from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
import enum
import fnmatch
import os
from pathlib import Path
import re
import sys
from typing import Any
from typing import Callable
from typing import Literal
from typing import Union
import uuid
import warnings

import execnet
import pytest

from xdist.plugin import _sys_path
import xdist.remote
from xdist.remote import Producer
from xdist.remote import WorkerInfo


def parse_tx_spec_config(config: pytest.Config) -> list[str]:
    xspeclist = []
    tx: list[str] = config.getvalue("tx")
    for xspec in tx:
        i = xspec.find("*")
        try:
            num = int(xspec[:i])
        except ValueError:
            xspeclist.append(xspec)
        else:
            xspeclist.extend([xspec[i + 1 :]] * num)
    if not xspeclist:
        raise pytest.UsageError(
            "MISSING test execution (tx) nodes: please specify --tx"
        )
    return xspeclist


# ---------------------------------------------------------------------------
#  Abstract Worker Factory
# ---------------------------------------------------------------------------


class AbstractWorkerFactory(ABC):
    """Abstract factory for creating worker controllers.

    Decouples worker instantiation from the NodeManager, enabling
    pluggable worker backends (process, thread, remote, etc.).
    """

    def __init__(self, nodemanager: NodeManager) -> None:
        self.nodemanager = nodemanager

    @abstractmethod
    def create_worker_controller(
        self,
        spec: execnet.XSpec,
        putevent: Callable[[tuple[str, dict[str, Any]]], None],
    ) -> WorkerController: ...

    @abstractmethod
    def teardown(self, timeout: float = 10.0) -> None: ...


# ---------------------------------------------------------------------------
#  Execnet-based factories (shared gateway lifecycle)
# ---------------------------------------------------------------------------


class _ExecnetWorkerFactory(AbstractWorkerFactory):
    """Base for execnet-based factories that share gateway lifecycle management."""

    def __init__(self, nodemanager: NodeManager) -> None:
        super().__init__(nodemanager)
        self._group = execnet.Group(execmodel="main_thread_only")

    def _ensure_execmodel(self, spec: execnet.XSpec) -> execnet.XSpec:
        if getattr(spec, "execmodel", None) != "main_thread_only":
            return execnet.XSpec(f"execmodel=main_thread_only//{spec}")
        return spec

    def _make_gateway(self, spec: execnet.XSpec) -> execnet.Gateway:
        spec = self._ensure_execmodel(spec)
        gw = self._group.makegateway(spec)
        self.nodemanager.config.hook.pytest_xdist_newgateway(gateway=gw)
        self.nodemanager.rsync_roots(gw)
        return gw

    def teardown(self, timeout: float = 10.0) -> None:
        self._group.terminate(timeout)


class ProcessWorkerFactory(_ExecnetWorkerFactory):
    """Creates workers via execnet *popen* gateways (local subprocesses)."""

    def create_worker_controller(
        self,
        spec: execnet.XSpec,
        putevent: Callable[[tuple[str, dict[str, Any]]], None],
    ) -> WorkerController:
        gw = self._make_gateway(spec)
        node = WorkerController(self.nodemanager, gw, self.nodemanager.config, putevent)
        gw.node = node  # type: ignore[attr-defined]
        node.setup()
        self.nodemanager.trace("started process-worker %r" % node)
        return node


class RemoteWorkerFactory(_ExecnetWorkerFactory):
    """Creates workers via execnet *ssh* / *socket* gateways (remote hosts)."""

    def create_worker_controller(
        self,
        spec: execnet.XSpec,
        putevent: Callable[[tuple[str, dict[str, Any]]], None],
    ) -> WorkerController:
        gw = self._make_gateway(spec)
        node = WorkerController(self.nodemanager, gw, self.nodemanager.config, putevent)
        gw.node = node  # type: ignore[attr-defined]
        node.setup()
        self.nodemanager.trace("started remote-worker %r" % node)
        return node


# ---------------------------------------------------------------------------
#  Thread-based worker factory
# ---------------------------------------------------------------------------


class ThreadWorkerFactory(AbstractWorkerFactory):
    """Creates workers that run inside the same process using threads.

    This is a lightweight alternative to subprocess workers, suitable
    for debugging or resource-constrained environments.
    """

    def __init__(self, nodemanager: NodeManager) -> None:
        super().__init__(nodemanager)
        self._controllers: list[WorkerController] = []

    def create_worker_controller(
        self,
        spec: execnet.XSpec,
        putevent: Callable[[tuple[str, dict[str, Any]]], None],
    ) -> WorkerController:
        gw = ThreadGateway(spec)
        self.nodemanager.config.hook.pytest_xdist_newgateway(gateway=gw)
        node = WorkerController(self.nodemanager, gw, self.nodemanager.config, putevent)
        gw.node = node  # type: ignore[attr-defined]
        node.setup()
        self._controllers.append(node)
        self.nodemanager.trace("started thread-worker %r" % node)
        return node

    def teardown(self, timeout: float = 10.0) -> None:
        for node in self._controllers:
            try:
                node.ensure_teardown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
#  ThreadGateway – minimal execnet.Gateway-compatible stub for thread workers
# ---------------------------------------------------------------------------


class ThreadGateway:
    """A duck-type compatible gateway that runs worker code in a thread."""

    def __init__(self, spec: execnet.XSpec) -> None:
        self.spec = spec
        self.id = getattr(spec, "id", None) or "thread-gw-%d" % id(self)
        self._channel: ThreadChannel | None = None

    def _rinfo(self) -> None:
        pass

    def remote_exec(self, module: Any) -> ThreadChannel:
        from threading import Thread

        ch = ThreadChannel(self)
        self._channel = ch

        def _run_worker() -> None:
            import io
            import pytest

            config = pytest.Config.fromdictargs(
                {"plugins": []}, []
            )
            config.workerinput = ch._workerinput
            config.pluginmanager.register(module)
            config.hook.pytest_cmdline_main(config=config)

        Thread(target=_run_worker, daemon=True).start()
        return ch

    def exit(self) -> None:
        if self._channel is not None and not self._channel.isclosed():
            self._channel.close()


class ThreadChannel:
    """Minimal channel implementation for thread-based workers."""

    def __init__(self, gateway: ThreadGateway) -> None:
        self.gateway = gateway
        self._closed = False
        self._callback: Callable[..., Any] | None = None
        self._endmarker: Any = None
        self._workerinput: dict[str, Any] = {}

    def send(self, data: Any) -> None:
        if self._closed:
            raise OSError("channel closed")

    def setcallback(
        self,
        callback: Callable[..., Any],
        endmarker: Any = None,
    ) -> None:
        self._callback = callback
        self._endmarker = endmarker

    def isclosed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True

    def _getremoteerror(self) -> object | None:
        return None


# ---------------------------------------------------------------------------
#  Factory Registry
# ---------------------------------------------------------------------------


class WorkerFactoryRegistry:
    """Registry that maps worker-type names to factory *classes*.

    Supports auto-detection of the appropriate factory from an execnet
    XSpec by inspecting ``spec.popen`` and ``spec.ssh`` attributes.
    """

    def __init__(self) -> None:
        self._factories: dict[str, type[AbstractWorkerFactory]] = {
            "process": ProcessWorkerFactory,
            "remote": RemoteWorkerFactory,
            "thread": ThreadWorkerFactory,
        }
        self._instances: dict[str, AbstractWorkerFactory] = {}

    def register(self, name: str, factory_cls: type[AbstractWorkerFactory]) -> None:
        self._factories[name] = factory_cls

    def get(self, name: str, nodemanager: NodeManager) -> AbstractWorkerFactory:
        if name not in self._instances:
            factory_cls = self._factories.get(name)
            if factory_cls is None:
                available = ", ".join(sorted(self._factories))
                raise ValueError(
                    f"Unknown worker type '{name}'. Available: {available}"
                )
            self._instances[name] = factory_cls(nodemanager)
        return self._instances[name]

    def get_by_spec(
        self,
        spec: execnet.XSpec,
        nodemanager: NodeManager,
    ) -> AbstractWorkerFactory:
        if getattr(spec, "worker_type", None) == "thread":
            return self.get("thread", nodemanager)
        if spec.ssh or spec.socket:
            return self.get("remote", nodemanager)
        return self.get("process", nodemanager)

    def teardown_all(self, timeout: float = 10.0) -> None:
        for factory in self._instances.values():
            factory.teardown(timeout)
        self._instances.clear()


# ---------------------------------------------------------------------------
#  NodeManager (refactored)
# ---------------------------------------------------------------------------


class NodeManager:
    EXIT_TIMEOUT = 10
    DEFAULT_IGNORES = [".*", "*.pyc", "*.pyo", "*~"]

    def __init__(
        self,
        config: pytest.Config,
        specs: Sequence[execnet.XSpec | str] | None = None,
        defaultchdir: str = "pyexecnetcache",
    ) -> None:
        self.config = config
        self.trace = self.config.trace.get("nodemanager")
        self.testrunuid = self.config.getoption("testrunuid")
        if self.testrunuid is None:
            self.testrunuid = uuid.uuid4().hex

        self.registry = WorkerFactoryRegistry()

        if specs is None:
            specs = self._gettxspecs()
        self.specs: list[execnet.XSpec] = []
        for spec in specs:
            if not isinstance(spec, execnet.XSpec):
                spec = execnet.XSpec(spec)
            if not spec.chdir and not spec.popen and not spec.ssh and not spec.socket:
                spec.chdir = defaultchdir
            self._ensure_gateway_id(spec)
            self.specs.append(spec)

        self.roots = self._getrsyncdirs()
        self.rsyncoptions = self._getrsyncoptions()
        self._rsynced_specs: set[tuple[Any, Any]] = set()

    def _ensure_gateway_id(self, spec: execnet.XSpec) -> None:
        factory = self.registry.get_by_spec(spec, self)
        if isinstance(factory, _ExecnetWorkerFactory):
            factory._group.allocate_id(spec)

    def rsync_roots(self, gateway: execnet.Gateway) -> None:
        if self.roots:
            for root in self.roots:
                self.rsync(gateway, root, **self.rsyncoptions)

    def setup_nodes(
        self,
        putevent: Callable[[tuple[str, dict[str, Any]]], None],
    ) -> list[WorkerController]:
        self.config.hook.pytest_xdist_setupnodes(config=self.config, specs=self.specs)
        self.trace("setting up nodes")
        return [self.setup_node(spec, putevent) for spec in self.specs]

    def setup_node(
        self,
        spec: execnet.XSpec,
        putevent: Callable[[tuple[str, dict[str, Any]]], None],
    ) -> WorkerController:
        factory = self.registry.get_by_spec(spec, self)
        node = factory.create_worker_controller(spec, putevent)
        self.trace("started node %r" % node)
        return node

    def teardown_nodes(self) -> None:
        self.registry.teardown_all(self.EXIT_TIMEOUT)

    def _gettxspecs(self) -> list[execnet.XSpec]:
        return [execnet.XSpec(x) for x in parse_tx_spec_config(self.config)]

    def _getrsyncdirs(self) -> list[Path]:
        for spec in self.specs:
            if not spec.popen or spec.chdir:
                break
        else:
            return []
        import _pytest
        import pytest

        def get_dir(p: str) -> str:
            stripped = p.rstrip("co")
            if os.path.basename(stripped) == "__init__.py":
                return os.path.dirname(p)
            else:
                return stripped

        pytestpath = get_dir(pytest.__file__)
        pytestdir = get_dir(_pytest.__file__)
        config = self.config
        candidates = [pytestpath, pytestdir]
        candidates += config.option.rsyncdir
        rsyncroots = config.getini("rsyncdirs")
        if rsyncroots:
            candidates.extend(rsyncroots)
        roots = []
        for root in candidates:
            root_path = Path(root).resolve()
            if not root_path.exists():
                raise pytest.UsageError(f"rsyncdir doesn't exist: {root!r}")
            if root_path not in roots:
                roots.append(root_path)
        return roots

    def _getrsyncoptions(self) -> dict[str, Any]:
        ignores = list(self.DEFAULT_IGNORES)
        ignores += [str(path) for path in self.config.option.rsyncignore]
        ignores += [str(path) for path in self.config.getini("rsyncignore")]

        return {
            "ignores": ignores,
            "verbose": getattr(self.config.option, "verbose", 0),
        }

    def rsync(
        self,
        gateway: execnet.Gateway,
        source: str | os.PathLike[str],
        notify: (
            Callable[[str, execnet.XSpec, str | os.PathLike[str]], Any] | None
        ) = None,
        verbose: int = False,
        ignores: Sequence[str] | None = None,
    ) -> None:
        rsync = HostRSync(source, verbose=verbose > 0, ignores=ignores)
        spec = gateway.spec
        if spec.popen and not spec.chdir:
            gateway.remote_exec(
                """
                import sys ; sys.path.insert(0, %r)
            """
                % os.path.dirname(str(source))
            ).waitclose()
            return
        if (spec, source) in self._rsynced_specs:
            return

        def finished() -> None:
            if notify:
                notify("rsyncrootready", spec, source)

        rsync.add_target_host(gateway, finished=finished)
        self._rsynced_specs.add((spec, source))
        self.config.hook.pytest_xdist_rsyncstart(source=source, gateways=[gateway])
        rsync.send()
        self.config.hook.pytest_xdist_rsyncfinish(source=source, gateways=[gateway])


class HostRSync(execnet.RSync):
    """RSyncer that filters out common files."""

    PathLike = Union[str, "os.PathLike[str]"]

    def __init__(
        self,
        sourcedir: PathLike,
        *,
        ignores: Sequence[PathLike] | None = None,
        verbose: bool = True,
    ) -> None:
        if ignores is None:
            ignores = []
        self._ignores = [re.compile(fnmatch.translate(os.fspath(x))) for x in ignores]
        super().__init__(sourcedir=Path(sourcedir), verbose=verbose)

    def filter(self, path: PathLike) -> bool:
        path = Path(path)
        for cre in self._ignores:
            if cre.match(path.name) or cre.match(str(path)):
                return False
        else:
            return True

    def add_target_host(
        self,
        gateway: execnet.Gateway,
        finished: Callable[[], None] | None = None,
    ) -> None:
        remotepath = os.path.basename(self._sourcedir)
        super().add_target(gateway, remotepath, finishedcallback=finished, delete=True)

    def _report_send_file(
        self,
        gateway: execnet.Gateway,  # type: ignore[override]
        modified_rel_path: str,
    ) -> None:
        if self._verbose > 0:
            path = os.path.basename(self._sourcedir) + "/" + modified_rel_path
            remotepath = gateway.spec.chdir
            print(f"{gateway.spec}:{remotepath} <= {path}")


def make_reltoroot(roots: Sequence[Path], args: list[str]) -> list[str]:
    splitcode = "::"
    result = []
    for arg in args:
        parts = arg.split(splitcode)
        fspath = Path(parts[0])
        try:
            exists = fspath.exists()
        except OSError:
            exists = False
        if not exists:
            result.append(arg)
            continue
        for root in roots:
            x: Path | None
            try:
                x = fspath.relative_to(root)
            except ValueError:
                x = None
            if x or fspath == root:
                parts[0] = root.name + "/" + str(x)
                break
        else:
            raise ValueError(f"arg {arg} not relative to an rsync root")
        result.append(splitcode.join(parts))
    return result


class Marker(enum.Enum):
    END = -1


class WorkerController:
    workerinfo: WorkerInfo

    class RemoteHook:
        @pytest.hookimpl(trylast=True)
        def pytest_xdist_getremotemodule(self) -> Any:
            return xdist.remote

    def __init__(
        self,
        nodemanager: NodeManager,
        gateway: execnet.Gateway,
        config: pytest.Config,
        putevent: Callable[[tuple[str, dict[str, Any]]], None],
    ) -> None:
        config.pluginmanager.register(self.RemoteHook())
        self.nodemanager = nodemanager
        self.putevent = putevent
        self.gateway = gateway
        self.config = config
        self.workerinput = {
            "workerid": gateway.id,
            "workercount": len(nodemanager.specs),
            "testrunuid": nodemanager.testrunuid,
            "mainargv": sys.argv,
        }
        self._down = False
        self._shutdown_sent = False
        self.log = Producer(f"workerctl-{gateway.id}", enabled=config.option.debug)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.gateway.id}>"

    @property
    def shutting_down(self) -> bool:
        return self._down or self._shutdown_sent

    def setup(self) -> None:
        self.log("setting up worker session")
        self.gateway._rinfo()
        spec = self.gateway.spec
        args = [str(x) for x in self.config.invocation_params.args or ()]
        option_dict = {}
        if not spec.popen or spec.chdir:
            args = make_reltoroot(self.nodemanager.roots, args)
        if spec.popen:
            name = "popen-%s" % self.gateway.id
            if hasattr(self.config, "_tmp_path_factory"):
                basetemp = self.config._tmp_path_factory.getbasetemp()
                option_dict["basetemp"] = str(basetemp / name)
        self.config.hook.pytest_configure_node(node=self)

        remote_module = self.config.hook.pytest_xdist_getremotemodule()
        self.channel = self.gateway.remote_exec(remote_module)
        change_sys_path = _sys_path if self.gateway.spec.popen else None
        self.channel.send((self.workerinput, args, option_dict, change_sys_path))

        if self.putevent:  # type: ignore[truthy-function]
            self.channel.setcallback(self.process_from_remote, endmarker=Marker.END)

    def ensure_teardown(self) -> None:
        if hasattr(self, "channel"):
            if not self.channel.isclosed():
                self.log("closing", self.channel)
                self.channel.close()
        if hasattr(self, "gateway"):
            self.log("exiting", self.gateway)
            self.gateway.exit()

    def send_runtest_some(self, indices: Sequence[int]) -> None:
        self.sendcommand("runtests", indices=indices)

    def send_runtest_all(self) -> None:
        self.sendcommand("runtests_all")

    def send_steal(self, indices: Sequence[int]) -> None:
        self.sendcommand("steal", indices=indices)

    def shutdown(self) -> None:
        if not self._down:
            try:
                self.sendcommand("shutdown")
            except OSError:
                pass
            self._shutdown_sent = True

    def sendcommand(self, name: str, **kwargs: object) -> None:
        self.log(f"sending command {name}(**{kwargs})")
        self.channel.send((name, kwargs))

    def notify_inproc(self, eventname: str, **kwargs: object) -> None:
        self.log(f"queuing {eventname}(**{kwargs})")
        self.putevent((eventname, kwargs))

    def process_from_remote(
        self, eventcall: tuple[str, dict[str, Any]] | Literal[Marker.END]
    ) -> None:
        try:
            if eventcall is Marker.END:
                err: object | None = self.channel._getremoteerror()  # type: ignore[no-untyped-call]
                if not self._down:
                    if not err or isinstance(err, EOFError):
                        err = "Not properly terminated"
                    self.notify_inproc("errordown", node=self, error=err)
                    self._down = True
                return
            eventname, kwargs = eventcall
            if eventname in ("collectionstart",):
                self.log(f"ignoring {eventname}({kwargs})")
            elif eventname == "workerready":
                self.notify_inproc(eventname, node=self, **kwargs)
            elif eventname == "internal_error":
                self.notify_inproc(eventname, node=self, **kwargs)
            elif eventname == "workerfinished":
                self._down = True
                self.workeroutput = kwargs["workeroutput"]
                self.notify_inproc("workerfinished", node=self)
            elif eventname in ("logstart", "logfinish"):
                self.notify_inproc(eventname, node=self, **kwargs)
            elif eventname in ("testreport", "collectreport", "teardownreport"):
                item_index = kwargs.pop("item_index", None)
                rep = self.config.hook.pytest_report_from_serializable(
                    config=self.config, data=kwargs["data"]
                )
                if item_index is not None:
                    rep.item_index = item_index
                self.notify_inproc(eventname, node=self, rep=rep)
            elif eventname == "collectionfinish":
                self.notify_inproc(eventname, node=self, ids=kwargs["ids"])
            elif eventname == "runtest_protocol_complete":
                self.notify_inproc(eventname, node=self, **kwargs)
            elif eventname == "unscheduled":
                self.notify_inproc(eventname, node=self, **kwargs)
            elif eventname == "logwarning":
                self.notify_inproc(
                    eventname,
                    message=kwargs["message"],
                    code=kwargs["code"],
                    nodeid=kwargs["nodeid"],
                    fslocation=kwargs["nodeid"],
                )
            elif eventname == "warning_recorded":
                warning_message = unserialize_warning_message(
                    kwargs["warning_message_data"]
                )
                self.notify_inproc(
                    eventname,
                    warning_message=warning_message,
                    when=kwargs["when"],
                    nodeid=kwargs["nodeid"],
                    location=kwargs["location"],
                )
            else:
                raise ValueError(f"unknown event: {eventname}")
        except KeyboardInterrupt:
            raise
        except BaseException:
            excinfo = pytest.ExceptionInfo.from_current()
            print("!" * 20, excinfo)
            self.config.notify_exception(excinfo)
            self.shutdown()
            self.notify_inproc("errordown", node=self, error=excinfo)


def unserialize_warning_message(data: dict[str, Any]) -> warnings.WarningMessage:
    import importlib

    if data["message_module"]:
        mod = importlib.import_module(data["message_module"])
        cls = getattr(mod, data["message_class_name"])
        message = None
        if data["message_args"] is not None:
            try:
                message = cls(*data["message_args"])
            except TypeError:
                pass
        if message is None:
            message_text = "{mod}.{cls}: {msg}".format(
                mod=data["message_module"],
                cls=data["message_class_name"],
                msg=data["message_str"],
            )
            message = Warning(message_text)
    else:
        message = data["message_str"]

    if data["category_module"]:
        mod = importlib.import_module(data["category_module"])
        category = getattr(mod, data["category_class_name"])
    else:
        category = None

    kwargs = {"message": message, "category": category}
    for attr_name in warnings.WarningMessage._WARNING_DETAILS:  # type: ignore[attr-defined]
        if attr_name in ("message", "category"):
            continue
        kwargs[attr_name] = data[attr_name]

    return warnings.WarningMessage(**kwargs)