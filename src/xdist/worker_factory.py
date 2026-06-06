
from __future__ import annotations

import abc
from collections.abc import Sequence
from enum import Enum
from typing import Any
from typing import Callable
from typing import Protocol
from typing import runtime_checkable

import execnet
import pytest

from xdist.remote import WorkerInfo


class WorkerType(Enum):
    PROCESS = "process"
    THREAD = "thread"
    REMOTE = "remote"


@runtime_checkable
class IWorker(Protocol):
    """Worker interface that all worker types must implement."""
    
    workerinfo: WorkerInfo
    workeroutput: dict[str, Any]
    shutting_down: bool
    
    def setup(self) -> None: ...
    def shutdown(self) -> None: ...
    def ensure_teardown(self) -> None: ...
    def send_runtest_some(self, indices: Sequence[int]) -> None: ...
    def send_runtest_all(self) -> None: ...
    def send_steal(self, indices: Sequence[int]) -> None: ...


class WorkerFactory(abc.ABC):
    """Abstract factory for creating worker instances."""
    
    def __init__(
        self,
        nodemanager: Any,
        gateway: execnet.Gateway,
        config: pytest.Config,
        putevent: Callable[[tuple[str, dict[str, Any]]], None],
    ) -> None:
        self.nodemanager = nodemanager
        self.gateway = gateway
        self.config = config
        self.putevent = putevent
    
    @abc.abstractmethod
    def create_worker(self) -> IWorker:
        """Create and return a configured worker instance."""
        pass
    
    @classmethod
    @abc.abstractmethod
    def supports_type(cls, worker_type: WorkerType) -> bool:
        """Check if this factory supports the given worker type."""
        pass


class ProcessWorkerFactory(WorkerFactory):
    """Factory for creating process-based workers."""
    
    def create_worker(self) -> IWorker:
        return ProcessWorkerController(
            nodemanager=self.nodemanager,
            gateway=self.gateway,
            config=self.config,
            putevent=self.putevent,
        )
    
    @classmethod
    def supports_type(cls, worker_type: WorkerType) -> bool:
        return worker_type == WorkerType.PROCESS


class ThreadWorkerFactory(WorkerFactory):
    """Factory for creating thread-based workers."""
    
    def create_worker(self) -> IWorker:
        return ThreadWorkerController(
            nodemanager=self.nodemanager,
            gateway=self.gateway,
            config=self.config,
            putevent=self.putevent,
        )
    
    @classmethod
    def supports_type(cls, worker_type: WorkerType) -> bool:
        return worker_type == WorkerType.THREAD


class RemoteWorkerFactory(WorkerFactory):
    """Factory for creating remote workers."""
    
    def create_worker(self) -> IWorker:
        return RemoteWorkerController(
            nodemanager=self.nodemanager,
            gateway=self.gateway,
            config=self.config,
            putevent=self.putevent,
        )
    
    @classmethod
    def supports_type(cls, worker_type: WorkerType) -> bool:
        return worker_type == WorkerType.REMOTE


class WorkerFactoryRegistry:
    """Registry for worker factories with automatic discovery."""
    
    _factories: list[type[WorkerFactory]] = []
    
    @classmethod
    def register(cls, factory_class: type[WorkerFactory]) -> None:
        if not issubclass(factory_class, WorkerFactory):
            raise TypeError(f"{factory_class} must be a subclass of WorkerFactory")
        cls._factories.append(factory_class)
    
    @classmethod
    def get_factory(
        cls,
        worker_type: WorkerType,
        nodemanager: Any,
        gateway: execnet.Gateway,
        config: pytest.Config,
        putevent: Callable[[tuple[str, dict[str, Any]]], None],
    ) -> WorkerFactory:
        for factory_class in cls._factories:
            if factory_class.supports_type(worker_type):
                return factory_class(nodemanager, gateway, config, putevent)
        raise ValueError(f"No factory registered for worker type: {worker_type}")
    
    @classmethod
    def unregister(cls, factory_class: type[WorkerFactory]) -> None:
        if factory_class in cls._factories:
            cls._factories.remove(factory_class)


WorkerFactoryRegistry.register(ProcessWorkerFactory)
WorkerFactoryRegistry.register(ThreadWorkerFactory)
WorkerFactoryRegistry.register(RemoteWorkerFactory)


class BaseWorkerController(IWorker):
    """Base class for all worker controllers with common functionality."""
    
    workerinfo: WorkerInfo
    workeroutput: dict[str, Any]
    
    def __init__(
        self,
        nodemanager: Any,
        gateway: execnet.Gateway,
        config: pytest.Config,
        putevent: Callable[[tuple[str, dict[str, Any]]], None],
    ) -> None:
        self.nodemanager = nodemanager
        self.putevent = putevent
        self.gateway = gateway
        self.config = config
        self.workerinput = {
            "workerid": gateway.id,
            "workercount": len(nodemanager.specs),
            "testrunuid": nodemanager.testrunuid,
            "mainargv": __import__("sys").argv,
        }
        self._down = False
        self._shutdown_sent = False
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.gateway.id}>"
    
    @property
    def shutting_down(self) -> bool:
        return self._down or self._shutdown_sent
    
    @abc.abstractmethod
    def setup(self) -> None:
        pass
    
    def shutdown(self) -> None:
        if not self._down:
            try:
                self.sendcommand("shutdown")
            except OSError:
                pass
            self._shutdown_sent = True
    
    def ensure_teardown(self) -> None:
        if hasattr(self, "channel"):
            if not self.channel.isclosed():
                self.channel.close()
        if hasattr(self, "gateway"):
            self.gateway.exit()
    
    @abc.abstractmethod
    def send_runtest_some(self, indices: Sequence[int]) -> None:
        pass
    
    @abc.abstractmethod
    def send_runtest_all(self) -> None:
        pass
    
    @abc.abstractmethod
    def send_steal(self, indices: Sequence[int]) -> None:
        pass
    
    def sendcommand(self, name: str, **kwargs: object) -> None:
        self.channel.send((name, kwargs))
    
    def notify_inproc(self, eventname: str, **kwargs: object) -> None:
        self.putevent((eventname, kwargs))


class ProcessWorkerController(BaseWorkerController):
    """Worker controller for process-based workers."""
    
    def setup(self) -> None:
        self.gateway._rinfo()
        spec = self.gateway.spec
        args = [str(x) for x in self.config.invocation_params.args or ()]
        option_dict = {}
        
        if not spec.popen or spec.chdir:
            from xdist.workermanage import make_reltoroot
            args = make_reltoroot(self.nodemanager.roots, args)
        
        if spec.popen:
            name = f"popen-{self.gateway.id}"
            if hasattr(self.config, "_tmp_path_factory"):
                basetemp = self.config._tmp_path_factory.getbasetemp()
                option_dict["basetemp"] = str(basetemp / name)
        
        self.config.hook.pytest_configure_node(node=self)
        
        import xdist.remote
        remote_module = self.config.hook.pytest_xdist_getremotemodule()
        self.channel = self.gateway.remote_exec(remote_module)
        
        from xdist.plugin import _sys_path
        change_sys_path = _sys_path if self.gateway.spec.popen else None
        self.channel.send((self.workerinput, args, option_dict, change_sys_path))
        
        if self.putevent:
            from xdist.workermanage import Marker
            self.channel.setcallback(self.process_from_remote, endmarker=Marker.END)
    
    def send_runtest_some(self, indices: Sequence[int]) -> None:
        self.sendcommand("runtests", indices=indices)
    
    def send_runtest_all(self) -> None:
        self.sendcommand("runtests_all")
    
    def send_steal(self, indices: Sequence[int]) -> None:
        self.sendcommand("steal", indices=indices)
    
    def process_from_remote(
        self, eventcall: tuple[str, dict[str, Any]] | Any
    ) -> None:
        from xdist.workermanage import Marker, unserialize_warning_message
        import pytest
        
        try:
            if eventcall is Marker.END:
                err = self.channel._getremoteerror()
                if not self._down:
                    if not err or isinstance(err, EOFError):
                        err = "Not properly terminated"
                    self.notify_inproc("errordown", node=self, error=err)
                    self._down = True
                return
            
            eventname, kwargs = eventcall
            if eventname in ("collectionstart",):
                pass
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
                warning_message = unserialize_warning_message(kwargs["warning_message_data"])
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


class ThreadWorkerController(BaseWorkerController):
    """Worker controller for thread-based workers."""
    
    def setup(self) -> None:
        self.gateway._rinfo()
        args = [str(x) for x in self.config.invocation_params.args or ()]
        option_dict = {}
        
        self.config.hook.pytest_configure_node(node=self)
        
        import xdist.remote
        remote_module = self.config.hook.pytest_xdist_getremotemodule()
        self.channel = self.gateway.remote_exec(remote_module)
        
        change_sys_path = None
        self.channel.send((self.workerinput, args, option_dict, change_sys_path))
        
        if self.putevent:
            from xdist.workermanage import Marker
            self.channel.setcallback(self.process_from_remote, endmarker=Marker.END)
    
    def send_runtest_some(self, indices: Sequence[int]) -> None:
        self.sendcommand("runtests", indices=indices)
    
    def send_runtest_all(self) -> None:
        self.sendcommand("runtests_all")
    
    def send_steal(self, indices: Sequence[int]) -> None:
        self.sendcommand("steal", indices=indices)
    
    def process_from_remote(
        self, eventcall: tuple[str, dict[str, Any]] | Any
    ) -> None:
        from xdist.workermanage import Marker, unserialize_warning_message
        import pytest
        
        try:
            if eventcall is Marker.END:
                err = self.channel._getremoteerror()
                if not self._down:
                    if not err or isinstance(err, EOFError):
                        err = "Not properly terminated"
                    self.notify_inproc("errordown", node=self, error=err)
                    self._down = True
                return
            
            eventname, kwargs = eventcall
            if eventname in ("collectionstart",):
                pass
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
                warning_message = unserialize_warning_message(kwargs["warning_message_data"])
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


class RemoteWorkerController(BaseWorkerController):
    """Worker controller for remote workers."""
    
    def setup(self) -> None:
        self.gateway._rinfo()
        args = [str(x) for x in self.config.invocation_params.args or ()]
        option_dict = {}
        
        from xdist.workermanage import make_reltoroot
        args = make_reltoroot(self.nodemanager.roots, args)
        
        self.config.hook.pytest_configure_node(node=self)
        
        import xdist.remote
        remote_module = self.config.hook.pytest_xdist_getremotemodule()
        self.channel = self.gateway.remote_exec(remote_module)
        
        change_sys_path = None
        self.channel.send((self.workerinput, args, option_dict, change_sys_path))
        
        if self.putevent:
            from xdist.workermanage import Marker
            self.channel.setcallback(self.process_from_remote, endmarker=Marker.END)
    
    def send_runtest_some(self, indices: Sequence[int]) -> None:
        self.sendcommand("runtests", indices=indices)
    
    def send_runtest_all(self) -> None:
        self.sendcommand("runtests_all")
    
    def send_steal(self, indices: Sequence[int]) -> None:
        self.sendcommand("steal", indices=indices)
    
    def process_from_remote(
        self, eventcall: tuple[str, dict[str, Any]] | Any
    ) -> None:
        from xdist.workermanage import Marker, unserialize_warning_message
        import pytest
        
        try:
            if eventcall is Marker.END:
                err = self.channel._getremoteerror()
                if not self._down:
                    if not err or isinstance(err, EOFError):
                        err = "Not properly terminated"
                    self.notify_inproc("errordown", node=self, error=err)
                    self._down = True
                return
            
            eventname, kwargs = eventcall
            if eventname in ("collectionstart",):
                pass
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
                warning_message = unserialize_warning_message(kwargs["warning_message_data"])
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
