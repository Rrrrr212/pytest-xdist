"""
This module is executed in remote subprocesses and helps to
control a remote testing session and relay back information.
It assumes that 'py' is importable and does not have dependencies
on the rest of the xdist code.  This means that the xdist-plugin
needs not to be installed in remote environments.
"""

from __future__ import annotations

import collections
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Sequence
import contextlib
import enum
import os
import sys
import time
import threading
from typing import Any
from typing import Callable
from typing import Literal
from typing import Optional
from typing import TypedDict
from typing import Union
import warnings

from _pytest.config import _prepareconfig
import execnet
import pytest


try:
    from setproctitle import setproctitle
except ImportError:

    def setproctitle(title: str) -> None:
        pass


class Producer:
    """
    Simplified implementation of the same interface as py.log, for backward compatibility
    since we dropped the dependency on pylib.
    Note: this is defined here because this module can't depend on xdist, so we need
    to have the other way around.
    """

    def __init__(self, name: str, *, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, enabled={self.enabled})"

    def __call__(self, *a: Any, **k: Any) -> None:
        if self.enabled:
            print(f"[{self.name}]", *a, **k, file=sys.stderr)

    def __getattr__(self, name: str) -> Producer:
        return type(self)(name, enabled=self.enabled)


def worker_title(title: str) -> None:
    try:
        setproctitle(title)
    except Exception:
        # changing the process name is very optional, no errors please
        pass


class Marker(enum.Enum):
    SHUTDOWN = 0


class TestQueue:
    """A simple queue that can be inspected and modified while the lock is held via the ``lock()`` method."""

    Item = Union[int, Literal[Marker.SHUTDOWN]]

    def __init__(self, execmodel: execnet.gateway_base.ExecModel):
        self._items: collections.deque[TestQueue.Item] = collections.deque()
        self._lock = execmodel.RLock()  # type: ignore[no-untyped-call]
        self._has_items_event = execmodel.Event()

    def get(self) -> Item:
        while True:
            with self.lock() as locked_items:
                if locked_items:
                    return locked_items.popleft()

            self._has_items_event.wait()

    def put(self, item: Item) -> None:
        with self.lock() as locked_items:
            locked_items.append(item)

    def replace(self, iterable: Iterable[Item]) -> None:
        with self.lock():
            self._items = collections.deque(iterable)

    @contextlib.contextmanager
    def lock(self) -> Generator[collections.deque[Item]]:
        with self._lock:
            try:
                yield self._items
            finally:
                if self._items:
                    self._has_items_event.set()
                else:
                    self._has_items_event.clear()


class WorkerInteractor:
    def __init__(self, config: pytest.Config, channel: execnet.Channel) -> None:
        self.config = config
        workerinput: dict[str, Any] = config.workerinput  # type: ignore[attr-defined]
        self.workerid = workerinput.get("workerid", "?")
        self.testrunuid = workerinput["testrunuid"]
        self.log = Producer(f"worker-{self.workerid}", enabled=config.option.debug)
        self.channel = channel
        self.torun = TestQueue(self.channel.gateway.execmodel)
        self.nextitem_index: int | None | Literal[Marker.SHUTDOWN] = None
        config.pluginmanager.register(self)

    def sendevent(self, name: str, **kwargs: object) -> None:
        self.log("sending", name, kwargs)
        self.channel.send((name, kwargs))

    @pytest.hookimpl
    def pytest_internalerror(self, excrepr: object) -> None:
        formatted_error = str(excrepr)
        for line in formatted_error.split("\n"):
            self.log("IERROR>", line)
        interactor.sendevent("internal_error", formatted_error=formatted_error)

    @pytest.hookimpl
    def pytest_sessionstart(self, session: pytest.Session) -> None:
        self.session = session
        workerinfo = getinfodict()
        self.sendevent("workerready", workerinfo=workerinfo)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_sessionfinish(self, exitstatus: int) -> Generator[None, object, None]:
        workeroutput: dict[str, Any] = self.config.workeroutput  # type: ignore[attr-defined]
        # in pytest 5.0+, exitstatus is an IntEnum object
        workeroutput["exitstatus"] = int(exitstatus)
        workeroutput["shouldfail"] = self.session.shouldfail
        workeroutput["shouldstop"] = self.session.shouldstop
        yield
        self.sendevent("workerfinished", workeroutput=workeroutput)

    @pytest.hookimpl
    def pytest_collection(self) -> None:
        self.sendevent("collectionstart")

    def handle_command(
        self, command: tuple[str, dict[str, Any]] | Literal[Marker.SHUTDOWN]
    ) -> None:
        if command is Marker.SHUTDOWN:
            self.torun.put(Marker.SHUTDOWN)
            return

        name, kwargs = command

        self.log("received command", name, kwargs)
        if name == "runtests":
            for i in kwargs["indices"]:
                self.torun.put(i)
        elif name == "runtests_all":
            for i in range(len(self.session.items)):
                self.torun.put(i)
        elif name == "shutdown":
            self.torun.put(Marker.SHUTDOWN)
        elif name == "steal":
            self.steal(kwargs["indices"])

    def steal(self, indices: Sequence[int]) -> None:
        """
        Remove tests from the queue.

        Removes either all requested tests, or none, if some of these tests
        are not in the queue (for example, if they were processed already).

        :param indices: indices of the tests to remove.
        """
        requested_set = set(indices)

        with self.torun.lock() as locked_queue:
            stolen = list(item for item in locked_queue if item in requested_set)

            # Stealing only if all requested tests are still pending
            if len(stolen) == len(requested_set):
                self.torun.replace(
                    item for item in locked_queue if item not in requested_set
                )
            else:
                stolen = []

        self.sendevent("unscheduled", indices=stolen)

    @pytest.hookimpl
    def pytest_runtestloop(self, session: pytest.Session) -> bool:
        self.log("entering main loop")
        self.channel.setcallback(self.handle_command, endmarker=Marker.SHUTDOWN)
        self.nextitem_index = self.torun.get()
        while self.nextitem_index is not Marker.SHUTDOWN:
            self.run_one_test()
            if session.shouldfail or session.shouldstop:
                break
        return True

    def run_one_test(self) -> None:
        assert isinstance(self.nextitem_index, int)
        self.item_index = self.nextitem_index
        self.nextitem_index = self.torun.get()

        items = self.session.items
        item = items[self.item_index]
        if self.nextitem_index is Marker.SHUTDOWN:
            nextitem = None
        else:
            assert self.nextitem_index is not None
            nextitem = items[self.nextitem_index]

        worker_title("[pytest-xdist running] %s" % item.nodeid)

        start = time.perf_counter()
        self.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)
        duration = time.perf_counter() - start

        worker_title("[pytest-xdist idle]")

        self.sendevent(
            "runtest_protocol_complete", item_index=self.item_index, duration=duration
        )

    def pytest_collection_modifyitems(
        self,
        config: pytest.Config,
        items: list[pytest.Item],
    ) -> None:
        # add the group name to nodeid as suffix if --dist=loadgroup
        if config.getvalue("loadgroup"):
            for item in items:
                gnames: set[str] = set()
                for mark in item.iter_markers("xdist_group"):
                    name = (
                        mark.args[0]
                        if len(mark.args) > 0
                        else mark.kwargs.get("name", "default")
                    )
                    gnames.add(str(name))
                if not gnames:
                    continue
                item._nodeid = f"{item.nodeid}@{'_'.join(sorted(gnames))}"

    @pytest.hookimpl
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.sendevent(
            "collectionfinish",
            topdir=str(self.config.rootpath),
            ids=[item.nodeid for item in session.items],
        )

    @pytest.hookimpl
    def pytest_runtest_logstart(
        self,
        nodeid: str,
        location: tuple[str, int | None, str],
    ) -> None:
        self.sendevent("logstart", nodeid=nodeid, location=location)

    @pytest.hookimpl
    def pytest_runtest_logfinish(
        self,
        nodeid: str,
        location: tuple[str, int | None, str],
    ) -> None:
        self.sendevent("logfinish", nodeid=nodeid, location=location)

    @pytest.hookimpl
    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        data = self.config.hook.pytest_report_to_serializable(
            config=self.config, report=report
        )
        data["item_index"] = self.item_index
        data["worker_id"] = self.workerid
        data["testrun_uid"] = self.testrunuid
        assert self.session.items[self.item_index].nodeid == report.nodeid
        self.sendevent("testreport", data=data)

    @pytest.hookimpl
    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        # send only reports that have not passed to controller as optimization (#330)
        if not report.passed:
            data = self.config.hook.pytest_report_to_serializable(
                config=self.config, report=report
            )
            self.sendevent("collectreport", data=data)

    @pytest.hookimpl
    def pytest_warning_recorded(
        self,
        warning_message: warnings.WarningMessage,
        when: str,
        nodeid: str,
        location: tuple[str, int, str] | None,
    ) -> None:
        self.sendevent(
            "warning_recorded",
            warning_message_data=serialize_warning_message(warning_message),
            when=when,
            nodeid=nodeid,
            location=location,
        )


def serialize_warning_message(
    warning_message: warnings.WarningMessage,
) -> dict[str, Any]:
    if isinstance(warning_message.message, Warning):
        message_module = type(warning_message.message).__module__
        message_class_name = type(warning_message.message).__name__
        message_str = str(warning_message.message)
        # check now if we can serialize the warning arguments (#349)
        # if not, we will just use the exception message on the controller node
        try:
            execnet.dumps(warning_message.message.args)
        except execnet.DumpError:
            message_args = None
        else:
            message_args = warning_message.message.args
    else:
        message_str = warning_message.message
        message_module = None
        message_class_name = None
        message_args = None
    if warning_message.category:
        category_module = warning_message.category.__module__
        category_class_name = warning_message.category.__name__
    else:
        category_module = None
        category_class_name = None

    result = {
        "message_str": message_str,
        "message_module": message_module,
        "message_class_name": message_class_name,
        "message_args": message_args,
        "category_module": category_module,
        "category_class_name": category_class_name,
    }
    # access private _WARNING_DETAILS because the attributes vary between Python versions
    for attr_name in warning_message._WARNING_DETAILS:  # type: ignore[attr-defined]
        if attr_name in ("message", "category"):
            continue
        attr = getattr(warning_message, attr_name)
        # Check if we can serialize the warning detail, marking `None` otherwise
        # Note that we need to define the attr (even as `None`) to allow deserializing
        try:
            execnet.dumps(attr)
        except execnet.DumpError:
            result[attr_name] = repr(attr)
        else:
            result[attr_name] = attr
    return result


class WorkerInfo(TypedDict):
    version: str
    version_info: tuple[int, int, int, str, int]
    sysplatform: str
    platform: str
    executable: str
    cwd: str
    id: str
    spec: execnet.XSpec


def getinfodict() -> WorkerInfo:
    import platform

    return dict(
        version=sys.version,
        version_info=tuple(sys.version_info),  # type: ignore[typeddict-item]
        sysplatform=sys.platform,
        platform=platform.platform(),
        executable=sys.executable,
        cwd=os.getcwd(),
    )


def setup_config(config: pytest.Config, basetemp: str | None) -> None:
    config.option.loadgroup = config.getvalue("dist") == "loadgroup"
    config.option.looponfail = False
    config.option.usepdb = False
    config.option.dist = "no"
    config.option.distload = False
    config.option.numprocesses = None
    config.option.maxprocesses = None
    config.option.basetemp = basetemp


if __name__ == "__channelexec__":
    channel: execnet.Channel = channel  # type: ignore[name-defined] # noqa: F821, PLW0127
    workerinput, args, option_dict, change_sys_path = channel.receive()  # type: ignore[name-defined]

    if change_sys_path is None:
        importpath = os.getcwd()
        sys.path.insert(0, importpath)
        os.environ["PYTHONPATH"] = (
            importpath + os.pathsep + os.environ.get("PYTHONPATH", "")
        )
    else:
        sys.path = change_sys_path

    os.environ["PYTEST_XDIST_TESTRUNUID"] = workerinput["testrunuid"]
    os.environ["PYTEST_XDIST_WORKER"] = workerinput["workerid"]
    os.environ["PYTEST_XDIST_WORKER_COUNT"] = str(workerinput["workercount"])

    config = _prepareconfig(args, None)

    setup_config(config, option_dict.get("basetemp"))
    config._parser.prog = os.path.basename(workerinput["mainargv"][0])
    config.workerinput = workerinput  # type: ignore[attr-defined]
    config.workeroutput = {}  # type: ignore[attr-defined]
    interactor = WorkerInteractor(config, channel)  # type: ignore[name-defined]
    config.hook.pytest_cmdline_main(config=config)


class RemoteWorkerStatus(enum.Enum):
    """Remote worker status enum."""
    IDLE = "idle"
    BUSY = "busy"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class RemoteWorker:
    """Remote worker that executes tests on a remote machine via SSH.
    
    This class provides SSH connectivity, connection pooling, heartbeat detection,
    and result transmission for distributed pytest execution.
    """

    DEFAULT_HEARTBEAT_INTERVAL = 30.0
    DEFAULT_CONNECTION_TIMEOUT = 60.0
    DEFAULT_POOL_SIZE = 5

    def __init__(
        self,
        worker_id: str,
        ssh_host: str,
        ssh_port: int = 22,
        ssh_username: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
        remote_dir: str = "pytest-xdist-remote",
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT,
        result_callback: Optional[Callable[[str, Any], None]] = None,
    ) -> None:
        """Initialize the RemoteWorker.
        
        Args:
            worker_id: Unique identifier for this worker
            ssh_host: SSH hostname or IP address
            ssh_port: SSH port number
            ssh_username: SSH username (defaults to current user)
            ssh_key_path: Path to SSH private key (optional)
            remote_dir: Directory on remote machine to execute tests
            heartbeat_interval: Interval in seconds for heartbeat checks
            connection_timeout: Connection timeout in seconds
            result_callback: Callback function for receiving results
        """
        self.worker_id = worker_id
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_username = ssh_username
        self.ssh_key_path = ssh_key_path
        self.remote_dir = remote_dir
        self.heartbeat_interval = heartbeat_interval
        self.connection_timeout = connection_timeout
        self.result_callback = result_callback
        
        self._gateway: Optional[execnet.Gateway] = None
        self._channel: Optional[execnet.Channel] = None
        self._status = RemoteWorkerStatus.DISCONNECTED
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_heartbeat: Optional[float] = None
        self._result_queue: list[tuple[str, Any]] = []

    def _build_ssh_spec(self) -> str:
        """Build the execnet SSH specification string.
        
        Returns:
            SSH specification string for execnet
        """
        spec_parts = ["ssh"]
        if self.ssh_username:
            spec_parts.append(f"//user={self.ssh_username}")
        spec_parts.append(f"//host={self.ssh_host}")
        spec_parts.append(f"//port={self.ssh_port}")
        if self.ssh_key_path:
            spec_parts.append(f"//ssh-identity={self.ssh_key_path}")
        if self.remote_dir:
            spec_parts.append(f"//chdir={self.remote_dir}")
        return "".join(spec_parts)

    def connect(self) -> bool:
        """Establish SSH connection to remote machine.
        
        Returns:
            True if connection successful, False otherwise
        """
        with self._lock:
            try:
                self.log(f"Connecting to {self.ssh_host}:{self.ssh_port}")
                ssh_spec = self._build_ssh_spec()
                self._gateway = execnet.makegateway(ssh_spec)
                self._status = RemoteWorkerStatus.IDLE
                self._last_heartbeat = time.time()
                self._start_heartbeat()
                self.log(f"Successfully connected to {self.ssh_host}")
                return True
            except Exception as e:
                self.log(f"Connection failed: {e}")
                self._status = RemoteWorkerStatus.ERROR
                return False

    def disconnect(self) -> None:
        """Disconnect from remote machine and clean up resources."""
        with self._lock:
            self._stop_heartbeat()
            if self._channel and not self._channel.isclosed():
                self._channel.close()
                self._channel = None
            if self._gateway:
                try:
                    self._gateway.exit()
                except Exception as e:
                    self.log(f"Error closing gateway: {e}")
                self._gateway = None
            self._status = RemoteWorkerStatus.DISCONNECTED
            self.log(f"Disconnected from {self.ssh_host}")

    def _start_heartbeat(self) -> None:
        """Start heartbeat monitoring thread."""
        self._heartbeat_stop_event.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        """Stop heartbeat monitoring thread."""
        self._heartbeat_stop_event.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5.0)

    def _heartbeat_loop(self) -> None:
        """Heartbeat loop to check remote worker status."""
        while not self._heartbeat_stop_event.is_set():
            try:
                if self._gateway and self._gateway._rinfo():
                    self._last_heartbeat = time.time()
                else:
                    self._status = RemoteWorkerStatus.ERROR
                    self.log("Heartbeat failed: gateway not responsive")
            except Exception as e:
                self.log(f"Heartbeat error: {e}")
                self._status = RemoteWorkerStatus.ERROR
            self._heartbeat_stop_event.wait(self.heartbeat_interval)

    def is_connected(self) -> bool:
        """Check if worker is connected.
        
        Returns:
            True if connected, False otherwise
        """
        return (
            self._status != RemoteWorkerStatus.DISCONNECTED
            and self._status != RemoteWorkerStatus.ERROR
        )

    def get_status(self) -> RemoteWorkerStatus:
        """Get current worker status.
        
        Returns:
            Current RemoteWorkerStatus
        """
        return self._status

    def setup_remote_session(self, worker_input: dict[str, Any]) -> bool:
        """Set up remote test session.
        
        Args:
            worker_input: Worker input configuration
            
        Returns:
            True if setup successful, False otherwise
        """
        with self._lock:
            if not self.is_connected():
                self.log("Cannot setup session: not connected")
                return False
            
            try:
                self.log("Setting up remote session")
                self._channel = self._gateway.remote_exec(sys.modules[__name__])
                self._channel.send(worker_input)
                self._channel.setcallback(self._handle_remote_event)
                self._status = RemoteWorkerStatus.IDLE
                return True
            except Exception as e:
                self.log(f"Session setup failed: {e}")
                self._status = RemoteWorkerStatus.ERROR
                return False

    def send_command(self, command: str, **kwargs: Any) -> None:
        """Send command to remote worker.
        
        Args:
            command: Command name
            **kwargs: Command arguments
        """
        with self._lock:
            if not self._channel or self._channel.isclosed():
                self.log("Cannot send command: channel not available")
                return
            
            try:
                self._channel.send((command, kwargs))
                self._status = RemoteWorkerStatus.BUSY
            except Exception as e:
                self.log(f"Failed to send command: {e}")
                self._status = RemoteWorkerStatus.ERROR

    def _handle_remote_event(self, event: tuple[str, dict[str, Any]] | Literal[Marker.SHUTDOWN]) -> None:
        """Handle events received from remote worker.
        
        Args:
            event: Event tuple or shutdown marker
        """
        if event is Marker.SHUTDOWN:
            self._status = RemoteWorkerStatus.IDLE
            self.log("Received shutdown from remote")
            return
        
        event_name, event_data = event
        self.log(f"Received event: {event_name}")
        
        self._result_queue.append((event_name, event_data))
        if self.result_callback:
            self.result_callback(event_name, event_data)
        
        if event_name == "workerfinished":
            self._status = RemoteWorkerStatus.IDLE

    def get_results(self) -> list[tuple[str, Any]]:
        """Get accumulated results.
        
        Returns:
            List of (event_name, event_data) tuples
        """
        with self._lock:
            results = list(self._result_queue)
            self._result_queue.clear()
            return results

    def log(self, message: str) -> None:
        """Log a message with worker identifier.
        
        Args:
            message: Message to log
        """
        print(f"[RemoteWorker:{self.worker_id}] {message}")

    def __repr__(self) -> str:
        return f"<RemoteWorker {self.worker_id}@{self.ssh_host} status={self._status.value}>"


class RemoteWorkerPool:
    """Pool of RemoteWorker instances for managing multiple remote workers."""

    def __init__(
        self,
        pool_size: int = RemoteWorker.DEFAULT_POOL_SIZE,
    ) -> None:
        """Initialize the worker pool.
        
        Args:
            pool_size: Maximum number of workers in the pool
        """
        self.pool_size = pool_size
        self._workers: dict[str, RemoteWorker] = {}
        self._lock = threading.Lock()

    def add_worker(self, worker: RemoteWorker) -> None:
        """Add a worker to the pool.
        
        Args:
            worker: RemoteWorker instance to add
        """
        with self._lock:
            if len(self._workers) >= self.pool_size:
                raise RuntimeError(f"Pool full (max {self.pool_size} workers)")
            self._workers[worker.worker_id] = worker

    def remove_worker(self, worker_id: str) -> Optional[RemoteWorker]:
        """Remove a worker from the pool.
        
        Args:
            worker_id: ID of worker to remove
            
        Returns:
            Removed RemoteWorker instance or None
        """
        with self._lock:
            return self._workers.pop(worker_id, None)

    def get_worker(self, worker_id: str) -> Optional[RemoteWorker]:
        """Get a worker from the pool by ID.
        
        Args:
            worker_id: ID of worker to get
            
        Returns:
            RemoteWorker instance or None
        """
        with self._lock:
            return self._workers.get(worker_id)

    def get_idle_worker(self) -> Optional[RemoteWorker]:
        """Get an idle worker from the pool.
        
        Returns:
            Idle RemoteWorker instance or None
        """
        with self._lock:
            for worker in self._workers.values():
                if worker.get_status() == RemoteWorkerStatus.IDLE:
                    return worker
            return None

    def get_all_workers(self) -> list[RemoteWorker]:
        """Get all workers in the pool.
        
        Returns:
            List of all RemoteWorker instances
        """
        with self._lock:
            return list(self._workers.values())

    def connect_all(self) -> dict[str, bool]:
        """Connect all workers in the pool.
        
        Returns:
            Dictionary mapping worker IDs to connection success status
        """
        results = {}
        with self._lock:
            for worker_id, worker in self._workers.items():
                results[worker_id] = worker.connect()
        return results

    def disconnect_all(self) -> None:
        """Disconnect all workers in the pool."""
        with self._lock:
            for worker in self._workers.values():
                worker.disconnect()

    def __len__(self) -> int:
        with self._lock:
            return len(self._workers)

    def __contains__(self, worker_id: str) -> bool:
        with self._lock:
            return worker_id in self._workers
