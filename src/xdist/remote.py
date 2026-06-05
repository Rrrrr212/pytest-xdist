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
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from typing import Callable
from typing import Literal
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


class ConnectionPool:
    """Manages a pool of SSH connections to remote hosts."""

    def __init__(self, max_connections: int = 10, connection_timeout: int = 30) -> None:
        self._max_connections = max_connections
        self._connection_timeout = connection_timeout
        self._connections: dict[str, execnet.Gateway] = {}
        self._lock = threading.Lock()
        self._usage_count: dict[str, int] = {}
        self._last_used: dict[str, float] = {}

    def get_connection(self, host: str, user: str | None = None, port: int = 22, ssh_config: dict[str, Any] | None = None) -> execnet.Gateway:
        spec_key = f"{user}@{host}:{port}" if user else f"{host}:{port}"

        with self._lock:
            if spec_key in self._connections:
                gw = self._connections[spec_key]
                if gw.isremote():
                    self._usage_count[spec_key] = self._usage_count.get(spec_key, 0) + 1
                    self._last_used[spec_key] = time.time()
                    return gw

            if len(self._connections) >= self._max_connections:
                self._evict_idle()

            ssh_opts = ""
            if ssh_config:
                if ssh_config.get("identity_file"):
                    ssh_opts += f" -i {ssh_config['identity_file']}"
                if ssh_config.get("ssh_config_path"):
                    ssh_opts += f" -F {ssh_config['ssh_config_path']}"

            spec_str = f"ssh={user}@{host}//port={port}//ssh={ssh_opts.strip()}" if ssh_opts else f"ssh={user}@{host}//port={port}"
            spec_str += "//execmodel=main_thread_only"

            gw = execnet.makegateway(spec_str)
            self._connections[spec_key] = gw
            self._usage_count[spec_key] = 1
            self._last_used[spec_key] = time.time()

            return gw

    def _evict_idle(self) -> None:
        now = time.time()
        idle_threshold = now - 300

        idle_connections = [
            key for key, last_used in self._last_used.items()
            if last_used < idle_threshold
        ]

        for key in idle_connections[:5]:
            self._remove_connection(key)

    def _remove_connection(self, key: str) -> None:
        if key in self._connections:
            try:
                self._connections[key].exit()
            except Exception:
                pass
            del self._connections[key]
            self._usage_count.pop(key, None)
            self._last_used.pop(key, None)

    def release(self, host: str, user: str | None = None, port: int = 22) -> None:
        spec_key = f"{user}@{host}:{port}" if user else f"{host}:{port}"
        with self._lock:
            self._remove_connection(spec_key)

    def close_all(self) -> None:
        with self._lock:
            for key in list(self._connections.keys()):
                self._remove_connection(key)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_connections": len(self._connections),
                "max_connections": self._max_connections,
                "connections": {
                    key: {
                        "usage_count": self._usage_count.get(key, 0),
                        "last_used": self._last_used.get(key, 0),
                    }
                    for key in self._connections
                }
            }


class HeartbeatDetector:
    """Monitors worker health via periodic heartbeat signals."""

    HEARTBEAT_INTERVAL = 5.0
    HEARTBEAT_TIMEOUT = 15.0
    MAX_MISSED_HEARTBEATS = 3

    def __init__(
        self,
        worker_id: str,
        channel: execnet.Channel,
        log: Producer,
        on_heartbeat_failure: Callable[[str], None] | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.channel = channel
        self.log = log
        self.on_heartbeat_failure = on_heartbeat_failure
        self._last_heartbeat = time.time()
        self._missed_count = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        self.log("heartbeat detector started for worker", self.worker_id)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self.log("heartbeat detector stopped for worker", self.worker_id)

    def record_heartbeat(self) -> None:
        with self._lock:
            self._last_heartbeat = time.time()
            self._missed_count = 0

    def _heartbeat_loop(self) -> None:
        while self._running:
            time.sleep(self.HEARTBEAT_INTERVAL)

            if not self._running:
                break

            now = time.time()
            with self._lock:
                elapsed = now - self._last_heartbeat

            if elapsed > self.HEARTBEAT_TIMEOUT:
                self._missed_count += 1
                self.log(
                    "heartbeat missed for worker",
                    self.worker_id,
                    "miss_count:",
                    self._missed_count,
                )

                if self._missed_count >= self.MAX_MISSED_HEARTBEATS:
                    self.log(
                        "heartbeat failure detected for worker",
                        self.worker_id,
                    )
                    if self.on_heartbeat_failure:
                        self.on_heartbeat_failure(self.worker_id)
                    break

    def send_heartbeat(self) -> None:
        try:
            self.channel.send(("heartbeat", {"timestamp": time.time(), "worker_id": self.worker_id}))
            self.record_heartbeat()
        except Exception as e:
            self.log("failed to send heartbeat for worker", self.worker_id, "error:", str(e))
            self.record_heartbeat()

    @property
    def is_alive(self) -> bool:
        with self._lock:
            return self._missed_count < self.MAX_MISSED_HEARTBEATS

    @property
    def last_heartbeat(self) -> float:
        with self._lock:
            return self._last_heartbeat


class ResultReporter:
    """Handles test result reporting from remote workers to controller."""

    def __init__(
        self,
        worker_id: str,
        testrunuid: str,
        channel: execnet.Channel,
        log: Producer,
    ) -> None:
        self.worker_id = worker_id
        self.testrunuid = testrunuid
        self.channel = channel
        self.log = log
        self._results_buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._total_tests_run = 0
        self._total_failures = 0
        self._total_errors = 0
        self._start_time = time.time()

    def report_test_result(self, report_data: dict[str, Any], item_index: int) -> None:
        result = {
            "item_index": item_index,
            "worker_id": self.worker_id,
            "testrun_uid": self.testrunuid,
            "data": report_data,
            "timestamp": time.time(),
        }

        with self._lock:
            self._results_buffer.append(result)
            self._total_tests_run += 1

            if report_data.get("failed"):
                self._total_failures += 1
            elif report_data.get("passed") is False and not report_data.get("skipped"):
                self._total_errors += 1

        self._flush_results()

    def report_collection(self, topdir: str, test_ids: list[str]) -> None:
        self.log("reporting collection finish", len(test_ids), "tests")
        self.channel.send((
            "collectionfinish",
            {"topdir": topdir, "ids": test_ids, "worker_id": self.worker_id}
        ))

    def report_worker_ready(self, worker_info: WorkerInfo) -> None:
        self.channel.send((
            "workerready",
            {"workerinfo": worker_info, "worker_id": self.worker_id}
        ))

    def report_worker_finished(self, worker_output: dict[str, Any]) -> None:
        worker_output["exitstatus"] = int(worker_output.get("exitstatus", 0))
        worker_output["shouldfail"] = worker_output.get("shouldfail", False)
        worker_output["shouldstop"] = worker_output.get("shouldstop", False)
        worker_output["stats"] = self.get_stats()

        self._flush_results()

        self.channel.send((
            "workerfinished",
            {"workeroutput": worker_output, "worker_id": self.worker_id}
        ))

    def report_internal_error(self, formatted_error: str) -> None:
        self.channel.send((
            "internal_error",
            {"formatted_error": formatted_error, "worker_id": self.worker_id}
        ))

    def _flush_results(self) -> None:
        with self._lock:
            if not self._results_buffer:
                return

            results_to_send = list(self._results_buffer)
            self._results_buffer.clear()

        for result in results_to_send:
            try:
                self.channel.send(("testreport", result))
            except Exception as e:
                self.log("failed to send test report", "error:", str(e))
                with self._lock:
                    self._results_buffer.extend(results_to_send)
                break

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self._start_time
            return {
                "total_tests_run": self._total_tests_run,
                "total_failures": self._total_failures,
                "total_errors": self._total_errors,
                "elapsed_time": elapsed,
                "tests_per_second": self._total_tests_run / elapsed if elapsed > 0 else 0,
                "buffer_size": len(self._results_buffer),
            }

    def flush_pending(self) -> None:
        self._flush_results()


class RemoteWorker:
    """
    A worker that executes tests on remote machines via SSH.
    
    Features:
    - SSH connection pooling for efficient resource usage
    - Heartbeat detection for monitoring worker health
    - Result reporting with buffering and reliable delivery
    - Automatic reconnection on connection failures
    """

    MAX_RECONNECT_ATTEMPTS = 3
    RECONNECT_DELAY = 5.0

    def __init__(
        self,
        worker_id: str,
        testrunuid: str,
        host: str,
        user: str | None = None,
        port: int = 22,
        ssh_config: dict[str, Any] | None = None,
        connection_pool: ConnectionPool | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.testrunuid = testrunuid
        self.host = host
        self.user = user
        self.port = port
        self.ssh_config = ssh_config or {}
        self.connection_pool = connection_pool or ConnectionPool()
        
        self.gateway: execnet.Gateway | None = None
        self.channel: execnet.Channel | None = None
        self.log = Producer(f"remote-worker-{worker_id}", enabled=True)
        
        self.heartbeat: HeartbeatDetector | None = None
        self.reporter: ResultReporter | None = None
        
        self._is_connected = False
        self._reconnect_count = 0
        self._lock = threading.Lock()

    def connect(self) -> bool:
        self.log("connecting to remote host", self.host)
        
        try:
            self.gateway = self.connection_pool.get_connection(
                host=self.host,
                user=self.user,
                port=self.port,
                ssh_config=self.ssh_config,
            )
            
            self.channel = self.gateway.remote_exec(
                "import xdist.remote; xdist.remote.start_remote_worker()"
            )
            
            self._is_connected = True
            self._reconnect_count = 0
            
            self.reporter = ResultReporter(
                worker_id=self.worker_id,
                testrunuid=self.testrunuid,
                channel=self.channel,
                log=self.log,
            )
            
            self.heartbeat = HeartbeatDetector(
                worker_id=self.worker_id,
                channel=self.channel,
                log=self.log,
                on_heartbeat_failure=self._on_heartbeat_failure,
            )
            self.heartbeat.start()
            
            self.log("successfully connected to", self.host)
            return True
            
        except Exception as e:
            self.log("connection failed:", str(e))
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        self.log("disconnecting from remote host", self.host)
        
        if self.heartbeat:
            self.heartbeat.stop()
            self.heartbeat = None
        
        if self.reporter:
            self.reporter.flush_pending()
            self.reporter = None
        
        if self.gateway:
            try:
                self.gateway.exit()
            except Exception:
                pass
            self.gateway = None
        
        self.channel = None
        self._is_connected = False

    def send_command(self, command: str, **kwargs: Any) -> bool:
        if not self.channel or not self._is_connected:
            self.log("cannot send command, not connected")
            return False
        
        try:
            self.channel.send((command, kwargs))
            if self.heartbeat:
                self.heartbeat.record_heartbeat()
            return True
        except Exception as e:
            self.log("failed to send command:", str(e))
            self._handle_connection_failure()
            return False

    def run_tests(self, indices: list[int]) -> bool:
        return self.send_command("runtests", indices=indices)

    def run_all_tests(self) -> bool:
        return self.send_command("runtests_all")

    def shutdown(self) -> bool:
        return self.send_command("shutdown")

    def steal_tests(self, indices: list[int]) -> bool:
        return self.send_command("steal", indices=indices)

    def _handle_connection_failure(self) -> None:
        with self._lock:
            self._is_connected = False
            self._reconnect_count += 1
            
            self.log(
                "connection failure, attempt",
                self._reconnect_count,
                "of",
                self.MAX_RECONNECT_ATTEMPTS,
            )
            
            if self._reconnect_count < self.MAX_RECONNECT_ATTEMPTS:
                self.log("waiting before reconnect...")
                time.sleep(self.RECONNECT_DELAY)
                self.connect()

    def _on_heartbeat_failure(self, worker_id: str) -> None:
        self.log("heartbeat failure detected, attempting recovery")
        self._handle_connection_failure()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_alive(self) -> bool:
        if not self._is_connected:
            return False
        if self.heartbeat:
            return self.heartbeat.is_alive
        return True

    def get_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "worker_id": self.worker_id,
            "host": self.host,
            "is_connected": self._is_connected,
            "reconnect_count": self._reconnect_count,
        }
        
        if self.reporter:
            stats["result_stats"] = self.reporter.get_stats()
        
        if self.heartbeat:
            stats["last_heartbeat"] = self.heartbeat.last_heartbeat
        
        stats["connection_pool"] = self.connection_pool.get_stats()
        
        return stats


def start_remote_worker() -> None:
    """Entry point for remote worker execution."""
    pass


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
