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
from dataclasses import dataclass
from dataclasses import field
import enum
import json
import os
import queue
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


@dataclass
class SSHConfig:
    """SSH 连接配置"""
    host: str
    port: int = 22
    username: str = ""
    password: str = ""
    key_filename: str = ""
    timeout: int = 30
    remote_python: str = "python"
    working_dir: str = ""


@dataclass
class SSHConnection:
    """SSH 连接实例"""
    config: SSHConfig
    process: subprocess.Popen | None = None
    connected: bool = False
    last_heartbeat: float = 0.0
    heartbeat_failures: int = 0
    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    result_queue: queue.Queue = field(default_factory=queue.Queue)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def is_alive(self) -> bool:
        """检查连接是否存活"""
        with self._lock:
            if not self.connected or self.process is None:
                return False
            return self.process.poll() is None

    def close(self) -> None:
        """关闭连接"""
        with self._lock:
            self.connected = False
            if self.process is not None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                self.process = None
            if self.stdout_thread and self.stdout_thread.is_alive():
                self.stdout_thread.join(timeout=1)
            if self.stderr_thread and self.stderr_thread.is_alive():
                self.stderr_thread.join(timeout=1)
            self.stdout_thread = None
            self.stderr_thread = None


class SSHConnectionPool:
    """SSH 连接池"""

    def __init__(
        self,
        max_connections: int = 10,
        heartbeat_interval: float = 30.0,
        max_heartbeat_failures: int = 3,
    ) -> None:
        self.max_connections = max_connections
        self.heartbeat_interval = heartbeat_interval
        self.max_heartbeat_failures = max_heartbeat_failures
        self._connections: dict[str, SSHConnection] = {}
        self._lock = threading.Lock()
        self._heartbeat_thread: threading.Thread | None = None
        self._running = False

    def __enter__(self) -> SSHConnectionPool:
        self.start_heartbeat_monitor()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop_heartbeat_monitor()
        self.close_all()

    def _get_connection_key(self, config: SSHConfig) -> str:
        """生成连接唯一标识"""
        return f"{config.username}@{config.host}:{config.port}"

    def get_connection(self, config: SSHConfig) -> SSHConnection:
        """获取连接，复用现有连接或创建新连接"""
        key = self._get_connection_key(config)

        with self._lock:
            if key in self._connections:
                conn = self._connections[key]
                if conn.is_alive():
                    return conn
                else:
                    self._remove_connection(key)

            if len(self._connections) >= self.max_connections:
                self._cleanup_dead()

            conn = SSHConnection(config=config)
            self._connections[key] = conn
            return conn

    def _remove_connection(self, key: str) -> None:
        """移除连接"""
        if key in self._connections:
            self._connections[key].close()
            del self._connections[key]

    def _cleanup_dead(self) -> None:
        """清理死连接"""
        dead_keys = [
            key for key, conn in self._connections.items() if not conn.is_alive()
        ]
        for key in dead_keys:
            self._remove_connection(key)

    def close_all(self) -> None:
        """关闭所有连接"""
        with self._lock:
            for key in list(self._connections.keys()):
                self._remove_connection(key)

    def start_heartbeat_monitor(self) -> None:
        """启动心跳检测线程"""
        if self._running:
            return
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    def stop_heartbeat_monitor(self) -> None:
        """停止心跳检测线程"""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1)
            self._heartbeat_thread = None

    def _heartbeat_loop(self) -> None:
        """心跳检测主循环"""
        while self._running:
            self._check_heartbeats()
            time.sleep(self.heartbeat_interval)

    def _check_heartbeats(self) -> None:
        """检查所有连接的心跳"""
        now = time.time()
        with self._lock:
            for key, conn in list(self._connections.items()):
                if not conn.is_alive():
                    self._remove_connection(key)
                    continue

                if now - conn.last_heartbeat >= self.heartbeat_interval:
                    if self._send_heartbeat(conn):
                        conn.last_heartbeat = now
                        conn.heartbeat_failures = 0
                    else:
                        conn.heartbeat_failures += 1
                        if conn.heartbeat_failures >= self.max_heartbeat_failures:
                            self._remove_connection(key)

    def _send_heartbeat(self, conn: SSHConnection) -> bool:
        """发送心跳检测"""
        try:
            if conn.process is None or conn.process.stdin is None:
                return False
            heartbeat_cmd = "echo 'HEARTBEAT'\n"
            conn.process.stdin.write(heartbeat_cmd)
            conn.process.stdin.flush()
            return True
        except Exception:
            return False


class RemoteWorker:
    """
    通过 SSH 在远程机器上执行测试的 RemoteWorker 类

    特性：
    - 连接池复用 SSH 连接，减少连接建立开销
    - 定期心跳检测监控连接状态，自动清理失效连接
    - 测试结果异步回传，支持实时获取执行状态
    """

    def __init__(
        self,
        connection_pool: SSHConnectionPool | None = None,
        default_timeout: int = 300,
    ) -> None:
        if connection_pool is None:
            connection_pool = SSHConnectionPool()
        self.pool = connection_pool
        self.default_timeout = default_timeout
        self._result_callbacks: list[Callable[[dict[str, Any]], None]] = []

    def add_result_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """添加结果回调函数，用于实时接收测试结果"""
        self._result_callbacks.append(callback)

    def _build_ssh_command(self, config: SSHConfig, command: str) -> str:
        """构建 SSH 命令"""
        ssh_cmd = ["ssh"]
        ssh_cmd.extend(["-p", str(config.port)])
        if config.key_filename:
            ssh_cmd.extend(["-i", config.key_filename])
        ssh_cmd.extend(["-o", "ConnectTimeout=%d" % config.timeout])
        if config.username:
            ssh_cmd.append(f"{config.username}@{config.host}")
        else:
            ssh_cmd.append(config.host)

        if config.working_dir:
            command = f"cd {config.working_dir} && {command}"

        ssh_cmd.append(command)
        return " ".join(ssh_cmd)

    def connect(self, config: SSHConfig) -> SSHConnection:
        """建立 SSH 连接并启动远程 Python 会话"""
        conn = self.pool.get_connection(config)
        if conn.is_alive():
            return conn

        command = config.remote_python + " -u"
        ssh_cmd = self._build_ssh_command(config, command)

        try:
            process = subprocess.Popen(
                ssh_cmd,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            conn.close()
            raise RuntimeError(f"Failed to connect to {config.host}: {e}") from e

        conn.process = process
        conn.connected = True
        conn.last_heartbeat = time.time()
        conn.heartbeat_failures = 0

        def read_output() -> None:
            """读取远程输出"""
            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                if not conn.connected:
                    break
                line = line.rstrip()
                if line:
                    self._parse_output(conn, line)

        conn.stdout_thread = threading.Thread(target=read_output, daemon=True)
        conn.stdout_thread.start()

        return conn

    def _parse_output(self, conn: SSHConnection, line: str) -> None:
        """解析远程输出，提取结果"""
        if line == "HEARTBEAT":
            conn.last_heartbeat = time.time()
            return

        if line.startswith('{"type":') and line.endswith("}"):
            try:
                data = json.loads(line)
                conn.result_queue.put(data)
                for callback in self._result_callbacks:
                    try:
                        callback(data)
                    except Exception:
                        pass
            except json.JSONDecodeError:
                pass
        else:
            pass

    def execute_test(
        self,
        config: SSHConfig,
        test_path: str,
        test_name: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """在远程执行单个测试用例并返回结果"""
        conn = self.connect(config)
        if timeout is None:
            timeout = self.default_timeout

        pytest_args = [test_path]
        if test_name:
            pytest_args[-1] += f"::{test_name}"

        pytest_args.extend(["-v", "--json-report"])
        remote_script = self._build_remote_execution(pytest_args)

        assert conn.process is not None
        assert conn.process.stdin is not None

        try:
            conn.process.stdin.write(remote_script + "\n")
            conn.process.stdin.flush()

            start_time = time.time()
            result: dict[str, Any] = {"status": "error", "message": "timeout"}

            while time.time() - start_time < timeout:
                try:
                    data = conn.result_queue.get(timeout=1)
                    if data.get("type") == "test_result":
                        result = data
                        break
                except queue.Empty:
                    if not conn.is_alive():
                        result = {"status": "error", "message": "connection closed"}
                        break
                    continue

            return result

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def execute_tests(
        self,
        config: SSHConfig,
        test_paths: list[str],
        timeout: int | None = None,
    ) -> list[dict[str, Any]]:
        """在远程执行多个测试用例"""
        conn = self.connect(config)
        if timeout is None:
            timeout = self.default_timeout

        pytest_args = test_paths + ["-v", "--json-report"]
        remote_script = self._build_remote_execution(pytest_args)

        assert conn.process is not None
        assert conn.process.stdin is not None

        try:
            conn.process.stdin.write(remote_script + "\n")
            conn.process.stdin.flush()

            start_time = time.time()
            results: list[dict[str, Any]] = []

            while time.time() - start_time < timeout:
                try:
                    data = conn.result_queue.get(timeout=1)
                    if data.get("type") == "test_result":
                        results.append(data)
                    elif data.get("type") == "session_finish":
                        break
                except queue.Empty:
                    if not conn.is_alive():
                        results.append(
                            {"status": "error", "message": "connection closed"}
                        )
                        break
                    continue

            return results

        except Exception as e:
            return [{"status": "error", "message": str(e)}]

    def _build_remote_execution(self, pytest_args: list[str]) -> str:
        """构建远程执行脚本代码，用于回传结果"""
        script = (
            """
import json
import sys
import pytest

def json_report_callback(data):
    print(json.dumps({"type": "test_result", **data}), flush=True)

def pytest_sessionfinish(session, exitstatus):
    print(json.dumps({"type": "session_finish", "exitstatus": exitstatus}), flush=True)

def pytest_runtest_logreport(report):
    if report.when == 'call':
        result = {
            "nodeid": report.nodeid,
            "outcome": report.outcome,
            "duration": report.duration,
            "passed": report.passed,
            "failed": report.failed,
            "skipped": report.skipped,
            "longrepr": str(report.longrepr) if report.longrepr else None
        }
        json_report_callback(result)

pytest.main(%r)
"""
            % pytest_args
        )
        return script

    def collect_tests(
        self, config: SSHConfig, test_paths: list[str]
    ) -> list[dict[str, Any]]:
        """收集远程机器上的测试用例"""
        conn = self.connect(config)

        script = """
import json
import sys
import pytest

class Collector:
    def __init__(self):
        self.tests = []
    
    def pytest_collection_modifyitems(self, config, items):
        for item in items:
            self.tests.append({
                "nodeid": item.nodeid,
                "name": item.name,
                "location": item.location
            })
    
    def pytest_sessionfinish(self, session, exitstatus):
        print(json.dumps({"type": "collection", "tests": self.tests}), flush=True)

collector = Collector()
pytest.main(%r + ["--collect-only", "-q"], collector)
""" % (
            test_paths,
        )

        assert conn.process is not None
        assert conn.process.stdin is not None

        try:
            conn.process.stdin.write(script + "\n")
            conn.process.stdin.flush()

            while True:
                try:
                    data = conn.result_queue.get(timeout=self.default_timeout)
                    if data.get("type") == "collection":
                        return data.get("tests", [])
                except queue.Empty:
                    if not conn.is_alive():
                        break
                    continue
        except Exception:
            pass

        return []

    def get_pending_results(self, config: SSHConfig) -> list[dict[str, Any]]:
        """获取所有待处理的结果"""
        conn = self.pool.get_connection(config)
        results: list[dict[str, Any]] = []
        try:
            while True:
                try:
                    results.append(conn.result_queue.get_nowait())
                except queue.Empty:
                    break
        except Exception:
            pass
        return results

    def health_check(self) -> dict[str, int]:
        """获取当前连接池健康状态"""
        with self.pool._lock:
            total = len(self.pool._connections)
            alive = sum(1 for conn in self.pool._connections.values() if conn.is_alive())
            return {
                "total_connections": total,
                "active_connections": alive,
                "dead_connections": total - alive,
            }

    def close_connection(self, config: SSHConfig) -> None:
        """关闭指定连接"""
        key = self.pool._get_connection_key(config)
        with self.pool._lock:
            if key in self.pool._connections:
                self.pool._remove_connection(key)

    def close(self) -> None:
        """关闭所有连接"""
        self.pool.close_all()
        self.pool.stop_heartbeat_monitor()
