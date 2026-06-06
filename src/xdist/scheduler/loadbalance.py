from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pytest

from xdist.remote import Producer
from xdist.scheduler.loadscope import LoadScopeScheduling
from xdist.workermanage import WorkerController


class LoadBalanceScheduling(LoadScopeScheduling):
    """Intelligent load balancing scheduling across nodes.

    Groups tests by file and distributes work units across workers
    to balance the total expected cost based on file size or
    historical execution time.

    Supports two modes via ``--load-group``:

    - ``size``: Weight work units by source file size.
    - ``time``: Weight work units by historical execution duration
      read from ``.pytest_cache/xdist_durations/durations.json``.
    """

    def __init__(self, config: pytest.Config, log: Producer | None = None) -> None:
        super().__init__(config, log)
        if log is None:
            self.log = Producer("loadbalancesched")
        else:
            self.log = log.loadbalancesched
        self._load_group: str = config.getoption("load_group", "size")
        self._node_unsent: dict[WorkerController, OrderedDict[str, dict[str, bool]]] = {}

    def _split_scope(self, nodeid: str) -> str:
        return nodeid.split("::", 1)[0]

    def _get_weight(self, scope: str, work_unit: dict[str, bool]) -> float:
        if self._load_group == "time":
            return self._get_file_duration(scope)
        return float(self._get_file_size(scope))

    def _get_file_size(self, filepath: str) -> int:
        path = Path(filepath)
        if not path.is_absolute():
            path = self.config.rootpath / filepath
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def _get_file_duration(self, filepath: str) -> float:
        try:
            cache_dir = self.config.cache.makedir("xdist_durations")
        except AttributeError:
            return 0.0
        durations_path = Path(cache_dir) / "durations.json"
        if not durations_path.is_file():
            return 0.0
        try:
            with open(durations_path, "r") as f:
                durations = json.load(f)
            return float(durations.get(filepath, 0.0))
        except (json.JSONDecodeError, ValueError):
            return 0.0

    def schedule(self) -> None:
        assert self.collection_is_completed

        if self.collection is not None:
            for node in self.nodes:
                self._reschedule(node)
            return

        if not self._check_nodes_have_same_collection():
            self.log("**Different tests collected, aborting run**")
            return

        self.collection = list(next(iter(self.registered_collections.values())))
        if not self.collection:
            return

        unsorted: dict[str, dict[str, bool]] = {}
        for nodeid in self.collection:
            scope = self._split_scope(nodeid)
            work_unit = unsorted.setdefault(scope, {})
            work_unit[nodeid] = False

        weighted = [
            (scope, self._get_weight(scope, unit), unit)
            for scope, unit in unsorted.items()
        ]
        weighted.sort(key=lambda x: x[1], reverse=True)

        nodes = list(self.assigned_work.keys())
        if not nodes:
            return

        loads = [0.0 for _ in nodes]
        node_assignments: list[OrderedDict[str, dict[str, bool]]] = [
            OrderedDict() for _ in nodes
        ]

        for scope, weight, unit in weighted:
            idx = min(range(len(nodes)), key=lambda i: loads[i])
            node_assignments[idx][scope] = unit
            loads[idx] += weight

        for i, node in enumerate(nodes):
            if node_assignments[i]:
                self.assigned_work[node] = node_assignments[i]
                self._node_unsent[node] = OrderedDict(node_assignments[i])
            else:
                self.assigned_work.pop(node)
                node.shutdown()

        for node in self.nodes:
            self._send_one(node)
        for node in self.nodes:
            self._send_one(node)

        if not self._has_unsent():
            for node in self.nodes:
                node.shutdown()

    def _has_unsent(self) -> bool:
        return any(bool(u) for u in self._node_unsent.values())

    def _send_one(self, node: WorkerController) -> None:
        unsent = self._node_unsent.get(node)
        if not unsent:
            return
        scope, work_unit = unsent.popitem(last=False)
        worker_collection = self.registered_collections[node]
        nodeids_indexes = [
            worker_collection.index(nodeid)
            for nodeid, completed in work_unit.items()
            if not completed
        ]
        node.send_runtest_some(nodeids_indexes)

    def _reschedule(self, node: WorkerController) -> None:
        if node.shutting_down:
            return

        unsent = self._node_unsent.get(node, OrderedDict())
        if not unsent:
            node.shutdown()
            return

        pending = self._pending_of(self.assigned_work.get(node, {}))
        if pending > 2:
            return

        self._send_one(node)

    def remove_node(self, node: WorkerController) -> str | None:
        workload = self.assigned_work.pop(node, {})
        unsent = self._node_unsent.pop(node, OrderedDict())

        if not self._pending_of(workload):
            return None

        crashitem: str | None = None
        for scope, work_unit in workload.items():
            if scope in unsent:
                continue
            for nodeid, completed in work_unit.items():
                if not completed:
                    crashitem = nodeid
                    break
            if crashitem:
                break

        if unsent and self.assigned_work:
            remaining_nodes = list(self.assigned_work.keys())
            for scope, work_unit in unsent.items():
                idx = min(
                    range(len(remaining_nodes)),
                    key=lambda i: len(self._node_unsent[remaining_nodes[i]]),
                )
                self._node_unsent[remaining_nodes[idx]][scope] = work_unit
                self.assigned_work[remaining_nodes[idx]][scope] = work_unit

        for n in self.assigned_work:
            self._reschedule(n)

        return crashitem