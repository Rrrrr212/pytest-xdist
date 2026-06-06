from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import pytest

from xdist.remote import Producer
from xdist.workermanage import WorkerController

from .loadscope import LoadScopeScheduling


DURATIONS_CACHE_FILENAME = ".xdist_durations.json"
DEFAULT_DURATION = 0.01


def get_file_size(filepath: str) -> int:
    try:
        return os.path.getsize(filepath)
    except OSError:
        return 0


def calculate_scope_file_sizes(
    collection: Sequence[str],
) -> dict[str, float]:
    scope_sizes: dict[str, float] = {}
    for nodeid in collection:
        scope = nodeid.split("::", 1)[0]
        if scope not in scope_sizes:
            scope_sizes[scope] = float(get_file_size(scope))
    return scope_sizes


def load_duration_cache(cache_path: str) -> dict[str, float]:
    try:
        with open(cache_path) as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {k: float(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        pass
    return {}


def save_duration_cache(cache_path: str, durations: dict[str, float]) -> None:
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    try:
        with open(cache_path, "w") as f:
            json.dump(durations, f, indent=2, sort_keys=True)
    except OSError:
        pass


def calculate_scope_durations(
    collection: Sequence[str],
    duration_cache: dict[str, float],
) -> dict[str, float]:
    scope_durations: dict[str, float] = {}
    for nodeid in collection:
        scope = nodeid.split("::", 1)[0]
        duration = duration_cache.get(nodeid, DEFAULT_DURATION)
        scope_durations[scope] = scope_durations.get(scope, 0.0) + duration
    return scope_durations


def balance_scopes_by_weight(
    scopes_with_weights: dict[str, float],
    num_nodes: int,
) -> list[list[str]]:
    if num_nodes <= 0:
        return []
    sorted_scopes = sorted(
        scopes_with_weights.items(), key=lambda item: item[1], reverse=True
    )
    node_assignments: list[list[str]] = [[] for _ in range(num_nodes)]
    node_weights: list[float] = [0.0] * num_nodes
    for scope, weight in sorted_scopes:
        min_node = min(range(num_nodes), key=lambda i: node_weights[i])
        node_assignments[min_node].append(scope)
        node_weights[min_node] += weight
    return node_assignments


def get_duration_cache_path(config: pytest.Config) -> str:
    rootdir = Path(str(config.rootdir))
    return str(rootdir / DURATIONS_CACHE_FILENAME)


class LoadBalanceScheduling(LoadScopeScheduling):
    """Implement load scheduling across nodes with intelligent weight-based grouping.

    This scheduler extends LoadScopeScheduling by distributing test scopes
    (files) to workers based on their estimated weight, which can be calculated
    from either file size or historical execution duration. This avoids uneven
    task distribution where some workers finish much earlier than others.

    The algorithm uses LPT (Longest Processing Time first) bin-packing:
    1. Calculate weight for each file scope (by file size or duration)
    2. Sort scopes by weight in descending order
    3. Assign each scope to the worker with the least current total weight

    When using duration mode, historical execution times are read from and
    written to a cache file (.xdist_durations.json) in the project root.
    """

    def __init__(
        self,
        config: pytest.Config,
        log: Producer | None = None,
    ) -> None:
        super().__init__(config, log)
        if log is None:
            self.log = Producer("loadbalancesched")
        else:
            self.log = log.loadbalancesched
        self._load_group_strategy: str = getattr(
            config.option, "loadgroup", None
        ) or "filesize"
        self._duration_cache: dict[str, float] = {}
        self._node_durations: dict[str, float] = {}
        self._duration_cache_path: str = ""

    def _split_scope(self, nodeid: str) -> str:
        return nodeid.split("::", 1)[0]

    def _calculate_scope_weights(self) -> dict[str, float]:
        assert self.collection is not None
        if self._load_group_strategy in ("duration", "auto"):
            return self._calculate_duration_weights()
        return calculate_scope_file_sizes(self.collection)

    def _calculate_duration_weights(self) -> dict[str, float]:
        assert self.collection is not None
        self._duration_cache_path = get_duration_cache_path(self.config)
        self._duration_cache = load_duration_cache(self._duration_cache_path)
        if self._duration_cache:
            return calculate_scope_durations(self.collection, self._duration_cache)
        self.log(
            "No duration cache found, falling back to file size based weighting"
        )
        return calculate_scope_file_sizes(self.collection)

    def _save_durations(self) -> None:
        if not self._node_durations:
            return
        cache_path = self._duration_cache_path or get_duration_cache_path(self.config)
        existing = load_duration_cache(cache_path)
        existing.update(self._node_durations)
        save_duration_cache(cache_path, existing)

    def mark_test_complete(
        self, node: WorkerController, item_index: int, duration: float = 0
    ) -> None:
        if duration > 0 and self._load_group_strategy in ("duration", "auto"):
            nodeid = self.registered_collections[node][item_index]
            self._node_durations[nodeid] = duration
        super().mark_test_complete(node, item_index, duration)
        if self._load_group_strategy in ("duration", "auto") and self.tests_finished:
            self._save_durations()

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

        scope_weights = self._calculate_scope_weights()
        self.log(f"Load-group strategy: {self._load_group_strategy}")
        self.log(f"Scope weights calculated for {len(scope_weights)} scopes")

        unsorted_workqueue: dict[str, dict[str, bool]] = {}
        for nodeid in self.collection:
            scope = self._split_scope(nodeid)
            work_unit = unsorted_workqueue.setdefault(scope, {})
            work_unit[nodeid] = False

        num_nodes = len(self.nodes)
        if num_nodes > 0 and scope_weights:
            node_scope_assignments = balance_scopes_by_weight(
                scope_weights, num_nodes
            )
            self.log(
                "Balanced scope distribution: "
                + ", ".join(
                    f"node{i}={len(scopes)}scopes"
                    for i, scopes in enumerate(node_scope_assignments)
                )
            )
            self.workqueue = OrderedDict()
            node_list = self.nodes
            assigned_scopes: set[str] = set()

            for i, scopes in enumerate(node_scope_assignments):
                if i < len(node_list):
                    node = node_list[i]
                    for scope in scopes:
                        if scope in unsorted_workqueue:
                            work_unit = unsorted_workqueue[scope]
                            assigned_to_node = self.assigned_work.setdefault(node, {})
                            assigned_to_node[scope] = dict(work_unit)
                            assigned_scopes.add(scope)

            for scope, work_unit in unsorted_workqueue.items():
                if scope not in assigned_scopes:
                    self.workqueue[scope] = work_unit
        else:
            for scope, nodeids in unsorted_workqueue.items():
                self.workqueue[scope] = nodeids

        extra_nodes = len(self.nodes) - len(unsorted_workqueue)
        if extra_nodes > 0:
            self.log(f"Shutting down {extra_nodes} nodes")
            for _ in range(extra_nodes):
                unused_node, _assigned = self.assigned_work.popitem()
                self.log(f"Shutting down unused node {unused_node}")
                unused_node.shutdown()

        for node in self.nodes:
            assigned = self.assigned_work.get(node, {})
            if assigned:
                worker_collection = self.registered_collections[node]
                nodeids_indexes = []
                for work_unit in assigned.values():
                    for nodeid, completed in work_unit.items():
                        if not completed:
                            nodeids_indexes.append(worker_collection.index(nodeid))
                if nodeids_indexes:
                    node.send_runtest_some(nodeids_indexes)

        for node in self.nodes:
            self._reschedule(node)

        if not self.workqueue:
            for node in self.nodes:
                node.shutdown()
