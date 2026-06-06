from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Sequence
from typing import NoReturn

import pytest

from xdist.remote import Producer
from xdist.report import report_collection_diff
from xdist.workermanage import parse_tx_spec_config
from xdist.workermanage import WorkerController


def get_test_file_size(nodeid: str) -> int:
    """
    获取测试文件的大小（字节）。
    
    :param nodeid: 测试用例的完整标识符
    :return: 文件大小（字节），如果无法获取则返回默认值 1000
    """
    try:
        # 从 nodeid 中提取文件路径（在第一个 '::' 之前）
        filepath = nodeid.split("::", 1)[0]
        if os.path.exists(filepath):
            return os.path.getsize(filepath)
    except (ValueError, OSError):
        pass
    # 如果无法获取文件大小，返回默认值
    return 1000


def get_historical_execution_time(nodeid: str, history_file: str | None = None) -> float:
    """
    获取测试用例的历史执行时间。
    
    :param nodeid: 测试用例的完整标识符
    :param history_file: 历史数据文件路径，默认为 .pytest-xdist-history.json
    :return: 历史执行时间（秒），如果没有历史数据则返回默认值 1.0
    """
    if history_file is None:
        history_file = ".pytest-xdist-history.json"
    
    try:
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history = json.load(f)
                if nodeid in history:
                    return float(history[nodeid])
    except (json.JSONDecodeError, OSError, ValueError):
        pass
    # 如果没有历史数据，返回默认值
    return 1.0


def save_execution_time(nodeid: str, duration: float, history_file: str | None = None) -> None:
    """
    保存测试用例的执行时间到历史文件。
    
    :param nodeid: 测试用例的完整标识符
    :param duration: 执行时间（秒）
    :param history_file: 历史数据文件路径，默认为 .pytest-xdist-history.json
    """
    if history_file is None:
        history_file = ".pytest-xdist-history.json"
    
    try:
        history = {}
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history = json.load(f)
        
        history[nodeid] = duration
        
        with open(history_file, "w") as f:
            json.dump(history, f)
    except (json.JSONDecodeError, OSError):
        pass


class SmartLoadGroupScheduling:
    """
    实现跨节点的智能负载均衡调度，支持按文件大小或历史执行时间分组。
    
    这个调度器继承了 LoadScopeScheduling 的功能，但增加了智能分组能力，
    能够根据测试文件的大小或历史执行时间来平衡各个节点的工作负载，
    避免任务分布不均的情况。
    """
    
    def __init__(self, config: pytest.Config, log: Producer | None = None) -> None:
        self.numnodes = len(parse_tx_spec_config(config))
        self.collection: list[str] | None = None
        
        self.workqueue: OrderedDict[str, dict[str, bool]] = OrderedDict()
        self.assigned_work: dict[WorkerController, dict[str, dict[str, bool]]] = {}
        self.registered_collections: dict[WorkerController, list[str]] = {}
        self.scope_weights: dict[str, float] = {}
        
        if log is None:
            self.log = Producer("smartloadgroupsched")
        else:
            self.log = log.smartloadgroupsched
        
        self.config = config
        self.grouping_strategy = config.option.loadgroup or "size"
    
    @property
    def nodes(self) -> list[WorkerController]:
        """A list of all active nodes in the scheduler."""
        return list(self.assigned_work.keys())
    
    @property
    def collection_is_completed(self) -> bool:
        """Boolean indication initial test collection is complete."""
        return len(self.registered_collections) >= self.numnodes
    
    @property
    def tests_finished(self) -> bool:
        """Return True if all tests have been executed by the nodes."""
        if not self.collection_is_completed:
            return False
        
        if self.workqueue:
            return False
        
        for assigned_unit in self.assigned_work.values():
            if self._pending_of(assigned_unit) >= 2:
                return False
        
        return True
    
    @property
    def has_pending(self) -> bool:
        """Return True if there are pending test items."""
        if self.workqueue:
            return True
        
        for assigned_unit in self.assigned_work.values():
            if self._pending_of(assigned_unit) > 0:
                return True
        
        return False
    
    def add_node(self, node: WorkerController) -> None:
        """Add a new node to the scheduler."""
        assert node not in self.assigned_work
        self.assigned_work[node] = {}
    
    def remove_node(self, node: WorkerController) -> str | None:
        """Remove a node from the scheduler."""
        workload = self.assigned_work.pop(node)
        if not self._pending_of(workload):
            return None
        
        for work_unit in workload.values():
            for nodeid, completed in work_unit.items():
                if not completed:
                    crashitem = nodeid
                    break
            else:
                continue
            break
        else:
            raise RuntimeError(
                "Unable to identify crashitem on a workload with pending items"
            )
        
        self.workqueue.update(workload)
        
        for node in self.assigned_work:
            self._reschedule(node)
        
        return crashitem
    
    def add_node_collection(
        self, node: WorkerController, collection: Sequence[str]
    ) -> None:
        """Add the collected test items from a node."""
        assert node in self.assigned_work
        
        if self.collection_is_completed:
            assert self.collection
            
            if collection != self.collection:
                other_node = next(iter(self.registered_collections.keys()))
                
                msg = report_collection_diff(
                    self.collection, collection, other_node.gateway.id, node.gateway.id
                )
                self.log(msg)
                return
        
        self.registered_collections[node] = list(collection)
    
    def mark_test_complete(
        self, node: WorkerController, item_index: int, duration: float = 0
    ) -> None:
        """Mark test item as completed by node."""
        nodeid = self.registered_collections[node][item_index]
        scope = self._split_scope(nodeid)
        
        self.assigned_work[node][scope][nodeid] = True
        
        # 保存执行时间到历史文件
        save_execution_time(nodeid, duration)
        
        self._reschedule(node)
    
    def mark_test_pending(self, item: str) -> NoReturn:
        raise NotImplementedError()
    
    def remove_pending_tests_from_node(
        self,
        node: WorkerController,
        indices: Sequence[int],
    ) -> None:
        raise NotImplementedError()
    
    def _calculate_scope_weight(self, scope: str, work_unit: dict[str, bool]) -> float:
        """
        计算一个 scope 的权重。
        
        根据配置的策略，权重可以基于文件大小或历史执行时间。
        """
        if self.grouping_strategy == "size":
            # 基于文件大小计算权重
            total_size = 0
            for nodeid in work_unit.keys():
                total_size += get_test_file_size(nodeid)
            return total_size
        elif self.grouping_strategy == "time":
            # 基于历史执行时间计算权重
            total_time = 0.0
            for nodeid in work_unit.keys():
                total_time += get_historical_execution_time(nodeid)
            return total_time
        else:
            # 默认策略：按测试数量
            return len(work_unit)
    
    def _assign_work_unit(self, node: WorkerController) -> None:
        """Assign a work unit to a node."""
        assert self.workqueue
        
        scope, work_unit = self.workqueue.popitem(last=False)
        
        assigned_to_node = self.assigned_work.setdefault(node, {})
        assigned_to_node[scope] = work_unit
        
        worker_collection = self.registered_collections[node]
        nodeids_indexes = [
            worker_collection.index(nodeid)
            for nodeid, completed in work_unit.items()
            if not completed
        ]
        
        node.send_runtest_some(nodeids_indexes)
    
    def _split_scope(self, nodeid: str) -> str:
        """Determine the scope (grouping) of a nodeid."""
        return nodeid.rsplit("::", 1)[0]
    
    def _pending_of(self, workload: dict[str, dict[str, bool]]) -> int:
        """Return the number of pending tests in a workload."""
        pending = sum(list(scope.values()).count(False) for scope in workload.values())
        return pending
    
    def _get_node_current_load(self, node: WorkerController) -> float:
        """获取节点当前的负载权重。"""
        load = 0.0
        for scope in self.assigned_work[node].keys():
            load += self.scope_weights.get(scope, 1.0)
        return load
    
    def _reschedule(self, node: WorkerController) -> None:
        """Maybe schedule new items on the node."""
        if node.shutting_down:
            return
        
        if not self.workqueue:
            node.shutdown()
            return
        
        self.log("Number of units waiting for node:", len(self.workqueue))
        
        if self._pending_of(self.assigned_work[node]) > 2:
            return
        
        self._assign_work_unit(node)
    
    def schedule(self) -> None:
        """Initiate distribution of the test collection."""
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
        
        unsorted_workqueue: dict[str, dict[str, bool]] = {}
        for nodeid in self.collection:
            scope = self._split_scope(nodeid)
            work_unit = unsorted_workqueue.setdefault(scope, {})
            work_unit[nodeid] = False
        
        # 计算每个 scope 的权重
        self.scope_weights = {}
        for scope, work_unit in unsorted_workqueue.items():
            self.scope_weights[scope] = self._calculate_scope_weight(scope, work_unit)
        
        # 按权重从大到小排序 scopes，以便优先分配大任务
        sorted_scopes = sorted(
            unsorted_workqueue.items(),
            key=lambda item: -self.scope_weights[item[0]]
        )
        
        for scope, nodeids in sorted_scopes:
            self.workqueue[scope] = nodeids
        
        extra_nodes = len(self.nodes) - len(self.workqueue)
        
        if extra_nodes > 0:
            self.log(f"Shutting down {extra_nodes} nodes")
            
            for _ in range(extra_nodes):
                unused_node, _assigned = self.assigned_work.popitem()
                
                self.log(f"Shutting down unused node {unused_node}")
                unused_node.shutdown()
        
        # 智能分配初始工作负载 - 最小-最大分配策略
        self._smart_initial_distribution()
        
        for node in self.nodes:
            self._reschedule(node)
        
        if not self.workqueue:
            for node in self.nodes:
                node.shutdown()
    
    def _smart_initial_distribution(self) -> None:
        """
        使用智能策略分配初始工作负载。
        
        这个方法使用最小-最大分配策略，确保每个节点的负载尽可能均衡。
        """
        # 为每个节点创建一个负载跟踪器
        node_loads: dict[WorkerController, float] = {
            node: 0.0 for node in self.nodes
        }
        
        # 按权重降序处理工作单元
        work_items = list(self.workqueue.items())
        
        # 重新创建 workqueue 以便正确分配
        temp_workqueue = OrderedDict(self.workqueue)
        self.workqueue.clear()
        
        for scope, work_unit in work_items:
            # 找到当前负载最小的节点
            min_load_node = min(node_loads.items(), key=lambda x: x[1])[0]
            
            # 将工作单元分配给负载最小的节点
            assigned_to_node = self.assigned_work.setdefault(min_load_node, {})
            assigned_to_node[scope] = work_unit
            
            # 更新该节点的负载
            node_loads[min_load_node] += self.scope_weights[scope]
            
            # 发送测试给节点
            worker_collection = self.registered_collections[min_load_node]
            nodeids_indexes = [
                worker_collection.index(nodeid)
                for nodeid, completed in work_unit.items()
                if not completed
            ]
            
            min_load_node.send_runtest_some(nodeids_indexes)
    
    def _check_nodes_have_same_collection(self) -> bool:
        """Return True if all nodes have collected the same items."""
        node_collection_items = list(self.registered_collections.items())
        first_node, col = node_collection_items[0]
        same_collection = True
        
        for node, collection in node_collection_items[1:]:
            msg = report_collection_diff(
                col, collection, first_node.gateway.id, node.gateway.id
            )
            if not msg:
                continue
            
            same_collection = False
            self.log(msg)
            
            rep = pytest.CollectReport(
                nodeid=node.gateway.id,
                outcome="failed",
                longrepr=msg,
                result=[],
            )
            self.config.hook.pytest_collectreport(report=rep)
        
        return same_collection
