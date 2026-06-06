import os
import json
from typing import List, Dict

import pytest
from xdist.remote import Producer
from xdist.scheduler.loadscope import LoadScopeScheduling

def get_file_size(filepath: str) -> int:
    try:
        return os.path.getsize(filepath)
    except OSError:
        return 0

def get_history_time(filepath: str, history_file: str = ".pytest_xdist_history.json") -> float:
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
            return history.get(filepath, 0.0)
    except Exception:
        pass
    return 0.0

def smart_group_tests(nodeids: List[str], strategy: str, num_groups: int = 4) -> Dict[str, str]:
    """
    Intelligently group tests based on file size or history execution time.
    Returns a mapping of nodeid to its assigned group name.
    """
    # Calculate weights for each file
    file_weights = {}
    for nodeid in nodeids:
        filepath = nodeid.split("::")[0]
        if filepath not in file_weights:
            if strategy == "size":
                file_weights[filepath] = get_file_size(filepath)
            elif strategy == "history":
                file_weights[filepath] = get_history_time(filepath)
            else:
                file_weights[filepath] = 1  # Default fallback weight
                
    # Greedy bin-packing algorithm to balance the load among num_groups
    sorted_files = sorted(file_weights.items(), key=lambda x: x[1], reverse=True)
    
    group_loads = {f"group_{i}": 0.0 for i in range(num_groups)}
    file_to_group = {}
    
    for filepath, weight in sorted_files:
        # Find the group with the minimum current load
        min_group = min(group_loads, key=group_loads.get)
        file_to_group[filepath] = min_group
        group_loads[min_group] += weight

    # Map each nodeid to its assigned group
    nodeid_to_group = {}
    for nodeid in nodeids:
        filepath = nodeid.split("::")[0]
        nodeid_to_group[nodeid] = file_to_group.get(filepath, "group_0")
        
    return nodeid_to_group

class LoadBalanceScheduling(LoadScopeScheduling):
    """
    Implement load scheduling across nodes by intelligently grouping tests
    based on test file size or history execution time.
    """
    def __init__(self, config: pytest.Config, log: Producer | None = None) -> None:
        super().__init__(config, log)
        if log is None:
            self.log = Producer("loadbalancesched")
        else:
            self.log = log.loadbalancesched
        self._nodeid_to_group: Dict[str, str] = {}
        self._strategy = config.getoption("loadgroup_strategy", "none")

    def _check_nodes_have_same_collection(self) -> bool:
        same = super()._check_nodes_have_same_collection()
        if same and not self._nodeid_to_group:
            collection = list(next(iter(self.registered_collections.values())))
            num_workers = len(self.nodes) if self.nodes else 1
            if self._strategy in ("size", "history"):
                self._nodeid_to_group = smart_group_tests(collection, self._strategy, num_workers)
        return same

    def _split_scope(self, nodeid: str) -> str:
        if self._strategy in ("size", "history") and nodeid in self._nodeid_to_group:
            return self._nodeid_to_group[nodeid]
        return super()._split_scope(nodeid)
