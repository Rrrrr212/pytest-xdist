from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any
from typing import Literal


LoadGroupMode = Literal["filesize", "duration"]


def load_duration_map(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}

    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        return {}

    durations: dict[str, float] = {}
    for key, value in raw.items():
        duration = _extract_duration(value)
        if duration is not None and duration > 0:
            durations[str(key)] = duration
    return durations


def build_load_groups(
    collection: Sequence[str],
    num_groups: int,
    rootpath: Path,
    mode: LoadGroupMode,
    duration_map: Mapping[str, float] | None = None,
) -> list[list[int]]:
    if num_groups < 1:
        raise ValueError("num_groups must be greater than 0")
    if not collection:
        return []

    file_groups = _group_collection_by_file(collection)
    bucket_count = min(num_groups, len(file_groups))
    buckets: list[list[int]] = [[] for _ in range(bucket_count)]
    bucket_weights = [0.0] * bucket_count

    group_weights = {
        test_file: _resolve_group_weight(
            collection=collection,
            indices=indices,
            rootpath=rootpath,
            test_file=test_file,
            mode=mode,
            duration_map=duration_map,
        )
        for test_file, indices in file_groups.items()
    }

    for test_file, indices in sorted(
        file_groups.items(),
        key=lambda item: (-group_weights[item[0]], item[0]),
    ):
        target = min(range(bucket_count), key=lambda index: (bucket_weights[index], index))
        buckets[target].extend(indices)
        bucket_weights[target] += group_weights[test_file]

    for bucket in buckets:
        bucket.sort()

    return [bucket for bucket in buckets if bucket]


def _group_collection_by_file(collection: Sequence[str]) -> OrderedDict[str, list[int]]:
    file_groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, nodeid in enumerate(collection):
        test_file = _resolve_test_file(nodeid)
        file_groups.setdefault(test_file, []).append(index)
    return file_groups


def _resolve_group_weight(
    collection: Sequence[str],
    indices: Sequence[int],
    rootpath: Path,
    test_file: str,
    mode: LoadGroupMode,
    duration_map: Mapping[str, float] | None,
) -> float:
    if mode == "duration":
        duration_weight = _resolve_duration_weight(
            collection=collection,
            indices=indices,
            test_file=test_file,
            duration_map=duration_map,
        )
        if duration_weight > 0:
            return duration_weight

    file_size = _resolve_file_size(rootpath, test_file)
    return float(max(file_size, len(indices), 1))


def _resolve_duration_weight(
    collection: Sequence[str],
    indices: Sequence[int],
    test_file: str,
    duration_map: Mapping[str, float] | None,
) -> float:
    if not duration_map:
        return 0.0

    weight = sum(max(duration_map.get(collection[index], 0.0), 0.0) for index in indices)
    if weight > 0:
        return weight

    return max(duration_map.get(test_file, 0.0), 0.0)


def _resolve_file_size(rootpath: Path, test_file: str) -> int:
    file_path = Path(test_file)
    if not file_path.is_absolute():
        file_path = rootpath / file_path

    try:
        return file_path.stat().st_size
    except OSError:
        return 0


def _resolve_test_file(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def _extract_duration(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, dict):
        for key in ("duration", "seconds", "elapsed"):
            duration = value.get(key)
            if isinstance(duration, (int, float)):
                return float(duration)

    return None
