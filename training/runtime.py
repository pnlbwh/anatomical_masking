"""Opt-in thread bootstrap and live evidence; import before numerical modules."""

from __future__ import annotations

import os
import sys
import time

# The hard-tail thread limits are opt-in and must be established before any numerical
# library is imported. Keep this lexical boundary stdlib-only: later environment edits
# are deliberately unable to turn a marker-absent import into an enabled process.
_HARDTAIL_THREAD_LIMIT_MARKER = "MASKER_HARDTAIL_THREAD_LIMITS"
_HARDTAIL_THREAD_LIMIT_MARKER_VALUE = "1"
_HARDTAIL_THREAD_LIMIT_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}
_HARDTAIL_THREAD_LIMIT_PREIMPORT_MODULES = (
    "numpy", "scipy", "torch", "nibabel", "threadpoolctl", "google.colab",
    "train_model_colab", "train_model", "augment_scan", "synth_masker_dataset",
    "augmentation",
)
_HARDTAIL_THREAD_LIMIT_BOOTSTRAP_MONOTONIC_NS = time.monotonic_ns()
_HARDTAIL_THREAD_LIMIT_BOOTSTRAP_WALL_TIME_NS = time.time_ns()
_HARDTAIL_THREAD_LIMIT_MARKER_REQUESTED = (
    os.environ.get(_HARDTAIL_THREAD_LIMIT_MARKER)
    == _HARDTAIL_THREAD_LIMIT_MARKER_VALUE
)
_HARDTAIL_THREAD_LIMIT_PRELOADED_MODULES = {
    name: bool(name in sys.modules)
    for name in _HARDTAIL_THREAD_LIMIT_PREIMPORT_MODULES
}
_HARDTAIL_THREAD_LIMIT_ORIGINAL_ENVIRONMENT = {
    _HARDTAIL_THREAD_LIMIT_MARKER: os.environ.get(_HARDTAIL_THREAD_LIMIT_MARKER),
    **{
        name: os.environ.get(name)
        for name in _HARDTAIL_THREAD_LIMIT_ENVIRONMENT
    },
}
if _HARDTAIL_THREAD_LIMIT_MARKER_REQUESTED:
    for _hardtail_env_name, _hardtail_env_value in (
            _HARDTAIL_THREAD_LIMIT_ENVIRONMENT.items()):
        os.environ[_hardtail_env_name] = _hardtail_env_value
_HARDTAIL_THREAD_LIMIT_EFFECTIVE_ENVIRONMENT = {
    _HARDTAIL_THREAD_LIMIT_MARKER: os.environ.get(_HARDTAIL_THREAD_LIMIT_MARKER),
    **{
        name: os.environ.get(name)
        for name in _HARDTAIL_THREAD_LIMIT_ENVIRONMENT
    },
}
_HARDTAIL_THREAD_LIMIT_IMPORT_TIMESTAMPS_NS = {}


def _hardtail_record_import_timestamp(name: str) -> None:
    previous = max(
        [_HARDTAIL_THREAD_LIMIT_BOOTSTRAP_MONOTONIC_NS]
        + list(_HARDTAIL_THREAD_LIMIT_IMPORT_TIMESTAMPS_NS.values()))
    observed = time.monotonic_ns()
    while observed <= previous:
        observed = time.monotonic_ns()
    _HARDTAIL_THREAD_LIMIT_IMPORT_TIMESTAMPS_NS[name] = observed


import copy
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
_hardtail_record_import_timestamp("numpy")
import nibabel as nib
_hardtail_record_import_timestamp("nibabel")
import torch
_hardtail_record_import_timestamp("torch")
from threadpoolctl import threadpool_info

_HARDTAIL_THREAD_LIMIT_BOOTSTRAP_ACTIONS = {
    "torch_num_threads": "not_requested",
    "torch_num_interop_threads": "not_requested",
}
_HARDTAIL_THREAD_LIMIT_PERMANENT_FAILURE_CODES = []


def _hardtail_configure_torch_threads(
        *, action_key: str, getter, setter) -> None:
    if not _HARDTAIL_THREAD_LIMIT_MARKER_REQUESTED:
        return
    try:
        current = getter()
    except BaseException:
        _HARDTAIL_THREAD_LIMIT_BOOTSTRAP_ACTIONS[action_key] = "getter_failed"
        _HARDTAIL_THREAD_LIMIT_PERMANENT_FAILURE_CODES.append(
            f"{action_key}_getter_failed")
        return
    if type(current) is int and current == 1:
        _HARDTAIL_THREAD_LIMIT_BOOTSTRAP_ACTIONS[action_key] = "already_one"
        return
    try:
        setter(1)
    except BaseException:
        _HARDTAIL_THREAD_LIMIT_BOOTSTRAP_ACTIONS[action_key] = "setter_failed"
        _HARDTAIL_THREAD_LIMIT_PERMANENT_FAILURE_CODES.append(
            f"{action_key}_setter_failed")
        return
    try:
        post_set = getter()
    except BaseException:
        _HARDTAIL_THREAD_LIMIT_BOOTSTRAP_ACTIONS[action_key] = "getter_failed"
        _HARDTAIL_THREAD_LIMIT_PERMANENT_FAILURE_CODES.append(
            f"{action_key}_postcheck_getter_failed")
        return
    if type(post_set) is not int or post_set != 1:
        _HARDTAIL_THREAD_LIMIT_BOOTSTRAP_ACTIONS[action_key] = "set_to_one"
        _HARDTAIL_THREAD_LIMIT_PERMANENT_FAILURE_CODES.append(
            f"{action_key}_postcheck_not_one")
        return
    _HARDTAIL_THREAD_LIMIT_BOOTSTRAP_ACTIONS[action_key] = "set_to_one"


_hardtail_configure_torch_threads(
    action_key="torch_num_threads",
    getter=torch.get_num_threads,
    setter=torch.set_num_threads,
)
_hardtail_configure_torch_threads(
    action_key="torch_num_interop_threads",
    getter=torch.get_num_interop_threads,
    setter=torch.set_num_interop_threads,
)

_HARDTAIL_THREAD_LIMIT_IDENTITY = {
    "marker": {
        "name": _HARDTAIL_THREAD_LIMIT_MARKER,
        "value": _HARDTAIL_THREAD_LIMIT_MARKER_VALUE,
    },
    "environment": dict(_HARDTAIL_THREAD_LIMIT_ENVIRONMENT),
    "outer_workers": 4,
    "inner_slice_workers": 4,
    "torch_version_policy": "runtime-recorded-live-gated/v1",
    "dataloader_private_api": {
        "iterator_class":
            "torch.utils.data.dataloader._MultiProcessingDataLoaderIter",
        "shutdown_method": "_shutdown_workers",
        "workers_attr": "_workers",
        "pin_thread_attr": "_pin_memory_thread",
        "loader_iterator_attr": "_iterator",
    },
    "colab_bootstrap_cell_source_sha256":
        "b584c576a8083169800bb1b3870869bdafd77d393dd706be922734529d28306e",
    "colab_bootstrap_receipt_protocol":
        "masker-hardtail-colab-first-cell-bootstrap-v1",
    "runtime_metadata_functions": {
        "train_model": "hardtail_thread_limit_metadata",
        "train_model_colab": "hardtail_thread_limit_metadata",
    },
}
_HARDTAIL_THREAD_LIMIT_IDENTITY_KEYS = {
    "marker", "environment", "outer_workers", "inner_slice_workers",
    "torch_version_policy", "dataloader_private_api",
    "colab_bootstrap_cell_source_sha256",
    "colab_bootstrap_receipt_protocol", "runtime_metadata_functions",
}
_HARDTAIL_THREAD_LIMIT_POOL_KEYS = {
    "user_api", "internal_api", "prefix", "num_threads", "version",
    "threading_layer", "architecture",
}
_HARDTAIL_THREAD_LIMIT_CHECK_KEYS = {
    "stable_identity_contract_exact", "cached_marker_requested",
    "lexical_effective_environment_exact",
    "import_timestamps_complete_and_after_bootstrap",
    "permanent_failure_codes_empty", "live_environment_exact",
    "torch_version_actual_nonempty", "torch_num_threads_one",
    "torch_num_interop_threads_one", "threadpool_info_nonempty",
    "threadpool_num_threads_exact_int_at_most_one",
}


def _hardtail_thread_limit_identity_contract_exact(identity) -> bool:
    return (
        isinstance(identity, dict)
        and set(identity) == _HARDTAIL_THREAD_LIMIT_IDENTITY_KEYS
        and identity.get("marker") == {
            "name": _HARDTAIL_THREAD_LIMIT_MARKER,
            "value": _HARDTAIL_THREAD_LIMIT_MARKER_VALUE,
        }
        and identity.get("environment") == _HARDTAIL_THREAD_LIMIT_ENVIRONMENT
        and identity.get("outer_workers") == 4
        and identity.get("inner_slice_workers") == 4
        and identity.get("torch_version_policy")
            == "runtime-recorded-live-gated/v1"
        and identity.get("dataloader_private_api") == {
            "iterator_class":
                "torch.utils.data.dataloader._MultiProcessingDataLoaderIter",
            "shutdown_method": "_shutdown_workers",
            "workers_attr": "_workers",
            "pin_thread_attr": "_pin_memory_thread",
            "loader_iterator_attr": "_iterator",
        }
        and identity.get("colab_bootstrap_cell_source_sha256")
            == "b584c576a8083169800bb1b3870869bdafd77d393dd706be922734529d28306e"
        and identity.get("colab_bootstrap_receipt_protocol")
            == "masker-hardtail-colab-first-cell-bootstrap-v1"
        and identity.get("runtime_metadata_functions") == {
            "train_model": "hardtail_thread_limit_metadata",
            "train_model_colab": "hardtail_thread_limit_metadata",
        }
    )


def _hardtail_sanitized_threadpool_info(rows) -> List[Dict[str, Any]]:
    if not isinstance(rows, (list, tuple)):
        return []
    projected = []
    for row in rows:
        if not isinstance(row, dict):
            return []
        projected.append({key: copy.deepcopy(row.get(key))
                          for key in _HARDTAIL_THREAD_LIMIT_POOL_KEYS})
    return sorted(
        projected,
        key=lambda item: json.dumps(
            item, sort_keys=True, separators=(",", ":"), default=str),
    )


def hardtail_thread_limit_metadata() -> Dict[str, Any]:
    """Return fresh, path-free evidence for the opt-in hard-tail thread contract."""
    live_environment = {
        _HARDTAIL_THREAD_LIMIT_MARKER:
            os.environ.get(_HARDTAIL_THREAD_LIMIT_MARKER),
        **{
            name: os.environ.get(name)
            for name in _HARDTAIL_THREAD_LIMIT_ENVIRONMENT
        },
    }
    try:
        torch_num_threads = torch.get_num_threads()
    except BaseException:
        torch_num_threads = None
    try:
        torch_num_interop_threads = torch.get_num_interop_threads()
    except BaseException:
        torch_num_interop_threads = None
    try:
        pools = _hardtail_sanitized_threadpool_info(threadpool_info())
    except BaseException:
        pools = []
    observation_monotonic_ns = time.monotonic_ns()
    latest_import = max(_HARDTAIL_THREAD_LIMIT_IMPORT_TIMESTAMPS_NS.values())
    while observation_monotonic_ns <= latest_import:
        observation_monotonic_ns = time.monotonic_ns()
    expected_live_environment = {
        _HARDTAIL_THREAD_LIMIT_MARKER: _HARDTAIL_THREAD_LIMIT_MARKER_VALUE,
        **dict(_HARDTAIL_THREAD_LIMIT_ENVIRONMENT),
    }
    import_timestamps_exact = (
        tuple(_HARDTAIL_THREAD_LIMIT_IMPORT_TIMESTAMPS_NS)
        == ("numpy", "nibabel", "torch")
        and all(
            type(value) is int
            and value > _HARDTAIL_THREAD_LIMIT_BOOTSTRAP_MONOTONIC_NS
            for value in _HARDTAIL_THREAD_LIMIT_IMPORT_TIMESTAMPS_NS.values()
        )
        and all(
            first < second
            for first, second in zip(
                _HARDTAIL_THREAD_LIMIT_IMPORT_TIMESTAMPS_NS.values(),
                tuple(_HARDTAIL_THREAD_LIMIT_IMPORT_TIMESTAMPS_NS.values())[1:],
            )
        )
        and observation_monotonic_ns > latest_import
    )
    pool_contract = (
        bool(pools)
        and all(
            isinstance(row, dict)
            and set(row) == _HARDTAIL_THREAD_LIMIT_POOL_KEYS
            and type(row.get("num_threads")) is int
            and row["num_threads"] <= 1
            for row in pools
        )
    )
    checks = {
        "stable_identity_contract_exact":
            _hardtail_thread_limit_identity_contract_exact(
                _HARDTAIL_THREAD_LIMIT_IDENTITY),
        "cached_marker_requested": _HARDTAIL_THREAD_LIMIT_MARKER_REQUESTED,
        "lexical_effective_environment_exact":
            _HARDTAIL_THREAD_LIMIT_EFFECTIVE_ENVIRONMENT
            == expected_live_environment,
        "import_timestamps_complete_and_after_bootstrap":
            import_timestamps_exact,
        "permanent_failure_codes_empty":
            not _HARDTAIL_THREAD_LIMIT_PERMANENT_FAILURE_CODES,
        "live_environment_exact": live_environment == expected_live_environment,
        "torch_version_actual_nonempty": bool(str(torch.__version__)),
        "torch_num_threads_one":
            type(torch_num_threads) is int and torch_num_threads == 1,
        "torch_num_interop_threads_one":
            type(torch_num_interop_threads) is int
            and torch_num_interop_threads == 1,
        "threadpool_info_nonempty": bool(pools),
        "threadpool_num_threads_exact_int_at_most_one": pool_contract,
    }
    if set(checks) != _HARDTAIL_THREAD_LIMIT_CHECK_KEYS:
        raise RuntimeError("hard-tail thread metadata check schema drift")
    return {
        "identity": copy.deepcopy(_HARDTAIL_THREAD_LIMIT_IDENTITY),
        "observation_monotonic_ns": observation_monotonic_ns,
        "lexical_evidence": {
            "bootstrap_monotonic_ns":
                _HARDTAIL_THREAD_LIMIT_BOOTSTRAP_MONOTONIC_NS,
            "bootstrap_wall_time_ns":
                _HARDTAIL_THREAD_LIMIT_BOOTSTRAP_WALL_TIME_NS,
            "marker_requested_at_lexical_boundary":
                _HARDTAIL_THREAD_LIMIT_MARKER_REQUESTED,
            "preloaded_modules": copy.deepcopy(
                _HARDTAIL_THREAD_LIMIT_PRELOADED_MODULES),
            "original_environment": copy.deepcopy(
                _HARDTAIL_THREAD_LIMIT_ORIGINAL_ENVIRONMENT),
            "effective_environment": copy.deepcopy(
                _HARDTAIL_THREAD_LIMIT_EFFECTIVE_ENVIRONMENT),
            "import_timestamps_ns": copy.deepcopy(
                _HARDTAIL_THREAD_LIMIT_IMPORT_TIMESTAMPS_NS),
        },
        "bootstrap_actions": copy.deepcopy(
            _HARDTAIL_THREAD_LIMIT_BOOTSTRAP_ACTIONS),
        "permanent_failure_codes": list(
            _HARDTAIL_THREAD_LIMIT_PERMANENT_FAILURE_CODES),
        "environment": live_environment,
        "torch_version_actual": str(torch.__version__),
        "torch_num_threads": torch_num_threads,
        "torch_num_interop_threads": torch_num_interop_threads,
        "threadpool_info": pools,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _hardtail_identity_contains_path_key(value) -> bool:
    if isinstance(value, dict):
        return any(
            "path" in str(key).lower()
            or _hardtail_identity_contains_path_key(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_hardtail_identity_contains_path_key(child) for child in value)
    return False


def _validate_hardtail_thread_limit_activation(num_workers):
    if type(num_workers) is not int or num_workers != 4:
        raise ValueError(
            "positive hard_artifact_tail_fraction requires exact built-in int "
            "num_workers == 4")
    logical = os.cpu_count() or 0
    if type(logical) is not int or logical < 16 or 4 * 4 > logical:
        raise RuntimeError(
            "positive hard_artifact_tail_fraction requires at least 16 logical "
            "CPUs for the frozen 4 outer x 4 inner worker budget")
    metadata = hardtail_thread_limit_metadata()
    required_record_keys = {
        "identity", "observation_monotonic_ns", "lexical_evidence",
        "bootstrap_actions", "permanent_failure_codes", "environment",
        "torch_version_actual", "torch_num_threads", "torch_num_interop_threads",
        "threadpool_info", "checks", "passed",
    }
    if (not isinstance(metadata, dict)
            or set(metadata) != required_record_keys
            or metadata.get("passed") is not True
            or not isinstance(metadata.get("checks"), dict)
            or set(metadata["checks"]) != _HARDTAIL_THREAD_LIMIT_CHECK_KEYS
            or not all(metadata["checks"].values())
            or not _hardtail_thread_limit_identity_contract_exact(
                metadata.get("identity"))):
        raise RuntimeError(
            "positive hard_artifact_tail_fraction requires an effective pre-import "
            "MASKER_HARDTAIL_THREAD_LIMITS=1 bootstrap with exact live thread caps")
    return copy.deepcopy(metadata["identity"]), metadata


