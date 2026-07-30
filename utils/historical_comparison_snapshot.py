"""Exact server-side snapshots for historical provider comparisons."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import pandas as pd

from utils.dashboard_snapshot_cache import (
    build_source_key,
    get_or_build_snapshot,
    is_snapshot_reference,
    resolve_snapshot_manifest,
)
from utils.export_flow_data import engine
from utils.provider_flow_snapshot import (
    NAMESPACE as PROVIDER_FLOW_NAMESPACE,
    fetch_provider_flow_mapping_state,
)


NAMESPACE = "historical-provider-comparison-v1"


def _reference_identity(value: Any) -> dict[str, str]:
    if not is_snapshot_reference(value):
        raise ValueError(
            "An exact provider source reference is required for "
            "historical comparisons"
        )
    return {
        "namespace": str(value["namespace"]),
        "source_key": str(value["source_key"]),
        "revision": str(value["revision"]),
    }


def _mapping_state_for_base(base_reference) -> dict[str, str]:
    manifest = resolve_snapshot_manifest(
        base_reference,
        engine,
        expected_namespace=PROVIDER_FLOW_NAMESPACE,
    )
    source_state = manifest.get("source_state")
    if not isinstance(source_state, Mapping):
        raise RuntimeError(
            "Provider source reference has no immutable source state"
        )
    required_fields = ("mapping_hash", "ea_balance_mapping_hash")
    if not all(field in source_state for field in required_fields):
        raise RuntimeError(
            "Provider source reference has incomplete mapping state"
        )
    return {
        field: str(source_state.get(field) or "")
        for field in required_fields
    }


def clear_historical_comparison_source_state() -> None:
    """Compatibility hook; source state now comes from persisted manifests."""


def get_historical_comparison_frame(
    *,
    direction: str,
    base_reference: Any,
    selection: Mapping[str, Any],
    query_dependencies: Mapping[str, Any],
    builder: Callable[[], pd.DataFrame],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Resolve one immutable comparison frame for an exact source revision."""

    base_identity = _reference_identity(base_reference)
    mapping_state = _mapping_state_for_base(base_reference)
    source_key = build_source_key(
        NAMESPACE,
        {
            "direction": str(direction),
            "base_reference": base_identity,
            "mapping_state": mapping_state,
            "selection": dict(selection),
            "query_dependencies": dict(query_dependencies),
        },
    )

    def build_stable_frame() -> pd.DataFrame:
        frame = builder()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Historical comparison builder must return a DataFrame")
        observed_mapping_state = {
            key: str(value or "")
            for key, value in fetch_provider_flow_mapping_state().items()
        }
        if observed_mapping_state != mapping_state:
            raise RuntimeError(
                "Provider comparison mappings changed during snapshot construction"
            )
        return frame

    reference, payload = get_or_build_snapshot(
        engine,
        namespace=NAMESPACE,
        source_key=source_key,
        builder=build_stable_frame,
        manifest={
            "direction": str(direction),
            "base_reference": base_identity,
            "mapping_state": mapping_state,
            "selection": dict(selection),
            "query_dependencies": dict(query_dependencies),
        },
    )
    if not isinstance(payload, pd.DataFrame):
        raise TypeError("Historical comparison snapshot did not resolve a DataFrame")
    return reference, payload.copy(deep=True)
