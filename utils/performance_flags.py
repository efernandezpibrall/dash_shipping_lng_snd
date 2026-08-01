"""Environment-controlled rollback switches for dashboard performance work."""

from __future__ import annotations

import os


REVISION_AWARE_REFRESH_ENV = "DASH_REVISION_AWARE_REFRESH_ENABLED"
FLEET_ARROW_SOURCE_ENV = "DASH_FLEET_ARROW_SOURCE_ENABLED"
FLEET_RENDER_SNAPSHOT_ENV = "DASH_FLEET_RENDER_SNAPSHOT_ENABLED"
FLEET_STAGED_RENDER_ENV = "DASH_FLEET_STAGED_RENDER_ENABLED"


def env_flag(name: str, *, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().casefold() not in {
        "0",
        "false",
        "no",
        "off",
        "disabled",
    }


def revision_aware_refresh_enabled() -> bool:
    return env_flag(REVISION_AWARE_REFRESH_ENV, default=False)


def fleet_arrow_source_enabled() -> bool:
    return env_flag(FLEET_ARROW_SOURCE_ENV, default=False)


def fleet_render_snapshot_enabled() -> bool:
    return env_flag(FLEET_RENDER_SNAPSHOT_ENV, default=False)


def fleet_staged_render_enabled() -> bool:
    return env_flag(FLEET_STAGED_RENDER_ENV, default=False)
