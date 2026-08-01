#!/usr/bin/env python3
"""Precompute the canonical Fleet source and staged render bundle."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build Fleet's default source and render snapshots with read-only "
            "SQL and the same persistent cache used by Dash."
        )
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.cache_dir is not None:
        os.environ["DASHBOARD_SNAPSHOT_CACHE_DIR"] = str(
            args.cache_dir.expanduser().resolve(strict=False)
        )
    os.environ.setdefault(
        "DASHBOARD_SNAPSHOT_LOCAL_PERSISTENCE_ENABLED",
        "1",
    )
    # Precompute the new immutable artifacts before the serving workers enable
    # the corresponding traffic switches.
    os.environ.setdefault("DASH_FLEET_ARROW_SOURCE_ENABLED", "1")
    os.environ.setdefault("DASH_FLEET_RENDER_SNAPSHOT_ENABLED", "1")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from pages import fleet_metrics
    from utils import dashboard_snapshot_cache as snapshots
    from utils.detail_snapshot_precompute import audit_read_only_sql

    if not snapshots.local_snapshot_persistence_enabled():
        raise SystemExit("Fleet precompute requires local snapshot persistence")
    snapshots._get_persistent_stores()
    started = time.perf_counter()
    try:
        with audit_read_only_sql([fleet_metrics.engine]) as sql_audit:
            result = fleet_metrics.precompute_default_fleet_metrics()

        snapshots.clear_local_snapshots()
        snapshots.close_persistent_snapshot_cache()
        fleet_metrics._resolve_fleet_render_bundle(
            result["bundle_reference"]
        )
        for section in ("summary", "signals", "detail"):
            fleet_metrics._resolve_fleet_render_artifact(
                result["bundle_reference"],
                section,
            )
        stores = snapshots._get_persistent_stores()
        result.update({
            "duration_seconds": round(time.perf_counter() - started, 6),
            "sql_audit": {
                "read_only_statement_count": sql_audit.statement_count,
                "rejected_statement_count": sql_audit.rejected_statement_count,
            },
            "cache": {
                "volume_bytes": int(stores.cache.volume()),
                "size_limit_bytes": int(stores.cache.size_limit),
            },
        })
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        logging.exception("Fleet metrics precompute failed")
        print(json.dumps({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }, sort_keys=True))
        return 1
    finally:
        snapshots.close_persistent_snapshot_cache()
        fleet_metrics.engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
