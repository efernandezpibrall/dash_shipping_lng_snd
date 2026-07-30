#!/usr/bin/env python3
"""Precompute exporter/importer detail snapshots into the local disk cache."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _importer_target(value: str) -> tuple[str, str]:
    aggregation, separator, selected_value = str(value).partition("=")
    if not separator or not aggregation.strip() or not selected_value.strip():
        raise argparse.ArgumentTypeError(
            "Importer targets must use AGGREGATION=VALUE, for example "
            "country=China"
        )
    return aggregation.strip(), selected_value.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build normal-navigation exporter/importer detail snapshots "
            "without changing the SQL schema or writing SQL data."
        )
    )
    parser.add_argument(
        "--exporter",
        action="append",
        default=[],
        metavar="COUNTRY",
        help=(
            "Exporter country to precompute. Repeat for multiple countries."
        ),
    )
    parser.add_argument(
        "--importer",
        action="append",
        default=[],
        type=_importer_target,
        metavar="AGGREGATION=VALUE",
        help=(
            "Importer selection to precompute, for example country=China. "
            "Repeat for multiple selections."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Persistent snapshot directory shared with the Dash process. "
            "Defaults to DASHBOARD_SNAPSHOT_CACHE_DIR or the application "
            "default."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    exporters = list(
        dict.fromkeys(
            country.strip()
            for country in args.exporter
            if country.strip()
        )
    )
    importers = list(dict.fromkeys(args.importer))
    if not exporters and not importers:
        exporters = ["United States"]
        importers = [("country", "China")]

    if args.cache_dir is not None:
        os.environ["DASHBOARD_SNAPSHOT_CACHE_DIR"] = str(
            args.cache_dir.expanduser().resolve(strict=False)
        )
    os.environ.setdefault(
        "DASHBOARD_SNAPSHOT_LOCAL_PERSISTENCE_ENABLED",
        "1",
    )

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from pages import exporter_detail, importer_detail
    from utils import dashboard_snapshot_cache as snapshots
    from utils.detail_snapshot_precompute import (
        DetailSnapshotPrecomputeError,
        precompute_detail_snapshots,
    )

    try:
        summary = precompute_detail_snapshots(
            exporter_detail,
            importer_detail,
            exporter_countries=exporters,
            importer_targets=importers,
        )
    except DetailSnapshotPrecomputeError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 1
    except Exception as exc:
        logging.exception("Detail snapshot precompute failed")
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            )
        )
        return 1
    finally:
        snapshots.close_persistent_snapshot_cache()
        exporter_detail.engine.dispose()
        importer_detail.engine.dispose()

    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
