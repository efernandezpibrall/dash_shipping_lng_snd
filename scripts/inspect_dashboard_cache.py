#!/usr/bin/env python3
"""Inspect or safely prune the same-host dashboard snapshot cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report cache usage by namespace and optionally prune only "
            "safe non-latest records. Pruning is a dry run unless --apply "
            "is supplied."
        )
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--staged-older-than-days", type=int, default=7)
    parser.add_argument("--retired-older-than-days", type=int, default=30)
    parser.add_argument(
        "--retired-namespace",
        action="append",
        default=[],
        help=(
            "Retired namespace eligible for non-latest pruning after the "
            "access grace period. Repeat for multiple namespaces."
        ),
    )
    parser.add_argument(
        "--include-records",
        action="store_true",
        help="Include every record in inspection output.",
    )
    parser.add_argument(
        "--coordinated-restart-confirmed",
        action="store_true",
        help=(
            "Required with --apply when pruning any retired namespace."
        ),
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.apply and not args.prune:
        raise SystemExit("--apply requires --prune")
    if args.cache_dir is not None:
        os.environ["DASHBOARD_SNAPSHOT_CACHE_DIR"] = str(
            args.cache_dir.expanduser().resolve(strict=False)
        )
    os.environ.setdefault(
        "DASHBOARD_SNAPSHOT_LOCAL_PERSISTENCE_ENABLED",
        "1",
    )

    from utils import dashboard_snapshot_cache as snapshots

    try:
        if args.prune:
            result = snapshots.prune_persistent_snapshot_cache(
                apply=args.apply,
                staged_older_than_days=args.staged_older_than_days,
                retired_namespaces=args.retired_namespace,
                retired_older_than_days=args.retired_older_than_days,
                coordinated_restart_confirmed=(
                    args.coordinated_restart_confirmed
                ),
            )
        else:
            result = snapshots.inspect_persistent_snapshot_cache()
            if not args.include_records:
                result = {
                    key: value
                    for key, value in result.items()
                    if key != "records"
                }
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    finally:
        snapshots.close_persistent_snapshot_cache()


if __name__ == "__main__":
    raise SystemExit(main())
