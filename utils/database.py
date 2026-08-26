from __future__ import annotations

import configparser
import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, event, exc
from sqlalchemy.engine import Engine


DEFAULT_POOL_SIZE = 2
DEFAULT_MAX_OVERFLOW = 0
DEFAULT_POOL_TIMEOUT_SECONDS = 30
DEFAULT_POOL_RECYCLE_SECONDS = 1800
DEFAULT_APPLICATION_NAME = "dash_shipping_lng_snd"


def _read_positive_int(name: str, default: int, *, allow_zero: bool = False) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "zero or greater" if allow_zero else "greater than zero"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config.ini"


CONFIG_FILE_PATH = Path(
    os.getenv("DASHBOARD_CONFIG_FILE", str(_default_config_path()))
).expanduser()

_config_reader = configparser.ConfigParser(interpolation=None)
_config_reader.read(CONFIG_FILE_PATH)

DB_CONNECTION_STRING = _config_reader.get(
    "DATABASE",
    "CONNECTION_STRING",
    fallback=None,
)
DB_SCHEMA = _config_reader.get("DATABASE", "SCHEMA", fallback="at_lng")


@lru_cache(maxsize=1)
def get_database_engine() -> Engine:
    if not DB_CONNECTION_STRING:
        raise ValueError(
            f"Missing DATABASE CONNECTION_STRING in {CONFIG_FILE_PATH}"
        )

    engine = create_engine(
        DB_CONNECTION_STRING,
        pool_size=_read_positive_int(
            "DASH_DB_POOL_SIZE",
            DEFAULT_POOL_SIZE,
        ),
        max_overflow=_read_positive_int(
            "DASH_DB_MAX_OVERFLOW",
            DEFAULT_MAX_OVERFLOW,
            allow_zero=True,
        ),
        pool_timeout=_read_positive_int(
            "DASH_DB_POOL_TIMEOUT_SECONDS",
            DEFAULT_POOL_TIMEOUT_SECONDS,
        ),
        pool_recycle=_read_positive_int(
            "DASH_DB_POOL_RECYCLE_SECONDS",
            DEFAULT_POOL_RECYCLE_SECONDS,
        ),
        pool_pre_ping=True,
        pool_use_lifo=True,
        connect_args={
            "connect_timeout": _read_positive_int(
                "DASH_DB_CONNECT_TIMEOUT_SECONDS",
                10,
            ),
            "application_name": os.getenv(
                "DASH_DB_APPLICATION_NAME",
                DEFAULT_APPLICATION_NAME,
            ),
        },
    )

    @event.listens_for(engine, "connect")
    def remember_process(_connection, record):
        record.info["pid"] = os.getpid()

    @event.listens_for(engine, "checkout")
    def reject_forked_connection(_connection, record, proxy):
        if record.info.get("pid") != os.getpid():
            record.dbapi_connection = proxy.dbapi_connection = None
            raise exc.DisconnectionError(
                "Database connection belongs to another process"
            )

    return engine


engine = get_database_engine()
