from __future__ import annotations

import logging
import time
from functools import wraps

from dash._utils import to_json


def _payload_size_bytes(value) -> int | None:
    try:
        return len(to_json(value).encode("utf-8"))
    except Exception:
        return None


def log_callback_timing(metric_name: str):
    """Log callback wall time and serialized output size at INFO level."""

    def decorator(callback):
        callback_logger = logging.getLogger(callback.__module__)

        @wraps(callback)
        def timed_callback(*args, **kwargs):
            started = time.perf_counter()
            result = callback(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - started) * 1000
            payload_bytes = (
                _payload_size_bytes(result)
                if callback_logger.isEnabledFor(logging.INFO)
                else None
            )
            callback_logger.info(
                "dash_callback metric=%s elapsed_ms=%.1f payload_bytes=%s",
                metric_name,
                elapsed_ms,
                payload_bytes,
            )
            return result

        return timed_callback

    return decorator
