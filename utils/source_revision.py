from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


SourceRevisionStatus = Literal["fresh", "stale", "unavailable"]


@dataclass(frozen=True)
class SourceRevision:
    status: SourceRevisionStatus
    revision_key: Any
    watermark: Any
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "SourceRevision":
        if not isinstance(value, Mapping):
            return cls(
                status="unavailable",
                revision_key=None,
                watermark=None,
                message="Source revision is unavailable.",
            )
        status = value.get("status")
        if status not in {"fresh", "stale", "unavailable"}:
            status = "unavailable"
        return cls(
            status=status,
            revision_key=value.get("revision_key"),
            watermark=value.get("watermark"),
            message=value.get("message"),
        )


def source_revision_from_context(
    source_context: Mapping[str, Any] | None,
) -> SourceRevision:
    if not isinstance(source_context, Mapping):
        return SourceRevision.from_mapping(None)
    if "source_revision" not in source_context:
        source_watermark = source_context.get("source_watermark")
        if source_watermark is not None:
            return SourceRevision(
                status="fresh",
                revision_key=source_watermark,
                watermark=source_watermark,
            )
    return SourceRevision.from_mapping(source_context.get("source_revision"))
