"""Shared finding/severity types for the assurance layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Severity(IntEnum):
    INFO = 0
    CAUTION = 1
    FAIL = 2


@dataclass(frozen=True)
class Finding:
    code: str  # stable identifier, e.g. "VAL-HYDROSTATIC-THIN"
    severity: Severity
    message: str
    value: float | None = None
    limit: str | None = None

    def __str__(self) -> str:
        return f"[{self.severity.name}] {self.code}: {self.message}"


def worst(findings: list[Finding]) -> Severity:
    return max((f.severity for f in findings), default=Severity.INFO)
