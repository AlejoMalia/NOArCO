"""
noarco.assurance.manifest — Reproducibility manifest
====================================================

Adapted from SFSA ``RME``. Adds what a review board actually needs: a hash of
the *inputs* (canonical JSON), the exact versions of numerical dependencies,
and two digests — ``content_hash`` (inputs + code versions + seed; identical on
any machine that reproduces the run) and ``seal`` (everything, incl. host and
timestamp; detects tampering).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from typing import Any

_TRACKED = ("noarco", "numpy", "scipy", "pydantic", "sfsa")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode()).hexdigest()


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in _TRACKED:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return out


@dataclass
class Manifest:
    name: str
    created_unix: float
    inputs: dict[str, Any]
    seed: int | None
    versions: dict[str, str]
    host: dict[str, str]
    python: str
    content_hash: str = ""
    seal: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def _content_payload(self) -> dict[str, Any]:
        return {"inputs": self.inputs, "seed": self.seed, "versions": self.versions,
                "python": self.python.rsplit(".", 1)[0], "extra": self.extra}

    def _seal_payload(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("seal")
        return d

    def verify(self) -> bool:
        """True iff neither inputs, versions, nor any other field were altered."""
        return (self.content_hash == _sha(self._content_payload())
                and self.seal == _sha(self._seal_payload()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_manifest(name: str, inputs: dict[str, Any], seed: int | None = None,
                   extra: dict[str, Any] | None = None) -> Manifest:
    m = Manifest(
        name=name, created_unix=time.time(), inputs=json.loads(_canonical(inputs)), seed=seed,
        versions=_versions(),
        host={"system": platform.system(), "release": platform.release(),
              "machine": platform.machine()},
        python=sys.version.split()[0], extra=extra or {},
    )
    m.content_hash = _sha(m._content_payload())
    m.seal = _sha(m._seal_payload())
    return m
