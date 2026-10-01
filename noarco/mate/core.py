"""
noarco.mate.core
================
Core implementation of the MATE + TRIADA methodology for NOArCO.

TRIADA protocol (must run before any expensive computation):
    T1 · INVENTORY  — Cache hit? Already computed?
    T2 · MATHEMATICS  — Closed-form formula available? Write it BEFORE running.
    T3 · MATE        — Can a previous result be reused/adapted?

MATE projection rules (during computation):
    R1 · Keep tension — don't force-close a hypothesis prematurely.
    R2 · Rules are invariant — don't reframe physics to save a result.
    R3 · Project last 3 — if 2-3 forced steps already fix the verdict,
         STOP and certify the projection. Log what was NOT expanded.
    R4 · Advance by critical squares — ontology → kinematics → metrology.
    R5 · Mate ≠ theater — completing a computation whose verdict is
         already forced adds cost, not truth.

Implementation:
    - `NOArCoCache`: thread-safe LRU cache keyed by (function_name, arg_hash)
    - `@triada`: decorator that runs T1-T2-T3 before executing the function
    - `mate_project`: function to certify a projected result and log the projection
    - `MATEResult`: wrapper around any computation result with MATE metadata

Author credit: MATE + TRIADA methodology by Alejo Malia.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger("noarco.mate")

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# MATEStatus — epistemic status of a computation result
# ---------------------------------------------------------------------------

class MATEStatus(str, Enum):
    """How was this result obtained?"""
    FULL_COMPUTE   = "full_compute"    # Fully computed — no shortcut taken
    CACHE_HIT      = "cache_hit"       # T1: returned from cache
    CLOSED_FORM    = "closed_form"     # T2: answered by analytic formula
    REUSED         = "reused"          # T3: adapted from a previous result
    PROJECTED      = "projected"       # R3: MATE projection — result is
                                       #     forced by last 2-3 steps;
                                       #     remaining computation skipped


# ---------------------------------------------------------------------------
# MATEResult — wrapper for any NOArCO computation
# ---------------------------------------------------------------------------

@dataclass
class MATEResult:
    """A computation result annotated with MATE metadata.

    Every public computation in NOArCO that passes through `@triada`
    or `mate_project()` returns a `MATEResult`. This makes the
    compute-efficiency strategy fully auditable.

    Parameters
    ----------
    value : Any
        The actual computation result.
    status : MATEStatus
        How this result was obtained.
    compute_time_s : float
        Wall-clock time spent (0.0 for cache hits and closed-form).
    cache_key : str, optional
        The cache key used to store/retrieve this result.
    closed_form_equation : str, optional
        If status is CLOSED_FORM, the analytic formula used (LaTeX or Python).
    projection_basis : str, optional
        If status is PROJECTED, documents the 2-3 forced steps that
        fixed the verdict. MANDATORY for projected results (R3).
    unexpanded_branches : list[str], optional
        If status is PROJECTED, lists what was NOT computed and WHY
        it cannot change the outcome. Auditable per R5.
    source_call : str, optional
        The function name + module that produced this result.
    warnings : list[str]
        Any MATE warnings (e.g. high uncertainty, near saturation).
    """
    value: Any
    status: MATEStatus = MATEStatus.FULL_COMPUTE
    compute_time_s: float = 0.0
    cache_key: str | None = None
    closed_form_equation: str | None = None
    projection_basis: str | None = None
    unexpanded_branches: list[str] = field(default_factory=list)
    source_call: str | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable MATE audit line."""
        lines = [
            f"[MATE] {self.source_call or 'unknown'} → {self.status.value}",
            f"  Compute time : {self.compute_time_s*1000:.2f} ms",
        ]
        if self.status == MATEStatus.CLOSED_FORM and self.closed_form_equation:
            lines.append(f"  Formula      : {self.closed_form_equation}")
        if self.status == MATEStatus.PROJECTED:
            lines.append(f"  Projection   : {self.projection_basis}")
            for b in self.unexpanded_branches:
                lines.append(f"  Not expanded : {b}")
        for w in self.warnings:
            lines.append(f"  ⚠ WARNING    : {w}")
        return "\n".join(lines)

    def unwrap(self) -> Any:
        """Return the raw value, discarding MATE metadata."""
        return self.value


# ---------------------------------------------------------------------------
# NOArCoCache — central LRU cache for the session
# ---------------------------------------------------------------------------

class NOArCoCache:
    """Thread-safe in-memory cache for NOArCO computations.

    Used by the `@triada` decorator (T1 — INVENTORY check).
    Keyed by a hash of (function name, arguments). Invalidated
    when `PlanetState` changes in a `NOArCoSession`.

    Parameters
    ----------
    maxsize : int
        Maximum number of entries. Oldest entries are evicted (LRU).
    """

    def __init__(self, maxsize: int = 512) -> None:
        self._store: dict[str, MATEResult] = {}
        self._access_order: list[str] = []
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self._maxsize = maxsize
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        """Create a deterministic cache key from function name + arguments."""
        try:
            payload = json.dumps(
                {"fn": func_name, "args": args, "kw": kwargs},
                sort_keys=True,
                default=str,  # Handles Pydantic models via __str__
            )
        except (TypeError, ValueError):
            # Fallback: use repr (less stable but always works)
            payload = repr((func_name, args, kwargs))
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> MATEResult | None:
        """Retrieve a cached result, or None if absent."""
        with self._lock:
            if key in self._store:
                self.hits += 1
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._store[key]
            self.misses += 1
            return None

    def set(self, key: str, result: MATEResult) -> None:
        """Store a result in the cache, evicting LRU entry if needed."""
        with self._lock:
            if key in self._store:
                self._access_order.remove(key)
            elif len(self._store) >= self._maxsize:
                lru_key = self._access_order.pop(0)
                del self._store[lru_key]
            self._store[key] = result
            self._access_order.append(key)

    def invalidate(self, prefix: str | None = None) -> int:
        """Invalidate cache entries. If prefix given, only matching keys.

        Returns number of entries removed.
        """
        with self._lock:
            if prefix is None:
                n = len(self._store)
                self._store.clear()
                self._access_order.clear()
                return n
            to_remove = [k for k in self._store if k.startswith(prefix)]
            for k in to_remove:
                del self._store[k]
                self._access_order.remove(k)
            return len(to_remove)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._store),
                "maxsize": self._maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total > 0 else 0.0,
            }


# ---------------------------------------------------------------------------
# Global default cache (used when no session cache is provided)
# ---------------------------------------------------------------------------

_DEFAULT_CACHE: NOArCoCache = NOArCoCache(maxsize=256)


def get_default_cache() -> NOArCoCache:
    """Return the module-level default cache."""
    return _DEFAULT_CACHE


def reset_default_cache() -> None:
    """Invalidate and reset the module-level default cache."""
    _DEFAULT_CACHE.invalidate()


# ---------------------------------------------------------------------------
# @triada — decorator implementing the full TRIADA protocol
# ---------------------------------------------------------------------------

def triada(
    *,
    closed_form: Callable[..., Any] | None = None,
    closed_form_eq: str | None = None,
    cache: NOArCoCache | None = None,
    skip_cache: bool = False,
) -> Callable[[F], F]:
    """Decorator implementing the TRIADA compute-efficiency protocol.

    Wraps a function with the three TRIADA checks, in order:

        T1 · INVENTORY  — Check cache before computing.
        T2 · MATHEMATICS  — If `closed_form` is provided and its result is
                           not None, return it as CLOSED_FORM (no simulation).
        T3 · MATE        — Fall through to full computation (marked FULL_COMPUTE).

    The decorated function returns a `MATEResult` instead of a raw value.
    Use `.unwrap()` on the result to get the plain value.

    Parameters
    ----------
    closed_form : Callable, optional
        A function with the same signature as the decorated function that
        returns an analytic result, or None if it cannot answer the query.
        Implements T2 (MATHEMATICS): write the formula BEFORE running.
    closed_form_eq : str, optional
        LaTeX or Python string documenting the closed-form formula (for audit).
    cache : NOArCoCache, optional
        Cache instance to use. Defaults to the module-level cache.
    skip_cache : bool
        If True, bypass cache (useful for stochastic computations).

    Examples
    --------
    >>> @triada(closed_form_eq="M = P * 4*pi*R^2 / g")
    ... def mass_per_pascal(planet):
    ...     return planet.surface_area_m2 / planet.gravity_ms2
    ...
    >>> result = mass_per_pascal(MARS)
    >>> result.status
    MATEStatus.FULL_COMPUTE  # first call: computed
    >>> result2 = mass_per_pascal(MARS)
    >>> result2.status
    MATEStatus.CACHE_HIT     # second call: from cache
    """
    _cache = cache or _DEFAULT_CACHE

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> MATEResult:
            fn_name = f"{fn.__module__}.{fn.__qualname__}"
            key = _cache._make_key(fn_name, args, kwargs)

            # ── T1: INVENTORY — cache check ───────────────────────────
            if not skip_cache:
                cached = _cache.get(key)
                if cached is not None:
                    logger.debug("[TRIADA-T1] Cache hit: %s", fn_name)
                    # Return a fresh MATEResult referencing the cached value
                    return MATEResult(
                        value=cached.value,
                        status=MATEStatus.CACHE_HIT,
                        compute_time_s=0.0,
                        cache_key=key,
                        source_call=fn_name,
                    )

            # ── T2: MATHEMATICS — closed-form check ─────────────────────
            if closed_form is not None:
                try:
                    cf_value = closed_form(*args, **kwargs)
                except Exception:  # noqa: BLE001 - a failing closed form falls through to T3
                    cf_value = None

                if cf_value is not None:
                    logger.debug("[TRIADA-T2] Closed-form: %s", fn_name)
                    result = MATEResult(
                        value=cf_value,
                        status=MATEStatus.CLOSED_FORM,
                        compute_time_s=0.0,
                        cache_key=key,
                        closed_form_equation=closed_form_eq,
                        source_call=fn_name,
                    )
                    if not skip_cache:
                        _cache.set(key, result)
                    return result

            # ── T3: MATE — full computation ────────────────────────────
            logger.debug("[TRIADA-T3] Full compute: %s", fn_name)
            t0 = time.perf_counter()
            value = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0

            result = MATEResult(
                value=value,
                status=MATEStatus.FULL_COMPUTE,
                compute_time_s=elapsed,
                cache_key=key,
                source_call=fn_name,
            )
            if not skip_cache:
                _cache.set(key, result)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# mate_project — certify a projected result (R3)
# ---------------------------------------------------------------------------

def mate_project(
    value: Any,
    projection_basis: str,
    unexpanded_branches: list[str],
    source_call: str | None = None,
    warnings: list[str] | None = None,
) -> MATEResult:
    """Certify a MATE projection result (Rule R3).

    Use this when the last 2-3 forced steps already fix the verdict
    and continuing the full computation would not change the outcome.
    The projection_basis and unexpanded_branches are MANDATORY for
    auditability (Rule R5: mate ≠ theater).

    Parameters
    ----------
    value : Any
        The projected result value.
    projection_basis : str
        Description of the 2-3 forced steps that fix the verdict.
        Must be specific and falsifiable.
    unexpanded_branches : list[str]
        What was NOT computed, and WHY it cannot change the outcome.
        Each entry should be: "<branch>: <reason it cannot change verdict>"
    source_call : str, optional
        Function or context name for the audit log.
    warnings : list[str], optional
        Any caveats about the projection.

    Returns
    -------
    MATEResult with status=PROJECTED

    Examples
    --------
    >>> # CO2 inventory gap for E2: we know endogenous CO2 is ~5e16 kg
    >>> # and E2 needs ~2e19 kg. The gap is 400x. Even adding the most
    >>> # optimistic regolith estimate (4e18 kg) still leaves a 4x gap.
    >>> result = mate_project(
    ...     value={"feasible": False, "gap_kg": 1.6e19},
    ...     projection_basis=(
    ...         "Step 1: E2 needs ~2e19 kg CO2. "
    ...         "Step 2: Best-case available = 4e18 kg (Jakosky 2019 upper bound). "
    ...         "Step 3: Gap = 1.6e19 kg > 0 regardless of regolith uncertainty range."
    ...     ),
    ...     unexpanded_branches=[
    ...         "Subsurface carbonates: even upper estimate (2e19 kg) requires "
    ...         "complete mobilization with 100% efficiency — physically impossible.",
    ...         "Cometary import: excluded by isru_only=True constraint.",
    ...     ],
    ...     source_call="co2_inventory_gap_kg",
    ... )
    """
    logger.info(
        "[MATE-R3] Projection certified by %s. Basis: %s",
        source_call or "unknown",
        projection_basis[:80],
    )
    for branch in unexpanded_branches:
        logger.info("[MATE-R3] Not expanded: %s", branch[:80])

    return MATEResult(
        value=value,
        status=MATEStatus.PROJECTED,
        compute_time_s=0.0,
        projection_basis=projection_basis,
        unexpanded_branches=unexpanded_branches,
        source_call=source_call,
        warnings=warnings or [],
    )
