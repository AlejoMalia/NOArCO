"""
noarco.screening — compare many candidates, sweep one axis, ask "what do I need?"
================================================================================

Thin batch layer over :func:`noarco.verdict.verdict`; no new physics. It exists so that trades are
one call instead of a hand-written loop, and so that every row stays a full, citeable verdict.

* :func:`required_availability` - what each axis needs for a target (no availability supplied);
* :func:`rank` - candidates ordered viable-first, then by ``Pi_min`` (ties broken by name: deterministic);
* :func:`sweep` - ``Pi_min`` and limiting axis as one availability varies (+ the exact break-even value);
* :func:`what_if` - the verdict before/after changing some inputs, and whether the limiting axis flipped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from noarco.core.desired_state import DesiredState
from noarco.verdict import Verdict, verdict

_AXIS_KWARG = {"mass": "available_inventory_kg", "throughput": "available_mass_flow_kg_s",
               "power": "available_power_w", "forcing": "available_forcing_wm2"}


@dataclass
class Candidate:
    name: str
    body: Any
    target: DesiredState
    kwargs: dict[str, Any] = field(default_factory=dict)       # build_time_years, available_* ...


@dataclass
class Ranked:
    rank: int
    name: str
    verdict: Verdict


def required_availability(body: Any, target: DesiredState, build_time_years: float = 1000.0) -> dict[str, float]:
    """Availability each axis needs for ``target`` (kg, kg/s, W, W/m2), without assuming any supply."""
    from noarco.bodies import resolve_body
    from noarco.feasibility import endpoint_requirements

    req = endpoint_requirements(resolve_body(body).planet, target, build_time_years)
    return {"mass": req.atmosphere_mass_kg, "throughput": req.mean_mass_flow_kg_s, "power": req.mean_power_w,
            "forcing": req.delta_forcing_toa_wm2}


def rank(candidates: Sequence[Candidate]) -> list[Ranked]:
    """Order candidates: viable first, then higher ``Pi_min``; undetermined last; ties by name."""
    rows = [(c.name, verdict(c.body, c.target, **c.kwargs)) for c in candidates]

    def key(row: tuple[str, Verdict]) -> tuple[int, float, str]:
        v = row[1]
        group = 0 if v.viable else (2 if v.viable is None else 1)
        pi = -(v.pi_min if v.pi_min is not None and math.isfinite(v.pi_min) else 1e300)
        return group, pi, row[0]

    return [Ranked(i + 1, n, v) for i, (n, v) in enumerate(sorted(rows, key=key))]


@dataclass
class SweepPoint:
    value: float
    pi_min: float | None
    limiting_axis: str | None
    viable: bool | None


def sweep(body: Any, target: DesiredState, axis: str, values: Sequence[float], **fixed: Any) -> list[SweepPoint]:
    """``Pi_min`` as the availability of ``axis`` takes each of ``values`` (other availabilities in ``fixed``)."""
    if axis not in _AXIS_KWARG:
        raise ValueError(f"axis must be one of {sorted(_AXIS_KWARG)}")
    out = []
    for x in values:
        v = verdict(body, target, **{**fixed, _AXIS_KWARG[axis]: float(x)})
        out.append(SweepPoint(float(x), v.pi_min, v.limiting_axis, v.viable))
    return out


def breakeven(body: Any, target: DesiredState, axis: str, build_time_years: float = 1000.0) -> float:
    """Availability of ``axis`` at which that axis alone reaches ``Pi = 1`` (exact: Pi is available / required)."""
    need = required_availability(body, target, build_time_years)[axis]
    return need


def what_if(base: Candidate, **changes: Any) -> dict[str, Any]:
    """Verdict before and after changing inputs (``build_time_years``, ``available_*``); reports a flip."""
    before = verdict(base.body, base.target, **base.kwargs)
    after = verdict(base.body, base.target, **{**base.kwargs, **changes})
    return {
        "before": before, "after": after,
        "viability_changed": before.viable != after.viable,
        "limiting_axis_changed": before.limiting_axis != after.limiting_axis,
        "pi_min_ratio": (after.pi_min / before.pi_min) if (before.pi_min and after.pi_min is not None
                                                          and math.isfinite(before.pi_min)
                                                          and math.isfinite(after.pi_min)) else None,
    }
