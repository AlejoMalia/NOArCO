"""
noarco.radiative.literature — published warming anchors for the CO2-only Mars atmosphere
=========================================================================================

NOArCO's default ("simple") radiative model is a log-law CO2 forcing mapped through a grey atmosphere.
It is quick, but it is biased low against GCMs at high CO2 pressure (documented in
``noarco.validation``). This module provides the *other* mode: ``radiative="literature_calibrated"``.

In that mode NOArCO does **not** compute the warming; it interpolates, in ``ln P``, between **published
anchor points** and reports the range with its provenance:

========  ===========================  =====================================================
P_total   warming (low, high) K        source / class
========  ===========================  =====================================================
610 Pa    (0, 0)                       present state (exact)
2 kPa     (0, 10)                      Jakosky & Edwards 2018: climate models give < 10 K at
                                       ~20 mbar (``literature_bound``: an upper bound only)
100 kPa   (50, 70)                     Jakosky & Edwards 2018: ~1 bar CO2 brings the surface
                                       close to melting, i.e. ~60 K of the ~60 K needed; the
                                       +/-10 K range is an **assumption** of this module
                                       (``literature_range_assumed``)
========  ===========================  =====================================================

Only Mars has a curve. The mode is for stating a published number, not for discovering a new one; it
does not model the physics (water-vapour and dust feedbacks are inside the published figures).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Anchor:
    p_total_pa: float
    dt_low_k: float
    dt_high_k: float
    source: str
    klass: str


MARS_CO2_WARMING_ANCHORS: tuple[Anchor, ...] = (
    Anchor(610.0, 0.0, 0.0, "present state", "exact"),
    Anchor(2_000.0, 0.0, 10.0, "Jakosky & Edwards 2018 (climate models: < 10 K at ~20 mbar)", "literature_bound"),
    Anchor(100_000.0, 50.0, 70.0,
           "Jakosky & Edwards 2018 (~1 bar brings the surface close to melting, ~60 K); +/-10 K assumed",
           "literature_range_assumed"),
)
P_MAX_PA = 1.5e5


def mars_co2_warming_k(p_total_pa: float) -> tuple[float, float, float]:
    """``(low, mid, high)`` warming (K) for a Mars CO2 atmosphere of total pressure ``p_total_pa``.

    Log-linear interpolation between the anchors (monotone); 0 at or below the present pressure.
    Raises ``ValueError`` above ``P_MAX_PA`` (no published anchor beyond ~1 bar).
    """
    if p_total_pa <= MARS_CO2_WARMING_ANCHORS[0].p_total_pa:
        return 0.0, 0.0, 0.0
    if p_total_pa > P_MAX_PA:
        raise ValueError(f"No published warming anchor above {P_MAX_PA:.3g} Pa")
    x = [math.log(a.p_total_pa) for a in MARS_CO2_WARMING_ANCHORS]
    lo = float(np.interp(math.log(p_total_pa), x, [a.dt_low_k for a in MARS_CO2_WARMING_ANCHORS]))
    hi = float(np.interp(math.log(p_total_pa), x, [a.dt_high_k for a in MARS_CO2_WARMING_ANCHORS]))
    return lo, (lo + hi) / 2.0, hi


def mars_pressure_for_warming(delta_t_k: float) -> float:
    """Total pressure (Pa) at which the *mid* published warming equals ``delta_t_k`` (bisection; monotone)."""
    if delta_t_k <= 0:
        return MARS_CO2_WARMING_ANCHORS[0].p_total_pa
    if delta_t_k > mars_co2_warming_k(P_MAX_PA)[1]:
        raise ValueError("Requested warming exceeds the published curve")
    lo, hi = MARS_CO2_WARMING_ANCHORS[0].p_total_pa, P_MAX_PA
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if mars_co2_warming_k(mid)[1] < delta_t_k:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)
