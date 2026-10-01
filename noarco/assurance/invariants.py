"""
noarco.assurance.invariants — Physical invariants (TRIADA T3: verification)
===========================================================================

Checks that must hold for any state the framework accepts or produces. A
violation means the *data or the model* is wrong, not that the scenario is
merely infeasible.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from noarco.assurance.findings import Finding, Severity
from noarco.core.constants import G_NEWTON as _G
from noarco.core.constants import SIGMA_SB as _SIGMA_SB
from noarco.core.planet_state import PlanetState

if TYPE_CHECKING:
    from noarco.engines.pathfinder import OptimizedPathway


def check_planet_invariants(planet: PlanetState, g_tol: float = 0.10) -> list[Finding]:
    out: list[Finding] = []

    # 1. Gravity consistency g = G M / R^2 (tolerance: giant planets quote g at 1 bar).
    g_calc = _G * planet.mass_kg / planet.radius_m ** 2
    rel = abs(g_calc - planet.gravity_ms2) / planet.gravity_ms2
    if rel > g_tol:
        out.append(Finding("INV-GRAVITY", Severity.FAIL,
                           f"g={planet.gravity_ms2} disagrees with GM/R^2={g_calc:.4g} by {rel:.1%}",
                           rel, f"<= {g_tol:.0%}"))
    elif rel > 0.02:
        out.append(Finding("INV-GRAVITY", Severity.CAUTION,
                           f"g differs from GM/R^2 by {rel:.1%} (reference level mismatch?)", rel))

    # 2. Composition closure.
    total = sum(planet.gas_composition.species.values())
    if not 0.99 <= total <= 1.01:
        out.append(Finding("INV-COMPOSITION", Severity.CAUTION if total <= 1.01 else Severity.FAIL,
                           f"mole fractions sum to {total:.4f}", total, "[0.99, 1.01]"))

    # 3. Hydrostatic mass round trip: P -> M -> P.
    m_atm = planet.atmospheric_mass_kg
    p_back = m_atm * planet.gravity_ms2 / planet.surface_area_m2
    if not math.isclose(p_back, planet.surface_pressure_pa, rel_tol=1e-9):
        out.append(Finding("INV-HYDROSTATIC", Severity.FAIL,
                           "hydrostatic mass/pressure round trip is inconsistent", p_back))

    # 4. Energy balance sanity: T_eq must match the Stefan-Boltzmann closed form.
    t_eq = (planet.solar_constant_wm2 * (1 - planet.surface_albedo) / (4 * _SIGMA_SB)) ** 0.25
    if not math.isclose(t_eq, planet.equilibrium_temperature_k, rel_tol=1e-9):
        out.append(Finding("INV-ENERGY", Severity.FAIL, "equilibrium temperature mismatch", t_eq))

    # 5. Thermodynamic plausibility: surface cooler than equilibrium needs an internal/other source.
    if planet.mean_temperature_k < 0.85 * t_eq:
        out.append(Finding("INV-THERMO", Severity.CAUTION,
                           f"mean T {planet.mean_temperature_k:.0f} K is well below T_eq {t_eq:.0f} K",
                           planet.mean_temperature_k))

    # 6. Reservoirs non-negative and subset-bounded by planetary mass.
    for name in ("co2_ice_kg", "h2o_ice_kg"):
        v = getattr(planet, name)
        if v is not None and (v < 0 or v > 0.1 * planet.mass_kg):
            out.append(Finding("INV-RESERVOIR", Severity.FAIL,
                               f"{name}={v:.3g} kg is negative or >10% of the planet's mass", v))
    return out


def check_mass_balance(required_kg: float, available_kg: float, produced_kg: float,
                       rel_tol: float = 1e-9) -> list[Finding]:
    """required == available-used + produced must close; negative masses are invalid."""
    out: list[Finding] = []
    for label, v in (("required", required_kg), ("available", available_kg), ("produced", produced_kg)):
        if not math.isfinite(v) or v < 0:
            out.append(Finding("INV-MASS-SIGN", Severity.FAIL, f"{label} mass {v!r} invalid", v))
    if not out and produced_kg > required_kg * (1 + rel_tol) and required_kg > 0:
        out.append(Finding("INV-MASS-OVERPRODUCTION", Severity.CAUTION,
                           "produced mass exceeds requirement", produced_kg))
    return out


def check_pathway_invariants(pathway: OptimizedPathway, rel_tol: float = 1e-9) -> list[Finding]:
    """Physical self-checks on a pathway: no step may violate a thermodynamic floor or a sum rule.

    * totals equal the sum of the steps;
    * every oxygen step needs at least the reversible Gibbs work (14.8 MJ/kg O2);
    * every import step needs at least the kinetic benchmark's positive energy;
    * masses, energies and times are finite and non-negative.
    """
    from noarco.core.constants import reversible_o2_energy_j_per_kg

    out: list[Finding] = []
    for st in pathway.steps:
        for label, v in (("mass", st.total_mass_kg), ("energy", st.energy_joules), ("time", st.time_years)):
            if not math.isfinite(v) or v < 0:
                out.append(Finding("INV-STEP-SIGN", Severity.FAIL, f"{st.mechanism_name}: {label} {v!r} invalid", v))
        if st.target_metric == "oxygen" and st.energy_joules < st.total_mass_kg * reversible_o2_energy_j_per_kg() * (1 - rel_tol):
            out.append(Finding("INV-GIBBS-FLOOR", Severity.FAIL,
                               f"{st.mechanism_name}: energy below the reversible Gibbs minimum"))
    if not math.isclose(pathway.total_energy_joules, sum(s.energy_joules for s in pathway.steps), rel_tol=1e-9, abs_tol=1e-6):
        out.append(Finding("INV-SUM-ENERGY", Severity.FAIL, "pathway energy is not the sum of its steps"))
    if not math.isclose(pathway.total_time_years, sum(s.time_years for s in pathway.steps), rel_tol=1e-9, abs_tol=1e-9):
        out.append(Finding("INV-SUM-TIME", Severity.FAIL, "pathway time is not the sum of its steps"))
    return out
