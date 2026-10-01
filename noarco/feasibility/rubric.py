"""
noarco.feasibility.rubric — endpoint requirements and the dimensionless feasibility rubric
==========================================================================================

Implements the requirement bookkeeping and the conjunctive feasibility rubric of
Turyshev (2026), arXiv:2603.00402, Sec. II.C and IV-VII:

* **Mass inventory** (Eq. 2, 17-19):      ``M_req = K * dP`` with ``K = 4 pi R^2 / g``.
* **Oxygen** (Eq. 5, 67, 72):             ``M_O2 = K p_O2`` and ``E_min = 14.8 MJ/kg * M_O2``.
* **Radiative control** (Eq. 31, 33):     direct forcing ``dF = sigma (T_e^4 - T_e0^4)`` or grey
  optical depth ``tau_IR = (4/3)(T_s/T_e)^4 - 2/3``.
* **Throughput and power** (Eq. 97-98):   ``Mdot = M / t_build``, ``P = E / t_build``.
* **Rubric** (Eq. 9-11):                   ``Pi_x = available / required`` and the
  conjunctive criterion ``Pi_min = min(Pi_x)``; an endpoint is feasible only if
  ``Pi_min >~ 1``.

The rubric is deliberately simple: it yields *lower bounds and discriminators*, not
predictions. Availabilities are inputs; nothing is assumed when they are omitted (the
corresponding ``Pi`` is ``None`` and does not enter ``Pi_min``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from noarco.core.constants import YEAR_S, reversible_o2_energy_j_per_kg
from noarco.core.desired_state import DesiredState
from noarco.core.planet_state import PlanetState
from noarco.inventory.atmosphere import AtmosphericInventory
from noarco.radiative.balance import RadiativeBalance


@dataclass
class EndpointRequirements:
    """Physical requirements implied by a target state (lower bounds)."""

    body: str
    scope: str                      # "global" | "enclosed"
    build_time_years: float
    pressure_target_pa: float | None
    delta_pressure_pa: float
    atmosphere_mass_kg: float       # global mass to add (0 for enclosed/no pressure target)
    o2_mass_kg: float
    o2_min_energy_j: float          # reversible electrolysis work
    buffer_gas_mass_kg: float       # atmosphere_mass - O2 (inert fill implied by the target)
    delta_forcing_toa_wm2: float    # direct-forcing route (mirrors/albedo)
    delta_tau_ir: float             # greenhouse route (added grey IR optical depth)
    mean_mass_flow_kg_s: float      # atmosphere_mass / t_build
    mean_power_w: float             # E_min / t_build
    notes: list[str] = field(default_factory=list)


@dataclass
class PiAssessment:
    """Dimensionless feasibility numbers (``None`` when no availability was supplied)."""

    requirements: EndpointRequirements
    pi_mass: float | None
    pi_forcing: float | None
    pi_throughput: float | None
    pi_power: float | None
    pi_stability: float | None

    @property
    def values(self) -> dict[str, float]:
        d = {"mass": self.pi_mass, "forcing": self.pi_forcing, "throughput": self.pi_throughput,
             "power": self.pi_power, "stability": self.pi_stability}
        return {k: v for k, v in d.items() if v is not None}

    @property
    def pi_min(self) -> float | None:
        """Conjunctive criterion (Eq. 11); ``None`` if no availability was supplied."""
        vals = self.values
        return min(vals.values()) if vals else None

    @property
    def binding_constraint(self) -> str | None:
        vals = self.values
        return min(vals, key=lambda k: vals[k]) if vals else None

    @property
    def feasible(self) -> bool | None:
        m = self.pi_min
        return None if m is None else m >= 1.0


def endpoint_requirements(
    planet: PlanetState,
    target: DesiredState,
    build_time_years: float = 1000.0,
) -> EndpointRequirements:
    """Requirements of reaching ``target`` from ``planet`` within ``build_time_years``.

    Enclosed (regional) targets have no global mass requirement: the gas mass scales with
    the covered area (``A P / g``, Eq. 80), which the caller supplies separately.
    """
    if build_time_years <= 0 or not math.isfinite(build_time_years):
        raise ValueError("build_time_years must be finite and > 0.")
    inv = AtmosphericInventory(planet)
    rb = RadiativeBalance(planet)
    notes: list[str] = []

    enclosed = target.enclosed
    p_target = target.min_surface_pressure_pa
    area = target.region_area_m2
    regional = enclosed and area is not None and p_target is not None
    if enclosed or p_target is None:
        d_p = 0.0
        if regional:
            notes.append("Enclosed/regional target with a declared area: gas mass = A P / g (Turyshev 2026, Eq. 80).")
        elif enclosed:
            notes.append("Enclosed/regional target: gas mass scales with covered area (A P / g).")
    else:
        d_p = max(p_target - planet.surface_pressure_pa, 0.0)
    mass = inv.mass_per_pascal() * d_p
    if regional:
        assert area is not None and p_target is not None
        mass = inv.regional_gas_mass_kg(area, p_target)
        d_p = p_target

    o2_mass = 0.0
    if target.min_o2_partial_pressure_pa and regional:
        assert area is not None
        o2_mass = inv.regional_gas_mass_kg(area, target.min_o2_partial_pressure_pa)
    elif target.min_o2_partial_pressure_pa and not enclosed:
        p_o2_now = planet.gas_composition.mole_fraction("O2") * planet.surface_pressure_pa
        o2_mass = inv.oxygen_mass_for_partial_pressure(max(target.min_o2_partial_pressure_pa - p_o2_now, 0.0))
    e_min = o2_mass * reversible_o2_energy_j_per_kg()
    buffer_mass = max(mass - o2_mass, 0.0)

    d_f = 0.0
    d_tau = 0.0
    if target.target_mean_temperature_k is not None and not enclosed:
        ts_now = planet.mean_temperature_k
        ts_tgt = target.target_mean_temperature_k
        if ts_tgt > ts_now:
            d_f = rb.forcing_needed_for_surface_temperature(ts_tgt)
            d_tau = rb.required_ir_optical_depth(ts_tgt) - rb.required_ir_optical_depth(ts_now)

    return EndpointRequirements(
        body=planet.body_name,
        scope="enclosed" if enclosed else "global",
        build_time_years=build_time_years,
        pressure_target_pa=p_target,
        delta_pressure_pa=d_p,
        atmosphere_mass_kg=mass,
        o2_mass_kg=o2_mass,
        o2_min_energy_j=e_min,
        buffer_gas_mass_kg=buffer_mass,
        delta_forcing_toa_wm2=max(d_f, 0.0),
        delta_tau_ir=max(d_tau, 0.0),
        mean_mass_flow_kg_s=mass / (build_time_years * YEAR_S),
        mean_power_w=e_min / (build_time_years * YEAR_S),
        notes=notes,
    )


def _ratio(available: float | None, required: float) -> float | None:
    """``available / required``; ``inf`` when nothing is required; ``None`` if not supplied."""
    if available is None:
        return None
    if required <= 0:
        return math.inf
    return available / required


def assess(
    requirements: EndpointRequirements,
    *,
    available_inventory_kg: float | None = None,
    available_mass_flow_kg_s: float | None = None,
    available_power_w: float | None = None,
    available_forcing_wm2: float | None = None,
    available_delta_tau_ir: float | None = None,
    replenishment_kg_s: float | None = None,
    loss_kg_s: float | None = None,
) -> PiAssessment:
    """Evaluate the rubric numbers for the availabilities that were supplied.

    ``Pi_F`` uses the direct-forcing route when ``available_forcing_wm2`` is given and the
    greenhouse route when ``available_delta_tau_ir`` is given (the larger of the two if
    both are). ``Pi_S = replenishment / loss`` (Eq. 10).
    """
    pi_f: float | None = None
    candidates = [
        x for x in (
            _ratio(available_forcing_wm2, requirements.delta_forcing_toa_wm2),
            _ratio(available_delta_tau_ir, requirements.delta_tau_ir),
        ) if x is not None
    ]
    if candidates:
        pi_f = max(candidates)
    pi_s = None
    if replenishment_kg_s is not None and loss_kg_s is not None:
        pi_s = _ratio(replenishment_kg_s, loss_kg_s)
    return PiAssessment(
        requirements=requirements,
        pi_mass=_ratio(available_inventory_kg, requirements.atmosphere_mass_kg),
        pi_forcing=pi_f,
        pi_throughput=_ratio(available_mass_flow_kg_s, requirements.mean_mass_flow_kg_s),
        pi_power=_ratio(available_power_w, requirements.mean_power_w),
        pi_stability=pi_s,
    )
