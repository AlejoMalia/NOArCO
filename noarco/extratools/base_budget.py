"""
noarco.extratools.base_budget — one conjunctive budget for a habitat + its ISRU plant
=====================================================================================

Closes the loop between :class:`HabitatDimensioner` (what the crew and shell need) and
:class:`ISRUEvaluator` (what the site can make): the ISRU plant is sized to the habitat's net
make-up demand, power is summed, and the same conjunctive rule as the planetary verdict is applied

    Pi_min = min(available / required)   over the axes whose availability is supplied,

on three axes: ``power`` (kW), ``oxygen`` (ISRU O2 capacity, kg/day) and ``cargo`` (kg that must be
brought: hull, buffer gas, initial O2, water make-up). An axis is evaluated only if its availability is
given (``None`` = unknown, not assumed). Same inputs -> same numbers; every sub-report keeps its own
warnings and ``model_class``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from noarco.core.planet_state import PlanetState
from noarco.extratools.habitat import HabitatDimensioner, HabitatReport, HabitatSpecification
from noarco.extratools.isru import (
    ISRUEvaluator,
    ISRUProductionReport,
    ISRURequirement,
    ISRUSiteProfile,
)

AXES = ("power", "oxygen", "cargo")


@dataclass
class BaseBudget:
    body: str
    habitat: HabitatReport
    isru: ISRUProductionReport
    required: dict[str, float]                   # axis -> requirement (kW, kg/day, kg)
    pi: dict[str, float] = field(default_factory=dict)
    pi_min: float | None = None
    limiting_axis: str | None = None
    viable: bool | None = None
    solar_array_area_m2: float | None = None
    warnings: list[str] = field(default_factory=list)
    model_class: str = "engineering_estimate"

    def summary(self) -> str:
        head = (f"{self.body} base: power {self.required['power']:.1f} kW, ISRU O2 {self.required['oxygen']:.2f} kg/day, "
                f"cargo {self.required['cargo']:.0f} kg")
        if self.pi_min is None:
            return head + " -> undetermined (no availability supplied)."
        return head + f" -> {'viable' if self.viable else 'NOT viable'} (Pi_min {self.pi_min:.3g}, limited by {self.limiting_axis})."


def size_base(
    planet: PlanetState,
    spec: HabitatSpecification,
    site: ISRUSiteProfile,
    *,
    available_power_kw: float | None = None,
    available_o2_capacity_kg_day: float | None = None,
    available_cargo_kg: float | None = None,
    isru_requirement: ISRURequirement | None = None,
) -> BaseBudget:
    """Size habitat + ISRU for ``spec`` at ``site`` and apply the conjunctive rule to what is supplied."""
    hab = HabitatDimensioner.dimension(planet, spec)
    days = float(spec.mission_days)
    leak_o2_day = hab.mission_o2_makeup_kg / days - hab.net_daily_o2_makeup_kg
    o2_demand = max(hab.net_daily_o2_makeup_kg + leak_o2_day, 1e-6)
    water_demand = hab.daily_water_makeup_kg
    base_req = isru_requirement or ISRURequirement()
    req = ISRURequirement(
        o2_kg_day=o2_demand, water_kg_day=water_demand, metal_kg_day=0.0,
        operating_hours_per_day=base_req.operating_hours_per_day, water_recovery_fraction=base_req.water_recovery_fraction,
        excavation_kwh_per_t=base_req.excavation_kwh_per_t, solar_array_efficiency=base_req.solar_array_efficiency,
        solar_derate=base_req.solar_derate)
    isru = ISRUEvaluator().evaluate_site(site, req)

    power_kw = hab.continuous_power_kw + isru.continuous_power_kw
    cargo_kg = hab.hull_mass_kg + hab.n2_mass_kg + hab.o2_mass_kg + hab.mission_water_makeup_kg
    required = {"power": power_kw, "oxygen": o2_demand, "cargo": cargo_kg}
    supplied = {"power": available_power_kw, "oxygen": available_o2_capacity_kg_day, "cargo": available_cargo_kg}
    pi: dict[str, float] = {}
    for a in AXES:
        have = supplied[a]
        if have is not None:
            pi[a] = have / required[a] if required[a] > 0 else float("inf")
    pi_min = min(pi.values()) if pi else None
    limiting = min(pi, key=lambda a: pi[a]) if pi else None

    warnings = [f"habitat: {w}" for w in hab.warnings] + [f"isru: {w}" for w in isru.warnings]
    if hab.net_daily_o2_makeup_kg == 0 and spec.crew_size > 0:
        warnings.append("Loop closure of 100 % makes the crew O2 make-up zero; leak make-up still sets the ISRU demand.")
    solar = isru.solar_array_area_m2
    return BaseBudget(
        body=planet.body_name, habitat=hab, isru=isru, required=required, pi=pi, pi_min=pi_min, limiting_axis=limiting,
        viable=None if pi_min is None else pi_min >= 1.0, solar_array_area_m2=solar, warnings=warnings)
