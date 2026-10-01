"""
noarco.engines.impact — ImpactEngine
====================================
Simulates and tracks the coupled thermodynamic and geochemical impact of
interventions on both AIR (atmosphere) and SOIL (soil/lithosphere).
"""

from __future__ import annotations

from dataclasses import dataclass

from noarco.core.planet_state import PlanetState
from noarco.mechanisms.base import AtmosphereDelta, Mechanism, SoilDelta


@dataclass
class ImpactReport:
    """Full impact assessment of an intervention on AIR and SOIL."""
    mechanism_name: str
    target_body: str
    dose: float
    atmosphere_impact: AtmosphereDelta
    soil_impact: SoilDelta
    energy_joules: float
    energy_kwh: float
    time_estimate_years: float
    feasibility_score: float
    summary: str


class ImpactEngine:
    """Engine that computes the multidirectional physical impact of any intervention."""

    @classmethod
    def evaluate_mechanism(
        cls,
        mechanism: Mechanism,
        planet: PlanetState,
        dose: float = 5.0,
        available_power_w: float = 1.0e9,
    ) -> ImpactReport:
        """Evaluate the combined impact of a mechanism on AIR and SOIL."""
        atm_delta = mechanism.impact_on_atmosphere(planet, dose)
        soil_delta = mechanism.impact_on_soil(planet, dose)
        energy_j = mechanism.energy_cost(dose)
        time_yrs = mechanism.time_estimate(dose, power_w=available_power_w)
        feasibility = mechanism.feasibility(planet)

        summary_text = (
            f"Mechanism {mechanism.name} on {planet.body_name}: "
            f"ΔT={atm_delta.delta_temperature_k:+.1f} K, "
            f"Forcing={atm_delta.delta_forcing_wm2:+.1f} W/m2, "
            f"Energy={energy_j:.2e} J ({energy_j/3.6e6:.2e} kWh), "
            f"Estimated time={time_yrs:.2f} years at {available_power_w/1e6:.1f} MW."
        )

        return ImpactReport(
            mechanism_name=mechanism.name,
            target_body=planet.body_name,
            dose=dose,
            atmosphere_impact=atm_delta,
            soil_impact=soil_delta,
            energy_joules=energy_j,
            energy_kwh=energy_j / 3.6e6,
            time_estimate_years=time_yrs,
            feasibility_score=feasibility,
            summary=summary_text,
        )
