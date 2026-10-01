"""
noarco.mechanisms.base
======================
Abstract base class for all terraforming intervention mechanisms.

Every mechanism (aerosols, PFCs, mirrors, aerogel, etc.) inherits from
`Mechanism` and must implement `compute()`, which takes the current
`PlanetState` and a target forcing/delta, and returns a `MechanismResult`
with mass, power, throughput, time, and risk.

References
----------
Turyshev, S. G. (2026). arXiv:2603.00402 (Section 5, Table 5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from noarco.core.planet_state import PlanetState


@dataclass
class MechanismResult:
    """Output of a mechanism computation.

    Parameters
    ----------
    mechanism_name : str
        Name of the mechanism.
    total_mass_kg : float
        Total mass of material required (kg).
    power_w : float
        Continuous power required (W). 0.0 if purely passive.
    throughput_kg_s : float
        Required industrial throughput rate (kg/s).
    delta_forcing_wm2 : float
        Radiative forcing produced (W/m2). Positive = warming.
    delta_temperature_k : float
        Estimated surface temperature change (K).
    deployment_time_years : float
        Time to deploy the mechanism (years).
    effect_duration_years : float
        Duration of the effect before maintenance needed (years).
        float('inf') for permanent effects.
    risk_index : float
        Qualitative risk score [0-1]. 0 = negligible, 1 = extreme.
    isru_fraction : float
        Fraction of mass that can be sourced in-situ [0-1].
    notes : str
        Additional notes or warnings.
    source_references : list[str]
        Citations for the estimates in this result.
    """
    mechanism_name: str
    total_mass_kg: float
    power_w: float
    throughput_kg_s: float
    delta_forcing_wm2: float
    delta_temperature_k: float
    deployment_time_years: float
    effect_duration_years: float
    risk_index: float
    isru_fraction: float = 0.0
    target_volume_m3: float | None = None
    notes: str = ""
    source_references: list[str] = field(default_factory=list)
    #: Fraction of the planetary surface over which the effect acts (1.0 = global).
    #: ``delta_forcing_wm2`` is always the *global-mean* forcing, i.e. already
    #: multiplied by this fraction for regional mechanisms; ``delta_temperature_k``
    #: is the local change where the mechanism acts.
    area_fraction: float = 1.0
    #: Model-validity caveats that the caller must surface next to the numbers.
    warnings: list[str] = field(default_factory=list)
    #: Credibility class: "first_principles" (closed-form physics), "reference_scaling"
    #: (published order-of-magnitude scaling) or "engineering_estimate" (parametrised,
    #: needs validation against data).
    model_class: str = "engineering_estimate"

    def summary(self) -> str:
        """Return a formatted summary of the mechanism result."""
        eff_dur = (
            f"{self.effect_duration_years:.1f} yr"
            if self.effect_duration_years != float("inf")
            else "permanent"
        )
        lines = [
            f"=== MechanismResult: {self.mechanism_name} ===",
            f"  Total mass required  : {self.total_mass_kg:.3e} kg",
            f"  Power required       : {self.power_w:.3e} W  ({self.power_w/1e12:.2f} TW)",
            f"  Throughput needed    : {self.throughput_kg_s:.3e} kg/s",
            f"  Radiative forcing    : {self.delta_forcing_wm2:+.2f} W/m2",
            f"  Temperature change   : {self.delta_temperature_k:+.1f} K",
            f"  Deployment time      : {self.deployment_time_years:.1f} yr",
            f"  Effect duration      : {eff_dur}",
            f"  Risk index           : {self.risk_index:.2f} / 1.0",
            f"  ISRU fraction        : {self.isru_fraction:.0%}",
        ]
        lines.append(f"  Model class          : {self.model_class}")
        if self.area_fraction < 1.0:
            lines.append(f"  Area fraction        : {self.area_fraction:.3e} (regional)")
        for w in self.warnings:
            lines.append(f"  WARNING              : {w}")
        if self.notes:
            lines.append(f"  Notes                : {self.notes}")
        for ref in self.source_references:
            lines.append(f"  Ref: {ref}")
        return "\n".join(lines)


@dataclass
class AtmosphereDelta:
    """Atmospheric modifications induced by a mechanism intervention."""
    delta_temperature_k: float = 0.0
    delta_pressure_pa: float = 0.0
    delta_composition: dict[str, float] = field(default_factory=dict)
    delta_forcing_wm2: float = 0.0
    notes: str = ""


@dataclass
class SoilDelta:
    """Soil and regolith modifications induced by a mechanism intervention."""
    regolith_processed_kg: float = 0.0
    water_ice_extracted_kg: float = 0.0
    metals_recovered_kg: dict[str, float] = field(default_factory=dict)
    perchlorates_neutralized_kg: float = 0.0
    notes: str = ""


class Mechanism(ABC):
    """Abstract base class for all terraforming mechanisms.

    Subclasses implement specific intervention methods and expose a
    consistent interface for use by the cascade planner (`noarco.paths`).

    Parameters
    ----------
    name : str
        Human-readable name of the mechanism.
    description : str
        Brief description of the physical process.
    """

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def compute(
        self,
        planet: PlanetState,
        target_delta_t_k: float | None = None,
        target_delta_f_wm2: float | None = None,
        deployment_years: float = 10.0,
    ) -> MechanismResult:
        """Compute the resources needed for this mechanism.

        Subclasses must implement this method. Either `target_delta_t_k`
        or `target_delta_f_wm2` should be provided (not both).
        """
        ...

    @abstractmethod
    def is_applicable(self, planet: PlanetState) -> bool:
        """Check if this mechanism is physically applicable to the given body."""
        ...

    def required_materials(self, current_state: PlanetState, target: dict[str, Any] | None = None) -> dict[str, float]:
        """Compute required raw materials for this mechanism."""
        res = self.compute(current_state, target_delta_t_k=5.0)
        return {"total_mass_kg": res.total_mass_kg}

    def impact_on_atmosphere(self, current_state: PlanetState, dose: float) -> AtmosphereDelta:
        """Compute atmospheric impact vector for a given intervention dose."""
        res = self.compute(current_state, target_delta_t_k=dose)
        return AtmosphereDelta(
            delta_temperature_k=res.delta_temperature_k,
            delta_forcing_wm2=res.delta_forcing_wm2,
            notes=res.notes,
        )

    def impact_on_soil(self, current_state: PlanetState, dose: float) -> SoilDelta:
        """Compute soil/regolith impact vector for a given intervention dose."""
        return SoilDelta(notes=f"Standard intervention for {self.name}")

    def energy_cost(self, dose: float) -> float:
        """Compute total energetic cost in Joules for a given dose."""
        return 1.0e15 * abs(dose)

    def time_estimate(self, dose: float, scale: float = 1.0, power_w: float = 1.0e9) -> float:
        """Estimate deployment time in years as a function of dose and available power."""
        e_joules = self.energy_cost(dose) * scale
        return e_joules / (power_w * 3.1536e7)

    def feasibility(self, current_state: PlanetState) -> float:
        """Calculate feasibility score [0.0 - 1.0] for the current planetary state."""
        return 0.85 if self.is_applicable(current_state) else 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

