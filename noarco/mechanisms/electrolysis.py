"""
noarco.mechanisms.electrolysis
==============================
Oxygen production mechanisms: water electrolysis + photosynthesis.

References
----------
McKay, C. P. et al. (1991). Making Mars habitable. Nature 352, 489-496.
Turyshev, S. G. (2026). arXiv:2603.00402, Section 4.7.
DeBenedictis et al. (2025). Nature Astronomy 9, 634-639.
NASA MOXIE experiment (Mars 2020 Perseverance).
"""

from __future__ import annotations

from enum import Enum

from noarco.core.constants import DELTA_G_H2O_J_MOL, MOLAR_MASS, YEAR_S
from noarco.core.planet_state import PlanetState
from noarco.mechanisms.base import Mechanism, MechanismResult


class OxygenMode(str, Enum):
    ELECTROLYSIS   = "electrolysis"
    PHOTOSYNTHESIS = "photosynthesis"
    COMBINED       = "combined"


_MU_O2 = MOLAR_MASS["O2"]               # kg/mol
_MU_H2O = MOLAR_MASS["H2O"]              # kg/mol
_PHOTO_RATE_MARS_G_M2_DAY = 1.5          # g O2/m2/day Mars estimate


class Electrolysis(Mechanism):
    """O2 production via water electrolysis and/or photosynthesis."""

    def __init__(
        self,
        mode: OxygenMode = OxygenMode.ELECTROLYSIS,
        available_power_w: float = 1e12,
        electrolysis_efficiency: float = 0.70,
        photosynthesis_area_m2: float = 1e12,
        target_o2_partial_pressure_pa: float = 16_000.0,
    ) -> None:
        super().__init__(
            name=f"O2Production ({mode.value})",
            description=(
                f"Atmospheric O2 production via {mode.value}. "
                f"Target: pO2 = {target_o2_partial_pressure_pa/1000:.1f} kPa. "
                "Ref: McKay et al. (1991); Turyshev (2026); NASA MOXIE."
            ),
        )
        self.mode = mode
        self.available_power_w = available_power_w
        self.electrolysis_efficiency = electrolysis_efficiency
        self.photosynthesis_area_m2 = photosynthesis_area_m2
        self.target_o2_partial_pressure_pa = target_o2_partial_pressure_pa

    def is_applicable(self, planet: PlanetState) -> bool:
        has_water = (planet.h2o_ice_kg or 0.0) > 0.0
        has_co2_for_photo = planet.gas_composition.mole_fraction("CO2") > 0.01
        return has_water or (self.mode == OxygenMode.PHOTOSYNTHESIS and has_co2_for_photo)

    def _o2_mass_needed(self, planet: PlanetState) -> float:
        current_po2 = planet.gas_composition.mole_fraction("O2") * planet.surface_pressure_pa
        delta_po2 = max(0.0, self.target_o2_partial_pressure_pa - current_po2)
        return delta_po2 * planet.surface_area_m2 / planet.gravity_ms2

    def _electrolysis_rate_kg_s(self) -> float:
        # Reversible (Gibbs) work per mol O2 is 2 dG (two H2O are split): 474 kJ/mol = 14.8 MJ/kg.
        # ``electrolysis_efficiency`` = reversible work / actual electrical input (0.70 -> 21.2 MJ/kg).
        energy_per_mol_o2 = 2.0 * DELTA_G_H2O_J_MOL
        power_effective = self.available_power_w * self.electrolysis_efficiency
        mol_per_s = power_effective / energy_per_mol_o2
        return mol_per_s * _MU_O2

    def _photosynthesis_rate_kg_s(self, planet: PlanetState) -> float:
        solar_scale = planet.solar_constant_wm2 / 1361.0
        rate_g_m2_day = _PHOTO_RATE_MARS_G_M2_DAY * min(solar_scale, 1.0)
        rate_kg_m2_s = rate_g_m2_day * 1e-3 / (24 * 3600)
        return rate_kg_m2_s * self.photosynthesis_area_m2

    def compute(
        self,
        planet: PlanetState,
        target_delta_t_k: float | None = None,
        target_delta_f_wm2: float | None = None,
        deployment_years: float = 1000.0,
    ) -> MechanismResult:
        if not self.is_applicable(planet):
            raise ValueError(
                f"Electrolysis/Photosynthesis not applicable to {planet.body_name}: "
                f"no H2O available and CO2 too low."
            )

        o2_needed_kg = self._o2_mass_needed(planet)

        if self.mode == OxygenMode.ELECTROLYSIS:
            rate_kg_s = self._electrolysis_rate_kg_s()
            mode_note = f"PEM electrolysis at {self.available_power_w:.2e} W. "
        elif self.mode == OxygenMode.PHOTOSYNTHESIS:
            rate_kg_s = self._photosynthesis_rate_kg_s(planet)
            mode_note = f"Photosynthesis over {self.photosynthesis_area_m2:.2e} m2. "
        else:
            rate_elec = self._electrolysis_rate_kg_s()
            rate_photo = self._photosynthesis_rate_kg_s(planet)
            rate_kg_s = rate_elec + rate_photo
            mode_note = f"Combined electrolysis + photosynthesis ({rate_kg_s:.2e} kg/s). "

        if rate_kg_s > 0:
            time_to_target_s = o2_needed_kg / rate_kg_s
            time_to_target_yr = time_to_target_s / (YEAR_S)
        else:
            time_to_target_yr = float("inf")

        seconds = deployment_years * YEAR_S
        o2_produced = rate_kg_s * seconds
        fraction_complete = min(o2_produced / o2_needed_kg, 1.0) if o2_needed_kg > 0 else 1.0

        h2o_consumed = o2_needed_kg * (_MU_H2O * 2) / _MU_O2
        h2o_available = planet.h2o_ice_kg or 0.0
        h2o_feasible = h2o_available >= h2o_consumed if self.mode != OxygenMode.PHOTOSYNTHESIS else True

        warnings = [
            ("O2 production only: no buffer gas (N2/Ar), no O2 sink filling (oxidation of "
            "FeO-bearing regolith can absorb ~1e18 kg, Turyshev 2026 Eq. 76) and no loss terms."),
        ]
        if self.mode != OxygenMode.ELECTROLYSIS:
            warnings.append(
                "Photosynthetic rate is a gross, Earth-average NPP-equivalent O2 flux scaled by "
                "insolation; only the burial/export fraction beta << 1 accumulates as net "
                "atmospheric O2 (Turyshev 2026, Eq. 74), so this is an upper bound."
            )
        notes = (
            f"{mode_note}"
            f"O2 needed for pO2={self.target_o2_partial_pressure_pa:.0f} Pa: {o2_needed_kg:.2e} kg. "
            f"Time to target: {time_to_target_yr:.0f} yr. "
            f"O2 produced in {deployment_years:.0f} yr: {o2_produced:.2e} kg ({fraction_complete:.1%}). "
            f"H2O consumed: {h2o_consumed:.2e} kg (available: {h2o_available:.2e} kg, feasible={h2o_feasible}). "
            f"Note: O2 is not a GHG (dT=0)."
        )

        return MechanismResult(
            mechanism_name=self.name,
            total_mass_kg=o2_produced,
            power_w=self.available_power_w if self.mode != OxygenMode.PHOTOSYNTHESIS else 0.0,
            throughput_kg_s=rate_kg_s,
            delta_forcing_wm2=0.0,
            delta_temperature_k=0.0,
            deployment_time_years=time_to_target_yr,
            effect_duration_years=float("inf"),
            risk_index=0.30,
            isru_fraction=1.0,
            notes=notes,
            model_class="first_principles" if self.mode == OxygenMode.ELECTROLYSIS else "engineering_estimate",
            warnings=warnings,
            source_references=[
                "McKay, C. P. et al. (1991). Nature 352, 489-496.",
                "Turyshev, S. G. (2026). arXiv:2603.00402, Sec. VII and Eqs. (5), (67)-(74).",
                "DeBenedictis et al. (2025). Nature Astronomy 9, 634-639.",
            ],
        )


# Canonical alias for compatibility
WaterElectrolysis = Electrolysis

