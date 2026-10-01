"""
noarco.mechanisms.co2_mobilization
===================================
Endogenous CO2 mobilisation mechanism.

References
----------
Jakosky, B. M. (2019). The CO2 inventory on Mars.
    Planetary and Space Science, 175, 52-59.
Jakosky, B. M. & Edwards, C. S. (2018).
    Inventory of CO2 available for terraforming Mars.
    Nature Astronomy, 2, 634-639.
Turyshev, S. G. (2026). arXiv:2603.00402, Sec. V (endogenous CO2, ~20 mbar reference case)
    and Table III (inventories).
"""

from __future__ import annotations

from enum import Enum

from noarco.core.constants import CO2_SUBLIMATION_J_KG, YEAR_S
from noarco.core.planet_state import PlanetState
from noarco.mechanisms.base import Mechanism, MechanismResult
from noarco.radiative.balance import RadiativeBalance


class MobilisationMode(str, Enum):
    POLAR_HEATING    = "polar_heating"
    ALBEDO_DARKENING = "albedo_darkening"


#: Mars CO2 reservoirs (kg). Every entry is tied to a published figure:
#:  * polar_deposit_south_kg: buried south-polar deposit ~6 mbar, 2.3e16 kg
#:    (Turyshev 2026, Sec. V.B, citing Jakosky & Edwards 2018).
#:  * accessible_reference_kg: "representative accessible endogenous CO2" ~20 mbar,
#:    7.78e16 kg (Turyshev 2026, Table III and Sec. V.A; Jakosky & Edwards 2018: ~0.02 bar).
#:  * crustal_upper_bound_kg: extreme *sequestered* bound, ~1 bar = 3.89e18 kg
#:    (Jakosky 2019: between ~10 mbar and ~1 bar is stored in regolith/crust). It is
#:    NOT an accessible inventory: mining it is a carbonate-processing problem.
_MARS_CO2_INVENTORY = {
    "polar_deposit_south_kg": 2.3e16,
    "accessible_reference_kg": 7.78e16,
    "crustal_upper_bound_kg": 3.89e18,
}

_L_SUB_CO2 = CO2_SUBLIMATION_J_KG

#: Upper bound of the water-vapour feedback fraction f_w in dT_eff = dT_dry (1 + f_w): an *estimated* bound
#: (a doubling of the dry response); the 1-bar published case implies f_w ~ 1.0 (see noarco.validation).
WATER_VAPOR_FEEDBACK_MAX = 1.0

_FORCING_WARNING = (
    "CO2 greenhouse forcing uses a logarithmic law (alpha per doubling); for a thin, "
    "partly band-saturated Mars atmosphere this is an engineering estimate. Turyshev (2026) "
    "cites climate models giving < 10 K warming for ~20 mbar CO2."
)


class CO2Mobilization(Mechanism):
    """Endogenous CO2 mobilisation (polar cap sublimation) mechanism."""

    def __init__(
        self,
        mode: MobilisationMode = MobilisationMode.POLAR_HEATING,
        co2_inventory_kg: float | None = None,
        include_regolith: bool = False,
        heat_source_efficiency: float = 0.70,
        forcing_per_doubling_wm2: float = 6.0,
        radiative_mode: str = "simple",
        water_vapor_feedback_fraction: float = 0.0,
    ) -> None:
        if radiative_mode not in ("simple", "literature_calibrated"):
            raise ValueError("radiative_mode must be 'simple' or 'literature_calibrated'")
        if not 0.0 <= water_vapor_feedback_fraction <= WATER_VAPOR_FEEDBACK_MAX:
            raise ValueError(f"water_vapor_feedback_fraction must be in [0, {WATER_VAPOR_FEEDBACK_MAX}]")
        if mode is MobilisationMode.ALBEDO_DARKENING:
            raise NotImplementedError(
                "ALBEDO_DARKENING is not modelled: only POLAR_HEATING (sublimation energy balance) is."
            )
        super().__init__(
            name=f"CO2Mobilization ({mode.value})",
            description=(
                "Sublimation of polar CO2 ice to increase atmospheric pressure. "
                f"Mode: {mode.value}. "
                "Accessible endogenous CO2 is an order-of-tens-of-mbar resource "
                "(~20 mbar reference; Turyshev 2026, Sec. V)."
            ),
        )
        self.mode = mode
        self.co2_inventory_kg = co2_inventory_kg
        self.include_regolith = include_regolith
        self.heat_source_efficiency = heat_source_efficiency
        self.forcing_per_doubling_wm2 = forcing_per_doubling_wm2
        self.radiative_mode = radiative_mode
        self.water_vapor_feedback_fraction = water_vapor_feedback_fraction

    def is_applicable(self, planet: PlanetState) -> bool:
        return self._get_inventory(planet) > 0.0

    def _get_inventory(self, planet: PlanetState) -> float:
        if self.co2_inventory_kg is not None:
            return self.co2_inventory_kg
        if planet.co2_ice_kg is not None:
            base = planet.co2_ice_kg
        elif planet.body_name == "Mars":
            base = _MARS_CO2_INVENTORY["polar_deposit_south_kg"]
        else:
            base = 0.0  # never borrow Mars' reservoirs for another body
        if self.include_regolith and planet.body_name == "Mars":
            # Polar + adsorbed + near-surface carbonates: the representative accessible case.
            base = max(base, _MARS_CO2_INVENTORY["accessible_reference_kg"])
        return base

    def _delta_pressure_from_co2(self, planet: PlanetState, co2_mass_kg: float) -> float:
        return co2_mass_kg * planet.gravity_ms2 / planet.surface_area_m2

    def _power_to_sublimate(self, co2_mass_kg: float, years: float) -> float:
        t = years * YEAR_S
        return (co2_mass_kg * _L_SUB_CO2) / (t * self.heat_source_efficiency)

    # ------------------------------------------------------------------ radiative modes
    def _warming_for_delta_p(self, planet: PlanetState, rb: RadiativeBalance, delta_p: float) -> tuple[float, float]:
        """``(delta_T, delta_F)`` for an added CO2 pressure, in the selected radiative mode."""
        x_co2 = planet.gas_composition.mole_fraction("CO2")
        p_old = x_co2 * planet.surface_pressure_pa
        if delta_p <= 0 or p_old <= 0:
            return 0.0, 0.0
        if self.radiative_mode == "literature_calibrated":
            from noarco.radiative.literature import mars_co2_warming_k

            dt = mars_co2_warming_k(planet.surface_pressure_pa + delta_p)[1]
            return dt, rb.greenhouse_forcing_for_temperature_rise(dt) if dt > 0 else 0.0
        d_f = rb.forcing_from_pressure_ratio(p_old, p_old + delta_p, self.forcing_per_doubling_wm2)
        dt_dry = rb.temperature_rise_for_greenhouse_forcing(d_f)
        return dt_dry * (1.0 + self.water_vapor_feedback_fraction), d_f

    def _delta_p_for_warming(self, planet: PlanetState, rb: RadiativeBalance, delta_t: float) -> float:
        """Added CO2 pressure (Pa) that produces ``delta_t`` in the selected radiative mode."""
        if delta_t <= 0:
            return 0.0
        if self.radiative_mode == "literature_calibrated":
            from noarco.radiative.literature import mars_pressure_for_warming

            return max(mars_pressure_for_warming(delta_t) - planet.surface_pressure_pa, 0.0)
        p_old = planet.gas_composition.mole_fraction("CO2") * planet.surface_pressure_pa
        d_f = rb.greenhouse_forcing_for_temperature_rise(delta_t / (1.0 + self.water_vapor_feedback_fraction))
        return float(p_old * (2.0 ** (d_f / self.forcing_per_doubling_wm2)) - p_old) if p_old > 0 else 0.0

    def _mode_warnings(self) -> list[str]:
        if self.radiative_mode == "literature_calibrated":
            return [(
                "RADIATIVE MODE literature_calibrated: warming is interpolated from published anchors "
                "(Jakosky & Edwards 2018; 20 mbar < 10 K bound, ~1 bar ~ 60 K with an assumed +/-10 K range), "
                "not computed. Use for stating a published number; Mars only."
            )]
        w = [
            _FORCING_WARNING,
            (
                "RADIATIVE MODE simple: absolute warming is biased low versus GCMs at high CO2 pressure "
                "(~2x low at 1 bar, see noarco.validation); do not use it as a hard dT gate. Mass, power and "
                "throughput are unaffected."
            ),
        ]
        if self.water_vapor_feedback_fraction > 0:
            w.append(f"Water-vapour feedback fraction f_w = {self.water_vapor_feedback_fraction:.2f} applied "
                     "(dT_eff = dT_dry x (1 + f_w)); an estimated, bounded parameter, not a computed feedback.")
        return w

    def compute(
        self,
        planet: PlanetState,
        target_delta_t_k: float | None = None,
        target_delta_f_wm2: float | None = None,
        deployment_years: float = 200.0,
    ) -> MechanismResult:
        if self.radiative_mode == "literature_calibrated" and planet.body_name != "Mars":
            raise ValueError("radiative='literature_calibrated' has a published curve for Mars only")
        inventory_kg = self._get_inventory(planet)
        rb = RadiativeBalance(planet)
        area, g = planet.surface_area_m2, planet.gravity_ms2

        delta_p_max = self._delta_pressure_from_co2(planet, inventory_kg)
        try:
            delta_t_max, delta_f_max = self._warming_for_delta_p(planet, rb, delta_p_max)
        except ValueError:                                  # beyond the published curve
            delta_t_max, delta_f_max = self._warming_for_delta_p(planet, rb, 1.5e5 - planet.surface_pressure_pa)
        warnings = self._mode_warnings()
        refs = [
            "Jakosky, B. M. (2019). Planet. Space Sci. 175, 52-59.",
            "Jakosky, B. M. & Edwards, C. S. (2018). Nature Astronomy 2, 634-639.",
            "Turyshev, S. G. (2026). arXiv:2603.00402, Sec. V and Table III.",
        ]
        mclass = "literature_calibrated" if self.radiative_mode == "literature_calibrated" else "engineering_estimate"

        if target_delta_t_k is not None and target_delta_t_k > delta_t_max * 1.05:
            return MechanismResult(
                mechanism_name=self.name, total_mass_kg=inventory_kg,
                power_w=self._power_to_sublimate(inventory_kg, deployment_years),
                throughput_kg_s=inventory_kg / (deployment_years * YEAR_S),
                delta_forcing_wm2=delta_f_max, delta_temperature_k=delta_t_max,
                deployment_time_years=deployment_years, effect_duration_years=float("inf"),
                risk_index=0.25, isru_fraction=1.0,
                notes=(
                    f"[MATE R3 PROJECTION] Target ΔT={target_delta_t_k:.1f} K exceeds "
                    f"max achievable ΔT={delta_t_max:.2f} K from all endogenous CO2. "
                    f"Available inventory: {inventory_kg:.2e} kg. "
                    f"Max ΔP: {delta_p_max:.1f} Pa ({delta_p_max/100:.2f} mbar). "
                    f"CONCLUSION: CO2 mobilisation alone CANNOT reach target."
                ),
                model_class=mclass, warnings=warnings, source_references=refs,
            )

        if target_delta_t_k is not None:
            delta_t = min(target_delta_t_k, delta_t_max)
            delta_p_needed = self._delta_p_for_warming(planet, rb, delta_t)
        elif target_delta_f_wm2 is not None:
            delta_f_t = min(target_delta_f_wm2, delta_f_max)
            delta_t = rb.temperature_rise_for_greenhouse_forcing(delta_f_t) * (
                1.0 + (self.water_vapor_feedback_fraction if self.radiative_mode == "simple" else 0.0))
            delta_t = min(delta_t, delta_t_max)
            delta_p_needed = self._delta_p_for_warming(planet, rb, delta_t)
        else:
            delta_t, delta_p_needed = delta_t_max, delta_p_max
        delta_t_final, delta_f = self._warming_for_delta_p(planet, rb, delta_p_needed)
        delta_t = delta_t_final if delta_p_needed > 0 else 0.0
        co2_mass_needed = min(delta_p_needed * area / g, inventory_kg)

        power_w = self._power_to_sublimate(co2_mass_needed, deployment_years)
        throughput_kg_s = co2_mass_needed / (deployment_years * YEAR_S)
        notes = (
            f"CO2 inventory used: {co2_mass_needed:.2e} kg "
            f"(of {inventory_kg:.2e} kg available). "
            f"ΔP achieved: {delta_p_needed:.1f} Pa ({delta_p_needed/100:.2f} mbar). "
            f"Power required: {power_w:.2e} W. "
            f"Effect is permanent."
        )
        return MechanismResult(
            mechanism_name=self.name, total_mass_kg=co2_mass_needed, power_w=power_w,
            throughput_kg_s=throughput_kg_s, delta_forcing_wm2=delta_f, delta_temperature_k=delta_t,
            deployment_time_years=deployment_years, effect_duration_years=float("inf"),
            risk_index=0.25, isru_fraction=1.0, notes=notes, model_class=mclass,
            warnings=warnings, source_references=refs,
        )
