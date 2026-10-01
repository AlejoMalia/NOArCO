"""
noarco.mechanisms.aerogel
=========================
Silica aerogel paraterraforming mechanism (regional solid-state greenhouse).

A cm-scale layer of silica aerogel is transparent to visible light but has very
low thermal conductivity and is opaque to thermal infrared. Sunlight is
absorbed at the base of the layer, and the layer's thermal resistance lifts the
temperature *beneath* it above the temperature of its top surface.

Physics (steady state, one dimension)
-------------------------------------
The top of the layer radiates to space, so the planetary energy budget is
unchanged and the top sits at the no-greenhouse temperature

    sigma T_top^4 = f S (1 - A)                                   (top balance)

with ``f`` the geometric/diurnal insolation factor (1/4 for a global mean). The
base absorbs the transmitted visible flux ``F_b = f S (1 - A_s) T_vis`` and
loses it by conduction and through the infrared window:

    F_b = k (T_b - T_top) / d + T_ir sigma (T_b^4 - T_top^4)      (base balance)

which is solved for ``T_b``. No empirical cap is imposed. The default
``k = 0.01 W/m/K`` is the value Wordsworth et al. (2019) quote for silica
aerogel at martian atmospheric pressures (0.02 W/m/K at 1 bar).

Because the top-of-atmosphere budget is unchanged, the mechanism has **zero
global radiative forcing**; it is a *local* temperature change over the covered
area. ``MechanismResult.delta_forcing_wm2`` is therefore 0 and
``area_fraction`` carries the coverage.

Limits of this model (reported in every result)
------------------------------------------------
Steady-state, no diurnal/seasonal storage, no atmospheric back-radiation, no
latent-heat exchange and no lateral or base heat loss. It therefore returns an
**upper bound** on the sub-layer warming. The value of ``k`` (effective, including
radiative conduction), ``T_vis`` and ``T_ir`` must be measured for the actual
aerogel. Wordsworth et al. (2019) measured warming > 45 K under a 3 cm aerogel
particle layer at 150 W/m2 (and > 50 K for 2 cm tiles) in a laboratory set-up
that loses heat through its sides and base, and report, from a coupled
radiative-thermal model, sub-layer temperatures above the melting point of
water throughout the year for a 2.5 cm layer at one mid-latitude ice-rich site.

References
----------
Wordsworth, R., Kerber, L. & Cockell, C. (2019). Nature Astronomy 3, 898-903.
Turyshev, S. G. (2026). arXiv:2603.00402, Sec. VI (regional aerogel) and Table VIII.
"""

from __future__ import annotations

from scipy.optimize import brentq

from noarco.core.constants import SIGMA_SB, WATER_TRIPLE_POINT_K, WATER_TRIPLE_POINT_PA, YEAR_S
from noarco.core.planet_state import PlanetState
from noarco.mechanisms.base import Mechanism, MechanismResult


class SilicaAerogel(Mechanism):
    """Silica aerogel solid-state greenhouse (regional paraterraforming)."""

    def __init__(
        self,
        thickness_m: float = 0.025,
        coverage_area_m2: float = 1e10,
        aerogel_density_kg_m3: float = 150.0,
        transmittance_vis: float = 0.75,
        transmittance_ir: float = 0.02,
        lifetime_years: float = 10.0,
        thermal_conductivity_w_m_k: float = 0.01,
        insolation_factor: float = 0.25,
        energy_per_kg_aerogel_j: float = 100e6,
    ) -> None:
        if thickness_m <= 0 or coverage_area_m2 <= 0 or aerogel_density_kg_m3 <= 0:
            raise ValueError("thickness, coverage area and density must be > 0.")
        if not (0.0 < transmittance_vis <= 1.0 and 0.0 <= transmittance_ir <= 1.0):
            raise ValueError("transmittances must be in (0, 1] (visible) and [0, 1] (IR).")
        if thermal_conductivity_w_m_k <= 0 or not 0.0 < insolation_factor <= 1.0:
            raise ValueError("conductivity must be > 0 and insolation_factor in (0, 1].")
        super().__init__(
            name="SilicaAerogel (solid-state greenhouse)",
            description=(
                f"Silica aerogel layer ({thickness_m*100:.1f} cm) over "
                f"{coverage_area_m2/1e6:.1f} km2. "
                f"T_vis={transmittance_vis:.2f}, T_IR={transmittance_ir:.3f}. "
                "Regional paraterraforming. Ref: Wordsworth et al. (2019)."
            ),
        )
        self.thickness_m = thickness_m
        self.coverage_area_m2 = coverage_area_m2
        self.aerogel_density_kg_m3 = aerogel_density_kg_m3
        self.transmittance_vis = transmittance_vis
        self.transmittance_ir = transmittance_ir
        self.lifetime_years = lifetime_years
        self.thermal_conductivity_w_m_k = thermal_conductivity_w_m_k
        self.insolation_factor = insolation_factor
        self.energy_per_kg_aerogel_j = energy_per_kg_aerogel_j

    def is_applicable(self, planet: PlanetState) -> bool:
        return planet.solar_constant_wm2 > 10.0

    def top_temperature_k(self, planet: PlanetState) -> float:
        """Temperature of the aerogel's top surface (no-greenhouse balance)."""
        absorbed = self.insolation_factor * planet.solar_constant_wm2 * (1.0 - planet.surface_albedo)
        return float((absorbed / SIGMA_SB) ** 0.25)

    def _sub_aerogel_temperature(self, planet: PlanetState) -> float:
        """Solve the base energy balance for the temperature under the layer (K)."""
        t_top = self.top_temperature_k(planet)
        f_b = (self.insolation_factor * planet.solar_constant_wm2
               * (1.0 - planet.surface_albedo) * self.transmittance_vis)
        k, d, t_ir = self.thermal_conductivity_w_m_k, self.thickness_m, self.transmittance_ir

        def residual(t_b: float) -> float:
            return k * (t_b - t_top) / d + t_ir * SIGMA_SB * (t_b**4 - t_top**4) - f_b

        # residual(t_top) = -f_b < 0 and grows monotonically with t_b.
        return float(brentq(residual, t_top, t_top + 5000.0, xtol=1e-9))

    def compute(
        self,
        planet: PlanetState,
        target_delta_t_k: float | None = None,
        target_delta_f_wm2: float | None = None,
        deployment_years: float = 5.0,
    ) -> MechanismResult:
        if not self.is_applicable(planet):
            raise ValueError(f"SilicaAerogel not applicable: S={planet.solar_constant_wm2} W/m2")

        t_top = self.top_temperature_k(planet)
        t_sub = self._sub_aerogel_temperature(planet)
        delta_t = t_sub - t_top
        area_fraction = min(self.coverage_area_m2 / planet.surface_area_m2, 1.0)

        mass_per_m2 = self.aerogel_density_kg_m3 * self.thickness_m
        total_mass = mass_per_m2 * self.coverage_area_m2
        sio2_fraction = (planet.regolith_composition or {}).get("SiO2", 0.0)
        # Aerogel is ~all SiO2 by mass: regolith with >= 20 wt% SiO2 is treated as an
        # adequate feedstock; below that the in-situ share scales down (heuristic).
        isru_fraction = min(sio2_fraction / 0.20, 1.0)

        seconds = deployment_years * YEAR_S
        total_energy = total_mass * self.energy_per_kg_aerogel_j
        power_w = total_energy / seconds
        throughput_kg_s = total_mass / seconds
        annual_replacement = total_mass / self.lifetime_years

        warnings = [
            ("UPPER BOUND: steady-state slab model: no diurnal/seasonal storage, no atmospheric "
            "back-radiation, no lateral losses; k_eff, T_vis and T_IR must be measured."),
            ("Global radiative forcing is zero (top-of-atmosphere budget unchanged); the "
            "warming is local to the covered area."),
            "Manufacturing energy per kg is an engineering placeholder (default 100 MJ/kg).",
        ]
        if planet.surface_pressure_pa < WATER_TRIPLE_POINT_PA:
            warnings.append(
                f"Mean pressure {planet.surface_pressure_pa:.0f} Pa is below the water triple point "
                f"({WATER_TRIPLE_POINT_PA:.0f} Pa): liquid water requires locally higher pressure "
                "(low elevation) or a sealed layer."
            )
        stable = t_sub >= WATER_TRIPLE_POINT_K
        notes = (
            f"Coverage: {self.coverage_area_m2/1e6:.1f} km2 ({area_fraction:.2e} of the surface). "
            f"Layer thickness: {self.thickness_m*100:.1f} cm. "
            f"Top T: {t_top:.1f} K, sub-aerogel T: {t_sub:.1f} K, local dT = {delta_t:.1f} K. "
            f"Mass/m2: {mass_per_m2:.2f} kg/m2. "
            f"Annual replacement: {annual_replacement:.2e} kg/yr. "
            f"Sub-aerogel T above water triple point: {'YES' if stable else 'NO'}."
        )
        return MechanismResult(
            mechanism_name=self.name,
            total_mass_kg=total_mass,
            power_w=power_w,
            throughput_kg_s=throughput_kg_s,
            delta_forcing_wm2=0.0,
            delta_temperature_k=delta_t,
            deployment_time_years=deployment_years,
            effect_duration_years=self.lifetime_years,
            risk_index=0.15,
            isru_fraction=isru_fraction,
            notes=notes,
            area_fraction=area_fraction,
            warnings=warnings,
            model_class="screening_upper_bound",
            source_references=[
                "Wordsworth, R. et al. (2019). Nature Astronomy 3, 898-903.",
                "Turyshev, S. G. (2026). arXiv:2603.00402, Sec. VI and Table VIII.",
            ],
        )
