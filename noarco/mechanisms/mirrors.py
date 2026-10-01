"""
noarco.mechanisms.mirrors
=========================
Orbital mirror / space reflector warming mechanism.

A reflector of area ``A_m`` at the planet's orbital distance intercepts
``S * A_m`` of sunlight; with an overall efficiency ``eta`` (reflectivity x
pointing x geometric losses) it adds an absorbed power ``eta * S * A_m``. For a
target *global-mean* top-of-atmosphere forcing ``dF``:

    A_m = 4 pi R^2 dF / (eta * S)                      (Turyshev 2026, Eq. 55)

The forcing needed for a temperature rise is the exact Stefan-Boltzmann balance
(Eq. 31), not its linearisation. Mars reference: dF = 20 W/m2, eta = 0.7,
S = 589 W/m2 gives A_m ~ 7.0e12 m2 (Eq. 56).

Not modelled (reported as warnings): mirror pointing and station-keeping
power, degradation, non-Keplerian orbit control, and the areal-mass budget of
structures (only a thin-film mass is computed).

References
----------
Turyshev, S. G. (2026). arXiv:2603.00402, Sec. VI.D, Eqs. (54)-(58), Fig. 3.
Zubrin, R. & McKay, C. (1993). AIAA Paper 93-2005.
McKay, C. P., Toon, O. B. & Kasting, J. F. (1991). Nature 352, 489-496.
"""

from __future__ import annotations

from enum import Enum

from noarco.core.constants import YEAR_S
from noarco.core.planet_state import PlanetState
from noarco.mechanisms.base import Mechanism, MechanismResult
from noarco.radiative.balance import RadiativeBalance


class MirrorMode(str, Enum):
    GLOBAL = "global"
    POLAR = "polar"


class OrbitalMirror(Mechanism):
    """Space-based mirror / solar reflector warming mechanism."""

    def __init__(
        self,
        mode: MirrorMode = MirrorMode.GLOBAL,
        reflectance: float = 0.90,
        film_thickness_m: float = 1e-6,
        film_density_kg_m3: float = 2700.0,
        polar_area_fraction: float = 0.02,
        orbital_altitude_km: float = 100_000.0,
        pointing_efficiency: float = 0.7 / 0.9,
        stationkeeping_w_per_m2: float = 0.0,
    ) -> None:
        if not 0.0 < reflectance <= 1.0 or not 0.0 < pointing_efficiency <= 1.0:
            raise ValueError("reflectance and pointing_efficiency must be in (0, 1].")
        if not 0.0 < polar_area_fraction <= 1.0:
            raise ValueError("polar_area_fraction must be in (0, 1].")
        super().__init__(
            name=f"OrbitalMirror ({mode.value})",
            description=(
                f"Thin-film orbital reflector ({mode.value} mode). "
                f"Reflectance={reflectance:.0%}, film={film_thickness_m*1e6:.1f} µm Al. "
                "Ref: Turyshev (2026) Sec. VI.D; McKay et al. (1991)."
            ),
        )
        self.mode = mode
        self.reflectance = reflectance
        self.film_thickness_m = film_thickness_m
        self.film_density_kg_m3 = film_density_kg_m3
        self.polar_area_fraction = polar_area_fraction
        self.orbital_altitude_km = orbital_altitude_km
        self.pointing_efficiency = pointing_efficiency
        self.stationkeeping_w_per_m2 = stationkeeping_w_per_m2

    @property
    def system_efficiency(self) -> float:
        """Overall efficiency eta = reflectance x pointing/geometry (default 0.7, Turyshev 2026)."""
        return self.reflectance * self.pointing_efficiency

    def is_applicable(self, planet: PlanetState) -> bool:
        return planet.solar_constant_wm2 > 50.0

    def _mirror_area_for_global_forcing(self, planet: PlanetState, delta_f_wm2: float) -> float:
        """Mirror area (m2) for a global-mean forcing: ``A = 4 pi R^2 dF / (eta S)`` (Eq. 55)."""
        return (delta_f_wm2 * planet.surface_area_m2) / (
            planet.solar_constant_wm2 * self.system_efficiency
        )

    def _mirror_area_for_polar_forcing(self, planet: PlanetState, delta_f_wm2: float) -> float:
        """Mirror area (m2) delivering ``delta_f_wm2`` over the polar fraction of the surface."""
        a_polar = planet.surface_area_m2 * self.polar_area_fraction
        return (delta_f_wm2 * a_polar) / (planet.solar_constant_wm2 * self.system_efficiency)

    def compute(
        self,
        planet: PlanetState,
        target_delta_t_k: float | None = None,
        target_delta_f_wm2: float | None = None,
        deployment_years: float = 50.0,
    ) -> MechanismResult:
        if not self.is_applicable(planet):
            raise ValueError(f"OrbitalMirror not applicable: S={planet.solar_constant_wm2} W/m2")

        rb = RadiativeBalance(planet)
        if target_delta_t_k is not None:
            delta_t = target_delta_t_k
            delta_f = rb.forcing_needed_for_delta_t(delta_t)  # exact, Eq. (31)
        elif target_delta_f_wm2 is not None:
            delta_f = target_delta_f_wm2
            delta_t = rb.temperature_rise_for_forcing(delta_f)
        else:
            raise ValueError("Provide target_delta_t_k or target_delta_f_wm2.")
        if delta_f < 0:
            raise ValueError("Mirrors can only add absorbed flux: a negative forcing is not supported.")

        if self.mode == MirrorMode.GLOBAL:
            mirror_area = self._mirror_area_for_global_forcing(planet, delta_f)
            mode_note = "Global diffuse illumination mode."
            area_fraction = 1.0
        else:
            mirror_area = self._mirror_area_for_polar_forcing(planet, delta_f)
            mode_note = f"Polar focused mode (targeting {self.polar_area_fraction:.1%} of surface)."
            area_fraction = self.polar_area_fraction

        total_mass = mirror_area * self.film_density_kg_m3 * self.film_thickness_m
        power_w = mirror_area * self.stationkeeping_w_per_m2
        seconds = deployment_years * YEAR_S
        throughput_kg_s = total_mass / seconds

        al2o3 = (planet.regolith_composition or {}).get("Al2O3", 0.0)
        al_fraction = al2o3 * (2 * 26.98) / 101.96  # mass fraction of Al in Al2O3
        # Heuristic: >= 5 wt% Al in the regolith is treated as an adequate feedstock.
        isru_fraction = min(al_fraction / 0.05, 1.0)

        warnings = [
            ("Station-keeping/pointing power is not modelled (stationkeeping_w_per_m2 = "
            f"{self.stationkeeping_w_per_m2:g}); degradation and replacement are not included."),
            ("Areal mass is the thin metal film only; real reflector structures are 1-10 g/m2 "
            "or more (Turyshev 2026, Eq. 58)."),
        ]
        notes = (
            f"{mode_note} Efficiency eta = {self.system_efficiency:.2f}. "
            f"Mirror area: {mirror_area:.3e} m2 "
            f"({mirror_area**0.5/1000:.0f} x {mirror_area**0.5/1000:.0f} km equivalent). "
            f"Total mass: {total_mass:.2e} kg."
        )
        return MechanismResult(
            mechanism_name=self.name,
            total_mass_kg=total_mass,
            power_w=power_w,
            throughput_kg_s=throughput_kg_s,
            delta_forcing_wm2=delta_f * area_fraction,
            delta_temperature_k=delta_t,
            deployment_time_years=deployment_years,
            effect_duration_years=float("inf"),
            risk_index=0.65,
            isru_fraction=isru_fraction,
            notes=notes,
            area_fraction=area_fraction,
            warnings=warnings,
            model_class="reference_scaling",
            source_references=[
                "Turyshev, S. G. (2026). arXiv:2603.00402, Sec. VI.D, Eqs. (54)-(58).",
                "Zubrin, R. & McKay, C. (1993). AIAA Paper 93-2005.",
                "McKay, C. P. et al. (1991). Nature 352, 489-496.",
            ],
        )
