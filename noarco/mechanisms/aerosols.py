"""
noarco.mechanisms.aerosols
==========================
Engineered IR-active aerosol (nanoparticle) warming, in the optical-depth /
column-mass framework of Turyshev (2026), Sec. VI.C.

Physics
-------
A particle population with column mass ``Sigma`` (kg m^-2) and mass absorption
coefficient ``kappa`` (m^2 kg^-1) has infrared absorption optical depth

    tau_a = kappa * Sigma

* **Mass.** For spheres (geometric optics) ``kappa = 3 Q_abs / (4 rho r)``;
  engineered nanorods are described directly by ``kappa`` (see below).
* **Maintenance.** With atmospheric residence time ``tau_p`` the injection rate that
  holds the column is ``Mdot = Sigma * A_planet / tau_p`` (Turyshev 2026, Eq. 53).
  Particle pathways are therefore *maintenance-dominated* (Eqs. 45-49).
* **Greenhouse mapping.** The grey atmosphere of Turyshev (2026), Eqs. (32)-(33),
  ``T_s^4 = (3/4) T_e^4 (tau + 2/3)``, is calibrated to the present surface temperature
  (``tau_0 = (4/3)(T_s/T_e)^4 - 2/3``) and the aerosol adds ``tau_a`` at fixed absorbed
  sunlight. The surface warming follows directly; the top-of-atmosphere forcing is the
  reduction of outgoing longwave radiation at fixed ``T_s``,
  ``dF = sigma T_s^4 [g(tau_0) - g(tau_0 + tau_a)]`` with ``g(tau) = 4/(3 tau + 2)``.
  ``dF`` saturates at the present OLR; forcing targets at or above it are reported as
  unreachable instead of being silently clipped.

Calibration of the nanorod presets (two independent sources, which disagree)
-----------------------------------------------------------------------------
* ``NANOROD`` (default) is calibrated to the **MarsWRF 3-D steady-state runs** of Richardson et al.
  (2025), Table S3 (60 nm x 8 um Al rods): the global warming reached for a given column density.
  A least-squares fit of the grey mapping above gives ``kappa = 3.15e4 m^2/kg`` (rms 18 %), and the
  column/release-rate balance of the same table gives a residence time of ~1.15 yr. Graphene disks:
  ``7.08e4 m^2/kg`` (rms 11 %), ~1.3 yr. ``noarco.validation`` repeats the fit with leave-one-out
  cross-validation.
* ``NANOROD_TURYSHEV`` uses the scaling of Turyshev (2026) Eq. 52-53 (column 1.6-5.4 mg/m2 for 30-100 d
  residence, +30 K), which implies ``kappa ~ 2.7e5 m^2/kg``. It is **~9x more optimistic** than the
  GCM-calibrated value (its short residence time and small column are not what the GCM produces); it
  is kept for reproducing the paper's numbers.

Not modelled (warnings are attached to every result)
----------------------------------------------------
Particle microphysics (coagulation, settling), lofting and global dispersal
(Richardson et al. 2025), visible-light absorption/scattering and their effect on
the solar budget, water-cycle feedbacks, and non-grey spectral structure.

References
----------
Ansari, S. et al. (2024). Science Advances 10, eadn4650.
Richardson, M. I. et al. (2025). Atmospheric dynamics of IR-active particles released
    from Mars' surface. arXiv:2504.01455 (Geophys. Res. Lett.).
Turyshev, S. G. (2026). arXiv:2603.00402, Sec. VI.C, Eqs. (32)-(33), (45)-(53).
van de Hulst (1957). Light Scattering by Small Particles (geometric-optics limit).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from noarco.core.constants import SIGMA_SB, YEAR_S
from noarco.core.planet_state import PlanetState
from noarco.mechanisms.base import Mechanism, MechanismResult
from noarco.radiative.balance import RadiativeBalance


def _g(tau: float) -> float:
    """Grey OLR ratio ``g(tau) = 4 / (3 tau + 2)``."""
    return 4.0 / (3.0 * tau + 2.0)

#: Mass absorption coefficient (m^2/kg) implied by Turyshev (2026) Eqs. (33) and (52)-(53)
#: for +30 K on Mars: delta-tau = 0.945 over a 3.5 mg/m^2 column.
IMPLIED_NANOROD_KAPPA_M2_KG = 2.7e5
#: Least-squares fit of the grey mapping to MarsWRF steady states (Richardson et al. 2025, Table S3).
MARSWRF_AL_ROD_KAPPA_M2_KG = 3.15e4
MARSWRF_GRAPHENE_KAPPA_M2_KG = 7.08e4
#: Residence times from the same table's mass balance, Sigma * A / (rho * Vdot) (Earth years).
MARSWRF_AL_ROD_RESIDENCE_YEARS = 1.15
MARSWRF_GRAPHENE_RESIDENCE_YEARS = 1.30
#: Default residence time of engineered particles: mid-range of 30-100 d (Turyshev 2026).
DEFAULT_RESIDENCE_YEARS = 65.0 / 365.25
#: Optical-depth cap when a target cannot be reached (1 - exp(-5) = 99.3 % of the maximum).
_TAU_CAP = 5.0


class AerosolMaterial(str, Enum):
    """Supported particle populations."""

    NANOROD = "nanorod"      # Al nanorods, calibrated to MarsWRF (Richardson et al. 2025)
    NANOROD_TURYSHEV = "nanorod_turyshev"  # same rods, Turyshev (2026) scaling (~9x more optimistic)
    GRAPHENE_DISKS = "graphene_disks"      # doped graphene disks, MarsWRF-calibrated
    IRON = "iron"            # Fe spheres (geometric optics)
    ALUMINIUM = "aluminium"  # Al spheres
    ALUMINA = "alumina"      # Al2O3 spheres
    GRAPHENE = "graphene"    # Graphene/graphite flakes (as spheres)
    MAGNETITE = "magnetite"  # Fe3O4 spheres


#: Material parameters. ``kappa_m2_kg`` (if present) is used directly; otherwise
#: ``kappa = 3 Q_abs / (4 rho r)``. ``Q_abs_ir`` of the sphere presets are
#: *assumed* order-unity values, not Mie calculations: treat them as placeholders.
_MATERIAL_PARAMS: dict[str, dict[str, Any]] = {
    "nanorod": {
        "kappa_m2_kg": MARSWRF_AL_ROD_KAPPA_M2_KG,
        "rho_kg_m3": 2_700.0,  # aluminium rods
        "optimal_radius_m": 30e-9,  # 60 nm diameter x 8 um rods (Richardson et al. 2025)
        "isru_from": "Al2O3 / Fe2O3 reduction",
        "isru_efficiency": 0.50,
        "residence_time_years": MARSWRF_AL_ROD_RESIDENCE_YEARS,
    },
    "nanorod_turyshev": {
        "kappa_m2_kg": IMPLIED_NANOROD_KAPPA_M2_KG,
        "rho_kg_m3": 2_700.0,
        "optimal_radius_m": 30e-9,
        "isru_from": "Al2O3 / Fe2O3 reduction",
        "isru_efficiency": 0.50,
        "residence_time_years": DEFAULT_RESIDENCE_YEARS,
    },
    "graphene_disks": {
        "kappa_m2_kg": MARSWRF_GRAPHENE_KAPPA_M2_KG,
        "rho_kg_m3": 2_267.0,
        "optimal_radius_m": 125e-9,
        "isru_from": "CO2 (carbon) + energy-intensive reduction",
        "isru_efficiency": 0.20,
        "residence_time_years": MARSWRF_GRAPHENE_RESIDENCE_YEARS,
    },
    "iron": {"Q_abs_ir": 0.80, "rho_kg_m3": 7_874.0, "optimal_radius_m": 0.9e-6,
             "isru_from": "Fe2O3 (hematite)", "isru_efficiency": 0.60,
             "residence_time_years": 0.5},
    "aluminium": {"Q_abs_ir": 0.40, "rho_kg_m3": 2_700.0, "optimal_radius_m": 1.0e-6,
                  "isru_from": "Al2O3 (alumina in regolith)", "isru_efficiency": 0.40,
                  "residence_time_years": 0.8},
    "alumina": {"Q_abs_ir": 0.60, "rho_kg_m3": 3_980.0, "optimal_radius_m": 0.5e-6,
                "isru_from": "Al2O3 in regolith", "isru_efficiency": 0.50,
                "residence_time_years": 1.5},
    "graphene": {"Q_abs_ir": 0.95, "rho_kg_m3": 2_267.0, "optimal_radius_m": 1.5e-6,
                 "isru_from": "CO2 (carbon) + energy-intensive reduction",
                 "isru_efficiency": 0.20, "residence_time_years": 2.0},
    "magnetite": {"Q_abs_ir": 0.70, "rho_kg_m3": 5_175.0, "optimal_radius_m": 0.7e-6,
                  "isru_from": "Fe3O4 in regolith", "isru_efficiency": 0.55,
                  "residence_time_years": 0.7},
}


class NanoparticleAerosol(Mechanism):
    """Engineered IR-active aerosol warming mechanism.

    Parameters
    ----------
    material : AerosolMaterial
        Particle population. Default: ``NANOROD``, calibrated to the MarsWRF runs of Richardson et al. (2025).
    particle_radius_m : float, optional
        Sphere radius (m) for the geometric-optics presets. Ignored for ``NANOROD``.
    target_optical_depth : float, optional
        Aerosol IR optical depth to deploy. If None it is derived from the target.
    coverage_fraction : float
        Fraction of the planetary surface carrying the layer (1.0 = global).
    mass_absorption_coefficient_m2_kg : float, optional
        Override of ``kappa`` (m^2/kg) when measured or modelled values exist.
    residence_time_years : float, optional
        Override of the atmospheric residence time.
    production_energy_j_per_kg : float
        Energy to manufacture particles (engineering placeholder, 3.5e6 J/kg).
    """

    def __init__(
        self,
        material: AerosolMaterial = AerosolMaterial.NANOROD,
        particle_radius_m: float | None = None,
        target_optical_depth: float | None = None,
        coverage_fraction: float = 1.0,
        mass_absorption_coefficient_m2_kg: float | None = None,
        residence_time_years: float | None = None,
        production_energy_j_per_kg: float = 3.5e6,
    ) -> None:
        if not 0.0 < coverage_fraction <= 1.0:
            raise ValueError("coverage_fraction must be in (0, 1].")
        if target_optical_depth is not None and target_optical_depth < 0:
            raise ValueError("target_optical_depth must be >= 0.")
        super().__init__(
            name=f"NanoparticleAerosol ({material.value})",
            description=(
                f"Engineered {material.value} particles suspended in the atmosphere to add "
                "infrared optical depth (Ansari et al. 2024; Turyshev 2026, Sec. VI.C)."
            ),
        )
        self.material = material
        self._params = _MATERIAL_PARAMS[material.value]
        self.particle_radius_m = particle_radius_m or self._params["optimal_radius_m"]
        self.target_optical_depth = target_optical_depth
        self.coverage_fraction = coverage_fraction
        self._kappa_override = mass_absorption_coefficient_m2_kg
        self._residence_override = residence_time_years
        self.production_energy_j_per_kg = production_energy_j_per_kg

    # ------------------------------------------------------------------ properties
    @property
    def mass_absorption_coefficient_m2_kg(self) -> float:
        """kappa (m^2/kg): override, preset value, or ``3 Q / (4 rho r)`` for spheres."""
        if self._kappa_override is not None:
            if self._kappa_override <= 0:
                raise ValueError("mass absorption coefficient must be > 0.")
            return self._kappa_override
        if "kappa_m2_kg" in self._params:
            return float(self._params["kappa_m2_kg"])
        q = self._params["Q_abs_ir"]
        return float(3.0 * q / (4.0 * self._params["rho_kg_m3"] * self.particle_radius_m))

    @property
    def residence_time_years(self) -> float:
        return float(self._residence_override or self._params["residence_time_years"])

    def is_applicable(self, planet: PlanetState) -> bool:
        """Particles need an atmosphere to stay aloft (at least ~1 Pa)."""
        return planet.surface_pressure_pa > 1.0

    # ------------------------------------------------------------------ physics
    def _tau_now(self, planet: PlanetState) -> float:
        rb = RadiativeBalance(planet)
        return max(rb.required_ir_optical_depth(planet.mean_temperature_k), 0.0)

    def max_forcing_wm2(self, planet: PlanetState) -> float:
        """Saturation forcing ``sigma T_s^4 g(tau_0)`` (the present OLR) times coverage."""
        ts = planet.mean_temperature_k
        return float(SIGMA_SB * ts**4 * _g(self._tau_now(planet)) * self.coverage_fraction)

    def forcing_for_optical_depth(self, planet: PlanetState, tau_a: float) -> float:
        """Global-mean TOA forcing (W/m2) of an aerosol optical depth ``tau_a`` over the coverage."""
        if tau_a < 0:
            raise ValueError("tau_a must be >= 0.")
        ts = planet.mean_temperature_k
        tau0 = self._tau_now(planet)
        return float(SIGMA_SB * ts**4 * (_g(tau0) - _g(tau0 + tau_a)) * self.coverage_fraction)

    def optical_depth_for_forcing(self, planet: PlanetState, delta_f_wm2: float) -> float:
        """Inverse of :meth:`forcing_for_optical_depth`; raises if at/above the saturation forcing."""
        if delta_f_wm2 < 0:
            raise ValueError("delta_f_wm2 must be >= 0.")
        f_max = self.max_forcing_wm2(planet)
        if delta_f_wm2 >= f_max:
            raise ValueError(
                f"Forcing {delta_f_wm2:.1f} W/m2 is at or above the saturation forcing {f_max:.1f} W/m2."
            )
        ts = planet.mean_temperature_k
        tau0 = self._tau_now(planet)
        g_new = _g(tau0) - delta_f_wm2 / (self.coverage_fraction * SIGMA_SB * ts**4)
        return float(4.0 / (3.0 * g_new) - 2.0 / 3.0 - tau0)

    def _required_delta_tau(self, planet: PlanetState, delta_t_k: float) -> float:
        """Added IR optical depth for ``+delta_t_k`` at the surface (Eddington mapping, Eq. 33)."""
        rb = RadiativeBalance(planet)
        ts = planet.mean_temperature_k
        return rb.required_ir_optical_depth(ts + delta_t_k) - rb.required_ir_optical_depth(ts)

    def _delta_t_for_delta_tau(self, planet: PlanetState, delta_tau: float) -> float:
        rb = RadiativeBalance(planet)
        ts = planet.mean_temperature_k
        tau_now = rb.required_ir_optical_depth(ts)
        return rb.eddington_surface_temperature(tau_now + delta_tau) - ts

    def _aerosol_optical_depth_for_delta_t(self, planet: PlanetState, delta_t_k: float) -> float:
        """Aerosol IR optical depth that raises the surface temperature by ``delta_t_k``."""
        return max(self._required_delta_tau(planet, delta_t_k), 0.0)

    def _mass_for_optical_depth(self, planet: PlanetState, tau_a: float) -> float:
        """Particle mass (kg) that carries optical depth ``tau_a`` over the covered area."""
        column = tau_a / self.mass_absorption_coefficient_m2_kg
        return float(column * planet.surface_area_m2 * self.coverage_fraction)

    # ------------------------------------------------------------------ compute
    def compute(
        self,
        planet: PlanetState,
        target_delta_t_k: float | None = None,
        target_delta_f_wm2: float | None = None,
        deployment_years: float = 10.0,
    ) -> MechanismResult:
        if not self.is_applicable(planet):
            raise ValueError(
                f"NanoparticleAerosol is not applicable to {planet.body_name}: "
                f"surface pressure {planet.surface_pressure_pa:.1f} Pa is too low."
            )
        warnings = [
            ("kappa is an inferred/assumed optical property, not a measurement; it dominates "
            "the uncertainty of mass and injection rate."),
            ("Microphysics, lofting/dispersal, solar-budget effects and water-cycle feedbacks "
            "are not modelled (Richardson et al. 2025); residence time is a parameter."),
            "Surface warming uses the grey Eddington mapping (+/-30-50 % on tau_IR).",
        ]

        if self.target_optical_depth is not None:
            tau_a = self.target_optical_depth
        elif target_delta_t_k is not None:
            tau_a = self._aerosol_optical_depth_for_delta_t(planet, target_delta_t_k)
        elif target_delta_f_wm2 is not None:
            try:
                tau_a = self.optical_depth_for_forcing(planet, target_delta_f_wm2)
            except ValueError as exc:
                tau_a = _TAU_CAP
                warnings.append(f"UNREACHABLE TARGET: {exc} Result is capped at tau = {_TAU_CAP}.")
        else:
            raise ValueError("Provide target_delta_t_k, target_delta_f_wm2 or target_optical_depth.")

        actual_delta_f = self.forcing_for_optical_depth(planet, tau_a)
        delta_t = self._delta_t_for_delta_tau(planet, tau_a * self.coverage_fraction)
        if target_delta_t_k is not None and self.target_optical_depth is None:
            # By construction the mapping reproduces the requested value (global coverage).
            delta_t = target_delta_t_k if self.coverage_fraction == 1.0 else delta_t

        total_mass_kg = self._mass_for_optical_depth(planet, tau_a)
        residence_yr = self.residence_time_years
        replenishment_kg_s = total_mass_kg / (residence_yr * YEAR_S)   # Eq. (53)
        throughput_kg_s = total_mass_kg / (deployment_years * YEAR_S)
        power_w = replenishment_kg_s * self.production_energy_j_per_kg

        isru = self._params["isru_efficiency"]
        comp = planet.regolith_composition
        isru_fraction = isru
        if comp is not None:
            if self.material in (AerosolMaterial.IRON, AerosolMaterial.MAGNETITE):
                fe = comp.get("Fe2O3", 0.0) + comp.get("FeO", 0.0)
                isru_fraction = min(isru * (fe / 0.18), 1.0)  # normalised to the Mars baseline
            elif self.material in (AerosolMaterial.ALUMINIUM, AerosolMaterial.ALUMINA,
                                   AerosolMaterial.NANOROD, AerosolMaterial.NANOROD_TURYSHEV):
                isru_fraction = min(isru * (comp.get("Al2O3", 0.0) / 0.093), 1.0)

        column_mg_m2 = tau_a / self.mass_absorption_coefficient_m2_kg * 1e6
        notes = (
            f"Aerosol IR optical depth tau_a={tau_a:.3f}; column {column_mg_m2:.2f} mg/m2 "
            f"(kappa = {self.mass_absorption_coefficient_m2_kg:.3g} m2/kg). "
            f"Residence time {residence_yr*365.25:.0f} d: replenishment {replenishment_kg_s:.2e} kg/s "
            f"({replenishment_kg_s*YEAR_S:.2e} kg/yr) to hold the column; maintenance-dominated "
            f"(Lambda = t_build/tau_p = {deployment_years/residence_yr:.1e})."
        )
        return MechanismResult(
            mechanism_name=self.name,
            total_mass_kg=total_mass_kg,
            power_w=power_w,
            throughput_kg_s=throughput_kg_s,
            delta_forcing_wm2=actual_delta_f,
            delta_temperature_k=delta_t,
            deployment_time_years=deployment_years,
            effect_duration_years=residence_yr,
            risk_index=0.35,
            isru_fraction=isru_fraction,
            notes=notes,
            area_fraction=self.coverage_fraction,
            warnings=warnings,
            model_class="reference_scaling",
            source_references=[
                "Ansari et al. (2024). Science Advances 10, eadn4650.",
                "Richardson et al. (2025). arXiv:2504.01455.",
                "Turyshev (2026). arXiv:2603.00402, Sec. VI.C, Eqs. (32)-(33), (52)-(53).",
            ],
        )


# Canonical alias for compatibility
MetalNanorodAerosols = NanoparticleAerosol
