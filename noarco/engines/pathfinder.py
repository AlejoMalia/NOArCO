"""
noarco.engines.pathfinder — PathEngine
======================================
Requirement-driven pathway synthesis. Every number is computed from the target
state; nothing is scripted. The engine

1. derives the endpoint requirements (mass, O2, minimum energy, forcing / optical
   depth, throughput, power) from Turyshev (2026) via :mod:`noarco.feasibility`;
2. allocates the thermal, pressure and oxygen requirements to mechanisms in a fixed,
   documented order (endogenous CO2 -> IR-active aerosols -> mirrors; import for any
   residual gas; electrolysis for O2), each with its own validity warnings;
3. reports the conjunctive feasibility number ``Pi_min`` (Eq. 11) against the
   availabilities supplied, with the binding constraint named.

The result is a *requirements-and-allocation* analysis (lower bounds), not a climate
prediction. ``overall_feasibility = min(1, Pi_min)`` is **not a probability**.

Applies the TRIADA protocol (Alejo Malia):
- T1 (INVENTORY): in-situ reservoirs bound what can be mobilised.
- T2 (MATHEMATICS): closed-form balances (hydrostatic mass, Stefan-Boltzmann, Gibbs work).
- T3 (MATE PROJECTION): infeasibility is certified as soon as one constraint fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from noarco.core.constants import CO2_SUBLIMATION_J_KG, YEAR_S, reversible_o2_energy_j_per_kg
from noarco.core.desired_state import DesiredState
from noarco.core.planet_state import PlanetState
from noarco.feasibility import assess, endpoint_requirements
from noarco.mechanisms.aerogel import SilicaAerogel
from noarco.mechanisms.aerosols import MetalNanorodAerosols
from noarco.mechanisms.base import Mechanism
from noarco.mechanisms.co2_mobilization import _MARS_CO2_INVENTORY, CO2Mobilization
from noarco.mechanisms.electrolysis import WaterElectrolysis
from noarco.mechanisms.greenhouse_gas import SuperGreenhouseGas
from noarco.mechanisms.mirrors import OrbitalMirror

#: Electrical efficiency of water electrolysis (reversible Gibbs work / electrical input).
ELECTROLYSIS_EFFICIENCY = 0.70


@dataclass
class PathwayStep:
    """An individual intervention step within an optimized terraforming pathway."""
    phase: int
    mechanism_name: str
    target_metric: str  # "warming", "pressure", "oxygen", "regional_paraterraforming"
    dose: float
    total_mass_kg: float
    energy_joules: float
    time_years: float
    risk_index: float
    isru_fraction: float
    soil_resources_utilized: dict[str, float]
    justification: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class OptimizedPathway:
    """Complete optimized sequence of actions from current state to desired state."""
    body_name: str
    initial_temperature_k: float
    target_temperature_k: float
    initial_pressure_pa: float
    target_pressure_pa: float
    target_o2_fraction: float
    steps: list[PathwayStep]
    total_energy_joules: float
    total_energy_kwh: float
    total_time_years: float
    overall_feasibility: float
    critical_path_summary: str
    pi_min: float | None = None
    binding_constraint: str | None = None
    warnings: list[str] = field(default_factory=list)


class PathEngine:
    """Requirement-driven pathway synthesis for planetary transitions."""

    @classmethod
    def get_standard_mechanisms(cls) -> dict[str, Mechanism]:
        """Return the library of mechanisms (each carries its own validity warnings)."""
        return {
            "nanorods": MetalNanorodAerosols(),
            "pfc_gases": SuperGreenhouseGas(),
            "aerogel": SilicaAerogel(),
            "mirrors": OrbitalMirror(),
            "co2_sublimation": CO2Mobilization(),
            "electrolysis": WaterElectrolysis(),
        }

    @staticmethod
    def _accessible_co2_kg(planet: PlanetState) -> float:
        """Accessible endogenous CO2: the published ~20 mbar reference on Mars, else the dataset value."""
        polar = planet.co2_ice_kg or 0.0
        if planet.body_name == "Mars":
            return max(polar, _MARS_CO2_INVENTORY["accessible_reference_kg"])
        return polar

    @classmethod
    def optimize_pathway(
        cls,
        current_planet: PlanetState,
        desired: DesiredState,
        available_power_w: float = 1.0e10,
        build_time_years: float = 1000.0,
        import_delta_v_km_s: float = 5.0,
    ) -> OptimizedPathway:
        """Allocate the requirements of ``desired`` to mechanisms and assess feasibility.

        Parameters
        ----------
        available_power_w : float
            Continuous power available to the programme (W). Step durations are
            ``energy / power``.
        build_time_years : float
            Build time used to express throughput and power requirements (Eq. 97-98).
        import_delta_v_km_s : float
            Delivery speed for the kinetic-energy benchmark of imported gas (Eq. 7).
        """
        if available_power_w <= 0:
            raise ValueError("available_power_w must be > 0.")
        planet = current_planet
        req = endpoint_requirements(planet, desired, build_time_years)
        warnings: list[str] = list(req.notes)
        steps: list[PathwayStep] = []
        phase = 1
        total_e_j = 0.0
        total_time = 0.0

        def add_step(step: PathwayStep) -> None:
            nonlocal phase, total_e_j, total_time
            steps.append(step)
            phase += 1
            total_e_j += step.energy_joules
            total_time += step.time_years

        ts_now = planet.mean_temperature_k
        ts_tgt = desired.target_mean_temperature_k
        d_t_needed = 0.0 if (ts_tgt is None or desired.enclosed) else max(ts_tgt - ts_now, 0.0)
        current = planet

        # --- Pressure: endogenous CO2 first (it also warms) -------------------------------
        imported_mass = 0.0
        if req.delta_pressure_pa > 0:
            accessible = cls._accessible_co2_kg(planet)
            mobilised = min(req.atmosphere_mass_kg, accessible)
            if mobilised > 0:
                used = mobilised
                co2 = CO2Mobilization(co2_inventory_kg=used)
                res = co2.compute(planet, target_delta_t_k=max(d_t_needed, 1.0))
                e_j = used * CO2_SUBLIMATION_J_KG / co2.heat_source_efficiency
                add_step(PathwayStep(
                    phase=phase, mechanism_name="CO2Mobilization (endogenous sublimation/outgassing)",
                    target_metric="pressure", dose=used * planet.gravity_ms2 / planet.surface_area_m2,
                    total_mass_kg=used, energy_joules=e_j, time_years=e_j / (available_power_w * YEAR_S),
                    risk_index=res.risk_index, isru_fraction=1.0,
                    soil_resources_utilized={"CO2_ice": used},
                    justification=(
                        f"Release {used:.2e} kg of CO2 (accessible reference {accessible:.2e} kg) for "
                        f"+{used * planet.gravity_ms2 / planet.surface_area_m2:.0f} Pa and "
                        f"+{res.delta_temperature_k:.1f} K."),
                    warnings=list(res.warnings),
                ))
                d_t_needed = max(d_t_needed - res.delta_temperature_k, 0.0)
                current = current.model_copy(update={
                    "surface_pressure_pa": planet.surface_pressure_pa + used * planet.gravity_ms2 / planet.surface_area_m2,
                    "mean_temperature_k": ts_now + res.delta_temperature_k,
                })
            imported_mass = max(req.atmosphere_mass_kg - mobilised, 0.0)
            if imported_mass > 0:
                e_imp = 0.5 * imported_mass * (import_delta_v_km_s * 1e3) ** 2
                add_step(PathwayStep(
                    phase=phase, mechanism_name="ExogenousGasImport (required)", target_metric="pressure",
                    dose=imported_mass * planet.gravity_ms2 / planet.surface_area_m2,
                    total_mass_kg=imported_mass, energy_joules=e_imp,
                    time_years=e_imp / (available_power_w * YEAR_S), risk_index=0.6, isru_fraction=0.0,
                    soil_resources_utilized={},
                    justification=(
                        f"Endogenous CO2 leaves a gap of {imported_mass:.2e} kg; importing it at "
                        f"{import_delta_v_km_s:.1f} km/s costs at least {e_imp:.2e} J (kinetic benchmark, "
                        "Turyshev 2026 Eq. 7; gravity assists and aerocapture could lower it)."),
                    warnings=[("Kinetic-energy benchmark only: capture, processing and delivery "
                               "logistics are not modelled.")],
                ))
                warnings.append(
                    f"Gas inventory shortfall of {imported_mass:.2e} kg: not coverable by endogenous CO2."
                )

        # --- Warming of the remainder ----------------------------------------------------
        if d_t_needed > 0.5:
            aero = MetalNanorodAerosols()
            tau_req = aero._aerosol_optical_depth_for_delta_t(current, d_t_needed)
            if tau_req <= 5.0:
                res_a = aero.compute(current, target_delta_t_k=d_t_needed, deployment_years=5.0)
                e_j = res_a.power_w * 5.0 * YEAR_S
                add_step(PathwayStep(
                    phase=phase, mechanism_name="IR-active aerosols (engineered nanorods)",
                    target_metric="warming", dose=d_t_needed, total_mass_kg=res_a.total_mass_kg,
                    energy_joules=e_j, time_years=5.0, risk_index=res_a.risk_index,
                    isru_fraction=res_a.isru_fraction,
                    soil_resources_utilized={"Al": res_a.total_mass_kg * 0.5, "Fe": res_a.total_mass_kg * 0.5},
                    justification=(
                        f"Add tau_IR = {tau_req:.2f} for +{d_t_needed:.1f} K with "
                        f"{res_a.total_mass_kg:.2e} kg of particles; maintenance-dominated "
                        f"(residence {res_a.effect_duration_years * 365.25:.0f} d)."),
                    warnings=list(res_a.warnings),
                ))
            else:
                mir = OrbitalMirror()
                res_m = mir.compute(current, target_delta_t_k=d_t_needed, deployment_years=50.0)
                e_j = res_m.power_w * 50.0 * YEAR_S
                add_step(PathwayStep(
                    phase=phase, mechanism_name="OrbitalMirror (global)", target_metric="warming",
                    dose=d_t_needed, total_mass_kg=res_m.total_mass_kg, energy_joules=e_j,
                    time_years=50.0, risk_index=res_m.risk_index, isru_fraction=res_m.isru_fraction,
                    soil_resources_utilized={}, warnings=list(res_m.warnings),
                    justification=(
                        f"Aerosols cannot supply tau_IR = {tau_req:.1f}; direct forcing of "
                        f"{res_m.delta_forcing_wm2:.0f} W/m2 needs {res_m.total_mass_kg:.2e} kg of reflector."),
                ))
                warnings.append("Aerosol route unreachable (tau_IR above cap); mirror route used.")

        # --- Oxygen ------------------------------------------------------------------------
        if req.o2_mass_kg > 0:
            e_o2 = req.o2_mass_kg * reversible_o2_energy_j_per_kg() / ELECTROLYSIS_EFFICIENCY
            add_step(PathwayStep(
                phase=phase, mechanism_name="Oxygenation (water electrolysis)", target_metric="oxygen",
                dose=req.o2_mass_kg, total_mass_kg=req.o2_mass_kg, energy_joules=e_o2,
                time_years=e_o2 / (available_power_w * YEAR_S), risk_index=0.3, isru_fraction=1.0,
                soil_resources_utilized={"H2O": req.o2_mass_kg * 1.1248},
                justification=(
                    f"{req.o2_mass_kg:.2e} kg of O2 from water: reversible minimum "
                    f"{req.o2_min_energy_j:.2e} J, {e_o2:.2e} J at efficiency {ELECTROLYSIS_EFFICIENCY:.2f}."),
                warnings=[("O2 sinks (oxidation of FeO-bearing regolith) and losses are not included "
                           "(Turyshev 2026, Eq. 75-76).")],
            ))

        pi = assess(
            req,
            available_inventory_kg=cls._accessible_co2_kg(planet) if req.atmosphere_mass_kg > 0 else None,
            available_power_w=available_power_w if req.mean_power_w > 0 else None,
        )
        pi_min = pi.pi_min
        overall = 1.0 if pi_min is None else max(0.0, min(1.0, pi_min))
        binding = pi.binding_constraint
        if pi.feasible is False:
            warnings.append(
                f"INFEASIBLE with the supplied availabilities: Pi_min = {pi_min:.3g} "
                f"(binding constraint: {binding})."
            )
        summary = (
            f"{len(steps)} allocation steps for {planet.body_name}. "
            f"Requirements: M = {req.atmosphere_mass_kg:.2e} kg, E_O2,min = {req.o2_min_energy_j:.2e} J, "
            f"dF_TOA = {req.delta_forcing_toa_wm2:.0f} W/m2, d_tau_IR = {req.delta_tau_ir:.2f}. "
            f"Total energy {total_e_j:.2e} J; {total_time:.1f} yr at {available_power_w/1e9:.1f} GW. "
            + (f"Pi_min = {pi_min:.3g} ({binding})." if pi_min is not None else "No availability bound applies.")
        )
        target_p = desired.min_surface_pressure_pa if desired.min_surface_pressure_pa is not None else planet.surface_pressure_pa
        target_o2 = (desired.min_o2_partial_pressure_pa / target_p) if (desired.min_o2_partial_pressure_pa and target_p > 0) else 0.0
        return OptimizedPathway(
            body_name=planet.body_name,
            initial_temperature_k=planet.mean_temperature_k,
            target_temperature_k=ts_tgt if ts_tgt is not None else planet.mean_temperature_k,
            initial_pressure_pa=planet.surface_pressure_pa,
            target_pressure_pa=target_p,
            target_o2_fraction=target_o2,
            steps=steps,
            total_energy_joules=total_e_j,
            total_energy_kwh=total_e_j / 3.6e6,
            total_time_years=total_time,
            overall_feasibility=overall,
            critical_path_summary=summary,
            pi_min=pi_min,
            binding_constraint=binding,
            warnings=warnings,
        )
