"""
noarco.inventory.inventory — InventoryEngine
============================================
Comprehensive resource inventory coupling Atmosphere (AIR) and Soil (SOIL).

The ``feasibility_score`` returned here is a **heuristic viability index**: a product of
fixed per-regime factors (lithosphere, retention, volatiles, energetics) with some
body-specific rules keyed on the body name. It is useful for ranking regimes quickly but is
not a probability and it is *not* the physical feasibility analysis; use
:mod:`noarco.feasibility` (Turyshev 2026 rubric: ``Pi_min``) for that.

Calculates:
1. Volatiles needed (O2, N2, CO2, H2O)
2. In-situ availability vs deficits (ISRU gap analysis)
3. Soil extraction vs Atmospheric extraction pathways
4. Sourcing of buffer gases (N2, Ar)
5. Comprehensive energy and mass budgeting
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from noarco.analyzers.atmosphere import AtmosphereEngine, VolumetricRequirement
from noarco.analyzers.soil import OxygenExtractionPlan, SoilEngine
from noarco.core.constants import G_NEWTON, MOLAR_MASS, R_GAS, YEAR_S
from noarco.core.planet_state import PlanetState


@dataclass
class ComprehensiveInventoryReport:
    """Integrated inventory evaluation for a given planetary volume or mission scale."""
    body_name: str
    volume_m3: float
    mode: str
    target_pressure_pa: float
    target_temperature_k: float
    total_gas_mass_kg: float
    o2_needed_kg: float
    n2_needed_kg: float
    o2_plan: OxygenExtractionPlan
    volumetric_req: VolumetricRequirement
    n2_source: str
    n2_deficit_kg: float
    total_energy_joules: float
    total_energy_kwh: float
    time_years_at_power: dict[str, float]  # Time in years at different power ratings
    feasibility_score: float  # 0.0-1.0 HEURISTIC viability index (see ``viability_basis``)
    critical_bottleneck: str | None = None
    notes: str = ""
    viability_basis: str = (
        "heuristic: product of fixed per-regime factors keyed on body type; not a probability "
        "and not derived from the conjunctive rubric of noarco.feasibility"
    )


class InventoryEngine:
    """Engine that calculates and budgets mass and energy resources for terraforming."""

    @classmethod
    def evaluate(
        cls,
        planet: PlanetState,
        volume_m3: float,
        target_pa: float = 101_325.0,
        target_temp_k: float = 293.15,
        target_o2_frac: float = 0.21,
        power_levels_w: dict[str, float] | None = None,
    ) -> ComprehensiveInventoryReport:
        """Run full AIR + SOIL coupled resource evaluation for a target volume."""
        # 1. Atmospheric volumetric requirement
        vol_req = AtmosphereEngine.compute_volume_conditioning(
            planet=planet,
            volume_m3=volume_m3,
            target_pa=target_pa,
            target_temp_k=target_temp_k,
            target_o2_fraction=target_o2_frac,
        )

        # 2. Regolith / Soil oxygen extraction plan
        o2_plan = SoilEngine.compute_oxygen_extraction(
            planet=planet,
            o2_target_kg=vol_req.o2_mass_kg,
        )

        # 3. Buffer gas (Nitrogen) source analysis
        # Atmospheric N2 partial pressure
        n2_needed = vol_req.n2_mass_kg
        n2_deficit = 0.0

        if planet.body_name in ("Jupiter", "Saturn", "Uranus", "Neptune"):
            n2_source = "Ammonia (NH3) extraction from deep cloud layers or external import"
        elif planet.body_name == "Venus":
            # Venus has ~3.2 bar of N2! (3.5% of 92 bar = 320 kPa) -> Superabundant N2
            n2_source = "Superabundant in-situ atmosphere (3.2 bar of N2 directly available)"
        elif planet.body_name == "Earth":
            n2_source = "Native terrestrial atmosphere (78% N2)"
        elif planet.body_name == "Mars":
            # Mars has 2.7% N2 at 610 Pa = 16.5 Pa. Total mass of N2 in Mars atm ~ 6e14 kg
            if n2_needed < 1e12:
                n2_source = "Cryogenic separation / PSA of Martian atmospheric N2 (16.5 Pa N2)"
            else:
                n2_deficit = max(0.0, n2_needed - 6e14)
                n2_source = "Requires cometary/asteroidal nitrogen import for planetary scales"
        elif planet.body_name == "Pluto":
            n2_source = "In-situ pure N2-ice bedrock (Sputnik Planitia)"
        else:
            # Moon, Mercury
            n2_deficit = n2_needed
            n2_source = "Total deficit: requires N2 import or substitution by a noble buffer gas"

        # 4. Total coupled energy = Conditioning (thermal + compression) + O2 extraction
        total_energy_j = vol_req.total_conditioning_energy_joules + o2_plan.extraction_energy_joules

        # 5. Timescales at various power levels (e.g., 100 kW, 10 MW, 1 GW, 100 GW)
        # Power ratings in Watts
        if power_levels_w is None:
            # Scaled power presets based on scale
            if volume_m3 <= 100.0:
                power_levels_w = {"10 kW": 1e4, "100 kW": 1e5, "1 MW": 1e6}
            elif volume_m3 <= 1e7:
                power_levels_w = {"1 MW": 1e6, "10 MW": 1e7, "100 MW": 1e8}
            elif volume_m3 <= 1e13:
                power_levels_w = {"100 MW": 1e8, "1 GW": 1e9, "10 GW": 1e10}
            else:
                power_levels_w = {"1 GW": 1e9, "100 GW": 1e11, "10 TW": 1e13}

        # Conversion efficiency ~ 70%
        system_efficiency = 0.70
        seconds_per_year = YEAR_S

        times_years: dict[str, float] = {}
        for p_label, p_val in power_levels_w.items():
            if p_val > 0:
                t_sec = total_energy_j / (p_val * system_efficiency)
                times_years[p_label] = t_sec / seconds_per_year
            else:
                times_years[p_label] = float("inf")

        # 6. Multi-Factor Transparent Viability Scoring (0.0 to 1.0)
        # Factors:
        # - f_lithosphere: Surface solid support (0.0 for gas giants open, 0.75 for aerostats)
        # - f_retention: Hydrodynamic Jeans escape stability (v_esc vs 6 * v_thermal at 293 K)
        # - f_volatiles: Buffer gas (N2) & H2O in-situ availability vs import requirements
        # - f_energetics: Thermodynamic feasibility of mass removal / cooling (Venus bottleneck)
        
        reasons: list[str] = []
        is_open = (vol_req.mode == "open_atmosphere")

        # 1. Lithosphere factor
        if is_open and planet.body_name in ("Jupiter", "Saturn", "Uranus", "Neptune"):
            f_lithosphere = 0.0
            reasons.append("No lithosphere: gas giants lack a solid surface for an open atmosphere (0%).")
        elif not is_open and planet.body_name in ("Jupiter", "Saturn", "Uranus", "Neptune"):
            f_lithosphere = 0.75
            reasons.append("Floating aerostat regime at the 1 bar layer (requires buoyancy compensation vs H2/He).")
        else:
            f_lithosphere = 1.0

        # 2. Retention factor (Jeans escape)
        if not is_open:
            f_retention = 1.0
        else:
            v_th_target = float(np.sqrt(3.0 * R_GAS * target_temp_k / MOLAR_MASS["N2"]))
            v_esc = planet.radius_m * np.sqrt(2.0 * G_NEWTON * planet.mass_kg / (planet.radius_m**3)) if planet.radius_m > 0 else 0.0
            lambda_esc = v_esc / v_th_target if v_th_target > 0 else 10.0

            if lambda_esc < 3.0:
                # e.g. Pluto (v_esc = 1210 m/s vs v_th = 511 m/s -> lambda = 2.37)
                f_retention = 0.15
                reasons.append(
                    f"Thermal hydrodynamic escape (v_esc={v_esc:.0f} m/s vs 6*v_th={6*v_th_target:.0f} m/s): "
                    f"At 293 K the atmosphere dissipates into interplanetary space within decades without a sealed dome."
                )
            elif planet.body_name == "Mercury":
                # Mercury: lambda ~ 8.3, but intense solar wind sputtering at 0.387 AU without magnetic dynamo
                f_retention = 0.30
                reasons.append(
                    "Extreme solar-wind erosion at 0.387 AU and no dipolar magnetic field (loss rate ~10³ kg/s; requires an L1 shield)."
                )
            elif planet.body_name == "Moon":
                f_retention = 0.20
                reasons.append("Lunar gravity is insufficient to retain an open atmosphere at 293 K.")
            else:
                f_retention = 1.0

        # 3. Volatiles factor (Buffer gas N2 & water reserves)
        if not is_open:
            f_volatiles = 1.0
        else:
            if n2_deficit > 1.0e17:
                # Mars: massive nitrogen deficit (~3e18 kg needed vs ~6e14 kg in-situ)
                f_volatiles = 0.45
                reasons.append(f"Critical buffer-nitrogen deficit ({n2_deficit:.2e} kg); requires massive cometary import.")
            elif planet.body_name in ("Mercury", "Moon"):
                f_volatiles = 0.40
                reasons.append("Total absence of nitrogen in the regolith (no in-situ source).")
            else:
                f_volatiles = 1.0

        # 4. Energetics factor (Venus thermal dump / sequestration)
        if is_open and planet.body_name == "Venus":
            f_energetics = 0.10
            reasons.append(
                "Colossal thermodynamic bottleneck: requires removing 4.72e20 kg of CO2 (>2e28 J) "
                "and dissipating 2.4e26 J of heat; infeasible without Type II stellar megaengineering."
            )
        else:
            f_energetics = 1.0

        # Composite score
        if f_lithosphere == 0.0:
            feasibility = 0.0
        else:
            feasibility = max(0.05, min(1.0, f_lithosphere * f_retention * f_volatiles * f_energetics))

        if not is_open:
            feasibility = min(1.0, max(0.75, feasibility * 0.95))

        bottleneck_summary = " | ".join(reasons) if reasons else "Stable nominal conditions"

        return ComprehensiveInventoryReport(
            body_name=planet.body_name,
            volume_m3=volume_m3,
            mode=vol_req.mode,
            target_pressure_pa=target_pa,
            target_temperature_k=target_temp_k,
            total_gas_mass_kg=vol_req.total_gas_mass_kg,
            o2_needed_kg=vol_req.o2_mass_kg,
            n2_needed_kg=vol_req.n2_mass_kg,
            o2_plan=o2_plan,
            volumetric_req=vol_req,
            n2_source=n2_source,
            n2_deficit_kg=n2_deficit,
            total_energy_joules=total_energy_j,
            total_energy_kwh=total_energy_j / 3.6e6,
            time_years_at_power=times_years,
            feasibility_score=round(feasibility, 2),
            critical_bottleneck=bottleneck_summary,
            notes=vol_req.regime_description,
        )
