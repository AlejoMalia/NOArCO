"""
noarco.extratools.habitat — NOArCO-Habitat
=========================================
Closed Habitat and Life Support Dimensioner (ECLSS / Paraterraforming).
Computes gas inventory, crew metabolic consumption, continuous power,
ambient purge/back-pressure and thermal balances for domes, lava tubes and bases.

Framework: NOArCO
Framework and TRIADA protocol author: Alejo Malia
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from noarco.core.constants import MOLAR_MASS, R_GAS, reversible_o2_energy_j_per_kg
from noarco.core.planet_state import PlanetState

# Crew consumption/production per person-day: NASA Baseline Values and Assumptions Document
# (BVAD, NASA/TP-2015-218570); the human-system requirements themselves are in NASA-STD-3001.
O2_CONSUMPTION_KG_DAY_PER_PERSON = 0.84       # kg O2 / person / day
CO2_PRODUCTION_KG_DAY_PER_PERSON = 1.00        # kg CO2 / person / day
H2O_CONSUMPTION_KG_DAY_PER_PERSON = 2.50       # kg H2O (drinking + food) / day
METABOLIC_HEAT_WATTS_PER_PERSON = 120.0        # W of sensible heat dissipated


@dataclass
class HabitatSpecification:
    """Design specifications for an extraterrestrial pressurized habitat."""
    volume_m3: float
    crew_size: int
    mission_days: int = 365
    interior_pressure_pa: float = 101_325.0    # 1 standard atm (or 70.3 kPa for exploration habitats)
    interior_temperature_k: float = 293.15     # 20 °C
    o2_fraction: float = 0.21
    n2_fraction: float = 0.78
    co2_max_fraction: float = 0.005            # Safe CO2 limit
    eclss_loop_closure_o2: float = 0.90        # 90% O2 recovery via Sabatier/Bosch
    eclss_loop_closure_h2o: float = 0.95       # 95% water recycling (urine + condensate)
    # Engineering assumptions (no measured values): override them with mission data.
    envelope_u_value_cold_w_m2_k: float = 0.35  # multilayer insulation, heat loss to a cold exterior
    envelope_u_value_hot_w_m2_k: float = 0.50   # heat gain from a hot exterior
    cooling_cop: float = 1.5                    # coefficient of performance of active cooling
    leak_fraction_per_year: float = 0.005       # leakage, fraction of gas mass per year AT the reference surface/volume ratio
    envelope_area_m2: float | None = None       # actual envelope area; None = equivalent sphere of the volume
    electrolysis_efficiency: float = 0.70       # reversible Gibbs work / electrical input
    scrubber_w_per_person: float = 200.0        # CO2 scrubbing (absorption/regeneration)
    base_w_per_person: float = 500.0            # instrumentation, pumps, lighting
    base_w_per_m3: float = 1.0                  # volume-proportional base load
    # Pressure shell (thin-wall sphere, t = dP r / (2 sigma)); defaults: Al 6061-T6, yield 276 MPa, safety factor 2.
    hull_allowable_stress_pa: float = 1.38e8
    hull_density_kg_m3: float = 2700.0
    hull_min_thickness_m: float = 0.002         # manufacturing / micrometeoroid minimum

    def __post_init__(self) -> None:
        if not (self.volume_m3 > 0):
            raise ValueError(f"volume_m3 must be > 0, got {self.volume_m3!r}")
        if not (isinstance(self.crew_size, int) and self.crew_size >= 0):
            raise ValueError(f"crew_size must be an integer >= 0, got {self.crew_size!r}")
        if self.envelope_area_m2 is not None:
            min_area = (36.0 * math.pi) ** (1.0 / 3.0) * self.volume_m3 ** (2.0 / 3.0)   # sphere = minimum
            if self.envelope_area_m2 < 0.999 * min_area:
                raise ValueError("envelope_area_m2 is below the isoperimetric minimum for this volume")
        if not (self.mission_days > 0):
            raise ValueError(f"mission_days must be > 0, got {self.mission_days!r}")
        if not (self.interior_pressure_pa > 0 and self.interior_temperature_k > 0):
            raise ValueError("interior pressure and temperature must be > 0")
        for name in ("o2_fraction", "n2_fraction", "co2_max_fraction",
                     "eclss_loop_closure_o2", "eclss_loop_closure_h2o"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v!r}")
        if self.o2_fraction + self.n2_fraction > 1.0 + 1e-9:
            raise ValueError("o2_fraction + n2_fraction must not exceed 1.0")
        if not (0.0 < self.electrolysis_efficiency <= 1.0 and self.cooling_cop > 0):
            raise ValueError("electrolysis_efficiency must be in (0, 1] and cooling_cop > 0")
        if not (self.hull_allowable_stress_pa > 0 and self.hull_density_kg_m3 > 0 and self.hull_min_thickness_m >= 0):
            raise ValueError("hull stress and density must be > 0; minimum thickness >= 0")
        for name in ("envelope_u_value_cold_w_m2_k", "envelope_u_value_hot_w_m2_k", "leak_fraction_per_year",
                     "scrubber_w_per_person", "base_w_per_person", "base_w_per_m3"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass
class HabitatReport:
    """Integrated habitat and ECLSS dimensioning results."""
    volume_m3: float
    crew_size: int
    total_gas_mass_kg: float
    o2_mass_kg: float
    n2_mass_kg: float
    daily_o2_consumption_kg: float
    net_daily_o2_makeup_kg: float
    daily_co2_scrubbed_kg: float
    daily_water_makeup_kg: float
    displacement_work_kwh: float               # Purge work against external pressure
    continuous_power_kw: float                 # Constant-operation electrical power
    hvac_cooling_power_kw: float               # Thermal control against the external environment
    leak_rate_annual_kg: float                 # annual makeup for joint leakage
    recommendations: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    # --- added: shell mechanics and mission ledger ---
    hull_wall_thickness_m: float = 0.0
    hull_mass_kg: float = 0.0
    hull_governed_by: str = "tension"               # tension | minimum_gauge | external_pressure_buckling
    mission_o2_makeup_kg: float = 0.0               # net O2 import over the mission (loop-closure aware) + leak O2
    mission_water_makeup_kg: float = 0.0
    mission_gas_leak_kg: float = 0.0                # gas lost through the envelope over the mission
    mission_energy_kwh: float = 0.0
    warnings: list[str] = field(default_factory=list)
    model_class: str = "engineering_estimate"


class HabitatDimensioner:
    """
    NOArCO space habitat dimensioning engine.
    Analytically and instantly determines the resources needed to
    guarantee human habitability on any planetary body.
    """

    @staticmethod
    def dimension(
        planet: PlanetState,
        spec: HabitatSpecification,
    ) -> HabitatReport:
        # 1. Initial gas mass to pressurize the volume (ideal gas, mixture molar mass from the fractions;
        #    the remainder beyond O2 + N2 is taken as argon).
        x_o2, x_n2 = spec.o2_fraction, spec.n2_fraction
        x_rest = max(1.0 - x_o2 - x_n2, 0.0)
        mu_mix = x_o2 * MOLAR_MASS["O2"] + x_n2 * MOLAR_MASS["N2"] + x_rest * MOLAR_MASS["Ar"]
        total_gas_mass = spec.interior_pressure_pa * spec.volume_m3 * mu_mix / (R_GAS * spec.interior_temperature_k)
        o2_mass = total_gas_mass * x_o2 * MOLAR_MASS["O2"] / mu_mix
        n2_mass = total_gas_mass * x_n2 * MOLAR_MASS["N2"] / mu_mix

        # 2. Daily metabolic consumption
        daily_o2_gross = spec.crew_size * O2_CONSUMPTION_KG_DAY_PER_PERSON
        net_daily_o2_makeup = daily_o2_gross * (1.0 - spec.eclss_loop_closure_o2)
        daily_co2_scrubbed = spec.crew_size * CO2_PRODUCTION_KG_DAY_PER_PERSON
        daily_h2o_gross = spec.crew_size * H2O_CONSUMPTION_KG_DAY_PER_PERSON
        net_daily_h2o_makeup = daily_h2o_gross * (1.0 - spec.eclss_loop_closure_h2o)

        # 3. Purge / back-pressure work against the external environment
        delta_p = max(0.0, planet.surface_pressure_pa - spec.interior_pressure_pa)
        displacement_work_j = delta_p * spec.volume_m3
        displacement_work_kwh = displacement_work_j / 3.6e6

        # 4. Required life-support power (ECLSS)
        # - All metabolic O2 is regenerated by water electrolysis (loop closure only decides how
        #   much *water* must be imported), so the electrolysis power is set by the gross O2 rate:
        #   reversible Gibbs work 14.8 MJ/kg divided by the electrolyser efficiency.
        o2_energy_j_per_kg = reversible_o2_energy_j_per_kg() / spec.electrolysis_efficiency
        eclss_power_kw = (daily_o2_gross * o2_energy_j_per_kg / 86400.0) / 1000.0
        scrubber_power_kw = (spec.crew_size * spec.scrubber_w_per_person) / 1000.0
        base_power_kw = (spec.crew_size * spec.base_w_per_person + spec.volume_m3 * spec.base_w_per_m3) / 1000.0

        # 5. Thermal control (HVAC): conduction through an equivalent-sphere envelope.
        delta_t = spec.interior_temperature_k - planet.mean_temperature_k
        crew_heat_kw = (spec.crew_size * METABOLIC_HEAT_WATTS_PER_PERSON) / 1000.0
        radius = (3.0 * spec.volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0)
        sphere_area_m2 = 4.0 * math.pi * (radius ** 2)
        area_m2 = spec.envelope_area_m2 if spec.envelope_area_m2 is not None else sphere_area_m2

        if delta_t > 0:
            # Cold exterior (Mars, Moon, Pluto): net heating after crew heat.
            thermal_loss_kw = (area_m2 * spec.envelope_u_value_cold_w_m2_k * delta_t) / 1000.0
            hvac_power_kw = max(0.0, thermal_loss_kw - crew_heat_kw)
        else:
            # Hot exterior (Venus, illuminated Mercury): active cooling of ingress + crew heat.
            heat_ingress_kw = (area_m2 * spec.envelope_u_value_hot_w_m2_k * abs(delta_t)) / 1000.0
            hvac_power_kw = (heat_ingress_kw + crew_heat_kw) / spec.cooling_cop

        total_continuous_power_kw = eclss_power_kw + scrubber_power_kw + base_power_kw + hvac_power_kw

        # 6. Structural leakage
        # Leakage goes through the envelope, so the fraction of gas lost per year scales with the
        # surface/volume ratio relative to the reference (a 2500 m3 sphere, A/V = 0.357 1/m).
        ref_av = 3.0 / ((3.0 * 2500.0 / (4.0 * math.pi)) ** (1.0 / 3.0))
        leak_fraction = spec.leak_fraction_per_year * (area_m2 / spec.volume_m3) / ref_av
        leak_rate_annual_kg = total_gas_mass * leak_fraction * (spec.mission_days / 365.25)

        # 7. Operational recommendations
        recs: list[str] = []
        if planet.surface_pressure_pa > 1.0e6:
            recs.append("Hyperbaric external environment: requires a double-ring structural hull to withstand crushing.")
        elif planet.surface_pressure_pa < 1000.0:
            recs.append("High-vacuum environment: positive pressurization; hull under pure tension. Prioritize regolith anchoring.")

        if abs(delta_t) > 100.0:
            recs.append(f"Severe thermal gradient ({abs(delta_t):.0f} K): requires a Silica Aerogel envelope or a 2 m regolith blanket.")

        # 8. Pressure shell: thin-walled sphere in tension, t = dP r / (2 sigma_allow); mass = A t rho.
        warnings: list[str] = []
        d_p_shell = spec.interior_pressure_pa - planet.surface_pressure_pa
        governed = "tension"
        if d_p_shell < 0:
            governed = "external_pressure_buckling"
            warnings.append("External pressure exceeds the interior: tension formula is not valid (buckling governs); "
                            "the thickness reported is a lower bound from |dP|.")
        t_hull = abs(d_p_shell) * radius / (2.0 * spec.hull_allowable_stress_pa)
        if t_hull < spec.hull_min_thickness_m:
            t_hull = spec.hull_min_thickness_m
            governed = "minimum_gauge" if d_p_shell >= 0 else governed
        hull_mass = area_m2 * t_hull * spec.hull_density_kg_m3

        # 9. Mission ledger: make-up over the mission, with the leak taken at its O2 fraction.
        mission_leak = leak_rate_annual_kg  # already scaled by mission_days/365.25
        mission_o2 = net_daily_o2_makeup * spec.mission_days + mission_leak * x_o2 * MOLAR_MASS["O2"] / mu_mix
        mission_h2o = net_daily_h2o_makeup * spec.mission_days
        mission_kwh = total_continuous_power_kw * 24.0 * spec.mission_days

        return HabitatReport(
            hull_wall_thickness_m=t_hull, hull_mass_kg=hull_mass, hull_governed_by=governed,
            mission_o2_makeup_kg=mission_o2, mission_water_makeup_kg=mission_h2o,
            mission_gas_leak_kg=mission_leak, mission_energy_kwh=mission_kwh, warnings=warnings,
            volume_m3=spec.volume_m3,
            crew_size=spec.crew_size,
            total_gas_mass_kg=total_gas_mass,
            o2_mass_kg=o2_mass,
            n2_mass_kg=n2_mass,
            daily_o2_consumption_kg=daily_o2_gross,
            net_daily_o2_makeup_kg=net_daily_o2_makeup,
            daily_co2_scrubbed_kg=daily_co2_scrubbed,
            daily_water_makeup_kg=net_daily_h2o_makeup,
            displacement_work_kwh=displacement_work_kwh,
            continuous_power_kw=total_continuous_power_kw,
            hvac_cooling_power_kw=hvac_power_kw,
            leak_rate_annual_kg=leak_rate_annual_kg,
            recommendations=recs,
            assumptions=[
                ("Thermal model is steady-state conduction through an equivalent sphere with the "
                "specification's U-values; solar/radiative loads, thermal mass and night cycles are not modelled."),
                ("Envelope U-values, cooling COP, leak fraction (scaled with surface/volume) and electrical loads are engineering "
                "assumptions (HabitatSpecification fields), not measured values."),
                "Crew rates follow NASA BVAD (O2 0.84, CO2 1.00 kg/person/day).",
            ],
        )
