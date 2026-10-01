"""
noarco.extratools.agro — NOArCO-Agro
===================================
Extraterrestrial Bioremediation, Detoxification and Agriculture Designer.
Converts sterile, toxic regolith (with perchlorates) into fertile arable soil
for agriculture in pressurized habitats and greenhouses.

Framework: NOArCO
Framework and TRIADA protocol author: Alejo Malia
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SoilConditioningSpec:
    """Design parameters for the agronomic conversion of regolith."""
    greenhouse_area_m2: float = 1000.0          # 1000 m2 of farmland
    soil_depth_m: float = 0.40                  # 40 cm arable layer depth
    perchlorate_fraction: float = 0.005         # 0.5% ClO4 in Martian regolith
    target_organic_matter_fraction: float = 0.03 # 3% mature organic matter
    target_moisture_fraction: float = 0.18      # 18% field-capacity moisture
    initial_nitrogen_ppm: float = 25.0
    target_nitrogen_ppm: float = 220.0          # Standard balanced NPK
    soil_bulk_density_kg_m3: float = 1500.0     # Bulk density of compacted regolith
    # Engineering assumptions (not measurements): replace with experimental values.
    nitrogen_fixation_g_m2_day: float = 0.05    # biological N2 fixation rate
    food_yield_kg_m2_year: float = 25.0         # optimistic controlled-environment yield
    detox_energy_mj_per_kg_o2: float = 6.0      # catalytic perchlorate decomposition
    crew_kcal_per_person_day: float = 2500.0    # NASA-STD-3001 order of magnitude for an active adult
    food_kcal_per_kg: float = 800.0             # fresh potato ~770 kcal/kg (USDA); the crop is an assumption

    def __post_init__(self) -> None:
        if not (self.greenhouse_area_m2 > 0 and self.soil_depth_m > 0 and self.soil_bulk_density_kg_m3 > 0):
            raise ValueError("area, depth and bulk density must be > 0")
        for n in ("perchlorate_fraction", "target_organic_matter_fraction", "target_moisture_fraction"):
            if not 0.0 <= getattr(self, n) <= 1.0:
                raise ValueError(f"{n} must be in [0, 1]")
        if not (self.crew_kcal_per_person_day > 0 and self.food_kcal_per_kg > 0):
            raise ValueError("crew_kcal_per_person_day and food_kcal_per_kg must be > 0")
        if not (self.nitrogen_fixation_g_m2_day > 0 and self.food_yield_kg_m2_year >= 0 and self.detox_energy_mj_per_kg_o2 >= 0):
            raise ValueError("fixation rate must be > 0; yield and energy >= 0")


@dataclass
class AgroReadinessReport:
    """Technical report on soil conditioning and food production."""
    total_soil_volume_m3: float
    total_soil_mass_kg: float
    perchlorate_mass_to_remove_kg: float
    oxygen_liberated_from_detox_kg: float
    detoxification_energy_kwh: float
    water_required_hydration_kg: float
    nitrogen_fertilizer_needed_kg: float
    organic_biomass_amendment_kg: float
    cyanobacteria_conditioning_days: float      # Biological inoculation time
    annual_food_yield_kg: float                 # Estimated caloric/vegetable yield
    recommendations: list[str] = field(default_factory=list)
    crew_supported: float = 0.0                 # people whose energy need the annual yield covers (calorie basis only)
    area_per_person_m2: float = 0.0             # farmland per person on that basis
    model_class: str = "engineering_estimate"


class AstroAgroPlanner:
    """
    NOArCO space agroecological planner.
    Applies thermodynamic and stoichiometric principles to transform
    inorganic regolith into a biologically active substrate.
    """

    @staticmethod
    def plan_soil_conversion(spec: SoilConditioningSpec) -> AgroReadinessReport:
        # 1. Total soil volume and mass
        volume_m3 = spec.greenhouse_area_m2 * spec.soil_depth_m
        soil_mass_kg = volume_m3 * spec.soil_bulk_density_kg_m3

        # 2. Perchlorate detoxification (ClO4- -> Cl- + 2 O2)
        # ClO4 (99.45 g/mol) -> 2 O2 (64 g/mol)
        perchlorate_mass_kg = soil_mass_kg * spec.perchlorate_fraction
        oxygen_liberated_kg = perchlorate_mass_kg * (64.0 / 99.45)

        # Thermal catalytic decomposition energy (~6 MJ / kg O2)
        detox_energy_mj = oxygen_liberated_kg * spec.detox_energy_mj_per_kg_o2
        detox_energy_kwh = detox_energy_mj / 3.6

        # 3. Water required for saturation at field capacity
        water_hydration_kg = soil_mass_kg * spec.target_moisture_fraction

        # 4. Assimilable nitrogen deficit
        delta_n_ppm = max(0.0, spec.target_nitrogen_ppm - spec.initial_nitrogen_ppm)
        nitrogen_needed_kg = (soil_mass_kg * delta_n_ppm) / 1.0e6

        # 5. Organic biomass amendment (humus / biocompost from algae reactors)
        organic_biomass_kg = soil_mass_kg * spec.target_organic_matter_fraction

        # 6. Soil maturation time via N2-fixing cyanobacteria inoculation
        # Assumed N2-fixation rate (engineering assumption, see SoilConditioningSpec)
        nitrogen_per_m2_needed = (nitrogen_needed_kg / spec.greenhouse_area_m2) * 1000.0 # g N / m2
        fixation_rate_g_m2_day = spec.nitrogen_fixation_g_m2_day
        conditioning_days = max(45.0, nitrogen_per_m2_needed / fixation_rate_g_m2_day)

        # 7. Annual food yield estimate (hydroponic crops / enriched substrate)
        # Yield is an input assumption (default 25 kg/m2/yr, optimistic controlled-environment potato)
        annual_food_yield_kg = spec.greenhouse_area_m2 * spec.food_yield_kg_m2_year

        recs: list[str] = [
            f"Phase 1 (Detoxification): The process will release {oxygen_liberated_kg:.1f} kg of usable O₂ for the habitat.",
            f"Phase 2 (Inoculation): Inoculate with nitrogen-fixing cyanobacteria for ~{conditioning_days:.0f} days.",
            f"Phase 3 (Planting): Estimated food support capacity of {annual_food_yield_kg/1000:.1f} tonnes/year.",
        ]

        food_kg_per_person_year = spec.crew_kcal_per_person_day * 365.25 / spec.food_kcal_per_kg
        crew_supported = annual_food_yield_kg / food_kg_per_person_year
        area_per_person = spec.greenhouse_area_m2 / crew_supported if crew_supported > 0 else float("inf")
        recs.append(f"Calorie basis only: {crew_supported:.1f} people supported ({area_per_person:.0f} m2 each); "
                    "protein, micronutrients, light and CO2 budgets are not modelled.")

        return AgroReadinessReport(
            crew_supported=crew_supported, area_per_person_m2=area_per_person,
            total_soil_volume_m3=volume_m3,
            total_soil_mass_kg=soil_mass_kg,
            perchlorate_mass_to_remove_kg=perchlorate_mass_kg,
            oxygen_liberated_from_detox_kg=oxygen_liberated_kg,
            detoxification_energy_kwh=detox_energy_kwh,
            water_required_hydration_kg=water_hydration_kg,
            nitrogen_fertilizer_needed_kg=nitrogen_needed_kg,
            organic_biomass_amendment_kg=organic_biomass_kg,
            cyanobacteria_conditioning_days=conditioning_days,
            annual_food_yield_kg=annual_food_yield_kg,
            recommendations=recs,
        )
