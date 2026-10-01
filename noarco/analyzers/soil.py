"""
noarco.analyzers.soil — SoilEngine
==================================
Deep characterization and ISRU extraction modeling of planetary regolith and lithosphere.

Models:
1. Mineralogical & oxide assay (SiO2, Fe2O3, Al2O3, MgO, CaO, SO3, TiO2, perchlorates)
2. Elemental recovery stoichiometry (exact TRIADA T2 method)
3. Oxygen extraction pathways:
   - Molten Regolith Electrolysis (MRE) / Carbothermal reduction: 35-45 MJ/kg O2
   - Subsurface Water Ice Mining + Electrolysis: 20-25 MJ/kg O2
4. Structural metal co-products (Fe, Al, Si) for habitat construction and shielding
5. Perchlorate remediation and soil de-toxification
6. Identification of volatile ice reserves (H2O, CO2, N2)
"""

from __future__ import annotations

from dataclasses import dataclass

from noarco.core.planet_state import PlanetState
from noarco.data.elements import get_element_recovery, oxygen_in_oxides_fraction


@dataclass
class SoilAnalysis:
    """Comprehensive characterization of planetary regolith and lithosphere."""
    body_name: str
    has_solid_surface: bool
    regolith_density_kg_m3: float
    oxide_fractions: dict[str, float]
    elemental_mass_fractions: dict[str, float]
    h2o_ice_reserve_kg: float
    co2_ice_reserve_kg: float
    extractable_o2_from_regolith_pct: float  # e.g. ~42% by mass
    mre_specific_energy_mj_kg_o2: float       # ~40 MJ/kg O2
    h2o_electrolysis_energy_mj_kg_o2: float   # ~20 MJ/kg O2
    perchlorate_toxicity_present: bool
    structural_metal_potential: str          # Description of available Fe, Al, Si
    notes: str = ""


@dataclass
class OxygenExtractionPlan:
    """ISRU extraction plan to provide a requested mass of oxygen."""
    o2_target_kg: float
    primary_source: str  # "water_ice" | "molten_regolith_electrolysis" | "atmospheric_co2" | "none"
    regolith_mined_kg: float
    water_ice_consumed_kg: float
    metals_coproduced_kg: dict[str, float]
    extraction_energy_joules: float
    extraction_energy_kwh: float
    processing_method: str
    isru_feasible: bool
    limiting_factor: str | None = None
    notes: str = ""


class SoilEngine:
    """Engine for regolith mineralogy, volatile reserves, and ISRU extraction."""

    # Typical specific energies
    MRE_ENERGY_MJ_KG_O2 = 40.0         # Molten Regolith Electrolysis (high-temp smelting)
    H2O_MINING_ENERGY_MJ_KG = 3.0      # Sublimation / thermal mining of ice
    H2O_ELECTROLYSIS_MJ_KG_O2 = 21.2   # Water electrolysis: reversible 14.8 MJ/kg O2 (2 dG/M_O2) / 0.70 efficiency; 1.125 kg H2O per kg O2
    PERCHLORATE_WASH_MJ_KG = 1.5       # Detoxification energy per kg soil

    @classmethod
    def analyze(cls, planet: PlanetState) -> SoilAnalysis:
        """Perform complete 100% characterization of the planetary soil/crust."""
        is_gas_giant = planet.body_name in ("Jupiter", "Saturn", "Uranus", "Neptune")
        if is_gas_giant or planet.regolith_composition is None:
            return SoilAnalysis(
                body_name=planet.body_name,
                has_solid_surface=False,
                regolith_density_kg_m3=0.0,
                oxide_fractions={},
                elemental_mass_fractions={},
                h2o_ice_reserve_kg=0.0,
                co2_ice_reserve_kg=0.0,
                extractable_o2_from_regolith_pct=0.0,
                mre_specific_energy_mj_kg_o2=0.0,
                h2o_electrolysis_energy_mj_kg_o2=0.0,
                perchlorate_toxicity_present=False,
                structural_metal_potential="None (gaseous body with no accessible solid crust)",
                notes="Gas giant: lacks a solid lithosphere.",
            )

        regolith = planet.regolith_composition
        # Calculate elemental mass fractions from oxides
        elements_recovered = {}
        for elem in ("Si", "Fe", "Al", "Mg", "Ca", "S"):
            elements_recovered[elem] = get_element_recovery(regolith, elem)

        # Regolith bulk density
        density = 1500.0  # kg/m3 standard loose regolith (Mars/Moon)
        if planet.body_name == "Mercury":
            density = 1700.0
        elif planet.body_name == "Venus":
            density = 2800.0  # Dense basalt bedrock
        elif planet.body_name == "Pluto":
            density = 1000.0  # Porous water/nitrogen ice

        # Exact oxygen mass fraction bound in the oxide assay: sum_i w_i * f_O(oxide_i).
        o2_bound_pct = oxygen_in_oxides_fraction(regolith)

        h2o_res = planet.h2o_ice_kg or 0.0
        co2_res = planet.co2_ice_kg or 0.0
        has_perchlorates = regolith.get("Cl", 0.0) > 0.0

        # Structural metal potential
        fe_pct = elements_recovered.get("Fe", 0.0) * 100
        al_pct = elements_recovered.get("Al", 0.0) * 100
        si_pct = elements_recovered.get("Si", 0.0) * 100
        metal_desc = f"Rich in Fe ({fe_pct:.1f}%), Al ({al_pct:.1f}%) and Si ({si_pct:.1f}%) suitable for in-situ metallurgy."

        return SoilAnalysis(
            body_name=planet.body_name,
            has_solid_surface=True,
            regolith_density_kg_m3=density,
            oxide_fractions=dict(regolith),
            elemental_mass_fractions=elements_recovered,
            h2o_ice_reserve_kg=h2o_res,
            co2_ice_reserve_kg=co2_res,
            extractable_o2_from_regolith_pct=o2_bound_pct * 100.0,
            mre_specific_energy_mj_kg_o2=cls.MRE_ENERGY_MJ_KG_O2,
            h2o_electrolysis_energy_mj_kg_o2=cls.H2O_ELECTROLYSIS_MJ_KG_O2 + cls.H2O_MINING_ENERGY_MJ_KG * (18.015 / 15.999),
            perchlorate_toxicity_present=has_perchlorates,
            structural_metal_potential=metal_desc,
            notes=planet.notes or "",
        )

    @classmethod
    def compute_oxygen_extraction(
        cls,
        planet: PlanetState,
        o2_target_kg: float,
        preferred_source: str = "auto",
    ) -> OxygenExtractionPlan:
        """Compute the mining, mass processing, energy, and co-products for O2 extraction."""
        analysis = cls.analyze(planet)

        if not analysis.has_solid_surface:
            return OxygenExtractionPlan(
                o2_target_kg=o2_target_kg,
                primary_source="none",
                regolith_mined_kg=0.0,
                water_ice_consumed_kg=0.0,
                metals_coproduced_kg={},
                extraction_energy_joules=0.0,
                extraction_energy_kwh=0.0,
                processing_method="Infeasible: no solid surface for mining",
                isru_feasible=False,
                limiting_factor="No lithosphere",
            )

        # Dynamic multi-source selection based on soil/regolith geochemical assay
        # Priority hierarchy by specific energy and operational suitability:
        # 1. Perchlorate decomposition (~6 MJ/kg O2) for local scales where Cl is present
        # 2. Subsurface Water Ice electrolysis (~21 MJ/kg O2) when H2O reserves exist
        # 3. Iron oxide carbothermal reduction (~28 MJ/kg O2) when Fe2O3 is abundant
        # 4. Molten Regolith Electrolysis (MRE) (~40 MJ/kg O2) for general silicate lithospheres

        has_perchlorates = analysis.perchlorate_toxicity_present and ("Cl" in analysis.oxide_fractions or planet.body_name == "Mars")
        fe2o3_pct = analysis.oxide_fractions.get("Fe2O3", 0.0)
        h2o_reserve = analysis.h2o_ice_reserve_kg

        # Decision logic
        if preferred_source == "auto":
            if has_perchlorates and o2_target_kg <= 1.0e6:
                chosen_source = "perchlorate_decomposition"
            elif h2o_reserve >= (o2_target_kg * 1.125):
                chosen_source = "water_ice"
            elif fe2o3_pct >= 0.05 and o2_target_kg <= 1.0e14:
                chosen_source = "iron_oxide_reduction"
            else:
                chosen_source = "molten_regolith_electrolysis"
        else:
            chosen_source = preferred_source

        # --- Pathway 1: Perchlorate Decomposition ---
        if chosen_source == "perchlorate_decomposition":
            # ClO4- -> Cl- + 2 O2 (~64% O2 by mass)
            perchlorates_needed = o2_target_kg / 0.64
            # Cl concentration in Martian regolith ~0.5%
            cl_frac = max(0.005, analysis.oxide_fractions.get("Cl", 0.005))
            regolith_mined = perchlorates_needed / cl_frac
            energy_per_kg_j = 6.0e6
            total_energy_j = o2_target_kg * energy_per_kg_j

            return OxygenExtractionPlan(
                o2_target_kg=o2_target_kg,
                primary_source="perchlorate_decomposition",
                regolith_mined_kg=regolith_mined,
                water_ice_consumed_kg=0.0,
                metals_coproduced_kg={"Cl_salts": perchlorates_needed * 0.36},
                extraction_energy_joules=total_energy_j,
                extraction_energy_kwh=total_energy_j / 3.6e6,
                processing_method="Perchlorate Washing and Catalytic Decomposition (Soil Detoxification)",
                isru_feasible=True,
                notes=(
                    f"Detoxifies {regolith_mined:.2e} kg of regolith by extracting perchlorates at 400 °C. "
                    f"Highly efficient O2 yield with minimal energy expenditure (6 MJ/kg)."
                ),
            )

        # --- Pathway 2: Water Ice Mining + PEM Electrolysis ---
        elif chosen_source == "water_ice":
            h2o_needed = o2_target_kg * (18.015 / 15.999)
            energy_per_kg_o2_j = (cls.H2O_MINING_ENERGY_MJ_KG * (18.015 / 15.999) + cls.H2O_ELECTROLYSIS_MJ_KG_O2) * 1.0e6
            total_energy_j = o2_target_kg * energy_per_kg_o2_j
            h2_produced = o2_target_kg * (2.016 / 31.999)

            return OxygenExtractionPlan(
                o2_target_kg=o2_target_kg,
                primary_source="water_ice",
                regolith_mined_kg=0.0,
                water_ice_consumed_kg=h2o_needed,
                metals_coproduced_kg={"H2": h2_produced},
                extraction_energy_joules=total_energy_j,
                extraction_energy_kwh=total_energy_j / 3.6e6,
                processing_method="Thermal mining of subsurface ice + PEM Electrolysis",
                isru_feasible=h2o_reserve >= h2o_needed,
                notes=f"Consumes {h2o_needed:.2e} kg of in-situ H2O ice; produces {h2_produced:.2e} kg of useful H2.",
            )

        # --- Pathway 3: Iron Oxide Carbothermal / Direct Reduction ---
        elif chosen_source == "iron_oxide_reduction":
            # Fe2O3 -> 2 Fe + 1.5 O2 (yield ~30% O2)
            fe2o3_needed = o2_target_kg / 0.301
            regolith_mined = fe2o3_needed / max(0.01, fe2o3_pct)
            fe_metal_produced = fe2o3_needed * (111.69 / 159.69)
            energy_per_kg_j = 28.0e6  # 28 MJ/kg O2 at 1000 °C
            total_energy_j = o2_target_kg * energy_per_kg_j

            return OxygenExtractionPlan(
                o2_target_kg=o2_target_kg,
                primary_source="iron_oxide_reduction",
                regolith_mined_kg=regolith_mined,
                water_ice_consumed_kg=0.0,
                metals_coproduced_kg={"Fe": fe_metal_produced},
                extraction_energy_joules=total_energy_j,
                extraction_energy_kwh=total_energy_j / 3.6e6,
                processing_method="Carbothermal Reduction of Hematite/Magnetite (Fe2O3) at 1000 °C",
                isru_feasible=True,
                notes=(
                    f"Processes {regolith_mined:.2e} kg of iron-rich regolith ({fe2o3_pct*100:.1f}% Fe2O3). "
                    f"Co-produces {fe_metal_produced:.2e} kg of pure metallic iron for structural engineering."
                ),
            )

        # --- Pathway 4: Molten Regolith Electrolysis (MRE) ---
        else:
            o2_fraction = (analysis.extractable_o2_from_regolith_pct / 100.0) * 0.85
            if o2_fraction <= 0.05:
                o2_fraction = 0.35
            regolith_mined = o2_target_kg / o2_fraction

            fe_mass = regolith_mined * analysis.elemental_mass_fractions.get("Fe", 0.10) * 0.90
            al_mass = regolith_mined * analysis.elemental_mass_fractions.get("Al", 0.05) * 0.85
            si_mass = regolith_mined * analysis.elemental_mass_fractions.get("Si", 0.20) * 0.80
            total_energy_j = o2_target_kg * cls.MRE_ENERGY_MJ_KG_O2 * 1.0e6

            return OxygenExtractionPlan(
                o2_target_kg=o2_target_kg,
                primary_source="molten_regolith_electrolysis",
                regolith_mined_kg=regolith_mined,
                water_ice_consumed_kg=0.0,
                metals_coproduced_kg={"Fe": fe_mass, "Al": al_mass, "Si": si_mass},
                extraction_energy_joules=total_energy_j,
                extraction_energy_kwh=total_energy_j / 3.6e6,
                processing_method="Molten Regolith Electrolysis (MRE) at 1600 °C",
                isru_feasible=True,
                notes=(
                    f"Mining of {regolith_mined:.2e} kg of basaltic/silicate regolith. "
                    f"Co-produces structural metals: Fe={fe_mass:.2e} kg, Al={al_mass:.2e} kg, Si={si_mass:.2e} kg."
                ),
            )
