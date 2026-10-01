"""
noarco.engines.knowledge — Mechanism Knowledge Engine (v2)
=========================================================
Mechanism and reaction knowledge engine for terraforming
with analytical temperature and pressure support.

Given the AIR and SOIL composition as well as the ambient
TEMPERATURE and PRESSURE, it can:
1. Instantly filter physically and thermodynamically viable mechanisms.
2. Compute continuous penalties or bonuses according to proximity to the optimal range.
3. Automatically adjust the required energy per kg (get_adjusted_energy) and
   the mass efficiency (get_adjusted_efficiency).
4. Recommend optimal routes with zero heavy numerical loops (TRIADA).

Framework and TRIADA protocol author: Alejo Malia
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from noarco.analyzers.atmosphere import AtmosphereEngine
from noarco.analyzers.soil import SoilEngine
from noarco.core.planet_state import PlanetState

# ============================================================
# 1. Enums and basic types
# ============================================================

class Domain(str, Enum):
    ATMOSPHERE = "atmosphere"
    SOIL = "soil"
    BOTH = "both"


class ProcessType(str, Enum):
    CHEMICAL = "chemical"
    PHYSICAL = "physical"
    BIOLOGICAL = "biological"
    RADIATIVE = "radiative"
    MECHANICAL = "mechanical"


# ============================================================
# 2. Main data structures
# ============================================================

@dataclass
class Species:
    """Chemical element or compound."""
    formula: str
    name: str
    molar_mass: float          # g/mol
    is_gas: bool = False
    is_solid: bool = False
    is_liquid: bool = False


@dataclass
class ReactionStoichiometry:
    """Stoichiometry of a chemical or physical reaction."""
    reactants: dict[str, float]   # formula: coefficient
    products: dict[str, float]    # formula: coefficient


@dataclass
class MechanismEvaluation:
    """Detailed physical suitability evaluation of a mechanism under given T and P."""
    mechanism_id: str
    name: str
    applicable: bool
    score: float                         # 0.0 - 1.0 (weighted)
    temperature_score: float             # 0.0 - 1.0
    pressure_score: float                # 0.0 - 1.0
    composition_score: float             # 0.0 - 1.0
    base_energy_mj: float                # MJ / kg at optimal conditions
    adjusted_energy_mj: float            # MJ / kg corrected for T and P
    base_efficiency: float               # 0.0 - 1.0
    adjusted_efficiency: float           # 0.0 - 1.0 corrected for T and P
    limiting_factors: list[str] = field(default_factory=list)


@dataclass
class KnowledgeMechanism:
    """
    A terraforming mechanism or process within the knowledge base (v2).
    Holds the physical, chemical and thermodynamic metadata for fast matching
    with dynamic calculation as a function of temperature and pressure.
    """
    id: str
    name: str
    description: str
    process_type: ProcessType
    domain: Domain

    # Composition requirements
    required_atmosphere: dict[str, float] = field(default_factory=dict)  # formula: minimum fraction
    required_soil: dict[str, float] = field(default_factory=dict)        # formula: minimum fraction
    required_elements: set[str] = field(default_factory=set)

    # Stoichiometry
    stoichiometry: ReactionStoichiometry | None = None

    # Base parameters (at optimal reference conditions)
    energy_per_kg_product: float = 0.0          # MJ / kg of main product
    main_product: str = ""                      # formula of the main product
    mass_efficiency: float = 1.0                # 0-1 (convertible fraction at optimum)

    # Operating ranges and optimal conditions
    min_temperature: float = 150.0              # K (absolute minimum operating temperature)
    max_temperature: float = 2000.0             # K (maximum operating temperature)
    optimal_temperature: float = 300.0          # K (where the process performs best)
    min_pressure: float = 10.0                  # Pa
    max_pressure: float = 1.0e7                 # Pa
    optimal_pressure: float = 101325.0          # Pa (1 atm standard reference)

    # Approximate impact (for ranking and quick planning)
    warming_potential: float = 0.0              # K per arbitrary unit
    oxygen_yield: float = 0.0                   # kg O2 per kg of main reactant
    co2_removal: float = 0.0                    # kg CO2 removed per kg of reactant

    tags: list[str] = field(default_factory=list)

    # ---------------------------------------------------------
    # Temperature and pressure scoring methods
    # ---------------------------------------------------------

    def _range_score(self, value: float, min_v: float, max_v: float, optimal: float) -> float:
        """
        Return a 0-1 score according to how close 'value' is to the optimum
        within the allowed range [min_v, max_v].
        """
        if value < min_v or value > max_v:
            return 0.0

        # Normalized distance to the optimum
        if optimal <= min_v or optimal >= max_v:
            return 1.0  # No optimum defined in the interior

        span = max(optimal - min_v, max_v - optimal)
        distance = abs(value - optimal)
        score = max(0.0, 1.0 - (distance / span))
        return score

    def temperature_score(self, temperature: float) -> float:
        """Thermal suitability score (0 to 1)."""
        return self._range_score(
            temperature,
            self.min_temperature,
            self.max_temperature,
            self.optimal_temperature,
        )

    def pressure_score(self, pressure: float) -> float:
        """Barometric suitability score (0 to 1)."""
        return self._range_score(
            pressure,
            self.min_pressure,
            self.max_pressure,
            self.optimal_pressure,
        )

    def get_adjusted_energy(self, temperature: float, pressure: float) -> float:
        """
        Adjust the required energy according to T and P.
        Away from optimal conditions, the required energy increases due to
        compression work and pre-heating or pre-cooling.
        """
        t_score = self.temperature_score(temperature)
        p_score = self.pressure_score(pressure)

        # Outside the physical operating range: drastic penalty
        if t_score == 0.0 or p_score == 0.0:
            return self.energy_per_kg_product * 10.0

        # Continuous penalty according to distance from the optimum
        penalty = 1.0 + (1.0 - t_score) * 1.5 + (1.0 - p_score) * 0.8
        return self.energy_per_kg_product * penalty

    def get_adjusted_efficiency(self, temperature: float, pressure: float) -> float:
        """
        Adjust the mass conversion efficiency according to T and P.
        Efficiency degrades progressively away from the optimal regime.
        """
        t_score = self.temperature_score(temperature)
        p_score = self.pressure_score(pressure)

        if t_score == 0.0 or p_score == 0.0:
            return 0.0

        return self.mass_efficiency * (0.55 + 0.45 * t_score) * (0.70 + 0.30 * p_score)

    def evaluate(
        self,
        atmosphere: dict[str, float],
        soil: dict[str, float],
        temperature: float,
        pressure: float,
    ) -> MechanismEvaluation:
        """
        Exhaustively evaluate the mechanism under the given conditions,
        returning the full analytical breakdown of T, P, energy and efficiency.
        """
        limiting_factors: list[str] = []

        # 1. Temperature and pressure filters
        t_score = self.temperature_score(temperature)
        p_score = self.pressure_score(pressure)

        if temperature < self.min_temperature:
            limiting_factors.append(f"Temperature ({temperature:.1f} K) below minimum ({self.min_temperature:.1f} K)")
        elif temperature > self.max_temperature:
            limiting_factors.append(f"Temperature ({temperature:.1f} K) above maximum ({self.max_temperature:.1f} K)")

        if pressure < self.min_pressure:
            limiting_factors.append(f"Pressure ({pressure:.1f} Pa) below minimum ({self.min_pressure:.1f} Pa)")
        elif pressure > self.max_pressure:
            limiting_factors.append(f"Pressure ({pressure:.1f} Pa) above maximum ({self.max_pressure:.1f} Pa)")

        if t_score == 0.0 or p_score == 0.0:
            return MechanismEvaluation(
                mechanism_id=self.id,
                name=self.name,
                applicable=False,
                score=0.0,
                temperature_score=t_score,
                pressure_score=p_score,
                composition_score=0.0,
                base_energy_mj=self.energy_per_kg_product,
                adjusted_energy_mj=self.get_adjusted_energy(temperature, pressure),
                base_efficiency=self.mass_efficiency,
                adjusted_efficiency=0.0,
                limiting_factors=limiting_factors,
            )

        # 2. Atmosphere requirements
        comp_scores: list[float] = []
        for species, min_frac in self.required_atmosphere.items():
            current = atmosphere.get(species, 0.0)
            if current < min_frac * 0.3:
                limiting_factors.append(f"Insufficient atmospheric gas: {species} ({current*100:.2f}% < {min_frac*100:.2f}%)")
                return MechanismEvaluation(
                    mechanism_id=self.id,
                    name=self.name,
                    applicable=False,
                    score=0.0,
                    temperature_score=t_score,
                    pressure_score=p_score,
                    composition_score=0.0,
                    base_energy_mj=self.energy_per_kg_product,
                    adjusted_energy_mj=self.get_adjusted_energy(temperature, pressure),
                    base_efficiency=self.mass_efficiency,
                    adjusted_efficiency=self.get_adjusted_efficiency(temperature, pressure),
                    limiting_factors=limiting_factors,
                )
            comp_scores.append(min(1.0, current / (min_frac + 1e-9)))

        # 3. Soil / regolith requirements
        for species, min_frac in self.required_soil.items():
            current = soil.get(species, 0.0)
            if current < min_frac * 0.3:
                limiting_factors.append(f"Insufficient soil mineral: {species} ({current*100:.2f}% < {min_frac*100:.2f}%)")
                return MechanismEvaluation(
                    mechanism_id=self.id,
                    name=self.name,
                    applicable=False,
                    score=0.0,
                    temperature_score=t_score,
                    pressure_score=p_score,
                    composition_score=0.0,
                    base_energy_mj=self.energy_per_kg_product,
                    adjusted_energy_mj=self.get_adjusted_energy(temperature, pressure),
                    base_efficiency=self.mass_efficiency,
                    adjusted_efficiency=self.get_adjusted_efficiency(temperature, pressure),
                    limiting_factors=limiting_factors,
                )
            comp_scores.append(min(1.0, current / (min_frac + 1e-9)))

        # 4. Required elements
        for elem in self.required_elements:
            in_soil = elem in soil or any(elem in k for k in soil)
            in_atm = elem in atmosphere or any(elem in k for k in atmosphere)
            if not (in_soil or in_atm):
                limiting_factors.append(f"Element absent in-situ: {elem}")
                return MechanismEvaluation(
                    mechanism_id=self.id,
                    name=self.name,
                    applicable=False,
                    score=0.0,
                    temperature_score=t_score,
                    pressure_score=p_score,
                    composition_score=0.0,
                    base_energy_mj=self.energy_per_kg_product,
                    adjusted_energy_mj=self.get_adjusted_energy(temperature, pressure),
                    base_efficiency=self.mass_efficiency,
                    adjusted_efficiency=self.get_adjusted_efficiency(temperature, pressure),
                    limiting_factors=limiting_factors,
                )

        composition_score = sum(comp_scores) / len(comp_scores) if comp_scores else 1.0

        # Integrated applicability score: 45% temperature + 25% pressure + 30% composition
        integrated_score = (t_score * 0.45 + p_score * 0.25 + composition_score * 0.30)
        final_score = max(0.0, min(1.0, integrated_score))

        return MechanismEvaluation(
            mechanism_id=self.id,
            name=self.name,
            applicable=True,
            score=final_score,
            temperature_score=t_score,
            pressure_score=p_score,
            composition_score=composition_score,
            base_energy_mj=self.energy_per_kg_product,
            adjusted_energy_mj=self.get_adjusted_energy(temperature, pressure),
            base_efficiency=self.mass_efficiency,
            adjusted_efficiency=self.get_adjusted_efficiency(temperature, pressure),
            limiting_factors=limiting_factors,
        )

    def is_applicable(
        self,
        atmosphere: dict[str, float],
        soil: dict[str, float],
        temperature: float,
        pressure: float,
    ) -> tuple[bool, float]:
        """
        Return (is_applicable, total_score 0-1).
        The score dynamically combines composition + temperature + pressure.
        """
        evaluation = self.evaluate(atmosphere, soil, temperature, pressure)
        return evaluation.applicable, evaluation.score


# Canonical alias
Mechanism = KnowledgeMechanism


# ============================================================
# 3. Mechanism catalog (the precomputed knowledge base)
# ============================================================

def build_default_catalog() -> dict[str, KnowledgeMechanism]:
    """
    Build the exhaustive catalog of physico-chemical mechanisms and reactions
    with thermodynamic calibration of T and P ranges and optima.
    """
    catalog: dict[str, KnowledgeMechanism] = {}

    # ---------- 1. Water electrolysis (H2O) ----------
    catalog["electrolysis_h2o"] = KnowledgeMechanism(
        id="electrolysis_h2o",
        name="Water Electrolysis (PEM / SOEC)",
        description="2 H₂O → 2 H₂ + O₂. Direct, mass-efficient oxygen source from ice or liquid water.",
        process_type=ProcessType.CHEMICAL,
        domain=Domain.BOTH,
        required_soil={"H2O": 0.005},
        required_elements={"H", "O"},
        stoichiometry=ReactionStoichiometry(
            reactants={"H2O": 2},
            products={"H2": 2, "O2": 1}
        ),
        energy_per_kg_product=21.2,          # MJ/kg O2: reversible 14.8 MJ/kg (2 dG/M_O2) / 0.70 system efficiency
        main_product="O2",
        mass_efficiency=0.95,
        oxygen_yield=0.888,                  # 16/18 kg O2 per kg H2O
        min_temperature=180.0,
        max_temperature=500.0,
        optimal_temperature=353.15,          # 80 °C (PEM / SOEC thermal optimum)
        min_pressure=50.0,
        max_pressure=2.0e7,                  # Hasta 200 bar
        optimal_pressure=101325.0,           # 1 bar
        tags=["oxygen", "water", "isru", "life_support"]
    )

    # ---------- 2. Iron oxide reduction / MRE ----------
    catalog["iron_oxide_reduction"] = KnowledgeMechanism(
        id="iron_oxide_reduction",
        name="Molten Regolith Electrolysis (MRE) / Fe₂O₃ Reduction",
        description="2 Fe₂O₃ → 4 Fe + 3 O₂. High-temperature oxygen extraction from basaltic regolith.",
        process_type=ProcessType.CHEMICAL,
        domain=Domain.SOIL,
        required_soil={"Fe2O3": 0.03},
        required_elements={"Fe", "O"},
        stoichiometry=ReactionStoichiometry(
            reactants={"Fe2O3": 2},
            products={"Fe": 4, "O2": 3}
        ),
        energy_per_kg_product=38.0,          # ~35-45 MJ/kg O2
        main_product="O2",
        mass_efficiency=0.85,
        oxygen_yield=0.301,                  # 96 / 319.4 kg O2 per kg Fe2O3
        min_temperature=150.0,
        max_temperature=2200.0,
        optimal_temperature=1873.15,         # 1600 °C (silicate slag melting)
        min_pressure=10.0,
        max_pressure=1.0e7,
        optimal_pressure=101325.0,
        tags=["oxygen", "regolith", "metals", "high_temperature", "dry_worlds"]
    )

    # ---------- 3. Sabatier reaction ----------
    catalog["sabatier"] = KnowledgeMechanism(
        id="sabatier",
        name="Sabatier Methanation",
        description="CO₂ + 4 H₂ → CH₄ + 2 H₂O. Reduces atmospheric CO₂ by fixing it into water and methane for propellant/heating.",
        process_type=ProcessType.CHEMICAL,
        domain=Domain.ATMOSPHERE,
        required_atmosphere={"CO2": 0.05},
        required_elements={"C", "O", "H"},
        stoichiometry=ReactionStoichiometry(
            reactants={"CO2": 1, "H2": 4},
            products={"CH4": 1, "H2O": 2}
        ),
        energy_per_kg_product=8.5,           # Exothermic with Ni/Ru catalysis
        main_product="H2O",
        mass_efficiency=0.92,
        co2_removal=1.0,
        min_temperature=200.0,
        max_temperature=800.0,
        optimal_temperature=573.15,          # 300 °C (peak Sabatier catalytic yield)
        min_pressure=500.0,
        max_pressure=1.0e7,
        optimal_pressure=300000.0,           # 3 bar favors Le Chatelier equilibrium
        tags=["co2_removal", "water_production", "methane", "fuel"]
    )

    # ---------- 4. Bosch reaction ----------
    catalog["bosch"] = KnowledgeMechanism(
        id="bosch",
        name="Bosch Reduction (Solid Carbon Fixation)",
        description="CO₂ + 2 H₂ → C(s) + 2 H₂O. Removes CO₂ by fixing solid elemental carbon (graphite) and recycling water.",
        process_type=ProcessType.CHEMICAL,
        domain=Domain.ATMOSPHERE,
        required_atmosphere={"CO2": 0.05},
        required_elements={"C", "O", "H"},
        stoichiometry=ReactionStoichiometry(
            reactants={"CO2": 1, "H2": 2},
            products={"C": 1, "H2O": 2}
        ),
        energy_per_kg_product=14.0,
        main_product="C",
        mass_efficiency=0.88,
        co2_removal=1.0,
        min_temperature=300.0,
        max_temperature=950.0,
        optimal_temperature=800.0,           # ~530 °C, Fe/Ni catalyzed
        min_pressure=1000.0,
        max_pressure=1.0e7,
        optimal_pressure=101325.0,
        tags=["co2_removal", "carbon_sink", "water_production", "venus"]
    )

    # ---------- 5. Catalytic perchlorate decomposition ----------
    catalog["perchlorate_decomposition"] = KnowledgeMechanism(
        id="perchlorate_decomposition",
        name="Catalytic Perchlorate Decomposition",
        description="ClO₄⁻ → Cl⁻ + 2 O₂. Detoxification of Martian soils with simultaneous release of pure oxygen.",
        process_type=ProcessType.CHEMICAL,
        domain=Domain.SOIL,
        required_soil={"Cl": 0.002},
        required_elements={"Cl", "O"},
        stoichiometry=ReactionStoichiometry(
            reactants={"ClO4": 1},
            products={"Cl": 1, "O2": 2}
        ),
        energy_per_kg_product=6.0,           # Ru/Fe-catalyzed thermolysis
        main_product="O2",
        mass_efficiency=0.90,
        oxygen_yield=0.6435,
        min_temperature=150.0,
        max_temperature=750.0,
        optimal_temperature=473.15,          # 200 °C
        min_pressure=10.0,
        max_pressure=1.0e7,
        optimal_pressure=101325.0,
        tags=["oxygen", "soil_detoxification", "mars", "agriculture"]
    )

    # ---------- 6. Mineral CO2 sequestration in carbonates ----------
    catalog["carbonate_mineralization"] = KnowledgeMechanism(
        id="carbonate_mineralization",
        name="Mineral Silicate Carbonation (Enhanced Weathering)",
        description="Mg₂SiO₄ + 2 CO₂ → 2 MgCO₃ + SiO₂. Massive, permanent chemical sequestration of CO₂ in the lithosphere.",
        process_type=ProcessType.CHEMICAL,
        domain=Domain.BOTH,
        required_atmosphere={"CO2": 0.10},
        required_soil={"MgO": 0.05},
        required_elements={"Mg", "Si", "C", "O"},
        stoichiometry=ReactionStoichiometry(
            reactants={"Mg2SiO4": 1, "CO2": 2},
            products={"MgCO3": 2, "SiO2": 1}
        ),
        energy_per_kg_product=3.5,
        main_product="MgCO3",
        mass_efficiency=0.80,
        co2_removal=0.625,                  # 88 kg CO2 fixed per 140 kg olivine
        min_temperature=220.0,
        max_temperature=850.0,
        optimal_temperature=450.0,
        min_pressure=100.0,
        max_pressure=1.0e8,
        optimal_pressure=9.2e6,             # High pressure (e.g. Venus) accelerates carbonation
        tags=["co2_removal", "carbon_sink", "planetary_scale", "venus"]
    )

    # ---------- 7. Nanoparticles / Metallic aerosols ----------
    catalog["metal_aerosol_warming"] = KnowledgeMechanism(
        id="metal_aerosol_warming",
        name="Metallic Nanorod Aerosols (Al/Fe)",
        description="Optimized Al/Fe nanorods dispersed in the atmosphere for IR radiative forcing (Ansari et al. 2024).",
        process_type=ProcessType.RADIATIVE,
        domain=Domain.BOTH,
        required_soil={"Fe2O3": 0.03},
        required_elements={"Fe"},
        energy_per_kg_product=18.5,          # Mining + nanorod fabrication
        main_product="aerosol",
        warming_potential=25.0,              # Very mass-efficient ΔT
        min_temperature=50.0,
        max_temperature=450.0,
        optimal_temperature=220.0,          # Martian cryosphere
        min_pressure=1.0,
        max_pressure=2.0e6,
        optimal_pressure=610.0,              # Martian surface pressure
        tags=["warming", "aerosol", "fast", "isru", "mars"]
    )

    # ---------- 8. Silica Aerogel ----------
    catalog["silica_aerogel"] = KnowledgeMechanism(
        id="silica_aerogel",
        name="Silica Aerogel Domes / Layers",
        description="Solid-state greenhouse effect (Wordsworth et al. 2019). A 2-3 cm layer that warms by +40 K and blocks UV.",
        process_type=ProcessType.PHYSICAL,
        domain=Domain.SOIL,
        required_soil={"SiO2": 0.15},
        required_elements={"Si", "O"},
        energy_per_kg_product=12.0,          # Supercritical drying
        main_product="aerogel_shield",
        warming_potential=40.0,
        min_temperature=50.0,
        max_temperature=380.0,
        optimal_temperature=210.0,
        min_pressure=1.0,
        max_pressure=2.0e6,
        optimal_pressure=610.0,
        tags=["warming", "paraterraforming", "regional", "uv_shield", "isru"]
    )

    # ---------- 9. Polar / adsorbed CO₂ release ----------
    catalog["co2_release"] = KnowledgeMechanism(
        id="co2_release",
        name="Thermal Sublimation of Polar CO₂ and Regolith Outgassing",
        description="Focused heating of ice caps and regolith to release CO₂ into the atmosphere and raise the background pressure.",
        process_type=ProcessType.PHYSICAL,
        domain=Domain.BOTH,
        required_soil={"CO2_ice": 0.005},
        required_elements={"C", "O"},
        energy_per_kg_product=0.59,          # Latent heat of sublimation ~590 kJ/kg
        main_product="CO2",
        warming_potential=5.0,
        min_temperature=120.0,
        max_temperature=320.0,
        optimal_temperature=216.58,         # Punto triple CO2 (5.18 bar / 216.58 K)
        min_pressure=10.0,
        max_pressure=1.0e7,
        optimal_pressure=610.0,
        tags=["pressure_increase", "warming", "polar", "mars"]
    )

    # ---------- 10. Biological photosynthesis / Cyanobacteria ----------
    catalog["cyanobacteria_photosynthesis"] = KnowledgeMechanism(
        id="cyanobacteria_photosynthesis",
        name="Photosynthetic Cyanobacteria Bioremediation",
        description="6 CO₂ + 6 H₂O + luz → C₆H₁₂O₆ + 6 O₂. Continuous passive biological production with nitrogen fixation.",
        process_type=ProcessType.BIOLOGICAL,
        domain=Domain.BOTH,
        required_atmosphere={"CO2": 0.01},
        required_soil={"H2O": 0.02},
        required_elements={"C", "H", "O", "N"},
        stoichiometry=ReactionStoichiometry(
            reactants={"CO2": 6, "H2O": 6},
            products={"C6H12O6": 1, "O2": 6}
        ),
        energy_per_kg_product=0.5,           # Minimal auxiliary energy (sunlight is primary)
        main_product="O2",
        mass_efficiency=0.75,
        oxygen_yield=0.727,
        co2_removal=1.0,
        min_temperature=273.15,              # Requires liquid water (≥ 0 °C)
        max_temperature=335.0,
        optimal_temperature=298.15,         # 25 °C
        min_pressure=610.0,                  # Minimum pressure: water triple point
        max_pressure=5.0e5,
        optimal_pressure=101325.0,
        tags=["biological", "oxygen", "co2_removal", "soil_enrichment", "passive"]
    )

    return catalog


# ============================================================
# 4. El Motor Principal: MechanismKnowledgeEngine
# ============================================================

class MechanismKnowledgeEngine:
    """
    Mechanism and reaction knowledge engine for terraforming (v2).
    Given the AIR and SOIL composition together with PRESSURE and TEMPERATURE,
    it instantly filters and recommends the best chemical and physical routes,
    dynamically adjusting energy and yields without heavy simulations.
    """

    def __init__(self, catalog: dict[str, KnowledgeMechanism] | None = None) -> None:
        self.catalog: dict[str, KnowledgeMechanism] = (
            catalog if catalog is not None else build_default_catalog()
        )

    def evaluate_all(
        self,
        atmosphere: dict[str, float],
        soil: dict[str, float],
        temperature: float = 280.0,
        pressure: float = 101_325.0,
        goal_tags: list[str] | None = None,
    ) -> list[MechanismEvaluation]:
        """
        Analytically evaluate all catalog mechanisms against the given conditions,
        returning the evaluations sorted by descending score.
        """
        evaluations: list[MechanismEvaluation] = []

        for mech in self.catalog.values():
            eval_res = mech.evaluate(atmosphere, soil, temperature, pressure)

            if not eval_res.applicable:
                continue

            # Additional weighting if it matches the requested goals
            if goal_tags:
                matched_tags = set(mech.tags) & set(goal_tags)
                if matched_tags:
                    tag_bonus = 1.0 + 0.50 * len(matched_tags)
                    eval_res.score = eval_res.score * tag_bonus
                else:
                    # If it matches none of the explicit goals, penalize
                    eval_res.score *= 0.25

            evaluations.append(eval_res)

        evaluations.sort(key=lambda x: x.score, reverse=True)
        return evaluations

    def find_applicable(
        self,
        atmosphere: dict[str, float],
        soil: dict[str, float],
        temperature: float = 280.0,
        pressure: float = 101_325.0,
        goal_tags: list[str] | None = None,
    ) -> list[tuple[KnowledgeMechanism, float]]:
        """
        Return a list of (mechanism, suitability_score) sorted from highest to lowest.
        Fully preserves compatibility with the v1 API.
        """
        evals = self.evaluate_all(atmosphere, soil, temperature, pressure, goal_tags)
        results: list[tuple[KnowledgeMechanism, float]] = []
        for ev in evals:
            mech = self.catalog[ev.mechanism_id]
            results.append((mech, ev.score))
        return results

    def recommend(
        self,
        atmosphere: dict[str, float],
        soil: dict[str, float],
        temperature: float = 280.0,
        pressure: float = 101_325.0,
        goals: list[str] | None = None,
        top_n: int = 5,
        verbose: bool = True,
    ) -> list[tuple[KnowledgeMechanism, float]]:
        """
        Generate and print the best recommendations with a breakdown of
        energy and efficiency dynamically adjusted for temperature and pressure.
        """
        goals = goals or []
        out = print if verbose else (lambda *_a, **_k: None)
        evals = self.evaluate_all(atmosphere, soil, temperature, pressure, goals)

        out("=" * 80)
        out("MECHANISM RECOMMENDATIONS (Mechanism Knowledge Engine v2 — NOArCO)")
        out("=" * 80)
        out(f"Ambient Conditions: T = {temperature:.1f} K | P = {pressure:.1f} Pa")
        out(f"Selected Goals : {', '.join(goals) if goals else 'General / Multi-objective'}")
        out("-" * 80)

        if not evals:
            out("No viable mechanisms found under the current physical and chemical conditions.")
            return []

        selected = evals[:top_n]
        for i, ev in enumerate(selected, 1):
            mech = self.catalog[ev.mechanism_id]
            energy_diff_pct = ((ev.adjusted_energy_mj - ev.base_energy_mj) / (ev.base_energy_mj + 1e-9)) * 100.0
            sign = "+" if energy_diff_pct >= 0 else ""

            out(f"\n{i}. {mech.name} [ID: {mech.id}]")
            out(f"   Suitability Score: {ev.score:.3f} | T_score: {ev.temperature_score:.2f} | P_score: {ev.pressure_score:.2f}")
            out(f"   Type: {mech.process_type.value.upper()} | Domain: {mech.domain.value.upper()}")
            out(f"   Description: {mech.description}")
            out(f"   Base Energy     : {ev.base_energy_mj:.2f} MJ/kg → Adjusted for T/P: {ev.adjusted_energy_mj:.2f} MJ/kg ({sign}{energy_diff_pct:.1f}%)")
            out(f"   Mass Efficiency: {ev.base_efficiency*100:.1f}% → Adjusted for T/P: {ev.adjusted_efficiency*100:.1f}%")
            if mech.oxygen_yield > 0:
                out(f"   O₂ Yield       : {mech.oxygen_yield:.3f} kg O₂ / kg reactant")
            if mech.warming_potential > 0:
                out(f"   Warming Potential: +{mech.warming_potential:.1f} K")
            if mech.co2_removal > 0:
                out(f"   CO₂ Removal     : {mech.co2_removal:.3f} kg CO₂ / kg reactant")
            out(f"   Tags: {', '.join(mech.tags)}")

        # Return compatible tuples
        return [(self.catalog[ev.mechanism_id], ev.score) for ev in selected]

    def recommend_for_planet(
        self,
        planet: PlanetState,
        goals: list[str] | None = None,
        top_n: int = 5,
        verbose: bool = True,
    ) -> list[tuple[KnowledgeMechanism, float]]:
        """
        Shortcut that automatically extracts the AIR and SOIL composition,
        TEMPERATURE and PRESSURE from the PlanetState using AtmosphereEngine and SoilEngine.
        """
        atm_analysis = AtmosphereEngine.analyze(planet)
        soil_analysis = SoilEngine.analyze(planet)

        # Build soil dictionary with oxides + ice
        soil_dict = dict(soil_analysis.oxide_fractions)
        if planet.h2o_ice_kg and planet.h2o_ice_kg > 0:
            soil_dict["H2O"] = 0.05
        if planet.co2_ice_kg and planet.co2_ice_kg > 0:
            soil_dict["CO2_ice"] = 0.02

        # Include detected main elements
        for elem, frac in soil_analysis.elemental_mass_fractions.items():
            soil_dict[elem] = frac

        return self.recommend(
            atmosphere=atm_analysis.composition_moles,
            soil=soil_dict,
            temperature=planet.mean_temperature_k,
            pressure=planet.surface_pressure_pa,
            goals=goals,
            top_n=top_n,
            verbose=verbose,
        )

    def get_mechanism(self, mechanism_id: str) -> KnowledgeMechanism | None:
        """Retrieve a mechanism by its unique identifier."""
        return self.catalog.get(mechanism_id)

    def add_mechanism(self, mechanism: KnowledgeMechanism) -> None:
        """Allow extending the knowledge base at runtime."""
        self.catalog[mechanism.id] = mechanism


# ============================================================
# 5. Canonical engine aliases for the framework
# ============================================================
ProcessCatalogEngine = MechanismKnowledgeEngine
ReactionPathEngine = MechanismKnowledgeEngine
KnowledgeEngine = MechanismKnowledgeEngine
MechanismMatcher = MechanismKnowledgeEngine
