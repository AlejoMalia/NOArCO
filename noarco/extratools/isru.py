"""
noarco.extratools.isru — NOArCO-ISRU
===================================
In-Situ Resource Utilization (ISRU) and Mining Site Evaluator.
Computes daily regolith excavation tonnage, reactor energy balances,
metallic by-product mass and compares viability across different sites.

Framework: NOArCO
Framework and TRIADA protocol author: Alejo Malia
"""

from __future__ import annotations

from dataclasses import dataclass, field

from noarco.core.constants import MOLAR_MASS, reversible_o2_energy_j_per_kg
from noarco.engines.knowledge import MechanismKnowledgeEngine

# Atomic / molar masses (kg/mol, IUPAC 2013 standard atomic weights) for the stoichiometric ledger.
_M_FE, _M_O, _M_CL = 55.845e-3, 15.999e-3, 35.45e-3
_M_FE2O3 = 2 * _M_FE + 3 * _M_O
_M_CLO4 = _M_CL + 4 * _M_O
#: Standard Gibbs energy of formation of Fe2O3 (hematite), 298 K: -742.2 kJ/mol (NIST-JANAF).
_DG_F_FE2O3_J_MOL = 742_200.0


@dataclass(frozen=True)
class Chemistry:
    """Stoichiometry of an O2-releasing reaction: what is mined, what co-product appears, the energy floor."""
    active_mineral: str                 # assay key of the reactant
    alt_key: str | None                 # alternative assay key and its mass conversion to the reactant
    alt_factor: float
    coproduct: str | None
    coproduct_per_o2: float             # kg co-product per kg O2 (from molar masses)
    floor_j_per_kg_o2: float            # reversible (Gibbs) minimum work per kg O2; 0 for exergonic steps
    floor_source: str


CHEMISTRIES: dict[str, Chemistry] = {
    "electrolysis_h2o": Chemistry("H2O", None, 1.0, "H2_fuel_kg", 2 * MOLAR_MASS["H2"] / MOLAR_MASS["O2"],
                                  reversible_o2_energy_j_per_kg(), "2 dG_f(H2O) / M_O2 (CODATA/NBS)"),
    "iron_oxide_reduction": Chemistry("Fe2O3", None, 1.0, "Fe_metal_kg", 4 * _M_FE / (3 * MOLAR_MASS["O2"]),
                                      _DG_F_FE2O3_J_MOL / (1.5 * MOLAR_MASS["O2"]), "dG_f(Fe2O3) = -742.2 kJ/mol (NIST-JANAF)"),
    "perchlorate_decomposition": Chemistry("ClO4", "Cl", _M_CLO4 / _M_CL, "Cl_salt_kg", _M_CL / (2 * MOLAR_MASS["O2"]),
                                           0.0, "exergonic: no positive thermodynamic floor"),
}


@dataclass
class ISRURequirement:
    """Production requirements for the ISRU plant."""
    o2_kg_day: float = 100.0               # kg of oxygen per day
    water_kg_day: float = 50.0             # kg of potable water / propellant per day
    metal_kg_day: float = 20.0             # kg of structural iron/aluminum per day
    operating_hours_per_day: float = 24.0   # Effective hours (continuous solar or nuclear reactor)
    # Engineering assumptions (not measurements): replace with plant data.
    water_recovery_fraction: float = 0.90   # fraction of the regolith H2O recovered by heating
    excavation_kwh_per_t: float = 1.5       # excavation + grinding energy per tonne of regolith
    solar_array_efficiency: float = 0.25    # electrical efficiency of the array
    solar_derate: float = 0.80              # dust, pointing, degradation

    def __post_init__(self) -> None:
        for name in ("o2_kg_day", "water_kg_day", "metal_kg_day", "excavation_kwh_per_t"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)!r}")
        if not 0 < self.operating_hours_per_day <= 24:
            raise ValueError("operating_hours_per_day must be in (0, 24]")
        for name in ("water_recovery_fraction", "solar_array_efficiency", "solar_derate"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass
class ISRUSiteProfile:
    """Geological and mineralogical profile of a landing site."""
    name: str
    planet_name: str
    temperature_k: float
    pressure_pa: float
    mineralogy: dict[str, float]          # Compound: mass fraction (SiO2, Fe2O3, H2O, Cl, etc.)
    solar_irradiance_w_m2: float = 590.0

    def __post_init__(self) -> None:
        if not (self.temperature_k > 0 and self.pressure_pa >= 0 and self.solar_irradiance_w_m2 >= 0):
            raise ValueError("temperature_k must be > 0; pressure_pa, irradiance must be >= 0")
        if any(f < 0 for f in self.mineralogy.values()) or sum(self.mineralogy.values()) > 1.0 + 1e-6:
            raise ValueError("mineralogy mass fractions must be >= 0 and sum to <= 1")


@dataclass
class ISRUProductionReport:
    """Technical and engineering report of the ISRU plant."""
    site_name: str
    selected_mechanism_id: str
    selected_mechanism_name: str
    daily_o2_kg: float
    daily_water_kg: float
    daily_metals_kg: float
    daily_regolith_mined_kg: float        # Total mass of soil to remove and process
    daily_tailings_waste_kg: float        # Leftover tailings (regolith minus everything extracted)
    reactor_energy_kwh_day: float         # Electrical/thermal energy consumed
    continuous_power_kw: float            # Continuous plant power
    specific_energy_kwh_per_kg_o2: float  # kWh / kg net O2
    co_products: dict[str, float] = field(default_factory=dict)
    feasibility_notes: list[str] = field(default_factory=list)
    # --- ledger and integrity (added) ---
    reactant_consumed_kg_day: float = 0.0     # mass of active mineral consumed to make the O2
    unaccounted_mass_kg_day: float = 0.0      # reactant - O2 - co-products (should be ~0 for a closed reaction)
    energy_floor_kwh_day: float = 0.0         # reversible (Gibbs) minimum for the O2 made
    exergy_efficiency: float | None = None    # floor / actual (None when the floor is 0)
    water_from_regolith_kg_day: float = 0.0
    solar_array_area_m2: float | None = None  # array that powers the plant during its operating hours
    warnings: list[str] = field(default_factory=list)
    model_class: str = "engineering_estimate"


class ISRUEvaluator:
    """
    NOArCO space-mining prospecting and analytical calculation engine.
    Determines the most efficient processing chain by crossing the local mineralogy
    with the Mechanism Knowledge Engine, then closes a mass ledger (reactant = O2 + co-products),
    checks the energy against the reversible thermodynamic floor and sizes the solar array.
    """

    def __init__(self, knowledge_engine: MechanismKnowledgeEngine | None = None) -> None:
        self.knowledge_engine = knowledge_engine or MechanismKnowledgeEngine()

    def evaluate_site(
        self,
        site: ISRUSiteProfile,
        req: ISRURequirement,
    ) -> ISRUProductionReport:
        evals = self.knowledge_engine.evaluate_all(
            atmosphere={"CO2": 0.95},
            soil=site.mineralogy,
            temperature=site.temperature_k,
            pressure=site.pressure_pa,
            goal_tags=["oxygen", "isru"],
        )
        if not evals:
            raise ValueError(f"No viable ISRU mechanisms for site {site.name} given its current mineralogy.")

        # Best ranked evaluation whose chemistry NOArCO can close stoichiometrically.
        best_eval = next((e for e in evals if e.mechanism_id in CHEMISTRIES), None)
        if best_eval is None:
            raise ValueError(
                f"No viable ISRU mechanism with a known stoichiometric ledger for site {site.name} "
                f"(have {[e.mechanism_id for e in evals]}; supported {sorted(CHEMISTRIES)})."
            )
        mech = self.knowledge_engine.get_mechanism(best_eval.mechanism_id)
        if mech is None:
            raise ValueError(f"Mechanism {best_eval.mechanism_id} not found in the catalog.")
        chem = CHEMISTRIES[mech.id]
        warnings: list[str] = []

        # 1. Yield and efficiency: no silent defaults.
        if mech.oxygen_yield <= 0:
            raise ValueError(f"Mechanism {mech.id} declares no oxygen yield; cannot size the plant.")
        yield_o2 = mech.oxygen_yield
        efficiency = best_eval.adjusted_efficiency
        if efficiency <= 0:
            raise ValueError(f"Mechanism {mech.id} has zero efficiency at T={site.temperature_k} K, P={site.pressure_pa} Pa.")

        # 2. Fraction of the active mineral in the assay (alternative key converted by stoichiometry).
        if chem.active_mineral in site.mineralogy:
            mineral_frac = site.mineralogy[chem.active_mineral]
        elif chem.alt_key and chem.alt_key in site.mineralogy:
            mineral_frac = site.mineralogy[chem.alt_key] * chem.alt_factor
            warnings.append(f"{chem.active_mineral} inferred from {chem.alt_key} by stoichiometry (x{chem.alt_factor:.3f}).")
        else:
            raise ValueError(f"Assay of {site.name} lacks {chem.active_mineral}"
                             f"{f' (or {chem.alt_key})' if chem.alt_key else ''}; cannot size {mech.id}.")
        if mineral_frac <= 0:
            raise ValueError(f"Assay of {site.name} has zero {chem.active_mineral}.")

        # 3. Mass ledger. O2 made = reactant * yield * efficiency  ->  reactant consumed = O2 / (yield * eff)
        reactant_kg = req.o2_kg_day / (yield_o2 * efficiency)
        h2o_frac = site.mineralogy.get("H2O", 0.0)
        water_is_reactant = chem.active_mineral == "H2O"
        regolith_for_o2 = reactant_kg / (mineral_frac * (req.water_recovery_fraction if water_is_reactant else 1.0))
        regolith_mined = regolith_for_o2
        water_from_regolith = 0.0
        if req.water_kg_day > 0:
            if h2o_frac <= 0:
                warnings.append("Water requested but the assay has no H2O: water must be imported.")
            else:
                # the same H2O feeds electrolysis (reactant) and the delivered water: one stream, one demand
                demand = req.water_kg_day + (reactant_kg if water_is_reactant else 0.0)
                regolith_for_water = demand / (h2o_frac * req.water_recovery_fraction)
                regolith_mined = max(regolith_for_o2, regolith_for_water)
                water_from_regolith = min(req.water_kg_day,
                                          max(regolith_mined * h2o_frac * req.water_recovery_fraction
                                              - (reactant_kg if water_is_reactant else 0.0), 0.0))
        if water_from_regolith < req.water_kg_day - 1e-9 and h2o_frac > 0:
            warnings.append("Water demand exceeds what the mined regolith releases.")

        co_products: dict[str, float] = {}
        if chem.coproduct:
            co_products[chem.coproduct] = req.o2_kg_day * chem.coproduct_per_o2
        # Conservation of the reaction: reactant (mass of the active mineral consumed *by the reaction*)
        # equals O2 + co-product; any rest is reported, not hidden.
        reacted_kg = req.o2_kg_day / yield_o2          # stoichiometric (efficiency affects what is *mined*, not the balance)
        unaccounted = reacted_kg - req.o2_kg_day - sum(co_products.values())
        if abs(unaccounted) > 0.01 * reacted_kg:
            warnings.append(f"Stoichiometric ledger leaves {unaccounted:.2f} kg/day unaccounted ({unaccounted/reacted_kg:.1%}).")

        tailings_kg = max(0.0, regolith_mined - req.o2_kg_day - sum(co_products.values()) - water_from_regolith)

        # 4. Energy: catalogue value, never below the reversible floor.
        floor_mj = req.o2_kg_day * chem.floor_j_per_kg_o2 / 1e6
        reactor_energy_mj = req.o2_kg_day * best_eval.adjusted_energy_mj
        if reactor_energy_mj < floor_mj - 1e-9:
            warnings.append("Catalogue energy is below the thermodynamic floor; floor used.")
            reactor_energy_mj = floor_mj
        excavation_energy_kwh = (regolith_mined / 1000.0) * req.excavation_kwh_per_t
        reactor_energy_kwh = reactor_energy_mj / 3.6 + excavation_energy_kwh
        continuous_power_kw = reactor_energy_kwh / req.operating_hours_per_day
        specific_energy_kwh_kg = reactor_energy_kwh / max(1e-3, req.o2_kg_day)
        exergy_eff = (floor_mj / 3.6) / reactor_energy_kwh if floor_mj > 0 and reactor_energy_kwh > 0 else None

        solar_area = None
        if site.solar_irradiance_w_m2 > 0:
            solar_area = continuous_power_kw * 1000.0 / (site.solar_irradiance_w_m2 * req.solar_array_efficiency * req.solar_derate)

        notes: list[str] = []
        if regolith_mined > 10_000:
            notes.append(f"Large-scale mining: requires a fleet of autonomous excavators (> {regolith_mined/1000:.1f} t/day).")
        if continuous_power_kw > 100:
            notes.append(f"High electrical demand ({continuous_power_kw:.1f} kW): requires a surface nuclear mini-reactor (Kilopower type).")

        return ISRUProductionReport(
            site_name=site.name,
            selected_mechanism_id=mech.id,
            selected_mechanism_name=mech.name,
            daily_o2_kg=req.o2_kg_day,
            daily_water_kg=req.water_kg_day,
            daily_metals_kg=co_products.get("Fe_metal_kg", req.metal_kg_day),
            daily_regolith_mined_kg=regolith_mined,
            daily_tailings_waste_kg=tailings_kg,
            reactor_energy_kwh_day=reactor_energy_kwh,
            continuous_power_kw=continuous_power_kw,
            specific_energy_kwh_per_kg_o2=specific_energy_kwh_kg,
            co_products=co_products,
            feasibility_notes=notes,
            reactant_consumed_kg_day=reactant_kg,
            unaccounted_mass_kg_day=unaccounted,
            energy_floor_kwh_day=floor_mj / 3.6,
            exergy_efficiency=exergy_eff,
            water_from_regolith_kg_day=water_from_regolith,
            solar_array_area_m2=solar_area,
            warnings=warnings,
        )
