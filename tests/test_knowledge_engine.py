"""
tests.test_knowledge_engine
===========================
Unit tests for MechanismKnowledgeEngine v2 (ProcessCatalogEngine / ReactionPathEngine).
Validates instant knowledge matching with Pressure and Temperature intelligence.
"""

import pytest

from noarco.data.planets import MARS, VENUS
from noarco.engines.knowledge import (
    Domain,
    Mechanism,
    MechanismKnowledgeEngine,
    ProcessCatalogEngine,
    ProcessType,
    build_default_catalog,
)


class TestMechanismKnowledgeEngine:
    def test_default_catalog_initialization(self):
        catalog = build_default_catalog()
        assert len(catalog) >= 10
        assert "electrolysis_h2o" in catalog
        assert "iron_oxide_reduction" in catalog
        assert "sabatier" in catalog
        assert "bosch" in catalog
        assert "metal_aerosol_warming" in catalog
        assert "silica_aerogel" in catalog
        assert "carbonate_mineralization" in catalog
        assert "perchlorate_decomposition" in catalog
        assert "cyanobacteria_photosynthesis" in catalog

    def test_mars_oxygen_matching(self):
        engine = MechanismKnowledgeEngine()
        mars_atm = {"CO2": 0.95, "N2": 0.027, "Ar": 0.016, "O2": 0.0015, "H2O": 0.0003}
        mars_soil = {
            "SiO2": 0.45,
            "Fe2O3": 0.18,
            "Al2O3": 0.09,
            "MgO": 0.08,
            "CaO": 0.06,
            "H2O": 0.03,
            "CO2_ice": 0.02,
            "Cl": 0.005,
        }

        recs = engine.find_applicable(
            atmosphere=mars_atm,
            soil=mars_soil,
            temperature=210.0,
            pressure=610.0,
            goal_tags=["oxygen"],
        )

        mech_ids = [m.id for m, score in recs]
        assert "electrolysis_h2o" in mech_ids
        assert "iron_oxide_reduction" in mech_ids
        assert "perchlorate_decomposition" in mech_ids

    def test_mars_warming_matching(self):
        engine = MechanismKnowledgeEngine()
        mars_atm = {"CO2": 0.95, "N2": 0.027}
        mars_soil = {"SiO2": 0.45, "Fe2O3": 0.18, "CO2_ice": 0.02}

        recs = engine.find_applicable(
            atmosphere=mars_atm,
            soil=mars_soil,
            temperature=210.0,
            pressure=610.0,
            goal_tags=["warming"],
        )

        mech_ids = [m.id for m, score in recs]
        assert "silica_aerogel" in mech_ids
        assert "metal_aerosol_warming" in mech_ids
        assert "co2_release" in mech_ids

    def test_venus_co2_removal_matching(self):
        engine = MechanismKnowledgeEngine()
        venus_atm = {"CO2": 0.965, "N2": 0.035}
        venus_soil = {"SiO2": 0.45, "MgO": 0.11, "Fe2O3": 0.09}

        recs = engine.find_applicable(
            atmosphere=venus_atm,
            soil=venus_soil,
            temperature=737.0,
            pressure=9.2e6,
            goal_tags=["co2_removal"],
        )

        mech_ids = [m.id for m, score in recs]
        # Silicate carbonation is applicable at Venus conditions
        assert "carbonate_mineralization" in mech_ids

    def test_recommend_for_planet(self):
        engine = MechanismKnowledgeEngine()
        recs_mars = engine.recommend_for_planet(MARS, goals=["oxygen"], top_n=3)
        assert len(recs_mars) >= 2

        recs_venus = engine.recommend_for_planet(VENUS, goals=["co2_removal"], top_n=2)
        assert len(recs_venus) >= 1

    def test_temperature_and_pressure_scoring(self):
        catalog = build_default_catalog()
        mech = catalog["electrolysis_h2o"]

        # Optimal temperature (353.15 K) should yield score 1.0
        t_opt_score = mech.temperature_score(mech.optimal_temperature)
        assert pytest.approx(t_opt_score, 1e-4) == 1.0

        # Mars cold (210 K) should be > 0 but < 1.0
        t_mars_score = mech.temperature_score(210.0)
        assert 0.0 < t_mars_score < 1.0

        # Below minimum temperature should be 0.0
        t_too_cold = mech.temperature_score(100.0)
        assert t_too_cold == 0.0

        # Pressure score at optimal
        p_opt_score = mech.pressure_score(mech.optimal_pressure)
        assert pytest.approx(p_opt_score, 1e-4) == 1.0

    def test_energy_and_efficiency_adjustment(self):
        catalog = build_default_catalog()
        mech = catalog["electrolysis_h2o"]

        # Base energy and efficiency at optimal conditions
        e_opt = mech.get_adjusted_energy(mech.optimal_temperature, mech.optimal_pressure)
        eff_opt = mech.get_adjusted_efficiency(mech.optimal_temperature, mech.optimal_pressure)
        assert pytest.approx(e_opt, 1e-4) == mech.energy_per_kg_product
        assert pytest.approx(eff_opt, 1e-4) == mech.mass_efficiency

        # At cold Mars conditions (210 K, 610 Pa), energy should be higher and efficiency lower
        e_mars = mech.get_adjusted_energy(210.0, 610.0)
        eff_mars = mech.get_adjusted_efficiency(210.0, 610.0)
        assert e_mars > mech.energy_per_kg_product
        assert eff_mars < mech.mass_efficiency

        # Out-of-range temperature penalty
        e_out = mech.get_adjusted_energy(50.0, 101325.0)
        assert e_out == mech.energy_per_kg_product * 10.0
        assert mech.get_adjusted_efficiency(50.0, 101325.0) == 0.0

    def test_evaluate_method_and_limiting_factors(self):
        catalog = build_default_catalog()
        mech = catalog["cyanobacteria_photosynthesis"]

        # Cyanobacteria requires liquid water (≥ 273.15 K)
        # On Mars at 210 K, it must not be applicable due to low temperature
        eval_mars = mech.evaluate(
            atmosphere={"CO2": 0.95},
            soil={"H2O": 0.05},
            temperature=210.0,
            pressure=610.0,
        )
        assert not eval_mars.applicable
        assert eval_mars.score == 0.0
        assert any("Temperature" in factor for factor in eval_mars.limiting_factors)

        # In a warm greenhouse (298 K, 101325 Pa), it should be highly applicable
        eval_warm = mech.evaluate(
            atmosphere={"CO2": 0.05, "N2": 0.78},
            soil={"H2O": 0.05},
            temperature=298.15,
            pressure=101325.0,
        )
        assert eval_warm.applicable
        assert eval_warm.score > 0.8
        assert eval_warm.temperature_score > 0.95

    def test_add_custom_mechanism_runtime(self):
        engine = ProcessCatalogEngine()
        custom = Mechanism(
            id="custom_plasma_electrolysis",
            name="CO2 Plasma Electrolysis",
            description="CO2 -> CO + 0.5 O2 via non-thermal microwave plasma",
            process_type=ProcessType.CHEMICAL,
            domain=Domain.ATMOSPHERE,
            required_atmosphere={"CO2": 0.5},
            required_elements={"C", "O"},
            energy_per_kg_product=22.0,
            main_product="O2",
            oxygen_yield=0.36,
            min_temperature=150.0,
            max_temperature=2000.0,
            optimal_temperature=300.0,
            min_pressure=10.0,
            max_pressure=1.0e7,
            optimal_pressure=101325.0,
            tags=["oxygen", "plasma", "mars"],
        )
        engine.add_mechanism(custom)
        assert engine.get_mechanism("custom_plasma_electrolysis") is not None

        matches = engine.find_applicable(
            atmosphere={"CO2": 0.95},
            soil={},
            temperature=210.0,
            pressure=610.0,
            goal_tags=["plasma"],
        )
        assert any(m.id == "custom_plasma_electrolysis" for m, s in matches)
