"""Knowledge catalogue and soil data: mass conservation, derived yields and thermodynamic floors.

Every expectation is derived from atomic weights or thermochemistry, not copied from the code.
"""

import pytest

from noarco.analyzers.soil import SoilEngine
from noarco.core.constants import reversible_o2_energy_j_per_kg
from noarco.data.elements import (
    molar_mass_g_mol,
    oxygen_in_oxides_fraction,
    oxygen_mass_fraction,
    parse_formula,
    reaction_mass_balance,
)
from noarco.data.planets import ALL_SOLAR_BODIES, MARS
from noarco.engines.knowledge import build_default_catalog

CATALOG = build_default_catalog()
REACTIVE = {k: m for k, m in CATALOG.items() if m.stoichiometry and m.stoichiometry.reactants}


class TestFormulaEngine:
    def test_parse_and_molar_mass(self):
        assert parse_formula("Mg2SiO4") == {"Mg": 2, "Si": 1, "O": 4}
        assert molar_mass_g_mol("H2O") == pytest.approx(18.015, abs=0.005)
        assert molar_mass_g_mol("CO2") == pytest.approx(44.009, abs=0.005)
        assert molar_mass_g_mol("C6H12O6") == pytest.approx(180.156, abs=0.02)

    @pytest.mark.parametrize("bad", ["", "h2o", "Xx", "(OH)2", "H2O+"])
    def test_rejects_unsupported(self, bad):
        with pytest.raises(ValueError):
            parse_formula(bad)

    def test_oxygen_fractions(self):
        assert oxygen_mass_fraction("SiO2") == pytest.approx(0.5326, abs=1e-3)
        assert oxygen_mass_fraction("Fe2O3") == pytest.approx(0.3006, abs=1e-3)
        assert oxygen_mass_fraction("Fe") == 0.0

    def test_oxide_assay_oxygen_is_additive(self):
        comp = {"SiO2": 0.5, "Fe2O3": 0.3}
        assert oxygen_in_oxides_fraction(comp) == pytest.approx(0.5 * 0.5326 + 0.3 * 0.3006, abs=1e-3)
        assert oxygen_in_oxides_fraction({"Cl": 0.01}) == 0.0
        assert oxygen_in_oxides_fraction({"Silicates": 0.5, "H2O": 0.5}) == pytest.approx(0.5 * 0.8881, abs=1e-3)


class TestCatalogueMassConservation:
    @pytest.mark.parametrize("mid", sorted(REACTIVE))
    def test_reactions_conserve_mass(self, mid):
        r, p = reaction_mass_balance(REACTIVE[mid].stoichiometry.reactants, REACTIVE[mid].stoichiometry.products)
        assert r == pytest.approx(p, rel=2e-3), mid

    def test_declared_oxygen_yields_match_stoichiometry(self):
        expect = {
            "electrolysis_h2o": 2 * molar_mass_g_mol("H2O"),          # kg H2O per 1 mol O2 (31.999 g)
            "iron_oxide_reduction": 2 * molar_mass_g_mol("Fe2O3"),
            "perchlorate_decomposition": molar_mass_g_mol("ClO4"),
        }
        o2 = molar_mass_g_mol("O2")
        assert CATALOG["electrolysis_h2o"].oxygen_yield == pytest.approx(o2 / expect["electrolysis_h2o"], abs=2e-3)
        assert CATALOG["iron_oxide_reduction"].oxygen_yield == pytest.approx(3 * o2 / expect["iron_oxide_reduction"], abs=2e-3)
        assert CATALOG["perchlorate_decomposition"].oxygen_yield == pytest.approx(2 * o2 / expect["perchlorate_decomposition"], abs=3e-3)

    def test_declared_co2_removal_of_carbonation(self):
        co2_per_olivine = 2 * molar_mass_g_mol("CO2") / molar_mass_g_mol("Mg2SiO4")
        assert CATALOG["carbonate_mineralization"].co2_removal == pytest.approx(co2_per_olivine, abs=3e-3)

    def test_yields_and_efficiencies_are_fractions(self):
        for m in CATALOG.values():
            assert 0.0 <= m.mass_efficiency <= 1.0 and m.oxygen_yield >= 0 and m.co2_removal >= 0
            assert m.min_temperature < m.optimal_temperature <= m.max_temperature or m.optimal_temperature <= m.max_temperature


class TestThermodynamicFloors:
    def test_water_electrolysis_cannot_beat_gibbs_work(self):
        assert CATALOG["electrolysis_h2o"].energy_per_kg_product >= reversible_o2_energy_j_per_kg() / 1e6

    def test_oxide_reduction_energy_exceeds_the_formation_enthalpy_floor(self):
        """Fe2O3 -> 2 Fe + 1.5 O2 needs at least |dHf| = 824.2 kJ/mol Fe2O3 (NIST) per 48 g O2."""
        floor_mj_per_kg_o2 = 824.2e3 / 0.047996 / 1e6
        assert CATALOG["iron_oxide_reduction"].energy_per_kg_product >= floor_mj_per_kg_o2

    def test_sublimation_energy_is_the_latent_heat(self):
        assert CATALOG["co2_release"].energy_per_kg_product == pytest.approx(0.59, abs=0.02)   # NIST ~0.57-0.59 MJ/kg

    def test_soil_energies_are_consistent_with_the_catalogue(self):
        assert SoilEngine.H2O_ELECTROLYSIS_MJ_KG_O2 == pytest.approx(
            CATALOG["electrolysis_h2o"].energy_per_kg_product)
        assert SoilEngine.H2O_ELECTROLYSIS_MJ_KG_O2 >= reversible_o2_energy_j_per_kg() / 1e6


class TestSoilAnalysis:
    @pytest.mark.parametrize("name", [n for n, p in ALL_SOLAR_BODIES.items() if p.regolith_composition])
    def test_oxygen_fraction_is_exact_and_bounded(self, name):
        p = ALL_SOLAR_BODIES[name]
        a = SoilEngine.analyze(p)
        assert a.extractable_o2_from_regolith_pct / 100.0 == pytest.approx(
            oxygen_in_oxides_fraction(p.regolith_composition))
        assert 0.0 < a.extractable_o2_from_regolith_pct < 89.0            # no compound here exceeds water's 88.8 % O

    def test_mars_regolith_oxygen_is_about_40_percent(self):
        assert 35.0 < SoilEngine.analyze(MARS).extractable_o2_from_regolith_pct < 46.0

    def test_perchlorate_flag_follows_the_assay_not_the_name(self):
        assert SoilEngine.analyze(MARS).perchlorate_toxicity_present is True
        dry = MARS.model_copy(update={"regolith_composition": {"SiO2": 0.5, "Fe2O3": 0.2}})
        assert SoilEngine.analyze(dry).perchlorate_toxicity_present is False

    def test_elemental_fractions_do_not_exceed_the_oxide_mass(self):
        a = SoilEngine.analyze(MARS)
        assert sum(a.elemental_mass_fractions.values()) + a.extractable_o2_from_regolith_pct / 100.0 <= 1.0 + 1e-9
