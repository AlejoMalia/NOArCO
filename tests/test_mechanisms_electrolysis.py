"""Electrolysis / O2 production: Gibbs-energy stoichiometry, rates, inventories and limits."""

import math

import pytest

from noarco.core.constants import DELTA_G_H2O_J_MOL, YEAR_S, reversible_o2_energy_j_per_kg
from noarco.data.planets import MARS, MOON
from noarco.mechanisms import Electrolysis, OxygenMode

MU_O2, MU_H2O = 31.9988e-3, 18.01528e-3


class TestElectrolysisPhysics:
    def test_reversible_energy_is_14p8_mj_per_kg_o2(self):
        assert reversible_o2_energy_j_per_kg() == pytest.approx(2 * DELTA_G_H2O_J_MOL / MU_O2)
        assert reversible_o2_energy_j_per_kg() == pytest.approx(14.8e6, rel=5e-3)

    @pytest.mark.parametrize("eta", [0.5, 0.7, 0.95, 1.0])
    def test_rate_from_power_and_efficiency(self, eta):
        e = Electrolysis(available_power_w=2e9, electrolysis_efficiency=eta)
        expected = 2e9 * eta / reversible_o2_energy_j_per_kg()
        assert e._electrolysis_rate_kg_s() == pytest.approx(expected)

    def test_no_double_efficiency_penalty(self):
        # Regression: 'practical' 350 kJ/mol energy was divided by eta a second time.
        e = Electrolysis(available_power_w=1e9, electrolysis_efficiency=0.7)
        specific = 1e9 / e._electrolysis_rate_kg_s()                       # J per kg O2 actually consumed
        assert specific == pytest.approx(14.82e6 / 0.7, rel=5e-3)           # ~21.2 MJ/kg, not ~30

    def test_efficiency_above_one_is_unphysical_but_linear(self):
        # The class does not validate eta; 1.0 is the reversible limit (energy >= Gibbs minimum).
        assert Electrolysis(available_power_w=1e9, electrolysis_efficiency=1.0)._electrolysis_rate_kg_s() == \
            pytest.approx(1e9 / reversible_o2_energy_j_per_kg())

    def test_water_consumption_stoichiometry(self):
        e = Electrolysis(target_o2_partial_pressure_pa=1000.0, available_power_w=1e12)
        r = e.compute(MARS)
        o2 = e._o2_mass_needed(MARS)
        assert f"{o2 * 2 * MU_H2O / MU_O2:.2e}" in r.notes                   # 1.1248 kg H2O per kg O2


class TestInventoriesAndTimes:
    def test_o2_mass_is_hydrostatic_column_weight(self):
        e = Electrolysis(target_o2_partial_pressure_pa=21_000.0)
        expected = (21_000.0 - MARS.gas_composition.mole_fraction("O2") * MARS.surface_pressure_pa) \
            * MARS.surface_area_m2 / MARS.gravity_ms2
        assert e._o2_mass_needed(MARS) == pytest.approx(expected)

    def test_no_oxygen_needed_when_target_already_met(self):
        e = Electrolysis(target_o2_partial_pressure_pa=1.0)
        assert e._o2_mass_needed(MARS) == 0.0
        assert Electrolysis(target_o2_partial_pressure_pa=1.0).compute(MARS).notes.startswith("PEM")

    def test_time_to_target_is_mass_over_rate(self):
        e = Electrolysis(available_power_w=1e13, target_o2_partial_pressure_pa=5000.0)
        r = e.compute(MARS)
        assert r.deployment_time_years == pytest.approx(e._o2_mass_needed(MARS) / e._electrolysis_rate_kg_s() / YEAR_S)

    def test_twice_the_power_halves_the_time(self):
        a = Electrolysis(available_power_w=1e12).compute(MARS).deployment_time_years
        b = Electrolysis(available_power_w=2e12).compute(MARS).deployment_time_years
        assert b == pytest.approx(a / 2)

    def test_paper_scale_breathable_o2_energy_and_time(self):
        """Turyshev (2026): reversible minimum 1.2e25 J for 21 kPa; ~380 TW over 1000 yr."""
        e = Electrolysis(target_o2_partial_pressure_pa=21_000.0, electrolysis_efficiency=1.0, available_power_w=380e12)
        r = e.compute(MARS)
        assert r.deployment_time_years == pytest.approx(1000.0, rel=0.05)

    def test_water_feasibility_flag(self):
        poor = MARS.model_copy(update={"h2o_ice_kg": 1e10})
        assert "feasible=False" in Electrolysis().compute(poor).notes
        assert "feasible=True" in Electrolysis().compute(MARS).notes


class TestModesAndValidity:
    def test_photosynthesis_rate_scales_with_area_and_insolation(self):
        a = Electrolysis(OxygenMode.PHOTOSYNTHESIS, photosynthesis_area_m2=1e12)._photosynthesis_rate_kg_s(MARS)
        b = Electrolysis(OxygenMode.PHOTOSYNTHESIS, photosynthesis_area_m2=2e12)._photosynthesis_rate_kg_s(MARS)
        assert b == pytest.approx(2 * a)
        expected = 1.5e-3 / 86400 * (MARS.solar_constant_wm2 / 1361.0) * 1e12
        assert a == pytest.approx(expected)

    def test_combined_is_the_sum(self):
        c = Electrolysis(OxygenMode.COMBINED, available_power_w=1e12, photosynthesis_area_m2=1e12)
        assert c.compute(MARS).throughput_kg_s == pytest.approx(
            Electrolysis(available_power_w=1e12)._electrolysis_rate_kg_s()
            + Electrolysis(OxygenMode.PHOTOSYNTHESIS, photosynthesis_area_m2=1e12)._photosynthesis_rate_kg_s(MARS))

    def test_photosynthesis_is_flagged_as_gross_upper_bound(self):
        r = Electrolysis(OxygenMode.PHOTOSYNTHESIS).compute(MARS)
        assert any("upper bound" in w for w in r.warnings) and r.model_class == "engineering_estimate"
        assert r.power_w == 0.0

    def test_electrolysis_is_first_principles_and_warns_about_sinks(self):
        r = Electrolysis().compute(MARS)
        assert r.model_class == "first_principles" and any("sink" in w for w in r.warnings)
        assert r.delta_forcing_wm2 == 0.0 and r.delta_temperature_k == 0.0

    def test_not_applicable_without_water(self):
        dry = MOON.model_copy(update={"h2o_ice_kg": 0.0})
        assert not Electrolysis().is_applicable(dry)
        with pytest.raises(ValueError):
            Electrolysis().compute(dry)

    def test_infinite_time_without_power(self):
        r = Electrolysis(available_power_w=0.0).compute(MARS)
        assert math.isinf(r.deployment_time_years)
