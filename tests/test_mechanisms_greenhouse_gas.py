"""SuperGreenhouseGas: IPCC AR5 constants, concentration/mass bookkeeping and validity warnings."""

import pytest

from noarco.core.constants import YEAR_S
from noarco.data.planets import EARTH, MARS
from noarco.inventory.atmosphere import AtmosphericInventory
from noarco.mechanisms import PFCGas, SuperGreenhouseGas
from noarco.mechanisms.greenhouse_gas import _GAS_PARAMS, LINEAR_REGIME_LIMIT_PPB

# IPCC AR5 (2013) WG1 Table 8.A.1 (pp. 731-738), read from the published table:
# gas: (lifetime yr, radiative efficiency W m-2 ppb-1, GWP100)
AR5 = {
    "CF4": (50_000.0, 0.09, 6_630), "C2F6": (10_000.0, 0.25, 11_100),
    "SF6": (3_200.0, 0.57, 23_500), "C3F8": (2_600.0, 0.28, 8_900),
}


class TestAR5Constants:
    @pytest.mark.parametrize("gas,ref", list(AR5.items()))
    def test_parameters_match_the_published_table(self, gas, ref):
        life, re, gwp = ref
        p = _GAS_PARAMS[gas]
        assert p["lifetime_yr"] == life and p["alpha_wm2_per_ppb"] == re and p["gwp_100"] == gwp

    def test_radiative_efficiency_is_per_ppb_not_per_ppt(self):
        # Regression: values were 1000x too small (ppt mislabelled as ppb).
        assert all(0.05 < p["alpha_wm2_per_ppb"] < 1.0 for k, p in _GAS_PARAMS.items())

    def test_molar_masses(self):
        assert _GAS_PARAMS["CF4"]["molar_mass"] == pytest.approx(0.088004, rel=1e-4)
        assert _GAS_PARAMS["SF6"]["molar_mass"] == pytest.approx(0.146055, rel=1e-4)

    def test_mixed_is_the_equal_mole_blend(self):
        pure = [v for k, v in _GAS_PARAMS.items() if k != "mixed"]
        m = _GAS_PARAMS["mixed"]
        assert m["alpha_wm2_per_ppb"] == pytest.approx(sum(p["alpha_wm2_per_ppb"] for p in pure) / 4)
        assert m["lifetime_yr"] == min(p["lifetime_yr"] for p in pure)


class TestBookkeeping:
    @pytest.mark.parametrize("gas", [g for g in PFCGas if g != PFCGas.MIXED])
    def test_forcing_is_linear_in_concentration(self, gas):
        m = SuperGreenhouseGas(gas)
        assert m._forcing_for_concentration(200.0) == pytest.approx(2 * m._forcing_for_concentration(100.0))
        assert m._forcing_for_concentration(100.0) == pytest.approx(AR5[gas.value][1] * 100.0)

    def test_concentration_forcing_round_trip(self):
        m = SuperGreenhouseGas(PFCGas.SF6)
        assert m._concentration_for_forcing(m._forcing_for_concentration(37.0)) == pytest.approx(37.0)

    def test_mass_uses_the_planets_own_mean_molar_mass(self):
        m = SuperGreenhouseGas(PFCGas.C2F6)
        n_atm_mars = MARS.atmospheric_mass_kg / AtmosphericInventory(MARS)._mean_molar_mass_kg_per_mol()
        assert m._mass_for_concentration(MARS, 100.0) == pytest.approx(100e-9 * n_atm_mars * 0.138012)
        # Earth-like molar mass (~29 g/mol) gives a different mole count than Mars (~43 g/mol)
        n_atm_earth = EARTH.atmospheric_mass_kg / AtmosphericInventory(EARTH)._mean_molar_mass_kg_per_mol()
        assert m._mass_for_concentration(EARTH, 100.0) == pytest.approx(100e-9 * n_atm_earth * 0.138012)

    def test_mars_reference_numbers_for_10_wm2_of_c2f6(self):
        r = SuperGreenhouseGas(PFCGas.C2F6).compute(MARS, target_delta_f_wm2=10.0)
        assert r.total_mass_kg == pytest.approx(3.0e9, rel=0.02)              # 40 ppb on Mars' 2.4e16 kg atmosphere
        assert "40.0 ppb" in r.notes

    def test_three_ways_to_specify_the_target_agree(self):
        by_f = SuperGreenhouseGas().compute(MARS, target_delta_f_wm2=8.0)
        by_c = SuperGreenhouseGas(target_concentration_ppb=8.0 / 0.25).compute(MARS)
        assert by_c.total_mass_kg == pytest.approx(by_f.total_mass_kg)
        by_t = SuperGreenhouseGas().compute(MARS, target_delta_t_k=by_f.delta_temperature_k)
        assert by_t.total_mass_kg == pytest.approx(by_f.total_mass_kg, rel=1e-6)

    def test_replacement_and_power(self):
        m = SuperGreenhouseGas(PFCGas.C3F8, synthesis_energy_j_per_kg=2e7)
        r = m.compute(MARS, target_delta_f_wm2=5.0, deployment_years=50.0)
        assert r.power_w == pytest.approx(r.total_mass_kg * 2e7 / (50 * YEAR_S))
        assert r.effect_duration_years == 2600.0
        assert f"{r.total_mass_kg / (2600 * YEAR_S):.2e}" in r.notes

    def test_isru_fraction_from_difficulty(self):
        assert SuperGreenhouseGas(PFCGas.SF6).compute(MARS, target_delta_f_wm2=1.0).isru_fraction == pytest.approx(0.3)


class TestValidityAndErrors:
    def test_linear_regime_warning(self):
        ok = SuperGreenhouseGas(PFCGas.CF4).compute(MARS, target_delta_f_wm2=5.0)
        big = SuperGreenhouseGas(PFCGas.CF4, target_concentration_ppb=LINEAR_REGIME_LIMIT_PPB + 1).compute(MARS)
        assert not any("small-perturbation" in w for w in ok.warnings)
        assert any("small-perturbation" in w for w in big.warnings)

    def test_forcing_above_the_outgoing_longwave_radiation_is_rejected(self):
        with pytest.raises(ValueError, match="outgoing longwave"):
            SuperGreenhouseGas(target_concentration_ppb=5000.0).compute(MARS)

    def test_always_declares_engineering_estimate_and_earth_re_caveat(self):
        r = SuperGreenhouseGas().compute(MARS, target_delta_f_wm2=1.0)
        assert r.model_class == "engineering_estimate" and any("terrestrial radiative efficiency" in w for w in r.warnings)

    def test_requires_a_target(self):
        with pytest.raises(ValueError):
            SuperGreenhouseGas().compute(MARS)

    def test_not_applicable_without_atmosphere(self):
        vac = MARS.model_copy(update={"surface_pressure_pa": 0.05})
        assert not SuperGreenhouseGas().is_applicable(vac)
        with pytest.raises(ValueError):
            SuperGreenhouseGas().compute(vac, target_delta_f_wm2=1.0)

    def test_inventory_helper_matches_mechanism(self):
        assert AtmosphericInventory(MARS).pfc_mass_for_forcing(10.0) == pytest.approx(
            SuperGreenhouseGas(PFCGas.C2F6).compute(MARS, target_delta_f_wm2=10.0).total_mass_kg)
        with pytest.raises(ValueError):
            AtmosphericInventory(MARS).pfc_mass_for_forcing(-1.0)
        with pytest.raises(ValueError):
            AtmosphericInventory(MARS).pfc_mass_for_forcing(1.0, radiative_efficiency_wm2_per_ppb=0.0)
