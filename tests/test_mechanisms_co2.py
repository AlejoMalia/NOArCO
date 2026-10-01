"""CO2Mobilization: published reservoirs, mass/pressure/forcing chain, limits and projection."""

import math

import pytest

from noarco.core.constants import CO2_SUBLIMATION_J_KG, YEAR_S
from noarco.data.planets import MARS, MOON
from noarco.inventory.atmosphere import AtmosphericInventory
from noarco.mechanisms import CO2Mobilization
from noarco.mechanisms.co2_mobilization import _MARS_CO2_INVENTORY
from noarco.radiative.balance import RadiativeBalance

K = AtmosphericInventory(MARS).mass_per_pascal()
RB = RadiativeBalance(MARS)


class TestReservoirs:
    def test_reservoirs_correspond_to_the_published_pressures(self):
        assert _MARS_CO2_INVENTORY["polar_deposit_south_kg"] / K == pytest.approx(600.0, rel=0.06)       # ~6 mbar
        assert _MARS_CO2_INVENTORY["accessible_reference_kg"] / K == pytest.approx(2000.0, rel=0.02)     # ~20 mbar
        assert _MARS_CO2_INVENTORY["crustal_upper_bound_kg"] / K == pytest.approx(1.0e5, rel=0.02)       # ~1 bar

    def test_crustal_bound_is_not_an_accessible_inventory(self):
        assert _MARS_CO2_INVENTORY["crustal_upper_bound_kg"] > 10 * _MARS_CO2_INVENTORY["accessible_reference_kg"]

    def test_inventory_selection(self):
        assert CO2Mobilization()._get_inventory(MARS) == MARS.co2_ice_kg
        assert CO2Mobilization(include_regolith=True)._get_inventory(MARS) == _MARS_CO2_INVENTORY["accessible_reference_kg"]
        assert CO2Mobilization(co2_inventory_kg=1e15)._get_inventory(MARS) == 1e15
        no_ice = MARS.model_copy(update={"co2_ice_kg": None})
        assert CO2Mobilization()._get_inventory(no_ice) == _MARS_CO2_INVENTORY["polar_deposit_south_kg"]

    def test_not_applicable_without_inventory(self):
        assert not CO2Mobilization().is_applicable(MOON.model_copy(update={"co2_ice_kg": 0.0}))
        assert not CO2Mobilization(co2_inventory_kg=0.0).is_applicable(MARS)


class TestPhysicsChain:
    def test_mass_to_pressure_is_hydrostatic(self):
        m = CO2Mobilization()
        assert m._delta_pressure_from_co2(MARS, 1e16) == pytest.approx(1e16 / K)

    def test_sublimation_power(self):
        m = CO2Mobilization(heat_source_efficiency=0.5)
        assert m._power_to_sublimate(1e15, 100.0) == pytest.approx(1e15 * CO2_SUBLIMATION_J_KG / (100 * YEAR_S * 0.5))

    def test_max_warming_of_the_accessible_reservoir_is_below_10_kelvin(self):
        """Turyshev (2026), Sec. V.A: ~20 mbar CO2 yields < 10 K under present insolation."""
        r = CO2Mobilization(include_regolith=True).compute(MARS, target_delta_t_k=100.0)
        assert 0.0 < r.delta_temperature_k < 10.0

    def test_forcing_uses_the_logarithmic_law_and_exact_inversion(self):
        m = CO2Mobilization(include_regolith=True, forcing_per_doubling_wm2=6.0)
        r = m.compute(MARS)                                                   # default: deliver the maximum
        p_old = MARS.gas_composition.mole_fraction("CO2") * MARS.surface_pressure_pa
        p_new = p_old + m._get_inventory(MARS) / K
        assert r.delta_forcing_wm2 == pytest.approx(6.0 * math.log2(p_new / p_old))
        assert r.delta_temperature_k == pytest.approx(RB.temperature_rise_for_greenhouse_forcing(r.delta_forcing_wm2))

    def test_forcing_scales_with_the_doubling_coefficient(self):
        a = CO2Mobilization(forcing_per_doubling_wm2=3.0).compute(MARS).delta_forcing_wm2
        b = CO2Mobilization(forcing_per_doubling_wm2=6.0).compute(MARS).delta_forcing_wm2
        assert b == pytest.approx(2 * a)

    def test_small_target_uses_only_the_needed_mass(self):
        m = CO2Mobilization(include_regolith=True)
        r = m.compute(MARS, target_delta_t_k=1.0)
        assert r.total_mass_kg < m._get_inventory(MARS)
        assert r.delta_temperature_k == pytest.approx(1.0)
        # mass delivers exactly the pCO2 increase that the log law needs
        p_old = MARS.gas_composition.mole_fraction("CO2") * MARS.surface_pressure_pa
        p_new = p_old * 2 ** (r.delta_forcing_wm2 / 6.0)
        assert r.total_mass_kg == pytest.approx((p_new - p_old) * K)

    def test_target_forcing_is_capped_by_inventory(self):
        m = CO2Mobilization()
        capped = m.compute(MARS, target_delta_f_wm2=1e6)
        assert capped.delta_forcing_wm2 == pytest.approx(m.compute(MARS).delta_forcing_wm2)

    def test_power_and_throughput_follow_mass_and_deployment(self):
        r = CO2Mobilization().compute(MARS, target_delta_t_k=0.5, deployment_years=50.0)
        assert r.throughput_kg_s == pytest.approx(r.total_mass_kg / (50 * YEAR_S))
        assert r.power_w == pytest.approx(r.total_mass_kg * CO2_SUBLIMATION_J_KG / (50 * YEAR_S * 0.70))


class TestProjectionAndValidity:
    def test_excessive_target_is_projected_to_the_inventory_limit(self):
        m = CO2Mobilization()
        r = m.compute(MARS, target_delta_t_k=60.0)
        assert "MATE R3 PROJECTION" in r.notes and "CANNOT reach target" in r.notes
        assert r.total_mass_kg == m._get_inventory(MARS)
        assert r.delta_temperature_k < 60.0

    def test_warnings_and_class(self):
        r = CO2Mobilization().compute(MARS, target_delta_t_k=1.0)
        assert r.model_class == "engineering_estimate" and any("logarithmic" in w for w in r.warnings)
        assert r.effect_duration_years == math.inf and r.isru_fraction == 1.0

    def test_no_co2_atmosphere_gives_zero_warming(self):
        from noarco.core.planet_state import GasComposition
        n2 = MARS.model_copy(update={"gas_composition": GasComposition(species={"N2": 1.0})})
        r = CO2Mobilization().compute(n2)
        assert r.delta_temperature_k == 0.0 and r.delta_forcing_wm2 == 0.0


class TestScopeGuards:
    def test_unmodelled_mode_is_rejected_not_silently_ignored(self):
        from noarco.mechanisms import MobilisationMode
        with pytest.raises(NotImplementedError):
            CO2Mobilization(mode=MobilisationMode.ALBEDO_DARKENING)

    def test_other_bodies_never_borrow_mars_reservoirs(self):
        moon = MOON.model_copy(update={"co2_ice_kg": None})
        assert CO2Mobilization(include_regolith=True)._get_inventory(moon) == 0.0
