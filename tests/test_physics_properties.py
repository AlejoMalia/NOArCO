"""Physical-law and scaling properties, checked across every body and on seeded random states."""

import numpy as np
import pytest

from noarco.analyzers.atmosphere import AtmosphereEngine
from noarco.analyzers.soil import SoilEngine
from noarco.core.planet_state import PlanetState
from noarco.data.planets import ALL_SOLAR_BODIES, MARS
from noarco.inventory.atmosphere import AtmosphericInventory
from noarco.inventory.inventory import InventoryEngine
from noarco.radiative.balance import RadiativeBalance

SIGMA = 5.670374419e-8
BODIES = sorted(ALL_SOLAR_BODIES)


class TestHydrostatics:
    def test_turyshev_mars_reference_value(self):
        # 3.89e15 kg per mbar is the published validation target.
        assert AtmosphericInventory(MARS).mass_per_mbar() == pytest.approx(3.89e15, rel=0.01)

    @pytest.mark.parametrize("name", BODIES)
    def test_pressure_mass_round_trip(self, name):
        p = ALL_SOLAR_BODIES[name]
        inv = AtmosphericInventory(p)
        assert inv.mass_per_pascal() * p.surface_pressure_pa == pytest.approx(p.atmospheric_mass_kg)
        assert inv.target_atmospheric_mass_kg(p.surface_pressure_pa) == pytest.approx(p.atmospheric_mass_kg)
        assert inv.mass_per_mbar() == pytest.approx(100 * inv.mass_per_pascal())

    def test_delta_mass_is_linear_and_antisymmetric(self):
        inv = AtmosphericInventory(MARS)
        up = inv.delta_mass_for_pressure(MARS.surface_pressure_pa + 1000)
        down = inv.delta_mass_for_pressure(MARS.surface_pressure_pa - 500)
        assert up == pytest.approx(1000 * inv.mass_per_pascal())
        assert down == pytest.approx(-500 * inv.mass_per_pascal())
        assert inv.delta_mass_for_pressure(MARS.surface_pressure_pa) == 0.0

    def test_delta_mass_independent_of_species(self):
        inv = AtmosphericInventory(MARS)
        assert inv.delta_mass_for_pressure(5000, "CO2") == inv.delta_mass_for_pressure(5000, "N2")

    def test_gap_analysis_consistency(self):
        g = AtmosphericInventory(MARS).co2_inventory_gap_kg(6000.0, include_regolith_co2_kg=1e18)
        assert g["isru_feasible"] is True and g["gap_kg"] == 0.0
        g0 = AtmosphericInventory(MARS).co2_inventory_gap_kg(6000.0)
        assert g0["gap_kg"] == pytest.approx(g0["co2_needed_kg"] - g0["co2_available_kg"])
        assert g0["gap_pa"] == pytest.approx(g0["gap_kg"] / AtmosphericInventory(MARS).mass_per_pascal())

    def test_pfc_mass_linear_in_forcing_and_inverse_in_radiative_efficiency(self):
        inv = AtmosphericInventory(MARS)
        assert inv.pfc_mass_for_forcing(20) == pytest.approx(2 * inv.pfc_mass_for_forcing(10))
        assert inv.pfc_mass_for_forcing(10, radiative_efficiency_wm2_per_ppb=0.5) == pytest.approx(
            inv.pfc_mass_for_forcing(10) / 2)


class TestRadiative:
    @pytest.mark.parametrize("name", BODIES)
    def test_equilibrium_temperature_closed_form(self, name):
        p = ALL_SOLAR_BODIES[name]
        expect = (p.solar_constant_wm2 * (1 - p.surface_albedo) / (4 * SIGMA)) ** 0.25
        assert RadiativeBalance(p).equilibrium_temperature_k() == pytest.approx(expect)
        assert RadiativeBalance(p).equilibrium_temperature_k() == pytest.approx(p.equilibrium_temperature_k)

    def test_teq_scaling_laws(self):
        rb = lambda **kw: RadiativeBalance(MARS.model_copy(update=kw)).equilibrium_temperature_k()
        assert rb(solar_constant_wm2=MARS.solar_constant_wm2 * 16) == pytest.approx(2 * rb(), rel=1e-12)
        assert rb(surface_albedo=0.0) > rb() > rb(surface_albedo=0.9)

    def test_greenhouse_limits(self):
        rb = RadiativeBalance(MARS)
        assert rb.greenhouse_temperature_gray(0.0) == pytest.approx(rb.equilibrium_temperature_k())
        ts = [rb.greenhouse_temperature_gray(t) for t in (0, 1, 2, 4, 8)]
        assert ts == sorted(ts)

    def test_default_forcing_is_exact_and_linearisation_underestimates_large_steps(self):
        rb = RadiativeBalance(MARS)
        t = rb.equilibrium_temperature_k()
        assert rb.forcing_needed_for_delta_t(60.0) == pytest.approx(rb.forcing_needed_exact(t, t + 60.0))
        small = rb.forcing_needed_for_delta_t(0.1) / rb.forcing_needed_linearized(0.1)
        big = rb.forcing_needed_for_delta_t(100.0) / rb.forcing_needed_linearized(100.0)
        assert small == pytest.approx(1.0, abs=1e-3) and big > 1.5

    def test_forcing_and_temperature_rise_are_inverse(self):
        rb = RadiativeBalance(MARS)
        for dt in (-30.0, 0.0, 5.0, 60.0):
            assert rb.temperature_rise_for_forcing(rb.forcing_needed_for_delta_t(dt)) == pytest.approx(dt, abs=1e-9)
        for dt in (-10.0, 0.0, 5.0, 60.0):
            assert rb.temperature_rise_for_greenhouse_forcing(
                rb.greenhouse_forcing_for_temperature_rise(dt)) == pytest.approx(dt, abs=1e-9)

    def test_greenhouse_route_warms_more_per_wm2_than_the_direct_route(self):
        rb = RadiativeBalance(MARS)
        assert rb.temperature_rise_for_greenhouse_forcing(40.0) > rb.temperature_rise_for_forcing(40.0) > 0
        with pytest.raises(ValueError):
            rb.temperature_rise_for_greenhouse_forcing(rb.planet.solar_constant_wm2)

    def test_surface_temperature_forcing_reduces_to_eq31_for_mars(self):
        rb = RadiativeBalance(MARS)
        assert rb.forcing_needed_for_surface_temperature(273.0) == pytest.approx(
            rb.forcing_needed_exact(rb.equilibrium_temperature_k(), 273.0 * rb.equilibrium_temperature_k() / 210.0))
        with pytest.raises(ValueError):
            rb.forcing_needed_for_surface_temperature(0.0)

    def test_feedback_factor_divides_forcing(self):
        rb = RadiativeBalance(MARS)
        assert rb.forcing_needed_for_delta_t(30, 2.0) == pytest.approx(rb.forcing_needed_for_delta_t(30) / 2)
        assert rb.forcing_needed_exact(210, 273, 4.0) == pytest.approx(rb.forcing_needed_exact(210, 273) / 4)
        with pytest.raises(ValueError):
            rb.forcing_needed_exact(210, 273, 0.0)

    def test_co2_log_forcing(self):
        rb = RadiativeBalance(MARS)
        assert rb.forcing_from_pressure_ratio(100, 200, 6.0) == pytest.approx(6.0)
        assert rb.forcing_from_pressure_ratio(100, 100) == pytest.approx(0.0)
        assert rb.forcing_from_pressure_ratio(100, 50, 6.0) == pytest.approx(-6.0)
        assert rb.forcing_from_co2_doubling(3, 6.0) == 18.0
        for bad in ((0, 1), (1, 0), (-1, 1)):
            with pytest.raises(ValueError):
                rb.forcing_from_pressure_ratio(*bad)

    def test_albedo_forcing_sign_and_linearity(self):
        rb = RadiativeBalance(MARS)
        assert rb.albedo_forcing(-0.1) > 0 > rb.albedo_forcing(0.1)
        assert rb.albedo_forcing(-0.1, 0.5) == pytest.approx(rb.albedo_forcing(-0.1) / 2)
        assert rb.albedo_forcing(-0.1) == pytest.approx(MARS.solar_constant_wm2 / 4 * 0.1)

    def test_summary(self):
        assert "RadiativeBalance: Mars" in RadiativeBalance(MARS).summary(273.0)


class TestEnginesAcrossAllBodies:
    @pytest.mark.parametrize("name", BODIES)
    def test_atmosphere_and_soil_analysis_are_finite(self, name):
        p = ALL_SOLAR_BODIES[name]
        a = AtmosphereEngine.analyze(p)
        s = SoilEngine.analyze(p)
        assert np.isfinite(a.total_atmospheric_mass_kg) and a.total_atmospheric_mass_kg >= 0
        assert s.has_solid_surface in (True, False)

    @pytest.mark.parametrize("name", BODIES)
    @pytest.mark.parametrize("volume", [2.0, 2500.0, 1.0e9])
    def test_inventory_report_is_sane(self, name, volume):
        r = InventoryEngine.evaluate(ALL_SOLAR_BODIES[name], volume_m3=volume)
        assert 0.0 <= r.feasibility_score <= 1.0
        assert r.volumetric_req.o2_mass_kg >= 0 and r.volumetric_req.total_gas_mass_kg >= 0

    def test_closed_habitat_gas_mass_follows_ideal_gas_law(self):
        r = AtmosphereEngine.compute_volume_conditioning(MARS, volume_m3=1000.0)
        assert r.mode == "closed_volume"
        # P V = m R_specific T  with R_specific of the target mixture (~287 J/kg/K for air-like)
        expected = 101_325.0 * 1000.0 / (287.0 * 293.15)
        assert r.total_gas_mass_kg == pytest.approx(expected, rel=0.05)

    def test_gas_mass_scales_linearly_with_volume_and_pressure(self):
        a = AtmosphereEngine.compute_volume_conditioning(MARS, 1000.0, target_pa=50_000.0)
        b = AtmosphereEngine.compute_volume_conditioning(MARS, 2000.0, target_pa=50_000.0)
        c = AtmosphereEngine.compute_volume_conditioning(MARS, 1000.0, target_pa=100_000.0)
        assert b.total_gas_mass_kg == pytest.approx(2 * a.total_gas_mass_kg, rel=1e-9)
        assert c.total_gas_mass_kg == pytest.approx(2 * a.total_gas_mass_kg, rel=1e-9)

    def test_mean_molar_mass_bounds(self):
        for p in ALL_SOLAR_BODIES.values():
            mu = AtmosphereEngine.get_mean_molar_mass(p.gas_composition)
            assert 0.002 < mu < 0.2


class TestRandomisedStates:
    """Seeded sweep over plausible states: invariants must hold for every draw."""

    def test_assurance_invariants_on_random_planets(self):
        from noarco.assurance import assure
        rng = np.random.default_rng(42)
        for _ in range(40):
            r = float(rng.uniform(1e6, 7e6))
            m = float(rng.uniform(1e22, 6e24))
            g = 6.6743e-11 * m / r**2
            p = PlanetState(
                body_name="R", surface_pressure_pa=float(10 ** rng.uniform(1, 6)),
                mean_temperature_k=float(rng.uniform(100, 500)),
                gas_composition={"species": {"CO2": 0.6, "N2": 0.4}},
                surface_albedo=float(rng.uniform(0, 0.9)), gravity_ms2=g, radius_m=r, mass_kg=m,
                solar_constant_wm2=float(rng.uniform(50, 2500)), ir_optical_depth=float(rng.uniform(0, 3)))
            rep = assure(p, mc_samples=0)
            assert not any(f.code.startswith("INV-") and f.severity.name == "FAIL" for f in rep.findings)
            assert rep.outputs["equilibrium_temperature"].output.nominal == pytest.approx(
                p.equilibrium_temperature_k, rel=1e-6)
