"""Any world from a list of conditions: derivations, assumptions, errors and integration."""

import math

import pytest

from noarco.bodies import ASSUMED_ROCKY_DENSITY_KG_M3, resolve_body
from noarco.core.constants import G_NEWTON, SIGMA_SB
from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.data.planets import MARS
from noarco.extratools.exoplanet import ExoplanetProfile
from noarco.projection import project
from noarco.session import NOArCoSession
from noarco.workflow import clear_cache, run_assessment

E3 = DesiredState.from_tier(HabitabilityTier.E3_ARMSTRONG)


class TestInputKinds:
    def test_planetstate_and_names_pass_through(self):
        assert resolve_body(MARS).planet is MARS
        assert resolve_body("mars").planet is MARS and resolve_body(" Moon ").planet.body_name == "Moon"
        with pytest.raises(ValueError, match="Unknown body"):
            resolve_body("Atlantis")

    def test_exoplanet_profile(self):
        r = resolve_body(ExoplanetProfile("X", 1.0, 1.0, 1.0, 1.0, albedo=0.3))
        p = r.planet
        assert p.gravity_ms2 == pytest.approx(9.82, abs=0.05) and p.solar_constant_wm2 == pytest.approx(1361, rel=0.01)
        assert "solar_constant_wm2" in r.derived and p.mean_temperature_k == pytest.approx(254.6, abs=1.5)

    def test_unsupported_type(self):
        with pytest.raises(TypeError):
            resolve_body(42)


class TestDerivations:
    def test_gravity_from_mass_and_radius(self):
        r = resolve_body({"mass_kg": 6.4171e23, "radius_m": 3.3895e6, "solar_constant_wm2": 586.2, "albedo": 0.25})
        assert r.planet.gravity_ms2 == pytest.approx(3.7279, rel=1e-3) and "gravity_ms2" in r.derived

    def test_mass_from_gravity_and_radius(self):
        r = resolve_body({"gravity_ms2": 3.72, "radius_m": 3.39e6, "solar_constant_wm2": 586.0})
        assert r.planet.mass_kg == pytest.approx(3.72 * 3.39e6**2 / G_NEWTON) and "mass_kg" in r.derived

    def test_radius_from_mass_and_density(self):
        r = resolve_body({"mass_kg": 1e20, "density_kg_m3": 2000.0, "solar_constant_wm2": 100.0})
        assert r.planet.radius_m == pytest.approx((3e20 / (4 * math.pi * 2000.0)) ** (1 / 3))

    def test_radius_from_gravity_and_density_round_trip(self):
        r = resolve_body({"gravity_ms2": 0.01, "density_kg_m3": 1500.0, "solar_constant_wm2": 10.0})
        p = r.planet
        assert G_NEWTON * p.mass_kg / p.radius_m**2 == pytest.approx(0.01)
        assert p.mass_kg / (4 / 3 * math.pi * p.radius_m**3) == pytest.approx(1500.0)

    def test_gravity_only_records_an_assumption(self):
        r = resolve_body({"gravity_ms2": 0.5, "solar_constant_wm2": 100.0})
        assert not r.is_fully_specified and any(f"{ASSUMED_ROCKY_DENSITY_KG_M3:.0f}" in a for a in r.assumptions)

    def test_insolation_from_luminosity_and_distance(self):
        r = resolve_body({"gravity_ms2": 9.8, "radius_m": 6.371e6, "luminosity_solar": 1.0, "distance_au": 1.0})
        assert r.planet.solar_constant_wm2 == pytest.approx(1361, rel=0.01)
        assert "solar_constant_wm2" in r.derived

    def test_equilibrium_temperature_default_and_vacuum_assumptions(self):
        r = resolve_body({"gravity_ms2": 1.6, "radius_m": 1.74e6, "solar_constant_wm2": 1361.0, "albedo": 0.11})
        assert r.planet.mean_temperature_k == pytest.approx((1361 * 0.89 / (4 * SIGMA_SB)) ** 0.25)
        assert any("no atmosphere" in a for a in r.assumptions)

    def test_airless_body_insolation_from_temperature(self):
        r = resolve_body({"gravity_ms2": 1.0, "radius_m": 1e6, "temperature_k": 250.0, "albedo": 0.1})
        assert r.planet.equilibrium_temperature_k == pytest.approx(250.0, rel=1e-6)

    def test_microgravity_habitat_is_clamped_and_flagged(self):
        r = resolve_body({"gravity_ms2": 1e-9, "radius_m": 100.0, "solar_constant_wm2": 1361.0, "pressure_pa": 101325.0,
                          "composition": {"N2": 0.78, "O2": 0.22}, "name": "station"})
        assert r.planet.gravity_ms2 == 1e-6 and any("microgravity" in a for a in r.assumptions)


class TestErrors:
    @pytest.mark.parametrize("cond", [
        {}, {"mass_kg": 1e22}, {"radius_m": 1e6}, {"gravity_ms2": 1.0},               # no insolation / no size
        {"gravity_ms2": -1.0, "solar_constant_wm2": 1.0}, {"gravity_ms2": 1.0, "solar_constant_wm2": 1.0, "bogus": 1},
        {"gravity_ms2": 1.0, "solar_constant_wm2": 1.0, "albedo": 2.0},
        {"mass_kg": 6e24, "radius_m": 6.4e6, "gravity_ms2": 3.0, "solar_constant_wm2": 1.0},   # inconsistent
    ])
    def test_unclosable_or_inconsistent_bodies_are_rejected(self, cond):
        with pytest.raises(ValueError):
            resolve_body(cond)


class TestIntegration:
    def test_workflow_accepts_conditions_and_reports_assumptions(self):
        clear_cache()
        r = run_assessment({"name": "my-moon", "gravity_ms2": 1.3, "radius_m": 1.5e6, "solar_constant_wm2": 50.0,
                            "pressure_pa": 200.0, "temperature_k": 120.0, "composition": {"N2": 0.9, "CH4": 0.1}}, E3)
        assert r.planet == "my-moon" and r.verdict in {"INFEASIBLE", "FEASIBLE", "UNDETERMINED"}
        assert any(f.code == "BODY-ASSUMPTION" for f in r.findings)

    def test_projection_accepts_names_and_conditions(self):
        assert str(project("Mars", E3).pi_min) == str(project(MARS, E3).pi_min)
        p = project({"gravity_ms2": 3.7, "radius_m": 3.39e6, "solar_constant_wm2": 589.0, "pressure_pa": 610.0,
                     "temperature_k": 210.0, "composition": {"CO2": 1.0}, "co2_ice_kg": 7.8e16}, E3)
        assert p.body == "custom-body" and 0.0 <= p.probability_feasible <= 1.0

    def test_session_from_conditions_and_exoplanet(self):
        s = NOArCoSession.from_conditions(ExoplanetProfile("Z", 0.7, 0.92, 0.029, 0.000553, 2566.0, 0.25))
        assert s.planet.body_name == "Z" and s.inventory.mass_per_pascal() > 0
        assert NOArCoSession.from_conditions("Mars").planet is MARS
