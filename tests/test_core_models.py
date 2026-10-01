"""PlanetState / GasComposition / DesiredState / ConstraintSet / Endpoints / StateEngine contracts."""

import math

import pytest
from pydantic import ValidationError

from noarco.core.constraints import ConstraintSet
from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.core.endpoints import MARS_ENDPOINTS
from noarco.core.planet_state import GasComposition, PlanetState, SourceType
from noarco.core.state import StateEngine
from noarco.data.planets import ALL_SOLAR_BODIES, MARS

_BASE = {
    "body_name": "T", "surface_pressure_pa": 1e5, "mean_temperature_k": 288.0,
    "gas_composition": GasComposition(species={"N2": 0.78, "O2": 0.21, "Ar": 0.01}),
    "surface_albedo": 0.3, "gravity_ms2": 9.81, "radius_m": 6.371e6, "mass_kg": 5.972e24,
    "solar_constant_wm2": 1361.0,
}


def mk(**kw):
    return PlanetState(**{**_BASE, **kw})


class TestGasComposition:
    def test_negative_fraction_rejected(self):
        with pytest.raises(ValidationError):
            GasComposition(species={"N2": -0.1, "O2": 1.1})

    def test_sum_over_tolerance_rejected(self):
        with pytest.raises(ValidationError):
            GasComposition(species={"N2": 0.9, "O2": 0.3})

    def test_small_rounding_excess_is_normalised(self):
        g = GasComposition(species={"N2": 0.6, "O2": 0.404})  # sums to 1.004
        assert sum(g.species.values()) == pytest.approx(1.0)

    def test_partial_pressures_sum_to_total(self):
        g = GasComposition(species={"CO2": 0.5, "N2": 0.5})
        assert sum(g.partial_pressure_pa(1000.0).values()) == pytest.approx(1000.0)

    def test_absent_species_is_zero(self):
        assert GasComposition(species={"N2": 1.0}).mole_fraction("O2") == 0.0


class TestPlanetStateValidation:
    @pytest.mark.parametrize("field,val", [
        ("surface_pressure_pa", 0.0), ("surface_pressure_pa", -5.0), ("mean_temperature_k", 0.0),
        ("surface_albedo", 1.01), ("surface_albedo", -0.01), ("gravity_ms2", 0.0),
        ("radius_m", -1.0), ("mass_kg", 0.0), ("solar_constant_wm2", 0.0), ("ir_optical_depth", -0.1),
    ])
    def test_invalid_fields_rejected(self, field, val):
        with pytest.raises(ValidationError):
            mk(**{field: val})

    def test_missing_required_field_rejected(self):
        d = dict(_BASE)
        d.pop("radius_m")
        with pytest.raises(ValidationError):
            PlanetState(**d)

    def test_defaults(self):
        p = mk()
        assert p.ir_optical_depth == 0.0 and p.source_type == SourceType.ESTIMATED
        assert p.co2_ice_kg is None and p.name == "T"

    def test_earth_like_derived_quantities(self):
        p = mk()
        assert p.escape_velocity_m_s == pytest.approx(11186, rel=0.01)
        assert p.equilibrium_temperature_k == pytest.approx(254.6, abs=1.0)
        assert p.atmospheric_mass_kg == pytest.approx(5.27e18, rel=0.05)
        assert p.surface_area_m2 == pytest.approx(5.1e14, rel=0.01)

    def test_greenhouse_delta_monotonic_in_tau(self):
        ds = [mk(ir_optical_depth=t).greenhouse_delta_t_k for t in (0.0, 0.5, 1.0, 5.0)]
        assert ds[0] == 0.0 and ds == sorted(ds)

    def test_partial_pressures_and_summary(self):
        p = mk()
        assert sum(p.partial_pressures_pa().values()) == pytest.approx(p.surface_pressure_pa)
        assert "PlanetState: T" in p.summary()

    def test_roundtrip_serialisation(self):
        p = MARS.model_dump(mode="json")
        assert PlanetState.model_validate(p) == MARS


class TestDatasets:
    @pytest.mark.parametrize("name", sorted(ALL_SOLAR_BODIES))
    def test_every_body_is_self_consistent(self, name):
        p = ALL_SOLAR_BODIES[name]
        assert p.body_name == name
        assert 0.95 <= sum(p.gas_composition.species.values()) <= 1.0 + 1e-9
        assert 0 <= p.surface_albedo <= 1 and p.radius_m > 1e5 and p.mass_kg > 1e19
        assert p.equilibrium_temperature_k > 0 and math.isfinite(p.escape_velocity_m_s)

    def test_mars_reference_values(self):
        assert MARS.surface_pressure_pa == 610.0 and MARS.mean_temperature_k == 210.0
        assert MARS.gravity_ms2 == pytest.approx(3.72, abs=0.01)


class TestDesiredStateAndEndpoints:
    def test_each_tier_builds(self):
        for tier in HabitabilityTier:
            if tier is HabitabilityTier.CUSTOM:
                continue
            assert DesiredState.from_tier(tier).habitability_tier == tier

    def test_tier_definitions_follow_turyshev_2026(self):
        e1 = DesiredState.from_tier(HabitabilityTier.E1_TRIPLE_POINT)
        e2 = DesiredState.from_tier(HabitabilityTier.E2_PROTECTED)
        e3 = DesiredState.from_tier(HabitabilityTier.E3_ARMSTRONG)
        e4 = DesiredState.from_tier(HabitabilityTier.E4_BREATHABLE)
        assert e1.min_surface_pressure_pa == pytest.approx(611.657) and e1.target_mean_temperature_k == pytest.approx(273.16)
        assert e2.enclosed and e2.is_regional and e2.min_surface_pressure_pa == 10_000.0
        assert e3.min_surface_pressure_pa == 6_270.0 and not e3.is_regional      # the Armstrong limit, not 19.3 kPa
        assert e4.min_o2_partial_pressure_pa == 16_000.0 and e4.max_co2_partial_pressure_pa == 500.0

    def test_global_tiers_are_ordered_and_e2_is_regional(self):
        ps = [DesiredState.from_tier(t).min_surface_pressure_pa for t in (
            HabitabilityTier.E1_TRIPLE_POINT, HabitabilityTier.E3_ARMSTRONG, HabitabilityTier.E4_BREATHABLE)]
        assert ps == sorted(ps) and ps[0] < ps[-1]

    def test_old_e2_name_is_gone_not_silently_redefined(self):
        assert not hasattr(HabitabilityTier, "E2_LIGHT_SUIT")

    def test_endpoints_match_the_published_criteria(self):
        order = [MARS_ENDPOINTS[k] for k in ("E0", "E1", "E2", "E3", "E4")]
        assert [e.label for e in order] == ["E0", "E1", "E2", "E3", "E4"]
        assert MARS_ENDPOINTS["E1"].min_pressure_pa == pytest.approx(611.657)
        assert MARS_ENDPOINTS["E2"].enclosed and not MARS_ENDPOINTS["E3"].enclosed
        assert MARS_ENDPOINTS["E3"].min_pressure_pa == pytest.approx(6270.0)
        assert all(e.min_temperature_k > 0 and e.max_co2_pa >= 0 for e in order)
        # global end states (E0, E1, E3, E4) are ordered by pressure
        glob = [MARS_ENDPOINTS[k].min_pressure_pa for k in ("E0", "E1", "E3", "E4")]
        assert glob == sorted(glob)

    def test_endpoint_is_frozen(self):
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
            MARS_ENDPOINTS["E0"].label = "X"


class TestConstraintSet:
    def test_defaults_unconstrained(self):
        c = ConstraintSet()
        assert c.max_time_years is None and not c.isru_only and c.allow_biological

    def test_summary_shows_values(self):
        s = ConstraintSet(isru_only=True, max_power_w=1e9).summary()
        assert "ISRU only" in s and "True" in s

    def test_type_validation(self):
        with pytest.raises(ValidationError):
            ConstraintSet(max_power_w="lots")


class TestStateEngine:
    @pytest.mark.parametrize("name", sorted(ALL_SOLAR_BODIES))
    def test_loads_every_body(self, name):
        st = StateEngine.load(ALL_SOLAR_BODIES[name])
        assert st.planet.body_name == name and st.atmosphere and st.soil
        assert name in st.summary()

    def test_gas_giants_have_no_solid_surface(self):
        assert StateEngine.load(ALL_SOLAR_BODIES["Jupiter"]).soil.has_solid_surface is False
        assert StateEngine.load(MARS).soil.has_solid_surface is True
