"""NOArCoSession: validated updates, reactive invalidation, history, deltas, and diagnostics."""

import pytest

from noarco.core.desired_state import HabitabilityTier
from noarco.data.planets import EARTH, MARS, VENUS
from noarco.mate.core import MATEStatus
from noarco.mechanisms import NanoparticleAerosol
from noarco.session import NOArCoSession


@pytest.fixture
def s():
    return NOArCoSession.for_mars()


class TestValidatedUpdates:
    @pytest.mark.parametrize("bad", [
        {"surface_albedo": 5.0}, {"surface_albedo": -0.1}, {"surface_pressure_pa": -1.0},
        {"surface_pressure_pa": 0.0}, {"gravity_ms2": 0.0}, {"mean_temperature_k": -5.0},
    ])
    def test_out_of_range_values_rejected(self, s, bad):
        before = s.planet
        with pytest.raises(ValueError):
            s.update(**bad)
        assert s.planet is before and s.history() == []

    def test_unknown_field_rejected(self, s):
        with pytest.raises(ValueError, match="Unknown"):
            s.update(surface_presure_pa=1.0)  # typo must not be silently ignored

    def test_unknown_target_and_constraint_fields_rejected(self, s):
        with pytest.raises(ValueError):
            s.update_target(nonsense=1)
        with pytest.raises(ValueError):
            s.update_constraints(max_power_w="not-a-number")

    def test_valid_update_applies_and_is_chainable(self, s):
        assert s.update(surface_pressure_pa=1220.0).update(mean_temperature_k=225.0) is s
        assert (s.planet.surface_pressure_pa, s.planet.mean_temperature_k) == (1220.0, 225.0)

    def test_original_planet_object_not_mutated(self, s):
        s.update(surface_pressure_pa=999.0)
        assert MARS.surface_pressure_pa == 610.0


class TestReactivity:
    def test_inventory_and_radiative_rebuilt_after_update(self, s):
        inv0, rad0 = s.inventory, s.radiative
        assert s.inventory is inv0            # memoised while state unchanged
        s.update(solar_constant_wm2=700.0)
        assert s.inventory is not inv0 and s.radiative is not rad0
        assert s.radiative.equilibrium_temperature_k() > rad0.equilibrium_temperature_k()

    def test_mass_per_pascal_tracks_radius(self, s):
        m0 = s.inventory.mass_per_pascal()
        s.update(radius_m=MARS.radius_m * 1.1)
        assert s.inventory.mass_per_pascal() == pytest.approx(m0 * 1.1**2, rel=1e-9)

    def test_history_records_old_and_new(self, s):
        s.update(surface_albedo=0.2)
        h = s.history()
        assert h[0]["fields_changed"] == ["surface_albedo"]
        assert h[0]["old_values"]["surface_albedo"] == MARS.surface_albedo
        assert h[0]["new_values"]["surface_albedo"] == 0.2

    def test_history_is_a_copy(self, s):
        s.update(surface_albedo=0.2)
        s.history().clear()
        assert len(s.history()) == 1

    def test_reset_to_new_body(self, s):
        s.reset_to(VENUS)
        assert s.planet.body_name == "Venus" and s.history()[-1] == {"reset_to": "Venus"}
        assert s.inventory.planet is VENUS

    def test_mate_stats_shape(self, s):
        _ = s.co2_feasibility
        st = s.mate_stats()
        assert {"size", "hits", "misses", "hit_rate", "history_entries"} <= set(st)

    def test_verbose_prints(self, capsys):
        v = NOArCoSession(MARS, verbose=True)
        v.update(surface_albedo=0.2)
        _ = v.inventory
        assert "State updated" in capsys.readouterr().out


class TestDeltas:
    def test_delta_properties_match_target(self, s):
        assert s.delta_temperature_k == pytest.approx(
            s.target.target_mean_temperature_k - MARS.mean_temperature_k)
        assert s.delta_pressure_pa == pytest.approx(
            s.target.min_surface_pressure_pa - MARS.surface_pressure_pa)

    def test_forcing_required_consistent_with_radiative(self, s):
        assert s.forcing_required_wm2 == pytest.approx(
            s.radiative.forcing_needed_for_surface_temperature(s.target.target_mean_temperature_k))

    def test_mass_required_positive_only_when_pressure_gap(self):
        e4 = NOArCoSession.for_mars(target_tier=HabitabilityTier.E4_BREATHABLE)
        assert e4.mass_required_kg and e4.mass_required_kg > 1e18
        ok = NOArCoSession(EARTH)
        assert ok.mass_required_kg is None or ok.mass_required_kg >= 0

    def test_co2_feasibility_full_gap_for_armstrong_target(self):
        r = NOArCoSession.for_mars(target_tier=HabitabilityTier.E3_ARMSTRONG).co2_feasibility
        assert r.status == MATEStatus.FULL_COMPUTE
        assert r.value["feasible"] is False and r.value["gap_kg"] > 0
        assert r.value["pi_accessible"] == pytest.approx(0.35, abs=0.02)        # ~20 mbar vs ~56 mbar needed
        assert r.value["pi_crustal_bound"] > 1.0

    def test_breathable_target_exceeds_even_the_crustal_bound(self):
        r = NOArCoSession.for_mars(target_tier=HabitabilityTier.E4_BREATHABLE).co2_feasibility
        assert r.status == MATEStatus.PROJECTED and r.value["feasible"] is False

    def test_co2_feasibility_projects_infeasibility_for_huge_targets(self):
        big = NOArCoSession.for_mars()
        big.update_target(min_surface_pressure_pa=2.0e5)   # beyond even the optimistic regolith bound
        r = big.co2_feasibility
        assert r.status == MATEStatus.PROJECTED and r.value["feasible"] is False
        assert r.value["ratio_optimistic"] > 1 and r.projection_basis and r.unexpanded_branches

    def test_co2_feasibility_positive_for_tiny_gap(self):
        r = NOArCoSession.for_mars().co2_feasibility
        assert r.value["feasible"] is True


class TestMechanismsAndSummary:
    def test_add_clear_chain(self, s):
        m = NanoparticleAerosol()
        assert s.add_mechanism(m) is s and s.mechanisms == [m]
        assert s.clear_mechanisms() is s and s.mechanisms == []

    def test_evaluate_mechanism_returns_mate_result(self, s):
        r = s.evaluate_mechanism(NanoparticleAerosol(), target_delta_t_k=10.0)
        assert r.value is not None

    def test_summary_mentions_body(self, s):
        assert "Mars" in s.summary()

    def test_assure_integration(self, s):
        r = s.assure(mc_samples=50)
        assert r.planet == "Mars" and r.manifest.verify()
