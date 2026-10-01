"""PathEngine: requirement-driven allocation; every number is checked against the requirement formulas."""

import pytest

from noarco.core.constants import YEAR_S, reversible_o2_energy_j_per_kg
from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.data.planets import MARS, MOON, VENUS
from noarco.engines.pathfinder import ELECTROLYSIS_EFFICIENCY, PathEngine
from noarco.engines.timeline import TimelineEngine
from noarco.feasibility import endpoint_requirements
from noarco.inventory.atmosphere import AtmosphericInventory
from noarco.mechanisms.co2_mobilization import _MARS_CO2_INVENTORY

K = AtmosphericInventory(MARS).mass_per_pascal()
T = HabitabilityTier
E1, E2, E3, E4 = (DesiredState.from_tier(t) for t in (T.E1_TRIPLE_POINT, T.E2_PROTECTED, T.E3_ARMSTRONG, T.E4_BREATHABLE))


class TestAllocation:
    def test_e1_needs_almost_no_gas_and_is_not_inventory_limited(self):
        pw = PathEngine.optimize_pathway(MARS, E1)
        assert pw.steps[0].mechanism_name.startswith("CO2Mobilization")
        assert pw.steps[0].total_mass_kg == pytest.approx(K * (611.657 - 610.0), rel=1e-6)
        assert pw.binding_constraint == "mass" and pw.pi_min > 100 and pw.overall_feasibility == 1.0
        assert not any("INFEASIBLE" in w for w in pw.warnings)

    def test_e3_is_inventory_limited_with_the_published_ratio(self):
        pw = PathEngine.optimize_pathway(MARS, E3)
        needed = K * (6270.0 - 610.0)
        assert pw.pi_min == pytest.approx(_MARS_CO2_INVENTORY["accessible_reference_kg"] / needed, rel=1e-6)
        assert 0.3 < pw.overall_feasibility < 0.4 and pw.binding_constraint == "mass"
        names = [s.mechanism_name for s in pw.steps]
        assert names[0].startswith("CO2Mobilization") and "ExogenousGasImport (required)" in names
        imp = next(s for s in pw.steps if s.mechanism_name.startswith("Exogenous"))
        assert imp.total_mass_kg == pytest.approx(needed - _MARS_CO2_INVENTORY["accessible_reference_kg"], rel=1e-6)
        assert any("INFEASIBLE" in w for w in pw.warnings)

    def test_import_energy_is_the_kinetic_benchmark(self):
        pw = PathEngine.optimize_pathway(MARS, E3, import_delta_v_km_s=5.0)
        imp = next(s for s in pw.steps if s.mechanism_name.startswith("Exogenous"))
        assert imp.energy_joules == pytest.approx(0.5 * imp.total_mass_kg * 5000.0**2)
        assert any("Kinetic" in w for w in imp.warnings)

    def test_e4_oxygen_step_uses_gibbs_work_over_efficiency(self):
        pw = PathEngine.optimize_pathway(MARS, E4)
        o2 = next(s for s in pw.steps if s.target_metric == "oxygen")
        req = endpoint_requirements(MARS, E4)
        assert o2.total_mass_kg == pytest.approx(req.o2_mass_kg)
        assert o2.energy_joules == pytest.approx(req.o2_mass_kg * reversible_o2_energy_j_per_kg() / ELECTROLYSIS_EFFICIENCY)
        assert pw.binding_constraint in {"mass", "power"} and pw.overall_feasibility < 1e-2

    def test_step_times_are_energy_over_power_and_scale_inversely(self):
        a = PathEngine.optimize_pathway(MARS, E4, available_power_w=1e10)
        b = PathEngine.optimize_pathway(MARS, E4, available_power_w=1e11)
        o2a = next(s for s in a.steps if s.target_metric == "oxygen")
        o2b = next(s for s in b.steps if s.target_metric == "oxygen")
        assert o2a.time_years == pytest.approx(o2a.energy_joules / (1e10 * YEAR_S))
        assert o2b.time_years == pytest.approx(o2a.time_years / 10)

    def test_totals_are_sums_and_phases_ordered(self):
        pw = PathEngine.optimize_pathway(MARS, E4)
        assert pw.total_energy_joules == pytest.approx(sum(s.energy_joules for s in pw.steps))
        assert pw.total_time_years == pytest.approx(sum(s.time_years for s in pw.steps))
        assert [s.phase for s in pw.steps] == list(range(1, len(pw.steps) + 1))
        assert pw.total_energy_kwh == pytest.approx(pw.total_energy_joules / 3.6e6)

    def test_warming_step_is_present_and_carries_model_warnings(self):
        pw = PathEngine.optimize_pathway(MARS, E1)
        warm = next(s for s in pw.steps if s.target_metric == "warming")
        assert warm.dose > 50.0 and warm.warnings and warm.total_mass_kg > 0

    def test_enclosed_target_has_no_global_allocation(self):
        pw = PathEngine.optimize_pathway(MARS, E2)
        assert pw.steps == [] and pw.overall_feasibility == 1.0 and any("Enclosed" in w for w in pw.warnings)

    def test_other_bodies_do_not_borrow_mars_reservoirs(self):
        t = DesiredState(habitability_tier=T.CUSTOM, min_surface_pressure_pa=5_000.0)
        pw = PathEngine.optimize_pathway(MOON, t)
        assert all(not s.mechanism_name.startswith("CO2Mobilization") for s in pw.steps)
        assert pw.pi_min == 0.0 and pw.overall_feasibility == 0.0

    def test_cooling_target_needs_no_warming(self):
        t = DesiredState(habitability_tier=T.CUSTOM, target_mean_temperature_k=150.0)
        assert PathEngine.optimize_pathway(MARS, t).steps == []

    def test_no_hidden_constants_feasibility_is_not_fixed(self):
        vals = {round(PathEngine.optimize_pathway(MARS, t).overall_feasibility, 6) for t in (E1, E3, E4)}
        assert len(vals) == 3 and 0.82 not in vals

    def test_invalid_power(self):
        with pytest.raises(ValueError):
            PathEngine.optimize_pathway(MARS, E3, available_power_w=0.0)

    def test_venus_runs_without_inventing_reservoirs(self):
        pw = PathEngine.optimize_pathway(VENUS, E1)
        assert pw.body_name == "Venus"


class TestTimelineOnPathway:
    def test_more_power_shortens_the_timeline_and_totals_match(self):
        pw = PathEngine.optimize_pathway(MARS, E3)
        slow = TimelineEngine.generate_schedule(pw, power_installed_gw=1.0)
        fast = TimelineEngine.generate_schedule(pw, power_installed_gw=100.0)
        assert fast.total_duration_years == pytest.approx(slow.total_duration_years / 100, rel=1e-2)
        assert slow.milestones[-1].energy_delivered_pct == pytest.approx(100.0, abs=0.1)
