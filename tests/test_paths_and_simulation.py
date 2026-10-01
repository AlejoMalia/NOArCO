"""PathFinder (global end-state graph + Pareto front) and the 0-D TemporalSimulator."""

import math

import numpy as np
import pytest

from noarco.core.constants import SIGMA_SB
from noarco.core.constraints import ConstraintSet
from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.data.planets import MARS
from noarco.mechanisms import (
    CO2Mobilization,
    Electrolysis,
    NanoparticleAerosol,
    OrbitalMirror,
    SilicaAerogel,
    SuperGreenhouseGas,
)
from noarco.paths.cascade import CascadeStep, PathFinder, TerraformingPath
from noarco.radiative.balance import RadiativeBalance
from noarco.sim.temporal import TemporalSimulator

E3 = DesiredState.from_tier(HabitabilityTier.E3_ARMSTRONG)
E4 = DesiredState.from_tier(HabitabilityTier.E4_BREATHABLE)
RB = RadiativeBalance(MARS)


def _mechs(with_o2=False):
    base = [NanoparticleAerosol(), OrbitalMirror(), SuperGreenhouseGas(), CO2Mobilization()]
    return base + ([Electrolysis()] if with_o2 else [])


def _path(name, t, m):
    return TerraformingPath(name=name, steps=[], total_time_years=t, total_mass_kg=m,
                            peak_power_w=1.0, mean_risk=0.1)


class TestPathFinder:
    def test_graph_is_the_global_end_state_graph_without_e2(self):
        g = PathFinder(MARS, E3, ConstraintSet(), _mechs(True)).build_transition_graph()
        assert set(g.nodes) == {"E0", "E1", "E2", "E3", "E4"}
        assert not any("E2" in e for e in g.edges)          # regional: no planetary transition
        assert ("E0", "E1") in g.edges and ("E1", "E3") in g.edges and ("E3", "E4") in g.edges

    def test_edges_carry_physical_requirements(self):
        g = PathFinder(MARS, E3, ConstraintSet(), _mechs(True)).build_transition_graph()
        for _, _, d in g.edges(data=True):
            assert d["duration_years"] > 0 and d["mass_other_kg"] + d["gas_mass_kg"] > 0 and 0 <= d["risk"] <= 1
            assert d["gas_mass_kg"] >= d["co2_used_kg"] >= 0 and d["import_mass_kg"] >= 0

    def test_edge_gas_mass_is_hydrostatic(self):
        g = PathFinder(MARS, E3, ConstraintSet(), _mechs(True)).build_transition_graph()
        k = MARS.surface_area_m2 / MARS.gravity_ms2
        assert g["E1"]["E3"]["gas_mass_kg"] == pytest.approx(k * (6270.0 - 611.657))

    def test_oxygen_edge_requires_electrolysis(self):
        assert ("E3", "E4") not in PathFinder(MARS, E4, ConstraintSet(), _mechs(False)).build_transition_graph().edges
        assert ("E3", "E4") in PathFinder(MARS, E4, ConstraintSet(), _mechs(True)).build_transition_graph().edges

    def test_shared_co2_reservoir_is_not_reused_along_a_path(self):
        pf = PathFinder(MARS, E4, ConstraintSet(), _mechs(True))
        k = MARS.surface_area_m2 / MARS.gravity_ms2
        for p in pf.find_all_paths("E0", "E4"):
            # Endpoint pressures telescope: total gas along any path E0 -> E4 is K * (101325 - 610).
            total_gas = k * (101_325.0 - 610.0)
            assert p.total_import_kg == pytest.approx(total_gas - pf._accessible_co2_kg(), rel=1e-3)

    def test_paths_are_contiguous_and_totals_add_up(self):
        paths = PathFinder(MARS, E4, ConstraintSet(), _mechs(True)).find_all_paths("E0", "E4")
        assert paths
        for p in paths:
            assert p.steps[0].from_endpoint == "E0" and p.steps[-1].to_endpoint == "E4"
            for a, b in zip(p.steps, p.steps[1:], strict=False):
                assert a.to_endpoint == b.from_endpoint
            assert p.total_time_years == pytest.approx(sum(s.duration_years for s in p.steps))
            assert p.total_mass_kg == pytest.approx(sum(s.mass_kg for s in p.steps))
            assert p.peak_power_w == max(s.power_w for s in p.steps)
            assert "E0" in p.summary()

    def test_unreachable_goal_and_empty_mechanisms(self):
        pf = PathFinder(MARS, E3, ConstraintSet(), _mechs())
        pf.build_transition_graph()
        assert pf.find_all_paths("E3", "E0") == []
        assert PathFinder(MARS, E3, ConstraintSet(), []).find_all_paths("E0", "E3") == []

    def test_constraints_prune_edges(self):
        no_import = PathFinder(MARS, E3, ConstraintSet(isru_only=True), _mechs(True)).build_transition_graph()
        assert ("E1", "E3") not in no_import.edges           # needs imported gas
        tight = PathFinder(MARS, E3, ConstraintSet(max_power_w=1.0), _mechs(True)).build_transition_graph()
        assert tight.number_of_edges() == 0

    def test_pareto_front_is_non_dominated_and_complete(self):
        pf = PathFinder(MARS, E3)
        paths = [_path("a", 10, 10), _path("b", 5, 20), _path("c", 20, 5), _path("dominated", 25, 25), _path("dup", 10, 10)]
        front = pf.find_pareto_front(paths)
        assert "dominated" not in {p.name for p in front} and {"a", "b", "c"} <= {p.name for p in front}
        for p in front:
            assert not any(q.total_time_years <= p.total_time_years and q.total_mass_kg <= p.total_mass_kg
                           and (q.total_time_years < p.total_time_years or q.total_mass_kg < p.total_mass_kg)
                           for q in paths)

    def test_pareto_front_random_property(self):
        rng = np.random.default_rng(0)
        pf = PathFinder(MARS, E3)
        for _ in range(20):
            paths = [_path(str(i), float(t), float(m)) for i, (t, m) in enumerate(rng.uniform(1, 100, size=(30, 2)))]
            front = pf.find_pareto_front(paths)
            assert front
            for p in paths:
                if p not in front:
                    assert any(q.total_time_years <= p.total_time_years and q.total_mass_kg <= p.total_mass_kg for q in front)

    def test_empty_pareto_and_step_fields(self):
        assert PathFinder(MARS, E3).find_pareto_front([]) == []
        assert {"from_endpoint", "to_endpoint", "mechanism_name", "import_mass_kg"} <= CascadeStep.__dataclass_fields__.keys()


class TestTemporalSimulator:
    """0-D energy balance: each test compares with an independent closed-form steady state."""

    def test_no_mechanisms_is_an_exact_steady_state(self):
        tr = TemporalSimulator(MARS, []).run(duration_years=50, dt_years=1.0)
        assert np.allclose(tr.mean_temperature_k, MARS.mean_temperature_k, atol=1e-6)
        assert np.allclose(tr.surface_pressure_pa, MARS.surface_pressure_pa)
        assert np.allclose(tr.forcing_wm2, 0.0, atol=1e-9)

    def test_mirror_reaches_the_exact_stefan_boltzmann_equilibrium(self):
        tr = TemporalSimulator(MARS, [OrbitalMirror()], design_delta_t_k=30.0).run(100, 1.0)
        t_e = RB.equilibrium_temperature_k()
        r = t_e / MARS.mean_temperature_k                                  # ~1 for Mars
        assert tr.mean_temperature_k[-1] == pytest.approx(MARS.mean_temperature_k + 30.0, abs=0.1)
        assert tr.forcing_wm2[-1] == pytest.approx(SIGMA_SB * ((t_e + 30.0) ** 4 - t_e**4), rel=1e-6)
        assert r == pytest.approx(1.0, abs=0.002)

    def test_aerosol_equilibrium_matches_the_eddington_mapping(self):
        a = NanoparticleAerosol()
        tr = TemporalSimulator(MARS, [a], design_delta_t_k=30.0).run(60, 1.0)
        tau0 = RB.required_ir_optical_depth(MARS.mean_temperature_k)
        tau_a = a._aerosol_optical_depth_for_delta_t(MARS, 30.0)
        t_expected = RB.eddington_surface_temperature(tau0 + tau_a)
        assert tr.mean_temperature_k[-1] == pytest.approx(t_expected, rel=1e-6)
        assert tr.forcing_wm2[-1] == pytest.approx(a.forcing_for_optical_depth(MARS, tau_a), rel=1e-6)

    def test_ramp_is_monotone_and_reaches_full_effect_after_deployment(self):
        a = NanoparticleAerosol()
        tr = TemporalSimulator(MARS, [a], design_delta_t_k=20.0).run(40, 1.0)
        assert np.all(np.diff(tr.mean_temperature_k) >= -1e-9)
        deploy = a.compute(MARS, target_delta_t_k=20.0).deployment_time_years
        idx = math.ceil(deploy)
        # at the end of the ramp the column lags the forcing by its short radiative time-scale
        assert tr.mean_temperature_k[idx] == pytest.approx(tr.mean_temperature_k[-1], abs=0.2)
        assert tr.mean_temperature_k[idx + 4] == pytest.approx(tr.mean_temperature_k[-1], abs=1e-4)
        assert tr.mean_temperature_k[1] < tr.mean_temperature_k[-1]

    def test_co2_release_conserves_mass_and_is_capped_by_inventory(self):
        m = CO2Mobilization(include_regolith=True)
        tr = TemporalSimulator(MARS, [m]).run(400, 2.0)
        inventory = m._get_inventory(MARS)
        d_p = tr.surface_pressure_pa[-1] - MARS.surface_pressure_pa
        assert d_p == pytest.approx(inventory * MARS.gravity_ms2 / MARS.surface_area_m2, rel=1e-9)
        assert np.all(np.diff(tr.surface_pressure_pa) >= -1e-9)

    def test_co2_warming_is_consistent_with_the_mechanism_estimate_and_below_10_kelvin(self):
        m = CO2Mobilization(include_regolith=True)
        tr = TemporalSimulator(MARS, [m]).run(400, 2.0)
        expected = m.compute(MARS).delta_temperature_k
        warming = tr.mean_temperature_k[-1] - tr.mean_temperature_k[0]
        assert warming == pytest.approx(expected, rel=0.05) and warming < 10.0

    def test_oxygen_accumulates_hydrostatically(self):
        e = Electrolysis(available_power_w=1e12)
        tr = TemporalSimulator(MARS, [e]).run(1000, 10.0)
        produced = e.compute(MARS).throughput_kg_s * 1000 * 365.25 * 86400
        gain = tr.o2_partial_pressure_pa[-1] - tr.o2_partial_pressure_pa[0]
        assert gain == pytest.approx(produced * MARS.gravity_ms2 / MARS.surface_area_m2, rel=1e-6)
        assert tr.surface_pressure_pa[-1] - tr.surface_pressure_pa[0] == pytest.approx(gain)

    def test_regional_aerogel_has_no_global_effect(self):
        tr = TemporalSimulator(MARS, [SilicaAerogel()]).run(30, 1.0)
        assert np.allclose(tr.mean_temperature_k, MARS.mean_temperature_k, atol=1e-6)
        assert any("regional" in n for n in tr.notes)

    def test_deeper_thermal_reservoir_delays_but_does_not_change_the_equilibrium(self):
        shallow = TemporalSimulator(MARS, [OrbitalMirror()], design_delta_t_k=10.0, thermal_depth_m=2.0).run(120, 1.0)
        deep = TemporalSimulator(MARS, [OrbitalMirror()], design_delta_t_k=10.0, thermal_depth_m=300.0).run(120, 1.0)
        assert deep.mean_temperature_k[25] < shallow.mean_temperature_k[25]
        assert deep.mean_temperature_k[-1] == pytest.approx(shallow.mean_temperature_k[-1], abs=0.05)

    def test_milestones_use_the_published_end_state_definitions(self):
        tr = TemporalSimulator(MARS, [NanoparticleAerosol(), CO2Mobilization(include_regolith=True)],
                               design_delta_t_k=63.0).run(300, 2.0)
        assert any(k.startswith("E1") for k in tr.milestones_reached)
        assert not any(k.startswith("E3") for k in tr.milestones_reached)     # 20 mbar < 62.7 mbar
        assert not any(k.startswith("E4") for k in tr.milestones_reached)

    def test_initial_conditions_and_determinism(self):
        a = TemporalSimulator(MARS, [NanoparticleAerosol()]).run(20)
        b = TemporalSimulator(MARS, [NanoparticleAerosol()]).run(20)
        assert a.time_years[0] == 0.0 and a.surface_pressure_pa[0] == MARS.surface_pressure_pa
        assert list(a.mean_temperature_k) == list(b.mean_temperature_k)
        assert "Simulated Duration" in a.summary() and a.final_state is not None

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            TemporalSimulator(MARS, [], design_delta_t_k=-1.0)
        with pytest.raises(ValueError):
            TemporalSimulator(MARS, [], thermal_depth_m=0.0)
        with pytest.raises(ValueError):
            TemporalSimulator(MARS, []).run(0.0)
        with pytest.raises(ValueError):
            TemporalSimulator(MARS, []).run(10.0, dt_years=0.0)

    def test_inapplicable_mechanism_is_reported(self):
        vac = MARS.model_copy(update={"surface_pressure_pa": 0.5})
        tr = TemporalSimulator(vac, [NanoparticleAerosol()]).run(5)
        assert any("not applicable" in n for n in tr.notes)
