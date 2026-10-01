"""Probabilistic projection: calibration against analytic results, reproducibility, speed, honesty."""

import math
import time

import numpy as np
import pytest
from scipy import stats

from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.data.planets import MARS
from noarco.projection import Interval, Prior, clear_projection_cache, default_priors, project

T = HabitabilityTier
E1, E3, E4 = (DesiredState.from_tier(t) for t in (T.E1_TRIPLE_POINT, T.E3_ARMSTRONG, T.E4_BREATHABLE))
FROZEN = {  # remove every input uncertainty except the one under test
    "solar_constant_wm2": Prior(MARS.solar_constant_wm2, 1e-9, "normal"),
    "surface_albedo": Prior(MARS.surface_albedo, 1e-12, "normal"),
    "gravity_ms2": Prior(MARS.gravity_ms2, 1e-12, "normal"),
    "radius_m": Prior(MARS.radius_m, 1e-6, "normal"),
}


@pytest.fixture(autouse=True)
def _fresh():
    clear_projection_cache()


class TestCalibration:
    def test_probability_matches_the_analytic_lognormal_ratio(self):
        """Pi_mass = inventory / M_req is lognormal; P(Pi >= 1) is a normal CDF."""
        inv, sp = 1.0e17, 2.0
        p = project(MARS, E3, n=200_000, priors={**FROZEN, "accessible_inventory_kg": Prior(inv, sp)},
                    available_power_w=1e30)
        k = 4 * math.pi * MARS.radius_m**2 / MARS.gravity_ms2
        m_req = k * (6270.0 - 610.0)
        analytic = 1.0 - stats.norm.cdf((math.log(m_req) - math.log(inv)) / math.log(sp))
        assert p.probability_feasible == pytest.approx(analytic, abs=3e-3)

    def test_interval_quantiles_match_the_analytic_ones(self):
        inv, sp, conf = 5.0e16, 1.5, 0.95
        p = project(MARS, E3, n=200_000, confidence=conf, available_power_w=1e30,
                    priors={**FROZEN, "accessible_inventory_kg": Prior(inv, sp)})
        k = 4 * math.pi * MARS.radius_m**2 / MARS.gravity_ms2
        m_req = k * (6270.0 - 610.0)
        z = stats.norm.ppf(0.975)
        lo, hi = (inv / m_req) * math.exp(-z * math.log(sp)), (inv / m_req) * math.exp(z * math.log(sp))
        assert p.outputs["pi_mass"].lower == pytest.approx(lo, rel=0.02)
        assert p.outputs["pi_mass"].upper == pytest.approx(hi, rel=0.02)
        assert p.outputs["pi_mass"].median == pytest.approx(inv / m_req, rel=0.01)

    def test_empirical_coverage_of_the_reported_interval(self):
        """Across many independent seeds the fraction of true draws inside the 95 % interval is ~95 %."""
        inv, sp = 5.0e16, 1.5
        pr = {**FROZEN, "accessible_inventory_kg": Prior(inv, sp)}
        ref = project(MARS, E3, n=100_000, priors=pr, available_power_w=1e30, seed=1)
        lo, hi = ref.outputs["pi_mass"].lower, ref.outputs["pi_mass"].upper
        rng = np.random.default_rng(99)
        k = 4 * math.pi * MARS.radius_m**2 / MARS.gravity_ms2
        draws = inv * np.exp(rng.normal(0, math.log(sp), 50_000)) / (k * 5660.0)
        assert np.mean((draws >= lo) & (draws <= hi)) == pytest.approx(0.95, abs=0.01)

    def test_deterministic_outputs_with_no_uncertainty_collapse_to_the_point_value(self):
        pr = {**FROZEN, "accessible_inventory_kg": Prior(7.78e16, 1.0 + 1e-12),
              "electrolysis_efficiency": Prior(0.7, 1e-12, "normal")}
        p = project(MARS, E4, priors=pr, available_power_w=1e13)
        k = 4 * math.pi * MARS.radius_m**2 / MARS.gravity_ms2
        assert p.outputs["pi_mass"].median == pytest.approx(7.78e16 / (k * (101_325.0 - 610.0)), rel=1e-6)
        assert p.outputs["pi_mass"].relative_margin < 1e-6


class TestBehaviour:
    def test_paper_verdicts(self):
        assert project(MARS, E1).probability_feasible > 0.999
        assert project(MARS, E4).probability_feasible < 0.001
        e3 = project(MARS, E3)
        assert e3.probability_feasible < 0.2 and e3.pi_min.median == pytest.approx(0.354, abs=0.03)

    def test_more_power_and_inventory_raise_the_probability(self):
        base = project(MARS, E3)
        more = project(MARS, E3, available_inventory_kg=2.0e17)
        assert more.probability_feasible > base.probability_feasible
        assert project(MARS, E4, available_power_w=1e15).outputs["pi_power"].median > \
            project(MARS, E4, available_power_w=1e10).outputs["pi_power"].median

    def test_higher_confidence_gives_wider_intervals(self):
        a = project(MARS, E3, confidence=0.80).pi_min
        b = project(MARS, E3, confidence=0.99).pi_min
        assert b.upper > a.upper and b.lower < a.lower and b.confidence == 0.99

    def test_narrower_priors_shrink_the_margin(self):
        wide = project(MARS, E3, priors={"accessible_inventory_kg": Prior(7.78e16, 2.5)}).pi_min.relative_margin
        tight = project(MARS, E3, priors={"accessible_inventory_kg": Prior(7.78e16, 1.1)}).pi_min.relative_margin
        assert tight < wide

    def test_thermal_outputs_are_present_and_sensible(self):
        p = project(MARS, E3)
        assert 0 < p.outputs["co2_warming_k"].median < 10.0                      # paper: < 10 K for ~20 mbar
        assert p.outputs["aerosol_replenishment_kg_s"].median > 0
        assert "co2_warming_k" not in project(MARS, DesiredState.from_tier(T.E2_PROTECTED)).outputs

    def test_binding_constraint_shares_sum_to_one(self):
        s = project(MARS, E4).binding_constraint_shares
        assert sum(s.values()) == pytest.approx(1.0) and s["power"] > 0.99

    def test_statement_bands(self):
        assert "FEASIBLE" in project(MARS, E1).statement()
        assert "INFEASIBLE" in project(MARS, E4).statement()
        assert "LIKELY INFEASIBLE" in project(MARS, E3).statement()
        assert "UNCERTAIN" in project(MARS, E3, available_inventory_kg=2.2e17).statement()

    def test_notes_declare_model_form_limits(self):
        assert any("model-form" in n for n in project(MARS, E1).notes)


class TestEngineering:
    def test_reproducible_and_cached(self):
        a, b = project(MARS, E3), project(MARS, E3)
        assert a is b
        c = project(MARS, E3, use_cache=False)
        assert str(c.pi_min) == str(a.pi_min)
        assert str(project(MARS, E3, seed=7, use_cache=False).pi_min) != str(a.pi_min)

    def test_fast(self):
        project(MARS, E3)                                                         # warm-up imports
        t0 = time.perf_counter()
        for s in range(10):
            project(MARS, E4, seed=s, use_cache=False)
        assert (time.perf_counter() - t0) / 10 < 0.25

    def test_default_priors_are_documented(self):
        for k, v in default_priors(MARS).items():
            assert v.basis and v.spread > 0, k

    @pytest.mark.parametrize("kw", [{"confidence": 0.3}, {"confidence": 1.0}, {"n": 10},
                                    {"build_time_years": 0.0}, {"available_power_w": -1.0}])
    def test_validation(self, kw):
        with pytest.raises(ValueError):
            project(MARS, E3, **kw)

    def test_bad_prior(self):
        with pytest.raises(ValueError):
            project(MARS, E3, priors={"accessible_inventory_kg": Prior(1e17, 0.5)})
        with pytest.raises(ValueError):
            project(MARS, E3, priors={"accessible_inventory_kg": Prior(1e17, 1.5, kind="cauchy")})

    def test_interval_helpers(self):
        i = Interval(9.0, 10.0, 11.0, 0.95)
        assert i.relative_margin == pytest.approx(0.1) and "95%" in str(i)
        assert math.isinf(Interval(0, 0, 0, 0.95).relative_margin)

    def test_other_bodies_without_inventory_are_handled(self):
        from noarco.data.planets import MOON
        p = project(MOON, DesiredState(habitability_tier=T.CUSTOM, min_surface_pressure_pa=5000.0))
        assert p.probability_feasible == 0.0


class TestModelStructuralError:
    def test_exact_chains_have_no_structural_error(self):
        e = project(MARS, E4).model_error
        assert e is not None
        assert e.margin_model_structural["pi_mass"] == pytest.approx(e.margin_parametric["pi_mass"])
        assert e.probability_feasible == project(MARS, E4).probability_feasible

    def test_thermal_outputs_widen_with_model_error(self):
        e = project(MARS, E3).model_error
        assert e.margin_model_structural["aerosol_mass_kg"] > e.margin_parametric["aerosol_mass_kg"]
        assert e.margin_model_structural["co2_warming_k"] > e.margin_parametric["co2_warming_k"]

    def test_structural_error_can_change_the_verdict_when_a_thermal_constraint_binds(self):
        base = project(MARS, E3)
        avail = 1.25 * base.outputs["aerosol_required_delta_tau"].median       # ~25 % margin on tau_IR
        p = project(MARS, E3, available_delta_tau_ir=avail, available_inventory_kg=1e18)
        assert p.model_error.probability_feasible < p.probability_feasible     # model error lowers confidence
        assert p.outputs["pi_forcing"].median == pytest.approx(avail / p.outputs["aerosol_required_delta_tau"].median, rel=1e-6)

    def test_not_covered_is_documented(self):
        e = project(MARS, E3).model_error
        assert any("Feedbacks" in s for s in e.not_covered) and e.factors["grey_atmosphere_tau"] == 1.4

    def test_parametric_samples_are_unchanged_by_the_structural_stream(self):
        assert str(project(MARS, E3).pi_min) == str(project(MARS, E3, use_cache=False).pi_min)


class TestSensitivity:
    def test_ranks_the_inputs_that_control_pi_min(self):
        from noarco.projection import sensitivity
        rows = sensitivity(MARS, E3)
        assert rows[0].parameter == "accessible_inventory_kg" and rows[0].span_log10 > 0.3
        assert next(r for r in rows if r.parameter == "aerosol_kappa_m2_kg").span_log10 == pytest.approx(0.0, abs=1e-6)
        assert rows[0].flips_verdict is False or rows[0].flips_verdict is True

    def test_inventory_can_flip_e3_but_not_e4(self):
        from noarco.projection import sensitivity
        e3 = {r.parameter: r for r in sensitivity(MARS, E3)}
        e4 = {r.parameter: r for r in sensitivity(MARS, E4)}
        assert e3["accessible_inventory_kg"].pi_min_high > e3["accessible_inventory_kg"].pi_min_low
        assert not any(r.flips_verdict for r in e4.values())

    def test_forcing_robustness(self):
        from noarco.projection import forcing_robustness
        out = forcing_robustness(MARS, E3, available_delta_tau_ir=2.0, available_inventory_kg=1e18)
        pf = {r["requirement_error_pct"]: r["p_feasible"] for r in out["rows"]}
        assert pf[-50.0] >= pf[10.0] >= pf[100.0]
        base = project(MARS, E3, available_inventory_kg=1e18, available_delta_tau_ir=2.0, use_cache=False)
        assert out["break_even_pct"] == pytest.approx((base.outputs["pi_forcing"].median - 1.0) * 100.0)
