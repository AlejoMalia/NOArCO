"""End-to-end assessment: verdicts follow the rubric, projection skips work, cache hits are free."""

import pytest

from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.data.planets import MARS
from noarco.mate.core import MATEStatus
from noarco.workflow import clear_cache, run_assessment

T = HabitabilityTier


@pytest.fixture(autouse=True)
def _fresh():
    clear_cache()


def test_e1_is_feasible_and_fully_computed():
    r = run_assessment(MARS, DesiredState.from_tier(T.E1_TRIPLE_POINT))
    assert r.verdict == "FEASIBLE" and r.status == MATEStatus.FULL_COMPUTE and r.pathway is not None
    assert r.paths and r.pareto and "Verdict: FEASIBLE" in r.to_markdown()


def test_e4_is_projected_infeasible_and_skips_the_path_graph():
    r = run_assessment(MARS, DesiredState.from_tier(T.E4_BREATHABLE))
    assert r.verdict == "INFEASIBLE" and r.status == MATEStatus.PROJECTED
    assert r.pathway is None and r.paths == [] and "cannot change" in r.projection_basis


def test_e3_borderline_is_infeasible_but_computed_when_threshold_is_low():
    r = run_assessment(MARS, DesiredState.from_tier(T.E3_ARMSTRONG), projection_threshold=0.1)
    assert r.verdict == "INFEASIBLE" and r.status == MATEStatus.FULL_COMPUTE and r.pathway.binding_constraint == "mass"


def test_repeat_is_a_cache_hit_with_identical_content():
    t = DesiredState.from_tier(T.E1_TRIPLE_POINT)
    a, b = run_assessment(MARS, t), run_assessment(MARS, t)
    assert b.status == MATEStatus.CACHE_HIT and a.input_hash == b.input_hash and b.verdict == a.verdict


def test_any_input_change_invalidates_the_cache_key():
    t = DesiredState.from_tier(T.E1_TRIPLE_POINT)
    assert run_assessment(MARS, t).input_hash != run_assessment(MARS, t, build_time_years=500.0).input_hash
    assert run_assessment(MARS, t, use_cache=False).status != MATEStatus.CACHE_HIT


def test_enclosed_target_is_undetermined_without_availabilities():
    r = run_assessment(MARS, DesiredState.from_tier(T.E2_PROTECTED))
    assert r.verdict == "UNDETERMINED" and r.paths == []


def test_assessment_includes_a_probabilistic_projection_with_the_requested_confidence():
    r = run_assessment(MARS, DesiredState.from_tier(T.E3_ARMSTRONG), confidence=0.97)
    assert r.projection is not None and r.projection.confidence == 0.97
    assert "Probabilistic projection (97%" in r.to_markdown()
    assert run_assessment(MARS, DesiredState.from_tier(T.E3_ARMSTRONG), confidence=None).projection is None
