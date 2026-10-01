"""Radiative modes: simple (default) vs literature_calibrated, water-vapour feedback bounds."""

import pytest

from noarco.data.planets import MARS, MOON
from noarco.inventory.atmosphere import AtmosphericInventory
from noarco.mechanisms import CO2Mobilization
from noarco.radiative.literature import (
    MARS_CO2_WARMING_ANCHORS,
    mars_co2_warming_k,
    mars_pressure_for_warming,
)

BIG = 1e19  # kg of CO2, enough to reach ~1 bar for the curve tests


class TestLiteratureCurve:
    def test_anchors_are_reproduced(self):
        assert mars_co2_warming_k(610.0) == (0.0, 0.0, 0.0)
        assert mars_co2_warming_k(100_000.0)[1] == pytest.approx(60.0)
        assert mars_co2_warming_k(2_000.0)[2] == pytest.approx(10.0)

    def test_monotone_and_bounded(self):
        ps = [700, 1e3, 5e3, 2e4, 1e5]
        mids = [mars_co2_warming_k(p)[1] for p in ps]
        assert mids == sorted(mids)
        with pytest.raises(ValueError):
            mars_co2_warming_k(5e5)

    def test_inverse_round_trip(self):
        p = mars_pressure_for_warming(30.0)
        assert mars_co2_warming_k(p)[1] == pytest.approx(30.0, rel=1e-6)

    def test_every_anchor_has_a_class_and_source(self):
        assert all(a.source and a.klass for a in MARS_CO2_WARMING_ANCHORS)


class TestModes:
    def test_default_is_simple(self):
        assert CO2Mobilization().radiative_mode == "simple"

    def test_invalid_mode_and_feedback(self):
        with pytest.raises(ValueError):
            CO2Mobilization(radiative_mode="gcm")
        with pytest.raises(ValueError):
            CO2Mobilization(water_vapor_feedback_fraction=2.5)
        with pytest.raises(ValueError):
            CO2Mobilization(water_vapor_feedback_fraction=-0.1)

    def test_calibrated_mode_is_mars_only(self):
        with pytest.raises(ValueError):
            CO2Mobilization(radiative_mode="literature_calibrated", co2_inventory_kg=1e15).compute(MOON, 10.0)

    def test_calibrated_matches_published_curve(self):
        r = CO2Mobilization(radiative_mode="literature_calibrated", co2_inventory_kg=BIG).compute(MARS, 60.0)
        assert r.delta_temperature_k == pytest.approx(60.0, rel=0.02)
        assert r.model_class == "literature_calibrated"
        assert any("literature_calibrated" in w for w in r.warnings)

    def test_simple_mode_warns_about_absolute_dt(self):
        r = CO2Mobilization(co2_inventory_kg=BIG).compute(MARS, 20.0)
        assert any("RADIATIVE MODE simple" in w for w in r.warnings)

    def test_simple_is_lower_than_literature_at_one_bar(self):
        k = AtmosphericInventory(MARS).mass_per_pascal()
        inv = 1e5 * k
        s = CO2Mobilization(co2_inventory_kg=inv).compute(MARS, None, None)
        c = CO2Mobilization(radiative_mode="literature_calibrated", co2_inventory_kg=inv).compute(MARS)
        assert s.delta_temperature_k < 0.6 * c.delta_temperature_k

    def test_water_feedback_scales_dt(self):
        a = CO2Mobilization(co2_inventory_kg=1e18).compute(MARS)
        b = CO2Mobilization(co2_inventory_kg=1e18, water_vapor_feedback_fraction=0.5).compute(MARS)
        assert b.delta_temperature_k == pytest.approx(1.5 * a.delta_temperature_k, rel=0.02)
        assert any("f_w" in w for w in b.warnings)

    def test_mass_is_independent_of_mode_for_inventory_limited_case(self):
        a = CO2Mobilization(co2_inventory_kg=1e15).compute(MARS)
        b = CO2Mobilization(co2_inventory_kg=1e15, radiative_mode="literature_calibrated").compute(MARS)
        assert a.total_mass_kg == pytest.approx(b.total_mass_kg)


class TestVerdictConfidenceAndMetrics:
    def test_axis_confidence_labels(self):
        from noarco.verdict import axis_confidence

        s, c = axis_confidence("simple"), axis_confidence("literature_calibrated")
        assert all(s[a] == "high" for a in ("mass", "power", "throughput"))
        assert s["forcing"] == "low" and c["forcing"] == "medium"
        with pytest.raises(ValueError):
            axis_confidence("gcm")

    def test_verdict_record_carries_mode_confidence_and_warnings(self):
        from noarco.core.desired_state import DesiredState, HabitabilityTier
        from noarco.verdict import verdict

        t = DesiredState(habitability_tier=HabitabilityTier.E1_TRIPLE_POINT, target_mean_temperature_k=273.0)
        v = verdict(MARS, t, available_inventory_kg=7.78e16)
        d = v.to_dict()
        assert d["radiative_mode"] == "simple" and d["axis_confidence"]["forcing"] == "low"
        assert any("low-confidence" in w for w in d["warnings"])
        w = verdict(MARS, t, available_inventory_kg=7.78e16, radiative_mode="literature_calibrated")
        assert not any("low-confidence" in x for x in w.warnings) and w.input_hash != v.input_hash

    def test_engineering_metrics(self):
        from noarco.metrics import engineering_metrics, to_markdown

        m = engineering_metrics()
        assert m["registry_green"] and m["reproducible"] and m["missing_availability_is_undetermined"]
        assert m["anchors_passed"] == m["anchors_total"] == 22
        assert 0 < m["constants_sourced_pct"] <= 100
        assert "Engineering metric" in to_markdown()
