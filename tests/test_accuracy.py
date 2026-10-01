"""Accuracy tables by scale and world: calibration against analytic values and sanity of the trends."""

import math

import pytest
from scipy import stats

from noarco.accuracy import (
    QUALITY,
    STANDARD_ENCLOSURES,
    Enclosure,
    accuracy_for_enclosure,
    accuracy_table,
    default_exterior_spread,
    to_markdown,
)
from noarco.data.planets import EARTH, MARS, VENUS
from noarco.extratools.habitat import HabitatDimensioner, HabitatSpecification

BOX = STANDARD_ENCLOSURES[0]
DOME = STANDARD_ENCLOSURES[-1]


class TestGeometry:
    def test_box_volume_area_and_crew(self):
        e = Enclosure("x", 10.0, 3.0, 0.01)
        assert e.volume_m3 == 300.0 and e.area_m2 == pytest.approx(2 * 100 + 4 * 30) and e.crew == 1

    def test_standard_scales(self):
        vols = [e.volume_m3 for e in STANDARD_ENCLOSURES]
        assert vols == sorted(vols) and vols[0] == 1.0 and STANDARD_ENCLOSURES[-1].side_m**2 == pytest.approx(200 * 1e4)


class TestCalibration:
    def test_o2_hit_rate_equals_the_analytic_normal_probability(self):
        r = accuracy_for_enclosure(MARS, DOME, n=200_000, tolerance=0.10)
        analytic = stats.norm.cdf(1.0) - stats.norm.cdf(-1.0)           # sigma = 10 %, tolerance = 10 %
        assert r.hit_rate["o2_consumption_kg_day"] == pytest.approx(analytic, abs=0.004)

    def test_o2_margin_is_1p96_sigma(self):
        r = accuracy_for_enclosure(MARS, DOME, n=200_000, confidence=0.95)
        assert r.margin["o2_consumption_kg_day"] == pytest.approx(1.96 * 0.10, rel=0.02)

    def test_gas_mass_margin_matches_the_propagated_analytic_sigma(self):
        """sigma^2 = sigma_V^2 + sigma_P^2 + (sigma_T / T)^2 for gas = P V mu / R T."""
        r = accuracy_for_enclosure(MARS, BOX, n=200_000)
        s_v = math.sqrt(0.02**2 + (3 * 0.01 / 1.0) ** 2)
        sigma = math.sqrt(s_v**2 + 0.005**2 + (1.0 / 293.15) ** 2)
        assert r.margin["gas_mass_kg"] == pytest.approx(1.96 * sigma, rel=0.03)

    def test_nominal_values_are_the_habitat_model_values(self):
        r = accuracy_for_enclosure(EARTH, STANDARD_ENCLOSURES[1])
        rep = HabitatDimensioner.dimension(EARTH, HabitatSpecification(
            volume_m3=216.0, crew_size=r.crew, envelope_area_m2=STANDARD_ENCLOSURES[1].area_m2))
        assert r.nominal["gas_mass_kg"] == pytest.approx(rep.total_gas_mass_kg)
        assert r.nominal["continuous_power_kw"] == pytest.approx(rep.continuous_power_kw)


class TestTrends:
    def test_small_enclosures_are_less_certain_in_gas_mass_than_large(self):
        small = accuracy_for_enclosure(MARS, BOX).margin["gas_mass_kg"]
        large = accuracy_for_enclosure(MARS, DOME).margin["gas_mass_kg"]
        assert small > large

    def test_measured_inputs_beat_default_inputs_everywhere(self):
        for e in (STANDARD_ENCLOSURES[1], DOME):
            d, m = accuracy_for_enclosure(MARS, e, quality="default"), accuracy_for_enclosure(MARS, e, quality="measured")
            for k in ("leak_kg_year", "continuous_power_kw"):
                assert m.hit_rate[k] > d.hit_rate[k] and m.margin[k] < d.margin[k]

    def test_looser_tolerance_raises_hit_rate_and_higher_confidence_widens_margin(self):
        a = accuracy_for_enclosure(MARS, DOME, tolerance=0.10).hit_rate["continuous_power_kw"]
        b = accuracy_for_enclosure(MARS, DOME, tolerance=0.40).hit_rate["continuous_power_kw"]
        assert b > a
        assert accuracy_for_enclosure(MARS, DOME, confidence=0.99).margin["gas_mass_kg"] > \
            accuracy_for_enclosure(MARS, DOME, confidence=0.80).margin["gas_mass_kg"]

    def test_isothermal_venus_gives_better_thermal_accuracy_than_thin_atmosphere_mars(self):
        assert default_exterior_spread(VENUS) < default_exterior_spread(EARTH) < default_exterior_spread(MARS)

    def test_no_crew_makes_crew_outputs_not_applicable(self):
        r = accuracy_for_enclosure(MARS, BOX)
        assert r.crew == 0 and math.isnan(r.hit_rate["o2_consumption_kg_day"])
        assert 0.0 <= r.overall_hit_rate <= 1.0

    def test_any_world_description_is_accepted(self):
        r = accuracy_for_enclosure({"gravity_ms2": 1.6, "radius_m": 1.7e6, "solar_constant_wm2": 1361.0}, DOME)
        assert 0.0 <= r.overall_hit_rate <= 1.0


class TestApi:
    def test_table_and_markdown(self):
        rows = accuracy_table("Mars")
        md = to_markdown(rows, "Mars")
        assert len(rows) == len(STANDARD_ENCLOSURES) and md.count("\n| ") >= len(rows)
        assert "1 m box" in md and "200 ha dome" in md and "Hit rate" in md

    def test_reproducible(self):
        assert accuracy_for_enclosure(MARS, DOME).hit_rate == accuracy_for_enclosure(MARS, DOME).hit_rate
        assert accuracy_for_enclosure(MARS, DOME, seed=3).hit_rate != accuracy_for_enclosure(MARS, DOME).hit_rate

    @pytest.mark.parametrize("kw", [{"confidence": 0.2}, {"tolerance": 0.0}, {"n": 5}, {"quality": "perfect"}])
    def test_validation(self, kw):
        with pytest.raises(ValueError):
            accuracy_for_enclosure(MARS, DOME, **kw)

    def test_quality_presets_are_ordered(self):
        assert all(QUALITY["measured"][k] <= QUALITY["default"][k] for k in QUALITY["default"])
