"""Exoplanet habitability: Kopparapu (2013) habitable-zone fit and the Jeans escape parameter."""

import math

import pytest

from noarco.core.constants import G_NEWTON, K_BOLTZMANN
from noarco.extratools.exoplanet import (
    JEANS_RETAINED,
    KOPPARAPU_2013,
    ExoplanetClassifier,
    ExoplanetProfile,
    effective_stellar_flux,
    habitable_zone_au,
)

M_EARTH, R_EARTH, M_P = 5.972e24, 6.371e6, 1.67262e-27


def planet(**kw):
    base = {"name": "P", "mass_earth": 1.0, "radius_earth": 1.0, "semi_major_axis_au": 1.0,
            "stellar_luminosity_solar": 1.0, "stellar_teff_k": 5780.0}
    return ExoplanetProfile(**{**base, **kw})


class TestKopparapu2013:
    def test_table3_coefficients_as_published(self):
        """Values transcribed from Table 3 of Kopparapu et al. (2013), arXiv:1301.6674."""
        assert KOPPARAPU_2013["moist_greenhouse"][:3] == (1.0140, 8.1774e-5, 1.7063e-9)
        assert KOPPARAPU_2013["maximum_greenhouse"][0] == 0.3438
        assert KOPPARAPU_2013["runaway_greenhouse"][0] == 1.0512
        assert KOPPARAPU_2013["recent_venus"][0] == 1.7753 and KOPPARAPU_2013["early_mars"][0] == 0.3179

    def test_solar_constants_at_teff_5780(self):
        for limit, (s0, *_rest) in KOPPARAPU_2013.items():
            assert effective_stellar_flux(limit, 5780.0) == pytest.approx(s0)

    def test_paper_values_for_the_sun(self):
        """The paper quotes 0.99 AU (moist greenhouse) and 1.70 AU (maximum greenhouse) for the Sun."""
        assert habitable_zone_au("moist_greenhouse", 5780.0, 1.0) == pytest.approx(0.99, abs=0.01)
        assert habitable_zone_au("maximum_greenhouse", 5780.0, 1.0) == pytest.approx(1.70, abs=0.01)
        assert habitable_zone_au("runaway_greenhouse", 5780.0, 1.0) == pytest.approx(0.97, abs=0.01)

    def test_distance_scales_as_sqrt_luminosity(self):
        a = habitable_zone_au("moist_greenhouse", 5780.0, 1.0)
        assert habitable_zone_au("moist_greenhouse", 5780.0, 4.0) == pytest.approx(2 * a)

    def test_ordering_of_the_boundaries(self):
        d = {k: habitable_zone_au(k, 5780.0, 1.0) for k in KOPPARAPU_2013}
        assert d["recent_venus"] < d["runaway_greenhouse"] < d["moist_greenhouse"] < d["maximum_greenhouse"] < d["early_mars"]

    def test_cooler_stars_need_less_flux_so_the_zone_is_farther_at_fixed_luminosity(self):
        """Redder light is absorbed more strongly: S_eff falls with T_eff (Kopparapu 2013, Fig. 8)."""
        assert effective_stellar_flux("moist_greenhouse", 3500.0) < effective_stellar_flux("moist_greenhouse", 5780.0)
        assert habitable_zone_au("moist_greenhouse", 3500.0, 0.01) > habitable_zone_au("moist_greenhouse", 5780.0, 0.01)

    def test_unknown_limit_and_teff_clamping(self):
        with pytest.raises(ValueError):
            effective_stellar_flux("nonsense", 5780.0)
        assert effective_stellar_flux("moist_greenhouse", 2000.0) == effective_stellar_flux("moist_greenhouse", 2600.0)
        assert effective_stellar_flux("moist_greenhouse", 9000.0) == effective_stellar_flux("moist_greenhouse", 7200.0)


class TestClassification:
    def test_earth_twin(self):
        r = ExoplanetClassifier.classify(planet())
        assert r.is_in_habitable_zone
        assert r.habitable_zone_inner_au == pytest.approx(0.99, abs=0.01) and r.habitable_zone_outer_au == pytest.approx(1.71, abs=0.02)
        assert r.habitable_zone_optimistic_au[0] < r.habitable_zone_inner_au < r.habitable_zone_outer_au < r.habitable_zone_optimistic_au[1]

    def test_hz_membership_follows_the_distance(self):
        assert not ExoplanetClassifier.classify(planet(semi_major_axis_au=0.6)).is_in_habitable_zone
        assert not ExoplanetClassifier.classify(planet(semi_major_axis_au=2.5)).is_in_habitable_zone
        assert ExoplanetClassifier.classify(planet(semi_major_axis_au=1.5)).is_in_habitable_zone

    def test_optimistic_only_is_reported(self):
        r = ExoplanetClassifier.classify(planet(semi_major_axis_au=0.8))   # between recent Venus and moist greenhouse
        assert not r.is_in_habitable_zone
        assert any("optimistic" in d for d in r.diagnostics)

    def test_out_of_fit_range_star_is_flagged(self):
        r = ExoplanetClassifier.classify(planet(stellar_teff_k=2566.0, stellar_luminosity_solar=0.000553,
                                                semi_major_axis_au=0.029, mass_earth=0.692, radius_earth=0.92))
        assert any("outside" in d for d in r.diagnostics)

    def test_tfi_is_declared_heuristic(self):
        assert any("heuristic" in d for d in ExoplanetClassifier.classify(planet()).diagnostics)


class TestJeansParameter:
    def test_definition_matches_gmm_over_ktr(self):
        r = ExoplanetClassifier.classify(planet())
        m_n2 = 28.013 * M_P
        expected = G_NEWTON * M_EARTH * m_n2 / (K_BOLTZMANN * 288.15 * R_EARTH)
        assert r.jeans_parameters["N2"] == pytest.approx(expected, rel=1e-6)

    def test_earth_retains_n2_but_not_h2(self):
        r = ExoplanetClassifier.classify(planet())
        assert r.jeans_parameters["N2"] > JEANS_RETAINED
        assert r.jeans_parameters["H2"] < 2 * JEANS_RETAINED                # H2 is marginal/escaping on Earth
        assert r.jeans_parameters["H2"] < r.jeans_parameters["H2O"] < r.jeans_parameters["N2"] < r.jeans_parameters["CO2"]

    def test_small_world_loses_nitrogen(self):
        r = ExoplanetClassifier.classify(planet(mass_earth=0.012, radius_earth=0.27))   # Moon-like
        assert r.jeans_parameters["N2"] < JEANS_RETAINED
        assert math.isfinite(r.tfi_score_pct)

    def test_lambda_scales_with_mass_over_radius(self):
        a = ExoplanetClassifier.classify(planet(mass_earth=1.0, radius_earth=1.0)).jeans_parameters["N2"]
        b = ExoplanetClassifier.classify(planet(mass_earth=2.0, radius_earth=1.0)).jeans_parameters["N2"]
        assert b == pytest.approx(2 * a)
