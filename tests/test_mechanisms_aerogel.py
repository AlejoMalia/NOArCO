"""SilicaAerogel: energy-balance closure, limiting cases and the Wordsworth et al. (2019) laboratory bound."""

import pytest

from noarco.core.constants import SIGMA_SB
from noarco.data.planets import MARS, VENUS
from noarco.mechanisms import SilicaAerogel


def base_residual(a: SilicaAerogel, planet) -> float:
    """Base energy balance residual at the solved temperature (should be ~0)."""
    t_top = a.top_temperature_k(planet)
    t_b = a._sub_aerogel_temperature(planet)
    f_b = a.insolation_factor * planet.solar_constant_wm2 * (1 - planet.surface_albedo) * a.transmittance_vis
    return (a.thermal_conductivity_w_m_k * (t_b - t_top) / a.thickness_m
            + a.transmittance_ir * SIGMA_SB * (t_b**4 - t_top**4) - f_b)


class TestPhysics:
    def test_top_equals_no_greenhouse_temperature_for_global_mean(self):
        assert SilicaAerogel().top_temperature_k(MARS) == pytest.approx(MARS.equilibrium_temperature_k, rel=1e-12)

    @pytest.mark.parametrize("d", [0.005, 0.01, 0.025, 0.03, 0.05])
    def test_base_balance_closes(self, d):
        a = SilicaAerogel(thickness_m=d)
        assert base_residual(a, MARS) == pytest.approx(0.0, abs=1e-6)

    def test_thin_layer_limit_is_conductive_formula(self):
        a = SilicaAerogel(thickness_m=1e-4, transmittance_ir=0.0)
        f_b = 0.25 * MARS.solar_constant_wm2 * (1 - MARS.surface_albedo) * a.transmittance_vis
        dt = a._sub_aerogel_temperature(MARS) - a.top_temperature_k(MARS)
        assert dt == pytest.approx(f_b * a.thickness_m / a.thermal_conductivity_w_m_k, rel=1e-9)

    def test_ir_leak_reduces_warming(self):
        opaque = SilicaAerogel(transmittance_ir=0.0).compute(MARS).delta_temperature_k
        leaky = SilicaAerogel(transmittance_ir=0.3).compute(MARS).delta_temperature_k
        assert leaky < opaque

    def test_monotone_in_thickness_insolation_and_inverse_in_conductivity(self):
        dts = [SilicaAerogel(thickness_m=d).compute(MARS).delta_temperature_k for d in (0.005, 0.01, 0.02, 0.04)]
        assert dts == sorted(dts) and dts[0] > 0
        more_k = SilicaAerogel(thermal_conductivity_w_m_k=0.03).compute(MARS).delta_temperature_k
        assert more_k < SilicaAerogel().compute(MARS).delta_temperature_k
        sunnier = MARS.model_copy(update={"solar_constant_wm2": 1.5 * MARS.solar_constant_wm2})
        assert SilicaAerogel().compute(sunnier).delta_temperature_k > SilicaAerogel().compute(MARS).delta_temperature_k

    def test_wordsworth_2019_laboratory_lower_bound(self):
        """Measured warming > 45 K for a 3 cm particle layer at 150 W/m2 (a lossy set-up)."""
        a = SilicaAerogel(thickness_m=0.03, transmittance_vis=0.5, insolation_factor=150.0 / MARS.solar_constant_wm2)
        dt = a._sub_aerogel_temperature(MARS) - a.top_temperature_k(MARS)
        assert dt > 45.0

    def test_exceeds_the_50k_needed_for_melting_at_2p5_cm(self):
        r = SilicaAerogel(thickness_m=0.025).compute(MARS)
        assert r.delta_temperature_k > 63.0                                   # 210 K -> 273 K
        assert MARS.equilibrium_temperature_k + r.delta_temperature_k > 273.16


class TestResultSemantics:
    def test_global_forcing_is_zero_and_regional_fraction_reported(self):
        r = SilicaAerogel(coverage_area_m2=1e10).compute(MARS)
        assert r.delta_forcing_wm2 == 0.0
        assert r.area_fraction == pytest.approx(1e10 / MARS.surface_area_m2)
        assert r.model_class == "screening_upper_bound" and any("UPPER BOUND" in w for w in r.warnings)

    def test_mass_energy_and_replacement(self):
        a = SilicaAerogel(thickness_m=0.02, coverage_area_m2=2e9, aerogel_density_kg_m3=120.0, lifetime_years=8.0)
        r = a.compute(MARS, deployment_years=4.0)
        assert r.total_mass_kg == pytest.approx(120.0 * 0.02 * 2e9)
        assert r.throughput_kg_s == pytest.approx(r.total_mass_kg / (4.0 * 365.25 * 86400))
        assert r.power_w * 4.0 * 365.25 * 86400 == pytest.approx(r.total_mass_kg * a.energy_per_kg_aerogel_j)
        assert r.effect_duration_years == 8.0

    def test_isru_fraction_follows_regolith_silica(self):
        rich = MARS.model_copy(update={"regolith_composition": {"SiO2": 0.45}})
        poor = MARS.model_copy(update={"regolith_composition": {"SiO2": 0.05}})
        assert SilicaAerogel().compute(rich).isru_fraction == 1.0
        assert SilicaAerogel().compute(poor).isru_fraction == pytest.approx(0.25)
        assert SilicaAerogel().compute(MARS.model_copy(update={"regolith_composition": None})).isru_fraction == 0.0

    def test_mean_pressure_below_triple_point_is_flagged(self):
        r = SilicaAerogel().compute(MARS)
        assert any("triple point" in w for w in r.warnings)
        high_p = MARS.model_copy(update={"surface_pressure_pa": 2000.0})
        assert not any("triple point" in w for w in SilicaAerogel().compute(high_p).warnings)

    def test_full_coverage_is_capped_at_one(self):
        assert SilicaAerogel(coverage_area_m2=1e30).compute(MARS).area_fraction == 1.0

    def test_not_applicable_in_the_dark(self):
        dark = MARS.model_copy(update={"solar_constant_wm2": 5.0})
        assert not SilicaAerogel().is_applicable(dark)
        with pytest.raises(ValueError):
            SilicaAerogel().compute(dark)

    @pytest.mark.parametrize("kw", [
        {"thickness_m": 0.0}, {"coverage_area_m2": -1.0}, {"aerogel_density_kg_m3": 0.0},
        {"transmittance_vis": 0.0}, {"transmittance_vis": 1.2}, {"transmittance_ir": -0.1},
        {"thermal_conductivity_w_m_k": 0.0}, {"insolation_factor": 0.0}, {"insolation_factor": 1.5},
    ])
    def test_invalid_parameters_rejected(self, kw):
        with pytest.raises(ValueError):
            SilicaAerogel(**kw)

    def test_venus_like_bright_planet_runs(self):
        assert SilicaAerogel().compute(VENUS).delta_temperature_k > 0
