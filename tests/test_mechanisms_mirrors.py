"""OrbitalMirror: geometry (Turyshev 2026 Eq. 55), exact forcing, regional mode and validation."""

import math

import pytest

from noarco.data.planets import MARS
from noarco.mechanisms import MirrorMode, OrbitalMirror
from noarco.radiative.balance import RadiativeBalance

RB = RadiativeBalance(MARS)


class TestGeometry:
    def test_area_formula_eq55(self):
        m = OrbitalMirror()
        a = m._mirror_area_for_global_forcing(MARS, 20.0)
        assert a == pytest.approx(4 * math.pi * MARS.radius_m**2 * 20.0 / (0.7 * MARS.solar_constant_wm2))

    def test_absorbed_power_closes_energy_budget(self):
        """eta * S * A_mirror == delta_F * 4 pi R^2 (power delivered equals power required)."""
        m = OrbitalMirror(reflectance=0.85, pointing_efficiency=0.8)
        a = m._mirror_area_for_global_forcing(MARS, 35.0)
        assert m.system_efficiency * MARS.solar_constant_wm2 * a == pytest.approx(35.0 * MARS.surface_area_m2)

    def test_area_linear_in_forcing_and_inverse_in_efficiency(self):
        m = OrbitalMirror()
        assert m._mirror_area_for_global_forcing(MARS, 40.0) == pytest.approx(2 * m._mirror_area_for_global_forcing(MARS, 20.0))
        better = OrbitalMirror(reflectance=0.95, pointing_efficiency=1.0)
        assert better._mirror_area_for_global_forcing(MARS, 20.0) < m._mirror_area_for_global_forcing(MARS, 20.0)

    def test_polar_mode_is_a_fraction_of_global_area_for_same_local_forcing(self):
        m = OrbitalMirror(mode=MirrorMode.POLAR, polar_area_fraction=0.02)
        g = OrbitalMirror()._mirror_area_for_global_forcing(MARS, 20.0)
        assert m._mirror_area_for_polar_forcing(MARS, 20.0) == pytest.approx(0.02 * g)


class TestCompute:
    def test_target_temperature_uses_exact_forcing_and_inverts(self):
        r = OrbitalMirror().compute(MARS, target_delta_t_k=40.0)
        assert r.delta_forcing_wm2 == pytest.approx(RB.forcing_needed_exact(RB.equilibrium_temperature_k(),
                                                                          RB.equilibrium_temperature_k() + 40.0))
        assert RB.temperature_rise_for_forcing(r.delta_forcing_wm2) == pytest.approx(40.0)

    def test_target_forcing_path(self):
        r = OrbitalMirror().compute(MARS, target_delta_f_wm2=20.0)
        assert r.delta_forcing_wm2 == 20.0 and r.delta_temperature_k == pytest.approx(
            RB.temperature_rise_for_forcing(20.0))

    def test_polar_mode_reports_global_mean_forcing_and_area_fraction(self):
        r = OrbitalMirror(mode=MirrorMode.POLAR, polar_area_fraction=0.05).compute(MARS, target_delta_f_wm2=40.0)
        assert r.area_fraction == 0.05 and r.delta_forcing_wm2 == pytest.approx(40.0 * 0.05)

    def test_mass_throughput_and_power_defaults(self):
        m = OrbitalMirror(film_thickness_m=2e-6, film_density_kg_m3=2700.0)
        r = m.compute(MARS, target_delta_f_wm2=10.0, deployment_years=40.0)
        area = m._mirror_area_for_global_forcing(MARS, 10.0)
        assert r.total_mass_kg == pytest.approx(area * 2700.0 * 2e-6)
        assert r.throughput_kg_s == pytest.approx(r.total_mass_kg / (40.0 * 365.25 * 86400))
        assert r.power_w == 0.0 and any("Station-keeping" in w for w in r.warnings)
        assert OrbitalMirror(stationkeeping_w_per_m2=1e-3).compute(MARS, target_delta_f_wm2=10.0).power_w == \
            pytest.approx(1e-3 * area)

    def test_effect_is_permanent_and_class_labelled(self):
        r = OrbitalMirror().compute(MARS, target_delta_t_k=10.0)
        assert r.effect_duration_years == math.inf and r.model_class == "reference_scaling"

    def test_isru_heuristic_from_regolith_aluminium(self):
        al_rich = MARS.model_copy(update={"regolith_composition": {"Al2O3": 0.2}})
        assert OrbitalMirror().compute(al_rich, target_delta_f_wm2=5.0).isru_fraction == 1.0
        assert OrbitalMirror().compute(MARS.model_copy(update={"regolith_composition": None}),
                                       target_delta_f_wm2=5.0).isru_fraction == 0.0


class TestValidation:
    def test_requires_a_target(self):
        with pytest.raises(ValueError):
            OrbitalMirror().compute(MARS)

    def test_negative_forcing_rejected(self):
        with pytest.raises(ValueError):
            OrbitalMirror().compute(MARS, target_delta_f_wm2=-1.0)
        with pytest.raises(ValueError):
            OrbitalMirror().compute(MARS, target_delta_t_k=-10.0)

    @pytest.mark.parametrize("kw", [{"reflectance": 0.0}, {"reflectance": 1.1}, {"pointing_efficiency": 0.0},
                                    {"polar_area_fraction": 0.0}, {"polar_area_fraction": 1.5}])
    def test_bad_constructor_arguments(self, kw):
        with pytest.raises(ValueError):
            OrbitalMirror(**kw)

    def test_not_applicable_far_from_the_sun(self):
        far = MARS.model_copy(update={"solar_constant_wm2": 10.0})
        assert not OrbitalMirror().is_applicable(far)
        with pytest.raises(ValueError):
            OrbitalMirror().compute(far, target_delta_t_k=1.0)
