"""NanoparticleAerosol: optical-depth/column-mass relations, greenhouse mapping, maintenance and limits."""

import math

import pytest

from noarco.core.constants import SIGMA_SB, YEAR_S
from noarco.data.planets import MARS
from noarco.mechanisms import AerosolMaterial, NanoparticleAerosol
from noarco.mechanisms.aerosols import IMPLIED_NANOROD_KAPPA_M2_KG, MARSWRF_AL_ROD_KAPPA_M2_KG, _g
from noarco.radiative.balance import RadiativeBalance

RB = RadiativeBalance(MARS)
AREA = MARS.surface_area_m2


class TestOpticalProperties:
    def test_sphere_mass_absorption_coefficient(self):
        a = NanoparticleAerosol(AerosolMaterial.ALUMINA)
        expected = 3 * 0.60 / (4 * 3980.0 * 0.5e-6)              # 3 Q / (4 rho r)
        assert a.mass_absorption_coefficient_m2_kg == pytest.approx(expected)

    def test_kappa_scales_inversely_with_radius(self):
        small = NanoparticleAerosol(AerosolMaterial.IRON, particle_radius_m=0.5e-6)
        big = NanoparticleAerosol(AerosolMaterial.IRON, particle_radius_m=1.0e-6)
        assert small.mass_absorption_coefficient_m2_kg == pytest.approx(2 * big.mass_absorption_coefficient_m2_kg)

    def test_nanorod_default_uses_the_marswrf_calibrated_kappa(self):
        a = NanoparticleAerosol()
        assert a.material == AerosolMaterial.NANOROD
        assert a.mass_absorption_coefficient_m2_kg == MARSWRF_AL_ROD_KAPPA_M2_KG
        turyshev = NanoparticleAerosol(AerosolMaterial.NANOROD_TURYSHEV)
        assert turyshev.mass_absorption_coefficient_m2_kg == IMPLIED_NANOROD_KAPPA_M2_KG
        assert turyshev.mass_absorption_coefficient_m2_kg / a.mass_absorption_coefficient_m2_kg > 8   # documented ~9x

    def test_overrides(self):
        a = NanoparticleAerosol(mass_absorption_coefficient_m2_kg=1e4, residence_time_years=0.5)
        assert a.mass_absorption_coefficient_m2_kg == 1e4 and a.residence_time_years == 0.5
        with pytest.raises(ValueError):
            _ = NanoparticleAerosol(mass_absorption_coefficient_m2_kg=-1.0).mass_absorption_coefficient_m2_kg


class TestGreenhouseMapping:
    @pytest.mark.parametrize("dt", [5.0, 15.0, 30.0, 60.0])
    def test_target_temperature_is_reproduced_by_the_eddington_mapping(self, dt):
        a = NanoparticleAerosol()
        r = a.compute(MARS, target_delta_t_k=dt)
        tau_total = RB.required_ir_optical_depth(MARS.mean_temperature_k) + a._aerosol_optical_depth_for_delta_t(MARS, dt)
        assert RB.eddington_surface_temperature(tau_total) == pytest.approx(MARS.mean_temperature_k + dt, rel=1e-9)
        assert r.delta_temperature_k == pytest.approx(dt)

    def test_forcing_round_trip_and_saturation(self):
        a = NanoparticleAerosol()
        for tau in (0.1, 0.5, 1.0, 3.0):
            f = a.forcing_for_optical_depth(MARS, tau)
            assert a.optical_depth_for_forcing(MARS, f) == pytest.approx(tau, rel=1e-9)
        assert a.forcing_for_optical_depth(MARS, 1e10) == pytest.approx(a.max_forcing_wm2(MARS), rel=1e-6)
        with pytest.raises(ValueError):
            a.optical_depth_for_forcing(MARS, a.max_forcing_wm2(MARS))

    def test_forcing_equals_olr_reduction_at_fixed_surface_temperature(self):
        a = NanoparticleAerosol()
        tau0 = RB.required_ir_optical_depth(MARS.mean_temperature_k)
        tau_a = 0.8
        expected = SIGMA_SB * MARS.mean_temperature_k**4 * (_g(tau0) - _g(tau0 + tau_a))
        assert a.forcing_for_optical_depth(MARS, tau_a) == pytest.approx(expected)

    def test_forcing_is_monotone_and_concave(self):
        a = NanoparticleAerosol()
        f = [a.forcing_for_optical_depth(MARS, t) for t in (0, 0.5, 1.0, 1.5, 2.0)]
        assert f[0] == 0.0 and f == sorted(f)
        assert all(f[i + 1] - f[i] >= f[i + 2] - f[i + 1] for i in range(len(f) - 2))

    def test_unreachable_forcing_target_is_flagged_not_clipped_silently(self):
        a = NanoparticleAerosol()
        r = a.compute(MARS, target_delta_f_wm2=1e6)
        assert any("UNREACHABLE" in w for w in r.warnings)
        assert r.delta_forcing_wm2 < a.max_forcing_wm2(MARS)


class TestMassAndMaintenance:
    def test_mass_is_column_times_area(self):
        a = NanoparticleAerosol()
        r = a.compute(MARS, target_delta_t_k=30.0)
        tau = a._aerosol_optical_depth_for_delta_t(MARS, 30.0)
        assert r.total_mass_kg == pytest.approx(tau / a.mass_absorption_coefficient_m2_kg * AREA)

    def test_mass_linear_in_optical_depth_and_coverage(self):
        a = NanoparticleAerosol(target_optical_depth=1.0)
        b = NanoparticleAerosol(target_optical_depth=2.0)
        half = NanoparticleAerosol(target_optical_depth=1.0, coverage_fraction=0.5)
        assert b.compute(MARS).total_mass_kg == pytest.approx(2 * a.compute(MARS).total_mass_kg)
        assert half.compute(MARS).total_mass_kg == pytest.approx(0.5 * a.compute(MARS).total_mass_kg)

    def test_replenishment_follows_eq53(self):
        a = NanoparticleAerosol(target_optical_depth=1.0, residence_time_years=0.1)
        r = a.compute(MARS, deployment_years=10.0)
        column = 1.0 / a.mass_absorption_coefficient_m2_kg
        mdot = column * AREA / (0.1 * YEAR_S)                     # Sigma * A / tau_p
        assert r.power_w == pytest.approx(mdot * a.production_energy_j_per_kg)

    def test_shorter_residence_means_more_power(self):
        slow = NanoparticleAerosol(target_optical_depth=1.0, residence_time_years=1.0).compute(MARS)
        fast = NanoparticleAerosol(target_optical_depth=1.0, residence_time_years=0.1).compute(MARS)
        assert fast.power_w == pytest.approx(10 * slow.power_w)
        assert fast.effect_duration_years == pytest.approx(0.1)

    def test_regional_coverage_reports_fraction(self):
        r = NanoparticleAerosol(target_optical_depth=1.0, coverage_fraction=0.1).compute(MARS)
        assert r.area_fraction == 0.1
        full = NanoparticleAerosol(target_optical_depth=1.0).compute(MARS)
        assert r.delta_forcing_wm2 == pytest.approx(0.1 * full.delta_forcing_wm2)


class TestMaterialsAndValidation:
    @pytest.mark.parametrize("mat", list(AerosolMaterial))
    def test_every_material_gives_finite_positive_mass(self, mat):
        r = NanoparticleAerosol(mat).compute(MARS, target_delta_t_k=10.0)
        assert math.isfinite(r.total_mass_kg) and r.total_mass_kg > 0 and 0 <= r.isru_fraction <= 1
        assert r.model_class == "reference_scaling" and r.warnings

    def test_bulk_spheres_need_orders_of_magnitude_more_mass_than_nanorods(self):
        rod = NanoparticleAerosol().compute(MARS, target_delta_t_k=30.0).total_mass_kg
        sphere = NanoparticleAerosol(AerosolMaterial.ALUMINA).compute(MARS, target_delta_t_k=30.0).total_mass_kg
        assert sphere > 100 * rod

    def test_isru_fraction_tracks_regolith_iron_and_alumina(self):
        fe = MARS.model_copy(update={"regolith_composition": {"Fe2O3": 0.36}})
        assert NanoparticleAerosol(AerosolMaterial.IRON).compute(fe, target_delta_t_k=5.0).isru_fraction == \
            pytest.approx(min(0.6 * 0.36 / 0.18, 1.0))

    def test_requires_a_target_and_an_atmosphere(self):
        with pytest.raises(ValueError):
            NanoparticleAerosol().compute(MARS)
        vacuum = MARS.model_copy(update={"surface_pressure_pa": 0.5})
        assert not NanoparticleAerosol().is_applicable(vacuum)
        with pytest.raises(ValueError):
            NanoparticleAerosol().compute(vacuum, target_delta_t_k=5.0)

    @pytest.mark.parametrize("kw", [{"coverage_fraction": 0.0}, {"coverage_fraction": 1.5}, {"target_optical_depth": -1.0}])
    def test_bad_constructor_arguments(self, kw):
        with pytest.raises(ValueError):
            NanoparticleAerosol(**kw)

    def test_negative_inputs_rejected(self):
        a = NanoparticleAerosol()
        with pytest.raises(ValueError):
            a.forcing_for_optical_depth(MARS, -0.1)
        with pytest.raises(ValueError):
            a.optical_depth_for_forcing(MARS, -1.0)
