"""Extratools, pathway/timeline/impact/report engines, CLI and autodiscovery: behaviour and edge cases."""

import math

import pytest

from noarco.autodiscovery import (
    DerivabilityStatus,
    GapDetector,
    GapFiller,
    GapSeverity,
    SelfImprovingSession,
)
from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.data.planets import ALL_SOLAR_BODIES, EARTH, JUPITER, MARS, MOON, VENUS
from noarco.engines.impact import ImpactEngine
from noarco.engines.pathfinder import PathEngine
from noarco.engines.timeline import TimelineEngine
from noarco.extratools import (
    AstroAgroPlanner,
    ExoplanetClassifier,
    ExoplanetProfile,
    HabitatDimensioner,
    HabitatSpecification,
    ISRUEvaluator,
    ISRURequirement,
    ISRUSiteProfile,
    SoilConditioningSpec,
    VolatileLogisticsPlanner,
)
from noarco.extratools.cli import run_exoplanet_quick_demo, run_habitat_quick_demo
from noarco.extratools.logistics import STANDARD_SOURCES
from noarco.mechanisms import NanoparticleAerosol
from noarco.report.reporter import ReportEngine
from noarco.session import NOArCoSession

E3 = DesiredState.from_tier(HabitabilityTier.E3_ARMSTRONG)


class TestLogistics:
    def test_unknown_source_and_species_rejected(self):
        with pytest.raises(KeyError):
            VolatileLogisticsPlanner.plan_import(MARS, "N2", 1e15, "Oort")
        with pytest.raises(ValueError, match="Unsupported species"):
            VolatileLogisticsPlanner.plan_import(MARS, "Xe", 1e15)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_bad_mass_rejected(self, bad):
        with pytest.raises(ValueError):
            VolatileLogisticsPlanner.plan_import(MARS, "N2", bad)

    def test_source_without_species_rejected(self):
        with pytest.raises(ValueError, match="does not contain"):
            VolatileLogisticsPlanner.plan_import(MARS, "CO2", 1e15, "Titan")

    def test_gross_mass_accounts_for_retention(self):
        r = VolatileLogisticsPlanner.plan_import(MARS, "N2", 1e15, "Kuiper_Belt_Comets")
        src = STANDARD_SOURCES["Kuiper_Belt_Comets"]
        assert r.gross_import_mass_kg == pytest.approx(1e15 / r.atmospheric_retention_fraction / src.nitrogen_fraction)
        assert r.bodies_required_count * src.typical_body_mass_kg >= r.gross_import_mass_kg

    def test_retention_ordering_by_escape_velocity(self):
        ret = {n: VolatileLogisticsPlanner.plan_import(p, "N2", 1e15).atmospheric_retention_fraction
               for n, p in ALL_SOLAR_BODIES.items()}
        assert ret["Earth"] >= ret["Mars"] >= ret["Moon"] >= ret["Pluto"]

    def test_scales_linearly_with_required_mass(self):
        a = VolatileLogisticsPlanner.plan_import(MARS, "N2", 1e15)
        b = VolatileLogisticsPlanner.plan_import(MARS, "N2", 2e15)
        assert b.gross_import_mass_kg == pytest.approx(2 * a.gross_import_mass_kg)

    def test_massive_campaign_reports_bottleneck(self):
        r = VolatileLogisticsPlanner.plan_import(MARS, "N2", 3e18)
        assert r.bodies_required_count > 1000 and any("Massive scale" in b for b in r.logistics_bottlenecks)


class TestHabitat:
    def test_scaling_with_crew_and_volume(self):
        a = HabitatDimensioner.dimension(MARS, HabitatSpecification(volume_m3=1000, crew_size=4))
        b = HabitatDimensioner.dimension(MARS, HabitatSpecification(volume_m3=2000, crew_size=4))
        c = HabitatDimensioner.dimension(MARS, HabitatSpecification(volume_m3=1000, crew_size=8))
        assert b.total_gas_mass_kg == pytest.approx(2 * a.total_gas_mass_kg)
        assert c.daily_o2_consumption_kg == pytest.approx(2 * a.daily_o2_consumption_kg)
        assert c.daily_o2_consumption_kg == pytest.approx(8 * 0.84)

    def test_loop_closure_reduces_makeup(self):
        hi = HabitatDimensioner.dimension(MARS, HabitatSpecification(2000, 6, eclss_loop_closure_o2=0.99))
        lo = HabitatDimensioner.dimension(MARS, HabitatSpecification(2000, 6, eclss_loop_closure_o2=0.50))
        assert hi.net_daily_o2_makeup_kg < lo.net_daily_o2_makeup_kg

    @pytest.mark.parametrize("name", sorted(ALL_SOLAR_BODIES))
    def test_all_bodies_produce_finite_nonnegative_report(self, name):
        r = HabitatDimensioner.dimension(ALL_SOLAR_BODIES[name], HabitatSpecification(2500, 6))
        for f in ("total_gas_mass_kg", "o2_mass_kg", "continuous_power_kw", "hvac_cooling_power_kw",
                  "displacement_work_kwh", "leak_rate_annual_kg"):
            v = getattr(r, f)
            assert math.isfinite(v) and v >= 0, f

    def test_o2_plus_n2_not_more_than_total(self):
        r = HabitatDimensioner.dimension(EARTH, HabitatSpecification(1000, 4))
        assert r.o2_mass_kg + r.n2_mass_kg <= r.total_gas_mass_kg * 1.001


class TestISRUAndAgro:
    SITE = ISRUSiteProfile("Jezero", "Mars", 210.0, 610.0, {"Fe2O3": 0.18, "SiO2": 0.45, "H2O": 0.03})

    def test_isru_energy_and_mass_positive_and_scale_with_demand(self):
        ev = ISRUEvaluator()
        a = ev.evaluate_site(self.SITE, ISRURequirement(o2_kg_day=100, water_kg_day=0))
        b = ev.evaluate_site(self.SITE, ISRURequirement(o2_kg_day=200, water_kg_day=0))
        assert a.daily_o2_kg == pytest.approx(100) and b.daily_o2_kg == pytest.approx(200)
        assert b.daily_regolith_mined_kg == pytest.approx(2 * a.daily_regolith_mined_kg, rel=1e-6)
        assert a.continuous_power_kw > 0 and a.specific_energy_kwh_per_kg_o2 > 0

    def test_isru_rejects_bad_site_inputs(self):
        with pytest.raises(ValueError):
            ISRUSiteProfile("x", "Mars", -5.0, 610.0, {"SiO2": 0.5})
        with pytest.raises(ValueError):
            ISRUSiteProfile("x", "Mars", 210.0, 610.0, {"SiO2": 0.8, "Fe2O3": 0.5})
        with pytest.raises(ValueError):
            ISRURequirement(operating_hours_per_day=30)

    def test_site_without_viable_mechanism_raises(self):
        dead = ISRUSiteProfile("Dead", "Mars", 5.0, 0.0, {"SiO2": 0.9})
        with pytest.raises(ValueError, match="No viable ISRU"):
            ISRUEvaluator().evaluate_site(dead, ISRURequirement())

    def test_agro_mass_balance(self):
        spec = SoilConditioningSpec(greenhouse_area_m2=1000, soil_depth_m=0.4, soil_bulk_density_kg_m3=1500,
                                    perchlorate_fraction=0.005)
        r = AstroAgroPlanner.plan_soil_conversion(spec)
        assert r.total_soil_volume_m3 == pytest.approx(400)
        assert r.total_soil_mass_kg == pytest.approx(400 * 1500)
        assert r.perchlorate_mass_to_remove_kg == pytest.approx(0.005 * r.total_soil_mass_kg)
        assert 0 < r.oxygen_liberated_from_detox_kg < r.perchlorate_mass_to_remove_kg  # O2 < ClO4 mass
        assert r.recommendations

    def test_agro_zero_perchlorate(self):
        r = AstroAgroPlanner.plan_soil_conversion(SoilConditioningSpec(perchlorate_fraction=0.0))
        assert r.perchlorate_mass_to_remove_kg == 0 and r.oxygen_liberated_from_detox_kg == 0


class TestExoplanet:
    def _profile(self, **kw):
        base = {"name": "X", "mass_earth": 1.0, "radius_earth": 1.0, "semi_major_axis_au": 1.0,
                "stellar_luminosity_solar": 1.0, "albedo": 0.3}
        return ExoplanetProfile(**{**base, **kw})

    def test_earth_twin_is_habitable_and_matches_earth(self):
        r = ExoplanetClassifier.classify(self._profile())
        assert r.is_in_habitable_zone and r.surface_gravity_g == pytest.approx(1.0, rel=1e-2)
        assert r.equilibrium_temperature_k == pytest.approx(254.0, abs=2.0)
        assert r.escape_velocity_km_s == pytest.approx(11.19, abs=0.1)
        assert 0 <= r.tfi_score_pct <= 100

    def test_flux_inverse_square_law(self):
        near = ExoplanetClassifier.classify(self._profile(semi_major_axis_au=1.0))
        far = ExoplanetClassifier.classify(self._profile(semi_major_axis_au=2.0))
        assert far.stellar_flux_w_m2 == pytest.approx(near.stellar_flux_w_m2 / 4)
        assert far.equilibrium_temperature_k < near.equilibrium_temperature_k

    def test_hz_edges_bracket_one_au_for_sun(self):
        r = ExoplanetClassifier.classify(self._profile())
        assert r.habitable_zone_inner_au < 1.0 < r.habitable_zone_outer_au

    def test_too_hot_and_too_cold_leave_hz(self):
        assert not ExoplanetClassifier.classify(self._profile(semi_major_axis_au=0.3)).is_in_habitable_zone
        assert not ExoplanetClassifier.classify(self._profile(semi_major_axis_au=5.0)).is_in_habitable_zone

    def test_tfi_penalises_extreme_gravity_and_missing_magnetosphere(self):
        ok = ExoplanetClassifier.classify(self._profile()).tfi_score_pct
        heavy = ExoplanetClassifier.classify(self._profile(mass_earth=20, radius_earth=2.0)).tfi_score_pct
        nomag = ExoplanetClassifier.classify(self._profile(has_magnetic_field=False)).tfi_score_pct
        assert heavy < ok and nomag < ok

    def test_invalid_profiles_rejected(self):
        for kw in ({"mass_earth": 0}, {"radius_earth": -1}, {"albedo": 1.5}, {"stellar_teff_k": 0}):
            with pytest.raises(ValueError):
                self._profile(**kw)


class TestPathwayTimelineImpactReport:
    def test_pathway_is_consistent(self):
        pw = PathEngine.optimize_pathway(MARS, E3)
        assert pw.steps and pw.total_energy_joules == pytest.approx(sum(s.energy_joules for s in pw.steps))
        assert pw.total_energy_kwh == pytest.approx(pw.total_energy_joules / 3.6e6)
        assert 0 <= pw.overall_feasibility <= 1
        assert [s.phase for s in pw.steps] == sorted(s.phase for s in pw.steps)

    def test_more_power_shortens_timeline(self):
        pw = PathEngine.optimize_pathway(MARS, E3)
        slow = TimelineEngine.generate_schedule(pw, power_installed_gw=1.0)
        fast = TimelineEngine.generate_schedule(pw, power_installed_gw=100.0)
        assert fast.total_duration_years < slow.total_duration_years

    def test_growth_shortens_timeline_and_milestones_monotone(self):
        pw = PathEngine.optimize_pathway(MARS, E3)
        flat = TimelineEngine.generate_schedule(pw, annual_growth_rate=0.0)
        grow = TimelineEngine.generate_schedule(pw, annual_growth_rate=0.05)
        assert grow.total_duration_years <= flat.total_duration_years
        years = [m.year for m in flat.milestones]
        assert years == sorted(years)

    def test_impact_report_fields(self):
        r = ImpactEngine.evaluate_mechanism(NanoparticleAerosol(), MARS, dose=5.0)
        assert r.energy_kwh == pytest.approx(r.energy_joules / 3.6e6)
        assert 0 <= r.feasibility_score <= 1 and r.target_body == "Mars"
        slower = ImpactEngine.evaluate_mechanism(NanoparticleAerosol(), MARS, dose=5.0, available_power_w=1e6)
        assert slower.time_estimate_years > r.time_estimate_years

    def test_report_matrix_and_labels(self):
        reps = ReportEngine.evaluate_matrix(["Mars", "Venus"], [2.0, 2500.0])
        assert len(reps) == 4
        table = ReportEngine.build_markdown_summary_table(reps)
        assert table.count("\n") >= 5 and "Mars" in table and "Venus" in table
        assert ReportEngine.format_volume_label(2500.0)
        with pytest.raises(KeyError):
            ReportEngine.evaluate_matrix(["Atlantis"], [1.0])


class TestCli:
    def test_habitat_demo_output(self, capsys):
        run_habitat_quick_demo("mars", crew=4, volume_m3=1000)
        out = capsys.readouterr().out
        assert "SIZING REPORT" in out and "MARS" in out and "kWh" in out

    def test_unknown_planet_falls_back_to_mars(self, capsys):
        run_habitat_quick_demo("nowhere")
        assert "MARS" in capsys.readouterr().out

    def test_exoplanet_demo_output(self, capsys):
        run_exoplanet_quick_demo("TRAPPIST-1e")
        out = capsys.readouterr().out
        assert "TFI" in out and "TRAPPIST-1e" in out and "Jeans" in out


class TestAutodiscovery:
    def test_fillers_respect_validity_envelopes(self):
        f = GapFiller()
        assert f.fill_ir_optical_depth(MARS).value == pytest.approx(0.05, rel=0.05)
        assert f.fill_ir_optical_depth(VENUS) is None     # 9 MPa: linear scaling invalid
        assert f.fill_ir_optical_depth(EARTH) is None     # trace CO2: invalid
        assert f.fill_ir_optical_depth(JUPITER) is None

    def test_derived_values_carry_provenance(self):
        dv = GapFiller().fill_ir_optical_depth(MARS)
        assert dv.formula and dv.inputs_used and dv.references and dv.uncertainty_factor >= 1.0
        assert dv.source_type in ("modeled", "estimated") and "planet.ir_optical_depth" in dv.summary()

    def test_fill_all_returns_no_none(self):
        assert all(d is not None for d in GapFiller().fill_all_derivable(MARS))

    def test_detector_flags_missing_and_orders_by_severity(self):
        gaps = GapDetector().scan(MARS.model_copy(update={"co2_ice_kg": None, "ir_optical_depth": 0.0}))
        paths = {g.field_path for g in gaps}
        assert {"planet.co2_ice_kg", "planet.ir_optical_depth"} <= paths
        order = {GapSeverity.CRITICAL: 0, GapSeverity.SIGNIFICANT: 1, GapSeverity.MINOR: 2, GapSeverity.COSMETIC: 3}
        sev = [order[g.severity] for g in gaps]
        assert sev == sorted(sev)

    def test_mechanism_gap_depends_on_argument_not_global_state(self):
        assert any(g.field_path == "session.mechanisms" for g in GapDetector().scan(MARS))
        assert not any(g.field_path == "session.mechanisms"
                       for g in GapDetector().scan(MARS, has_mechanisms=True))
        assert not hasattr(MARS, "_mechanisms_checked")      # shared dataset must never be mutated

    def test_critical_and_derivable_views(self):
        d = GapDetector()
        assert all(g.severity == GapSeverity.CRITICAL for g in d.critical_gaps(MARS))
        assert all(g.derivability in (DerivabilityStatus.DERIVABLE, DerivabilityStatus.PARTIALLY)
                   for g in d.derivable_gaps(MARS))

    def test_self_improving_session_fills_without_touching_measured_data(self):
        partial = MARS.model_copy(update={"ir_optical_depth": 0.0, "uv_flux_wm2": None, "co2_ice_kg": None})
        s = SelfImprovingSession(NOArCoSession(partial))
        p = s.session.planet
        assert p.ir_optical_depth > 0 and p.co2_ice_kg is not None and p.uv_flux_wm2 is not None
        assert p.surface_pressure_pa == MARS.surface_pressure_pa and p.surface_albedo == MARS.surface_albedo
        assert s.derived_values() and s.session.mechanisms
        assert not s.diagnose()
        assert "AutoDiscovery Report" in s.diagnosis_report()
        assert MARS.co2_ice_kg == 2.3e16 and not hasattr(MARS, "_mechanisms_checked")

    def test_complete_state_needs_no_derivation(self):
        s = SelfImprovingSession(NOArCoSession(MARS), auto_fill=True)
        assert all(d.field_path != "planet.surface_pressure_pa" for d in s.derived_values())

    def test_force_fill_and_unknown_path(self):
        s = SelfImprovingSession(NOArCoSession(MARS.model_copy(update={"co2_ice_kg": None})), auto_fill=False)
        assert s.force_fill("planet.nonexistent") is None
        dv = s.force_fill("planet.co2_ice_kg")
        assert dv is not None and s.session.planet.co2_ice_kg == dv.value

    def test_moon_is_not_given_invented_ir_depth(self):
        assert GapFiller().fill_ir_optical_depth(MOON) is None


class TestTimelineClosedForm:
    def _pw(self):
        return PathEngine.optimize_pathway(MARS, E3)

    def test_zero_growth_equals_energy_over_power(self):
        pw = self._pw()
        sched = TimelineEngine.generate_schedule(pw, power_installed_gw=10.0, annual_growth_rate=0.0)
        expected = pw.total_energy_joules / (10e9 * 365.25 * 24 * 3600)
        assert sched.total_duration_years == pytest.approx(expected, rel=1e-3)

    def test_power_scaling_is_inverse_without_growth(self):
        pw = self._pw()
        a = TimelineEngine.generate_schedule(pw, 1.0).total_duration_years
        b = TimelineEngine.generate_schedule(pw, 10.0).total_duration_years
        assert a == pytest.approx(10 * b, rel=1e-3)

    def test_growth_matches_numerical_integral_of_power_curve(self):
        import numpy as np
        pw = self._pw()
        g, p0 = 0.03, 10.0
        sched = TimelineEngine.generate_schedule(pw, p0, g)
        t = np.linspace(0, sched.total_duration_years, 400_001)
        power_w = p0 * 1e9 * (1 + g) ** t
        delivered = float(np.sum((power_w[:-1] + power_w[1:]) / 2 * np.diff(t))) * 365.25 * 24 * 3600
        assert delivered == pytest.approx(pw.total_energy_joules, rel=1e-3)

    def test_phase_durations_sum_to_total_and_are_nonnegative(self):
        sched = TimelineEngine.generate_schedule(self._pw(), 10.0, 0.02)
        assert all(d >= 0 for d in sched.critical_phases_breakdown.values())
        assert sum(sched.critical_phases_breakdown.values()) == pytest.approx(sched.total_duration_years, abs=0.2)
        assert sched.milestones[-1].energy_delivered_pct == pytest.approx(100.0, abs=0.1)

    @pytest.mark.parametrize("kw", [{"power_installed_gw": 0.0}, {"power_installed_gw": -1.0},
                                    {"power_installed_gw": float("nan")}, {"annual_growth_rate": -1.0}])
    def test_invalid_parameters_rejected(self, kw):
        with pytest.raises(ValueError):
            TimelineEngine.generate_schedule(self._pw(), **kw)

    def test_shrinking_power_may_never_finish(self):
        sched = TimelineEngine.generate_schedule(self._pw(), 0.001, -0.5)
        assert math.isinf(sched.total_duration_years)
