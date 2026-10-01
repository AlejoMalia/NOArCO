"""Screening (headroom, ranking, sweeps, what-if), traceability (replay, schema, hashes) and the
integrated base budget / ISRU ledger / habitat shell / logistics delta-v."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.data.planets import EARTH, MARS, MOON, VENUS
from noarco.extratools import (
    AstroAgroPlanner,
    HabitatDimensioner,
    HabitatSpecification,
    ISRUEvaluator,
    ISRURequirement,
    ISRUSiteProfile,
    SoilConditioningSpec,
    VolatileLogisticsPlanner,
    size_base,
)
from noarco.extratools.logistics import hohmann_delta_v_km_s
from noarco.screening import Candidate, breakeven, rank, required_availability, sweep, what_if
from noarco.verdict import SCHEMA, code_hash, replay, result_schema, validate_record, verdict

ROOT = Path(__file__).resolve().parents[1]
T = DesiredState.from_tier(HabitabilityTier.E3_ARMSTRONG)
YEAR = 365.25 * 86400


# ------------------------------------------------------------------------------------ headroom
class TestHeadroom:
    def test_required_and_shortfall(self):
        v = verdict(MARS, T, available_inventory_kg=1e17)
        need = 5.66e3 * 4 * math.pi * MARS.radius_m**2 / MARS.gravity_ms2        # (6270 - 610) Pa
        assert v.required["mass"] == pytest.approx(need, rel=2e-3)
        assert v.shortfall["mass"] == pytest.approx(need - 1e17, rel=2e-3)

    def test_no_shortfall_when_viable(self):
        v = verdict(MARS, T, available_inventory_kg=1e19)
        assert v.shortfall["mass"] == 0.0 and v.viable

    def test_mass_limit_is_not_fixed_by_time(self):
        v = verdict(MARS, T, available_inventory_kg=1e16, available_mass_flow_kg_s=1e9)
        assert math.isinf(v.min_build_time_years)
        assert json.loads(v.to_json())["min_build_time_years"] == "inf"           # strict JSON, no Infinity

    def test_min_build_time_scales_flow_axis(self):
        r = required_availability(MARS, T, 1000.0)
        v = verdict(MARS, T, build_time_years=1000.0, available_inventory_kg=1e19, available_mass_flow_kg_s=r["throughput"] / 4)
        assert v.pi["throughput"] == pytest.approx(0.25)
        assert v.min_build_time_years == pytest.approx(4000.0, rel=1e-9)
        w = verdict(MARS, T, build_time_years=v.min_build_time_years, available_inventory_kg=1e19,
                    available_mass_flow_kg_s=r["throughput"] / 4)
        assert w.pi["throughput"] == pytest.approx(1.0, rel=1e-9) and w.viable

    def test_explain_names_the_limiting_axis(self):
        v = verdict(MARS, T, available_inventory_kg=1e17, available_power_w=1e12)
        text = v.explain()
        assert "NOT VIABLE" in text and "mass" in text and "confidence" in text
        assert "undetermined" in verdict(MARS, T).explain()

    def test_enclosed_regional_requirement_uses_area(self):
        t = DesiredState(min_surface_pressure_pa=50_000.0, enclosed=True, region_area_m2=1e4, min_o2_partial_pressure_pa=10_000.0)
        v = verdict(MARS, t, available_inventory_kg=1e6)
        assert v.requirements["atmosphere_mass_kg"] == pytest.approx(1e4 * 5e4 / MARS.gravity_ms2, rel=1e-9)
        assert v.requirements["o2_mass_kg"] == pytest.approx(1e4 * 1e4 / MARS.gravity_ms2, rel=1e-9)


# ------------------------------------------------------------------------------------ screening
class TestScreening:
    def test_rank_order_and_determinism(self):
        cs = [Candidate("z", "Mars", T, {"available_inventory_kg": 3e17}), Candidate("a", "Mars", T, {"available_inventory_kg": 3e17}),
              Candidate("low", "Mars", T, {"available_inventory_kg": 1e17}), Candidate("none", "Mars", T)]
        r = rank(cs)
        assert [x.name for x in r] == ["a", "z", "low", "none"] and [x.rank for x in r] == [1, 2, 3, 4]
        assert [x.name for x in rank(list(reversed(cs)))] == ["a", "z", "low", "none"]

    def test_sweep_crosses_one_exactly_at_the_breakeven(self):
        need = breakeven("Mars", T, "mass")
        pts = sweep("Mars", T, "mass", [0.5 * need, need, 2 * need])
        assert [p.viable for p in pts] == [False, True, True]
        assert pts[1].pi_min == pytest.approx(1.0, rel=1e-9)
        with pytest.raises(ValueError):
            sweep("Mars", T, "stability", [1.0])

    def test_what_if_flip(self):
        base = Candidate("b", "Mars", T, {"available_inventory_kg": 1e17})
        w = what_if(base, available_inventory_kg=1e19)
        assert w["viability_changed"] and not w["limiting_axis_changed"] and w["pi_min_ratio"] == pytest.approx(100.0)


# ------------------------------------------------------------------------------------ traceability
class TestTraceability:
    def test_replay_matches_and_detects_tampering(self):
        v = verdict(MARS, T, available_inventory_kg=1e17, available_power_w=1e12)
        rec = json.loads(v.to_json())
        r = replay(rec)
        assert r.matches and r.same_environment
        rec["inputs"]["availability"]["inventory_kg"] = 2e17          # tamper with an input
        assert not replay(rec).matches and "input_hash" in replay(rec).differences
        rec2 = json.loads(v.to_json())
        rec2["verdict"]["pi_min"] = 5.0                                # tamper with a result
        assert "verdict" in replay(rec2).differences

    def test_schema_validation(self):
        rec = json.loads(verdict(MARS, T, available_inventory_kg=1e17).to_json())
        assert validate_record(rec) == [] and rec["schema"] == SCHEMA
        bad = dict(rec)
        bad["input_hash"] = "xyz"
        bad["verdict"] = {**rec["verdict"], "limiting_axis": "luck"}
        assert len(validate_record(bad)) >= 2
        assert validate_record({}) != []

    def test_schema_file_is_in_sync(self):
        on_disk = json.loads((ROOT / "docs" / "noarco-result-1.1.schema.json").read_text())
        assert on_disk == result_schema()

    def test_hashes_present_and_stable(self):
        v = verdict(MARS, T, available_inventory_kg=1e17)
        assert len(code_hash()) == 64 and v.code_hash == code_hash()
        assert v.to_json() == verdict(MARS, T, available_inventory_kg=1e17).to_json()
        assert "code\\_hash" in v.bibtex() and v.input_hash in v.bibtex()

    def test_constants_used_cite_sources(self):
        v = verdict(MARS, T, available_inventory_kg=1e17, available_power_w=1e12, available_mass_flow_kg_s=1e6)
        names = {c["name"] for c in v.constants_used}
        assert {"E_O2_reversible", "YEAR_S (Julian)"} <= names
        assert all(c["source"] and c["class"] for c in v.constants_used)

    def test_strict_json_everywhere(self):
        v = verdict(MARS, T, available_inventory_kg=1e17, available_power_w=1e12)
        json.loads(v.to_json(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))   # no NaN/Infinity tokens

    def test_verify_record_script(self, tmp_path):
        f = tmp_path / "r.json"
        f.write_text(verdict(MARS, T, available_inventory_kg=1e17).to_json())
        ok = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_record.py"), str(f)], capture_output=True, text=True, check=False)
        assert ok.returncode == 0 and "MATCH" in ok.stdout
        rec = json.loads(f.read_text())
        rec["inputs"]["availability"]["inventory_kg"] = 1.0
        f.write_text(json.dumps(rec))
        bad = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_record.py"), str(f)], capture_output=True, text=True, check=False)
        assert bad.returncode == 1
        assert subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_record.py"), str(tmp_path / "x")],
                              capture_output=True, check=False).returncode == 2

    def test_citation_file(self):
        txt = (ROOT / "CITATION.cff").read_text()
        assert "cff-version" in txt and "CC-BY-4.0" in txt and "2603.00402" in txt


# ------------------------------------------------------------------------------------ ISRU / habitat / logistics
SITE = ISRUSiteProfile("Jezero", "Mars", 210.0, 610.0, {"SiO2": 0.45, "Fe2O3": 0.18, "H2O": 0.03, "Cl": 0.005})
DRY = ISRUSiteProfile("Dry", "Mars", 210.0, 610.0, {"SiO2": 0.45, "Fe2O3": 0.18})


class TestISRULedger:
    def test_energy_never_below_gibbs_floor_and_exergy_efficiency_in_range(self):
        for o2 in (1.0, 10.0, 100.0, 1000.0):
            r = ISRUEvaluator().evaluate_site(SITE, ISRURequirement(o2_kg_day=o2, water_kg_day=0))
            assert r.reactor_energy_kwh_day >= r.energy_floor_kwh_day
            assert r.energy_floor_kwh_day == pytest.approx(o2 * 14.82 / 3.6, rel=2e-3)
            assert 0 < r.exergy_efficiency <= 1

    def test_stoichiometric_closure(self):
        r = ISRUEvaluator().evaluate_site(DRY, ISRURequirement(o2_kg_day=100, water_kg_day=0))
        assert r.selected_mechanism_id == "iron_oxide_reduction"
        assert r.co_products["Fe_metal_kg"] == pytest.approx(100 * 4 * 55.845 / (3 * 31.9988), rel=1e-3)
        assert abs(r.unaccounted_mass_kg_day) < 0.01 * 100 / 0.3006
        assert r.daily_tailings_waste_kg == pytest.approx(r.daily_regolith_mined_kg - 100 - r.co_products["Fe_metal_kg"], rel=1e-9)

    def test_water_is_one_stream_for_reactant_and_delivery(self):
        r = ISRUEvaluator().evaluate_site(SITE, ISRURequirement(o2_kg_day=50, water_kg_day=20))
        assert r.selected_mechanism_id == "electrolysis_h2o"
        total_water_extracted = r.daily_regolith_mined_kg * 0.03 * 0.90
        assert total_water_extracted == pytest.approx(r.reactant_consumed_kg_day * (0.888 / 0.888) + 20, rel=2e-2)
        assert r.water_from_regolith_kg_day == pytest.approx(20, rel=1e-6)

    def test_missing_mineral_is_an_error_not_an_invented_fraction(self):
        site = ISRUSiteProfile("OnlySilica", "Mars", 210.0, 610.0, {"SiO2": 0.9, "H2O": 0.0})
        with pytest.raises(ValueError):
            ISRUEvaluator().evaluate_site(site, ISRURequirement(o2_kg_day=10))

    def test_solar_array_scales_with_power_and_irradiance(self):
        a = ISRUEvaluator().evaluate_site(SITE, ISRURequirement(o2_kg_day=50, water_kg_day=0))
        b = ISRUEvaluator().evaluate_site(SITE, ISRURequirement(o2_kg_day=100, water_kg_day=0))
        assert b.solar_array_area_m2 == pytest.approx(2 * a.solar_array_area_m2, rel=1e-9)
        assert a.solar_array_area_m2 == pytest.approx(a.continuous_power_kw * 1000 / (590 * 0.25 * 0.8), rel=1e-9)

    def test_bad_requirement_params(self):
        with pytest.raises(ValueError):
            ISRURequirement(water_recovery_fraction=0.0)


class TestHabitatShell:
    def test_hull_thickness_matches_thin_wall_formula(self):
        spec = HabitatSpecification(volume_m3=1000.0, crew_size=4, interior_pressure_pa=101_325.0)
        r = HabitatDimensioner.dimension(MARS, spec)
        radius = (3 * 1000.0 / (4 * math.pi)) ** (1 / 3)
        assert r.hull_wall_thickness_m == pytest.approx((101_325 - 610) * radius / (2 * 1.38e8), rel=1e-9)
        assert r.hull_mass_kg == pytest.approx(4 * math.pi * radius**2 * r.hull_wall_thickness_m * 2700, rel=1e-9)
        assert r.hull_governed_by == "tension"

    def test_minimum_gauge_and_external_pressure(self):
        tiny = HabitatDimensioner.dimension(EARTH, HabitatSpecification(volume_m3=10.0, crew_size=1, interior_pressure_pa=101_325.0))
        assert tiny.hull_governed_by == "minimum_gauge" and tiny.hull_wall_thickness_m == 0.002
        v = HabitatDimensioner.dimension(VENUS, HabitatSpecification(volume_m3=100.0, crew_size=2))
        assert v.hull_governed_by == "external_pressure_buckling" and v.warnings

    def test_mission_ledger_scales_with_days(self):
        a = HabitatDimensioner.dimension(MARS, HabitatSpecification(volume_m3=200, crew_size=3, mission_days=100))
        b = HabitatDimensioner.dimension(MARS, HabitatSpecification(volume_m3=200, crew_size=3, mission_days=200))
        assert b.mission_water_makeup_kg == pytest.approx(2 * a.mission_water_makeup_kg)
        assert b.mission_energy_kwh == pytest.approx(2 * a.mission_energy_kwh)
        assert a.mission_water_makeup_kg == pytest.approx(3 * 2.5 * 0.05 * 100)


class TestLogistics:
    def test_hohmann_reference_values(self):
        assert hohmann_delta_v_km_s(1.0, 1.524) == pytest.approx(5.6, abs=0.05)         # Earth -> Mars heliocentric
        assert hohmann_delta_v_km_s(1.0, 1.0) == pytest.approx(0.0, abs=1e-9)
        assert hohmann_delta_v_km_s(1.0, 5.2) == pytest.approx(hohmann_delta_v_km_s(5.2, 1.0), rel=1e-9)
        with pytest.raises(ValueError):
            hohmann_delta_v_km_s(0.0, 1.0)

    def test_moon_no_longer_inherits_mars_delta_v(self):
        r = VolatileLogisticsPlanner.plan_import(MOON, "H2O", 1e12, "Main_Belt_Asteroids")
        assert r.delta_v_method == "heliocentric_hohmann" and r.delta_v_km_s == pytest.approx(r.delta_v_hohmann_km_s)
        assert any("No catalogue delta-v" in w for w in r.warnings)

    def test_rocket_equation(self):
        r = VolatileLogisticsPlanner.plan_import(MARS, "N2", 1e15, isp_s=3000.0)
        ve = 3000.0 * 9.80665
        assert r.propellant_fraction == pytest.approx(1 - math.exp(-r.delta_v_km_s * 1000 / ve), rel=1e-9)
        assert r.propellant_mass_kg == pytest.approx(r.gross_import_mass_kg * (math.exp(r.delta_v_km_s * 1000 / ve) - 1), rel=1e-9)
        with pytest.raises(ValueError):
            VolatileLogisticsPlanner.plan_import(MARS, "N2", 1e15, isp_s=0.0)


class TestAgroCrew:
    def test_crew_supported_on_calorie_basis(self):
        r = AstroAgroPlanner.plan_soil_conversion(SoilConditioningSpec(greenhouse_area_m2=1000.0, food_yield_kg_m2_year=25.0))
        per_person = 2500 * 365.25 / 800
        assert r.crew_supported == pytest.approx(25000 / per_person, rel=1e-9)
        assert r.area_per_person_m2 == pytest.approx(1000 / r.crew_supported, rel=1e-9)


class TestBaseBudget:
    SPEC = HabitatSpecification(volume_m3=500, crew_size=4, mission_days=500)

    def test_conjunctive_budget(self):
        b0 = size_base(MARS, self.SPEC, SITE)
        assert b0.viable is None and b0.pi == {}
        b = size_base(MARS, self.SPEC, SITE, available_power_kw=b0.required["power"] / 2, available_o2_capacity_kg_day=100.0,
                      available_cargo_kg=b0.required["cargo"] * 3)
        assert b.limiting_axis == "power" and b.pi_min == pytest.approx(0.5) and b.viable is False
        assert "NOT viable" in b.summary()

    def test_power_is_the_sum_of_habitat_and_isru(self):
        b = size_base(MARS, self.SPEC, SITE)
        assert b.required["power"] == pytest.approx(b.habitat.continuous_power_kw + b.isru.continuous_power_kw)
        assert b.required["cargo"] == pytest.approx(b.habitat.hull_mass_kg + b.habitat.n2_mass_kg + b.habitat.o2_mass_kg
                                                    + b.habitat.mission_water_makeup_kg)

    def test_isru_sized_to_habitat_demand(self):
        b = size_base(MARS, self.SPEC, SITE)
        assert b.isru.daily_o2_kg == pytest.approx(b.required["oxygen"])
        assert b.isru.daily_o2_kg >= b.habitat.net_daily_o2_makeup_kg
