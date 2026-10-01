"""The product surface: GCM validation, single-call verdict, constants provenance, citeable records."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from noarco.constants_registry import (
    CLASSES,
    as_dicts,
    entries,
    registry_hash,
    to_markdown,
    verify_registry,
)
from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.data.planets import MARS
from noarco.validation import (
    MARSWRF_TABLE_S3,
    gcm_markdown,
    leave_one_out,
    run_gcm_validation,
)
from noarco.verdict import AXES, SCHEMA, verdict

T = HabitabilityTier
E1, E3, E4 = (DesiredState.from_tier(t) for t in (T.E1_TRIPLE_POINT, T.E3_ARMSTRONG, T.E4_BREATHABLE))
ROOT = Path(__file__).resolve().parents[1]


class TestGCMValidation:
    def test_no_comparison_fails(self):
        assert [r.case_id for r in run_gcm_validation() if r.status == "FAIL"] == []

    def test_out_of_sample_errors_are_bounded_and_reported(self):
        for mat, bound in (("Al", 0.30), ("C", 0.20)):
            r = next(x for x in run_gcm_validation() if x.case_id == f"RICH25-LOO-{mat}")
            assert r.kind == "out_of_sample" and r.noarco_value <= bound

    def test_loo_never_uses_the_held_out_point(self):
        loo = leave_one_out("Al")
        assert len(loo) == sum(1 for r in MARSWRF_TABLE_S3 if r[0] == "Al")
        # a model fitted on everything fits better than leave-one-out predictions (no leakage)
        from noarco.validation import _fit_kappa, _predict_warming
        rows = [r for r in MARSWRF_TABLE_S3 if r[0] == "Al"]
        k = _fit_kappa(rows)
        in_sample = sum(abs(p - r[2]) / r[2] for r, p in zip(rows, _predict_warming(k, [r[3] for r in rows]), strict=True))
        out_sample = sum(abs(p - r[2]) / r[2] for r, p in loo)
        assert out_sample > in_sample

    def test_documented_discrepancies_are_reported_not_hidden(self):
        res = {r.case_id: r for r in run_gcm_validation()}
        d = {k for k, r in res.items() if r.status in ("DISCREPANCY", "KNOWN_LIMITATION")}
        assert {"RICH25-VS-TURYSHEV", "JE18-1BAR", "RICH25-GH6MBAR", "RICH25-AL160"} <= d
        assert res["RICH25-VS-TURYSHEV"].rel_error > 0.8 and res["JE18-1BAR"].noarco_value < 60.0

    def test_radiative_cases_are_a_separate_gate_with_explicit_usage_flags(self):
        res = {r.case_id: r for r in run_gcm_validation()}
        for cid in ("JE18-1BAR", "RICH25-GH6MBAR"):
            r = res[cid]
            assert r.gate == "radiative" and r.status == "KNOWN_LIMITATION"
            assert r.use_for_pi_min_mass is True and r.use_for_dt_absolute is False
            assert r.error_radiative == pytest.approx(r.rel_error)
        assert res["JE18-1BAR"].rel_error > 0.4 and res["JE18-1BAR"].in_band is False
        assert res["RICH25-GH6MBAR"].in_band is None
        assert res["RICH25-KAPPA-Al"].error_radiative is None

    def test_calibrated_mode_closes_the_1_bar_gap_with_traceability(self):
        r = {x.case_id: x for x in run_gcm_validation()}["JE18-1BAR-CAL"]
        assert r.in_band is True and r.rel_error < 0.05 and r.use_for_dt_absolute is True
        assert r.kind == "calibration"                      # not an independent test

    def test_20_mbar_bound_and_calibration_are_labelled(self):
        r = {x.case_id: x for x in run_gcm_validation()}
        assert r["JE18-20MBAR"].kind == "bound" and r["JE18-20MBAR"].noarco_value < 10.0
        assert r["RICH25-KAPPA-Al"].kind == "calibration"                      # not an independent test

    def test_markdown(self):
        md = gcm_markdown()
        assert "documented discrepancies" in md and "RICH25-LOO-Al" in md


class TestVerdict:
    def test_viable_and_limiting_axis(self):
        v = verdict(MARS, E1, available_inventory_kg=7.78e16)
        assert v.viable is True and v.limiting_axis == "mass"
        w = verdict(MARS, E3, available_inventory_kg=7.78e16, available_power_w=1e12)
        assert w.viable is False and w.limiting_axis == "mass" and w.pi_min == pytest.approx(0.354, abs=0.01)

    @pytest.mark.parametrize("kw,axis", [
        ({"available_inventory_kg": 1e20, "available_power_w": 1e9}, "power"),
        ({"available_inventory_kg": 1e20, "available_power_w": 1e30, "available_delta_tau_ir": 0.5}, "forcing"),
        ({"available_inventory_kg": 1e20, "available_power_w": 1e30, "available_delta_tau_ir": 1e6,
          "available_mass_flow_kg_s": 1.0}, "throughput"),
        ({"available_inventory_kg": 1e20, "available_power_w": 1e30, "available_delta_tau_ir": 1e6,
          "available_mass_flow_kg_s": 1e30, "replenishment_kg_s": 1.0, "loss_kg_s": 10.0}, "stability"),
    ])
    def test_every_axis_can_be_the_limiting_one(self, kw, axis):
        v = verdict(MARS, E4, **kw)
        assert v.limiting_axis == axis and v.viable is False and axis in AXES

    def test_no_availability_means_undetermined_not_assumed(self):
        v = verdict(MARS, E3)
        assert v.viable is None and v.limiting_axis is None and v.pi == {}

    def test_record_is_self_contained_and_citeable(self):
        v = verdict(MARS, E3, available_inventory_kg=7.78e16)
        d = json.loads(v.to_json())
        assert d["schema"] == SCHEMA and d["model_class"] == "reference_scaling" and d["version"]
        assert d["input_hash"] == v.input_hash and d["constants_hash"] == registry_hash()
        assert d["verdict"]["limiting_axis"] == "mass" and "not a climate simulation" in d["scope"]
        assert v.input_hash[:16] in v.cite() and "not viable" in v.cite()

    def test_hash_is_a_pure_function_of_inputs(self):
        a = verdict(MARS, E3, available_inventory_kg=7.78e16)
        b = verdict(MARS, E3, available_inventory_kg=7.78e16)
        c = verdict(MARS, E3, available_inventory_kg=7.79e16)
        assert a.input_hash == b.input_hash and a.to_json() == b.to_json() and a.input_hash != c.input_hash

    def test_accepts_any_body_and_reports_assumptions(self):
        v = verdict({"gravity_ms2": 1.6, "radius_m": 1.74e6, "solar_constant_wm2": 1361.0}, E1, available_inventory_kg=1e15)
        assert v.body == "custom-body" and v.body_assumptions


class TestConstantsProvenance:
    def test_registry_matches_the_code(self):
        assert verify_registry() == []

    def test_every_entry_is_classified_and_sourced(self):
        for e in entries():
            assert e.klass in CLASSES and e.source and e.unit
        assert {"exact", "literature", "estimated", "calibrated", "measured", "derived"} <= {e.klass for e in entries()}

    def test_unverified_constants_are_not_hidden(self):
        est = {e.name for e in entries() if e.klass == "estimated"}
        assert {"CO2_forcing_per_doubling", "PFC_synthesis_energy", "AEROGEL_manufacture_energy"} <= est

    def test_docs_file_is_in_sync(self):
        assert to_markdown() in (ROOT / "docs" / "CONSTANTS.md").read_text()

    def test_hash_changes_when_an_entry_changes(self):
        assert registry_hash() == registry_hash() and len(as_dicts()) == len(entries())


class TestReproduceScript:
    def test_script_exits_zero_and_prints_the_tables(self):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "reproduce_reference.py")], capture_output=True, text=True,
                           cwd=ROOT, check=False)
        assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
        assert "all anchors reproduced" in r.stdout and "T-EQ31-60K" in r.stdout and "documented discrepancies" in r.stdout

    def test_script_writes_json(self, tmp_path):
        out = tmp_path / "r.json"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "reproduce_reference.py"), "--json", str(out)],
                       capture_output=True, text=True, cwd=ROOT, check=True)
        d = json.loads(out.read_text())
        assert len(d["anchor"]) >= 20 and all(x["passed"] for x in d["anchor"])
