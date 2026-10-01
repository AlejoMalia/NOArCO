"""External validation report and the hand-off of surviving scenarios to a GCM."""

import csv
import json

import pytest

from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.data.planets import MARS
from noarco.exchange import SCHEMA, export_candidates, scenario_dict
from noarco.validation import run_validation, to_markdown

T = HabitabilityTier
ALL = {t.value: DesiredState.from_tier(t) for t in (T.E1_TRIPLE_POINT, T.E2_PROTECTED, T.E3_ARMSTRONG, T.E4_BREATHABLE)}


class TestValidation:
    def test_every_external_case_passes(self):
        bad = [(r.case.case_id, r.rel_error) for r in run_validation() if not r.passed]
        assert bad == []

    def test_coverage_of_sources(self):
        srcs = {r.case.source.split(" ")[0] for r in run_validation()}
        assert {"Turyshev", "IPCC", "Kopparapu", "Wordsworth"} <= srcs and len(run_validation()) >= 20

    def test_published_values_are_not_taken_from_the_code(self):
        # AR5 constants are compared with literals, so a code change must be a deliberate edit of both
        r = {x.case.case_id: x for x in run_validation()}
        assert r["AR5-C2F6-RE"].case.published == 0.25 and r["AR5-SF6-GWP"].case.published == 23500.0

    def test_markdown_report(self):
        md = to_markdown()
        assert "| T-EQ31-60K |" in md and "cases pass" in md and "**NO**" not in md


class TestExchange:
    def test_scenario_schema_and_contents(self):
        sc = scenario_dict(MARS, ALL["E3_armstrong"], label="e3")
        assert sc["schema"] == SCHEMA and sc["units"] == "SI"
        assert sc["initial_state"]["surface_pressure_pa"] == 610.0
        assert sc["forcing_for_gcm"]["absorbed_flux_forcing_wm2"] == pytest.approx(110.9, rel=0.02)
        assert sc["screening"]["binding_constraint"] == "mass" and sc["screening"]["survives"] is False
        assert "provenance" in sc and sc["limitations"]
        json.dumps(sc, default=str)                                           # serialisable

    def test_only_survivors_are_written_and_the_index_lists_all(self, tmp_path):
        written = export_candidates(MARS, ALL, tmp_path)
        names = {p.stem for p in written}
        assert "E1_triple_point" in names and "E4_breathable" not in names and "E3_armstrong" not in names
        with open(tmp_path / "index.csv") as fh:
            rows = list(csv.DictReader(fh))
        assert {r["label"] for r in rows} == set(ALL)
        assert next(r for r in rows if r["label"] == "E4_breathable")["survives"] == "False"
        assert json.loads((tmp_path / "E1_triple_point.json").read_text())["schema"] == SCHEMA

    def test_more_resources_let_more_scenarios_survive(self, tmp_path):
        small = len(export_candidates(MARS, ALL, tmp_path / "a"))
        big = len(export_candidates(MARS, ALL, tmp_path / "b", available_inventory_kg=1e19, available_power_w=1e16))
        assert big >= small and big >= 3

    def test_accepts_any_body_description(self, tmp_path):
        out = export_candidates({"gravity_ms2": 3.7, "radius_m": 3.39e6, "solar_constant_wm2": 589.0, "pressure_pa": 610.0,
                                 "temperature_k": 210.0, "composition": {"CO2": 1.0}, "co2_ice_kg": 2.3e16}, {"e1": ALL["E1_triple_point"]}, tmp_path)
        sc = json.loads(out[0].read_text())
        assert sc["provenance"]["body_assumptions"] is not None
