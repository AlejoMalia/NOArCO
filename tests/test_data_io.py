"""Real input data with provenance: loaders, strict units, topography, hash and fact/conjecture split."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from noarco.assurance.units import UnitError
from noarco.bodies import resolve_body
from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.data.planets import MARS
from noarco.data_io import (
    BodyDataset,
    Sourced,
    attach_profile_csv,
    attach_topography_csv,
    dataset_to_planet,
    load_json,
    load_netcdf,
)
from noarco.workflow import clear_cache, run_assessment

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "data" / "mars_baseline.json"


class TestSourced:
    def test_validates_source_and_value(self):
        with pytest.raises(ValueError):
            Sourced(1.0, "Pa", "guess")
        with pytest.raises(ValueError):
            Sourced(math.nan, "Pa", "measured")

    def test_unit_conversion_and_alias(self):
        assert Sourced(6.1, "mbar", "measured").si("pressure_pa") == pytest.approx(610.0)
        assert Sourced(3.7, "m/s2", "measured").si("gravity_ms2") == pytest.approx(3.7)
        assert Sourced(-63.15, "degC", "measured").si("temperature_k") == pytest.approx(210.0)

    def test_wrong_dimension_is_rejected(self):
        with pytest.raises(UnitError):
            Sourced(1.0, "kg", "measured").si("pressure_pa")
        with pytest.raises(UnitError):
            Sourced(1.0, "furlong", "measured").si("radius_m")


class TestJsonExample:
    def test_example_reproduces_the_builtin_mars(self):
        ds = load_json(EXAMPLE)
        p = dataset_to_planet(ds)
        assert p.surface_pressure_pa == pytest.approx(MARS.surface_pressure_pa)
        assert p.radius_m == pytest.approx(MARS.radius_m)                        # "3.3895 Mm" converted strictly
        assert p.gravity_ms2 == pytest.approx(MARS.gravity_ms2) and p.co2_ice_kg == 2.3e16

    def test_provenance_separates_fact_from_conjecture(self):
        rep = load_json(EXAMPLE).provenance_report()
        assert "ir_optical_depth" in rep["conjecture"] and "composition" in rep["fact"]
        assert "pressure_pa" in rep["by_source"]["measured"] and rep["worst_source_type"] == "assumed"

    def test_findings_flag_assumed_inputs(self):
        assert [f.code for f in load_json(EXAMPLE).findings()] == ["DATA-ASSUMED"]

    def test_hash_is_stable_and_sensitive(self):
        a, b = load_json(EXAMPLE), load_json(EXAMPLE)
        assert a.input_hash == b.input_hash and len(a.input_hash) == 64
        b.fields["albedo"] = Sourced(0.26, "1", "literature")
        assert a.input_hash != b.input_hash

    def test_unknown_field_rejected(self, tmp_path):
        raw = json.loads(EXAMPLE.read_text())
        raw["fields"]["bogus"] = {"value": 1, "unit": "1", "data_source": "measured"}
        p = tmp_path / "x.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="Unknown field"):
            load_json(p)

    def test_missing_source_is_rejected(self, tmp_path):
        raw = json.loads(EXAMPLE.read_text())
        del raw["fields"]["albedo"]["data_source"]
        p = tmp_path / "x.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            load_json(p)


class TestProfileAndTopography:
    def test_profile_csv_takes_the_surface_from_the_deepest_level(self, tmp_path):
        f = tmp_path / "tp.csv"
        f.write_text("pressure [mbar],temperature [K]\n0.001,140\n1,170\n6.2,212\n")
        ds = attach_profile_csv(BodyDataset("X", fields={"gravity_ms2": Sourced(3.7, "m/s^2", "literature"),
                                                         "radius_m": Sourced(3.39e6, "m", "literature"),
                                                         "solar_constant_wm2": Sourced(589, "W/m^2", "literature")}),
                                f, "measured", "test")
        c = ds.to_conditions()
        assert c["pressure_pa"] == pytest.approx(620.0) and c["temperature_k"] == 212.0
        assert ds.profile_source.data_source == "measured"

    def test_profile_csv_requires_columns_and_rows(self, tmp_path):
        f = tmp_path / "bad.csv"
        f.write_text("p,T\n1,2\n")
        with pytest.raises(ValueError):
            attach_profile_csv(BodyDataset("X"), f, "measured")
        g = tmp_path / "empty.csv"
        g.write_text("pressure [Pa],temperature [K]\n")
        with pytest.raises(ValueError):
            attach_profile_csv(BodyDataset("X"), g, "measured")

    def test_topography_changes_the_mean_pressure_hydrostatically(self, tmp_path):
        ds = load_json(EXAMPLE)
        flat = ds.mean_surface_pressure_pa()
        assert flat == pytest.approx(610.0)                                   # no topography yet
        f = tmp_path / "z.csv"
        f.write_text("elevation [km]\n-4\n0\n4\n")
        attach_topography_csv(ds, f, "literature", "synthetic")
        h = ds._scale_height_m()
        expected = 610.0 * np.mean(np.exp(-np.array([-4000, 0, 4000.0]) / h))
        assert ds.mean_surface_pressure_pa() == pytest.approx(expected)
        assert ds.mean_surface_pressure_pa() > 610.0                          # convexity: mean exceeds the reference
        assert dataset_to_planet(ds).surface_pressure_pa == pytest.approx(expected)
        assert dataset_to_planet(ds, use_topography=False).surface_pressure_pa == pytest.approx(610.0)

    def test_topography_requires_scale_height_ingredients(self):
        ds = BodyDataset("Y", fields={"pressure_pa": Sourced(100.0, "Pa", "measured")})
        ds.topography_m = np.array([0.0, 1000.0])
        assert ds.mean_surface_pressure_pa() is None

    def test_netcdf_profile_round_trip(self, tmp_path):
        import xarray as xr
        f = tmp_path / "tp.nc"
        xr.Dataset({"pressure": ("lev", np.array([1.0, 6.0]), {"units": "mbar"}),
                    "temperature": ("lev", np.array([180.0, 210.0]), {"units": "K"})}).to_netcdf(f, engine="scipy")
        ds = load_netcdf(f, "NC", "measured", "unit test")
        assert ds.profile_pa_k[0].tolist() == pytest.approx([100.0, 600.0]) and ds.profile_pa_k[1][-1] == 210.0

    def test_netcdf_missing_variable(self, tmp_path):
        import xarray as xr
        f = tmp_path / "bad.nc"
        xr.Dataset({"pressure": ("lev", np.array([1.0]))}).to_netcdf(f, engine="scipy")
        with pytest.raises(ValueError):
            load_netcdf(f, "NC", "measured")


class TestIntegration:
    def test_dataset_flows_through_resolve_and_assessment(self):
        clear_cache()
        ds = load_json(EXAMPLE)
        r = resolve_body(ds)
        assert r.derived["input_hash"] == ds.input_hash and any("assumption" in a for a in r.assumptions)
        rep = run_assessment(ds, DesiredState.from_tier(HabitabilityTier.E3_ARMSTRONG))
        assert rep.planet == "Mars" and any(f.code == "BODY-ASSUMPTION" for f in rep.findings)

    def test_empty_dataset_is_flagged(self):
        assert [f.code for f in BodyDataset("E").findings()] == ["DATA-EMPTY"]
