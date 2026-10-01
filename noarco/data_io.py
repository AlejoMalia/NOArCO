"""
noarco.data_io — real input data with provenance
================================================

Anchor an assessment to a concrete world and epoch with measured or published data instead of
typed-in numbers. A :class:`BodyDataset` holds

* **scalar fields**, each a :class:`Sourced` value: ``value`` + ``unit`` + ``data_source`` +
  ``reference`` where ``data_source`` is one of ``measured``, ``literature`` or ``assumed``;
* an optional **T-P profile** (pressure, temperature, optional height) from which the surface state
  is taken (the deepest level);
* an optional **simplified topography** (elevation samples or a hypsometric curve) from which the
  *mean* surface pressure is derived hydrostatically: ``P(z) = P_ref exp(-(z - z_ref)/H)``.

Formats: JSON (full), CSV (profile or topography, one header row ``name [unit]``) and NetCDF
(through ``xarray``; the pure-Python ``scipy`` backend reads NetCDF3, ``netCDF4``/``h5netcdf`` extend
this when installed). Units are converted strictly with :mod:`noarco.assurance.units` (an unknown unit
raises). Every dataset has a canonical SHA-256 ``input_hash``, and ``provenance_report`` separates
fact (measured/literature) from conjecture (assumed), so the audit trail survives into the assessment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from noarco.assurance.findings import Finding, Severity
from noarco.assurance.units import UnitError, convert
from noarco.core.constants import R_GAS

DATA_SOURCES = ("measured", "literature", "assumed")
#: SI unit of every supported scalar field.
FIELD_UNITS = {
    "pressure_pa": "Pa", "temperature_k": "K", "gravity_ms2": "m/s^2", "mass_kg": "kg", "radius_m": "m",
    "density_kg_m3": "kg/m^3", "albedo": "1", "solar_constant_wm2": "W/m^2", "ir_optical_depth": "1",
    "h2o_ice_kg": "kg", "co2_ice_kg": "kg",
}
#: Units whose parser needs an alias (not in the strict registry).
_ALIASES = {"m/s2": "m/s^2", "m s-2": "m/s^2", "W/m2": "W/m^2", "kg/m3": "kg/m^3", "dimensionless": "1", "-": "1"}


@dataclass(frozen=True)
class Sourced:
    """A value with unit and provenance."""

    value: float
    unit: str
    data_source: str
    reference: str = ""

    def __post_init__(self) -> None:
        if self.data_source not in DATA_SOURCES:
            raise ValueError(f"data_source must be one of {DATA_SOURCES}, got {self.data_source!r}")
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")

    def si(self, name: str) -> float:
        unit = _ALIASES.get(self.unit.strip(), self.unit.strip())
        target = FIELD_UNITS[name]
        try:
            if unit in ("degC", "degF"):
                from noarco.assurance.units import convert_temperature
                return convert_temperature(self.value, unit, "K")
            return convert(self.value, unit, target)
        except UnitError as exc:
            raise UnitError(f"{name}: cannot convert '{self.unit}' to {target} ({exc})") from exc


@dataclass
class BodyDataset:
    name: str
    fields: dict[str, Sourced] = field(default_factory=dict)
    composition: dict[str, float] | None = None
    composition_source: Sourced | None = None
    regolith_composition: dict[str, float] | None = None
    profile_pa_k: tuple[np.ndarray, np.ndarray] | None = None          # (pressure Pa, temperature K)
    profile_source: Sourced | None = None
    topography_m: np.ndarray | None = None                             # elevation samples (m)
    topography_source: Sourced | None = None
    epoch: str = ""
    notes: str = ""

    # ------------------------------------------------------------------ derived inputs
    def _surface_from_profile(self) -> tuple[float, float] | None:
        if self.profile_pa_k is None:
            return None
        p, t = self.profile_pa_k
        i = int(np.argmax(p))
        return float(p[i]), float(t[i])

    def mean_surface_pressure_pa(self, scale_height_m: float | None = None) -> float | None:
        """Mean surface pressure from the topography: ``P_ref * mean(exp(-(z - z_ref)/H))``.

        ``z_ref`` is the elevation at which ``pressure_pa`` was measured (the dataset mean
        elevation if topography is given). ``H = R T / (mu g)`` is taken from the dataset
        (temperature, gravity and composition) unless ``scale_height_m`` is given.
        """
        if "pressure_pa" not in self.fields:
            return None
        p_ref = self.fields["pressure_pa"].si("pressure_pa")
        if self.topography_m is None:
            return p_ref
        z = self.topography_m
        h = scale_height_m or self._scale_height_m()
        if h is None:
            return None
        return float(p_ref * np.mean(np.exp(-(z - float(np.mean(z))) / h)))

    def _scale_height_m(self) -> float | None:
        if not {"temperature_k", "gravity_ms2"} <= set(self.fields) or not self.composition:
            return None
        from noarco.core.constants import MOLAR_MASS
        known = {k: v for k, v in self.composition.items() if k in MOLAR_MASS}
        tot = sum(known.values())
        if tot <= 0:
            return None
        mu = sum(MOLAR_MASS[k] * v for k, v in known.items()) / tot
        return R_GAS * self.fields["temperature_k"].si("temperature_k") / (mu * self.fields["gravity_ms2"].si("gravity_ms2"))

    def to_conditions(self, use_topography: bool = True) -> dict[str, Any]:
        """Conditions dict for :func:`noarco.bodies.resolve_body` (SI units)."""
        c: dict[str, Any] = {"name": self.name}
        for k, f in self.fields.items():
            c[{"albedo": "albedo"}.get(k, k)] = f.si(k)
        prof = self._surface_from_profile()
        if prof is not None:
            c.setdefault("pressure_pa", prof[0])
            c.setdefault("temperature_k", prof[1])
        if use_topography:
            mean_p = self.mean_surface_pressure_pa()
            if mean_p is not None:
                c["pressure_pa"] = mean_p
        if self.composition:
            c["composition"] = dict(self.composition)
        if self.regolith_composition:
            c["regolith_composition"] = dict(self.regolith_composition)
        c["source_type"] = self.worst_source_type()
        return c

    def worst_source_type(self) -> str:
        """Epistemic class of the weakest *input*: assumed > literature > measured -> planet source type."""
        srcs = [f.data_source for f in self.fields.values()]
        for s in (self.composition_source, self.profile_source, self.topography_source):
            if s is not None:
                srcs.append(s.data_source)
        if "assumed" in srcs:
            return "assumed"
        if "literature" in srcs:
            return "modeled"
        return "measured" if srcs else "assumed"

    # ------------------------------------------------------------------ audit
    def canonical(self) -> dict[str, Any]:
        return {
            "name": self.name, "epoch": self.epoch,
            "fields": {k: [v.value, v.unit, v.data_source, v.reference] for k, v in sorted(self.fields.items())},
            "composition": dict(sorted((self.composition or {}).items())),
            "regolith": dict(sorted((self.regolith_composition or {}).items())),
            "profile": None if self.profile_pa_k is None else [self.profile_pa_k[0].tolist(), self.profile_pa_k[1].tolist()],
            "topography": None if self.topography_m is None else self.topography_m.tolist(),
        }

    @property
    def input_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.canonical(), sort_keys=True).encode()).hexdigest()

    def provenance_report(self) -> dict[str, Any]:
        by: dict[str, list[str]] = {s: [] for s in DATA_SOURCES}
        for k, f in self.fields.items():
            by[f.data_source].append(k)
        for label, s in (("composition", self.composition_source), ("T-P profile", self.profile_source),
                         ("topography", self.topography_source)):
            if s is not None:
                by[s.data_source].append(label)
        return {"name": self.name, "epoch": self.epoch, "input_hash": self.input_hash, "by_source": by,
                "fact": by["measured"] + by["literature"], "conjecture": by["assumed"],
                "worst_source_type": self.worst_source_type()}

    def findings(self) -> list[Finding]:
        """Assurance findings: every assumed input and any input with no provenance."""
        out = [Finding("DATA-ASSUMED", Severity.CAUTION, f"input '{k}' is an assumption, not data")
               for k in self.provenance_report()["conjecture"]]
        if not self.fields:
            out.append(Finding("DATA-EMPTY", Severity.FAIL, "dataset has no scalar fields"))
        return out


# ---------------------------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------------------------
def _sourced(d: dict[str, Any], default_source: str | None = None) -> Sourced:
    return Sourced(float(d["value"]), d.get("unit", "1"), d.get("data_source", default_source or ""), d.get("reference", ""))


def load_json(path: str | Path) -> BodyDataset:
    """Load a dataset from JSON (schema in ``BodyDataset`` docs; every scalar needs ``data_source``)."""
    raw = json.loads(Path(path).read_text())
    ds = BodyDataset(name=raw["name"], epoch=raw.get("epoch", ""), notes=raw.get("notes", ""))
    for k, v in raw.get("fields", {}).items():
        if k not in FIELD_UNITS:
            raise ValueError(f"Unknown field '{k}'. Supported: {sorted(FIELD_UNITS)}")
        ds.fields[k] = _sourced(v)
    if "composition" in raw:
        ds.composition = dict(raw["composition"]["species"])
        ds.composition_source = Sourced(0.0, "1", raw["composition"]["data_source"], raw["composition"].get("reference", ""))
    ds.regolith_composition = raw.get("regolith_composition")
    if "profile" in raw:
        pr = raw["profile"]
        p = np.array([convert(x, pr.get("pressure_unit", "Pa"), "Pa") for x in pr["pressure"]], dtype=float)
        t = np.array(pr["temperature"], dtype=float)
        ds.profile_pa_k = (p, t)
        ds.profile_source = Sourced(0.0, "1", pr["data_source"], pr.get("reference", ""))
    if "topography" in raw:
        tp = raw["topography"]
        ds.topography_m = np.array([convert(x, tp.get("unit", "m"), "m") for x in tp["elevation"]], dtype=float)
        ds.topography_source = Sourced(0.0, "1", tp["data_source"], tp.get("reference", ""))
    return ds


def _read_csv_columns(path: str | Path) -> dict[str, tuple[str, np.ndarray]]:
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise ValueError("CSV needs a header row and at least one data row")
    cols: dict[str, tuple[str, np.ndarray]] = {}
    for j, head in enumerate(rows[0]):
        name, _, unit = head.partition("[")
        cols[name.strip()] = (unit.rstrip("]").strip() or "1", np.array([float(r[j]) for r in rows[1:]]))
    return cols


def attach_profile_csv(ds: BodyDataset, path: str | Path, data_source: str, reference: str = "") -> BodyDataset:
    """Add a T-P profile from CSV with columns ``pressure [unit]`` and ``temperature [unit]``."""
    cols = _read_csv_columns(path)
    if not {"pressure", "temperature"} <= set(cols):
        raise ValueError("profile CSV needs 'pressure [unit]' and 'temperature [unit]' columns")
    p = np.array([convert(x, cols["pressure"][0], "Pa") for x in cols["pressure"][1]])
    tu = cols["temperature"][0]
    from noarco.assurance.units import convert_temperature
    t = np.array([convert_temperature(x, tu, "K") if tu in ("degC", "degF") else convert(x, tu, "K")
                  for x in cols["temperature"][1]])
    ds.profile_pa_k = (p, t)
    ds.profile_source = Sourced(0.0, "1", data_source, reference)
    return ds


def attach_topography_csv(ds: BodyDataset, path: str | Path, data_source: str, reference: str = "") -> BodyDataset:
    """Add simplified topography from CSV with one column ``elevation [unit]``."""
    cols = _read_csv_columns(path)
    if "elevation" not in cols:
        raise ValueError("topography CSV needs an 'elevation [unit]' column")
    ds.topography_m = np.array([convert(x, cols["elevation"][0], "m") for x in cols["elevation"][1]])
    ds.topography_source = Sourced(0.0, "1", data_source, reference)
    return ds


def load_netcdf(path: str | Path, name: str, data_source: str, reference: str = "") -> BodyDataset:
    """Load a T-P profile (variables ``pressure`` and ``temperature`` with ``units`` attributes) from NetCDF."""
    import xarray as xr

    try:
        ds_nc = xr.open_dataset(path, engine="scipy")
    except Exception:  # noqa: BLE001 - fall back to whatever backend is installed
        ds_nc = xr.open_dataset(path)
    with ds_nc:
        for v in ("pressure", "temperature"):
            if v not in ds_nc:
                raise ValueError(f"NetCDF file lacks variable '{v}'")
        pu, tu = ds_nc["pressure"].attrs.get("units", "Pa"), ds_nc["temperature"].attrs.get("units", "K")
        p = np.array([convert(float(x), pu, "Pa") for x in ds_nc["pressure"].values.ravel()])
        t = np.array([convert(float(x), tu, "K") for x in ds_nc["temperature"].values.ravel()])
    out = BodyDataset(name=name)
    out.profile_pa_k = (p, t)
    out.profile_source = Sourced(0.0, "1", data_source, reference)
    return out


def dataset_to_planet(ds: BodyDataset, **kwargs: Any) -> Any:
    """Validated ``PlanetState`` for a dataset (see ``BodyDataset.to_conditions``)."""
    from noarco.bodies import resolve_body

    return resolve_body(ds.to_conditions(**kwargs)).planet
