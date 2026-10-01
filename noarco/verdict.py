"""
noarco.verdict — the product: one call, a viable / not-viable verdict and the axis that limits it
================================================================================================

``verdict(body, target, ...)`` evaluates the conjunctive rubric ``Pi_min = min(available / required)``
(Turyshev 2026, Eq. 9-11) and returns

* ``viable``: ``True`` / ``False`` / ``None`` (None = no availability was supplied for any axis);
* ``limiting_axis``: ``mass`` | ``power`` | ``forcing`` | ``throughput`` | ``stability``;
* every ``Pi`` and requirement, the inputs hash, the package version, the model class and the
  provenance of the constants (hash of :mod:`noarco.constants_registry`).

``Verdict.to_dict()`` is a self-contained JSON record (schema ``noarco-result/1.1``): a third party can
cite it and re-run it, because the same inputs reproduce the same ``input_hash`` and numbers. Timestamps
are deliberately excluded so the record is a pure function of its inputs.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from noarco import __version__
from noarco.bodies import ResolvedBody, resolve_body
from noarco.constants_registry import entries as registry_entries
from noarco.constants_registry import registry_hash
from noarco.core.desired_state import DesiredState
from noarco.feasibility import PiAssessment, assess, endpoint_requirements

SCHEMA = "noarco-result/1.1"
AXES = ("mass", "power", "forcing", "throughput", "stability")
RADIATIVE_MODES = ("simple", "literature_calibrated")


def axis_confidence(radiative_mode: str = "simple") -> dict[str, str]:
    """Confidence label per verdict axis.

    Mass, power and throughput are closed-form inventory/energy constraints that match the reference
    paper's anchors to a few percent (``high``). The forcing/dT axis rests on the radiative model: ``low``
    in the simple mode (biased low versus GCMs at high CO2 pressure), ``medium`` when a published
    literature curve is used. Stability depends on user-declared loss/replenishment (``medium``).
    """
    if radiative_mode not in RADIATIVE_MODES:
        raise ValueError(f"radiative_mode must be one of {RADIATIVE_MODES}")
    return {"mass": "high", "power": "high", "throughput": "high",
            "forcing": "low" if radiative_mode == "simple" else "medium", "stability": "medium"}


# availability keyword -> axis it feeds
_AXIS_REQUIREMENT = {          # axis -> (requirement field, unit)
    "mass": ("atmosphere_mass_kg", "kg"),
    "throughput": ("mean_mass_flow_kg_s", "kg/s"),
    "power": ("mean_power_w", "W"),
    "forcing": ("delta_forcing_toa_wm2", "W/m2"),
}
# registry constants each axis rests on (the rest of the inputs are the body's own data, captured in ``inputs``)
_AXIS_CONSTANTS = {
    "mass": (),
    "throughput": ("YEAR_S (Julian)",),
    "power": ("YEAR_S (Julian)", "DELTA_G_H2O", "E_O2_reversible"),
    "forcing": ("SIGMA_SB",),
    "stability": (),
}


def _finite(obj: Any) -> Any:
    """Strict-JSON form: ``inf`` -> the string ``"inf"`` (nothing required / no build time suffices)."""
    if isinstance(obj, float) and math.isinf(obj):
        return "inf" if obj > 0 else "-inf"
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    """SHA-256 over the source of the whole ``noarco`` package: ties a record to the exact code that made it."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _constants_used(axes: list[str]) -> list[dict[str, Any]]:
    names = sorted({n for a in axes for n in _AXIS_CONSTANTS.get(a, ())})
    by_name = {e.name: e for e in registry_entries()}
    return [{"name": n, "value": by_name[n].value, "unit": by_name[n].unit, "source": by_name[n].source,
             "class": by_name[n].klass} for n in names if n in by_name]


@dataclass
class Verdict:
    body: str
    target: str
    viable: bool | None
    limiting_axis: str | None
    pi_min: float | None
    pi: dict[str, float]
    requirements: dict[str, float]
    inputs: dict[str, Any]
    input_hash: str
    model_class: str = "reference_scaling"
    body_assumptions: list[str] = field(default_factory=list)
    version: str = __version__
    constants_hash: str = ""
    radiative_mode: str = "simple"
    axis_confidence: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    required: dict[str, float] = field(default_factory=dict)          # axis -> availability needed to pass
    shortfall: dict[str, float] = field(default_factory=dict)         # axis -> max(required - available, 0)
    min_build_time_years: float | None = None                         # time at which flow/power axes pass
    constants_used: list[dict[str, Any]] = field(default_factory=list)
    code_hash: str = ""

    def explain(self) -> str:
        """Plain-language reading of the verdict: the axes ranked from the most limiting."""
        if self.pi_min is None:
            return f"{self.body} -> {self.target}: undetermined (no availability supplied for any axis)."
        order = sorted(self.pi.items(), key=lambda kv: kv[1])
        lines = [
            (f"{self.body} -> {self.target}: {'VIABLE' if self.viable else 'NOT VIABLE'} "
             f"(Pi_min = {self.pi_min:.3g}, limited by {self.limiting_axis}).")
        ]
        for axis, val in order:
            conf = self.axis_confidence.get(axis, "?")
            need = self.required.get(axis)
            short = self.shortfall.get(axis, 0.0)
            tail = f" - need {need:.3g}, short by {short:.3g}" if need is not None and short > 0 else ""
            lines.append(f"  {axis:<10} Pi = {val:.3g}  [{conf} confidence]{tail}")
        if self.min_build_time_years is not None and math.isfinite(self.min_build_time_years):
            lines.append(f"  Flow/power axes would pass with a build time of {self.min_build_time_years:.3g} yr.")
        elif self.min_build_time_years is not None:
            lines.append("  No build time fixes this: the mass inventory is the limit (time does not create mass).")
        return "\n".join(lines)

    def bibtex(self) -> str:
        key = f"noarco_{self.input_hash[:12]}"
        return (f"@software{{{key},\n  title = {{NOArCO {self.version} screening verdict: {self.body} to {self.target}}},\n"
                f"  version = {{{self.version}}},\n  note = {{input\\_hash {self.input_hash}; constants\\_hash "
                f"{self.constants_hash}; code\\_hash {self.code_hash}}}\n}}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "version": self.version, "model_class": self.model_class,
            "body": self.body, "target": self.target, "verdict": {
                "viable": self.viable, "limiting_axis": self.limiting_axis, "pi_min": self.pi_min, "pi": self.pi},
            "requirements": self.requirements, "inputs": self.inputs, "input_hash": self.input_hash,
            "constants_hash": self.constants_hash, "radiative_mode": self.radiative_mode,
            "axis_confidence": self.axis_confidence, "warnings": self.warnings,
            "required": self.required, "shortfall": self.shortfall, "min_build_time_years": self.min_build_time_years,
            "constants_used": self.constants_used, "code_hash": self.code_hash, "body_assumptions": self.body_assumptions,
            "scope": "Screening verdict from closed-form requirements; not a climate simulation.",
        }

    def to_json(self) -> str:
        return json.dumps(_finite(self.to_dict()), sort_keys=True, indent=2, default=str, allow_nan=False)

    def cite(self) -> str:
        return (f"NOArCO {self.version} verdict {self.input_hash[:16]} (constants {self.constants_hash[:8]}): "
                f"{self.body} -> {self.target}: "
                f"{'viable' if self.viable else 'not viable' if self.viable is False else 'undetermined'}"
                f"{f', limited by {self.limiting_axis}' if self.limiting_axis else ''}.")


def verdict(
    body: Any,
    target: DesiredState,
    *,
    build_time_years: float = 1000.0,
    available_inventory_kg: float | None = None,
    available_power_w: float | None = None,
    available_mass_flow_kg_s: float | None = None,
    available_forcing_wm2: float | None = None,
    available_delta_tau_ir: float | None = None,
    replenishment_kg_s: float | None = None,
    loss_kg_s: float | None = None,
    radiative_mode: str = "simple",
) -> Verdict:
    """Conjunctive viability verdict for ``target`` on any body description (see :mod:`noarco.bodies`).

    An axis is evaluated only if its availability is supplied (``None`` = unknown, not assumed).
    """
    confidence = axis_confidence(radiative_mode)
    resolved: ResolvedBody = resolve_body(body)
    planet = resolved.planet
    req = endpoint_requirements(planet, target, build_time_years)
    pi: PiAssessment = assess(
        req, available_inventory_kg=available_inventory_kg, available_mass_flow_kg_s=available_mass_flow_kg_s,
        available_power_w=available_power_w, available_forcing_wm2=available_forcing_wm2,
        available_delta_tau_ir=available_delta_tau_ir, replenishment_kg_s=replenishment_kg_s, loss_kg_s=loss_kg_s)
    warnings: list[str] = []
    if target.target_mean_temperature_k is not None and radiative_mode == "simple":
        warnings.append(
            "Absolute surface-temperature target requested in radiative_mode='simple': the forcing axis is "
            "low-confidence (simple radiative model is biased low versus GCMs); do not use it as a hard "
            "gate. Use radiative_mode='literature_calibrated' (Mars) or a GCM."
        )
    undetermined = [a for a in ("mass", "power", "throughput", "forcing") if pi.values.get(a) is None]
    if undetermined:
        warnings.append(f"Axes not evaluated (no availability supplied): {', '.join(undetermined)}.")
    availability: dict[str, float] = {k: v for k, v in {
        "inventory_kg": available_inventory_kg, "power_w": available_power_w, "mass_flow_kg_s": available_mass_flow_kg_s,
        "forcing_wm2": available_forcing_wm2, "delta_tau_ir": available_delta_tau_ir,
        "replenishment_kg_s": replenishment_kg_s, "loss_kg_s": loss_kg_s}.items() if v is not None}
    inputs = {
        "planet": planet.model_dump(mode="json"), "target": target.model_dump(mode="json"),
        "build_time_years": build_time_years, "radiative_mode": radiative_mode,
        "availability": availability,
    }
    h = hashlib.sha256(json.dumps({"v": __version__, "in": inputs}, sort_keys=True, default=str).encode()).hexdigest()
    required, shortfall, t_min = _headroom(req, pi, build_time_years, availability)
    return Verdict(
        body=planet.body_name, target=target.habitability_tier.value, viable=pi.feasible,
        limiting_axis=pi.binding_constraint, pi_min=pi.pi_min, pi=pi.values,
        requirements={k: v for k, v in req.__dict__.items() if isinstance(v, float)},
        inputs=inputs, input_hash=h, body_assumptions=list(resolved.assumptions), constants_hash=registry_hash(),
        radiative_mode=radiative_mode, axis_confidence=confidence, warnings=warnings,
        required=required, shortfall=shortfall, min_build_time_years=t_min,
        constants_used=_constants_used(list(pi.values)), code_hash=code_hash(),
    )


def _headroom(req: Any, pi: PiAssessment, build_time_years: float, avail: dict[str, float]
              ) -> tuple[dict[str, float], dict[str, float], float | None]:
    """Availability needed per axis, shortfall of what was supplied, and the build time that clears flow/power."""
    supplied = {"mass": avail.get("inventory_kg"), "throughput": avail.get("mass_flow_kg_s"),
                "power": avail.get("power_w"), "forcing": avail.get("forcing_wm2")}
    required: dict[str, float] = {}
    shortfall: dict[str, float] = {}
    for axis, (field_name, _unit) in _AXIS_REQUIREMENT.items():
        have = supplied[axis]
        if have is None:
            continue
        need = float(getattr(req, field_name))
        required[axis] = need
        shortfall[axis] = max(need - have, 0.0)
    vals = pi.values
    if not vals:
        return required, shortfall, None
    if vals.get("mass") is not None and vals["mass"] < 1.0:
        return required, shortfall, math.inf
    ratios = [vals[a] for a in ("throughput", "power") if vals.get(a) is not None and math.isfinite(vals[a])]
    # mean flow and mean power scale as 1/t, so Pi_flow, Pi_power scale as t: t_min = t / min(Pi)
    t_min = build_time_years / min(ratios) if ratios and min(ratios) > 0 else None
    return required, shortfall, t_min


@dataclass
class ReplayResult:
    matches: bool
    differences: list[str]
    same_environment: bool          # same package version, constants hash and code hash as the record


def replay(record: dict[str, Any]) -> ReplayResult:
    """Re-run a ``noarco-result`` record from its own ``inputs`` and compare every number.

    ``matches`` is the scientific check (same inputs -> same verdict, hash and requirements).
    ``same_environment`` says whether the code and constants are the ones that produced the record; a
    record made by another version can legitimately differ, and the differences are listed.
    """
    from noarco.core.desired_state import DesiredState
    from noarco.core.planet_state import PlanetState

    inp = record["inputs"]
    kw = {f"available_{k}" if k in ("inventory_kg", "power_w", "mass_flow_kg_s", "forcing_wm2", "delta_tau_ir") else k: v
          for k, v in inp["availability"].items()}
    v = verdict(PlanetState.model_validate(inp["planet"]), DesiredState.model_validate(inp["target"]),
                build_time_years=inp["build_time_years"], radiative_mode=inp.get("radiative_mode", "simple"), **kw)
    new = json.loads(v.to_json())
    old = json.loads(json.dumps(_finite(record), default=str))
    diffs = [k for k in ("input_hash", "verdict", "requirements", "required", "shortfall", "min_build_time_years")
             if new.get(k) != old.get(k)]
    env = (new["version"] == old.get("version") and new["constants_hash"] == old.get("constants_hash")
           and new["code_hash"] == old.get("code_hash"))
    return ReplayResult(matches=not diffs, differences=diffs, same_environment=env)


def result_schema() -> dict[str, Any]:
    """JSON Schema (draft 2020-12) of a ``noarco-result`` record."""
    num_or_null = {"type": ["number", "null"]}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": SCHEMA, "type": "object",
        "required": ["schema", "version", "model_class", "body", "target", "verdict", "requirements", "inputs",
                     "input_hash", "constants_hash", "code_hash", "radiative_mode", "axis_confidence", "warnings"],
        "properties": {
            "schema": {"const": SCHEMA}, "version": {"type": "string"}, "model_class": {"type": "string"},
            "body": {"type": "string"}, "target": {"type": "string"},
            "verdict": {"type": "object", "required": ["viable", "limiting_axis", "pi_min", "pi"], "properties": {
                "viable": {"type": ["boolean", "null"]}, "limiting_axis": {"type": ["string", "null"], "enum": [*AXES, None]},
                "pi_min": num_or_null, "pi": {"type": "object", "propertyNames": {"enum": list(AXES)},
                       "additionalProperties": {"type": ["number", "string"], "description": "\"inf\" = nothing required"}}}},
            "requirements": {"type": "object"}, "inputs": {"type": "object", "required": ["planet", "target", "availability"]},
            "input_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "constants_hash": {"type": "string"}, "code_hash": {"type": "string"},
            "radiative_mode": {"enum": list(RADIATIVE_MODES)},
            "axis_confidence": {"type": "object", "propertyNames": {"enum": list(AXES)},
                                "additionalProperties": {"enum": ["high", "medium", "low"]}},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "required": {"type": "object"}, "shortfall": {"type": "object"}, "min_build_time_years": {"type": ["number", "string", "null"], "description": "\"inf\" = mass-limited"},
            "constants_used": {"type": "array", "items": {"type": "object",
                                                         "required": ["name", "value", "unit", "source", "class"]}},
            "body_assumptions": {"type": "array", "items": {"type": "string"}}, "scope": {"type": "string"},
        },
    }


def validate_record(record: dict[str, Any]) -> list[str]:
    """Structural check of a record against :func:`result_schema` (required keys, types, enums); ``[]`` = valid."""
    schema = result_schema()
    problems = [f"missing key: {k}" for k in schema["required"] if k not in record]
    if record.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    v = record.get("verdict", {})
    for k in schema["properties"]["verdict"]["required"]:
        if k not in v:
            problems.append(f"missing verdict.{k}")
    if v.get("limiting_axis") not in (*AXES, None):
        problems.append("verdict.limiting_axis not an allowed axis")
    if v.get("viable") not in (True, False, None):
        problems.append("verdict.viable must be boolean or null")
    h = record.get("input_hash", "")
    if not (isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)):
        problems.append("input_hash must be a 64-hex SHA-256")
    if record.get("radiative_mode") not in RADIATIVE_MODES:
        problems.append("radiative_mode not allowed")
    for a, c in record.get("axis_confidence", {}).items():
        if a not in AXES or c not in ("high", "medium", "low"):
            problems.append(f"bad axis_confidence entry {a}={c}")
    return problems
