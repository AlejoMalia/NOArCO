"""
noarco.exchange — hand surviving scenarios to a full climate model
==================================================================

NOArCO is a *filter*: it removes scenarios whose inventory, power, throughput or forcing requirements
already fail the rubric, and passes the rest to a model that can do transient physics (a GCM). This
module writes the scenarios that survive in a neutral, self-describing format
(``noarco-scenario/1.0`` JSON, plus an index CSV) with units, provenance and an input hash.

It does **not** emit model-specific namelists: variable names differ between GCMs and guessing them
would be wrong. Each file lists the quantities a GCM needs (surface pressure, composition changes,
absorbed-flux forcing, added longwave optical depth, initial state) so a short, explicit mapping to the
target model can be written and reviewed once.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from noarco.bodies import ResolvedBody, resolve_body
from noarco.core.desired_state import DesiredState
from noarco.feasibility import assess, endpoint_requirements

SCHEMA = "noarco-scenario/1.0"


def scenario_dict(
    body: Any,
    target: DesiredState,
    *,
    label: str,
    build_time_years: float = 1000.0,
    available_power_w: float = 1.0e10,
    available_inventory_kg: float | None = None,
) -> dict[str, Any]:
    """Self-describing scenario for one target on one body (plain Python types)."""
    resolved: ResolvedBody = resolve_body(body)
    planet = resolved.planet
    req = endpoint_requirements(planet, target, build_time_years)
    from noarco.engines.pathfinder import PathEngine
    inv = available_inventory_kg if available_inventory_kg is not None else (
        PathEngine._accessible_co2_kg(planet) if req.atmosphere_mass_kg > 0 else None)
    rub = assess(req, available_inventory_kg=inv, available_power_w=available_power_w if req.mean_power_w > 0 else None)
    return {
        "schema": SCHEMA,
        "label": label,
        "units": "SI",
        "initial_state": {
            "body": planet.body_name, "surface_pressure_pa": planet.surface_pressure_pa,
            "mean_surface_temperature_k": planet.mean_temperature_k, "gravity_ms2": planet.gravity_ms2,
            "radius_m": planet.radius_m, "bond_albedo": planet.surface_albedo,
            "solar_constant_wm2": planet.solar_constant_wm2, "composition_mole_fractions": dict(planet.gas_composition.species),
            "ir_optical_depth_estimate": planet.ir_optical_depth,
        },
        "target_state": {
            "tier": target.habitability_tier.value, "min_surface_pressure_pa": target.min_surface_pressure_pa,
            "target_mean_temperature_k": target.target_mean_temperature_k,
            "min_o2_partial_pressure_pa": target.min_o2_partial_pressure_pa,
            "max_co2_partial_pressure_pa": target.max_co2_partial_pressure_pa, "enclosed_only": target.enclosed,
        },
        "forcing_for_gcm": {
            "atmosphere_mass_to_add_kg": req.atmosphere_mass_kg, "o2_mass_to_add_kg": req.o2_mass_kg,
            "absorbed_flux_forcing_wm2": req.delta_forcing_toa_wm2,
            "added_longwave_optical_depth_grey": req.delta_tau_ir,
            "note": "Two alternative routes to the same surface target: direct absorbed-flux (mirrors/albedo) or added "
                    "longwave opacity (aerosols/gases). Grey mapping: +/-30-50 % on optical depth.",
        },
        "screening": {
            "build_time_years": build_time_years, "available_power_w": available_power_w,
            "available_inventory_kg": inv, "pi": rub.values, "pi_min": rub.pi_min,
            "binding_constraint": rub.binding_constraint, "survives": rub.feasible is not False,
        },
        "provenance": {
            "body_derived": resolved.derived, "body_assumptions": resolved.assumptions,
            "planet_source_type": planet.source_type.value, "planet_reference": planet.reference,
        },
        "limitations": [
            "NOArCO screens with closed-form requirements; it does not simulate transients or feedbacks.",
            "Intervals/probabilities are available from noarco.projection (parametric + structural).",
        ],
    }


def export_candidates(
    body: Any,
    targets: dict[str, DesiredState],
    directory: str | Path,
    **kwargs: Any,
) -> list[Path]:
    """Write one JSON file per *surviving* target and an ``index.csv`` of all screened targets.

    Returns the paths of the scenario files that were written (the survivors).
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    rows = []
    for label, tgt in targets.items():
        sc = scenario_dict(body, tgt, label=label, **kwargs)
        keep = bool(sc["screening"]["survives"])
        if keep:
            f = out_dir / f"{label}.json"
            f.write_text(json.dumps(sc, indent=2, sort_keys=True, default=str))
            written.append(f)
        rows.append({"label": label, "survives": keep, "pi_min": sc["screening"]["pi_min"],
                     "binding_constraint": sc["screening"]["binding_constraint"]})
    with open(out_dir / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "survives", "pi_min", "binding_constraint"])
        w.writeheader()
        w.writerows(rows)
    return written
