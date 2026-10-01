"""
noarco.metrics — engineering (process) quality metrics
======================================================

These measure how well the *process* is controlled — they are **not** climate-accuracy figures.

* ``constants_sourced_pct``: share of registry constants with a named source (every class except
  ``estimated``);
* ``registry_green``: every registry value matches the live constant in the code (``verify_registry``);
* ``anchors_passed`` / ``anchors_total``: external benchmark cases (``noarco.validation.run_validation``);
* ``reproducible``: the same inputs give the same input hash and the same JSON record;
* ``missing_availability_is_undetermined``: no availability supplied -> ``viable is None``.
"""

from __future__ import annotations

from typing import Any


def engineering_metrics() -> dict[str, Any]:
    from noarco.constants_registry import entries, verify_registry
    from noarco.core.desired_state import DesiredState, HabitabilityTier
    from noarco.data.planets import MARS
    from noarco.validation import run_validation
    from noarco.verdict import verdict

    es = entries()
    sourced = [e for e in es if e.klass != "estimated"]
    anchors = run_validation()
    tgt = DesiredState(habitability_tier=HabitabilityTier.E1_TRIPLE_POINT)
    a = verdict(MARS, tgt, available_inventory_kg=7.78e16)
    b = verdict(MARS, tgt, available_inventory_kg=7.78e16)
    none = verdict(MARS, tgt)
    return {
        "constants_total": len(es),
        "constants_sourced_pct": 100.0 * len(sourced) / len(es),
        "registry_green": verify_registry() == [],
        "anchors_passed": sum(r.passed for r in anchors),
        "anchors_total": len(anchors),
        "reproducible": a.input_hash == b.input_hash and a.to_json() == b.to_json(),
        "missing_availability_is_undetermined": none.viable is None,
    }


def to_markdown() -> str:
    m = engineering_metrics()
    return "\n".join([
        "| Engineering metric (process quality, not climate accuracy) | Value |",
        "|---|---|",
        f"| Constants with a named source | {m['constants_sourced_pct']:.0f} % of {m['constants_total']} |",
        f"| `verify_registry` (values match the code) | {'green' if m['registry_green'] else 'RED'} |",
        f"| External anchors reproduced | {m['anchors_passed']}/{m['anchors_total']} |",
        f"| Same inputs -> same JSON | {'yes' if m['reproducible'] else 'NO'} |",
        f"| Missing availability -> `indeterminado` (`viable = None`) | {'yes' if m['missing_availability_is_undetermined'] else 'NO'} |",
    ])
