"""
noarco.assurance.audit — One-call V&V assessment of a planetary scenario
========================================================================

``assure(planet, target)`` runs, in order: physical invariants (verification),
model-validity envelopes, first-order uncertainty propagation of the headline
outputs with a Monte-Carlo linearity cross-check, and stamps a reproducibility
manifest. The overall status is the worst finding, so a FAIL can gate a
pipeline in CI or before a report is issued.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from noarco.assurance import invariants, validity
from noarco.assurance.findings import Finding, Severity, worst
from noarco.assurance.manifest import Manifest, build_manifest
from noarco.assurance.uncertainty import PropagationResult, UncertainValue, propagate
from noarco.core.constants import SIGMA_SB as _SIGMA
from noarco.core.desired_state import DesiredState
from noarco.core.planet_state import PlanetState, SourceType

# Default 1-sigma *relative* uncertainty by epistemic status of the state.
# These are conservative engineering defaults, not measurements: override them
# with ``rel_sigma`` when a real error budget exists.
DEFAULT_REL_SIGMA: dict[SourceType, float] = {
    SourceType.MEASURED: 0.01,
    SourceType.MODELED: 0.05,
    SourceType.ESTIMATED: 0.20,
    SourceType.ASSUMED: 0.50,
}


@dataclass
class AssuranceReport:
    planet: str
    status: Severity
    findings: list[Finding]
    outputs: dict[str, PropagationResult]
    manifest: Manifest
    units: dict[str, str] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return {Severity.INFO: "PASS", Severity.CAUTION: "CAUTION", Severity.FAIL: "FAIL"}[self.status]

    @property
    def passed(self) -> bool:
        return self.status < Severity.FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "planet": self.planet,
            "status": self.verdict,
            "findings": [{"code": f.code, "severity": f.severity.name, "message": f.message}
                         for f in self.findings],
            "outputs": {
                k: {"nominal": r.output.nominal, "sigma": r.output.sigma,
                    "unit": self.units.get(k, ""), "linearity_ok": r.linearity_ok,
                    "variance_share": r.contributions}
                for k, r in self.outputs.items()
            },
            "manifest": self.manifest.to_dict(),
        }

    def to_markdown(self) -> str:
        lines = [f"# Assurance report — {self.planet}", f"**Status: {self.verdict}**", ""]
        if self.findings:
            lines += ["## Findings", *(f"- {f}" for f in self.findings), ""]
        lines += ["## Uncertainty (1σ, first-order + MC cross-check)", "",
                  "| Output | Value | σ | Rel. | Linear OK | Dominant input |", "|---|---|---|---|---|---|"]
        for k, r in self.outputs.items():
            dom = next(iter(r.contributions), "-")
            lines.append(f"| {k} [{self.units.get(k, '')}] | {r.output.nominal:.6g} | "
                         f"{r.output.sigma:.3g} | {r.output.relative:.2%} | {r.linearity_ok} | {dom} |")
        lines += ["", f"Manifest content hash: `{self.manifest.content_hash}`",
                  f"Seal: `{self.manifest.seal}`"]
        return "\n".join(lines)


def _dump(obj: object | None) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return repr(obj)


def _planet_inputs(planet: PlanetState, rel_sigma: float) -> dict[str, UncertainValue]:
    vals = {
        "solar_constant_wm2": planet.solar_constant_wm2,
        "surface_albedo": planet.surface_albedo,
        "gravity_ms2": planet.gravity_ms2,
        "radius_m": planet.radius_m,
        "ir_optical_depth": planet.ir_optical_depth,
    }
    return {k: UncertainValue(v, abs(v) * rel_sigma) for k, v in vals.items()}


def assure(
    planet: PlanetState,
    target: DesiredState | None = None,
    *,
    rel_sigma: float | None = None,
    mc_samples: int = 2000,
    seed: int = 20260930,
) -> AssuranceReport:
    """Run the full V&V pipeline on ``planet`` (and optionally a target)."""
    findings: list[Finding] = []
    findings += invariants.check_planet_invariants(planet)
    findings += validity.check_planet_validity(planet)

    if target is not None and target.target_mean_temperature_k is not None:
        p_tgt = None if target.enclosed else target.min_surface_pressure_pa
        findings += validity.check_forcing_request(planet, planet.surface_pressure_pa, p_tgt)

    sigma_rel = DEFAULT_REL_SIGMA[planet.source_type] if rel_sigma is None else rel_sigma
    if sigma_rel < 0:
        raise ValueError("rel_sigma must be >= 0")
    inputs = _planet_inputs(planet, sigma_rel)

    def t_eq(p: dict[str, float]) -> float:
        return float((p["solar_constant_wm2"] * (1 - p["surface_albedo"]) / (4 * _SIGMA)) ** 0.25)

    def t_surface(p: dict[str, float]) -> float:
        return float(t_eq(p) * (1 + p["ir_optical_depth"] / 2) ** 0.25)

    def kg_per_pa(p: dict[str, float]) -> float:
        return 4 * math.pi * p["radius_m"] ** 2 / p["gravity_ms2"]

    models = {
        "equilibrium_temperature": (t_eq, "K"),
        "greenhouse_surface_temperature": (t_surface, "K"),
        "atmosphere_mass_per_pascal": (kg_per_pa, "kg/Pa"),
    }
    outputs: dict[str, PropagationResult] = {}
    units: dict[str, str] = {}
    for name, (fn, unit) in models.items():
        res = propagate(fn, inputs, mc_samples=mc_samples, seed=seed)
        outputs[name] = res
        units[name] = unit
        if res.linearity_ok is False:
            findings.append(Finding("UQ-NONLINEAR", Severity.CAUTION,
                                    f"{name}: first-order propagation inadequate; use MC sigma"))

    manifest = build_manifest(
        f"assure:{planet.body_name}",
        inputs={"planet": planet.model_dump(mode="json"),
                "target": _dump(target),
                "rel_sigma": sigma_rel, "mc_samples": mc_samples},
        seed=seed,
    )
    return AssuranceReport(planet.body_name, worst(findings), findings, outputs, manifest, units)
