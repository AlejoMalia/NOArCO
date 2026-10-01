"""
noarco.workflow — end-to-end scenario assessment with TRIADA/MATE compute savings
=================================================================================

``run_assessment(planet, target)`` runs the whole chain in one call:

1. requirements (mass, O2, energy, forcing / optical depth, throughput, power);
2. the conjunctive feasibility rubric ``Pi_min`` (Turyshev 2026);
3. pathway allocation and, when the target is reachable, the path graph;
4. V&V (:func:`noarco.assurance.assure`) and mechanism-credibility findings;
5. a reproducibility manifest.

Compute savings (the TRIADA protocol):

* **T1 - inventory.** Results are cached by a hash of the inputs; repeating a scenario
  costs a dictionary lookup.
* **T2/T3 - closed form + MATE projection.** If the rubric already proves the endpoint
  infeasible by a margin (``Pi_min < projection_threshold`` on a hard inventory or power
  constraint), the verdict cannot change with more computation, so the expensive path
  graph is skipped and the result is certified as *projected*, with the basis recorded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from noarco.assurance import assure
from noarco.assurance.findings import Finding, Severity
from noarco.assurance.invariants import check_pathway_invariants
from noarco.assurance.validity import check_mechanism_result
from noarco.bodies import resolve_body
from noarco.core.constraints import ConstraintSet
from noarco.core.desired_state import DesiredState
from noarco.core.planet_state import PlanetState
from noarco.engines.pathfinder import OptimizedPathway, PathEngine
from noarco.extratools.exoplanet import ExoplanetProfile
from noarco.feasibility import EndpointRequirements, PiAssessment, assess, endpoint_requirements
from noarco.mate.core import MATEStatus, NOArCoCache
from noarco.mechanisms import (
    CO2Mobilization,
    Electrolysis,
    NanoparticleAerosol,
    OrbitalMirror,
    SuperGreenhouseGas,
)
from noarco.mechanisms.base import Mechanism
from noarco.paths.cascade import PathFinder, TerraformingPath
from noarco.projection import Projection, project

_CACHE = NOArCoCache(maxsize=128)


@dataclass
class AssessmentReport:
    planet: str
    status: MATEStatus
    requirements: EndpointRequirements
    rubric: PiAssessment
    pathway: OptimizedPathway | None
    paths: list[TerraformingPath]
    pareto: list[TerraformingPath]
    findings: list[Finding]
    verdict: str
    projection_basis: str | None = None
    projection: Projection | None = None
    input_hash: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        r, pi = self.requirements, self.rubric
        lines = [
            f"# Scenario assessment - {self.planet}",
            f"**Verdict: {self.verdict}** (compute status: {self.status.value})",
            "",
            (f"- Atmosphere mass to add: {r.atmosphere_mass_kg:.3e} kg; O2: {r.o2_mass_kg:.3e} kg "
            f"(E_min {r.o2_min_energy_j:.3e} J)"),
            f"- Forcing (direct): {r.delta_forcing_toa_wm2:.1f} W/m2; added IR optical depth: {r.delta_tau_ir:.2f}",
            (f"- Mean flow {r.mean_mass_flow_kg_s:.3e} kg/s, mean power {r.mean_power_w:.3e} W over "
            f"{r.build_time_years:.0f} yr"),
            f"- Pi_min = {pi.pi_min if pi.pi_min is not None else 'n/a'} (binding: {pi.binding_constraint})",
        ]
        if self.projection is not None:
            pj = self.projection
            lines += ["", f"Probabilistic projection ({pj.confidence:.0%}, n={pj.n_samples}): {pj.statement()}"]
        if self.projection_basis:
            lines += ["", f"MATE projection: {self.projection_basis}"]
        if self.findings:
            lines += ["", "## Findings", *(f"- {f}" for f in self.findings)]
        if self.paths:
            lines += ["", f"## Paths ({len(self.paths)}; {len(self.pareto)} Pareto-optimal)"]
        lines += ["", f"Input hash: `{self.input_hash}`"]
        return "\n".join(lines)


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _default_mechanisms() -> list[Mechanism]:
    return [NanoparticleAerosol(), OrbitalMirror(), SuperGreenhouseGas(), CO2Mobilization(), Electrolysis()]


def run_assessment(
    planet: PlanetState | str | dict[str, Any] | ExoplanetProfile,
    target: DesiredState,
    *,
    build_time_years: float = 1000.0,
    available_power_w: float = 1.0e10,
    available_inventory_kg: float | None = None,
    mechanisms: list[Mechanism] | None = None,
    constraints: ConstraintSet | None = None,
    projection_threshold: float = 0.5,
    use_cache: bool = True,
    confidence: float | None = 0.95,
) -> AssessmentReport:
    """Assess a scenario end to end (see module docstring for the compute-saving rules).

    ``planet`` may be a ``PlanetState``, a solar-system body name, an ``ExoplanetProfile`` or a dict of
    conditions describing any world (see :mod:`noarco.bodies`).
    """
    resolved = resolve_body(planet)
    planet = resolved.planet
    mechs = mechanisms if mechanisms is not None else _default_mechanisms()
    key = _hash({
        "planet": planet.model_dump(mode="json"), "target": target.model_dump(mode="json"),
        "t": build_time_years, "p": available_power_w, "inv": available_inventory_kg,
        "mechs": [m.name for m in mechs], "c": (constraints or ConstraintSet()).model_dump(mode="json"),
        "thr": projection_threshold, "conf": confidence,
    })
    if use_cache:
        hit = _CACHE.get(key)
        if hit is not None:
            rep: AssessmentReport = hit.value
            return AssessmentReport(**{**rep.__dict__, "status": MATEStatus.CACHE_HIT})

    req = endpoint_requirements(planet, target, build_time_years)
    inv = available_inventory_kg
    if inv is None:
        inv = PathEngine._accessible_co2_kg(planet) if req.atmosphere_mass_kg > 0 else None
    rubric = assess(req, available_inventory_kg=inv,
                    available_power_w=available_power_w if req.mean_power_w > 0 else None)

    findings: list[Finding] = list(assure(planet, target, mc_samples=0).findings)
    projected = rubric.pi_min is not None and rubric.pi_min < projection_threshold and \
        rubric.binding_constraint in {"mass", "power"}

    pathway: OptimizedPathway | None = None
    paths: list[TerraformingPath] = []
    pareto: list[TerraformingPath] = []
    basis: str | None = None
    if projected:
        status = MATEStatus.PROJECTED
        basis = (f"Pi_min = {rubric.pi_min:.3g} on the {rubric.binding_constraint} constraint is below "
                 f"{projection_threshold}; more computation cannot change the infeasible verdict, so the "
                 "path graph was skipped.")
        verdict = "INFEASIBLE"
    else:
        status = MATEStatus.FULL_COMPUTE
        pathway = PathEngine.optimize_pathway(planet, target, available_power_w, build_time_years)
        findings.extend(check_pathway_invariants(pathway))
        for step in pathway.steps:
            for w in step.warnings:
                findings.append(Finding("PATH-WARNING", Severity.INFO, f"{step.mechanism_name}: {w}"))
        if not target.enclosed:
            pf = PathFinder(planet, target, constraints, mechs)
            paths = pf.find_all_paths("E0", "E4" if target.min_o2_partial_pressure_pa else "E3")
            pareto = pf.find_pareto_front(paths)
            for m in mechs:
                if m.is_applicable(planet):
                    try:
                        findings.extend(check_mechanism_result(m.compute(planet, target_delta_t_k=10.0)))
                    except (ValueError, NotImplementedError):
                        continue
        verdict = {True: "FEASIBLE", False: "INFEASIBLE", None: "UNDETERMINED"}[rubric.feasible]

    proj = None
    if confidence is not None:
        proj = project(planet, target, build_time_years=build_time_years, available_power_w=available_power_w,
                       available_inventory_kg=available_inventory_kg, confidence=confidence)
    report = AssessmentReport(
        planet=planet.body_name, status=status, requirements=req, rubric=rubric, pathway=pathway,
        paths=paths, pareto=pareto, findings=findings, verdict=verdict, projection_basis=basis,
        projection=proj, input_hash=key,
    )
    if resolved.assumptions:
        findings.extend(Finding("BODY-ASSUMPTION", Severity.CAUTION, a) for a in resolved.assumptions)
    if use_cache:
        from noarco.mate.core import MATEResult
        _CACHE.set(key, MATEResult(value=report, status=status))
    return report


def clear_cache() -> int:
    """Drop all cached assessments; returns the number removed."""
    return _CACHE.invalidate()
