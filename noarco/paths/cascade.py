"""
noarco.paths.cascade — end-state transition graph and multi-objective path selection
====================================================================================

Builds a directed graph over the *global* end states of Turyshev (2026), Sec. II.A,
and enumerates the paths between them:

    E0 -> E1 -> E3 -> E4   (plus the skip edges E0->E3, E0->E4, E1->E4)

E2 (protected agriculture in enclosures) is a regional deployment-area problem with
no global pressure or temperature requirement, so it is **not** a node of this
planetary graph (it remains in :data:`noarco.core.endpoints.MARS_ENDPOINTS`).

Each edge is computed from the physical difference between its end states:

* **Pressure** ``dP`` -> gas mass ``K dP`` (hydrostatic). Endogenous CO2 is used first, up to
  the accessible reservoir (:mod:`noarco.mechanisms.co2_mobilization`); any residual is an
  explicit *import requirement* (``import_mass_kg``) and is flagged, never hidden.
* **Temperature** ``dT`` left after the CO2 contribution -> the lowest-mass thermal
  mechanism among the supplied aerosol / PFC / mirror mechanisms.
* **Oxygen** -> water electrolysis when an :class:`Electrolysis` mechanism is supplied.

Per edge the cheapest (lowest total mass) thermal option is selected; this greedy rule is
a documented modelling choice, not an optimality proof. Paths are then compared on time,
mass and risk, and the non-dominated set on (time, mass) is reported.

References
----------
Turyshev, S. G. (2026). arXiv:2603.00402, Sec. II.A, III, V, VI, VII.
Deb, K. (2001). Multi-Objective Optimization using Evolutionary Algorithms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import pairwise

import networkx as nx
import numpy as np

from noarco.core.constraints import ConstraintSet
from noarco.core.desired_state import DesiredState
from noarco.core.endpoints import MARS_ENDPOINTS
from noarco.core.planet_state import PlanetState
from noarco.mechanisms.aerosols import NanoparticleAerosol
from noarco.mechanisms.base import Mechanism, MechanismResult
from noarco.mechanisms.co2_mobilization import _MARS_CO2_INVENTORY, CO2Mobilization
from noarco.mechanisms.electrolysis import Electrolysis
from noarco.mechanisms.greenhouse_gas import SuperGreenhouseGas
from noarco.mechanisms.mirrors import OrbitalMirror

logger = logging.getLogger("noarco.paths")

#: Global transitions considered (see module docstring).
TRANSITIONS: tuple[tuple[str, str], ...] = (
    ("E0", "E1"), ("E1", "E3"), ("E3", "E4"), ("E0", "E3"), ("E1", "E4"), ("E0", "E4"),
)
_THERMAL = (NanoparticleAerosol, SuperGreenhouseGas, OrbitalMirror)


@dataclass
class CascadeStep:
    from_endpoint: str
    to_endpoint: str
    mechanism_name: str
    duration_years: float
    mass_kg: float
    power_w: float
    risk_score: float
    result: MechanismResult
    import_mass_kg: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class TerraformingPath:
    name: str
    steps: list[CascadeStep]
    total_time_years: float
    total_mass_kg: float
    peak_power_w: float
    mean_risk: float
    is_pareto_optimal: bool = False
    total_import_kg: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== TerraformingPath: {self.name} ===",
            f"  Total Duration : {self.total_time_years:,.1f} years",
            f"  Total Mass     : {self.total_mass_kg:.3e} kg",
            f"  Peak Power     : {self.peak_power_w:.3e} W ({self.peak_power_w/1e12:.2f} TW)",
            f"  Mean Risk Index: {self.mean_risk:.2f} / 1.0",
            f"  Steps ({len(self.steps)}):",
        ]
        for idx, s in enumerate(self.steps, 1):
            lines.append(
                f"    {idx}. {s.from_endpoint} -> {s.to_endpoint} via {s.mechanism_name} "
                f"[{s.duration_years:,.1f} yr, {s.mass_kg:.2e} kg, Risk: {s.risk_score:.2f}]"
            )
        return "\n".join(lines)


class PathFinder:
    """Cascade generator and multi-objective Pareto pathfinder."""

    def __init__(
        self,
        planet: PlanetState,
        target: DesiredState,
        constraints: ConstraintSet | None = None,
        available_mechanisms: list[Mechanism] | None = None,
    ) -> None:
        self.planet = planet
        self.target = target
        self.constraints = constraints or ConstraintSet()
        self.mechanisms = available_mechanisms or []
        self.graph = nx.DiGraph()

    def _accessible_co2_kg(self) -> float:
        polar = self.planet.co2_ice_kg or 0.0
        if self.planet.body_name == "Mars":
            return max(polar, _MARS_CO2_INVENTORY["accessible_reference_kg"])
        return polar

    def _edge(self, src: str, dst: str) -> dict[str, object] | None:
        """Compute one transition, or ``None`` if it is excluded by the constraints."""
        ep_s, ep_d = MARS_ENDPOINTS[src], MARS_ENDPOINTS[dst]
        d_t = max(ep_d.min_temperature_k - ep_s.min_temperature_k, 0.0)
        d_p = max(ep_d.min_pressure_pa - ep_s.min_pressure_pa, 0.0)
        state = self.planet.model_copy(update={
            "surface_pressure_pa": max(self.planet.surface_pressure_pa, ep_s.min_pressure_pa),
            "mean_temperature_k": max(self.planet.mean_temperature_k, ep_s.min_temperature_k),
        })
        mass_gas = d_p * state.surface_area_m2 / state.gravity_ms2
        names: list[str] = []
        warnings: list[str] = []
        mass = 0.0
        duration = 0.0
        power = 0.0
        risk = 0.0
        isru_min = 1.0
        import_mass = 0.0
        co2_used = 0.0
        primary: MechanismResult | None = None
        d_t_left = d_t

        if mass_gas > 0:
            used = min(mass_gas, self._accessible_co2_kg())
            if used > 0:
                co2 = CO2Mobilization(co2_inventory_kg=used)
                res = co2.compute(state, target_delta_t_k=max(d_t, 1.0))
                names.append(co2.name)
                co2_used = used
                duration = max(duration, res.deployment_time_years)
                power = max(power, res.power_w)
                risk = max(risk, res.risk_index)
                primary = res
                warnings += res.warnings
                d_t_left = max(d_t - res.delta_temperature_k, 0.0)
            import_mass = max(mass_gas - used, 0.0)
            if import_mass > 0:
                isru_min = 0.0
                warnings.append(f"Requires importing {import_mass:.2e} kg of gas (endogenous CO2 exhausted).")

        if d_t_left > 0.5:
            best: MechanismResult | None = None
            best_name = ""
            for mech in self.mechanisms:
                if not isinstance(mech, _THERMAL) or not mech.is_applicable(state):
                    continue
                try:
                    res = mech.compute(state, target_delta_t_k=d_t_left)
                except Exception as exc:  # noqa: BLE001 - skip candidates the mechanism cannot evaluate
                    logger.debug("Skipping %s: %s", mech.name, exc)
                    continue
                if any("UNREACHABLE" in w for w in res.warnings):
                    continue
                if best is None or res.total_mass_kg < best.total_mass_kg:
                    best, best_name = res, mech.name
            if best is None:
                return None
            names.append(best_name)
            mass += best.total_mass_kg
            duration = max(duration, best.deployment_time_years)
            power = max(power, best.power_w)
            risk = max(risk, best.risk_index)
            isru_min = min(isru_min, best.isru_fraction)
            primary = best
            warnings += best.warnings

        o2_needed = max(ep_d.min_o2_pa - ep_s.min_o2_pa, 0.0)
        if o2_needed > 0:
            ele = next((m for m in self.mechanisms if isinstance(m, Electrolysis)), None)
            if ele is None:
                return None
            ele = Electrolysis(mode=ele.mode, available_power_w=ele.available_power_w,
                               electrolysis_efficiency=ele.electrolysis_efficiency,
                               photosynthesis_area_m2=ele.photosynthesis_area_m2,
                               target_o2_partial_pressure_pa=ep_d.min_o2_pa)
            if not ele.is_applicable(state):
                return None
            res = ele.compute(state)
            names.append(ele.name)
            o2_mass = ele._o2_mass_needed(state)
            mass += o2_mass
            duration = max(duration, res.deployment_time_years)
            power = max(power, res.power_w)
            risk = max(risk, res.risk_index)
            primary = primary or res
            warnings += res.warnings

        if primary is None:
            return None
        if self.constraints.max_time_years and duration > self.constraints.max_time_years:
            return None
        if self.constraints.max_power_w and power > self.constraints.max_power_w:
            return None
        if self.constraints.isru_only and (import_mass > 0 or isru_min < 0.3):
            return None
        return {
            "mechanism": " + ".join(names), "duration_years": duration, "mass_kg": mass,
            "power_w": power, "risk": risk, "result": primary, "import_mass_kg": import_mass,
            "warnings": list(dict.fromkeys(warnings)), "gas_mass_kg": mass_gas,
            "co2_used_kg": co2_used, "mass_other_kg": mass,
        }

    def build_transition_graph(self) -> nx.DiGraph:
        """Construct the directed graph of global end-state transitions (see module docstring)."""
        self.graph.clear()
        for ep in MARS_ENDPOINTS:
            self.graph.add_node(ep, data=MARS_ENDPOINTS[ep])
        for src, dst in TRANSITIONS:
            edge = self._edge(src, dst)
            if edge is not None:
                self.graph.add_edge(src, dst, **edge)
        return self.graph

    def find_all_paths(self, start: str = "E0", goal: str = "E3") -> list[TerraformingPath]:
        """Find all feasible paths from start endpoint to goal."""
        if not self.graph.edges:
            self.build_transition_graph()

        if not nx.has_path(self.graph, start, goal):
            return []

        all_paths: list[TerraformingPath] = []
        raw_paths = list(nx.all_simple_paths(self.graph, start, goal))

        path_idx = 1
        for r_path in raw_paths:
            # Collect edge combinations along path
            edge_options = []
            for u, v in pairwise(r_path):
                edges_data = [self.graph[u][v]] # In simple DiGraph
                edge_options.append((u, v, edges_data[0]))

            steps = []
            tot_time = 0.0
            tot_mass = 0.0
            peak_pwr = 0.0
            risks = []

            remaining_co2 = self._accessible_co2_kg()
            for u, v, data in edge_options:
                # Endogenous CO2 is one reservoir shared by the whole path: allocate it
                # cumulatively so that later edges cannot reuse what earlier ones consumed.
                used = min(data["gas_mass_kg"], remaining_co2)
                remaining_co2 -= used
                import_mass = data["gas_mass_kg"] - used
                step_warnings = list(data["warnings"])
                if import_mass > 0 and not any("importing" in w for w in step_warnings):
                    step_warnings.append(f"Requires importing {import_mass:.2e} kg of gas.")
                step = CascadeStep(
                    from_endpoint=u,
                    to_endpoint=v,
                    mechanism_name=data["mechanism"],
                    duration_years=data["duration_years"],
                    mass_kg=data["mass_other_kg"] + data["gas_mass_kg"],  # gas (endogenous + imported) + hardware
                    power_w=data["power_w"],
                    risk_score=data["risk"],
                    result=data["result"],
                    import_mass_kg=import_mass,
                    warnings=step_warnings,
                )
                steps.append(step)
                tot_time += step.duration_years
                tot_mass += step.mass_kg
                peak_pwr = max(peak_pwr, step.power_w)
                risks.append(step.risk_score)

            all_paths.append(
                TerraformingPath(
                    name=f"Cascade-{path_idx}: {' -> '.join(r_path)} via {steps[0].mechanism_name}",
                    steps=steps,
                    total_time_years=tot_time,
                    total_mass_kg=tot_mass,
                    peak_power_w=peak_pwr,
                    mean_risk=float(np.mean(risks)) if risks else 0.0,
                    total_import_kg=sum(st.import_mass_kg for st in steps),
                )
            )
            path_idx += 1

        return all_paths

    def find_pareto_front(self, paths: list[TerraformingPath]) -> list[TerraformingPath]:
        """Filter paths to the non-dominated Pareto front (minimising time and total mass moved).

        Total mass includes the gas that must be delivered (endogenous and imported).
        """
        pareto: list[TerraformingPath] = []
        for p1 in paths:
            dominated = False
            for p2 in paths:
                if (p2.total_time_years <= p1.total_time_years and p2.total_mass_kg <= p1.total_mass_kg) and \
                   (p2.total_time_years < p1.total_time_years or p2.total_mass_kg < p1.total_mass_kg):
                    dominated = True
                    break
            if not dominated:
                p1.is_pareto_optimal = True
                pareto.append(p1)
        return pareto
