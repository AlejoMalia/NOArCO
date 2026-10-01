"""
noarco.session — Reactive connected-layer session
===================================================

`NOArCoSession` is the central object for interactive and programmatic
use of NOArCO. It connects ALL layers (inventory, radiative balance,
mechanisms, paths, sim) into a reactive network:

    When PlanetState changes → all dependent computations auto-invalidate
    and are lazily recomputed on next access.

This is the answer to "make all layers connected as a network so that when
a datum is edited, the framework corrects itself."

Design pattern
--------------
- PlanetState is immutable (pydantic). Updates produce a new instance.
- The session holds the current state and a NOArCoCache.
- All physics modules (inventory, radiative, mechanisms) are exposed as
  lazy @properties that check the cache (TRIADA T1) before recomputing.
- `update()` replaces the PlanetState and invalidates the entire cache.
- Individual field updates via `update(field=value)` use pydantic's
  `model_copy(update={...})` to produce a new state efficiently.

MATE integration
----------------
All session properties go through the TRIADA cache (T1: INVENTORY).
The session also exposes `mate_stats()` to inspect compute efficiency.

References
----------
Turyshev (2026) arXiv:2603.00402 — all physics in the connected modules.
Malia, A. (2026). MATE: Methodology of Advancement through Strategic Tension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from noarco.core.constraints import ConstraintSet
from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.core.planet_state import PlanetState
from noarco.inventory.atmosphere import AtmosphericInventory
from noarco.mate.core import MATEResult, MATEStatus, NOArCoCache, mate_project
from noarco.mechanisms.base import Mechanism
from noarco.radiative.balance import RadiativeBalance

_M = TypeVar("_M", bound=BaseModel)

if TYPE_CHECKING:
    from noarco.assurance import AssuranceReport
    from noarco.extratools.exoplanet import ExoplanetProfile


def _validated_update(model: _M, changes: dict[str, Any]) -> _M:
    """Return a copy of ``model`` with ``changes`` applied *and validated*.

    ``BaseModel.model_copy(update=...)`` skips validation, which would let
    out-of-range values (albedo = 5, negative pressure) or misspelled field
    names enter the session silently.
    """
    unknown = sorted(set(changes) - set(type(model).model_fields))
    if unknown:
        raise ValueError(f"Unknown {type(model).__name__} field(s): {unknown}")
    data = {name: getattr(model, name) for name in type(model).model_fields}
    data.update(changes)
    return type(model).model_validate(data)


class NOArCoSession:
    """Reactive multi-layer terraforming planning session.

    All physical modules are connected through a shared cache.
    Editing any part of the planet state auto-invalidates all
    dependent computations — they are re-derived on next access.

    Parameters
    ----------
    planet : PlanetState
        Initial planetary state.
    target : DesiredState, optional
        Target conditions. Defaults to E1 (triple point of water).
    constraints : ConstraintSet, optional
        Operational constraints. Defaults to unconstrained.
    mechanisms : list[Mechanism], optional
        Active intervention mechanisms. If None, all applicable
        mechanisms are auto-selected based on constraints.
    cache_size : int
        Maximum cache entries. Default 512.
    verbose : bool
        If True, print MATE status on each computation.

    Examples
    --------
    >>> from noarco.data.planets import MARS
    >>> from noarco.session import NOArCoSession
    >>> from noarco.core.desired_state import HabitabilityTier
    >>>
    >>> session = NOArCoSession.for_mars(target_tier=HabitabilityTier.E1_TRIPLE_POINT)
    >>> print(session.inventory.mass_per_mbar())       # 3.89e15 (Turyshev 2026)
    >>> print(session.radiative.equilibrium_temperature_k())  # ~209 K
    >>>
    >>> # Edit planet state — everything auto-updates
    >>> session.update(surface_pressure_pa=1220.0)     # double the pressure
    >>> print(session.inventory.mass_per_mbar())       # same formula, new state
    >>> print(session.mate_stats())                    # see what was recomputed
    """

    def __init__(
        self,
        planet: PlanetState,
        target: DesiredState | None = None,
        constraints: ConstraintSet | None = None,
        mechanisms: list[Mechanism] | None = None,
        cache_size: int = 512,
        verbose: bool = False,
    ) -> None:
        self._planet = planet
        self._target = target or DesiredState.from_tier(HabitabilityTier.E1_TRIPLE_POINT)
        self._constraints = constraints or ConstraintSet()
        self._mechanisms: list[Mechanism] = mechanisms or []
        self._cache = NOArCoCache(maxsize=cache_size)
        self.verbose = verbose

        # Lazily-built module instances (invalidated on state change)
        self._inventory: AtmosphericInventory | None = None
        self._radiative: RadiativeBalance | None = None

        # Edit history for audit trail
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Class-level constructors for canonical bodies
    # ------------------------------------------------------------------

    @classmethod
    def for_mars(
        cls,
        target_tier: HabitabilityTier = HabitabilityTier.E1_TRIPLE_POINT,
        constraints: ConstraintSet | None = None,
        **kwargs: Any,
    ) -> NOArCoSession:
        """Create a session pre-configured for Mars E0 baseline.

        Parameters
        ----------
        target_tier : HabitabilityTier
            Target endpoint. Default E1 (triple point of water).
        constraints : ConstraintSet, optional
            Operational constraints.

        Returns
        -------
        NOArCoSession
        """
        from noarco.data.planets import MARS

        target = DesiredState.from_tier(target_tier)
        return cls(planet=MARS, target=target, constraints=constraints, **kwargs)

    @classmethod
    def from_conditions(
        cls,
        body: PlanetState | str | dict[str, Any] | ExoplanetProfile,
        target_tier: HabitabilityTier = HabitabilityTier.E1_TRIPLE_POINT,
        **kwargs: Any,
    ) -> NOArCoSession:
        """Create a session for any body description (name, conditions dict, exoplanet profile)."""
        from noarco.bodies import planet_from

        return cls(planet=planet_from(body), target=DesiredState.from_tier(target_tier), **kwargs)

    @classmethod
    def for_venus(
        cls,
        target_tier: HabitabilityTier = HabitabilityTier.E1_TRIPLE_POINT,
        **kwargs: Any,
    ) -> NOArCoSession:
        """Create a session pre-configured for Venus E0 baseline."""
        from noarco.data.planets import VENUS

        target = DesiredState.from_tier(target_tier)
        return cls(planet=VENUS, target=target, **kwargs)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def planet(self) -> PlanetState:
        """Current PlanetState (immutable)."""
        return self._planet

    @property
    def target(self) -> DesiredState:
        """Current DesiredState."""
        return self._target

    @property
    def constraints(self) -> ConstraintSet:
        """Current ConstraintSet."""
        return self._constraints

    # ------------------------------------------------------------------
    # Reactive connected layers (lazy + cached)
    # ------------------------------------------------------------------

    @property
    def inventory(self) -> AtmosphericInventory:
        """Atmospheric inventory module, auto-bound to current PlanetState.

        Rebuilt automatically when PlanetState changes.
        Results are cached via TRIADA T1 (INVENTORY).
        """
        if self._inventory is None or self._inventory.planet is not self._planet:
            self._inventory = AtmosphericInventory(self._planet)
            if self.verbose:
                print(f"[Session] inventory rebuilt for {self._planet.body_name}")
        return self._inventory

    @property
    def radiative(self) -> RadiativeBalance:
        """Radiative balance module, auto-bound to current PlanetState.

        Rebuilt automatically when PlanetState changes.
        """
        if self._radiative is None or self._radiative.planet is not self._planet:
            self._radiative = RadiativeBalance(self._planet)
            if self.verbose:
                print(f"[Session] radiative rebuilt for {self._planet.body_name}")
        return self._radiative

    @property
    def mechanisms(self) -> list[Mechanism]:
        """List of active intervention mechanisms."""
        return self._mechanisms

    # ------------------------------------------------------------------
    # Verification & validation
    # ------------------------------------------------------------------

    def assure(self, **kwargs: Any) -> AssuranceReport:
        """Run the V&V pipeline (invariants, validity, UQ, manifest) on the current state.

        Keyword arguments are forwarded to :func:`noarco.assurance.assure`.
        """
        from noarco.assurance import assure

        return assure(self._planet, self._target, **kwargs)

    # ------------------------------------------------------------------
    # Delta properties: gap between current and target state
    # ------------------------------------------------------------------

    @property
    def delta_temperature_k(self) -> float | None:
        """Temperature gap between current state and target (K).

        Positive = target is warmer than current.
        Returns None if target temperature is not specified.
        """
        if self._target.target_mean_temperature_k is None:
            return None
        return self._target.target_mean_temperature_k - self._planet.mean_temperature_k

    @property
    def delta_pressure_pa(self) -> float | None:
        """Pressure gap between current state and target (Pa).

        Positive = target pressure is higher than current.
        Returns None if the target pressure is not specified or applies only
        inside enclosures (regional E2-type targets have no global pressure gap).
        """
        if self._target.min_surface_pressure_pa is None or self._target.enclosed:
            return None
        return self._target.min_surface_pressure_pa - self._planet.surface_pressure_pa

    @property
    def forcing_required_wm2(self) -> float | None:
        """Radiative forcing required to reach target temperature (W/m²).

        Exact Stefan-Boltzmann balance at fixed greenhouse opacity, no feedbacks
        (see :meth:`RadiativeBalance.forcing_needed_for_surface_temperature`).
        Returns None if the target temperature is not specified.

        Reference: Turyshev (2026), Eqs. (30)-(31).
        """
        tgt = self._target.target_mean_temperature_k
        if tgt is None:
            return None
        return self.radiative.forcing_needed_for_surface_temperature(tgt)

    @property
    def mass_required_kg(self) -> float | None:
        """Mass of additional atmosphere needed to reach target pressure (kg).

        Returns None if target pressure is not specified.

        Reference: Turyshev (2026), Eq. 1.
        """
        dp = self.delta_pressure_pa
        if dp is None or dp <= 0:
            return None
        return dp * self.inventory.mass_per_pascal()

    @property
    def co2_feasibility(self) -> MATEResult:
        """MATE-annotated CO2 inventory feasibility assessment.

        Applies TRIADA before computing:
            T1: Check cache.
            T2: If gap > 10x available, project infeasibility (MATE R3).
            T3: Otherwise, full gap calculation.

        Returns
        -------
        MATEResult
            .value = dict with 'feasible', 'gap_kg', 'ratio', 'conclusion'
        """
        if self._target.min_surface_pressure_pa is None or self._target.enclosed:
            return MATEResult(
                value={
                    "feasible": None,
                    "conclusion": "No global pressure target (none specified, or enclosed/regional).",
                },
                status=MATEStatus.CLOSED_FORM,
                source_call="NOArCoSession.co2_feasibility",
            )

        from noarco.mechanisms.co2_mobilization import _MARS_CO2_INVENTORY as _INV

        target_pa = self._target.min_surface_pressure_pa
        polar = self._planet.co2_ice_kg or 0.0
        is_mars = self._planet.body_name == "Mars"
        # Published Mars reservoirs (Turyshev 2026 Table III; Jakosky & Edwards 2018;
        # Jakosky 2019). No reference exists for other bodies: use the dataset value only.
        accessible = max(polar, _INV["accessible_reference_kg"]) if is_mars else polar
        crustal_bound = _INV["crustal_upper_bound_kg"] if is_mars else polar
        needed = self.inventory.delta_mass_for_pressure(target_pa)

        pi_polar = polar / needed if needed > 0 else float("inf")
        pi_accessible = accessible / needed if needed > 0 else float("inf")
        pi_crustal = crustal_bound / needed if needed > 0 else float("inf")
        ratio = needed / polar if polar > 0 else float("inf")

        if needed > 0 and needed > crustal_bound:
            # MATE R3: even the extreme crustal-sequestration bound cannot supply the
            # target, so the verdict cannot change with better regolith data.
            return mate_project(
                value={
                    "feasible": False,
                    "gap_kg": needed - crustal_bound,
                    "co2_needed_kg": needed,
                    "co2_available_optimistic_kg": crustal_bound,
                    "ratio": ratio,
                    "ratio_optimistic": needed / crustal_bound if crustal_bound > 0 else float("inf"),
                    "pi_polar": pi_polar,
                    "pi_accessible": pi_accessible,
                    "pi_crustal_bound": pi_crustal,
                    "conclusion": (
                        f"Endogenous CO2 is insufficient by {needed / crustal_bound:.1f}x even against "
                        "the extreme crustal-sequestration bound (~1 bar, Jakosky 2019)."
                        if crustal_bound > 0
                        else "No endogenous CO2 reservoir is known for this body."
                    ),
                },
                projection_basis=(
                    f"Step 1: target {target_pa:.0f} Pa needs {needed:.2e} kg of gas "
                    f"(M = K * dP, K = 4 pi R^2 / g). "
                    f"Step 2: largest conceivable endogenous CO2 = {crustal_bound:.2e} kg "
                    "(extreme crustal bound; the accessible reference is ~20 mbar). "
                    f"Step 3: needed/available = {needed / max(crustal_bound, 1e-300):.1f} > 1, "
                    "so the gap cannot close with in-situ CO2."
                ),
                unexpanded_branches=[
                    "Exogenous import: excluded here (in-situ only); see the logistics extratool.",
                    "Crustal carbonates beyond the ~1 bar bound: no data support a larger inventory.",
                ],
                source_call="NOArCoSession.co2_feasibility",
                warnings=[
                    (
                        "The crustal bound is a sequestered total, not an accessible inventory; "
                        "mobilising it is a carbonate-processing problem (Turyshev 2026, Sec. V.B)."
                    )
                ],
            )

        gap = max(needed - accessible, 0.0)
        value: dict[str, Any] = {
            "co2_needed_kg": needed,
            "co2_available_kg": accessible,
            "co2_polar_kg": polar,
            "gap_kg": max(needed - polar, 0.0),
            "gap_pa": max(needed - polar, 0.0) / self.inventory.mass_per_pascal(),
            "isru_feasible": needed <= polar,
            "ratio": ratio,
            "pi_polar": pi_polar,
            "pi_accessible": pi_accessible,
            "pi_crustal_bound": pi_crustal,
        }
        value["feasible"] = needed <= accessible
        if needed <= polar:
            value["conclusion"] = "Polar CO2 alone covers the pressure target."
        elif needed <= accessible:
            value["conclusion"] = (
                "Needs more than the polar deposit but fits the representative accessible "
                f"reservoir ({accessible:.2e} kg, ~20 mbar)."
            )
        else:
            value["conclusion"] = (
                f"Exceeds the accessible reference by {gap:.2e} kg; only crustal "
                "carbonate mining at industrial scale could supply it."
            )
        return MATEResult(
            value=value,
            status=MATEStatus.FULL_COMPUTE,
            source_call="NOArCoSession.co2_feasibility",
        )

    # ------------------------------------------------------------------
    # Mechanism management
    # ------------------------------------------------------------------

    def add_mechanism(self, mechanism: Mechanism) -> NOArCoSession:
        """Add an intervention mechanism to the session.

        Parameters
        ----------
        mechanism : Mechanism
            Any subclass of `noarco.mechanisms.base.Mechanism`.

        Returns
        -------
        self (for chaining)
        """
        self._mechanisms.append(mechanism)
        return self

    def clear_mechanisms(self) -> NOArCoSession:
        """Remove all mechanisms from the session."""
        self._mechanisms.clear()
        return self

    def evaluate_mechanism(
        self,
        mechanism: Mechanism | None = None,
        target_delta_t_k: float | None = None,
        deployment_years: float = 10.0,
    ) -> MATEResult:
        """Evaluate a mechanism against the current planet state and target.

        If `mechanism` is None, uses the first mechanism in the session list.
        If `target_delta_t_k` is None, uses `self.delta_temperature_k`.

        Returns
        -------
        MATEResult whose .value is a MechanismResult.
        """
        import time as _time

        mech = mechanism or (self._mechanisms[0] if self._mechanisms else None)
        if mech is None:
            raise ValueError(
                "No mechanism specified. Add one via session.add_mechanism() "
                "or pass mechanism= to evaluate_mechanism()."
            )

        dt = target_delta_t_k or self.delta_temperature_k
        if dt is None:
            raise ValueError(
                "No target temperature change available. "
                "Set DesiredState.target_mean_temperature_k or pass target_delta_t_k=."
            )

        t0 = _time.perf_counter()
        result = mech.compute(self._planet, target_delta_t_k=dt, deployment_years=deployment_years)
        elapsed = _time.perf_counter() - t0

        return MATEResult(
            value=result,
            status=MATEStatus.FULL_COMPUTE,
            compute_time_s=elapsed,
            source_call=f"NOArCoSession.evaluate_mechanism[{mech.name}]",
        )

    # ------------------------------------------------------------------
    # State mutation (reactive: invalidates all dependent layers)
    # ------------------------------------------------------------------

    def update(self, **kwargs: Any) -> NOArCoSession:
        """Update one or more PlanetState fields and propagate throughout.

        This is the ONLY way to modify the planet state in a session.
        After calling update():
          - A new PlanetState is created via pydantic model_copy.
          - The cache is fully invalidated (all layers will recompute).
          - The edit is recorded in the session history.

        Parameters
        ----------
        **kwargs
            Any field of PlanetState by name. E.g.:
                session.update(surface_pressure_pa=1220.0)
                session.update(mean_temperature_k=230.0, surface_albedo=0.20)

        Returns
        -------
        self (for chaining)

        Examples
        --------
        >>> session.update(surface_pressure_pa=1220.0, mean_temperature_k=225.0)
        >>> print(session.inventory.mass_per_mbar())  # auto-updated
        """
        old_state = self._planet
        self._planet = _validated_update(self._planet, kwargs)

        # Invalidate ALL dependent layers
        self._cache.invalidate()
        self._inventory = None
        self._radiative = None

        # Record in history for audit trail
        self._history.append({
            "fields_changed": list(kwargs.keys()),
            "old_values": {k: getattr(old_state, k) for k in kwargs},
            "new_values": kwargs,
        })

        if self.verbose:
            changed = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"[Session] State updated: {changed} → cache invalidated.")

        return self

    def update_target(self, **kwargs: Any) -> NOArCoSession:
        """Update the DesiredState and invalidate derived delta properties."""
        self._target = _validated_update(self._target, kwargs)
        self._cache.invalidate()
        if self.verbose:
            print(f"[Session] Target updated: {list(kwargs.keys())}")
        return self

    def update_constraints(self, **kwargs: Any) -> NOArCoSession:
        """Update the ConstraintSet."""
        self._constraints = _validated_update(self._constraints, kwargs)
        if self.verbose:
            print(f"[Session] Constraints updated: {list(kwargs.keys())}")
        return self

    def reset_to(self, planet: PlanetState) -> NOArCoSession:
        """Reset the session to a completely new PlanetState."""
        self._planet = planet
        self._cache.invalidate()
        self._inventory = None
        self._radiative = None
        self._history.append({"reset_to": planet.body_name})
        return self

    # ------------------------------------------------------------------
    # Diagnostics and audit
    # ------------------------------------------------------------------

    def mate_stats(self) -> dict[str, Any]:
        """Return MATE/TRIADA compute-efficiency statistics."""
        stats = self._cache.stats()
        stats["history_entries"] = len(self._history)
        return stats

    def history(self) -> list[dict[str, Any]]:
        """Return the full edit history of this session."""
        return list(self._history)

    def summary(self) -> str:
        """Full session status report."""
        lines = [
            "=" * 60,
            f" NOArCO Session — {self._planet.body_name}",
            "=" * 60,
            "",
            self._planet.summary(),
            "",
            f"Target       : {self._target.description or self._target.habitability_tier.value}",
            f"Min P target : {self._target.min_surface_pressure_pa} Pa",
            f"Target T     : {self._target.target_mean_temperature_k} K",
            f"Is regional  : {self._target.is_regional}",
            "",
        ]

        # Delta summary
        dt = self.delta_temperature_k
        dp = self.delta_pressure_pa
        df = self.forcing_required_wm2
        dm = self.mass_required_kg

        lines.append("--- Deltas (current → target) ---")
        lines.append(f"  ΔT required    : {dt:+.1f} K" if dt is not None else "  ΔT : N/A")
        lines.append(f"  ΔP required    : {dp:+.0f} Pa" if dp is not None else "  ΔP : N/A")
        lines.append(f"  ΔF required    : {df:+.2f} W/m²" if df is not None else "  ΔF : N/A")
        lines.append(f"  ΔM required    : {dm:.3e} kg" if dm is not None else "  ΔM : N/A")

        # CO2 feasibility (MATE-annotated)
        lines.append("")
        lines.append("--- CO₂ Feasibility (MATE) ---")
        feas = self.co2_feasibility
        lines.append(f"  Status         : {feas.status.value}")
        if isinstance(feas.value, dict):
            lines.append(f"  Conclusion     : {feas.value.get('conclusion', '')}")
        if feas.projection_basis:
            lines.append(f"  Projection     : {feas.projection_basis[:120]}…")

        # Mechanisms
        lines.append("")
        if self._mechanisms:
            lines.append(f"--- Active Mechanisms ({len(self._mechanisms)}) ---")
            for m in self._mechanisms:
                lines.append(f"  • {m.name}")
        else:
            lines.append("--- No mechanisms configured (use session.add_mechanism()) ---")

        # MATE stats
        lines.append("")
        lines.append("--- MATE / TRIADA Cache Stats ---")
        stats = self.mate_stats()
        lines.append(f"  Cache size     : {stats['size']} / {stats['maxsize']}")
        lines.append(
            f"  Hit rate       : {stats['hit_rate']:.0%} "
            f"({stats['hits']} hits, {stats['misses']} misses)"
        )
        lines.append(f"  Edit history   : {stats['history_entries']} changes")

        lines.append("")
        lines.append(self._constraints.summary())
        lines.append("=" * 60)

        return "\n".join(lines)
