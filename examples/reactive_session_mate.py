"""
NOArCO Example 2: Reactive Session + MATE/TRIADA
=================================================

Demonstrates:
  1. NOArCoSession as a reactive network: editing any planet parameter
     auto-propagates to inventory, radiative balance, and mechanism outputs.
  2. MATE + TRIADA: how the framework avoids wasted computation.
  3. CO2 feasibility MATE projection (R3): stops before full calculation
     when the verdict is already forced.

Usage
-----
    python examples/reactive_session_mate.py

Expected MATE output:
    - First session.summary() call: all FULL_COMPUTE (cold cache)
    - After session.update(): cache invalidated, recomputed on demand
    - CO2 gap for E2: MATEStatus.PROJECTED (forced verdict, no full sim)
    - Second call to same property: MATEStatus.CACHE_HIT (free)

References
----------
Turyshev (2026). arXiv:2603.00402.
Malia, A. (2026). MATE: Methodology of Advancement through Strategic Tension.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noarco.core.constraints import ConstraintSet
from noarco.core.desired_state import HabitabilityTier
from noarco.mate.core import MATEStatus
from noarco.mechanisms.aerosols import AerosolMaterial, NanoparticleAerosol
from noarco.session import NOArCoSession


def separator(title: str) -> None:
    print("\n" + "═" * 65)
    print(f"  {title}")
    print("═" * 65)


def main() -> None:
    # ──────────────────────────────────────────────────────────────────
    # 1. Create reactive session for Mars → E1
    # ──────────────────────────────────────────────────────────────────
    separator("1. Reactive Session — Mars E0 → E1")

    session = NOArCoSession.for_mars(
        target_tier=HabitabilityTier.E1_TRIPLE_POINT,
        constraints=ConstraintSet(max_time_years=500, isru_only=True),
        verbose=True,
    )

    # Add a mechanism
    session.add_mechanism(
        NanoparticleAerosol(material=AerosolMaterial.ALUMINA, coverage_fraction=1.0)
    )

    # Full session summary (first call — cold cache)
    print(session.summary())

    # ──────────────────────────────────────────────────────────────────
    # 2. MATE/TRIADA: CO2 feasibility projection
    # ──────────────────────────────────────────────────────────────────
    separator("2. CO₂ Feasibility — MATE Projection (E1 target)")

    feas_e1 = session.co2_feasibility
    print(f"  Status   : {feas_e1.status.value}")
    print(f"  Verdict  : {feas_e1.value.get('conclusion', '')}")
    if feas_e1.projection_basis:
        print(f"  Basis    : {feas_e1.projection_basis}")
    print()

    # Now check E2 (6 kPa target — much larger gap)
    separator("3. CO₂ Feasibility — MATE Projection (E2 target, 6 kPa)")

    session_e2 = NOArCoSession.for_mars(
        target_tier=HabitabilityTier.E2_LIGHT_SUIT,
        constraints=ConstraintSet(isru_only=True),
        verbose=False,
    )
    feas_e2 = session_e2.co2_feasibility
    print(f"  Status           : {feas_e2.status.value}")
    print(f"  CO₂ needed       : {feas_e2.value.get('co2_needed_kg', 0):.2e} kg")
    print(f"  CO₂ available    : {feas_e2.value.get('co2_available_optimistic_kg', 0):.2e} kg")
    print(f"  Gap ratio        : {feas_e2.value.get('ratio_optimistic', 0):.1f}×")
    print(f"  Conclusion       : {feas_e2.value.get('conclusion', '')}")
    if feas_e2.status == MATEStatus.PROJECTED:
        print("\n  [MATE R3] Computation stopped. Verdict forced. Unexpanded branches:")
        for b in feas_e2.unexpanded_branches:
            print(f"    • {b}")
    print()

    # ──────────────────────────────────────────────────────────────────
    # 4. Reactive update: change surface pressure, watch everything update
    # ──────────────────────────────────────────────────────────────────
    separator("4. Reactive Update — doubling surface pressure")

    print(f"  Before: P = {session.planet.surface_pressure_pa:.1f} Pa")
    print(f"  Before: Atm mass = {session.inventory.current_atmospheric_mass_kg():.4e} kg")
    print(f"  Before: ΔT needed = {session.delta_temperature_k:+.1f} K")
    print(f"  Before: ΔF needed = {session.forcing_required_wm2:+.2f} W/m²")
    print()

    # Edit → auto-propagates through ALL layers
    session.update(surface_pressure_pa=1220.0, mean_temperature_k=220.0)

    print(f"  After:  P = {session.planet.surface_pressure_pa:.1f} Pa")
    print(f"  After:  Atm mass = {session.inventory.current_atmospheric_mass_kg():.4e} kg")
    print(f"  After:  ΔT needed = {session.delta_temperature_k:+.1f} K")
    print(f"  After:  ΔF needed = {session.forcing_required_wm2:+.2f} W/m²")

    # ──────────────────────────────────────────────────────────────────
    # 5. Mechanism evaluation (uses updated state automatically)
    # ──────────────────────────────────────────────────────────────────
    separator("5. Mechanism Evaluation on Updated State")

    result = session.evaluate_mechanism(deployment_years=5.0)
    print(result.value.summary())
    print(f"\n  [MATE] Compute time: {result.compute_time_s*1000:.1f} ms")

    # ──────────────────────────────────────────────────────────────────
    # 6. MATE cache stats
    # ──────────────────────────────────────────────────────────────────
    separator("6. MATE / TRIADA Cache Statistics")

    stats = session.mate_stats()
    print(f"  Cache entries  : {stats['size']}")
    print(f"  Cache hit rate : {stats['hit_rate']:.0%}")
    print(f"  Hits           : {stats['hits']}")
    print(f"  Misses         : {stats['misses']}")
    print(f"  Edit history   : {stats['history_entries']} state changes")
    print()
    print("  [MATE summary: only missed computations that were truly needed]")
    print("  [All cache hits = 0 ms, 0 CPU, 0 €]")

    separator("NOArCO session example complete")


if __name__ == "__main__":
    main()
