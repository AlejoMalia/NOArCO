"""
NOArCO Example: Mars warming via engineered nanoparticle aerosols
=================================================================

This script demonstrates the core NOArCO workflow:
1. Load the Mars E0 baseline state
2. Define a desired state (target temperature increase)
3. Compute nanoparticle aerosol requirements
4. Print inventory gap analysis
5. Print radiative balance

Expected output (approximate):
    Mass per mbar (Mars): 3.89e+15 kg/mbar   <- validates Turyshev (2026)
    Total aerosol mass: ~2e+10 kg
    Temperature increase: ~15 K

Usage
-----
    python examples/mars_warming_aerosol.py

References
----------
Ansari et al. (2024). Science Advances, 10, eadn4650.
Turyshev (2026). arXiv:2603.00402.
"""

import os
import sys

# Allow running from project root without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noarco.core.constraints import ConstraintSet
from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.data.planets import MARS
from noarco.inventory.atmosphere import AtmosphericInventory
from noarco.mechanisms.aerosols import AerosolMaterial, NanoparticleAerosol
from noarco.radiative.balance import RadiativeBalance


def main() -> None:
    print("=" * 65)
    print(" NOArCO v0.1 -- Mars Aerosol Warming Example")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Current state
    # ------------------------------------------------------------------
    print("\n" + MARS.summary())

    # ------------------------------------------------------------------
    # 2. Desired state: E1 (triple point of water)
    # ------------------------------------------------------------------
    target = DesiredState.from_tier(HabitabilityTier.E1_TRIPLE_POINT)
    print(f"\nTarget: {target.description}")
    print(f"  Min pressure : {target.min_surface_pressure_pa} Pa")
    print(f"  Target T     : {target.target_mean_temperature_k} K")

    # ------------------------------------------------------------------
    # 3. Constraints: ISRU-only, 100-year window
    # ------------------------------------------------------------------
    constraints = ConstraintSet(
        max_time_years=100.0,
        isru_only=True,
        regional_only=False,
    )
    print("\n" + constraints.summary())

    # ------------------------------------------------------------------
    # 4. Atmospheric inventory (validates Turyshev 2026)
    # ------------------------------------------------------------------
    inv = AtmosphericInventory(MARS)
    print("\n" + inv.summary(target_pa=611.0))
    print(f"\n  [VALIDATION] kg per mbar = {inv.mass_per_mbar():.3e} kg/mbar")
    print(  "  [EXPECTED]   kg per mbar = 3.89e+15 kg/mbar  (Turyshev 2026)")

    # CO2 gap for E2 (6 kPa)
    gap_e2 = inv.co2_inventory_gap_kg(
        target_pa=6000.0,
        include_regolith_co2_kg=2e17,  # Optimistic regolith estimate
    )
    print("\n--- CO2 inventory gap for E2 (6 kPa target) ---")
    print(f"  CO2 needed    : {gap_e2['co2_needed_kg']:.3e} kg")
    print(f"  CO2 available : {gap_e2['co2_available_kg']:.3e} kg (ice + optimistic regolith)")
    print(f"  Gap           : {gap_e2['gap_kg']:.3e} kg")
    print(f"  ISRU feasible : {gap_e2['isru_feasible']}")
    print("  -> Conclusion: endogenous CO2 INSUFFICIENT for E2 (Jakosky 2019).")

    # ------------------------------------------------------------------
    # 5. Radiative balance
    # ------------------------------------------------------------------
    rb = RadiativeBalance(MARS)
    print("\n" + rb.summary(target_temperature_k=273.0))

    # ------------------------------------------------------------------
    # 6. Nanoparticle aerosol mechanism (alumina, Ansari et al. 2024)
    # ------------------------------------------------------------------
    print("\n--- Mechanism: Alumina Nanoparticle Aerosol ---")
    aerosol = NanoparticleAerosol(
        material=AerosolMaterial.ALUMINA,
        coverage_fraction=1.0,  # Global
    )
    result = aerosol.compute(
        planet=MARS,
        target_delta_t_k=15.0,  # First step: +15 K
        deployment_years=5.0,
    )
    print(result.summary())

    # Regional variant: 10,000 km^2 region only
    print("\n--- Regional variant (10,000 km2 paraterraforming) ---")
    aerosol_regional = NanoparticleAerosol(
        material=AerosolMaterial.ALUMINA,
        coverage_fraction=1e10 / MARS.surface_area_m2,  # 10,000 km^2
    )
    result_regional = aerosol_regional.compute(
        planet=MARS,
        target_delta_t_k=30.0,
        deployment_years=2.0,
    )
    print(result_regional.summary())

    print("\n" + "=" * 65)
    print(" NOArCO example complete. See noarco/ for full module docs.")
    print("=" * 65)


if __name__ == "__main__":
    main()
