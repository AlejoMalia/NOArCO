"""Quick smoke test for autodiscovery + session modules."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noarco.autodiscovery import GapDetector, SelfImprovingSession
from noarco.data.planets import MARS
from noarco.session import NOArCoSession

# Create a deliberately incomplete Mars state (simulating an unknown body)
mars_incomplete = MARS.model_copy(update={
    'ir_optical_depth': 0.0,
    'uv_flux_wm2': None,
    'co2_ice_kg': None,
    'h2o_ice_kg': None,
    'regolith_composition': None,
})

print("=== BEFORE self-improvement ===")
detector = GapDetector()
gaps_before = detector.scan(mars_incomplete)
print(f"Gaps detected: {len(gaps_before)}")
for g in gaps_before:
    print(f"  {g}")

print()

# Wrap in a self-improving session
base = NOArCoSession(mars_incomplete, verbose=False)
smart = SelfImprovingSession(base, auto_fill=True)

print(smart.diagnosis_report())

print("=== AFTER self-improvement ===")
p = smart.session.planet
print(f"  ir_optical_depth     : {p.ir_optical_depth:.4f}  (was 0.0)")
print(f"  uv_flux_wm2          : {p.uv_flux_wm2:.3e}  (was None)")
print(f"  co2_ice_kg           : {p.co2_ice_kg:.2e}  (was None)")
print(f"  h2o_ice_kg           : {p.h2o_ice_kg:.2e}  (was None)")
print(f"  regolith SiO2 frac   : {p.regolith_composition.get('SiO2', 'N/A')}  (was None)")

# Test reactive update
print()
print("=== Reactive update: increase pressure 10x ===")
smart.session.update(surface_pressure_pa=6100.0)
print(f"  New P: {smart.session.planet.surface_pressure_pa:.0f} Pa")
print(f"  New inv mass: {smart.session.inventory.current_atmospheric_mass_kg():.4e} kg")
print(f"  New delta_T needed: {smart.session.delta_temperature_k:.1f} K")
print(f"  MATE cache stats: {smart.session.mate_stats()}")
