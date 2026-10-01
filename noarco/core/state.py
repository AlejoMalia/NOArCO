"""
noarco.core.state — StateEngine and Unified Planetary State
===========================================================
Defines StateEngine: Validates, manages, and represents the combined
AIR (Atmosphere) and SOIL (Soil/Regolith) state of any planetary body.

Maintains backward compatibility with noarco.core.planet_state.
"""

from __future__ import annotations

from dataclasses import dataclass

from noarco.analyzers.atmosphere import AtmosphereAnalysis, AtmosphereEngine
from noarco.analyzers.soil import SoilAnalysis, SoilEngine
from noarco.core.planet_state import (
    PlanetState,
)


@dataclass
class UnifiedPlanetState:
    """Coupled representation of AIR + SOIL domains for a planetary body."""
    planet: PlanetState
    atmosphere: AtmosphereAnalysis
    soil: SoilAnalysis

    def summary(self) -> str:
        lines = [
            f"=== UnifiedPlanetState: {self.planet.body_name} ===",
            "  [AIR - Atmosphere]",
            f"    Surface pressure      : {self.atmosphere.surface_pressure_bar:.4f} bar ({self.atmosphere.surface_pressure_pa:.1f} Pa)",
            f"    Mean temperature    : {self.atmosphere.mean_temperature_k:.1f} K ({self.atmosphere.mean_temperature_c:.1f} °C)",
            f"    Total atmospheric mass: {self.atmosphere.total_atmospheric_mass_kg:.3e} kg",
            f"    Jeans retention       : {'Stable' if self.atmosphere.jeans_retention_stable else 'Unstable (hydrodynamic escape)'}",
            "  [SOIL - Soil and Lithosphere]",
            f"    Solid surface         : {'Yes' if self.soil.has_solid_surface else 'No (gas giant)'}",
            f"    Extractable oxygen    : {self.soil.extractable_o2_from_regolith_pct:.1f}% by regolith mass",
            f"    H2O ice reserve     : {self.soil.h2o_ice_reserve_kg:.2e} kg",
            f"    Metal potential       : {self.soil.structural_metal_potential}",
        ]
        return "\n".join(lines)


class StateEngine:
    """Engine for validating, ingesting, and updating planetary state representations."""

    @classmethod
    def load(cls, planet: PlanetState) -> UnifiedPlanetState:
        """Ingest PlanetState and perform 100% characterization of AIR and SOIL."""
        atm = AtmosphereEngine.analyze(planet)
        soil = SoilEngine.analyze(planet)
        return UnifiedPlanetState(
            planet=planet,
            atmosphere=atm,
            soil=soil,
        )
