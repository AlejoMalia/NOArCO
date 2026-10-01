"""
noarco.core.desired_state
=========================
Defines the DesiredState model: the target conditions the user wants
to achieve through terraforming or paraterraforming.

References
----------
Turyshev, S. G. (2026). arXiv:2603.00402, Sec. II.A (end states E0-E4).
DeBenedictis et al. (2025). Nature Astronomy, 9, 634-639.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from noarco.core.constants import ARMSTRONG_LIMIT_PA, WATER_TRIPLE_POINT_K, WATER_TRIPLE_POINT_PA


class HabitabilityTier(str, Enum):
    """Target habitability level (corresponds to Turyshev E0-E4 endpoints)."""
    E0_CURRENT       = "E0_current"       # Present state (no modification)
    E1_TRIPLE_POINT  = "E1_triple_point"  # Recurrent open-surface liquid water
    E2_PROTECTED     = "E2_protected"     # Protected agriculture in enclosures (regional)
    E3_ARMSTRONG     = "E3_armstrong"     # Pressure-unassisted exposure (>= 6.27 kPa)
    E4_BREATHABLE    = "E4_breathable"    # Breathable surface atmosphere
    CUSTOM           = "custom"           # User-defined targets


# Backwards-compatible alias of the old (mislabelled) E2 name; ``_member_map_`` makes the
# alias resolve to the same member without a duplicate-value definition.
HabitabilityTier._member_map_["E2_LIGHT_SUIT"] = HabitabilityTier.E2_PROTECTED


class DesiredState(BaseModel):
    """Target planetary conditions specified by the user.

    At minimum, the user specifies a `habitability_tier` or any
    combination of explicit physical targets. NOArCO will compute the
    delta between `PlanetState` (current) and `DesiredState` (target)
    to drive planning.

    Parameters
    ----------
    habitability_tier : HabitabilityTier
        High-level habitability goal. Used to auto-fill targets
        if explicit values are not provided.
    min_surface_pressure_pa : float, optional
        Minimum acceptable surface pressure (Pa).
    target_mean_temperature_k : float, optional
        Target global mean surface temperature (K).
    min_temperature_k : float, optional
        Minimum allowable temperature anywhere in the target region (K).
    min_o2_partial_pressure_pa : float, optional
        Minimum O2 partial pressure (Pa). Breathable air: 16,000 Pa.
    max_co2_partial_pressure_pa : float, optional
        Maximum safe CO2 partial pressure (Pa). TLV-TWA: 500 Pa (5000 ppm).
    region_area_m2 : float, optional
        If set, this is a regional (paraterraforming) target, not global.
        Units: m2. None implies global scope.
    description : str, optional
        Human-readable description of the desired state.

    Examples
    --------
    >>> target = DesiredState(
    ...     habitability_tier=HabitabilityTier.E1_TRIPLE_POINT,
    ...     min_surface_pressure_pa=611.0,
    ...     target_mean_temperature_k=273.0,
    ... )
    """

    habitability_tier: HabitabilityTier = Field(
        default=HabitabilityTier.E1_TRIPLE_POINT,
        description="Target habitability endpoint (Turyshev E0-E4).",
    )
    min_surface_pressure_pa: float | None = Field(
        default=None, description="Minimum target surface pressure (Pa)."
    )
    target_mean_temperature_k: float | None = Field(
        default=None, description="Target global mean temperature (K)."
    )
    min_temperature_k: float | None = Field(
        default=None,
        description="Minimum allowable temperature in target region (K).",
    )
    min_o2_partial_pressure_pa: float | None = Field(
        default=None, description="Minimum O2 partial pressure for breathing (Pa)."
    )
    max_co2_partial_pressure_pa: float | None = Field(
        default=None, description="Maximum safe CO2 partial pressure (Pa)."
    )
    region_area_m2: float | None = Field(
        default=None,
        description="Target region area (m2). None = global. Set for paraterraforming.",
    )
    target_volume_m3: float | None = Field(
        default=None,
        description="Target enclosed or regional volume to terraform (m3). If set, activates volumetric paraterraforming.",
    )
    enclosed: bool = Field(
        default=False,
        description="True if the pressure target applies inside enclosures only (regional problem).",
    )
    description: str | None = Field(
        default=None, description="Human-readable description of target."
    )

    @classmethod
    def from_tier(cls, tier: HabitabilityTier) -> DesiredState:
        """Create a DesiredState from a standard habitability tier.

        Threshold values are from Turyshev (2026) Table 1 and
        standard physiological/thermodynamic references.

        Parameters
        ----------
        tier : HabitabilityTier
            The target endpoint.

        Returns
        -------
        DesiredState
        """
        configs: dict[HabitabilityTier, dict[str, Any]] = {
            HabitabilityTier.E0_CURRENT: {
                "description": "E0: present state -- no modification.",
            },
            HabitabilityTier.E1_TRIPLE_POINT: {
                "min_surface_pressure_pa": WATER_TRIPLE_POINT_PA,
                "target_mean_temperature_k": WATER_TRIPLE_POINT_K,
                "description": "E1: recurrent open-surface liquid water.",
            },
            HabitabilityTier.E2_PROTECTED: {
                "min_surface_pressure_pa": 10_000.0,
                "min_temperature_k": 273.15,
                "enclosed": True,
                "description": "E2: protected agriculture; 10 kPa interior pressure in enclosures (regional).",
            },
            HabitabilityTier.E3_ARMSTRONG: {
                "min_surface_pressure_pa": ARMSTRONG_LIMIT_PA,
                "target_mean_temperature_k": 250.0,
                "description": "E3: pressure-unassisted exposure (>= 6.27 kPa, T_s >~ 250 K).",
            },
            HabitabilityTier.E4_BREATHABLE: {
                "min_surface_pressure_pa": 101_325.0,
                "target_mean_temperature_k": 288.0,
                "min_o2_partial_pressure_pa": 16_000.0,
                "max_co2_partial_pressure_pa": 500.0,
                "description": "E4: breathable surface atmosphere, no equipment needed.",
            },
        }
        cfg = configs.get(tier, {})
        return cls(habitability_tier=tier, **cfg)

    @property
    def is_regional(self) -> bool:
        """True if this is a regional (paraterraforming) target."""
        return self.region_area_m2 is not None or self.enclosed
