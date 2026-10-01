"""
noarco.core.constraints
=======================
Defines ConstraintSet: operational constraints on the terraforming
process imposed by engineering, logistics, and policy.

References
----------
Turyshev, S. G. (2026). arXiv:2603.00402 (Section 5, throughput analysis).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConstraintSet(BaseModel):
    """Operational constraints on the terraforming scenario.

    These constraints define the resource envelope within which
    NOArCO must find feasible paths. If a path cannot be found
    within these constraints, NOArCO will report the bottleneck.

    Parameters
    ----------
    max_time_years : float, optional
        Maximum allowed total terraforming time (years). None = unconstrained.
    max_total_mass_kg : float, optional
        Maximum total mass that can be processed/deployed (kg). None = unconstrained.
    max_power_w : float, optional
        Maximum continuous power available (W). None = unconstrained.
    max_throughput_kg_s : float, optional
        Maximum industrial mass throughput rate (kg/s). None = unconstrained.
    isru_only : bool
        If True, only in-situ resources may be used (no imports). Default False.
    regional_only : bool
        If True, only regional (paraterraforming) solutions are considered. Default False.
    allow_biological : bool
        If True, biological mechanisms (photosynthesis, microbes) are allowed. Default True.
    allow_orbital_infrastructure : bool
        If True, orbital reflectors and space-based mechanisms are allowed. Default True.
    notes : str, optional
        Free-text notes about these constraints.
    """

    max_time_years: float | None = Field(
        default=None, description="Maximum total time allowed (years)."
    )
    max_total_mass_kg: float | None = Field(
        default=None, description="Maximum total mass budget (kg)."
    )
    max_power_w: float | None = Field(
        default=None, description="Maximum available power (W)."
    )
    max_throughput_kg_s: float | None = Field(
        default=None, description="Maximum industrial throughput rate (kg/s)."
    )
    isru_only: bool = Field(
        default=False,
        description="If True, restrict to in-situ resources only.",
    )
    regional_only: bool = Field(
        default=False,
        description="If True, restrict to regional paraterraforming solutions.",
    )
    allow_biological: bool = Field(
        default=True, description="Allow biological mechanisms."
    )
    allow_orbital_infrastructure: bool = Field(
        default=True, description="Allow orbital mirrors/reflectors."
    )
    notes: str | None = Field(default=None)

    def summary(self) -> str:
        """Return human-readable constraint summary."""
        lines = ["=== ConstraintSet ==="]
        lines.append(f"  Max time       : {self.max_time_years or 'unconstrained'} yr")
        lines.append(f"  Max mass       : {self.max_total_mass_kg or 'unconstrained'} kg")
        lines.append(f"  Max power      : {self.max_power_w or 'unconstrained'} W")
        lines.append(f"  Max throughput : {self.max_throughput_kg_s or 'unconstrained'} kg/s")
        lines.append(f"  ISRU only      : {self.isru_only}")
        lines.append(f"  Regional only  : {self.regional_only}")
        lines.append(f"  Biological OK  : {self.allow_biological}")
        lines.append(f"  Orbital infra  : {self.allow_orbital_infrastructure}")
        return "\n".join(lines)
