"""
noarco.engines.timeline — TimelineEngine
========================================
Projects timelines, operational milestones, and scaling curves for terraforming
and paraterraforming projects across varying industrial power capacities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from noarco.core.constants import YEAR_S
from noarco.engines.pathfinder import OptimizedPathway


@dataclass
class Milestone:
    year: float
    title: str
    phase: int
    energy_delivered_pct: float
    description: str


@dataclass
class TimelineSchedule:
    body_name: str
    total_duration_years: float
    power_installed_gw: float
    milestones: list[Milestone]
    critical_phases_breakdown: dict[str, float]  # Phase name -> duration in years


class TimelineEngine:
    """Engine that computes dynamic timelines based on power growth curves."""

    @classmethod
    def generate_schedule(
        cls,
        pathway: OptimizedPathway,
        power_installed_gw: float = 10.0,
        annual_growth_rate: float = 0.0,
    ) -> TimelineSchedule:
        """Project operational milestones for an industrial power curve.

        The installed power follows ``P(t) = P0 * (1 + g)**t`` (P0 =
        ``power_installed_gw``, g = ``annual_growth_rate``). Phases run
        sequentially and each is energy-limited: a phase ends when the
        cumulative energy delivered by the power curve reaches the cumulative
        energy of the pathway up to that phase, which has the closed form

            t_end = ln(1 + E_cum * ln(1 + g) / P0) / ln(1 + g)     (g != 0)
            t_end = E_cum / P0                                     (g == 0)

        Growth defaults to 0 because exponential growth is unbounded and can
        shrink millennia into centuries; enable it only with a justified
        ceiling in mind.
        """
        if not (power_installed_gw > 0 and math.isfinite(power_installed_gw)):
            raise ValueError(f"power_installed_gw must be finite and > 0, got {power_installed_gw!r}")
        if not annual_growth_rate > -1.0:
            raise ValueError(f"annual_growth_rate must be > -1, got {annual_growth_rate!r}")

        p0_w = power_installed_gw * 1.0e9
        seconds_per_year = YEAR_S
        p0_j_per_year = p0_w * seconds_per_year
        log_g = math.log1p(annual_growth_rate)

        def year_when_delivered(energy_j: float) -> float:
            if energy_j <= 0:
                return 0.0
            if abs(log_g) < 1e-12:
                return energy_j / p0_j_per_year
            arg = 1.0 + energy_j * log_g / p0_j_per_year
            if arg <= 0.0:  # shrinking power: the energy is never delivered
                return math.inf
            return math.log(arg) / log_g

        milestones: list[Milestone] = []
        phase_breakdown: dict[str, float] = {}
        total_energy = pathway.total_energy_joules
        cum_energy = 0.0
        previous_year = 0.0

        for step in pathway.steps:
            cum_energy += step.energy_joules
            current_year = year_when_delivered(cum_energy)
            pct = (cum_energy / total_energy) * 100.0 if total_energy > 0 else 100.0

            phase_breakdown[step.mechanism_name] = current_year - previous_year
            milestones.append(
                Milestone(
                    year=round(current_year, 1),
                    title=f"Phase {step.phase}: {step.mechanism_name}",
                    phase=step.phase,
                    energy_delivered_pct=round(pct, 1),
                    description=step.justification,
                )
            )
            previous_year = current_year

        return TimelineSchedule(
            body_name=pathway.body_name,
            total_duration_years=round(previous_year, 1),
            power_installed_gw=power_installed_gw,
            milestones=milestones,
            critical_phases_breakdown=phase_breakdown,
        )
