"""
noarco.mate — MATE + TRIADA compute-efficiency methodology
==========================================================

MATE (Methodology of Advancement through Strategic Tension) and its
operational instrument TRIADA are integrated throughout NOArCO
to avoid wasted computation, just as a chess player stops
calculating when a forced mate in 3 is already visible.

TRIADA checks (before any computation):
    T1 · INVENTORY  — Is the result already in cache or on disk?
    T2 · MATHEMATICS  — Does a closed-form bound answer it without simulation?
    T3 · MATE        — Can a previous computation be reused?

MATE certifies (during computation):
    R3 · If the last 2-3 forced steps already fix the verdict,
         stop and project the result. Document what was NOT expanded
         and why it cannot change the outcome.

Author credit: MATE + TRIADA methodology by Alejo Malia.

References
----------
Malia, A. (2026). MATE: Methodology of Advancement through Strategic Tension.
    Internal method document.
"""

from noarco.mate.core import (
    MATEResult,
    MATEStatus,
    NOArCoCache,
    mate_project,
    triada,
)

__all__ = [
    "MATEResult",
    "MATEStatus",
    "NOArCoCache",
    "mate_project",
    "triada",
]
