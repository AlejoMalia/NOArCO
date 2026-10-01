"""
NOArCO — Planetary Terraforming & Planetary-Engineering Analysis Framework
==========================================================================

A scientific Python library for simulating, planning, and optimizing
terraforming and paraterraforming scenarios on planets, moons, and
hypothetical exoplanets.

References
----------
Turyshev, S. G. (2026). Terraforming Mars: Mass, Forcing, and Industrial
    Throughput Constraints. arXiv:2603.00402.
DeBenedictis et al. (2025). The case for Mars terraforming research.
    Nature Astronomy, 9, 634-639.
McKay et al. (1991). Making Mars habitable. Nature, 352, 489-496.
Wordsworth et al. (2019). Nature Astronomy, 3, 898-903.
Ansari et al. (2024). Science Advances, 10, eadn4650.
"""

from noarco import assurance, extratools
from noarco.analyzers.atmosphere import AtmosphereAnalysis, AtmosphereEngine, VolumetricRequirement
from noarco.analyzers.soil import OxygenExtractionPlan, SoilAnalysis, SoilEngine
from noarco.autodiscovery import GapDetector, GapFiller, SelfImprovingSession
from noarco.core.constraints import ConstraintSet
from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.core.endpoints import MARS_ENDPOINTS, Endpoint
from noarco.core.planet_state import GasComposition, PlanetState, SourceType
from noarco.core.state import StateEngine, UnifiedPlanetState
from noarco.engines.impact import ImpactEngine, ImpactReport
from noarco.engines.knowledge import (
    Domain,
    KnowledgeEngine,
    KnowledgeMechanism,
    Mechanism,
    MechanismEvaluation,
    MechanismKnowledgeEngine,
    MechanismMatcher,
    ProcessCatalogEngine,
    ProcessType,
    ReactionPathEngine,
    ReactionStoichiometry,
)
from noarco.engines.pathfinder import OptimizedPathway, PathEngine, PathwayStep
from noarco.engines.timeline import Milestone, TimelineEngine, TimelineSchedule
from noarco.inventory.atmosphere import AtmosphericInventory
from noarco.inventory.inventory import ComprehensiveInventoryReport, InventoryEngine
from noarco.mate.core import MATEResult, MATEStatus, NOArCoCache, mate_project, triada
from noarco.radiative.balance import RadiativeBalance
from noarco.report.reporter import ReportEngine
from noarco.session import NOArCoSession

NOArCOSession = NOArCoSession

__version__ = "0.1.0"
__author__ = "Alejo Malia & NOArCO Contributors"

__all__ = [
    "MARS_ENDPOINTS",
    "AtmosphereAnalysis",
    "AtmosphereEngine",
    "AtmosphericInventory",
    "ComprehensiveInventoryReport",
    "ConstraintSet",
    "DesiredState",
    "Domain",
    "Endpoint",
    "GapDetector",
    "GapFiller",
    "GasComposition",
    "HabitabilityTier",
    "ImpactEngine",
    "ImpactReport",
    "InventoryEngine",
    "KnowledgeEngine",
    "KnowledgeMechanism",
    "MATEResult",
    "MATEStatus",
    "Mechanism",
    "MechanismEvaluation",
    "MechanismKnowledgeEngine",
    "MechanismMatcher",
    "Milestone",
    "NOArCOSession",
    "NOArCoCache",
    "NOArCoSession",
    "OptimizedPathway",
    "OxygenExtractionPlan",
    "PathEngine",
    "PathwayStep",
    "PlanetState",
    "ProcessCatalogEngine",
    "ProcessType",
    "RadiativeBalance",
    "ReactionPathEngine",
    "ReactionStoichiometry",
    "ReportEngine",
    "SelfImprovingSession",
    "SoilAnalysis",
    "SoilEngine",
    "SourceType",
    "StateEngine",
    "TimelineEngine",
    "TimelineSchedule",
    "UnifiedPlanetState",
    "VolumetricRequirement",
    "assurance",
    "extratools",
    "mate_project",
    "triada",
]
