"""
noarco.extratools — NOArCO Derived Toolsuite
============================================================
Catalog of specialized satellite tools built on the NOArCO analytical core:

1. NOArCO-Habitat    (HabitatDimensioner): Sizing of closed habitats, domes and ECLSS life support.
2. NOArCO-ISRU       (ISRUEvaluator): Mineral prospecting and in-situ space mining.
3. NOArCO-Logistics  (VolatileLogisticsPlanner): Interplanetary supply chain and cometary redirection.
4. NOArCO-Exoplanet  (ExoplanetClassifier): Astrophysical habitability and Terraforming Feasibility Index (TFI).
5. NOArCO-Agro       (AstroAgroPlanner): Bioremediation, perchlorate detoxification and space agriculture.
6. NOArCO-CLI        (cli): Fast execution console and interactive reports.

Framework: NOArCO
Framework and TRIADA protocol author: Alejo Malia
"""

from __future__ import annotations

from noarco.extratools.agro import (
    AgroReadinessReport,
    AstroAgroPlanner,
    SoilConditioningSpec,
)
from noarco.extratools.base_budget import BaseBudget, size_base
from noarco.extratools.cli import (
    run_exoplanet_quick_demo,
    run_habitat_quick_demo,
)
from noarco.extratools.exoplanet import (
    ExoplanetClassifier,
    ExoplanetProfile,
    ExoplanetReport,
)
from noarco.extratools.habitat import (
    HabitatDimensioner,
    HabitatReport,
    HabitatSpecification,
)
from noarco.extratools.isru import (
    ISRUEvaluator,
    ISRUProductionReport,
    ISRURequirement,
    ISRUSiteProfile,
)
from noarco.extratools.logistics import (
    STANDARD_SOURCES,
    LogisticsCampaignReport,
    VolatileLogisticsPlanner,
    VolatileSource,
)

__all__ = [
    "STANDARD_SOURCES",
    "AgroReadinessReport",
    # Agro
    "AstroAgroPlanner",
    "BaseBudget",
    # Exoplanet
    "ExoplanetClassifier",
    "ExoplanetProfile",
    "ExoplanetReport",
    # Habitat
    "HabitatDimensioner",
    "HabitatReport",
    "HabitatSpecification",
    # ISRU
    "ISRUEvaluator",
    "ISRUProductionReport",
    "ISRURequirement",
    "ISRUSiteProfile",
    "LogisticsCampaignReport",
    "SoilConditioningSpec",
    # Logistics
    "VolatileLogisticsPlanner",
    "VolatileSource",
    "run_exoplanet_quick_demo",
    # CLI
    "run_habitat_quick_demo",
    "size_base",
]
