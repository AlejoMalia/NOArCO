"""
noarco.assurance — Verification & Validation layer (flight-assurance grade)
===========================================================================

Hardened adaptation of the SFSA engines most relevant to agency use:

=============  ==========================  =====================================
SFSA engine    Module here                 What it guarantees
=============  ==========================  =====================================
UDE            ``units``                   strict dimensional analysis
UQE            ``uncertainty``             covariance propagation + MC check
AIE / MRE      ``validity``                model applicability envelopes
TRIADA T3      ``invariants``              physical invariants
RME            ``manifest``                reproducible, tamper-evident runs
(orchestrator) ``audit.assure``            one-call V&V report with PASS/FAIL gate
=============  ==========================  =====================================
"""

from noarco.assurance.audit import DEFAULT_REL_SIGMA, AssuranceReport, assure
from noarco.assurance.findings import Finding, Severity
from noarco.assurance.manifest import Manifest, build_manifest
from noarco.assurance.uncertainty import UncertainValue, propagate
from noarco.assurance.units import UnitError, convert, parse_unit

__all__ = [
    "DEFAULT_REL_SIGMA", "AssuranceReport", "Finding", "Manifest", "Severity",
    "UncertainValue", "UnitError", "assure", "build_manifest", "convert",
    "parse_unit", "propagate",
]
