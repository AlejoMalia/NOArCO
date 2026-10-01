"""
noarco.assurance.sfsa_bridge — Optional link to the SFSA framework
==================================================================

SFSA (Standard Framework for Scientific Advancement) provides the campaign
management engines (MATE/ORE/VOI/AMF/...). NOArCO only *requires* the hardened
V&V layer in this package; SFSA is an optional accelerator::

    pip install -e "/path/to/SFSA/python"      # or:  pip install noarco[sfsa]

``open_sfsa_session(planet)`` returns an ``SFSASession`` whose layers are
pre-registered with the planet's state and whose model-risk engine (MRE) is
configured with NOArCO's validity envelopes, so campaign sweeps are gated by
the same regimes that ``assure()`` enforces.
"""

from __future__ import annotations

from typing import Any

from noarco.core.planet_state import PlanetState


class SFSAUnavailableError(ImportError):
    """Raised when the optional ``sfsa`` package is not installed."""


def open_sfsa_session(planet: PlanetState, name: str | None = None) -> Any:
    try:
        from sfsa import SFSASession
    except ImportError as exc:  # pragma: no cover - exercised only without sfsa
        raise SFSAUnavailableError(
            "SFSA is not installed. Install it with `pip install -e <SFSA>/python`."
        ) from exc

    session = SFSASession(name=name or f"NOArCO::{planet.body_name}")
    session.register_layer("planet", {
        "temp_k": planet.mean_temperature_k,
        "pressure_pa": planet.surface_pressure_pa,
        "albedo": planet.surface_albedo,
        "solar_constant_wm2": planet.solar_constant_wm2,
        "ir_optical_depth": planet.ir_optical_depth,
    })
    mre = getattr(session, "mre", None)
    if mre is not None:
        mre.register_envelope("ir_optical_depth", 0.0, 10.0, hard_min=0.0)
        mre.register_envelope("pressure_pa", 0.0, 1.0e7, hard_min=0.0)
        mre.register_envelope("albedo", 0.0, 1.0, hard_min=0.0, hard_max=1.0)
        mre.register_envelope("temp_k", 1.0, 1500.0, hard_min=0.0)
    return session
