"""
noarco.bodies — describe any world by a list of conditions
===========================================================

NOArCO is not limited to catalogued planets and moons. ``resolve_body`` accepts

* a :class:`PlanetState`,
* the name of a solar-system body (``"Mars"``, ``"Moon"``, ...),
* an :class:`ExoplanetProfile`,
* or a **dict of conditions** describing any body, derived or artificial world: an asteroid, a
  moon you define, a rotating habitat, a lab chamber, a lava tube, a hypothetical exoplanet.

Conditions (SI units; only some are needed)::

    name, pressure_pa, temperature_k, gravity_ms2, mass_kg, radius_m, density_kg_m3, albedo,
    solar_constant_wm2 | (luminosity_w | luminosity_solar, distance_au), composition,
    ir_optical_depth, h2o_ice_kg, co2_ice_kg, regolith_composition, source_type

Missing quantities are **derived from physics** where a closed form exists (``g = G M / R^2``,
``M = g R^2 / G``, ``R = 3 g / (4 pi G rho)``, ``S = L / (4 pi d^2)``, ``T = T_eq`` for an airless
body) and otherwise filled with a *recorded assumption*. Every derivation and assumption is
returned in the :class:`ResolvedBody`, so nothing is invented silently. A body that cannot be
closed from the given conditions raises ``ValueError`` listing exactly what is missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from noarco.core.constants import AU_M, G_NEWTON, SIGMA_SB
from noarco.core.planet_state import GasComposition, PlanetState, SourceType
from noarco.data.planets import ALL_SOLAR_BODIES
from noarco.extratools.exoplanet import M_EARTH, R_EARTH, ExoplanetProfile

_ALLOWED = {
    "name", "pressure_pa", "temperature_k", "gravity_ms2", "mass_kg", "radius_m", "density_kg_m3",
    "albedo", "solar_constant_wm2", "luminosity_w", "luminosity_solar", "distance_au", "composition",
    "ir_optical_depth", "h2o_ice_kg", "co2_ice_kg", "regolith_composition", "source_type",
}
_L_SUN_W = 3.828e26
#: Density assumed only when a body is given by gravity alone (rocky-body order of magnitude).
ASSUMED_ROCKY_DENSITY_KG_M3 = 3500.0
_MIN_PRESSURE_PA = 1e-9
_MIN_GRAVITY = 1e-6


@dataclass
class ResolvedBody:
    planet: PlanetState
    derived: dict[str, str] = field(default_factory=dict)       # quantity -> how it was derived
    assumptions: list[str] = field(default_factory=list)        # values filled without a closed form

    @property
    def is_fully_specified(self) -> bool:
        return not self.assumptions


def resolve_body(spec: PlanetState | str | dict[str, Any] | ExoplanetProfile | Any) -> ResolvedBody:
    """Turn any supported description into a validated :class:`PlanetState` (see module docstring)."""
    if isinstance(spec, PlanetState):
        return ResolvedBody(spec)
    if isinstance(spec, str):
        key = {k.lower(): v for k, v in ALL_SOLAR_BODIES.items()}.get(spec.strip().lower())
        if key is None:
            raise ValueError(f"Unknown body '{spec}'. Known: {sorted(ALL_SOLAR_BODIES)}; or pass a dict of conditions.")
        return ResolvedBody(key)
    if isinstance(spec, ExoplanetProfile):
        return _from_conditions({
            "name": spec.name, "mass_kg": spec.mass_earth * M_EARTH, "radius_m": spec.radius_earth * R_EARTH,
            "luminosity_solar": spec.stellar_luminosity_solar, "distance_au": spec.semi_major_axis_au,
            "albedo": spec.albedo,
        })
    if isinstance(spec, dict):
        return _from_conditions(spec)
    if hasattr(spec, "to_conditions") and hasattr(spec, "provenance_report"):   # noarco.data_io.BodyDataset
        r = _from_conditions(spec.to_conditions())
        r.assumptions.extend(f"input '{k}' is an assumption (data_source = assumed)"
                             for k in spec.provenance_report()["conjecture"])
        r.derived["input_hash"] = spec.input_hash
        return r
    raise TypeError(f"Unsupported body description: {type(spec).__name__}")


def _from_conditions(c: dict[str, Any]) -> ResolvedBody:
    unknown = sorted(set(c) - _ALLOWED)
    if unknown:
        raise ValueError(f"Unknown condition(s) {unknown}. Allowed: {sorted(_ALLOWED)}")
    for k in ("pressure_pa", "temperature_k", "gravity_ms2", "mass_kg", "radius_m", "density_kg_m3",
              "solar_constant_wm2", "luminosity_w", "luminosity_solar", "distance_au"):
        if k in c and not (isinstance(c[k], (int, float)) and math.isfinite(c[k]) and c[k] > 0 or
                           (k == "pressure_pa" and c[k] == 0)):
            raise ValueError(f"{k} must be a positive finite number, got {c[k]!r}")
    derived: dict[str, str] = {}
    assume: list[str] = []
    g, m, r, rho = (c.get(k) for k in ("gravity_ms2", "mass_kg", "radius_m", "density_kg_m3"))

    # --- size and gravity: close the (g, M, R, rho) quartet ---------------------------------
    if m and r:
        if g:
            calc = G_NEWTON * m / r**2
            if abs(calc - g) / g > 0.10:
                raise ValueError(f"Inconsistent gravity: {g} vs G*M/R^2 = {calc:.4g}")
        else:
            g = G_NEWTON * m / r**2
            derived["gravity_ms2"] = "g = G M / R^2"
    elif g and r:
        m = g * r**2 / G_NEWTON
        derived["mass_kg"] = "M = g R^2 / G"
    elif m and rho:
        r = (3.0 * m / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
        derived["radius_m"] = "R = (3 M / (4 pi rho))^(1/3)"
        g = G_NEWTON * m / r**2
        derived["gravity_ms2"] = "g = G M / R^2"
    elif g and rho:
        r = 3.0 * g / (4.0 * math.pi * G_NEWTON * rho)
        derived["radius_m"] = "R = 3 g / (4 pi G rho)"
        m = g * r**2 / G_NEWTON
        derived["mass_kg"] = "M = g R^2 / G"
    elif g:
        r = 3.0 * g / (4.0 * math.pi * G_NEWTON * ASSUMED_ROCKY_DENSITY_KG_M3)
        m = g * r**2 / G_NEWTON
        assume.append(f"radius and mass inferred from g with an assumed density of {ASSUMED_ROCKY_DENSITY_KG_M3:.0f} kg/m3")
    elif m or r or rho:
        raise ValueError("Cannot close the body's size: give gravity_ms2, or mass_kg with radius_m (or density_kg_m3), "
                         "or radius_m with density_kg_m3 only together with mass or gravity.")
    else:
        raise ValueError("Missing size: give gravity_ms2, or mass_kg and radius_m (or density_kg_m3).")
    assert g is not None and m is not None and r is not None
    if g < _MIN_GRAVITY:
        g = _MIN_GRAVITY
        assume.append("microgravity clamped to 1e-6 m/s2 (PlanetState needs g > 0)")

    # --- insolation and albedo -----------------------------------------------------------------
    albedo = c.get("albedo")
    if albedo is None:
        albedo = 0.30
        assume.append("Bond albedo assumed 0.30")
    if not 0.0 <= albedo <= 1.0:
        raise ValueError("albedo must be in [0, 1]")
    s = c.get("solar_constant_wm2")
    if s is None:
        lum = c.get("luminosity_w") or ((c["luminosity_solar"] * _L_SUN_W) if "luminosity_solar" in c else None)
        if lum and c.get("distance_au"):
            s = lum / (4.0 * math.pi * (c["distance_au"] * AU_M) ** 2)
            derived["solar_constant_wm2"] = "S = L / (4 pi d^2)"
        elif c.get("temperature_k") and (c.get("pressure_pa") or 0.0) < 1.0:
            s = 4.0 * SIGMA_SB * c["temperature_k"] ** 4 / (1.0 - albedo)
            assume.append("insolation back-computed from the given temperature treating it as T_eq (airless body)")
        else:
            raise ValueError("Missing insolation: give solar_constant_wm2, or luminosity and distance_au, "
                             "or temperature_k for an airless body.")

    # --- atmosphere and temperature --------------------------------------------------------------
    p = c.get("pressure_pa")
    if p is None or p == 0:
        p = _MIN_PRESSURE_PA
        assume.append("no atmosphere: pressure set to 1e-9 Pa")
    comp = c.get("composition")
    if comp is None:
        comp = {"CO2": 1.0}
        assume.append("composition unspecified; a nominal CO2 placeholder is used (irrelevant at vacuum)"
                      if p <= 1e-3 else "composition unspecified; nominal pure CO2 assumed")
    temp = c.get("temperature_k")
    if temp is None:
        temp = (s * (1.0 - albedo) / (4.0 * SIGMA_SB)) ** 0.25
        derived["temperature_k"] = "T = T_eq = [S (1 - A) / (4 sigma)]^(1/4) (no greenhouse)"

    st = c.get("source_type", SourceType.ESTIMATED)
    if isinstance(st, str):
        st = SourceType(st)
    planet = PlanetState(
        body_name=c.get("name", "custom-body"), surface_pressure_pa=p, mean_temperature_k=temp,
        gas_composition=GasComposition(species=comp), surface_albedo=albedo, gravity_ms2=g, radius_m=r,
        mass_kg=m, solar_constant_wm2=s, ir_optical_depth=c.get("ir_optical_depth", 0.0),
        co2_ice_kg=c.get("co2_ice_kg"), h2o_ice_kg=c.get("h2o_ice_kg"),
        regolith_composition=c.get("regolith_composition"), source_type=st,
        notes="Built from user conditions by noarco.bodies.resolve_body.",
    )
    return ResolvedBody(planet, derived, assume)


def planet_from(spec: PlanetState | str | dict[str, Any] | ExoplanetProfile) -> PlanetState:
    """Shortcut: the validated ``PlanetState`` for any supported description."""
    return resolve_body(spec).planet
