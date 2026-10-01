"""
noarco.assurance.validity — Model validity envelopes (assumption integrity)
===========================================================================

Every closed-form model in NOArCO carries an applicability regime. This module
makes those regimes explicit and checkable (SFSA ``AIE``/``MRE`` concept):
a result computed outside its envelope is reported as a finding instead of
being returned silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from noarco.assurance.findings import Finding, Severity
from noarco.core.constants import R_GAS as _R_GAS
from noarco.core.planet_state import PlanetState
from noarco.mechanisms.base import MechanismResult


@dataclass(frozen=True)
class Envelope:
    """Inclusive safe range [lo, hi]; ``margin`` widens it to a CAUTION band."""

    code: str
    what: str
    lo: float
    hi: float
    margin: float = 0.25  # fraction of span tolerated as CAUTION before FAIL

    def check(self, value: float) -> Finding | None:
        if not math.isfinite(value):
            return Finding(self.code, Severity.FAIL, f"{self.what} is not finite", value)
        if self.lo <= value <= self.hi:
            return None
        span = self.hi - self.lo
        over = (self.lo - value) if value < self.lo else (value - self.hi)
        sev = Severity.CAUTION if over <= self.margin * span else Severity.FAIL
        return Finding(self.code, sev,
                       f"{self.what} = {value:.4g} outside validity range [{self.lo:g}, {self.hi:g}]",
                       value, f"[{self.lo:g}, {self.hi:g}]")


# Regime limits for the analytic models used by the core engines.
GRAY_GREENHOUSE_TAU = Envelope("VAL-GRAY-TAU", "IR optical depth (1-layer gray model)", 0.0, 10.0)
IDEAL_GAS_PRESSURE = Envelope("VAL-IDEAL-GAS-P", "Surface pressure (ideal-gas regime)", 0.0, 1.0e7)
CO2_LOG_FORCING_RATIO = Envelope("VAL-CO2-LOG-RATIO",
                                 "CO2 pressure ratio (logarithmic forcing fit)", 0.1, 1.0e3, margin=0.0)
HYDROSTATIC_THIN_SHELL = Envelope("VAL-HYDRO-THIN", "Atmospheric scale height / radius", 0.0, 0.05)


def scale_height_m(planet: PlanetState) -> float:
    """Isothermal scale height H = R T / (mu g) using the planet's own composition."""
    from noarco.inventory.atmosphere import AtmosphericInventory

    mu = AtmosphericInventory(planet)._mean_molar_mass_kg_per_mol()
    return _R_GAS * planet.mean_temperature_k / (mu * planet.gravity_ms2)


def check_planet_validity(planet: PlanetState) -> list[Finding]:
    """Check that the planet's state lies inside the regimes of the core models."""
    out: list[Finding] = []
    checks = [
        GRAY_GREENHOUSE_TAU.check(planet.ir_optical_depth),
        IDEAL_GAS_PRESSURE.check(planet.surface_pressure_pa),
        HYDROSTATIC_THIN_SHELL.check(scale_height_m(planet) / planet.radius_m),
    ]
    out.extend(c for c in checks if c is not None)
    return out


def check_forcing_request(planet: PlanetState, p_current_pa: float | None = None,
                          p_target_pa: float | None = None) -> list[Finding]:
    """Validity of a CO2-forcing request (logarithmic law) against the planet's models.

    Temperature-forcing conversions in NOArCO use the exact Stefan-Boltzmann balance, so no
    linearisation envelope applies.
    """
    out: list[Finding] = []
    if p_current_pa and p_target_pa and p_current_pa > 0 and p_target_pa > 0:
        f = CO2_LOG_FORCING_RATIO.check(p_target_pa / p_current_pa)
        if f:
            out.append(f)
    return out


def check_mechanism_result(result: MechanismResult) -> list[Finding]:
    """Turn a mechanism result's credibility metadata into assurance findings.

    ``first_principles`` results add nothing; every other class, every unreachable target and
    every regional/upper-bound caveat is surfaced so that it cannot travel without its
    qualifier.
    """
    out: list[Finding] = []
    name = result.mechanism_name
    if result.model_class == "engineering_estimate":
        out.append(Finding("MECH-ENGINEERING", Severity.CAUTION,
                           f"{name}: parametrised engineering estimate; validate against data"))
    elif result.model_class == "screening_upper_bound":
        out.append(Finding("MECH-UPPER-BOUND", Severity.CAUTION,
                           f"{name}: screening upper bound, not a prediction"))
    elif result.model_class == "reference_scaling":
        out.append(Finding("MECH-SCALING", Severity.INFO,
                           f"{name}: published order-of-magnitude scaling"))
    for w in result.warnings:
        if "UNREACHABLE" in w:
            out.append(Finding("MECH-UNREACHABLE", Severity.CAUTION, f"{name}: {w}"))
    if result.area_fraction < 1.0:
        out.append(Finding("MECH-REGIONAL", Severity.INFO,
                           f"{name}: acts on {result.area_fraction:.2e} of the surface only"))
    return out
