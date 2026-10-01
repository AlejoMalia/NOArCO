"""
noarco.core.planet_state
========================
Defines the PlanetState data model: the complete physical state of a
planetary body at a given moment in time.

All physical quantities use SI units unless otherwise noted.
Each field carries a `source_type` attribute indicating whether the
value is measured, modeled, estimated, or assumed.

References
----------
Turyshev, S. G. (2026). arXiv:2603.00402 (Section 2, Table 1).
DeBenedictis et al. (2025). Nature Astronomy, 9, 634-639.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, field_validator

from noarco.core.constants import G_NEWTON, SIGMA_SB


class SourceType(str, Enum):
    """Epistemic status of a parameter value."""
    MEASURED   = "measured"    # Direct instrumental measurement
    MODELED    = "modeled"     # Output of a calibrated physical model
    ESTIMATED  = "estimated"   # Order-of-magnitude estimate with large uncertainty
    ASSUMED    = "assumed"     # Working assumption, may be poorly constrained


class GasComposition(BaseModel):
    """Molar composition of an atmospheric gas mixture.

    Parameters
    ----------
    species : dict[str, float]
        Mapping of gas species names (e.g. 'CO2', 'N2', 'O2', 'Ar', 'H2O')
        to their mole fractions. Fractions must sum to <= 1.0 (remainder
        is treated as unspecified trace gases).

    Examples
    --------
    >>> atm = GasComposition(species={'CO2': 0.953, 'N2': 0.027, 'Ar': 0.016})
    >>> atm.partial_pressure_pa(surface_pressure_pa=610.0)['CO2']
    581.33
    """
    species: dict[str, float] = Field(
        description="Mole fractions of each gas species (dimensionless)."
    )

    @field_validator("species")
    @classmethod
    def fractions_positive(cls, v: dict[str, float]) -> dict[str, float]:
        for k, frac in v.items():
            if frac < 0:
                raise ValueError(f"Mole fraction for '{k}' must be non-negative, got {frac}.")
        total = sum(v.values())
        if total > 1.005:
            raise ValueError(
                f"Sum of mole fractions ({total:.4f}) exceeds 1.0 by more than 0.5%."
            )
        # Auto-normalize: real atmospheric measurements always have small rounding
        # residuals (e.g. Mahaffy et al. 2013 SAM data sums to 1.002 due to rounding).
        # Only renormalise an *excess* (idempotent: a valid composition is returned
        # untouched, and a deliberately partial one keeps its unspecified remainder).
        if total > 1.0 + 1e-12:
            return {k: frac / total for k, frac in v.items()}
        return v

    def partial_pressure_pa(self, surface_pressure_pa: float) -> dict[str, float]:
        """Return partial pressure (Pa) for each species.

        Parameters
        ----------
        surface_pressure_pa : float
            Total surface pressure in Pascals.

        Returns
        -------
        dict[str, float]
            Species name -> partial pressure in Pa.
        """
        return {sp: frac * surface_pressure_pa for sp, frac in self.species.items()}

    def mole_fraction(self, species: str) -> float:
        """Return mole fraction of a given species, 0.0 if absent."""
        return self.species.get(species, 0.0)


class PlanetState(BaseModel):
    """Complete physical state of a planetary body.

    This is the central data model of NOArCO. All simulation and
    optimization modules consume and produce `PlanetState` instances.

    Parameters
    ----------
    body_name : str
        Common name of the body (e.g. 'Mars', 'Venus').
    surface_pressure_pa : float
        Mean surface pressure in Pascals. For Mars E0: 610 Pa.
    mean_temperature_k : float
        Global mean surface temperature in Kelvin. For Mars E0: ~210 K.
    gas_composition : GasComposition
        Mole fractions of atmospheric species.
    surface_albedo : float
        Bond albedo of the surface (0-1). Mars E0: ~0.25.
    gravity_ms2 : float
        Mean surface gravitational acceleration (m/s2). Mars: 3.72.
    radius_m : float
        Mean planetary radius (m). Mars: 3.3895e6.
    mass_kg : float
        Planetary mass (kg). Mars: 6.4171e23.
    solar_constant_wm2 : float
        Solar irradiance at the body's mean orbital distance (W/m2).
        Mars: ~586 W/m2.
    ir_optical_depth : float
        Effective infrared optical depth of the atmosphere. Mars E0: ~0.05.
    uv_flux_wm2 : float, optional
        Mean surface UV flux (200-400 nm) in W/m2. None = not computed.
    co2_ice_kg : float, optional
        Total mass of CO2 ice available (polar caps + subsurface), in kg.
        Mars: ~5e16 kg (measured) + uncertain subsurface.
    h2o_ice_kg : float, optional
        Total mass of H2O ice available, in kg. Mars: ~1.2e19 to 5e19 kg.
    regolith_composition : dict[str, float], optional
        Mass fractions of key elements/oxides in the regolith.
        Keys use standard mineral/oxide notation (e.g. 'SiO2', 'Fe2O3').
    source_type : SourceType
        Epistemic status of this state snapshot.
    reference : str, optional
        Citation or data source for this state.
    notes : str, optional
        Free-text notes about this state.

    Examples
    --------
    >>> from noarco.data.planets import MARS
    >>> print(MARS.body_name, MARS.surface_pressure_pa)
    Mars 610.0
    """

    # --- Identity ---
    body_name: str = Field(description="Name of the planetary body.")

    # --- Atmosphere ---
    surface_pressure_pa: float = Field(
        gt=0.0, description="Mean surface pressure (Pa)."
    )
    mean_temperature_k: float = Field(
        gt=0.0, description="Global mean surface temperature (K)."
    )
    gas_composition: GasComposition = Field(
        description="Atmospheric mole fractions."
    )

    # --- Surface and orbital ---
    surface_albedo: float = Field(
        ge=0.0, le=1.0, description="Bond albedo of the surface."
    )
    gravity_ms2: float = Field(gt=0.0, description="Surface gravity (m/s2).")
    radius_m: float = Field(gt=0.0, description="Mean planetary radius (m).")
    mass_kg: float = Field(gt=0.0, description="Planetary mass (kg).")
    solar_constant_wm2: float = Field(
        gt=0.0, description="Solar irradiance at mean orbital distance (W/m2)."
    )

    # --- Radiative ---
    ir_optical_depth: float = Field(
        ge=0.0,
        default=0.0,
        description="Effective IR optical depth of the atmosphere.",
    )
    uv_flux_wm2: float | None = Field(
        default=None, description="Mean surface UV flux 200-400 nm (W/m2)."
    )

    # --- Volatile inventories ---
    co2_ice_kg: float | None = Field(
        default=None, description="Total CO2 ice reservoir (kg)."
    )
    h2o_ice_kg: float | None = Field(
        default=None, description="Total H2O ice reservoir (kg)."
    )

    # --- Regolith ---
    regolith_composition: dict[str, float] | None = Field(
        default=None,
        description="Regolith mass fractions by mineral/oxide (dimensionless).",
    )

    # --- Metadata ---
    source_type: SourceType = Field(
        default=SourceType.ESTIMATED,
        description="Epistemic status of this state.",
    )
    reference: str | None = Field(
        default=None, description="Citation for state parameters."
    )
    notes: str | None = Field(
        default=None, description="Free-text notes."
    )

    # --- Derived properties ---

    @property
    def name(self) -> str:
        """Alias for body_name."""
        return self.body_name

    @property
    def escape_velocity_m_s(self) -> float:
        """Planetary escape velocity in m/s: v_esc = sqrt(2 * G * M / R)."""
        G = G_NEWTON
        return float(np.sqrt(2.0 * G * self.mass_kg / self.radius_m))

    @property
    def surface_area_m2(self) -> float:
        """Total surface area of the body (m2)."""
        return 4.0 * np.pi * self.radius_m**2

    @property
    def atmospheric_mass_kg(self) -> float:
        """Total atmospheric mass derived from hydrostatic balance (kg).

        Uses the column pressure relation:
            P = M_atm * g / (4 * pi * R^2)
        => M_atm = P * 4 * pi * R^2 / g

        Reference: Turyshev (2026), Eq. 1.
        """
        return self.surface_pressure_pa * self.surface_area_m2 / self.gravity_ms2

    @property
    def equilibrium_temperature_k(self) -> float:
        """Planetary equilibrium temperature without greenhouse effect (K).

        T_eq = [ S_sun * (1 - A_s) / (4 * sigma) ]^(1/4)

        Reference: Pierrehumbert (2010), Eq. 3.1.
        """
        sigma = SIGMA_SB
        return float(((self.solar_constant_wm2 * (1.0 - self.surface_albedo)) / (4.0 * sigma)) ** 0.25)

    @property
    def greenhouse_delta_t_k(self) -> float:
        """Greenhouse warming above equilibrium (K), 1-layer gray model.

        T_s = T_eq * (1 + tau_IR/2)^(1/4)
        => DeltaT = T_s - T_eq

        Reference: Pierrehumbert (2010), Section 4.3.
        """
        t_eq = self.equilibrium_temperature_k
        t_s_model = t_eq * (1.0 + self.ir_optical_depth / 2.0) ** 0.25
        return float(t_s_model - t_eq)

    def partial_pressures_pa(self) -> dict[str, float]:
        """Partial pressures (Pa) for each atmospheric species."""
        return self.gas_composition.partial_pressure_pa(self.surface_pressure_pa)

    def summary(self) -> str:
        """Return a human-readable one-page summary of this state."""
        pp = self.partial_pressures_pa()
        lines = [
            f"=== PlanetState: {self.body_name} ===",
            f"  Source type     : {self.source_type.value}",
            f"  Surface pressure: {self.surface_pressure_pa:.1f} Pa  ({self.surface_pressure_pa/1e3:.4f} kPa)",
            f"  Mean temperature: {self.mean_temperature_k:.1f} K  ({self.mean_temperature_k - 273.15:.1f} deg C)",
            f"  Equilibrium T   : {self.equilibrium_temperature_k:.1f} K",
            f"  Greenhouse dT   : {self.greenhouse_delta_t_k:.1f} K",
            f"  Atmospheric mass: {self.atmospheric_mass_kg:.3e} kg",
            f"  Surface gravity : {self.gravity_ms2:.3f} m/s2",
            f"  Solar constant  : {self.solar_constant_wm2:.1f} W/m2",
            f"  Bond albedo     : {self.surface_albedo:.3f}",
            "  Gas composition (partial pressures):",
        ]
        for sp, pp_val in sorted(pp.items(), key=lambda x: -x[1]):
            lines.append(f"    {sp:6s}: {pp_val:.2f} Pa")
        if self.co2_ice_kg is not None:
            lines.append(f"  CO2 ice reservoir: {self.co2_ice_kg:.2e} kg")
        if self.h2o_ice_kg is not None:
            lines.append(f"  H2O ice reservoir: {self.h2o_ice_kg:.2e} kg")
        if self.reference:
            lines.append(f"  Reference       : {self.reference}")
        return "\n".join(lines)
