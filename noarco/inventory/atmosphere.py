"""
noarco.inventory.atmosphere
===========================
Calculates mass inventories for atmospheric modification:
- Mass of atmosphere needed per unit pressure
- Delta-mass required to reach a target pressure
- CO2 available vs CO2 needed (gap analysis)
- PFC mass requirements for radiative forcing

Key reference result (validation target):
    For Mars: 3.89 x 10^15 kg per mbar of pressure
    Source: Turyshev (2026), arXiv:2603.00402, Section 2.

References
----------
Turyshev, S. G. (2026). arXiv:2603.00402.
Jakosky, B. M. (2019). Planet. Space Sci., 175, 52-59.
Marinova, M. M. et al. (2005). J. Geophys. Res. Planets, 110, E03002.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from noarco.core.constants import MOLAR_MASS
from noarco.core.planet_state import PlanetState


class AtmosphericInventory:
    """Computes mass inventories for atmospheric modification scenarios.

    Parameters
    ----------
    planet : PlanetState
        Current state of the target body.

    Examples
    --------
    >>> from noarco.data.planets import MARS
    >>> inv = AtmosphericInventory(MARS)
    >>> print(f"{inv.mass_per_mbar():.3e} kg/mbar")
    3.890e+15 kg/mbar
    >>> print(f"{inv.delta_mass_for_pressure(target_pa=6000):.3e} kg")
    2.091e+17 kg
    """

    # Molar masses (kg/mol)
    _MOLAR_MASS: ClassVar[dict[str, float]] = {
        **MOLAR_MASS,
        "C2F6": 138.012e-3,
        "CF4": 88.004e-3,
        "SF6": 146.055e-3,
    }

    def __init__(self, planet: PlanetState) -> None:
        self.planet = planet

    # ------------------------------------------------------------------
    # Core hydrostatic relation
    # ------------------------------------------------------------------

    def mass_per_pascal(self) -> float:
        """Atmospheric mass (kg) needed per Pascal of surface pressure.

        Derived from the hydrostatic relation:
            P = M_atm * g / (4 * pi * R^2)
        => dM/dP = 4 * pi * R^2 / g

        Returns
        -------
        float
            kg Pa^-1

        References
        ----------
        Turyshev (2026), Eq. 1.
        """
        return self.planet.surface_area_m2 / self.planet.gravity_ms2

    def mass_per_mbar(self) -> float:
        """Atmospheric mass (kg) per mbar (100 Pa) of surface pressure.

        Validation target for Mars: 3.89 x 10^15 kg/mbar.
        (Turyshev 2026, Table 2).

        Returns
        -------
        float
            kg mbar^-1
        """
        return self.mass_per_pascal() * 100.0  # 1 mbar = 100 Pa

    def current_atmospheric_mass_kg(self) -> float:
        """Total atmospheric mass of the current state (kg).

        Returns
        -------
        float
            Total atmospheric mass in kg.
        """
        return self.planet.atmospheric_mass_kg

    def target_atmospheric_mass_kg(self, target_pa: float) -> float:
        """Total atmospheric mass needed to reach a target pressure (kg).

        Parameters
        ----------
        target_pa : float
            Target surface pressure in Pascals.

        Returns
        -------
        float
            Required total atmospheric mass in kg.
        """
        return target_pa * self.planet.surface_area_m2 / self.planet.gravity_ms2

    def delta_mass_for_pressure(
        self, target_pa: float, species: str = "CO2"
    ) -> float:
        """Additional mass of a gas species needed to reach target pressure.

        Assumes the added gas is entirely composed of `species` and that
        the background atmosphere remains unchanged.

        Parameters
        ----------
        target_pa : float
            Target total surface pressure (Pa).
        species : str
            Gas species to add (default: 'CO2').

        Returns
        -------
        float
            Additional mass of `species` needed (kg). Negative means current
            pressure already exceeds target.

        Notes
        -----
        The hydrostatic relation fixes the *mass* of gas per unit pressure
        independently of its molar mass (P = M g / A), so the result is the
        same for any pure ``species``; the argument is validated and kept for
        API symmetry.

        For Mars (610 Pa -> 6 kPa) this returns ~2.1e17 kg, about 4x the
        5e16 kg of measured polar CO2 ice (Jakosky 2019; Turyshev 2026),
        which is the fundamental inventory bottleneck.
        """
        if species not in self._MOLAR_MASS:
            raise ValueError(
                f"Unknown species '{species}'. Known: {sorted(self._MOLAR_MASS)}"
            )
        if not np.isfinite(target_pa) or target_pa < 0.0:
            raise ValueError(f"target_pa must be finite and >= 0, got {target_pa!r}")
        delta_p = target_pa - self.planet.surface_pressure_pa
        return float(delta_p * self.mass_per_pascal())

    # ------------------------------------------------------------------
    # Endpoint-normalised bounds (Turyshev 2026, Sec. II.C and III-VII)
    # ------------------------------------------------------------------

    def species_mass_for_partial_pressure(
        self, partial_pressure_pa: float, species: str, mean_molar_mass_kg_mol: float
    ) -> float:
        """Mass (kg) of a constituent of a well-mixed atmosphere for a target partial pressure.

            M_i = K * (mu_i / mu_bar) * p_i      (Turyshev 2026, Eq. 4)

        with K = 4 pi R^2 / g the mass per pascal and ``p_i = x_i P`` the mole-fraction
        partial pressure. The ``mu_i / mu_bar`` factor is the mass fraction carried per
        unit mole fraction; it is ~0.05 for H2 in a CO2 atmosphere.
        """
        if partial_pressure_pa < 0 or mean_molar_mass_kg_mol <= 0:
            raise ValueError("partial pressure must be >= 0 and mean molar mass > 0.")
        if species not in self._MOLAR_MASS:
            raise ValueError(f"Unknown species '{species}'. Known: {sorted(self._MOLAR_MASS)}")
        return float(self.mass_per_pascal() * partial_pressure_pa
                     * self._MOLAR_MASS[species] / mean_molar_mass_kg_mol)

    def oxygen_mass_for_partial_pressure(self, po2_pa: float) -> float:
        """O2 mass (kg) for a partial pressure, ``K * p_O2`` (Turyshev 2026, Eq. 67).

        Mars, 21 kPa -> 8.2e17 kg. This is the hydrostatic column weight, i.e. it
        omits the ``mu_O2 / mu_bar`` (~1.1 for an Earth-like mix) factor of Eq. 4.
        """
        if po2_pa < 0:
            raise ValueError("po2_pa must be >= 0.")
        return float(self.mass_per_pascal() * po2_pa)

    @staticmethod
    def minimum_o2_production_energy_j(o2_mass_kg: float) -> float:
        """Reversible (Gibbs) minimum energy to electrolyse water into ``o2_mass_kg`` of O2.

        ``E_min = M_O2 * 2 dG / mu_O2`` = 14.8 MJ per kg (Turyshev 2026, Eq. 5 and 72).
        Real plants need this divided by an efficiency well below 1.
        """
        from noarco.core.constants import reversible_o2_energy_j_per_kg

        if o2_mass_kg < 0:
            raise ValueError("o2_mass_kg must be >= 0.")
        return float(o2_mass_kg * reversible_o2_energy_j_per_kg())

    def regional_gas_mass_kg(self, area_m2: float, pressure_pa: float) -> float:
        """Gas mass (kg) to pressurise an *open* region: ``A P / g`` (Turyshev 2026, Eq. 80)."""
        if area_m2 < 0 or pressure_pa < 0:
            raise ValueError("area and pressure must be >= 0.")
        return float(area_m2 * pressure_pa / self.planet.gravity_ms2)

    def max_pressurized_area_m2(
        self, mass_flow_kg_s: float, build_time_years: float, pressure_pa: float
    ) -> float:
        """Largest area that a net gas flow can pressurise in a build time (Eq. 82)."""
        from noarco.core.constants import YEAR_S

        if mass_flow_kg_s < 0 or build_time_years < 0 or pressure_pa <= 0:
            raise ValueError("flow and time must be >= 0 and pressure > 0.")
        return float(self.planet.gravity_ms2 * mass_flow_kg_s * build_time_years * YEAR_S / pressure_pa)

    def volumetric_conditioning_requirements(
        self,
        volume_m3: float,
        target_pa: float = 101_325.0,
        target_temp_k: float = 293.15,
        target_o2_fraction: float = 0.21,
        envelope_leak_rate_pct_day: float = 0.05,
    ) -> dict[str, float | str]:
        """Compute the gas mass, oxygen, thermal energy, and power
        required to condition a volume (m3), properly distinguishing
        closed habitats vs open atmospheres and accounting for ambient gas.
        """
        from noarco.analyzers.atmosphere import AtmosphereEngine

        req = AtmosphereEngine.compute_volume_conditioning(
            planet=self.planet,
            volume_m3=volume_m3,
            target_pa=target_pa,
            target_temp_k=target_temp_k,
            target_o2_fraction=target_o2_fraction,
        )

        return {
            "volume_m3": volume_m3,
            "mode": req.mode,
            "target_pressure_pa": target_pa,
            "target_temperature_k": target_temp_k,
            "total_gas_mass_kg": req.total_gas_mass_kg,
            "net_mass_delta_kg": req.net_mass_delta_kg,
            "o2_mass_kg": req.o2_mass_kg,
            "n2_mass_kg": req.n2_mass_kg,
            "initial_thermal_energy_j": req.thermal_energy_joules,
            "initial_thermal_kwh": req.thermal_energy_kwh,
            "compression_work_j": req.compression_work_joules,
            "total_conditioning_energy_j": req.total_conditioning_energy_joules,
            "total_conditioning_energy_kwh": req.total_conditioning_energy_kwh,
            "regime_description": req.regime_description,
        }


    # ------------------------------------------------------------------
    # Inventory gap analysis
    # ------------------------------------------------------------------

    def co2_inventory_gap_kg(
        self,
        target_pa: float,
        include_regolith_co2_kg: float = 0.0,
    ) -> dict[str, float]:
        """Compute the CO2 inventory gap between available and needed.

        Parameters
        ----------
        target_pa : float
            Target CO2 partial pressure (Pa).
        include_regolith_co2_kg : float
            Additional CO2 estimated in regolith (kg). Default 0.
            For Mars, this ranges from 2e17 to 4e18 kg (very uncertain).

        Returns
        -------
        dict with keys:
            'co2_needed_kg'     : Total CO2 mass needed
            'co2_available_kg'  : Estimated available CO2 (ice + regolith)
            'gap_kg'            : Shortfall (positive = deficit)
            'gap_mbar'          : Shortfall expressed in pressure (Pa)
            'isru_feasible'     : Whether ISRU can close the gap

        References
        ----------
        Jakosky (2019), Planet. Space Sci. 175, 52-59.
        Turyshev (2026), Table 3.
        """
        co2_needed = self.delta_mass_for_pressure(target_pa, "CO2")
        co2_ice = self.planet.co2_ice_kg or 0.0
        co2_available = co2_ice + include_regolith_co2_kg
        gap = co2_needed - co2_available
        gap_pa = gap / self.mass_per_pascal() if gap > 0 else 0.0

        return {
            "co2_needed_kg": co2_needed,
            "co2_available_kg": co2_available,
            "gap_kg": max(gap, 0.0),
            "gap_pa": gap_pa,
            "isru_feasible": gap <= 0.0,
        }

    # ------------------------------------------------------------------
    # PFC mass requirements
    # ------------------------------------------------------------------

    def pfc_mass_for_forcing(
        self,
        delta_f_wm2: float,
        radiative_efficiency_wm2_per_ppb: float = 0.25,
        molar_mass_kg_mol: float = 138.012e-3,
    ) -> float:
        """Mass (kg) of a fluorinated gas that produces a forcing at its terrestrial radiative efficiency.

            c [ppb] = dF / RE        M = c * 1e-9 * (M_atm / mu_atm) * mu_gas

        Defaults are C2F6 (IPCC AR5 Table 8.A.1: RE = 0.25 W m^-2 ppb^-1). The linear
        law holds only for small perturbations and Earth-like opacity, so this is an
        order-of-magnitude estimate for other bodies (see
        :class:`noarco.mechanisms.SuperGreenhouseGas` for gas catalogue and warnings).
        """
        if radiative_efficiency_wm2_per_ppb <= 0 or molar_mass_kg_mol <= 0:
            raise ValueError("radiative efficiency and molar mass must be > 0.")
        if delta_f_wm2 < 0:
            raise ValueError("delta_f_wm2 must be >= 0.")
        c_ppb = delta_f_wm2 / radiative_efficiency_wm2_per_ppb
        n_atm_mol = self.planet.atmospheric_mass_kg / self._mean_molar_mass_kg_per_mol()
        return float(c_ppb * 1e-9 * n_atm_mol * molar_mass_kg_mol)

    def _mean_molar_mass_kg_per_mol(self) -> float:
        """Mole-fraction-weighted mean molar mass of the planet's atmosphere.

        Falls back to CO2 (43.4 g/mol Mars-like) if no known species are present.
        """
        species = self.planet.gas_composition.species
        known = {sp: x for sp, x in species.items() if sp in self._MOLAR_MASS}
        total = sum(known.values())
        if total <= 0.0:
            return 43.4e-3
        return sum(x * self._MOLAR_MASS[sp] for sp, x in known.items()) / total

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def summary(self, target_pa: float | None = None) -> str:
        """Print a formatted inventory summary.

        Parameters
        ----------
        target_pa : float, optional
            If provided, include delta-mass analysis for this target pressure.
        """
        p = self.planet
        lines = [
            f"=== AtmosphericInventory: {p.body_name} ===",
            f"  Current pressure      : {p.surface_pressure_pa:.1f} Pa ({p.surface_pressure_pa/1e3:.4f} kPa)",
            f"  Atmospheric mass      : {self.current_atmospheric_mass_kg():.4e} kg",
            f"  kg per Pascal         : {self.mass_per_pascal():.4e} kg/Pa",
            f"  kg per mbar           : {self.mass_per_mbar():.4e} kg/mbar",
        ]
        if p.co2_ice_kg is not None:
            lines.append(f"  CO2 ice available     : {p.co2_ice_kg:.2e} kg")
        if p.h2o_ice_kg is not None:
            lines.append(f"  H2O ice available     : {p.h2o_ice_kg:.2e} kg")
        if target_pa is not None:
            dm = self.delta_mass_for_pressure(target_pa)
            gap = self.co2_inventory_gap_kg(target_pa)
            lines += [
                f"  --- Target: {target_pa:.0f} Pa ---",
                f"  Delta mass needed     : {dm:.4e} kg",
                f"  CO2 gap (vs ice only) : {gap['gap_kg']:.2e} kg",
                f"  ISRU feasible (ice)   : {gap['isru_feasible']}",
            ]
        return "\n".join(lines)
