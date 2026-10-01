"""
noarco.radiative.balance
========================
Zero-dimensional (global mean) radiative balance and forcing calculations.

This module implements:
- Equilibrium temperature (no-atmosphere)
- 1-layer gray greenhouse model
- Radiative forcing from CO2, PFCs, and aerosols
- Required forcing to achieve a target temperature

All models are order-of-magnitude / 1D. For higher fidelity,
couple to LMD Planetary Climate Model or MarsWRF.

References
----------
Pierrehumbert, R. T. (2010). Principles of Planetary Climate.
    Cambridge University Press. (Sections 3.1, 4.3)
Turyshev, S. G. (2026). arXiv:2603.00402 (Section 3).
Marinova, M. M. et al. (2005). J. Geophys. Res. Planets, 110, E03002.
Wordsworth, R. et al. (2010). Icarus, 210, 992-997.
"""

from __future__ import annotations

import numpy as np

from noarco.core.constants import SIGMA_SB as _SIGMA
from noarco.core.planet_state import PlanetState


class RadiativeBalance:
    """Global-mean (0D) radiative balance calculator.

    Parameters
    ----------
    planet : PlanetState
        Current state of the target body.

    Examples
    --------
    >>> from noarco.data.planets import MARS
    >>> rb = RadiativeBalance(MARS)
    >>> rb.equilibrium_temperature_k()
    209.0  # approximately
    >>> rb.forcing_needed_for_delta_t(delta_t_k=63.0)
    132.0  # W/m^2, linearized, no feedbacks (E0 -> E1); exact: see forcing_needed_exact
    """

    def __init__(self, planet: PlanetState) -> None:
        self.planet = planet

    def equilibrium_temperature_k(self) -> float:
        """Planetary equilibrium temperature without greenhouse effect (K).

        T_eq = [ S * (1 - A) / (4 * sigma) ]^(1/4)

        Returns
        -------
        float
            Equilibrium temperature in Kelvin.

        References
        ----------
        Pierrehumbert (2010), Eq. 3.1.
        """
        p = self.planet
        return float(((p.solar_constant_wm2 * (1.0 - p.surface_albedo)) / (4.0 * _SIGMA)) ** 0.25)

    def forcing_needed_for_delta_t(
        self,
        delta_t_k: float,
        feedback_factor: float = 1.0,
    ) -> float:
        """TOA forcing (W/m2) that raises the effective radiating temperature by ``delta_t_k``.

        Exact Stefan-Boltzmann balance (Turyshev 2026, Eq. 30-31):

            sigma * T_e^4 = S (1 - A) / 4 + dF_TOA
            dF_TOA = sigma * [(T_eq + dT)^4 - T_eq^4] / feedback_factor

        The linearized form ``4 sigma T_eq^3 dT`` under-estimates the requirement
        by ~40 % for dT = 60 K on Mars and is available as
        :meth:`forcing_needed_linearized`.

        Parameters
        ----------
        delta_t_k : float
            Required rise of the effective radiating temperature (K); negative
            values (cooling) are allowed while T_eq + dT > 0.
        feedback_factor : float
            Amplification by positive climate feedbacks (> 0). 1.0 = none.
            A value above 1 *reduces* the external forcing needed and must be
            justified by a model; it is an assumption, not a measurement.

        Returns
        -------
        float
            Required top-of-atmosphere forcing in W/m2.

        References
        ----------
        Turyshev (2026), arXiv:2603.00402, Eqs. (30)-(31): 30 K -> 78 W/m2 and
        60 K -> 191 W/m2 for Mars (T_e0 = 210 K).
        """
        if feedback_factor <= 0.0:
            raise ValueError("feedback_factor must be positive.")
        t_eq = self.equilibrium_temperature_k()
        if t_eq + delta_t_k <= 0.0:
            raise ValueError("Resulting temperature must be positive.")
        return float(_SIGMA * ((t_eq + delta_t_k) ** 4 - t_eq**4) / feedback_factor)

    def forcing_needed_for_surface_temperature(self, target_surface_temperature_k: float) -> float:
        """TOA forcing (W/m2) to reach a target *surface* temperature at fixed greenhouse opacity.

        Holding the emissivity mapping fixed, ``T_e / T_s`` stays equal to its present
        value ``r = T_eq / T_mean`` so the required effective temperature is
        ``T_e' = r * T_s'`` and ``dF = sigma (T_e'^4 - T_eq^4)``. For Mars
        (r ~ 1) this reduces to Turyshev (2026) Eq. (31). It does not include feedbacks
        or opacity changes; use :meth:`required_ir_optical_depth` for greenhouse pathways.
        """
        if target_surface_temperature_k <= 0.0:
            raise ValueError("target temperature must be positive.")
        r = self.equilibrium_temperature_k() / self.planet.mean_temperature_k
        t_e_target = r * target_surface_temperature_k
        return float(_SIGMA * (t_e_target**4 - self.equilibrium_temperature_k() ** 4))

    def temperature_rise_for_greenhouse_forcing(self, delta_f_wm2: float) -> float:
        """Surface warming (K) caused by a *greenhouse-type* forcing (reduced outgoing longwave).

        In the grey atmosphere of Turyshev (2026), Eq. (32), the absorbed flux (and so the
        effective temperature ``T_e``) is unchanged and the surface temperature satisfies
        ``T_s^4 = T_e^4 / g(tau)``. A forcing ``dF = OLR_before - OLR_after`` at fixed ``T_s``
        lowers ``g`` by ``dF / (sigma T_s^4)``; restoring balance gives the closed form

            T_s' = T_s * (1 - dF / (sigma T_e^4)) ** (-1/4)

        which warms the surface more per W/m2 than the direct-forcing route of
        :meth:`temperature_rise_for_forcing` (absorbed-flux change, ``T_s'``
        = ``T_s (1 + dF / sigma T_e^4)^(1/4)``). Valid for ``dF < sigma T_e^4`` (the OLR).
        """
        t_e = self.equilibrium_temperature_k()
        x = delta_f_wm2 / (_SIGMA * t_e**4)
        if x >= 1.0:
            raise ValueError("Forcing at or above the outgoing longwave radiation is unphysical here.")
        return float(self.planet.mean_temperature_k * ((1.0 - x) ** -0.25 - 1.0))

    def greenhouse_forcing_for_temperature_rise(self, delta_t_k: float) -> float:
        """Greenhouse-type forcing (W/m2) needed for a surface warming (inverse of the above)."""
        ts = self.planet.mean_temperature_k
        if ts + delta_t_k <= 0:
            raise ValueError("Resulting temperature must be positive.")
        t_e = self.equilibrium_temperature_k()
        return float(_SIGMA * t_e**4 * (1.0 - (ts / (ts + delta_t_k)) ** 4))

    def forcing_needed_linearized(
        self,
        delta_t_k: float,
        feedback_factor: float = 1.0,
    ) -> float:
        """Linearized forcing ``4 sigma T_eq^3 dT / feedback`` (valid for dT/T_eq << 0.1)."""
        if feedback_factor <= 0.0:
            raise ValueError("feedback_factor must be positive.")
        t_eq = self.equilibrium_temperature_k()
        return float(4.0 * _SIGMA * t_eq**3 * delta_t_k / feedback_factor)

    def temperature_rise_for_forcing(self, delta_f_wm2: float) -> float:
        """Inverse of :meth:`forcing_needed_for_delta_t` (no feedback): exact dT for a *direct*
        (absorbed-flux) forcing such as mirrors or albedo. For greenhouse-type forcings use
        :meth:`temperature_rise_for_greenhouse_forcing`."""
        t_eq = self.equilibrium_temperature_k()
        arg = t_eq**4 + delta_f_wm2 / _SIGMA
        if arg <= 0.0:
            raise ValueError("Forcing would drive the effective temperature to zero.")
        return float(arg**0.25 - t_eq)

    def forcing_needed_exact(
        self,
        t_from_k: float,
        t_to_k: float,
        feedback_factor: float = 1.0,
    ) -> float:
        """Non-linear radiative forcing to move emission from T_from to T_to (W/m2).

        DF = sigma * (T_to^4 - T_from^4) / feedback_factor

        Exact (no linearization error), valid for any temperature pair.
        """
        if t_from_k <= 0.0 or t_to_k <= 0.0:
            raise ValueError("Temperatures must be positive (K).")
        if feedback_factor <= 0.0:
            raise ValueError("feedback_factor must be positive.")
        return _SIGMA * (t_to_k**4 - t_from_k**4) / feedback_factor

    def forcing_from_co2_doubling(
        self,
        n_doublings: float,
        alpha_wm2: float = 6.0,
    ) -> float:
        """Radiative forcing from CO2 pressure increase (W/m2).

        DF = alpha * ln(P_CO2 / P_CO2_0) / ln(2) * N_doublings
        Simplified: DF = alpha * N_doublings  (per doubling)

        Parameters
        ----------
        n_doublings : float
            Number of doublings of CO2 pressure.
        alpha_wm2 : float
            Forcing per doubling of CO2 (W/m2). Mars estimate: 5-8 W/m2.
            Default 6.0. Reference: Wordsworth et al. (2010).

        Returns
        -------
        float
            Radiative forcing in W/m2.

        References
        ----------
        Wordsworth et al. (2010). Icarus, 210, 992-997.
        """
        return alpha_wm2 * n_doublings

    def forcing_from_pressure_ratio(
        self,
        p_current_pa: float,
        p_target_pa: float,
        alpha_wm2: float = 6.0,
    ) -> float:
        """Radiative forcing from a CO2 pressure change (W/m2).

        DF = alpha * log2(P_target / P_current)

        Parameters
        ----------
        p_current_pa : float
            Current CO2 partial pressure (Pa).
        p_target_pa : float
            Target CO2 partial pressure (Pa).
        alpha_wm2 : float
            Forcing per CO2 doubling (W/m2).

        Returns
        -------
        float
            Radiative forcing in W/m2.
        """
        if p_current_pa <= 0 or p_target_pa <= 0:
            raise ValueError("Pressures must be positive.")
        return float(alpha_wm2 * np.log2(p_target_pa / p_current_pa))

    def eddington_surface_temperature(self, tau_ir: float) -> float:
        """Surface temperature (K) from the Eddington grey relation (Turyshev 2026, Eq. 32).

            T_s^4 = (3/4) * T_e^4 * (tau_IR + 2/3)

        An *architecture-level* mapping between longwave optical depth and surface
        temperature; it is not meant for the thin present-day Mars atmosphere
        (tau << 1) and carries +/-30-50 % uncertainty on tau for real spectra.
        """
        if tau_ir < 0.0:
            raise ValueError("tau_ir must be >= 0.")
        return float(self.equilibrium_temperature_k() * (0.75 * (tau_ir + 2.0 / 3.0)) ** 0.25)

    def required_ir_optical_depth(self, target_surface_temperature_k: float) -> float:
        """Grey IR optical depth needed for a target surface temperature (Turyshev 2026, Eq. 33).

            tau_IR = (4/3) * (T_s / T_e)^4 - 2/3

        Mars (T_e = 210 K): 250 K -> 2.0, 273 K -> 3.1.
        """
        if target_surface_temperature_k <= 0.0:
            raise ValueError("target temperature must be positive.")
        ratio = target_surface_temperature_k / self.equilibrium_temperature_k()
        return float(4.0 / 3.0 * ratio**4 - 2.0 / 3.0)

    def greenhouse_temperature_gray(
        self, tau_ir: float | None = None
    ) -> float:
        """Surface temperature from 1-layer gray greenhouse model (K).

        T_s = T_eq * (1 + tau_IR / 2)^(1/4)

        Parameters
        ----------
        tau_ir : float, optional
            IR optical depth. If None, uses planet.ir_optical_depth.

        Returns
        -------
        float
            Surface temperature in Kelvin.

        References
        ----------
        Pierrehumbert (2010), Section 4.3.
        """
        tau = tau_ir if tau_ir is not None else self.planet.ir_optical_depth
        t_eq = self.equilibrium_temperature_k()
        return float(t_eq * (1.0 + tau / 2.0) ** 0.25)

    def albedo_forcing(
        self,
        delta_albedo: float,
        area_fraction: float = 1.0,
    ) -> float:
        """Radiative forcing from surface albedo change (W/m2).

        DF_albedo = -(S / 4) * DA * f_area

        Positive DF means warming (albedo decrease).

        Parameters
        ----------
        delta_albedo : float
            Change in albedo (negative = darker = more absorption = warming).
        area_fraction : float
            Fraction of planetary surface affected. Default 1.0 (global).

        Returns
        -------
        float
            Global mean radiative forcing (W/m2).
        """
        return -1.0 * (self.planet.solar_constant_wm2 / 4.0) * delta_albedo * area_fraction

    def summary(self, target_temperature_k: float | None = None) -> str:
        """Print a formatted radiative balance summary."""
        p = self.planet
        t_eq = self.equilibrium_temperature_k()
        t_gh = self.greenhouse_temperature_gray()
        lines = [
            f"=== RadiativeBalance: {p.body_name} ===",
            f"  Solar constant      : {p.solar_constant_wm2:.1f} W/m2",
            f"  Bond albedo         : {p.surface_albedo:.3f}",
            f"  Equilibrium T (no GH): {t_eq:.1f} K",
            f"  IR optical depth    : {p.ir_optical_depth:.3f}",
            f"  Greenhouse T (1-layer): {t_gh:.1f} K",
            f"  Observed mean T     : {p.mean_temperature_k:.1f} K",
            f"  GH warming (model)  : {t_gh - t_eq:.1f} K",
        ]
        if target_temperature_k is not None:
            dt = target_temperature_k - p.mean_temperature_k
            df = self.forcing_needed_for_delta_t(dt)
            lines.append(f"  --- Target: {target_temperature_k:.1f} K ---")
            lines.append(f"  Required DT         : {dt:.1f} K")
            lines.append(f"  Required DF_TOA     : {df:.2f} W/m2")
        return "\n".join(lines)
