"""
noarco.sim.temporal — zero-dimensional energy-balance and mass-bookkeeping simulator
====================================================================================

A transparent global-mean model with **no invented feedback constants**. State:
surface temperature ``T``, surface pressure ``P``, CO2 and O2 inventories.

Energy balance (grey atmosphere, Turyshev 2026 Eq. 30-33)
---------------------------------------------------------
    C dT/dt = F_abs(t) - g(tau) * sigma * T^4,         g(tau) = 4 / (3 (tau + 2/3))

* ``F_abs = S (1 - A) / 4 + dF_direct(t)`` with ``dF_direct`` the absorbed flux added by
  mirrors.
* ``tau = tau_0 + dtau(t)``; ``tau_0`` is calibrated so that the initial state is a steady
  state (``tau_0 = (4/3) (T_0 / T_e)^4 - 2/3``, Eq. 33). Greenhouse-type forcings (aerosols,
  PFCs, CO2 pressure) enter as an optical-depth increment with the same top-of-atmosphere
  forcing ``dF = sigma T^4 (g(tau_0) - g(tau_0 + dtau))``.
* ``C`` is the heat capacity of the *active* column: atmosphere (``c_p M / A``) plus a
  regolith layer of depth ``thermal_depth_m`` (default 2 m). It is integrated with an
  unconditionally stable implicit Euler step.

Mass bookkeeping
----------------
Hydrostatic (Eq. 17): ``dP = g dM / A``. CO2 release is capped by the mechanism's
inventory, O2 accumulates from electrolysis, and CO2 partial pressure feeds back on the
temperature through the logarithmic forcing law of :class:`CO2Mobilization`.

What this model does **not** contain (and therefore cannot predict)
-------------------------------------------------------------------
Latent-heat and polar-ice reservoirs (which delay and buffer warming for decades to
millennia), CO2 condensation/collapse, water-vapour, cloud and albedo feedbacks,
escape and geochemical sinks, latitudinal structure and aerosol microphysics. It is a
quasi-equilibrium bookkeeping tool: the temperature follows the forcing with the
column's short radiative time-scale. Use a 3-D GCM for any quantitative claim about
transients. Mechanisms act over their deployment time (linear ramp) and are then held.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from noarco.core.constants import SIGMA_SB, YEAR_S
from noarco.core.endpoints import MARS_ENDPOINTS
from noarco.core.planet_state import PlanetState
from noarco.mechanisms.aerogel import SilicaAerogel
from noarco.mechanisms.aerosols import NanoparticleAerosol
from noarco.mechanisms.base import Mechanism
from noarco.mechanisms.co2_mobilization import CO2Mobilization
from noarco.mechanisms.electrolysis import Electrolysis
from noarco.mechanisms.greenhouse_gas import SuperGreenhouseGas
from noarco.mechanisms.mirrors import OrbitalMirror

#: Specific heat (J kg^-1 K^-1) of CO2 gas and of basaltic regolith (order of magnitude).
_CP_GAS = 740.0
_RHO_CP_REGOLITH = 1500.0 * 800.0


@dataclass
class SimulationTrajectory:
    time_years: np.ndarray
    surface_pressure_pa: np.ndarray
    mean_temperature_k: np.ndarray
    forcing_wm2: np.ndarray
    o2_partial_pressure_pa: np.ndarray
    final_state: PlanetState
    milestones_reached: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: Declared model class: a reduced-order 0-D energy/mass balance, not a GCM.
    model_class: str = "reduced_order_0D"

    def summary(self) -> str:
        lines = [
            "=== SimulationTrajectory Summary (reduced-order 0-D, not a GCM) ===",
            f"  Simulated Duration   : {self.time_years[-1]:,.1f} years ({len(self.time_years)} steps)",
            f"  Initial Pressure     : {self.surface_pressure_pa[0]:.1f} Pa  -> Final: {self.surface_pressure_pa[-1]:.1f} Pa",
            f"  Initial Temperature  : {self.mean_temperature_k[0]:.1f} K  -> Final: {self.mean_temperature_k[-1]:.1f} K",
            f"  Final Net Forcing    : {self.forcing_wm2[-1]:+.2f} W/m²",
            f"  Final pO2            : {self.o2_partial_pressure_pa[-1]:.1f} Pa",
        ]
        if self.milestones_reached:
            lines.append("  Milestones Reached:")
            for m, yr in self.milestones_reached.items():
                lines.append(f"    • {m}: at year {yr:,.1f}")
        for n in self.notes:
            lines.append(f"  Note: {n}")
        return "\n".join(lines)


def _g(tau: float) -> float:
    """Grey emissivity factor ``g(tau) = 4 / (3 (tau + 2/3))`` (surface-to-space OLR ratio)."""
    return 4.0 / (3.0 * (tau + 2.0 / 3.0))


def _tau_from_g(g: float) -> float:
    return 4.0 / (3.0 * g) - 2.0 / 3.0


@dataclass
class _Driver:
    """Time-dependent contribution of one mechanism (all linear ramps over ``ramp_years``)."""

    kind: str                 # "tau" | "mirror" | "co2" | "o2"
    ramp_years: float
    amount: float             # tau increment | W/m2 | kg | kg
    inventory_kg: float = 0.0  # co2 / o2 total mass to release

    def fraction(self, t: float) -> float:
        return 1.0 if self.ramp_years <= 0 else min(max(t / self.ramp_years, 0.0), 1.0)


class TemporalSimulator:
    """Global-mean energy-balance and mass-bookkeeping propagator (see module docstring).

    Parameters
    ----------
    initial_planet : PlanetState
        Initial state; it is taken to be in steady state.
    active_mechanisms : list[Mechanism], optional
        Mechanisms acting globally: :class:`NanoparticleAerosol`, :class:`SuperGreenhouseGas`,
        :class:`OrbitalMirror`, :class:`CO2Mobilization`, :class:`Electrolysis`. Regional
        mechanisms (:class:`SilicaAerogel`) have no global effect and are reported in
        ``notes``.
    design_delta_t_k : float
        Target warming used to size aerosol / PFC / mirror mechanisms that were not given an
        explicit concentration or optical depth (default 15 K).
    thermal_depth_m : float
        Depth of the regolith layer taking part in the heat capacity (default 2 m).
    """

    def __init__(
        self,
        initial_planet: PlanetState,
        active_mechanisms: list[Mechanism] | None = None,
        design_delta_t_k: float = 15.0,
        thermal_depth_m: float = 2.0,
    ) -> None:
        if design_delta_t_k < 0 or thermal_depth_m <= 0:
            raise ValueError("design_delta_t_k must be >= 0 and thermal_depth_m > 0.")
        self.initial_planet = initial_planet
        self.mechanisms = list(active_mechanisms or [])
        self.design_delta_t_k = design_delta_t_k
        self.thermal_depth_m = thermal_depth_m

    # ------------------------------------------------------------------ drivers
    def _drivers(self, planet: PlanetState) -> tuple[list[_Driver], list[str]]:
        drivers: list[_Driver] = []
        notes: list[str] = []
        for m in self.mechanisms:
            if not m.is_applicable(planet):
                notes.append(f"{m.name}: not applicable to {planet.body_name}; ignored.")
                continue
            if isinstance(m, SilicaAerogel):
                notes.append(f"{m.name}: regional, zero global forcing; not included in the global mean.")
            elif isinstance(m, CO2Mobilization):
                inv = m._get_inventory(planet)
                drivers.append(_Driver("co2", 200.0, 0.0, inv))
            elif isinstance(m, Electrolysis):
                res = m.compute(planet)
                o2_total = res.throughput_kg_s * 1000.0 * YEAR_S
                drivers.append(_Driver("o2", 1000.0, 0.0, o2_total))
            elif isinstance(m, OrbitalMirror):
                res = m.compute(planet, target_delta_t_k=self.design_delta_t_k)
                drivers.append(_Driver("mirror", res.deployment_time_years, res.delta_forcing_wm2))
            elif isinstance(m, NanoparticleAerosol):
                res = m.compute(planet, target_delta_t_k=self.design_delta_t_k)
                tau = m.target_optical_depth
                if tau is None:
                    tau = m._aerosol_optical_depth_for_delta_t(planet, self.design_delta_t_k)
                drivers.append(_Driver("tau", res.deployment_time_years, tau * m.coverage_fraction))
            elif isinstance(m, SuperGreenhouseGas):
                res = m.compute(planet, target_delta_t_k=self.design_delta_t_k)
                drivers.append(_Driver("ghg_forcing", res.deployment_time_years, res.delta_forcing_wm2))
            else:
                notes.append(f"{m.name}: unsupported mechanism type; ignored.")
        return drivers, notes

    # ------------------------------------------------------------------ run
    def run(self, duration_years: float = 200.0, dt_years: float = 1.0) -> SimulationTrajectory:
        """Propagate the state for ``duration_years`` with a fixed step (implicit Euler)."""
        if duration_years <= 0 or dt_years <= 0:
            raise ValueError("duration_years and dt_years must be > 0.")
        p0 = self.initial_planet
        area = p0.surface_area_m2
        g = p0.gravity_ms2
        t_e0 = p0.equilibrium_temperature_k
        tau0 = 4.0 / 3.0 * (p0.mean_temperature_k / t_e0) ** 4 - 2.0 / 3.0
        if tau0 < 0:
            tau0 = 0.0  # state colder than T_e: no greenhouse (steady state then only approximate)
        drivers, notes = self._drivers(p0)

        t_arr = [0.0]
        p_arr = [p0.surface_pressure_pa]
        temp = p0.mean_temperature_k
        temps = [temp]
        forc = [0.0]
        po2 = p0.gas_composition.mole_fraction("O2") * p0.surface_pressure_pa
        po2s = [po2]
        x_co2 = p0.gas_composition.mole_fraction("CO2")
        pco2_0 = x_co2 * p0.surface_pressure_pa
        pco2 = pco2_0
        released = {id(d): 0.0 for d in drivers}
        milestones: dict[str, float] = {}

        co2_alpha = 6.0
        for m in self.mechanisms:
            if isinstance(m, CO2Mobilization):
                co2_alpha = m.forcing_per_doubling_wm2
        c_gas = _CP_GAS * (p0.surface_pressure_pa / g)           # J m^-2 K^-1 (column, initial)
        c_total = c_gas + _RHO_CP_REGOLITH * self.thermal_depth_m

        n_steps = math.ceil(duration_years / dt_years)
        t = 0.0
        pressure = p0.surface_pressure_pa
        for _ in range(n_steps):
            t_new = min(t + dt_years, duration_years)
            dt = t_new - t
            # --- mass sources -------------------------------------------------------
            d_mass_co2 = d_mass_o2 = 0.0
            for d in drivers:
                if d.kind in ("co2", "o2"):
                    target = d.inventory_kg * d.fraction(t_new)
                    step = max(target - released[id(d)], 0.0)
                    released[id(d)] += step
                    if d.kind == "co2":
                        d_mass_co2 += step
                    else:
                        d_mass_o2 += step
            d_p_co2 = d_mass_co2 * g / area
            d_p_o2 = d_mass_o2 * g / area
            pressure += d_p_co2 + d_p_o2
            pco2 += d_p_co2
            po2 += d_p_o2

            # --- radiative drivers at t_new -----------------------------------------
            f_direct = 0.0
            d_tau = 0.0
            ghg_f = 0.0
            for d in drivers:
                if d.kind == "mirror":
                    f_direct += d.amount * d.fraction(t_new)
                elif d.kind == "tau":
                    d_tau += d.amount * d.fraction(t_new)
                elif d.kind == "ghg_forcing":
                    ghg_f += d.amount * d.fraction(t_new)
            co2_f = co2_alpha * math.log2(pco2 / pco2_0) if pco2_0 > 0 and pco2 > pco2_0 else 0.0
            t_ref = temps[0]
            olr_ref = SIGMA_SB * t_ref**4
            # Convert OLR-reducing (greenhouse) forcings to an optical-depth increment.
            g_now = max(_g(tau0 + d_tau) - (co2_f + ghg_f) / olr_ref, 1e-6)
            tau = _tau_from_g(g_now)
            f_abs = p0.solar_constant_wm2 * (1.0 - p0.surface_albedo) / 4.0 + f_direct

            def residual(tn: float, told: float = temp, fa: float = f_abs,
                         gg: float = g_now, dt_y: float = dt) -> float:
                return c_total * (tn - told) / (dt_y * YEAR_S) - fa + gg * SIGMA_SB * tn**4

            temp = float(brentq(residual, 1.0, 5000.0, xtol=1e-10))
            total_forcing = f_direct + (_g(tau0) - g_now) * olr_ref
            t = t_new
            t_arr.append(t)
            p_arr.append(pressure)
            temps.append(temp)
            forc.append(total_forcing)
            po2s.append(po2)
            _ = tau

            # --- milestones from the end-state definitions -----------------------------
            for label, ep in (("E1", MARS_ENDPOINTS["E1"]), ("E3", MARS_ENDPOINTS["E3"]),
                              ("E4", MARS_ENDPOINTS["E4"])):
                if label in milestones:
                    continue
                if (pressure >= ep.min_pressure_pa and temp >= ep.min_temperature_k
                        and po2 >= ep.min_o2_pa and pco2 <= ep.max_co2_pa + (pco2_0 if label != "E4" else 0.0)):
                    milestones[f"{label} ({ep.name})"] = t

        final_planet = p0.model_copy(update={
            "surface_pressure_pa": pressure,
            "mean_temperature_k": temp,
        })
        notes.append(
            "Quasi-equilibrium 0-D model: no latent-heat/ice reservoirs, no CO2 condensation "
            "or collapse, no water-vapour/cloud/albedo feedbacks, no escape or sinks."
        )
        return SimulationTrajectory(
            time_years=np.array(t_arr),
            surface_pressure_pa=np.array(p_arr),
            mean_temperature_k=np.array(temps),
            forcing_wm2=np.array(forc),
            o2_partial_pressure_pa=np.array(po2s),
            final_state=final_planet,
            milestones_reached=milestones,
            notes=notes,
        )
