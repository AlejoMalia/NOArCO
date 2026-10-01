"""
noarco.mechanisms.greenhouse_gas
================================
Super-greenhouse gas (PFC-class) bookkeeping: concentration <-> mass <-> throughput.

Fluorinated compounds absorb strongly in the 8-13 um infrared window where CO2
and H2O are weak absorbers, and are very long lived. This module computes the
*inventory, production rate and replacement* implied by a target concentration,
and converts concentration to forcing with the **terrestrial radiative
efficiency** (W m^-2 ppb^-1, IPCC AR5 Table 8.A.1).

Validity and honesty
--------------------
* The terrestrial radiative efficiency is defined for small perturbations of
  Earth's atmosphere. Applying it to Mars, and to concentrations of ppm or more
  (where window saturation and the Mars-specific lapse-rate and surface
  temperature matter), is an **order-of-magnitude** estimate. Results above
  ``LINEAR_REGIME_LIMIT_PPB`` carry an explicit warning. Quantitative Mars
  forcing requires line-by-line/radiative-convective calculations such as
  Marinova et al. (2005).
* Turyshev (2026), Sec. VI.A, stresses that for PFC-class pathways the binding
  constraints are feedstock (fluorine) and synthesis throughput plus
  replacement, not the forcing law; this class reports those.

Supported gases (AR5 radiative efficiency RE in W m^-2 ppb^-1, GWP100, lifetime):
    CF4   RE 0.09  | GWP 6,630  | 50,000 yr
    C2F6  RE 0.25  | GWP 11,100 | 10,000 yr
    C3F8  RE 0.28  | GWP 8,900  |  2,600 yr
    SF6   RE 0.57  | GWP 23,500 |  3,200 yr
    mixed  equal-mole blend of the four (mean RE and GWP; shortest lifetime)

References
----------
IPCC AR5 (2013), WG1 Ch. 8, Table 8.A.1, pp. 731-738 (radiative efficiency, lifetime, GWP100;
    values checked against the published table).
Marinova, M. M., McKay, C. P. & Hashimoto, H. (2005). J. Geophys. Res. Planets 110, E03002.
Turyshev, S. G. (2026). arXiv:2603.00402, Sec. VI.A and Table VIII.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from noarco.core.constants import YEAR_S
from noarco.core.planet_state import PlanetState
from noarco.mechanisms.base import Mechanism, MechanismResult
from noarco.radiative.balance import RadiativeBalance

#: Above this concentration the small-perturbation terrestrial radiative efficiency is
#: not reliable (band saturation, Mars-specific thermal structure).
LINEAR_REGIME_LIMIT_PPB = 1000.0


class PFCGas(str, Enum):
    """Supported perfluorocarbon / super-greenhouse gases."""

    C2F6 = "C2F6"    # Hexafluoroethane
    CF4 = "CF4"      # Tetrafluoromethane
    SF6 = "SF6"      # Sulphur hexafluoride
    C3F8 = "C3F8"    # Octafluoropropane
    MIXED = "mixed"  # Equal-mole blend of the four gases above


_PURE_GAS_PARAMS: dict[str, dict[str, Any]] = {
    "C2F6": {"gwp_100": 11_100, "lifetime_yr": 10_000, "molar_mass": 138.012e-3,
             "alpha_wm2_per_ppb": 0.25, "isru_difficulty": 0.6,
             "isru_feedstocks": "C (from CO2 reduction) + F2 (from fluorite/HF electrolysis)"},
    "CF4": {"gwp_100": 6_630, "lifetime_yr": 50_000, "molar_mass": 88.004e-3,
            "alpha_wm2_per_ppb": 0.09, "isru_difficulty": 0.55,
            "isru_feedstocks": "C (from CO2) + F2 (electrolysis of HF)"},
    "SF6": {"gwp_100": 23_500, "lifetime_yr": 3_200, "molar_mass": 146.055e-3,
            "alpha_wm2_per_ppb": 0.57, "isru_difficulty": 0.7,
            "isru_feedstocks": "S (from sulfates in regolith) + F2 (electrolysis)"},
    "C3F8": {"gwp_100": 8_900, "lifetime_yr": 2_600, "molar_mass": 188.019e-3,
             "alpha_wm2_per_ppb": 0.28, "isru_difficulty": 0.65,
             "isru_feedstocks": "C (from CO2) + F2 (electrolysis)"},
}


def _mixed_params() -> dict[str, Any]:
    g = list(_PURE_GAS_PARAMS.values())
    n = len(g)
    return {
        "gwp_100": sum(x["gwp_100"] for x in g) / n,
        "lifetime_yr": min(x["lifetime_yr"] for x in g),
        "molar_mass": sum(x["molar_mass"] for x in g) / n,
        "alpha_wm2_per_ppb": sum(x["alpha_wm2_per_ppb"] for x in g) / n,
        "isru_difficulty": sum(x["isru_difficulty"] for x in g) / n,
        "isru_feedstocks": "Equal-mole blend of C2F6 + CF4 + SF6 + C3F8",
    }


_GAS_PARAMS: dict[str, dict[str, Any]] = {**_PURE_GAS_PARAMS, "mixed": _mixed_params()}


class SuperGreenhouseGas(Mechanism):
    """Super-greenhouse gas injection: concentration, mass, throughput and replacement."""

    def __init__(
        self,
        gas: PFCGas = PFCGas.C2F6,
        deployment_strategy: str = "continuous",
        target_concentration_ppb: float | None = None,
        synthesis_energy_j_per_kg: float = 30e6,
    ) -> None:
        params = _GAS_PARAMS[gas.value]
        super().__init__(
            name=f"SuperGreenhouseGas ({gas.value})",
            description=(
                f"Injection of {gas.value} (GWP={params['gwp_100']:,.0f}, "
                f"lifetime={params['lifetime_yr']:,} yr) for persistent IR forcing. "
                f"ISRU feedstocks: {params['isru_feedstocks']}. "
                "Reference: IPCC AR5 Table 8.A.1; Marinova et al. (2005); Turyshev (2026) Sec. VI.A."
            ),
        )
        self.gas = gas
        self._params = params
        self.deployment_strategy = deployment_strategy
        self.target_concentration_ppb = target_concentration_ppb
        self.synthesis_energy_j_per_kg = synthesis_energy_j_per_kg

    def is_applicable(self, planet: PlanetState) -> bool:
        return planet.surface_pressure_pa > 0.1

    def _forcing_for_concentration(self, c_ppb: float) -> float:
        """Forcing (W/m2) = RE (W/m2/ppb) x concentration (ppb); linear, small-perturbation."""
        return float(self._params["alpha_wm2_per_ppb"] * c_ppb)

    def _concentration_for_forcing(self, delta_f_wm2: float) -> float:
        return float(delta_f_wm2 / self._params["alpha_wm2_per_ppb"])

    def _mass_for_concentration(self, planet: PlanetState, c_ppb: float) -> float:
        """Mass (kg) of gas for a mole-fraction concentration in the planet's own atmosphere.

        ``M = c * 1e-9 * (M_atm / mu_atm) * mu_gas`` with ``mu_atm`` the mole-fraction-weighted
        mean molar mass of the planet's atmosphere (not a Mars constant).
        """
        from noarco.inventory.atmosphere import AtmosphericInventory

        mu_air = AtmosphericInventory(planet)._mean_molar_mass_kg_per_mol()
        n_atm = planet.atmospheric_mass_kg / mu_air
        return float(c_ppb * 1e-9 * n_atm * self._params["molar_mass"])

    def _required_power_for_synthesis(self, mass_kg: float, years: float) -> float:
        seconds = years * YEAR_S
        return mass_kg * self.synthesis_energy_j_per_kg / seconds if seconds > 0 else 0.0

    def compute(
        self,
        planet: PlanetState,
        target_delta_t_k: float | None = None,
        target_delta_f_wm2: float | None = None,
        deployment_years: float = 100.0,
    ) -> MechanismResult:
        if not self.is_applicable(planet):
            raise ValueError(f"SuperGreenhouseGas not applicable: P={planet.surface_pressure_pa} Pa")

        rb = RadiativeBalance(planet)
        if self.target_concentration_ppb is not None:
            c_ppb = self.target_concentration_ppb
            delta_f = self._forcing_for_concentration(c_ppb)
            delta_t = rb.temperature_rise_for_greenhouse_forcing(delta_f)
        elif target_delta_t_k is not None:
            delta_t = target_delta_t_k
            delta_f = rb.greenhouse_forcing_for_temperature_rise(delta_t)
            c_ppb = self._concentration_for_forcing(delta_f)
        elif target_delta_f_wm2 is not None:
            delta_f = target_delta_f_wm2
            c_ppb = self._concentration_for_forcing(delta_f)
            delta_t = rb.temperature_rise_for_greenhouse_forcing(delta_f)
        else:
            raise ValueError("Provide target_delta_t_k, target_delta_f_wm2, or target_concentration_ppb.")

        total_mass = self._mass_for_concentration(planet, c_ppb)
        power = self._required_power_for_synthesis(total_mass, deployment_years)
        seconds = deployment_years * YEAR_S
        throughput = total_mass / seconds if seconds > 0 else 0.0
        lifetime = float(self._params["lifetime_yr"])
        # Replacement flow needed to hold the burden against first-order loss M/tau.
        replacement_kg_s = total_mass / (lifetime * YEAR_S)
        isru_fraction = max(0.0, 1.0 - self._params["isru_difficulty"])

        warnings = [
            ("Forcing uses the terrestrial radiative efficiency (IPCC AR5) as an "
            "order-of-magnitude estimate for this atmosphere; Mars-specific radiative-convective "
            "calculations (e.g. Marinova et al. 2005) are needed for quantitative forcing."),
            ("Synthesis energy per kg is an engineering placeholder "
            f"({self.synthesis_energy_j_per_kg:.2e} J/kg)."),
        ]
        if c_ppb > LINEAR_REGIME_LIMIT_PPB:
            warnings.append(
                f"Concentration {c_ppb:.0f} ppb exceeds the small-perturbation regime "
                f"({LINEAR_REGIME_LIMIT_PPB:.0f} ppb): window saturation makes the linear forcing "
                "an over-estimate."
            )
        notes = (
            f"Target concentration: {c_ppb:.1f} ppb {self.gas.value}. "
            f"Build throughput: {throughput:.2e} kg/s over {deployment_years:.0f} yr. "
            f"Replacement to hold the burden (first-order loss, tau = {lifetime:,.0f} yr): "
            f"{replacement_kg_s:.2e} kg/s. "
            f"ISRU feedstocks: {self._params['isru_feedstocks']}."
        )
        return MechanismResult(
            mechanism_name=self.name,
            total_mass_kg=total_mass,
            power_w=power,
            throughput_kg_s=throughput,
            delta_forcing_wm2=delta_f,
            delta_temperature_k=delta_t,
            deployment_time_years=deployment_years,
            effect_duration_years=lifetime,
            risk_index=0.55,
            isru_fraction=isru_fraction,
            notes=notes,
            warnings=warnings,
            model_class="engineering_estimate",
            source_references=[
                "IPCC AR5 (2013), WG1 Ch. 8, Table 8.A.1.",
                "Marinova, M. M. et al. (2005). J. Geophys. Res. Planets 110, E03002.",
                "Turyshev, S. G. (2026). arXiv:2603.00402, Sec. VI.A and Table VIII.",
            ],
        )
