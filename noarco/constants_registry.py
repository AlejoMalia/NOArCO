"""
noarco.constants_registry — where every constant comes from
===========================================================

``constant -> value -> unit -> source -> class``. Classes:

* ``exact``      - defined by SI (no uncertainty);
* ``measured``   - experimental/observational value with a published source;
* ``derived``    - computed from other registry entries (formula in ``source``);
* ``literature`` - taken from a named publication (a model result or table);
* ``estimated``  - engineering estimate or placeholder **without** a verified source;
* ``calibrated`` - fitted to external data in this repository (the data are named).

``verify_registry()`` checks each entry's value against the live constant in the code, so the table
cannot drift from the implementation; ``to_markdown()`` generates ``docs/CONSTANTS.md``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

CLASSES = ("exact", "measured", "derived", "literature", "estimated", "calibrated")


@dataclass(frozen=True)
class ConstantEntry:
    name: str
    value: float
    unit: str
    source: str
    klass: str
    live: Callable[[], float]          # reads the value actually used by the code

    def __post_init__(self) -> None:
        if self.klass not in CLASSES:
            raise ValueError(f"unknown class {self.klass!r}")


def _c(name: str, value: float, unit: str, source: str, klass: str, live: Callable[[], float]) -> ConstantEntry:
    return ConstantEntry(name, value, unit, source, klass, live)


def _entries() -> list[ConstantEntry]:
    import inspect

    from noarco.core import constants as k
    from noarco.extratools import agro, habitat, isru, logistics
    from noarco.extratools import logistics as lg
    from noarco.extratools.exoplanet import KOPPARAPU_2013
    from noarco.mechanisms import aerosols, co2_mobilization, greenhouse_gas
    from noarco.mechanisms.co2_mobilization import _MARS_CO2_INVENTORY as inv
    from noarco.radiative import literature as lit

    ap = greenhouse_gas._GAS_PARAMS
    return [
        _c("SIGMA_SB", 5.670374419e-8, "W m-2 K-4", "CODATA 2018 (exact in SI 2019)", "exact", lambda: k.SIGMA_SB),
        _c("G_NEWTON", 6.67430e-11, "m3 kg-1 s-2", "CODATA 2018", "measured", lambda: k.G_NEWTON),
        _c("R_GAS", 8.314462618, "J mol-1 K-1", "N_A * k_B (exact)", "exact", lambda: k.R_GAS),
        _c("K_BOLTZMANN", 1.380649e-23, "J K-1", "SI 2019", "exact", lambda: k.K_BOLTZMANN),
        _c("AU", 149_597_870_700.0, "m", "IAU 2012", "exact", lambda: k.AU_M),
        _c("YEAR_S (Julian)", 31_557_600.0, "s", "IAU", "exact", lambda: k.YEAR_S),
        _c("SOLAR_CONSTANT_1AU", 1361.0, "W m-2", "Kopp & Lean 2011 (TSI)", "measured", lambda: k.SOLAR_CONSTANT_1AU),
        _c("DELTA_G_H2O", 237_129.0, "J mol-1", "NIST/CRC standard Gibbs energy of formation, H2O(l)", "measured",
           lambda: k.DELTA_G_H2O_J_MOL),
        _c("DELTA_H_H2O", 285_830.0, "J mol-1", "NIST/CRC standard enthalpy of formation, H2O(l)", "measured",
           lambda: k.DELTA_H_H2O_J_MOL),
        _c("E_O2_reversible", 14.82e6, "J kg-1", "2 * DELTA_G_H2O / M_O2 (Turyshev 2026 Eq. 5: 14.8 MJ/kg)", "derived",
           k.reversible_o2_energy_j_per_kg),
        _c("CO2_SUBLIMATION", 5.9e5, "J kg-1", "NIST WebBook ~0.57-0.59 MJ/kg; Turyshev 2026 uses 6e5", "measured",
           lambda: k.CO2_SUBLIMATION_J_KG),
        _c("WATER_TRIPLE_POINT_PA", 611.657, "Pa", "IAPWS", "exact", lambda: k.WATER_TRIPLE_POINT_PA),
        _c("WATER_TRIPLE_POINT_K", 273.16, "K", "ITS-90 definition", "exact", lambda: k.WATER_TRIPLE_POINT_K),
        _c("ARMSTRONG_LIMIT_PA", 6270.0, "Pa", "water vapour pressure at 37 C (47 mmHg); Turyshev 2026 Sec. II.A", "measured",
           lambda: k.ARMSTRONG_LIMIT_PA),
        _c("MARS_CO2_polar_deposit", 2.3e16, "kg", "Turyshev 2026 Sec. V.B (~6 mbar), after Jakosky & Edwards 2018", "literature",
           lambda: inv["polar_deposit_south_kg"]),
        _c("MARS_CO2_accessible_reference", 7.78e16, "kg", "Turyshev 2026 Table III (~20 mbar); Jakosky & Edwards 2018 (~0.02 bar)",
           "literature", lambda: inv["accessible_reference_kg"]),
        _c("MARS_CO2_crustal_upper_bound", 3.89e18, "kg", "Jakosky 2019 (up to ~1 bar sequestered; NOT accessible)", "literature",
           lambda: inv["crustal_upper_bound_kg"]),
        _c("CO2_forcing_per_doubling", 6.0, "W m-2", "log-law coefficient, engineering default (no verified Mars-specific source)",
           "estimated", lambda: co2_mobilization.CO2Mobilization().forcing_per_doubling_wm2),
        _c("WATER_VAPOR_FEEDBACK_MAX", 1.0, "1", "upper bound of f_w in dT_eff = dT_dry (1+f_w): a doubling of the dry response; "
           "the published 1-bar case implies f_w ~ 1.0 (noarco.validation); optional parameter, default 0",
           "estimated", lambda: co2_mobilization.WATER_VAPOR_FEEDBACK_MAX),
        _c("MARS_CO2_WARMING_1BAR_MID", 60.0, "K", "Jakosky & Edwards 2018, Nature Astron. 2, 634 (~1 bar CO2 close to melting); "
           "+/-10 K range assumed by NOArCO; used only by radiative='literature_calibrated'",
           "literature", lambda: lit.MARS_CO2_WARMING_ANCHORS[2].dt_low_k / 2 + lit.MARS_CO2_WARMING_ANCHORS[2].dt_high_k / 2),
        _c("MARS_CO2_WARMING_20MBAR_MAX", 10.0, "K", "Jakosky & Edwards 2018 (climate models: < 10 K from ~20 mbar); an upper bound",
           "literature", lambda: lit.MARS_CO2_WARMING_ANCHORS[1].dt_high_k),
        _c("AEROSOL_kappa_MarsWRF_Al", 3.15e4, "m2 kg-1", "least-squares fit to Richardson et al. 2025 Table S3 (rms 18 %)", "calibrated",
           lambda: aerosols.MARSWRF_AL_ROD_KAPPA_M2_KG),
        _c("AEROSOL_kappa_MarsWRF_graphene", 7.08e4, "m2 kg-1", "fit to Richardson et al. 2025 Table S3 (rms 11 %)", "calibrated",
           lambda: aerosols.MARSWRF_GRAPHENE_KAPPA_M2_KG),
        _c("AEROSOL_kappa_Turyshev", 2.7e5, "m2 kg-1", "inferred from Turyshev 2026 Eq. 33+52+53 (+30 K); ~9x the GCM fit", "literature",
           lambda: aerosols.IMPLIED_NANOROD_KAPPA_M2_KG),
        _c("AEROSOL_residence_Al", 1.15, "yr", "column/release mass balance of Richardson et al. 2025 Table S3", "calibrated",
           lambda: aerosols.MARSWRF_AL_ROD_RESIDENCE_YEARS),
        _c("PFC_RE_C2F6", 0.25, "W m-2 ppb-1", "IPCC AR5 Table 8.A.1 (Earth, small perturbation)", "literature",
           lambda: ap["C2F6"]["alpha_wm2_per_ppb"]),
        _c("PFC_RE_CF4", 0.09, "W m-2 ppb-1", "IPCC AR5 Table 8.A.1", "literature", lambda: ap["CF4"]["alpha_wm2_per_ppb"]),
        _c("PFC_RE_SF6", 0.57, "W m-2 ppb-1", "IPCC AR5 Table 8.A.1", "literature", lambda: ap["SF6"]["alpha_wm2_per_ppb"]),
        _c("PFC_RE_C3F8", 0.28, "W m-2 ppb-1", "IPCC AR5 Table 8.A.1", "literature", lambda: ap["C3F8"]["alpha_wm2_per_ppb"]),
        _c("PFC_GWP100_SF6", 23500.0, "1", "IPCC AR5 Table 8.A.1", "literature", lambda: float(ap["SF6"]["gwp_100"])),
        _c("PFC_synthesis_energy", 30e6, "J kg-1", "engineering placeholder", "estimated",
           lambda: greenhouse_gas.SuperGreenhouseGas().synthesis_energy_j_per_kg),
        _c("AEROGEL_k_eff", 0.01, "W m-1 K-1", "Wordsworth et al. 2019 (silica aerogel at martian pressures)", "literature",
           lambda: __import__("noarco.mechanisms.aerogel", fromlist=["SilicaAerogel"]).SilicaAerogel().thermal_conductivity_w_m_k),
        _c("AEROGEL_manufacture_energy", 100e6, "J kg-1", "engineering placeholder", "estimated",
           lambda: __import__("noarco.mechanisms.aerogel", fromlist=["SilicaAerogel"]).SilicaAerogel().energy_per_kg_aerogel_j),
        _c("KOPPARAPU_moist_Seff_sun", 1.0140, "1", "Kopparapu et al. 2013 Table 3", "literature",
           lambda: KOPPARAPU_2013["moist_greenhouse"][0]),
        _c("KOPPARAPU_maxgh_Seff_sun", 0.3438, "1", "Kopparapu et al. 2013 Table 3", "literature",
           lambda: KOPPARAPU_2013["maximum_greenhouse"][0]),
        _c("CREW_O2", 0.84, "kg person-1 day-1", "NASA BVAD (NASA/TP-2015-218570)", "literature",
           lambda: habitat.O2_CONSUMPTION_KG_DAY_PER_PERSON),
        _c("CREW_CO2", 1.00, "kg person-1 day-1", "NASA BVAD", "literature", lambda: habitat.CO2_PRODUCTION_KG_DAY_PER_PERSON),
        _c("HABITAT_U_cold", 0.35, "W m-2 K-1", "engineering assumption (HabitatSpecification field)", "estimated",
           lambda: habitat.HabitatSpecification(1.0, 1).envelope_u_value_cold_w_m2_k),
        _c("HABITAT_leak_fraction", 0.005, "1 yr-1", "engineering assumption at reference A/V", "estimated",
           lambda: habitat.HabitatSpecification(1.0, 1).leak_fraction_per_year),
        _c("CREW_H2O", 2.5, "kg person-1 day-1", "NASA BVAD (drinking + food water), NASA/TP-2015-218570", "literature",
           lambda: habitat.H2O_CONSUMPTION_KG_DAY_PER_PERSON),
        _c("CREW_metabolic_heat", 120.0, "W person-1", "NASA BVAD sensible heat per crew member", "literature",
           lambda: habitat.METABOLIC_HEAT_WATTS_PER_PERSON),
        _c("HULL_allowable_stress", 1.38e8, "Pa", "Al 6061-T6 yield 276 MPa (MMPDS) / safety factor 2; engineering default", "estimated",
           lambda: habitat.HabitatSpecification(volume_m3=1.0, crew_size=0).hull_allowable_stress_pa),
        _c("HULL_density", 2700.0, "kg m-3", "aluminium alloy 6061 (handbook)", "measured",
           lambda: habitat.HabitatSpecification(volume_m3=1.0, crew_size=0).hull_density_kg_m3),
        _c("ISRU_dGf_Fe2O3", 742_200.0, "J mol-1", "NIST-JANAF, hematite, 298 K (magnitude)", "measured", lambda: isru._DG_F_FE2O3_J_MOL),
        _c("ISRU_M_Fe", 55.845e-3, "kg mol-1", "IUPAC standard atomic weight", "measured", lambda: isru._M_FE),
        _c("ISRU_water_recovery", 0.90, "1", "engineering assumption (heating recovery of regolith H2O)", "estimated",
           lambda: isru.ISRURequirement().water_recovery_fraction),
        _c("ISRU_excavation_energy", 1.5, "kWh t-1", "engineering assumption (excavation + grinding)", "estimated",
           lambda: isru.ISRURequirement().excavation_kwh_per_t),
        _c("SOLAR_array_efficiency", 0.25, "1", "engineering assumption (space-grade multijunction on the surface)", "estimated",
           lambda: isru.ISRURequirement().solar_array_efficiency),
        _c("SOLAR_array_derate", 0.80, "1", "engineering assumption (dust, pointing, degradation)", "estimated",
           lambda: isru.ISRURequirement().solar_derate),
        _c("AGRO_food_kcal_per_kg", 800.0, "kcal kg-1", "USDA, fresh potato ~770 kcal/kg; crop choice is an assumption", "estimated",
           lambda: agro.SoilConditioningSpec().food_kcal_per_kg),
        _c("CREW_kcal", 2500.0, "kcal person-1 day-1", "NASA-STD-3001 order of magnitude, active adult", "estimated",
           lambda: agro.SoilConditioningSpec().crew_kcal_per_person_day),
        _c("GM_SUN", 1.32712440018e11, "km3 s-2", "IAU 2015 nominal solar mass parameter", "exact", lambda: lg._MU_SUN_KM3_S2),
        _c("G0_standard_gravity", 9.80665, "m s-2", "CGPM 1901 (exact)", "exact", lambda: lg._G0),
        _c("LOGISTICS_v_infinity", 5.0, "km s-1", "assumed hyperbolic excess speed at the target (parameter)", "estimated",
           lambda: inspect.signature(lg.VolatileLogisticsPlanner.plan_import).parameters["v_infinity_km_s"].default),
        _c("LOGISTICS_isp", 3000.0, "s", "assumed electric/mass-driver specific impulse (parameter)", "estimated",
           lambda: inspect.signature(lg.VolatileLogisticsPlanner.plan_import).parameters["isp_s"].default),
        _c("LOGISTICS_deflections_per_century", 5.0, "1", "engineering assumption (parameter)", "estimated",
           lambda: float(__import__("inspect").signature(logistics.VolatileLogisticsPlanner.plan_import)
                         .parameters["deflections_per_century"].default)),
    ]


def verify_registry() -> list[tuple[str, float, float]]:
    """Return ``(name, registry_value, live_value)`` for every entry that disagrees (should be empty)."""
    bad = []
    for e in _entries():
        live = float(e.live())
        if abs(live - e.value) > 1e-9 * max(1.0, abs(e.value)) and abs(live - e.value) / abs(e.value) > 5e-4:
            bad.append((e.name, e.value, live))
    return bad


def registry_hash() -> str:
    payload = [(e.name, e.value, e.unit, e.source, e.klass) for e in _entries()]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def to_markdown() -> str:
    lines = ["| Constant | Value | Unit | Source | Class |", "|---|---|---|---|---|"]
    for e in _entries():
        lines.append(f"| {e.name} | {e.value:.6g} | {e.unit} | {e.source} | **{e.klass}** |")
    counts: dict[str, int] = {}
    for e in _entries():
        counts[e.klass] = counts.get(e.klass, 0) + 1
    lines.append("\n" + ", ".join(f"{n} {c}" for c, n in sorted(counts.items())) + ".")
    return "\n".join(lines)


def entries() -> list[ConstantEntry]:
    return _entries()


def as_dicts() -> list[dict[str, Any]]:
    return [{"name": e.name, "value": e.value, "unit": e.unit, "source": e.source, "class": e.klass} for e in _entries()]
