"""
noarco.core.endpoints
=====================
Planetary end states (E0-E4) used for requirement analysis and planning.

The definitions follow Turyshev (2026), arXiv:2603.00402, Sec. II.A and
Tables III-IV. They are *bookkeeping labels with measurable criteria*, not a
monotone ladder in pressure:

=====  =============================================  =============  ===========
label  meaning (Turyshev 2026, Sec. II.A)             scope          pressure
=====  =============================================  =============  ===========
E0     present Mars                                   global         ~0.61 kPa
E1     recurrent open-surface liquid water            global/region  >= 611.657 Pa at 273.16 K
E2     protected agriculture in enclosed volumes      regional       internal 10-30 kPa
E3     pressure-unassisted exposure (Armstrong limit) global         >= 6.27 kPa, T_s >~ 250 K
E4     breathable surface atmosphere                  global         p_O2 >~ 21 kPa + buffer gas
=====  =============================================  =============  ===========

E2 is an *enclosure and deployment-area* problem: its pressure is an interior
pressure and does not require a global atmosphere.

References
----------
Turyshev, S. G. (2026). arXiv:2603.00402, Sec. II.A, Tables III-IV.
Armstrong limit: body-temperature water boils below ~6.3 kPa (47 mmHg).
"""

from __future__ import annotations

from dataclasses import dataclass

from noarco.core.constants import ARMSTRONG_LIMIT_PA, WATER_TRIPLE_POINT_K, WATER_TRIPLE_POINT_PA


@dataclass(frozen=True)
class Endpoint:
    """A named end state.

    Parameters
    ----------
    label : str
        Short identifier ('E0' ... 'E4').
    name : str
        Human-readable name.
    min_pressure_pa : float
        Minimum pressure (Pa). Global surface pressure unless ``enclosed``.
    min_temperature_k : float
        Minimum mean surface temperature (K) (or enclosure temperature if ``enclosed``).
    min_o2_pa : float
        Minimum O2 partial pressure (Pa).
    max_co2_pa : float
        Maximum CO2 partial pressure (Pa).
    habitability : str
        Short description of what the state enables.
    reference : str
        Source of the definition.
    enclosed : bool
        True if the pressure applies inside enclosures only (regional problem).
    """

    label: str
    name: str
    min_pressure_pa: float
    min_temperature_k: float
    min_o2_pa: float
    max_co2_pa: float
    habitability: str
    reference: str
    enclosed: bool = False


#: End states for Mars (Turyshev 2026, Sec. II.A).
MARS_ENDPOINTS: dict[str, Endpoint] = {
    "E0": Endpoint(
        label="E0",
        name="Present Mars",
        min_pressure_pa=610.0,
        min_temperature_k=210.0,
        min_o2_pa=0.0,
        max_co2_pa=float("inf"),
        habitability="Robotic/human-assisted operations; pressure suits and thermal control.",
        reference="Turyshev (2026), Sec. II.A, Table II.",
    ),
    "E1": Endpoint(
        label="E1",
        name="Recurrent open-surface liquid water",
        min_pressure_pa=WATER_TRIPLE_POINT_PA,
        min_temperature_k=WATER_TRIPLE_POINT_K,
        min_o2_pa=0.0,
        max_co2_pa=float("inf"),
        habitability="Liquid water sustainable at the surface under a favourable local energy balance.",
        reference="Turyshev (2026), Sec. II.A Eq. (1); water triple point (IAPWS).",
    ),
    "E2": Endpoint(
        label="E2",
        name="Protected agriculture (enclosed)",
        min_pressure_pa=10_000.0,
        min_temperature_k=273.15,
        min_o2_pa=0.0,
        max_co2_pa=float("inf"),
        habitability="Enclosed/paraterraformed volumes at 10-30 kPa internal pressure (regional).",
        reference="Turyshev (2026), Sec. II.A (internal pressure 10-30 kPa).",
        enclosed=True,
    ),
    "E3": Endpoint(
        label="E3",
        name="Pressure-unassisted exposure (Armstrong limit)",
        min_pressure_pa=ARMSTRONG_LIMIT_PA,
        min_temperature_k=250.0,
        min_o2_pa=0.0,
        max_co2_pa=float("inf"),
        habitability="No ebullism; supplemental O2 and low pCO2 still required.",
        reference="Turyshev (2026), Sec. II.A and Table IV (P_s >= 6.27 kPa, T_s >~ 250 K).",
    ),
    "E4": Endpoint(
        label="E4",
        name="Breathable surface atmosphere",
        min_pressure_pa=101_325.0,
        min_temperature_k=273.15,
        min_o2_pa=16_000.0,
        max_co2_pa=500.0,
        habitability="Humans without equipment; requires O2 and buffer gas (Earth-like: p_O2 ~ 21 kPa).",
        reference=(
            "Turyshev (2026), Sec. II.A and Table IV; p_O2 >= 16 kPa physiological minimum; "
            "p_CO2 <= ~500 Pa (5000 ppm TLV-TWA at 1 atm)."
        ),
    ),
}
