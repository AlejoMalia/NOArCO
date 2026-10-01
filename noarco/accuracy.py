"""
noarco.accuracy — expected accuracy of the engine by scale and by world
=======================================================================

How well does a NOArCO habitat/enclosure calculation hold at 1 m, 6 m, 1 ha ... 200 ha, and on
different bodies? ``accuracy_table`` answers with numbers computed, not asserted: the uncertain
inputs of the habitat model are sampled from documented priors (scale-dependent where physics says so),
the model is evaluated vectorised, and two metrics are reported per output:

* **margin** - half-width of the central ``confidence`` interval relative to the nominal value;
* **hit rate** - ``P(|true/nominal - 1| <= tolerance)``: the share of plausible worlds in which the
  engine's value is within the tolerance (default +/-10 %). This is the "% de acierto".

Scale dependence (why small and large differ)
---------------------------------------------
* effective volume: construction tolerance ``3 * sigma_L / L`` (sigma_L = 1 cm) plus 2 % internal
  displacement, so a 1 m box is ~3 % uncertain in volume and a dome ~2 %;
* leakage goes through the envelope, so the loss fraction scales with surface/volume (small
  enclosures leak proportionally more) with a factor-2 prior on the coefficient;
* envelope area vs the modelled shape: 10 % (structure, seams);
* thermal: U-value factor 1.3 and a site-to-site spread of the time-mean exterior temperature
  (`default_exterior_spread`: 15 % on thin-atmosphere worlds, 0.3 % on Venus), which dominates on
  worlds with large thermal contrasts; diurnal swings average out and only affect *peak* power;
* crew rates +/-10 % (BVAD), electrolyser efficiency +/-0.08, base loads +/-20 %.

What it is not
--------------
A *parametric* accuracy under these priors. It is not validated against measured habitats and it
excludes model-form error (conduction-only thermal model, no solar load, no structural dynamics). Use
it to know which outputs to trust and how much, not as a guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from noarco.bodies import resolve_body
from noarco.core.constants import MOLAR_MASS, R_GAS, reversible_o2_energy_j_per_kg
from noarco.core.planet_state import PlanetState
from noarco.extratools.habitat import (
    CO2_PRODUCTION_KG_DAY_PER_PERSON,
    METABOLIC_HEAT_WATTS_PER_PERSON,
    O2_CONSUMPTION_KG_DAY_PER_PERSON,
    HabitatDimensioner,
    HabitatSpecification,
)

#: Input-quality presets: multiplicative factors (x = lognormal 1-sigma, else sigma) of the priors.
#: ``default`` uses engineering priors; ``measured`` assumes the user supplied measured U-values,
#: leak tests, site temperatures, crew logs and electrolyser data.
QUALITY = {
    "default": {"u": 1.3, "leak": 2.0, "crew": 0.10, "eff": 0.08, "load": 1.2, "cop": 1.2, "site": 1.0},
    "measured": {"u": 1.05, "leak": 1.2, "crew": 0.03, "eff": 0.02, "load": 1.05, "cop": 1.05, "site": 0.25},
}

OUTPUTS = ("gas_mass_kg", "o2_consumption_kg_day", "leak_kg_year", "continuous_power_kw")


@dataclass(frozen=True)
class Enclosure:
    """A box-shaped enclosure: ``footprint`` (m x m) and interior ``height`` (m); floor included in the area."""

    label: str
    side_m: float
    height_m: float
    people_per_m2: float = 0.0   # crew density on the floor area

    @property
    def volume_m3(self) -> float:
        return self.side_m**2 * self.height_m

    @property
    def area_m2(self) -> float:
        return 2.0 * self.side_m**2 + 4.0 * self.side_m * self.height_m

    @property
    def crew(self) -> int:
        return round(self.side_m**2 * self.people_per_m2)


HA = 1e4
STANDARD_ENCLOSURES: tuple[Enclosure, ...] = (
    Enclosure("1 m box (1 m3)", 1.0, 1.0),
    Enclosure("6 m module (216 m3)", 6.0, 6.0, 1 / 9.0),
    Enclosure("100 m2 pad x 3 m", 10.0, 3.0, 0.01),
    Enclosure("1 ha dome x 5 m", math.sqrt(1 * HA), 5.0, 0.01),
    Enclosure("10 ha dome x 10 m", math.sqrt(10 * HA), 10.0, 0.01),
    Enclosure("50 ha dome x 15 m", math.sqrt(50 * HA), 15.0, 0.01),
    Enclosure("200 ha dome x 20 m", math.sqrt(200 * HA), 20.0, 0.01),
)


@dataclass
class AccuracyRow:
    enclosure: Enclosure
    crew: int
    margin: dict[str, float]      # relative half-width of the central interval, per output
    hit_rate: dict[str, float]    # P(|true/nominal - 1| <= tolerance), per output
    nominal: dict[str, float]
    confidence: float
    tolerance: float

    @property
    def overall_hit_rate(self) -> float:
        """Hit rate of the weakest *defined* output (the chain is only as good as its worst link)."""
        vals = [v for k, v in self.hit_rate.items() if not math.isnan(v)]
        return min(vals) if vals else math.nan


def _ln(rng: np.random.Generator, n: int, factor: float) -> np.ndarray:
    return np.exp(rng.normal(0.0, math.log(factor), n))


def default_exterior_spread(planet: PlanetState) -> float:
    """Site-to-site spread of the *time-mean* exterior temperature about the global mean (fraction of T).

    Engineering default by atmosphere thickness (latitude/elevation contrasts are large where the
    atmosphere cannot redistribute heat): < 1 kPa 15 %, 1-10 kPa 10 %, 10 kPa-1 MPa 4 %, > 1 MPa 0.3 %.
    Diurnal swings average out of a linear conduction model and only affect peak power.
    """
    p = planet.surface_pressure_pa
    return 0.15 if p < 1e3 else 0.10 if p < 1e4 else 0.04 if p < 1e6 else 0.003


def accuracy_for_enclosure(
    body: PlanetState | str | dict[str, Any] | Any,
    enclosure: Enclosure,
    *,
    interior_pressure_pa: float = 101_325.0,
    interior_temperature_k: float = 293.15,
    exterior_spread_fraction: float | None = None,
    confidence: float = 0.95,
    tolerance: float = 0.10,
    n: int = 20_000,
    seed: int = 20260930,
    quality: str = "default",
) -> AccuracyRow:
    """Accuracy of the habitat calculation for one enclosure on one world (see module docstring)."""
    if not 0.5 <= confidence < 1.0 or tolerance <= 0 or n < 100:
        raise ValueError("confidence in [0.5, 1), tolerance > 0 and n >= 100 required")
    if quality not in QUALITY:
        raise ValueError(f"quality must be one of {sorted(QUALITY)}")
    q = QUALITY[quality]
    planet = resolve_body(body).planet
    if exterior_spread_fraction is None:
        exterior_spread_fraction = default_exterior_spread(planet) * q["site"]
    spec = HabitatSpecification(
        volume_m3=enclosure.volume_m3, crew_size=enclosure.crew, interior_pressure_pa=interior_pressure_pa,
        interior_temperature_k=interior_temperature_k, envelope_area_m2=enclosure.area_m2,
    )
    rep = HabitatDimensioner.dimension(planet, spec)
    nominal = {
        "gas_mass_kg": rep.total_gas_mass_kg, "o2_consumption_kg_day": rep.daily_o2_consumption_kg,
        "leak_kg_year": rep.leak_rate_annual_kg, "continuous_power_kw": rep.continuous_power_kw,
    }

    rng = np.random.default_rng(seed)
    v, a0 = enclosure.volume_m3, enclosure.area_m2
    length = v ** (1.0 / 3.0)
    sigma_vol = math.sqrt(0.02**2 + (3.0 * 0.01 / length) ** 2)
    vol = v * (1.0 + rng.normal(0.0, sigma_vol, n))
    area = a0 * _ln(rng, n, 1.10)
    p_int = interior_pressure_pa * (1.0 + rng.normal(0.0, 0.005, n))
    t_int = interior_temperature_k + rng.normal(0.0, 1.0, n)
    mu = 0.21 * MOLAR_MASS["O2"] + 0.78 * MOLAR_MASS["N2"] + 0.01 * MOLAR_MASS["Ar"]   # kg/mol, as the habitat model
    gas = p_int * vol * mu / (R_GAS * t_int)

    crew = enclosure.crew
    o2_rate = crew * O2_CONSUMPTION_KG_DAY_PER_PERSON * (1.0 + rng.normal(0.0, q["crew"], n))
    co2_rate = crew * CO2_PRODUCTION_KG_DAY_PER_PERSON * (1.0 + rng.normal(0.0, q["crew"], n))

    ref_av = 3.0 / ((3.0 * 2500.0 / (4.0 * math.pi)) ** (1.0 / 3.0))
    leak_frac = spec.leak_fraction_per_year * _ln(rng, n, q["leak"]) * (area / vol) / ref_av
    leak = gas * leak_frac

    eff = np.clip(0.70 + rng.normal(0.0, q["eff"], n), 0.3, 0.95)
    eclss = o2_rate * (reversible_o2_energy_j_per_kg() / eff) / 86400.0 / 1000.0
    scrubber = crew * 200.0 * _ln(rng, n, q["load"]) / 1000.0
    base = (crew * 500.0 + v * 1.0) * _ln(rng, n, q["load"]) / 1000.0
    t_ext = planet.mean_temperature_k + rng.normal(0.0, exterior_spread_fraction * planet.mean_temperature_k, n)
    t_ext = np.maximum(t_ext, 1.0)
    dt = t_int - t_ext
    crew_heat = crew * METABOLIC_HEAT_WATTS_PER_PERSON / 1000.0
    u = _ln(rng, n, q["u"])
    cold = np.maximum(area * spec.envelope_u_value_cold_w_m2_k * u * dt / 1000.0 - crew_heat, 0.0)
    hot = (area * spec.envelope_u_value_hot_w_m2_k * u * np.abs(dt) / 1000.0 + crew_heat) / (spec.cooling_cop * _ln(rng, n, q["cop"]))
    hvac = np.where(dt > 0, cold, hot)
    power = eclss + scrubber + base + hvac
    _ = co2_rate

    samples = {"gas_mass_kg": gas, "o2_consumption_kg_day": o2_rate, "leak_kg_year": leak,
               "continuous_power_kw": power}
    alpha = 1.0 - confidence
    margin: dict[str, float] = {}
    hit: dict[str, float] = {}
    for k, x in samples.items():
        nom = nominal[k]
        if nom <= 0:
            margin[k], hit[k] = math.nan, math.nan        # output not applicable (e.g. no crew)
            continue
        lo, hi = np.quantile(x, [alpha / 2.0, 1.0 - alpha / 2.0])
        margin[k] = float((hi - lo) / 2.0 / nom)
        hit[k] = float(np.mean(np.abs(x / nom - 1.0) <= tolerance))
    return AccuracyRow(enclosure, crew, margin, hit, nominal, confidence, tolerance)


def accuracy_table(
    body: PlanetState | str | dict[str, Any] | Any,
    enclosures: tuple[Enclosure, ...] = STANDARD_ENCLOSURES,
    **kwargs: Any,
) -> list[AccuracyRow]:
    """Accuracy rows for a list of enclosures on one world."""
    return [accuracy_for_enclosure(body, e, **kwargs) for e in enclosures]


def to_markdown(rows: list[AccuracyRow], title: str = "") -> str:
    """Render accuracy rows as a Markdown table (hit rate of each output and of the weakest one)."""
    if not rows:
        return ""
    tol, conf = rows[0].tolerance, rows[0].confidence
    head = (f"| Enclosure | Crew | Gas mass | O2 use | Leakage | Power | **Weakest** | "
            f"Power margin ({conf:.0%}) |\n|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in rows:
        hr = {k: ("n/a" if math.isnan(v) else f"{v:.0%}") for k, v in r.hit_rate.items()}
        m = r.margin["continuous_power_kw"]
        lines.append(f"| {r.enclosure.label} | {r.crew} | {hr['gas_mass_kg']} | {hr['o2_consumption_kg_day']} | "
                     f"{hr['leak_kg_year']} | {hr['continuous_power_kw']} | **{r.overall_hit_rate:.0%}** | "
                     f"{'n/a' if math.isnan(m) else f'±{m:.0%}'} |")
    cap = f"{title}\n\n" if title else ""
    return cap + f"Hit rate = P(|true/nominal − 1| ≤ {tol:.0%}) under the documented priors.\n\n" + head + "\n".join(lines)
