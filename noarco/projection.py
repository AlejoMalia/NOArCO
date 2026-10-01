"""
noarco.projection — fast probabilistic projection with confidence intervals
===========================================================================

``project(planet, target)`` answers "how likely is this transition, and with what margin?" in
milliseconds. It draws correlated-free samples of the *uncertain inputs* from documented priors,
evaluates the closed-form requirement chain (vectorised, no loops over samples) and returns, for
each output, a central interval at the requested confidence together with
``P(feasible) = P(Pi_min >= 1)``.

Outputs (per sample)
--------------------
* ``pi_mass`` / ``pi_power`` / ``pi_min``  - rubric numbers of Turyshev (2026) Eq. 9-11;
* ``atmosphere_mass_kg``, ``o2_energy_j``, ``build_time_at_power_years``;
* ``co2_warming_k`` - greenhouse warming from the accessible CO2 (log-law);
* ``aerosol_mass_kg`` and ``aerosol_replenishment_kg_s`` - what IR-active aerosols would need to
  supply the remaining warming (maintenance-dominated, Eq. 53).

What the interval means - and does not
--------------------------------------
The interval propagates **parametric** uncertainty only (the priors below). It does not include
model-form error (grey atmosphere, log-law CO2 forcing, no feedbacks). A narrow interval therefore
means "insensitive to the listed inputs", not "certain". Priors are conservative defaults with their
basis stated; override them with ``priors=`` when better data exist. Results are reproducible for a
given ``seed`` and cached by input hash.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from noarco.assurance.audit import DEFAULT_REL_SIGMA
from noarco.bodies import resolve_body
from noarco.core.constants import SIGMA_SB, YEAR_S, reversible_o2_energy_j_per_kg
from noarco.core.desired_state import DesiredState
from noarco.core.planet_state import PlanetState
from noarco.engines.pathfinder import ELECTROLYSIS_EFFICIENCY, PathEngine
from noarco.extratools.exoplanet import ExoplanetProfile
from noarco.mechanisms.aerosols import MARSWRF_AL_ROD_KAPPA_M2_KG, MARSWRF_AL_ROD_RESIDENCE_YEARS


@dataclass(frozen=True)
class Prior:
    """Distribution of an uncertain input. ``kind``: 'lognormal' (``spread`` = multiplicative 1-sigma
    factor, e.g. 2.0), 'normal' (``spread`` = absolute sigma) or 'uniform' (``spread`` = half-width)."""

    nominal: float
    spread: float
    kind: str = "lognormal"
    basis: str = ""
    lower: float | None = None
    upper: float | None = None

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.kind == "lognormal":
            if self.spread < 1.0 or self.nominal <= 0:
                raise ValueError("lognormal needs nominal > 0 and spread >= 1")
            x = self.nominal * np.exp(rng.normal(0.0, math.log(self.spread), n))
        elif self.kind == "normal":
            x = rng.normal(self.nominal, self.spread, n)
        elif self.kind == "uniform":
            x = rng.uniform(self.nominal - self.spread, self.nominal + self.spread, n)
        else:
            raise ValueError(f"unknown prior kind {self.kind!r}")
        if self.lower is not None or self.upper is not None:
            x = np.clip(x, self.lower, self.upper)
        return np.asarray(x, dtype=float)


@dataclass(frozen=True)
class Interval:
    lower: float
    median: float
    upper: float
    confidence: float

    @property
    def relative_margin(self) -> float:
        """Half-width of the interval relative to the median."""
        return (self.upper - self.lower) / 2.0 / abs(self.median) if self.median else math.inf

    def __str__(self) -> str:
        return f"{self.median:.4g} [{self.lower:.4g}, {self.upper:.4g}] @ {self.confidence:.0%}"


@dataclass(frozen=True)
class StructuralError:
    """Model-form uncertainty of one model class: a multiplicative lognormal 1-sigma factor."""

    factor: float
    applies_to: str
    basis: str


#: Structural (model-form) error per model class. Exact closed forms (hydrostatic mass, Gibbs work,
#: Stefan-Boltzmann) carry none; the grey-atmosphere and CO2 log-law mappings do.
MODEL_STRUCTURAL_ERROR: dict[str, StructuralError] = {
    "grey_atmosphere_tau": StructuralError(
        1.4, "required added IR optical depth", "Turyshev 2026 Sec. IV.B: +/-30-50 % on tau_IR for real spectra"),
    "co2_log_forcing": StructuralError(
        1.5, "CO2 greenhouse forcing", "log law in a thin, band-saturated atmosphere; engineering factor"),
}
NOT_COVERED = [
    "Feedbacks (water vapour, clouds, albedo), CO2 condensation/collapse and latent-heat reservoirs.",
    "Aerosol microphysics, lofting and dispersal; spectral (non-grey) structure beyond the tau_IR factor.",
    "Escape and geochemical sinks; latitudinal/diurnal structure; correlations between inputs.",
    "Errors in the reference scalings themselves (Turyshev 2026) and in the published reservoir estimates.",
]


@dataclass
class ModelErrorReport:
    """Projection including model-form error, next to the parametric-only numbers."""

    probability_feasible: float
    outputs: dict[str, Interval]
    margin_parametric: dict[str, float]
    margin_model_structural: dict[str, float]    # combined parametric + structural
    factors: dict[str, float]
    not_covered: list[str]


@dataclass
class Projection:
    body: str
    confidence: float
    n_samples: int
    seed: int
    probability_feasible: float
    probability_feasible_ci: tuple[float, float]   # Wilson interval of the Monte-Carlo estimate
    outputs: dict[str, Interval]
    binding_constraint_shares: dict[str, float]
    priors: dict[str, Prior]
    notes: list[str] = field(default_factory=list)
    input_hash: str = ""
    model_error: ModelErrorReport | None = None

    @property
    def pi_min(self) -> Interval:
        return self.outputs["pi_min"]

    def statement(self) -> str:
        p, c = self.probability_feasible, self.confidence
        if p <= 1e-3:
            word = f"INFEASIBLE (P(feasible) < {p + 1e-3:.3f})"
        elif p >= 1 - 1e-3:
            word = f"FEASIBLE (P(infeasible) < {1 - p + 1e-3:.3f})"
        elif p <= 1 - c:
            word = f"LIKELY INFEASIBLE (P(infeasible) = {1 - p:.1%})"
        elif p >= c:
            word = f"LIKELY FEASIBLE (P(feasible) = {p:.1%})"
        else:
            word = f"UNCERTAIN (P(feasible) = {p:.1%})"
        top = max(self.binding_constraint_shares, key=lambda k: self.binding_constraint_shares[k]) \
            if self.binding_constraint_shares else "none"
        return f"{self.body}: {word}; Pi_min {self.pi_min}; binding constraint: {top}."


def default_priors(planet: PlanetState) -> dict[str, Prior]:
    """Documented default priors for the uncertain inputs of the projection."""
    rel = DEFAULT_REL_SIGMA[planet.source_type]
    inv = PathEngine._accessible_co2_kg(planet)
    pri = {
        "solar_constant_wm2": Prior(planet.solar_constant_wm2, rel, "normal", f"source type {planet.source_type.value}"),
        "surface_albedo": Prior(planet.surface_albedo, rel * planet.surface_albedo, "normal", "source type",
                                lower=0.0, upper=0.95),
        "gravity_ms2": Prior(planet.gravity_ms2, rel * planet.gravity_ms2, "normal", "source type", lower=1e-3),
        "radius_m": Prior(planet.radius_m, rel * planet.radius_m, "normal", "source type", lower=1.0),
        "electrolysis_efficiency": Prior(ELECTROLYSIS_EFFICIENCY, 0.08, "normal",
                                         "engineering spread around 0.70", lower=0.3, upper=0.95),
        "co2_forcing_per_doubling_wm2": Prior(6.0, 1.5, "normal", "log-law coefficient; Mars-specific value uncertain",
                                              lower=1.0, upper=12.0),
        "aerosol_kappa_m2_kg": Prior(MARSWRF_AL_ROD_KAPPA_M2_KG, 1.5, "lognormal",
                                     "fit to MarsWRF Table S3 (Richardson 2025), rms 18 % + extrapolation"),
        "aerosol_residence_years": Prior(MARSWRF_AL_ROD_RESIDENCE_YEARS, 1.3, "lognormal",
                                         "mass balance of MarsWRF Table S3 (1.07-1.25 yr)"),
    }
    if inv > 0:
        # Jakosky & Edwards (2018): accessible CO2 of order 0.01-0.05 bar -> factor ~1.6-2 (1-sigma).
        pri["accessible_inventory_kg"] = Prior(inv, 1.8, "lognormal", "Jakosky & Edwards 2018; Turyshev 2026 Sec. V")
    return pri


_CACHE: dict[str, Projection] = {}


def _tau_eddington(ts: np.ndarray, te: np.ndarray) -> np.ndarray:
    return np.asarray(4.0 / 3.0 * (ts / te) ** 4 - 2.0 / 3.0, dtype=float)


def project(
    planet: PlanetState | str | dict[str, Any] | ExoplanetProfile,
    target: DesiredState,
    *,
    build_time_years: float = 1000.0,
    available_power_w: float = 1.0e10,
    available_inventory_kg: float | None = None,
    confidence: float = 0.95,
    n: int = 20_000,
    seed: int = 20260930,
    priors: dict[str, Prior] | None = None,
    use_cache: bool = True,
    available_delta_tau_ir: float | None = None,
    thermal_requirement_scale: float = 1.0,
) -> Projection:
    """Probabilistic projection of ``target`` from ``planet`` (see module docstring).

    ``planet`` may be any body description accepted by :func:`noarco.bodies.resolve_body`.
    """
    planet = resolve_body(planet).planet
    if not 0.5 <= confidence < 1.0:
        raise ValueError("confidence must be in [0.5, 1)")
    if n < 100 or build_time_years <= 0 or available_power_w <= 0:
        raise ValueError("n >= 100, build_time_years > 0 and available_power_w > 0 required")
    pri = default_priors(planet)
    if available_inventory_kg is not None:
        pri["accessible_inventory_kg"] = Prior(available_inventory_kg, 1.8, "lognormal", "user supplied")
    pri.update(priors or {})

    key = hashlib.sha256(json.dumps(
        {"p": planet.model_dump(mode="json"), "t": target.model_dump(mode="json"), "b": build_time_years,
         "w": available_power_w, "c": confidence, "n": n, "s": seed, "dtau": available_delta_tau_ir,
         "trs": thermal_requirement_scale,
         "pri": {k: (v.nominal, v.spread, v.kind, v.lower, v.upper) for k, v in sorted(pri.items())}},
        sort_keys=True, default=str).encode()).hexdigest()
    if use_cache and key in _CACHE:
        return _CACHE[key]

    rng = np.random.default_rng(seed)
    x = {k: v.sample(rng, n) for k, v in pri.items()}
    # Structural (model-form) factors are drawn from an independent stream so that the parametric
    # samples are identical with and without them.
    rng_s = np.random.default_rng(seed + 1)
    f_tau = np.exp(rng_s.normal(0.0, math.log(MODEL_STRUCTURAL_ERROR["grey_atmosphere_tau"].factor), n))
    f_co2 = np.exp(rng_s.normal(0.0, math.log(MODEL_STRUCTURAL_ERROR["co2_log_forcing"].factor), n))
    ones = np.ones(n)

    g, r = x["gravity_ms2"], x["radius_m"]
    k_pa = 4.0 * np.pi * r**2 / g                                       # kg per Pa (Eq. 2)
    area = 4.0 * np.pi * r**2

    d_p = 0.0 if (target.enclosed or target.min_surface_pressure_pa is None) else \
        max(target.min_surface_pressure_pa - planet.surface_pressure_pa, 0.0)
    m_req = k_pa * d_p
    o2_req_pa = 0.0
    if target.min_o2_partial_pressure_pa and not target.enclosed:
        o2_req_pa = max(target.min_o2_partial_pressure_pa
                        - planet.gas_composition.mole_fraction("O2") * planet.surface_pressure_pa, 0.0)
    m_o2 = k_pa * o2_req_pa
    e_o2 = m_o2 * reversible_o2_energy_j_per_kg() / x["electrolysis_efficiency"]
    mean_power = e_o2 / (build_time_years * YEAR_S)
    has_thermal = target.target_mean_temperature_k is not None and not target.enclosed
    ts0 = planet.mean_temperature_k
    te = (x["solar_constant_wm2"] * (1.0 - x["surface_albedo"]) / (4.0 * SIGMA_SB)) ** 0.25

    def run(ftau: np.ndarray, fco2: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        big = np.inf
        pi_mass = x["accessible_inventory_kg"] / m_req if (m_req.max() > 0 and "accessible_inventory_kg" in x) else \
            np.full(n, big if m_req.max() <= 0 else 0.0)
        pi_power = available_power_w / mean_power if mean_power.max() > 0 else np.full(n, big)
        outs: dict[str, np.ndarray] = {
            "pi_mass": pi_mass, "pi_power": pi_power, "atmosphere_mass_kg": m_req, "o2_energy_j": e_o2,
            "build_time_at_power_years": e_o2 / available_power_w / YEAR_S,
        }
        pi_min = np.minimum(pi_mass, pi_power)
        if has_thermal:
            assert target.target_mean_temperature_k is not None
            dt_co2 = np.zeros(n)
            if "accessible_inventory_kg" in x:
                released = np.minimum(x["accessible_inventory_kg"], m_req if m_req.max() > 0 else x["accessible_inventory_kg"])
                p_co2_0 = planet.gas_composition.mole_fraction("CO2") * planet.surface_pressure_pa
                if p_co2_0 > 0:
                    f = x["co2_forcing_per_doubling_wm2"] * fco2 * np.log2((p_co2_0 + released / k_pa) / p_co2_0)
                    xx = np.clip(f / (SIGMA_SB * te**4), 0.0, 0.95)
                    dt_co2 = ts0 * ((1.0 - xx) ** -0.25 - 1.0)
                outs["co2_warming_k"] = dt_co2
            remaining = np.maximum(target.target_mean_temperature_k - ts0 - dt_co2, 0.0)
            d_tau = np.maximum(_tau_eddington(ts0 + dt_co2 + remaining, te)
                               - _tau_eddington(ts0 + dt_co2, te), 0.0) * ftau * thermal_requirement_scale
            mass = d_tau / x["aerosol_kappa_m2_kg"] * area
            outs["aerosol_required_delta_tau"] = d_tau
            outs["aerosol_mass_kg"] = mass
            outs["aerosol_replenishment_kg_s"] = mass / (x["aerosol_residence_years"] * YEAR_S)
            if available_delta_tau_ir is not None:
                pi_f = np.where(d_tau > 0, available_delta_tau_ir / np.maximum(d_tau, 1e-300), big)
                outs["pi_forcing"] = pi_f
                pi_min = np.minimum(pi_min, pi_f)
        outs["pi_min"] = pi_min
        return outs, pi_min >= 1.0

    outputs, feasible = run(ones, ones)
    outputs_ms, feasible_ms = run(f_tau, f_co2)
    pi_mass, pi_power = outputs["pi_mass"], outputs["pi_power"]

    alpha = 1.0 - confidence
    intervals: dict[str, Interval] = {}
    for name, arr in outputs.items():
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            intervals[name] = Interval(math.inf, math.inf, math.inf, confidence)
            continue
        lo, med, hi = np.quantile(finite, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0])
        intervals[name] = Interval(float(lo), float(med), float(hi), confidence)

    def _intervals(outs: dict[str, np.ndarray]) -> dict[str, Interval]:
        res: dict[str, Interval] = {}
        for name, arr in outs.items():
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                res[name] = Interval(math.inf, math.inf, math.inf, confidence)
                continue
            lo, med, hi = np.quantile(finite, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0])
            res[name] = Interval(float(lo), float(med), float(hi), confidence)
        return res

    intervals_ms = _intervals(outputs_ms)
    model_error = ModelErrorReport(
        probability_feasible=float(feasible_ms.mean()), outputs=intervals_ms,
        margin_parametric={k: v.relative_margin for k, v in intervals.items()},
        margin_model_structural={k: v.relative_margin for k, v in intervals_ms.items()},
        factors={k: v.factor for k, v in MODEL_STRUCTURAL_ERROR.items()},
        not_covered=list(NOT_COVERED),
    )
    p = float(feasible.mean())
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    shares = {"mass": float(np.mean(pi_mass <= pi_power)), "power": float(np.mean(pi_power < pi_mass))}

    proj = Projection(
        body=planet.body_name, confidence=confidence, n_samples=n, seed=seed, probability_feasible=p,
        probability_feasible_ci=(max(centre - half, 0.0), min(centre + half, 1.0)), outputs=intervals,
        binding_constraint_shares=shares, priors=pri,
        notes=[
            (
                "Intervals propagate parametric uncertainty only; model-form error (grey atmosphere, "
                "log-law CO2 forcing, no feedbacks) is not included."
            ),
            "Constants G, sigma and the Gibbs energy are taken as exact.",
        ],
        input_hash=key, model_error=model_error,
    )
    if use_cache:
        _CACHE[key] = proj
    return proj


def clear_projection_cache() -> int:
    n = len(_CACHE)
    _CACHE.clear()
    return n


@dataclass(frozen=True)
class SensitivityRow:
    parameter: str
    pi_min_low: float          # median Pi_min with the parameter one sigma "worse" direction (lower Pi_min side)
    pi_min_high: float
    span_log10: float          # |log10(high/low)|: how strongly Pi_min depends on this input
    flips_verdict: bool        # the +/-1 sigma range straddles Pi_min = 1


def _frozen(pr: Prior) -> Prior:
    if pr.kind == "lognormal":
        return Prior(pr.nominal, 1.0 + 1e-12, "lognormal", pr.basis, pr.lower, pr.upper)
    return Prior(pr.nominal, abs(pr.nominal) * 1e-12 + 1e-300, "normal", pr.basis, pr.lower, pr.upper)


def sensitivity(
    planet: PlanetState | str | dict[str, Any] | ExoplanetProfile,
    target: DesiredState,
    *,
    n: int = 2000,
    seed: int = 20260930,
    **kwargs: Any,
) -> list[SensitivityRow]:
    """One-at-a-time +/-1-sigma sensitivity of the median ``Pi_min`` (all other inputs at nominal).

    Tells which uncertain input actually controls the verdict and whether a plausible change of that
    input alone could flip it. ``kwargs`` are forwarded to :func:`project`.
    """
    body = resolve_body(planet).planet
    base = default_priors(body)
    if kwargs.get("available_inventory_kg") is not None:
        base["accessible_inventory_kg"] = Prior(kwargs["available_inventory_kg"], 1.8, "lognormal", "user supplied")
    base.update(kwargs.pop("priors", None) or {})
    frozen = {k: _frozen(v) for k, v in base.items()}
    rows: list[SensitivityRow] = []
    for name, pr in base.items():
        vals = []
        for sign in (-1.0, 1.0):
            if pr.kind == "lognormal":
                nominal = pr.nominal * pr.spread**sign
            else:
                nominal = pr.nominal + sign * pr.spread
            pri = {**frozen, name: _frozen(Prior(nominal, pr.spread, pr.kind, pr.basis, pr.lower, pr.upper))}
            pj = project(body, target, n=n, seed=seed, priors=pri, use_cache=False, **kwargs)
            vals.append(pj.pi_min.median)
        lo, hi = sorted(vals)
        span = abs(math.log10(hi / lo)) if lo > 0 and math.isfinite(hi) and math.isfinite(lo) else 0.0
        rows.append(SensitivityRow(name, lo, hi, span, lo < 1.0 <= hi))
    return sorted(rows, key=lambda r: -r.span_log10)


def forcing_robustness(
    planet: PlanetState | str | dict[str, Any] | ExoplanetProfile,
    target: DesiredState,
    available_delta_tau_ir: float,
    percents: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0),
    **kwargs: Any,
) -> dict[str, Any]:
    """"If the required forcing/optical depth is off by +/-X %, does Pi_min still pass?"

    Scales the thermal requirement by ``1 +/- X/100`` and reports ``P(feasible)`` and the median
    ``Pi_min``; ``break_even_pct`` is the error in the requirement at which the median Pi_min hits 1.
    """
    body = resolve_body(planet).planet
    rows = []
    for pct in percents:
        for sign in (+1, -1):
            scale = max(1.0 + sign * pct / 100.0, 1e-6)
            pj = project(body, target, available_delta_tau_ir=available_delta_tau_ir,
                         thermal_requirement_scale=scale, use_cache=False, **kwargs)
            rows.append({"requirement_error_pct": sign * pct, "p_feasible": pj.probability_feasible,
                         "pi_min_median": pj.pi_min.median})
    base = project(body, target, available_delta_tau_ir=available_delta_tau_ir, use_cache=False, **kwargs)
    pf = base.outputs.get("pi_forcing")
    break_even = None if pf is None or not math.isfinite(pf.median) else (pf.median - 1.0) * 100.0
    return {"baseline_pi_min": base.pi_min.median, "rows": rows, "break_even_pct": break_even}
