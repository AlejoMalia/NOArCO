"""
noarco.assurance.uncertainty — First-order propagation with covariance + MC cross-check
=======================================================================================

Adapted from SFSA ``UQE``. Differences: full input covariance (correlated
inputs), Jacobian by central differences with scale-aware steps, exact normal
coverage factor from the requested confidence, and a seeded Monte-Carlo
cross-check that flags when the linear (GUM) approximation is not adequate —
the practice expected by NASA-STD-7009 style model credibility assessments.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class UncertainValue:
    nominal: float
    sigma: float  # 1-sigma standard uncertainty (same unit as nominal)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.nominal) and math.isfinite(self.sigma)) or self.sigma < 0:
            raise ValueError(f"Invalid UncertainValue({self.nominal}, {self.sigma})")

    @property
    def relative(self) -> float:
        return self.sigma / abs(self.nominal) if self.nominal else math.inf

    def interval(self, confidence: float = 0.95) -> tuple[float, float]:
        k = coverage_factor(confidence)
        return self.nominal - k * self.sigma, self.nominal + k * self.sigma


def coverage_factor(confidence: float) -> float:
    """Two-sided normal coverage factor k (0.95 -> 1.96, 0.68 -> ~1.0)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return float(stats.norm.ppf(0.5 + confidence / 2.0))


@dataclass
class PropagationResult:
    output: UncertainValue
    contributions: dict[str, float]  # fraction of output variance per input
    sensitivities: dict[str, float]  # d(out)/d(in)
    linearity_ok: bool | None = None  # set when a MC cross-check was run
    mc_mean: float | None = None
    mc_sigma: float | None = None
    notes: list[str] = field(default_factory=list)


def propagate(
    fn: Callable[[dict[str, float]], float],
    inputs: Mapping[str, UncertainValue],
    correlation: Mapping[tuple[str, str], float] | None = None,
    *,
    mc_samples: int = 0,
    seed: int = 0,
    linearity_tol: float = 0.10,
) -> PropagationResult:
    """Propagate input uncertainties through ``fn``.

    ``correlation[(a, b)]`` is the Pearson coefficient between inputs a and b
    (symmetric). With ``mc_samples > 0`` a seeded Monte-Carlo run (normal
    inputs, same covariance) is compared to the first-order result; if the
    sigmas differ by more than ``linearity_tol`` the result is flagged
    non-linear and the MC sigma should be preferred.
    """
    names = list(inputs)
    nominal = {k: v.nominal for k, v in inputs.items()}
    y0 = float(fn(dict(nominal)))
    if not math.isfinite(y0):
        raise ValueError("Model output is not finite at the nominal point")

    n = len(names)
    cov = np.diag([inputs[k].sigma ** 2 for k in names])
    for (a, b), r in (correlation or {}).items():
        if a not in inputs or b not in inputs:
            raise KeyError(f"Correlation refers to unknown input: {(a, b)}")
        if not -1.0 <= r <= 1.0:
            raise ValueError(f"Correlation {r} out of [-1, 1]")
        i, j = names.index(a), names.index(b)
        cov[i, j] = cov[j, i] = r * inputs[a].sigma * inputs[b].sigma
    if n and np.min(np.linalg.eigvalsh(cov)) < -1e-12 * max(1.0, float(np.max(np.abs(cov)))):
        raise ValueError("Covariance matrix is not positive semi-definite")

    jac = np.zeros(n)
    for i, k in enumerate(names):
        h = max(abs(nominal[k]), inputs[k].sigma, 1e-12) * 1e-6
        hi, lo = dict(nominal), dict(nominal)
        hi[k] += h
        lo[k] -= h
        jac[i] = (float(fn(hi)) - float(fn(lo))) / (2.0 * h)

    var = float(jac @ cov @ jac)
    sigma = math.sqrt(max(var, 0.0))
    # Variance shares from the diagonal terms (covariance terms can be negative).
    diag_terms = {k: (jac[i] * inputs[k].sigma) ** 2 for i, k in enumerate(names)}
    tot = sum(diag_terms.values())
    shares = {k: (v / tot if tot > 0 else 0.0) for k, v in diag_terms.items()}

    res = PropagationResult(
        output=UncertainValue(y0, sigma),
        contributions=dict(sorted(shares.items(), key=lambda kv: -kv[1])),
        sensitivities={k: float(jac[i]) for i, k in enumerate(names)},
    )
    if mc_samples > 0 and n:
        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal([nominal[k] for k in names], cov, size=mc_samples)
        ys = []
        for row in draws:
            try:
                y = float(fn(dict(zip(names, row.tolist(), strict=True))))
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            if math.isfinite(y):
                ys.append(y)
        if len(ys) < max(10, mc_samples // 2):
            res.linearity_ok = False
            res.notes.append("MC cross-check: model failed/non-finite on most draws")
        else:
            arr = np.asarray(ys)
            res.mc_mean, res.mc_sigma = float(arr.mean()), float(arr.std(ddof=1))
            denom = max(res.mc_sigma, sigma, 1e-300)
            res.linearity_ok = abs(res.mc_sigma - sigma) / denom <= linearity_tol
            if not res.linearity_ok:
                res.notes.append(
                    f"Linear sigma {sigma:.4g} differs from MC sigma {res.mc_sigma:.4g}; "
                    "prefer the Monte-Carlo estimate."
                )
    return res


def combine_independent(values: Sequence[UncertainValue]) -> UncertainValue:
    """Sum of independent uncertain values."""
    return UncertainValue(sum(v.nominal for v in values),
                          math.sqrt(sum(v.sigma ** 2 for v in values)))
