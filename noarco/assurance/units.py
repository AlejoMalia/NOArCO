"""
noarco.assurance.units — Strict dimensional analysis
====================================================

Compound-unit parser over the seven SI base dimensions. Adapted from the
SFSA ``UnitDimensionalEngine`` with the flight-software-grade differences:

* Unknown units **raise** ``UnitError`` (SFSA silently returns "dimensionless",
  which is how Pa-vs-bar type mission losses happen).
* Compound expressions: ``"W m^-2 K^-4"``, ``"kg/m^2/s"``, ``"mbar"``.
* Rational exponents (``Fraction``) so ``sqrt`` of a dimension is exact.
* Affine scales (degC, degF) are handled only by ``convert_temperature``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

_BASE = ("M", "L", "T", "K", "N", "I", "J")  # kg, m, s, K, mol, A, cd


class UnitError(ValueError):
    """Unknown unit, malformed expression, or dimensional mismatch."""


@dataclass(frozen=True)
class Dimension:
    """Exponent vector over (kg, m, s, K, mol, A, cd)."""

    exps: tuple[Fraction, ...] = (Fraction(0),) * 7

    @staticmethod
    def of(**kw: int | Fraction) -> Dimension:
        return Dimension(tuple(Fraction(kw.get(k, 0)) for k in _BASE))

    def __mul__(self, o: Dimension) -> Dimension:
        return Dimension(tuple(a + b for a, b in zip(self.exps, o.exps, strict=True)))

    def __truediv__(self, o: Dimension) -> Dimension:
        return Dimension(tuple(a - b for a, b in zip(self.exps, o.exps, strict=True)))

    def __pow__(self, p: int | Fraction) -> Dimension:
        return Dimension(tuple(a * Fraction(p) for a in self.exps))

    @property
    def dimensionless(self) -> bool:
        return all(e == 0 for e in self.exps)

    def __str__(self) -> str:
        parts = [f"{b}^{e}" for b, e in zip(_BASE, self.exps, strict=True) if e != 0]
        return "·".join(parts) or "1"


_D = Dimension.of
_ONE = _D()
_PRESSURE = _D(M=1, L=-1, T=-2)
_ENERGY = _D(M=1, L=2, T=-2)
_POWER = _D(M=1, L=2, T=-3)
_FORCE = _D(M=1, L=1, T=-2)

# name -> (dimension, SI scale). Symbols are case-sensitive (mol vs Mol, m vs M).
_UNITS: dict[str, tuple[Dimension, float]] = {
    "1": (_ONE, 1.0), "dimensionless": (_ONE, 1.0), "fraction": (_ONE, 1.0),
    "percent": (_ONE, 1e-2), "ppm": (_ONE, 1e-6), "ppb": (_ONE, 1e-9),
    "rad": (_ONE, 1.0), "deg": (_ONE, 0.017453292519943295),
    "m": (_D(L=1), 1.0), "g": (_D(M=1), 1e-3), "s": (_D(T=1), 1.0),
    "K": (_D(K=1), 1.0), "mol": (_D(N=1), 1.0), "A": (_D(I=1), 1.0), "cd": (_D(J=1), 1.0),
    "min": (_D(T=1), 60.0), "h": (_D(T=1), 3600.0), "hour": (_D(T=1), 3600.0),
    "day": (_D(T=1), 86400.0), "yr": (_D(T=1), 31_557_600.0),  # Julian year
    "tonne": (_D(M=1), 1e3), "t": (_D(M=1), 1e3),
    "L": (_D(L=3), 1e-3), "AU": (_D(L=1), 149_597_870_700.0),
    "Pa": (_PRESSURE, 1.0), "bar": (_PRESSURE, 1e5), "atm": (_PRESSURE, 101_325.0),
    "torr": (_PRESSURE, 101_325.0 / 760.0),
    "J": (_ENERGY, 1.0), "eV": (_ENERGY, 1.602176634e-19), "Wh": (_ENERGY, 3600.0),
    "N": (_FORCE, 1.0), "W": (_POWER, 1.0), "Hz": (_D(T=-1), 1.0),
    "C": (_D(I=1, T=1), 1.0), "V": (_D(M=1, L=2, T=-3, I=-1), 1.0),
}
# SI prefixes usable on the units flagged below (kg is handled through "g").
_PREFIXES = {
    "Y": 1e24, "Z": 1e21, "E": 1e18, "P": 1e15, "T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3,
    "h": 1e2, "d": 1e-1, "c": 1e-2, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9,
    "p": 1e-12, "f": 1e-15,
}
_PREFIXABLE = {"m", "g", "s", "K", "mol", "A", "cd", "Pa", "bar", "J", "eV", "Wh", "N",
               "W", "Hz", "C", "V", "L", "t", "tonne"}

_TOKEN = re.compile(r"^([A-Za-zµ]+|1)(?:\^|\*\*)?(-?\d+(?:/\d+)?)?$")


def _atom(sym: str) -> tuple[Dimension, float]:
    if sym in _UNITS:
        return _UNITS[sym]
    if len(sym) > 1 and sym[0] in _PREFIXES and sym[1:] in _PREFIXABLE:
        dim, scale = _UNITS[sym[1:]]
        return dim, scale * _PREFIXES[sym[0]]
    raise UnitError(f"Unknown unit '{sym}'")


def parse_unit(expr: str) -> tuple[Dimension, float]:
    """Parse a unit expression into ``(Dimension, scale_to_SI)``.

    Grammar: factors separated by whitespace, ``*`` or ``·`` (multiply) and
    ``/`` (divide everything after it, left to right), each factor
    ``symbol[^exp]`` with integer or rational exponent.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise UnitError("Empty unit expression")
    text = expr.strip().replace("·", "*").replace("**", "^")
    dim, scale = _ONE, 1.0
    sign = 1
    for chunk in re.split(r"([/])", text):
        if chunk == "/":
            sign = -1
            continue
        for tok in re.split(r"[\s*]+", chunk.strip()):
            if not tok:
                continue
            m = _TOKEN.match(tok.replace("**", "^"))
            if not m:
                raise UnitError(f"Malformed unit token '{tok}' in '{expr}'")
            d, sc = _atom(m.group(1))
            p = Fraction(m.group(2)) if m.group(2) else Fraction(1)
            p *= sign
            dim = dim * (d ** p)
            scale *= sc ** float(p)
    return dim, scale


def compatible(a: str, b: str) -> bool:
    return parse_unit(a)[0] == parse_unit(b)[0]


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between dimensionally compatible units; raises on mismatch."""
    d1, s1 = parse_unit(from_unit)
    d2, s2 = parse_unit(to_unit)
    if d1 != d2:
        raise UnitError(f"Cannot convert '{from_unit}' [{d1}] to '{to_unit}' [{d2}]")
    return value * s1 / s2


def to_si(value: float, unit: str) -> float:
    return value * parse_unit(unit)[1]


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Affine temperature conversion among K, degC, degF."""
    to_k = {"K": lambda v: v, "degC": lambda v: v + 273.15,
            "degF": lambda v: (v - 32.0) * 5.0 / 9.0 + 273.15}
    from_k = {"K": lambda v: v, "degC": lambda v: v - 273.15,
              "degF": lambda v: (v - 273.15) * 9.0 / 5.0 + 32.0}
    if from_unit not in to_k or to_unit not in from_k:
        raise UnitError(f"Unsupported temperature unit: {from_unit!r} -> {to_unit!r}")
    kelvin = to_k[from_unit](value)
    if kelvin < 0.0:
        raise UnitError("Temperature below absolute zero")
    return float(from_k[to_unit](kelvin))


def assert_equation(lhs: str, rhs: str, what: str = "equation") -> None:
    """Raise if both sides of an equation do not share a dimension."""
    d1, d2 = parse_unit(lhs)[0], parse_unit(rhs)[0]
    if d1 != d2:
        raise UnitError(f"Dimensional inhomogeneity in {what}: [{d1}] != [{d2}]")
