"""
noarco.data.elements — Planetary geochemistry, mineralogy, and ISRU database.
=============================================================================

Provides elemental and mineralogical properties, standard regolith compositions,
chemical stoichiometry, and energetic extraction thresholds for In-Situ Resource
Utilization (ISRU) on Mars, the Moon, Venus, and asteroids.

References
----------
Grotzinger, J. P. et al. (2014). Science, 343, 1242777 (Curiosity APXS).
Taylor, S. R. & McLennan, S. M. (1985). The Continental Crust.
Surkov, Y. A. et al. (1984). J. Geophys. Res., 89, B393 (Venera 13/14).
Turyshev, S. G. (2026). arXiv:2603.00402.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElementInfo:
    symbol: str
    name: str
    atomic_number: int
    atomic_weight_g_mol: float
    crustal_abundance_mars_pct: float
    primary_minerals: list[str]
    isru_extraction_energy_mj_kg: float


# Primary elements utilized in planetary engineering / terraforming
ELEMENTS: dict[str, ElementInfo] = {
    "Si": ElementInfo("Si", "Silicon", 14, 28.085, 20.1, ["SiO2 (Quartz/Silicates)", "Olivine", "Pyroxene"], 32.0),
    "Fe": ElementInfo("Fe", "Iron", 26, 55.845, 12.8, ["Fe2O3 (Hematite)", "Fe3O4 (Magnetite)", "FeO"], 18.5),
    "Al": ElementInfo("Al", "Aluminium", 13, 26.982, 4.9, ["Al2O3 (Alumina/Feldspar)", "Plagioclase"], 45.0),
    "Mg": ElementInfo("Mg", "Magnesium", 12, 24.305, 5.3, ["MgO", "Olivine", "Pyroxene"], 38.0),
    "Ca": ElementInfo("Ca", "Calcium", 20, 40.078, 5.2, ["CaO", "Plagioclase", "Carbonates"], 25.0),
    "S": ElementInfo("S", "Sulfur", 16, 32.065, 2.2, ["MgSO4", "CaSO4 (Gypsum)", "Jarosite"], 8.0),
    "C": ElementInfo("C", "Carbon", 6, 12.011, 0.1, ["CO2 (Atm/Ice)", "Carbonates"], 12.0),
    "H": ElementInfo("H", "Hydrogen", 1, 1.008, 0.05, ["H2O (Ice/Clay)", "Hydrated minerals"], 142.0),
    "O": ElementInfo("O", "Oxygen", 8, 15.999, 44.5, ["Silicates", "Oxides", "H2O", "CO2"], 0.0),
    "N": ElementInfo("N", "Nitrogen", 7, 14.007, 0.01, ["Atmospheric N2", "Nitrates"], 15.0),
    "F": ElementInfo("F", "Fluorine", 9, 18.998, 0.03, ["Fluorite (CaF2)", "Apatite"], 65.0),
}


# Standard oxide conversion factors
OXIDE_STOICHIOMETRY: dict[str, float] = {
    # Oxide: mass fraction of target metal/element
    "SiO2": 28.085 / (28.085 + 2 * 15.999),      # ~0.4674 Si
    "Fe2O3": (2 * 55.845) / (2 * 55.845 + 3 * 15.999), # ~0.6994 Fe
    "Al2O3": (2 * 26.982) / (2 * 26.982 + 3 * 15.999), # ~0.5293 Al
    "MgO": 24.305 / (24.305 + 15.999),           # ~0.6030 Mg
    "CaO": 40.078 / (40.078 + 15.999),           # ~0.7147 Ca
    "SO3": 32.065 / (32.065 + 3 * 15.999),       # ~0.4005 S
}


def get_element_recovery(oxide_mass_fractions: dict[str, float], element: str) -> float:
    """Calculate maximum extractable element mass fraction from regolith oxide assay.

    Applies TRIADA T2 (MATHEMATICS): Exact stoichiometry without iteration.
    """
    total = 0.0
    for oxide, frac in oxide_mass_fractions.items():
        if element == "Si" and oxide == "SiO2":
            total += frac * OXIDE_STOICHIOMETRY["SiO2"]
        elif element == "Fe" and oxide in ("Fe2O3", "FeO"):
            stoich = OXIDE_STOICHIOMETRY["Fe2O3"] if oxide == "Fe2O3" else (55.845 / 71.844)
            total += frac * stoich
        elif element == "Al" and oxide == "Al2O3":
            total += frac * OXIDE_STOICHIOMETRY["Al2O3"]
        elif element == "Mg" and oxide == "MgO":
            total += frac * OXIDE_STOICHIOMETRY["MgO"]
        elif element == "Ca" and oxide == "CaO":
            total += frac * OXIDE_STOICHIOMETRY["CaO"]
        elif element == "S" and oxide == "SO3":
            total += frac * OXIDE_STOICHIOMETRY["SO3"]
    return total


# ---------------------------------------------------------------------------
# Exact formula stoichiometry (no per-oxide special cases)
# ---------------------------------------------------------------------------

#: IUPAC standard atomic weights (g/mol) of every element that appears in the regolith
#: assays and in the mechanism catalog (ELEMENTS covers the framework's primary set).
ATOMIC_WEIGHT_G_MOL: dict[str, float] = {
    **{k: v.atomic_weight_g_mol for k, v in ELEMENTS.items()},
    "Ti": 47.867, "Mn": 54.938, "Cr": 51.996, "Na": 22.990, "K": 39.098, "P": 30.974,
    "Cl": 35.45, "Ni": 58.693,
}


def parse_formula(formula: str) -> dict[str, int]:
    """Element counts of a simple chemical formula such as ``'Mg2SiO4'`` or ``'C6H12O6'``.

    Parentheses, charges and hydrates are not supported and raise ``ValueError``.
    """
    import re

    if not re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", formula):
        raise ValueError(f"Unsupported formula: {formula!r}")
    counts: dict[str, int] = {}
    for sym, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if sym not in ATOMIC_WEIGHT_G_MOL:
            raise ValueError(f"Unknown element '{sym}' in {formula!r}")
        counts[sym] = counts.get(sym, 0) + (int(n) if n else 1)
    return counts


def molar_mass_g_mol(formula: str) -> float:
    """Molar mass (g/mol) from the standard atomic weights."""
    return sum(ATOMIC_WEIGHT_G_MOL[e] * n for e, n in parse_formula(formula).items())


def oxygen_mass_fraction(formula: str) -> float:
    """Mass fraction of oxygen in a compound (0 if it contains none)."""
    counts = parse_formula(formula)
    return ATOMIC_WEIGHT_G_MOL["O"] * counts.get("O", 0) / molar_mass_g_mol(formula)


def oxygen_in_oxides_fraction(oxide_mass_fractions: dict[str, float]) -> float:
    """Exact mass fraction of oxygen bound in an oxide assay: ``sum_i w_i * f_O(i)``.

    Every entry that is a chemical formula contributes its oxygen (oxides, but also ices such as
    ``H2O``); elements (``'Cl'``) contribute nothing; descriptive labels that are not formulas
    (e.g. ``'Silicates'``) are unspecified composition and are skipped.
    """
    total = 0.0
    for f, w in oxide_mass_fractions.items():
        try:
            total += w * oxygen_mass_fraction(f)
        except ValueError:
            continue
    return total


def reaction_mass_balance(reactants: dict[str, float], products: dict[str, float]) -> tuple[float, float]:
    """Total reactant and product mass (g) for a reaction given as ``{formula: coefficient}``."""
    return (sum(c * molar_mass_g_mol(f) for f, c in reactants.items()),
            sum(c * molar_mass_g_mol(f) for f, c in products.items()))
