"""
noarco.core.constants — physical constants (single source of truth)
====================================================================

SI units. Values are CODATA 2018 / IAU 2015 exact or recommended values unless
a different source is named. Model modules import from here instead of
re-declaring literals so that a constant is defined (and audited) once.
"""

from __future__ import annotations

#: Stefan-Boltzmann constant (W m^-2 K^-4), CODATA 2018 (exact).
SIGMA_SB = 5.670374419e-8
#: Newtonian constant of gravitation (m^3 kg^-1 s^-2), CODATA 2018.
G_NEWTON = 6.67430e-11
#: Universal gas constant (J mol^-1 K^-1), exact (N_A * k_B).
R_GAS = 8.314462618
#: Boltzmann constant (J K^-1), exact.
K_BOLTZMANN = 1.380649e-23
#: Avogadro constant (mol^-1), exact.
N_AVOGADRO = 6.02214076e23
#: Speed of light in vacuum (m s^-1), exact.
C_LIGHT = 299_792_458.0
#: Julian year (s), IAU.
YEAR_S = 365.25 * 86_400.0
#: Astronomical unit (m), IAU 2012 (exact).
AU_M = 149_597_870_700.0
#: Solar constant at 1 AU (W m^-2), total solar irradiance ~1361 (Kopp & Lean 2011).
SOLAR_CONSTANT_1AU = 1361.0

# --- Molar masses (kg/mol), IUPAC standard atomic weights -------------------------------
MOLAR_MASS = {
    "H2": 2.01588e-3, "He": 4.002602e-3, "N2": 28.0134e-3, "O2": 31.9988e-3,
    "Ar": 39.948e-3, "CO2": 44.0095e-3, "CO": 28.0101e-3, "H2O": 18.01528e-3,
    "CH4": 16.0425e-3, "SO2": 64.0638e-3, "Ne": 20.1797e-3, "Na": 22.98977e-3,
}

# --- Thermochemistry of water splitting 2 H2O(l) -> 2 H2 + O2 at 298.15 K, 1 bar -------------
#: Standard Gibbs energy of formation of liquid water (J/mol H2O), reversible work per mol H2O.
DELTA_G_H2O_J_MOL = 237_129.0
#: Standard enthalpy of formation of liquid water (J/mol H2O) (thermoneutral energy).
DELTA_H_H2O_J_MOL = 285_830.0


def reversible_o2_energy_j_per_kg() -> float:
    """Minimum (reversible, Gibbs) work to liberate 1 kg of O2 from water: ~14.8 MJ/kg.

    2 mol H2O are split per mol O2, so W = 2 * dG / M_O2.
    """
    return 2.0 * DELTA_G_H2O_J_MOL / MOLAR_MASS["O2"]


def reversible_h2_energy_j_per_kg() -> float:
    """Minimum (reversible, Gibbs) work to liberate 1 kg of H2 from water: ~1.18e8 J/kg."""
    return DELTA_G_H2O_J_MOL / MOLAR_MASS["H2"]


#: Latent heat of sublimation of CO2 ice near 150-195 K (J/kg), NIST WebBook (~0.57-0.59 MJ/kg).
CO2_SUBLIMATION_J_KG = 5.9e5
#: Water triple point (K, Pa), exact by the 1990 ITS / IAPWS.
WATER_TRIPLE_POINT_K = 273.16
WATER_TRIPLE_POINT_PA = 611.657
#: Armstrong limit: pressure below which body-temperature water boils (ebullism), 6.27 kPa
#: (47 mmHg vapour pressure of water at 37 degC).
ARMSTRONG_LIMIT_PA = 6_270.0
