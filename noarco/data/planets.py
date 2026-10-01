"""
noarco.data.planets
===================
Pre-built PlanetState instances for well-characterised bodies.
All values are clearly tagged with their epistemic status.

Mars E0 is the primary reference case, validated against:
    Turyshev (2026), arXiv:2603.00402, Table 2.
    Mars Climate Database v6.1 (Millour et al. 2022).
    Curiosity / Perseverance APXS data.

References
----------
Turyshev, S. G. (2026). arXiv:2603.00402.
Millour, E. et al. (2022). Mars Climate Database v6.1.
Grotzinger, J. P. et al. (2014). Science, 343, 1242777 (regolith).
"""

from noarco.core.planet_state import GasComposition, PlanetState, SourceType

# ---------------------------------------------------------------------------
# MARS -- E0 baseline (current state)
# ---------------------------------------------------------------------------

MARS = PlanetState(
    body_name="Mars",
    # --- Atmospheric state ---
    # Mean surface pressure: ~610 Pa (6.1 mbar). Varies seasonally ~500-700 Pa.
    # Source: Mars Climate Database, Viking landers, Curiosity REMS.
    surface_pressure_pa=610.0,
    # Global mean surface temperature ~210 K.
    # Varies from ~130 K (poles in winter) to ~300 K (tropics in summer).
    # Source: MCD v6.1.
    mean_temperature_k=210.0,
    gas_composition=GasComposition(
        species={
            "CO2": 0.9532,   # Mahaffy et al. (2013), SAM/Curiosity
            "N2":  0.0268,   # Mahaffy et al. (2013)
            "Ar":  0.0193,   # Mahaffy et al. (2013)
            "O2":  0.00174,  # Mahaffy et al. (2013)
            "CO":  0.000747, # Mahaffy et al. (2013)
            "H2O": 0.00030,  # Seasonal average, MCD
        }
    ),
    # --- Surface/orbital ---
    # Bond albedo ~0.25. Dust-covered regions: 0.1-0.4.
    surface_albedo=0.250,
    gravity_ms2=3.7207,           # GM/R^2 from the mass/radius below (Turyshev 2026 uses 3.71)
    radius_m=3_389_500.0,         # Mean radius (IAU 2015)
    mass_kg=6.4171e23,            # NASA planetary fact sheet
    solar_constant_wm2=586.2,     # 1361 W/m^2 / 1.524^2 AU (Turyshev 2026 uses the orbit-averaged 589)
    # --- Radiative ---
    ir_optical_depth=0.05,        # Estimated, dusty/clear conditions
    uv_flux_wm2=None,             # Not computed in baseline
    # --- Volatile inventories ---
    # CO2 ice: buried south-polar deposit ~6 mbar = 2.3e16 kg (Turyshev 2026, Sec. V.B,
    # after Jakosky & Edwards 2018). The representative *accessible* endogenous CO2
    # (polar + adsorbed + near-surface carbonate) is ~20 mbar = 7.8e16 kg; the extreme
    # crustal bound is ~1 bar (Jakosky 2019) and is not an accessible inventory.
    co2_ice_kg=2.3e16,            # Polar deposit only (see comment above)
    # H2O ice: 1.2e19 to 5e19 kg total (isotopic models + radar).
    # Reference: Villanueva et al. (2015), Science.
    h2o_ice_kg=2.0e19,            # Central estimate
    # --- Regolith composition (mass fractions) ---
    # Source: Curiosity APXS, Sol 700 average. Grotzinger et al. (2014).
    regolith_composition={
        "SiO2":  0.430,  # Silica
        "Fe2O3": 0.165,  # Iron oxide (hematite/goethite)
        "Al2O3": 0.093,  # Alumina
        "MgO":   0.088,  # Magnesium oxide
        "CaO":   0.073,  # Calcium oxide
        "SO3":   0.054,  # Sulfate
        "TiO2":  0.009,  # Titania
        "Cr2O3": 0.004,  # Chromia
        "MnO":   0.004,  # Manganese oxide
        "P2O5":  0.009,  # Phosphate
        "Cl":    0.005,  # Chlorine (incl. perchlorates)
    },
    source_type=SourceType.MEASURED,
    reference=(
        "Turyshev (2026) arXiv:2603.00402; Mars Climate Database v6.1; "
        "Mahaffy et al. (2013) Science 341:263; "
        "Grotzinger et al. (2014) Science 343:1242777; "
        "Jakosky (2019) Planet. Space Sci. 175:52."
    ),
    notes=(
        "Mars E0 baseline. Seasonal pressure range 500-700 Pa. "
        "CO2 ice inventory is polar cap only (measured lower bound); "
        "regolith-adsorbed CO2 adds 2e17-4e18 kg but is highly uncertain."
    ),
)

# ---------------------------------------------------------------------------
# VENUS -- E0 baseline (current state)
# ---------------------------------------------------------------------------

VENUS = PlanetState(
    body_name="Venus",
    surface_pressure_pa=9.2e6,    # ~92 bar
    mean_temperature_k=737.0,     # Mean surface temperature
    gas_composition=GasComposition(
        species={
            "CO2": 0.9650,
            "N2":  0.0350,
            "SO2": 0.000150,
            "H2O": 0.000020,
        }
    ),
    surface_albedo=0.770,          # High albedo due to cloud cover
    gravity_ms2=8.870,
    radius_m=6_051_800.0,
    mass_kg=4.8675e24,
    solar_constant_wm2=2601.0,     # At 0.723 AU
    ir_optical_depth=100.0,        # Extreme greenhouse
    regolith_composition={
        "SiO2": 0.451,  # Venera 13/14 basaltic crust
        "Al2O3": 0.158,
        "Fe2O3": 0.093,
        "MgO": 0.114,
        "CaO": 0.071,
        "TiO2": 0.016,
        "SO3": 0.016,
    },
    source_type=SourceType.MEASURED,
    reference="Venus Express; Seiff et al. (1986) J. Geophys. Res.; Surkov et al. (1984)",
    notes="Venus E0 baseline. Terraforming goal: cooling, CO2 removal, water import.",
)

# ---------------------------------------------------------------------------
# EARTH -- reference (for validation)
# ---------------------------------------------------------------------------

EARTH = PlanetState(
    body_name="Earth",
    surface_pressure_pa=101_325.0,
    mean_temperature_k=288.0,
    gas_composition=GasComposition(
        species={
            "N2":  0.7809,
            "O2":  0.2095,
            "Ar":  0.0093,
            "CO2": 0.000421,  # ~421 ppm (2024)
            "H2O": 0.0010,    # ~0.1% global mean (dry-air basis; moist tropics ~2%)
        }
    ),
    surface_albedo=0.300,
    gravity_ms2=9.807,
    radius_m=6_371_000.0,
    mass_kg=5.9722e24,
    solar_constant_wm2=1361.0,
    ir_optical_depth=0.78,         # Effective gray-atmosphere value
    co2_ice_kg=None,
    h2o_ice_kg=2.6e19,             # Cryosphere estimate
    regolith_composition={
        "SiO2": 0.606,  # Continental crust average (Taylor & McLennan)
        "Al2O3": 0.159,
        "Fe2O3": 0.067,
        "CaO": 0.064,
        "MgO": 0.047,
        "TiO2": 0.007,
    },
    source_type=SourceType.MEASURED,
    reference="CODATA 2018; NOAA Global Monitoring Laboratory (CO2 2024); Taylor & McLennan (1985).",
    notes="Earth reference state for model validation.",
)

# ---------------------------------------------------------------------------
# MERCURY
# ---------------------------------------------------------------------------
MERCURY = PlanetState(
    body_name="Mercury",
    surface_pressure_pa=1e-9,  # Exosphere trace ~1 nPa
    mean_temperature_k=440.0,  # Mean diurnal: ~100 K night to 700 K day
    gas_composition=GasComposition(species={"O2": 0.42, "Na": 0.29, "H2": 0.22, "He": 0.06}),
    surface_albedo=0.088,
    gravity_ms2=3.70,
    radius_m=2_439_700.0,
    mass_kg=3.3011e23,
    solar_constant_wm2=9126.6,  # 0.387 AU
    ir_optical_depth=0.0,
    co2_ice_kg=0.0,
    h2o_ice_kg=1e12,  # Polar shadowed craters (MESSENGER)
    regolith_composition={
        "SiO2": 0.420,  # MESSENGER XRS data
        "MgO": 0.250,
        "Al2O3": 0.080,
        "CaO": 0.080,
        "Fe2O3": 0.030,
        "SO3": 0.040,
    },
    source_type=SourceType.MEASURED,
    reference="NASA Planetary Fact Sheet; MESSENGER XRS (Nittler et al. 2011).",
)

# ---------------------------------------------------------------------------
# MOON (LUNA)
# ---------------------------------------------------------------------------
MOON = PlanetState(
    body_name="Moon",
    surface_pressure_pa=3e-10,  # ~0.3 nPa exosphere
    mean_temperature_k=250.0,  # ~100 K night to 390 K day
    gas_composition=GasComposition(species={"Ar": 0.40, "He": 0.40, "Ne": 0.20}),
    surface_albedo=0.120,
    gravity_ms2=1.62,
    radius_m=1_737_400.0,
    mass_kg=7.342e22,
    solar_constant_wm2=1361.0,
    ir_optical_depth=0.0,
    co2_ice_kg=0.0,
    h2o_ice_kg=6e11,  # LCROSS / Mini-SAR polar ice estimates
    regolith_composition={
        "SiO2": 0.450,  # Apollo sample baseline
        "Al2O3": 0.180,
        "Fe2O3": 0.120,
        "CaO": 0.130,
        "MgO": 0.090,
        "TiO2": 0.030,
    },
    source_type=SourceType.MEASURED,
    reference="Apollo surface data; LCROSS; Heiken et al. (1991) Lunar Sourcebook.",
)

# ---------------------------------------------------------------------------
# JUPITER (at 1 bar level)
# ---------------------------------------------------------------------------
JUPITER = PlanetState(
    body_name="Jupiter",
    surface_pressure_pa=100_000.0,  # 1 bar reference altitude
    mean_temperature_k=165.0,       # T at 1 bar
    gas_composition=GasComposition(species={"H2": 0.898, "He": 0.102}),
    surface_albedo=0.503,
    gravity_ms2=24.79,
    radius_m=69_911_000.0,
    mass_kg=1.8982e27,
    solar_constant_wm2=50.3,        # 5.20 AU
    ir_optical_depth=2.5,
    source_type=SourceType.MEASURED,
    reference="Galileo probe; Juno mission.",
)

# ---------------------------------------------------------------------------
# SATURN (at 1 bar level)
# ---------------------------------------------------------------------------
SATURN = PlanetState(
    body_name="Saturn",
    surface_pressure_pa=100_000.0,
    mean_temperature_k=134.0,
    gas_composition=GasComposition(species={"H2": 0.963, "He": 0.0325, "CH4": 0.0045}),
    surface_albedo=0.342,
    gravity_ms2=10.44,
    radius_m=58_232_000.0,
    mass_kg=5.6834e26,
    solar_constant_wm2=14.9,        # 9.58 AU
    ir_optical_depth=3.0,
    source_type=SourceType.MEASURED,
    reference="Cassini-Huygens data.",
)

# ---------------------------------------------------------------------------
# URANUS (at 1 bar level)
# ---------------------------------------------------------------------------
URANUS = PlanetState(
    body_name="Uranus",
    surface_pressure_pa=100_000.0,
    mean_temperature_k=76.0,
    gas_composition=GasComposition(species={"H2": 0.825, "He": 0.152, "CH4": 0.023}),
    surface_albedo=0.300,
    gravity_ms2=8.69,
    radius_m=25_362_000.0,
    mass_kg=8.6810e25,
    solar_constant_wm2=3.69,        # 19.2 AU
    ir_optical_depth=4.0,
    source_type=SourceType.MEASURED,
    reference="Voyager 2.",
)

# ---------------------------------------------------------------------------
# NEPTUNE (at 1 bar level)
# ---------------------------------------------------------------------------
NEPTUNE = PlanetState(
    body_name="Neptune",
    surface_pressure_pa=100_000.0,
    mean_temperature_k=72.0,
    gas_composition=GasComposition(species={"H2": 0.800, "He": 0.190, "CH4": 0.010}),
    surface_albedo=0.290,
    gravity_ms2=11.15,
    radius_m=24_622_000.0,
    mass_kg=1.0241e26,
    solar_constant_wm2=1.50,        # 30.05 AU
    ir_optical_depth=4.5,
    source_type=SourceType.MEASURED,
    reference="Voyager 2.",
)

# ---------------------------------------------------------------------------
# PLUTO
# ---------------------------------------------------------------------------
PLUTO = PlanetState(
    body_name="Pluto",
    surface_pressure_pa=1.0,        # ~1 Pa (10 microbar, New Horizons)
    mean_temperature_k=44.0,
    gas_composition=GasComposition(species={"N2": 0.990, "CH4": 0.005, "CO": 0.005}),
    surface_albedo=0.550,
    gravity_ms2=0.620,
    radius_m=1_188_300.0,
    mass_kg=1.303e22,
    solar_constant_wm2=0.87,        # 39.48 AU
    ir_optical_depth=0.01,
    h2o_ice_kg=1e21,               # Water ice bedrock
    regolith_composition={
        "H2O": 0.65,  # Water ice bedrock
        "CH4": 0.15,  # Methane frost
        "N2":  0.10,  # Nitrogen ice (Sputnik Planitia)
        "CO":  0.05,  # Carbon monoxide ice
        "Silicates": 0.05, # Insoluble mineral dust/core
    },
    source_type=SourceType.MEASURED,
    reference="New Horizons (Stern et al. 2015).",
)

ALL_SOLAR_BODIES: dict[str, PlanetState] = {
    "Mercury": MERCURY,
    "Venus": VENUS,
    "Earth": EARTH,
    "Moon": MOON,
    "Mars": MARS,
    "Jupiter": JUPITER,
    "Saturn": SATURN,
    "Uranus": URANUS,
    "Neptune": NEPTUNE,
    "Pluto": PLUTO,
}

