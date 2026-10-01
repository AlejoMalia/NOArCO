"""
noarco.extratools.cli — NOArCO-CLI & Interactive Console
=======================================================
Interactive console tool to quickly evaluate habitats, ISRU sites,
interplanetary logistics and exoplanets within the NOArCO framework.

Framework: NOArCO
Framework and TRIADA protocol author: Alejo Malia
"""

from __future__ import annotations

from noarco.data.planets import EARTH, JUPITER, MARS, MERCURY, NEPTUNE, PLUTO, SATURN, URANUS, VENUS
from noarco.extratools.exoplanet import JEANS_RETAINED, ExoplanetClassifier, ExoplanetProfile
from noarco.extratools.habitat import HabitatDimensioner, HabitatSpecification

PLANET_MAP = {
    "mercury": MERCURY,
    "venus": VENUS,
    "earth": EARTH,
    "mars": MARS,
    "jupiter": JUPITER,
    "saturn": SATURN,
    "uranus": URANUS,
    "neptune": NEPTUNE,
    "pluto": PLUTO,
}


def run_habitat_quick_demo(planet_name: str = "mars", crew: int = 6, volume_m3: float = 2500.0) -> None:
    """Run a quick habitat sizing evaluation."""
    planet = PLANET_MAP.get(planet_name.lower(), MARS)
    spec = HabitatSpecification(volume_m3=volume_m3, crew_size=crew)
    report = HabitatDimensioner.dimension(planet, spec)

    print("=" * 75)
    print(f"NOArCO-HABITAT: SIZING REPORT ({planet.name.upper()})")
    print("=" * 75)
    print(f"Volume: {report.volume_m3:,.0f} m³ | Crew: {report.crew_size} persons")
    print(f"Total gas mass in envelope: {report.total_gas_mass_kg:,.1f} kg")
    print(f"  - Oxygen (O₂): {report.o2_mass_kg:,.1f} kg")
    print(f"  - Nitrogen (N₂): {report.n2_mass_kg:,.1f} kg")
    print(f"Net daily O₂ consumption (makeup): {report.net_daily_o2_makeup_kg:.2f} kg/day")
    print(f"Net water makeup: {report.daily_water_makeup_kg:.2f} kg/day")
    print(f"Initial purge work: {report.displacement_work_kwh:.2f} kWh")
    print(f"Required continuous power: {report.continuous_power_kw:.2f} kW")
    print(f"HVAC climate-control power: {report.hvac_cooling_power_kw:.2f} kW")
    for rec in report.recommendations:
        print(f"  * {rec}")
    print("=" * 75)


def run_exoplanet_quick_demo(name: str = "TRAPPIST-1e") -> None:
    """Classify a known exoplanet and compute its TFI."""
    trappist_1e = ExoplanetProfile(
        name=name,
        mass_earth=0.692,
        radius_earth=0.920,
        semi_major_axis_au=0.029,
        stellar_luminosity_solar=0.000553,
        stellar_teff_k=2566.0,
        albedo=0.25,
        has_magnetic_field=True,
    )
    report = ExoplanetClassifier.classify(trappist_1e)

    print("=" * 75)
    print(f"NOArCO-EXOPLANET: HABITABILITY & TFI ASSESSMENT ({report.name})")
    print("=" * 75)
    print(f"Astrophysical Classification: {report.classification}")
    print(f"Surface Gravity: {report.surface_gravity_g:.2f} g ({report.surface_gravity_m_s2:.2f} m/s²)")
    print(f"Escape Velocity: {report.escape_velocity_km_s:.2f} km/s")
    print(f"Equilibrium Temperature: {report.equilibrium_temperature_k:.1f} K")
    print(f"In Habitable Zone: {'YES (potential liquid water)' if report.is_in_habitable_zone else 'NO'}")
    print(f"Required Radiative Forcing: {report.required_forcing_w_m2:+.1f} W/m²")
    print(f"TERRAFORMING FEASIBILITY INDEX (TFI): {report.tfi_score_pct:.1f}%")
    print("Gas Retention (Jeans parameter λ = m v_esc²/2kT at 288 K; stable if ≥ 54):")
    for gas, val in report.jeans_parameters.items():
        status = "Stable Retention" if val >= JEANS_RETAINED else "Progressive Escape"
        print(f"  - {gas}: λ = {val:.2f} ({status})")
    for d in report.diagnostics:
        print(f"  * {d}")
    print("=" * 75)


if __name__ == "__main__":
    run_habitat_quick_demo("mars", crew=6, volume_m3=2500)
    print("\n")
    run_exoplanet_quick_demo("TRAPPIST-1e")
