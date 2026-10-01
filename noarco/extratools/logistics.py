"""
noarco.extratools.logistics — NOArCO-Logistics
=============================================
Interplanetary Supply Chain and Volatile Logistics Planner.
Resolves planetary bottlenecks (N2 deficit on Mars, lack of volatiles on Mercury)
by computing orbital maneuvers (Delta v), cometary redirection, impact energy and retention.

Framework: NOArCO
Framework and TRIADA protocol author: Alejo Malia
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from noarco.core.planet_state import PlanetState

_MU_SUN_KM3_S2 = 1.32712440018e11        # GM_sun (IAU 2015 nominal)
_AU_KM = 149_597_870.7
_G0 = 9.80665


def hohmann_delta_v_km_s(r1_au: float, r2_au: float) -> float:
    """Total heliocentric delta-v (km/s) of a Hohmann transfer between circular orbits ``r1`` and ``r2``.

    ``dv = |v1 (sqrt(2 r2 / (r1 + r2)) - 1)| + |v2 (1 - sqrt(2 r1 / (r1 + r2)))|``. Closed form, coplanar,
    no planetary-gravity assists and no capture: a *lower-bound reference* for the heliocentric part only.
    """
    if r1_au <= 0 or r2_au <= 0:
        raise ValueError("orbit radii must be > 0")
    r1, r2 = r1_au * _AU_KM, r2_au * _AU_KM
    v1, v2 = math.sqrt(_MU_SUN_KM3_S2 / r1), math.sqrt(_MU_SUN_KM3_S2 / r2)
    return abs(v1 * (math.sqrt(2 * r2 / (r1 + r2)) - 1)) + abs(v2 * (1 - math.sqrt(2 * r1 / (r1 + r2))))


@dataclass
class VolatileSource:
    """Candidate volatile source in the Solar System."""
    name: str
    location_au: float
    nitrogen_fraction: float              # Mass fraction of N2 / NH3
    water_fraction: float                 # Mass fraction of H2O
    co2_fraction: float                   # Mass fraction of CO2
    typical_body_mass_kg: float           # Typical comet / asteroid mass (e.g. 2e14 kg)
    delta_v_to_mars_km_s: float           # Insertion delta-v
    delta_v_to_mercury_km_s: float


# Standard catalog of volatile sources
STANDARD_SOURCES: dict[str, VolatileSource] = {
    "Titan": VolatileSource(
        name="Titan Atmosphere",
        location_au=9.54,
        nitrogen_fraction=0.95,
        water_fraction=0.00,
        co2_fraction=0.00,
        typical_body_mass_kg=1.0e15,       # Orbital compression tanker
        delta_v_to_mars_km_s=6.8,
        delta_v_to_mercury_km_s=11.2,
    ),
    "Kuiper_Belt_Comets": VolatileSource(
        name="Kuiper Belt Comets",
        location_au=40.0,
        nitrogen_fraction=0.05,            # Mainly ammonia (NH3) and ions
        water_fraction=0.75,
        co2_fraction=0.15,
        typical_body_mass_kg=2.5e14,       # ~7 km diameter
        delta_v_to_mars_km_s=3.5,          # Via Neptune/Jupiter gravitational perturbation
        delta_v_to_mercury_km_s=6.5,
    ),
    "Main_Belt_Asteroids": VolatileSource(
        name="Carbonaceous Asteroids (C-type - Main Belt)",
        location_au=2.8,
        nitrogen_fraction=0.015,
        water_fraction=0.10,
        co2_fraction=0.05,
        typical_body_mass_kg=1.0e16,
        delta_v_to_mars_km_s=2.8,
        delta_v_to_mercury_km_s=7.9,
    ),
}


@dataclass
class LogisticsCampaignReport:
    """Comprehensive volatile transport and resupply plan."""
    target_planet_name: str
    target_species: str
    required_mass_kg: float
    source_name: str
    delta_v_km_s: float
    propulsive_energy_joules: float
    bodies_required_count: int
    impact_velocity_km_s: float
    kinetic_energy_impact_joules: float
    atmospheric_retention_fraction: float # Fraction of gas that does not escape after impact
    gross_import_mass_kg: float           # Gross mass to divert, accounting for losses
    estimated_campaign_years: float
    logistics_bottlenecks: list[str] = field(default_factory=list)
    # --- added ---
    delta_v_method: str = "catalogue"                 # catalogue | heliocentric_hohmann
    delta_v_hohmann_km_s: float = 0.0                 # heliocentric Hohmann transfer, source orbit -> target orbit
    propellant_fraction: float = 0.0                  # 1 - exp(-dv / ve) of the redirected mass (rocket equation)
    propellant_mass_kg: float = 0.0                   # propellant to redirect all bodies at the given Isp
    warnings: list[str] = field(default_factory=list)
    model_class: str = "screening_upper_bound"


class VolatileLogisticsPlanner:
    """
    NOArCO astronautical logistics and cometary redirection planner.
    Determines analytically the kinetic and engineering effort
    required to mitigate global elemental deficits.
    """

    @staticmethod
    def plan_import(
        target_planet: PlanetState,
        species: str,
        required_net_mass_kg: float,
        source_key: str = "Kuiper_Belt_Comets",
        deflections_per_century: float = 5.0,
        v_infinity_km_s: float = 5.0,
        isp_s: float = 3000.0,
    ) -> LogisticsCampaignReport:
        if source_key not in STANDARD_SOURCES:
            raise KeyError(f"Source '{source_key}' not recognized. Options: {list(STANDARD_SOURCES.keys())}")

        source = STANDARD_SOURCES[source_key]

        if not deflections_per_century > 0:
            raise ValueError("deflections_per_century must be > 0")
        if not (v_infinity_km_s >= 0 and isp_s > 0):
            raise ValueError("v_infinity_km_s must be >= 0 and isp_s > 0")
        if not (required_net_mass_kg > 0 and math.isfinite(required_net_mass_kg)):
            raise ValueError(f"required_net_mass_kg must be finite and > 0, got {required_net_mass_kg!r}")

        # 1. Species fraction in the source
        if species.upper() in ["N2", "NITROGEN", "N"]:
            mass_frac = source.nitrogen_fraction
        elif species.upper() in ["H2O", "WATER"]:
            mass_frac = source.water_fraction
        elif species.upper() in ["CO2"]:
            mass_frac = source.co2_fraction
        else:
            raise ValueError(
                f"Unsupported species '{species}'. Supported: N2 (N, nitrogen), H2O (water), CO2."
            )

        if mass_frac <= 0.0:
            raise ValueError(f"Source {source.name} does not contain {species}.")

        # 2. Atmospheric retention according to the planet's escape velocity
        # Typical impact velocity: v_impact = sqrt(v_esc^2 + v_inf^2)
        v_esc = target_planet.escape_velocity_m_s / 1000.0  # km/s
        v_inf = v_infinity_km_s  # hyperbolic excess speed at the target (assumption, parameter)
        v_impact = math.sqrt(v_esc**2 + v_inf**2)

        # Melosh & Vickery (1989) retention rule:
        # The higher the impact velocity and the lower the gravity, the more atmospheric mass is expelled by the impact plume.
        if v_esc > 8.0:
            retention_frac = 0.95  # Massive planets (Earth, Venus) retain almost everything
        elif v_esc > 4.0:
            retention_frac = 0.85  # 4-8 km/s escape velocity (e.g. Mars, 5 km/s)
        elif v_esc > 2.0:
            retention_frac = 0.65  # 2-4 km/s escape velocity (e.g. Ceres-class / large moons)
        else:
            retention_frac = 0.25  # Small bodies (Mercury, Moon, Pluto) lose most of it by blow-off

        gross_mass_species_needed = required_net_mass_kg / retention_frac
        total_source_mass_needed = gross_mass_species_needed / mass_frac

        # 3. Number of bodies / comets
        bodies_count = math.ceil(total_source_mass_needed / source.typical_body_mass_kg)

        # 4. Delta-v and propulsive energy
        # Target heliocentric orbit radius from its solar constant (S = 1361 W/m2 at 1 AU): r = sqrt(1361 / S).
        r_target_au = math.sqrt(1361.0 / target_planet.solar_constant_wm2)
        dv_hohmann = hohmann_delta_v_km_s(source.location_au, r_target_au)
        warnings: list[str] = []
        if "Mercury" in target_planet.name:
            delta_v, method = source.delta_v_to_mercury_km_s, "catalogue"
        elif "Mars" in target_planet.name:
            delta_v, method = source.delta_v_to_mars_km_s, "catalogue"
        else:
            delta_v, method = dv_hohmann, "heliocentric_hohmann"
            warnings.append(f"No catalogue delta-v for {target_planet.name}: heliocentric Hohmann estimate used "
                            f"({dv_hohmann:.2f} km/s; no gravity assists, no capture).")
        if method == "catalogue" and delta_v < 0.5 * dv_hohmann:
            warnings.append(f"Catalogue delta-v ({delta_v:.1f} km/s) is below half the heliocentric Hohmann value "
                            f"({dv_hohmann:.1f} km/s): it presumes gravity assists / perturbation.")

        # Kinetic energy for orbital deflection: E = 0.5 * m * (delta_v)^2
        delta_v_m_s = delta_v * 1000.0
        propulsive_energy_j = 0.5 * total_source_mass_needed * (delta_v_m_s**2)

        # 5. Total impact kinetic energy deposited on the target planet
        v_impact_m_s = v_impact * 1000.0
        kinetic_energy_impact_j = 0.5 * total_source_mass_needed * (v_impact_m_s**2)

        # 6. Campaign duration:
        # Assuming an advanced industrial deflection rate of ``deflections_per_century`` bodies per century
        # or propulsion via nuclear mass-drivers.
        campaign_years = max(50.0, (bodies_count / deflections_per_century) * 100.0)

        ve = isp_s * _G0                                              # m/s
        prop_fraction = 1.0 - math.exp(-delta_v_m_s / ve)
        prop_mass = total_source_mass_needed * (math.exp(delta_v_m_s / ve) - 1.0)
        warnings.append("Retention is a step-function heuristic after Melosh & Vickery (1989), not a hydrocode result.")

        bottlenecks: list[str] = []
        if bodies_count > 1000:
            bottlenecks.append(f"Massive scale: requires diverting {bodies_count:,} independent comets/bodies.")
        if kinetic_energy_impact_j > 1.0e26:
            bottlenecks.append(
                f"Destructive impact thermal energy ({kinetic_energy_impact_j:.1e} J): "
                "Exceeds the crust melting threshold; requires aerodynamic braking or fragmentation in the upper atmosphere."
            )
        if retention_frac < 0.50:
            bottlenecks.append(f"Low retention ({retention_frac*100:.0f}%): Impact-plume escape dissipates most of the gas.")

        return LogisticsCampaignReport(
            target_planet_name=target_planet.name,
            target_species=species,
            required_mass_kg=required_net_mass_kg,
            source_name=source.name,
            delta_v_km_s=delta_v,
            propulsive_energy_joules=propulsive_energy_j,
            bodies_required_count=bodies_count,
            impact_velocity_km_s=v_impact,
            kinetic_energy_impact_joules=kinetic_energy_impact_j,
            atmospheric_retention_fraction=retention_frac,
            gross_import_mass_kg=total_source_mass_needed,
            estimated_campaign_years=campaign_years,
            logistics_bottlenecks=bottlenecks,
            delta_v_method=method, delta_v_hohmann_km_s=dv_hohmann, propellant_fraction=prop_fraction,
            propellant_mass_kg=prop_mass, warnings=warnings,
        )
