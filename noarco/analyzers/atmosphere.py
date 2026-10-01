"""
noarco.analyzers.atmosphere — AtmosphereEngine
===============================================
Deep characterization and volumetric modeling of planetary atmospheres.

Distinguishes rigorously between:
1. `closed_volume` (enclosed habitats, bio-domes, pressure envelopes)
2. `open_atmosphere` (global planetary scale, hydrostatic column balance)

Takes into account:
- Ambient surface pressure P0 and composition
- Planetary gravity g and scale height H
- Sensible heating/cooling requirements (with COP for heat pumps/refrigeration)
- Compression/pumping work against ambient atmospheres (e.g. Venus 92 bar)
- Volatile addition (Mars) vs volatile removal/sequestration (Venus)
- Aerostat regime for Gas Giants (absence of solid surface)
- Jeans escape limits on low-gravity bodies (Moon, Pluto)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from noarco.core.constants import G_NEWTON as G_GRAV
from noarco.core.constants import R_GAS
from noarco.core.planet_state import GasComposition, PlanetState

MOLAR_MASS_AIR = 0.02897  # kg/mol (Earth air: ~78% N2, 21% O2, 1% Ar)
MOLAR_MASS_O2 = 0.031999  # kg/mol
MOLAR_MASS_N2 = 0.028014  # kg/mol
MOLAR_MASS_CO2 = 0.044009  # kg/mol
MOLAR_MASS_H2 = 0.002016  # kg/mol
MOLAR_MASS_HE = 0.004003  # kg/mol
CP_AIR = 1005.0  # J / (kg * K)
CV_AIR = 718.0  # J / (kg * K)


@dataclass
class AtmosphereAnalysis:
    """Complete diagnostic of an ambient planetary atmosphere."""
    body_name: str
    surface_pressure_pa: float
    surface_pressure_bar: float
    mean_temperature_k: float
    mean_temperature_c: float
    molar_mass_mean_kg_mol: float
    density_surface_kg_m3: float
    scale_height_m: float
    total_atmospheric_mass_kg: float
    equivalent_atmosphere_volume_m3: float
    escape_velocity_ms: float
    jeans_retention_stable: bool
    is_gas_giant: bool
    composition_moles: dict[str, float]
    composition_mass_fractions: dict[str, float]
    notes: str = ""


@dataclass
class VolumetricRequirement:
    """Resource, thermodynamic, and mechanical requirements for conditioning a volume."""
    volume_m3: float
    mode: str  # "closed_volume" | "open_atmosphere" | "floating_aerostat"
    target_pressure_pa: float
    target_temperature_k: float
    target_o2_fraction: float
    total_gas_mass_kg: float
    o2_mass_kg: float
    n2_mass_kg: float
    net_mass_delta_kg: float  # Positive: add mass; Negative: remove mass (e.g. Venus)
    thermal_energy_joules: float
    thermal_energy_kwh: float
    compression_work_joules: float
    total_conditioning_energy_joules: float
    total_conditioning_energy_kwh: float
    jeans_blowoff_warning: bool
    structural_enclosure_required: bool
    regime_description: str
    notes: str = ""


class AtmosphereEngine:
    """Engine for atmospheric analysis and volumetric terraforming calculations."""

    @staticmethod
    def get_mean_molar_mass(gas_comp: GasComposition) -> float:
        """Compute mean molar mass (kg/mol) from species mole fractions."""
        species_weights = {
            "CO2": MOLAR_MASS_CO2,
            "N2": MOLAR_MASS_N2,
            "O2": MOLAR_MASS_O2,
            "Ar": 0.039948,
            "H2O": 0.018015,
            "CO": 0.028010,
            "SO2": 0.064066,
            "CH4": 0.016043,
            "H2": MOLAR_MASS_H2,
            "He": MOLAR_MASS_HE,
            "Na": 0.022990,
            "Ne": 0.020180,
        }
        total_frac = 0.0
        weighted_mass = 0.0
        for sp, frac in gas_comp.species.items():
            mw = species_weights.get(sp, 0.028)
            weighted_mass += frac * mw
            total_frac += frac
        if total_frac <= 0:
            return MOLAR_MASS_AIR
        return weighted_mass / total_frac

    @classmethod
    def analyze(cls, planet: PlanetState) -> AtmosphereAnalysis:
        """Perform comprehensive 100% characterization of the planetary atmosphere."""
        p0 = planet.surface_pressure_pa
        t0 = planet.mean_temperature_k
        g = planet.gravity_ms2
        r = planet.radius_m
        surface_area = 4.0 * np.pi * r**2

        mu = cls.get_mean_molar_mass(planet.gas_composition)
        # Density rho = P * mu / (R * T)
        density = (p0 * mu) / (R_GAS * t0) if t0 > 0 else 0.0
        # Scale height H = R * T / (mu * g)
        scale_height = (R_GAS * t0) / (mu * g) if (mu > 0 and g > 0) else 0.0
        total_mass = (p0 * surface_area) / g if g > 0 else 0.0
        atm_volume = surface_area * scale_height

        # Escape velocity v_esc = sqrt(2 * G * M / R)
        v_esc = float(np.sqrt(2.0 * G_GRAV * planet.mass_kg / r)) if r > 0 else 0.0
        # Thermal speed of N2 at T0
        v_th_n2 = float(np.sqrt(3.0 * R_GAS * t0 / MOLAR_MASS_N2)) if t0 > 0 else 0.0
        # Atmosphere retention rule of thumb: v_esc > 6 * v_thermal
        jeans_stable = bool(v_esc >= (6.0 * v_th_n2))

        is_gas_giant = planet.body_name in ("Jupiter", "Saturn", "Uranus", "Neptune")

        # Mass fractions
        mass_fracs: dict[str, float] = {}
        denom = sum(frac * cls.get_species_molar_mass(sp) for sp, frac in planet.gas_composition.species.items())
        if denom > 0:
            for sp, frac in planet.gas_composition.species.items():
                mass_fracs[sp] = (frac * cls.get_species_molar_mass(sp)) / denom

        return AtmosphereAnalysis(
            body_name=planet.body_name,
            surface_pressure_pa=p0,
            surface_pressure_bar=p0 / 1e5,
            mean_temperature_k=t0,
            mean_temperature_c=t0 - 273.15,
            molar_mass_mean_kg_mol=mu,
            density_surface_kg_m3=density,
            scale_height_m=scale_height,
            total_atmospheric_mass_kg=total_mass,
            equivalent_atmosphere_volume_m3=atm_volume,
            escape_velocity_ms=v_esc,
            jeans_retention_stable=jeans_stable,
            is_gas_giant=is_gas_giant,
            composition_moles=dict(planet.gas_composition.species),
            composition_mass_fractions=mass_fracs,
            notes=planet.notes or "",
        )

    @staticmethod
    def get_species_molar_mass(sp: str) -> float:
        mapping = {
            "CO2": MOLAR_MASS_CO2,
            "N2": MOLAR_MASS_N2,
            "O2": MOLAR_MASS_O2,
            "Ar": 0.039948,
            "H2O": 0.018015,
            "CO": 0.028010,
            "SO2": 0.064066,
            "CH4": 0.016043,
            "H2": MOLAR_MASS_H2,
            "He": MOLAR_MASS_HE,
            "Na": 0.022990,
            "Ne": 0.020180,
        }
        return mapping.get(sp, 0.028)

    @classmethod
    def compute_volume_conditioning(
        cls,
        planet: PlanetState,
        volume_m3: float,
        target_pa: float = 101_325.0,
        target_temp_k: float = 293.15,
        target_o2_fraction: float = 0.21,
    ) -> VolumetricRequirement:
        """Compute conditioning requirements distinguishing closed habitats vs open atmospheres.

        Volumes < 1e13 m3: Closed paraterraforming / structural enclosure.
        Volumes >= 1e13 m3: Open planetary scale / hydrostatic atmosphere.
        """
        analysis = cls.analyze(planet)
        p0 = planet.surface_pressure_pa
        t0 = planet.mean_temperature_k
        g = planet.gravity_ms2
        r = planet.radius_m
        surface_area = 4.0 * np.pi * r**2

        is_open = volume_m3 >= 1e13
        is_gas_giant = analysis.is_gas_giant

        if is_gas_giant and not is_open:
            mode = "floating_aerostat"
            structural = True
            regime = "Floating aerostat at the 1 bar layer (buoyancy with hot H2 / vacuum aerogel)"
        elif is_open:
            mode = "open_atmosphere"
            structural = False
            regime = "Global open atmosphere (planetary hydrostatic equilibrium)"
        else:
            mode = "closed_volume"
            structural = True
            regime = "Pressurized closed habitat (paraterraforming with a sealed dome/structure)"

        if not is_open:
            # CLOSED ENCLOSURE (or floating aerostat)
            # Ideal gas target inside volume V
            n_target = (target_pa * volume_m3) / (R_GAS * target_temp_k)
            total_gas_mass_kg = n_target * MOLAR_MASS_AIR
            o2_mass_kg = n_target * target_o2_fraction * MOLAR_MASS_O2
            n2_mass_kg = total_gas_mass_kg - o2_mass_kg

            # Existing ambient gas mass inside volume V
            ambient_density = (p0 * analysis.molar_mass_mean_kg_mol) / (R_GAS * t0) if t0 > 0 else 0.0
            ambient_mass_in_vol = ambient_density * volume_m3

            # On Venus (P0 = 92 bar): ambient toxic CO2 must be evacuated/purged against 92 bar!
            if planet.body_name == "Venus":
                # Work to pump out ambient gas against external pressure P0: W ~ P0 * V
                compression_work_j = p0 * volume_m3
                net_mass_delta_kg = total_gas_mass_kg - ambient_mass_in_vol  # Massive negative delta: remove ~130 kg CO2 per 2 m3!
                # Refrigeration energy: must cool from 737 K down to 293.15 K
                delta_t_chill = max(0.0, t0 - target_temp_k)
                q_thermal = total_gas_mass_kg * CV_AIR * delta_t_chill
                # Refrigeration Carnot COP = T_cold / (T_hot - T_cold)
                cop_fridge = max(0.2, target_temp_k / max(1.0, delta_t_chill))
                thermal_work_j = q_thermal / cop_fridge
            elif is_gas_giant:
                # Ambient is H2/He at 1 bar. Must displace/purge H2 and replace with N2/O2.
                compression_work_j = 0.5 * p0 * volume_m3  # Displacement work
                net_mass_delta_kg = total_gas_mass_kg  # Must supply air
                # Heating from cryogenic T0 to 293 K
                delta_t_heat = max(0.0, target_temp_k - t0)
                thermal_work_j = total_gas_mass_kg * CV_AIR * delta_t_heat
            else:
                # Near-vacuum or low-pressure cold worlds (Mars, Moon, Mercury, Pluto)
                # Purging work is negligible (external pressure < 1 kPa)
                compression_work_j = target_pa * volume_m3 * 0.1  # Injection pumping work
                net_mass_delta_kg = total_gas_mass_kg
                # Sensible heating from ambient T0
                if t0 < target_temp_k:
                    delta_t_heat = target_temp_k - t0
                    thermal_work_j = total_gas_mass_kg * CV_AIR * delta_t_heat
                else:
                    # e.g. Mercury daytime (up to 700 K) requires active cooling
                    delta_t_cool = t0 - target_temp_k
                    cop = max(0.2, target_temp_k / max(1.0, delta_t_cool))
                    thermal_work_j = (total_gas_mass_kg * CV_AIR * delta_t_cool) / cop

            jeans_warning = False
            total_energy_j = thermal_work_j + compression_work_j

        else:
            # OPEN PLANETARY SCALE (Hydrostatic column balance)
            # Atmospheric volume shell is V_atm = 4 * pi * R^2 * H
            # Mass of column over an area A is M = P * A / g
            h_scale = max(100.0, analysis.scale_height_m)
            total_atm_volume = surface_area * h_scale

            # Effective surface area covered by volume_m3 in an unconfined atmosphere
            # If volume exceeds total planetary atmospheric shell, area is 100% of planet (4 * pi * R^2)
            if volume_m3 >= total_atm_volume:
                eff_area = surface_area
            else:
                eff_area = min(surface_area, volume_m3 / h_scale)

            # Hydrostatic mass: M = P * Area / g
            atm_mass_target = (target_pa * eff_area) / g
            # Background mass of current atmosphere over the same effective area
            current_atm_mass = (p0 * eff_area) / g if g > 0 else 0.0
            net_mass_delta_kg = atm_mass_target - current_atm_mass

            total_gas_mass_kg = atm_mass_target
            o2_mass_kg = total_gas_mass_kg * target_o2_fraction * (MOLAR_MASS_O2 / MOLAR_MASS_AIR)
            n2_mass_kg = total_gas_mass_kg - o2_mass_kg

            if is_gas_giant:
                regime = "INFEASIBLE: gas giants lack a solid lithospheric surface for an open atmosphere."
                compression_work_j = 0.0
                thermal_work_j = 0.0
                total_energy_j = 0.0
                jeans_warning = False
            elif planet.body_name == "Venus":
                regime = (
                    "Venusian open atmosphere: requires REMOVAL/SEQUESTRATION of 4.72e20 kg of CO2 "
                    "and dissipation of planetary heat (cooling from 737 K to 293 K)."
                )
                # Realistic Venus thermodynamics:
                # 1. Ejection / Sequestration of CO2: kinetic ejection to escape velocity (10.36 km/s)
                #    or mining and carbonating 1.5e21 kg of basaltic crust ~ 50 MJ / kg CO2
                sequestration_energy_j = abs(net_mass_delta_kg) * 5.0e7
                # 2. Planetary cooling: sensible heat of atmosphere (4.7e20 kg * 1000 J/kg/K * 444 K)
                #    plus upper crust cooling ~ 2.4e26 J total thermal dissipation
                delta_t = t0 - target_temp_k
                thermal_cooling_j = abs(net_mass_delta_kg) * CP_AIR * delta_t + 1.0e26 * (eff_area / surface_area)
                total_energy_j = sequestration_energy_j + thermal_cooling_j
                compression_work_j = sequestration_energy_j
                thermal_work_j = thermal_cooling_j
                jeans_warning = False
            else:
                # Mars, Mercury, Moon, Pluto
                # Evaluate thermal escape at the target terraformed temperature (293 K)
                v_th_target = float(np.sqrt(3.0 * R_GAS * target_temp_k / MOLAR_MASS_N2))
                jeans_warning = bool(analysis.escape_velocity_ms < (6.0 * v_th_target))
                if jeans_warning:
                    regime += (
                        f" [WARNING: Insufficient gravity (v_esc = {analysis.escape_velocity_ms:.0f} m/s vs 6*v_th = {6.0*v_th_target:.0f} m/s). "
                        f"Massive Jeans escape at {target_temp_k:.0f} K without a sealed dome / worldhouse]"
                    )
                # Heating planet atmosphere to target temp
                delta_t = max(0.0, target_temp_k - t0)
                thermal_work_j = total_gas_mass_kg * CP_AIR * delta_t
                compression_work_j = 0.0
                total_energy_j = thermal_work_j

        return VolumetricRequirement(
            volume_m3=volume_m3,
            mode=mode,
            target_pressure_pa=target_pa,
            target_temperature_k=target_temp_k,
            target_o2_fraction=target_o2_fraction,
            total_gas_mass_kg=total_gas_mass_kg,
            o2_mass_kg=o2_mass_kg,
            n2_mass_kg=n2_mass_kg,
            net_mass_delta_kg=net_mass_delta_kg,
            thermal_energy_joules=thermal_work_j,
            thermal_energy_kwh=thermal_work_j / 3.6e6,
            compression_work_joules=compression_work_j,
            total_conditioning_energy_joules=total_energy_j,
            total_conditioning_energy_kwh=total_energy_j / 3.6e6,
            jeans_blowoff_warning=jeans_warning,
            structural_enclosure_required=structural,
            regime_description=regime,
        )
