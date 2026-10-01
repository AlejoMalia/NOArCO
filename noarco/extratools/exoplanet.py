"""
noarco.extratools.exoplanet — NOArCO-Exoplanet
=============================================
Exoplanet Habitability Classifier and Terraforming Feasibility Index (TFI).
Evaluates surface gravity, equilibrium temperature, Kopparapu (2013) limits,
Jeans thermal escape and the radiative forcing needed for any extrasolar world.

Framework: NOArCO
Framework and TRIADA protocol author: Alejo Malia
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from noarco.core.constants import G_NEWTON as G_GRAV
from noarco.core.constants import K_BOLTZMANN
from noarco.core.constants import SIGMA_SB as STEFAN_BOLTZMANN
from noarco.core.constants import SOLAR_CONSTANT_1AU as S_SOLAR_EARTH

M_EARTH = 5.972e24                    # kg
R_EARTH = 6.371e6                     # m
M_PROTON = 1.67262e-27                # kg
STANDARD_GRAVITY = 9.80665            # m/s2

#: Kopparapu et al. (2013), ApJ 765, 131 (arXiv:1301.6674), Table 3: coefficients of
#: ``S_eff = S_sun + a T + b T^2 + c T^3 + d T^4`` with ``T = T_eff - 5780 K``, valid for
#: 2600 K <= T_eff <= 7200 K and a 1 Earth-mass planet. Columns: S_sun, a, b, c, d.
KOPPARAPU_2013 = {
    "recent_venus": (1.7753, 1.4316e-4, 2.9875e-9, -7.5702e-12, -1.1635e-15),
    "runaway_greenhouse": (1.0512, 1.3242e-4, 1.5418e-8, -7.9895e-12, -1.8328e-15),
    "moist_greenhouse": (1.0140, 8.1774e-5, 1.7063e-9, -4.3241e-12, -6.6462e-16),
    "maximum_greenhouse": (0.3438, 5.8942e-5, 1.6558e-9, -3.0045e-12, -5.2983e-16),
    "early_mars": (0.3179, 5.4513e-5, 1.5313e-9, -2.7786e-12, -4.8997e-16),
}
_TEFF_RANGE = (2600.0, 7200.0)


def effective_stellar_flux(limit: str, teff_k: float) -> float:
    """Habitable-zone boundary flux ``S_eff`` (relative to Earth's) for a star (Kopparapu 2013, Eq. 2).

    Raises ``ValueError`` for an unknown ``limit``. ``teff_k`` outside 2600-7200 K is clamped to
    the fitted range (the caller should report that).
    """
    if limit not in KOPPARAPU_2013:
        raise ValueError(f"Unknown limit '{limit}'. Known: {sorted(KOPPARAPU_2013)}")
    t = min(max(teff_k, _TEFF_RANGE[0]), _TEFF_RANGE[1]) - 5780.0
    s0, a, b, c, d = KOPPARAPU_2013[limit]
    return s0 + a * t + b * t**2 + c * t**3 + d * t**4


def habitable_zone_au(limit: str, teff_k: float, luminosity_solar: float) -> float:
    """Distance (AU) of a HZ boundary: ``d = sqrt((L/L_sun) / S_eff)`` (Kopparapu 2013, Eq. 3)."""
    return math.sqrt(luminosity_solar / effective_stellar_flux(limit, teff_k))


#: Jeans parameters above this value correspond to ``v_esc >= 6 v_rms`` (lambda = 1.5 (v_esc/v_rms)^2 = 54),
#: the classical criterion for retaining a gas over billions of years; 24 corresponds to 4 v_rms.
JEANS_RETAINED = 54.0
JEANS_MARGINAL = 24.0


@dataclass
class ExoplanetProfile:
    """Fundamental astrophysical parameters of an exoplanet."""
    name: str
    mass_earth: float                     # In Earth masses (M_Earth)
    radius_earth: float                   # In Earth radii (R_Earth)
    semi_major_axis_au: float             # Orbital distance in AU
    stellar_luminosity_solar: float       # Stellar luminosity (L_Sun)
    stellar_teff_k: float = 5778.0        # Stellar effective temperature
    albedo: float = 0.30                  # Estimated Bond albedo
    has_magnetic_field: bool = True       # Presence of an intrinsic magnetosphere

    def __post_init__(self) -> None:
        for name in ("mass_earth", "radius_earth", "semi_major_axis_au",
                     "stellar_luminosity_solar", "stellar_teff_k"):
            v = getattr(self, name)
            if not (v > 0):
                raise ValueError(f"{name} must be > 0, got {v!r}")
        if not 0.0 <= self.albedo <= 1.0:
            raise ValueError(f"albedo must be in [0, 1], got {self.albedo!r}")


@dataclass
class ExoplanetReport:
    """Comprehensive astrophysical habitability and TFI report from NOArCO."""
    name: str
    surface_gravity_m_s2: float
    surface_gravity_g: float
    escape_velocity_km_s: float
    stellar_flux_w_m2: float
    stellar_flux_earth_ratio: float
    equilibrium_temperature_k: float
    is_in_habitable_zone: bool
    habitable_zone_inner_au: float
    habitable_zone_outer_au: float
    jeans_parameters: dict[str, float]    # Jeans lambda = m v_esc^2 / (2 k T) at 288 K, for H2, H2O, N2, O2, CO2
    required_forcing_w_m2: float          # Greenhouse trapping needed to hold 288.15 K (sigma T^4 - absorbed flux)
    tfi_score_pct: float                  # Terraforming Feasibility Index (0-100%)
    classification: str                   # Super-Earth, Ice World, Desert World, etc.
    diagnostics: list[str] = field(default_factory=list)
    habitable_zone_optimistic_au: tuple[float, float] = (0.0, 0.0)  # recent Venus .. early Mars


class ExoplanetClassifier:
    """
    NOArCO analytical astrophysics engine for evaluating and classifying exoplanets
    and determining their planetary-engineering potential.
    """

    @staticmethod
    def classify(planet: ExoplanetProfile) -> ExoplanetReport:
        # 1. Absolute mass and radius
        m_kg = planet.mass_earth * M_EARTH
        r_m = planet.radius_earth * R_EARTH

        # Surface gravity: g = G * M / R^2
        g_m_s2 = (G_GRAV * m_kg) / (r_m ** 2)
        g_earth = g_m_s2 / STANDARD_GRAVITY

        # Escape velocity: v_esc = sqrt(2 * G * M / R)
        v_esc_m_s = math.sqrt((2.0 * G_GRAV * m_kg) / r_m)
        v_esc_km_s = v_esc_m_s / 1000.0

        # 2. Insolation and Equilibrium Temperature
        # Stellar flux: S = (L / a^2) * S_Earth
        flux_w_m2 = (planet.stellar_luminosity_solar / (planet.semi_major_axis_au ** 2)) * S_SOLAR_EARTH
        flux_ratio = flux_w_m2 / S_SOLAR_EARTH

        # T_eq = [ S * (1 - A) / (4 * sigma) ]^(1/4)
        t_eq = ((flux_w_m2 * (1.0 - planet.albedo)) / (4.0 * STEFAN_BOLTZMANN)) ** 0.25

        # 3. Habitable Zone limits (Kopparapu et al. 2013, Eq. 2-3): conservative (moist
        #    greenhouse / maximum greenhouse) and optimistic (recent Venus / early Mars).
        teff = planet.stellar_teff_k
        lum = planet.stellar_luminosity_solar
        hz_inner_au = habitable_zone_au("moist_greenhouse", teff, lum)
        hz_outer_au = habitable_zone_au("maximum_greenhouse", teff, lum)
        hz_inner_opt_au = habitable_zone_au("recent_venus", teff, lum)
        hz_outer_opt_au = habitable_zone_au("early_mars", teff, lum)
        in_hz = hz_inner_au <= planet.semi_major_axis_au <= hz_outer_au
        in_hz_optimistic = hz_inner_opt_au <= planet.semi_major_axis_au <= hz_outer_opt_au

        # 4. Jeans escape parameter at the surface radius and 288 K:
        #    lambda = G M m / (k T r) = m v_esc^2 / (2 k T). A screening value: the real
        #    exobase is hotter and higher, which lowers lambda.
        T_REF = 288.15
        species_masses = {
            "H2": 2.016 * M_PROTON,
            "H2O": 18.015 * M_PROTON,
            "N2": 28.013 * M_PROTON,
            "O2": 31.998 * M_PROTON,
            "CO2": 44.01 * M_PROTON,
        }

        jeans_params: dict[str, float] = {}
        for sp, m_sp in species_masses.items():
            jeans_params[sp] = m_sp * v_esc_m_s**2 / (2.0 * K_BOLTZMANN * T_REF)

        # 5. Radiative Forcing Required for T_surface = 288 K
        # Greenhouse trapping needed to hold 288.15 K: sigma T_s^4 - S (1 - A) / 4.
        f_desired = STEFAN_BOLTZMANN * (288.15 ** 4)
        f_current = (flux_w_m2 * (1.0 - planet.albedo)) / 4.0
        delta_f = f_desired - f_current

        # 6. NOArCO Terraforming Feasibility Index (TFI) calculation
        # Factores:
        # f_grav: penalizes extreme gravities (< 0.3g or > 2.0g)
        if 0.6 <= g_earth <= 1.5:
            f_grav = 1.0
        elif 0.3 <= g_earth <= 2.2:
            f_grav = 0.75
        else:
            f_grav = 0.20

        # f_ret: N2 retention (lambda >= 54, i.e. v_esc >= 6 v_rms, retains it for Gyr)
        lambda_n2 = jeans_params["N2"]
        if lambda_n2 >= JEANS_RETAINED:
            f_ret = 1.0
        elif lambda_n2 >= JEANS_MARGINAL:
            f_ret = 0.50
        else:
            f_ret = 0.10

        # f_mag: protection against stellar wind / flares
        f_mag = 1.0 if planet.has_magnetic_field else 0.40

        # f_rad: radiative forcing feasibility (viable if |Delta F| < 100 W/m2)
        if abs(delta_f) < 40.0:
            f_rad = 1.0
        elif abs(delta_f) < 120.0:
            f_rad = 0.70
        elif abs(delta_f) < 250.0:
            f_rad = 0.35
        else:
            f_rad = 0.05

        tfi_score = (f_grav * 0.30 + f_ret * 0.30 + f_mag * 0.15 + f_rad * 0.25) * 100.0

        # 7. Diagnostics and Classification
        diag: list[str] = []
        if planet.mass_earth > 5.0:
            classification = "Massive Super-Earth (potential Sub-Neptune)"
            diag.append("Risk of a dense primordial H/He envelope; high surface gravity.")
        elif planet.radius_earth < 0.6:
            classification = "Icy / Rocky Sub-Earth"
            diag.append("Low escape velocity; prone to Jeans thermal escape.")
        elif in_hz:
            classification = "Temperate Candidate in the Habitable Zone"
            diag.append("Inside the conservative habitable zone (moist greenhouse to maximum greenhouse).")
        elif t_eq < 200.0:
            classification = "Ultra-Distant Cryogenic World"
            diag.append("Requires massive radiative forcing with greenhouse supergases.")
        else:
            classification = "Hyperthermal / Torrid World"
            diag.append("Excessive insolation; requires an orbital sunshade at L1 to prevent a runaway greenhouse effect.")

        if not _TEFF_RANGE[0] <= teff <= _TEFF_RANGE[1]:
            diag.append(
                f"Stellar T_eff = {teff:.0f} K is outside the {_TEFF_RANGE[0]:.0f}-{_TEFF_RANGE[1]:.0f} K range of the "
                "Kopparapu (2013) fit; boundaries are evaluated at the nearest limit."
            )
        if in_hz_optimistic and not in_hz:
            diag.append("Inside the optimistic (recent Venus / early Mars) but outside the conservative zone.")
        diag.append("TFI is a heuristic screening index with fixed weights, not a probability.")

        return ExoplanetReport(
            name=planet.name,
            surface_gravity_m_s2=g_m_s2,
            surface_gravity_g=g_earth,
            escape_velocity_km_s=v_esc_km_s,
            stellar_flux_w_m2=flux_w_m2,
            stellar_flux_earth_ratio=flux_ratio,
            equilibrium_temperature_k=t_eq,
            is_in_habitable_zone=in_hz,
            habitable_zone_inner_au=hz_inner_au,
            habitable_zone_outer_au=hz_outer_au,
            jeans_parameters=jeans_params,
            required_forcing_w_m2=delta_f,
            tfi_score_pct=tfi_score,
            classification=classification,
            diagnostics=diag,
            habitable_zone_optimistic_au=(hz_inner_opt_au, hz_outer_opt_au),
        )
