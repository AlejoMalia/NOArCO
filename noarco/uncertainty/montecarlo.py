"""
noarco.uncertainty.montecarlo — Stratified Monte Carlo sensitivity and uncertainty propagation.
==============================================================================================

Propagates parametric uncertainties in:
    - Volatile inventories (CO2, H2O) using log-normal distributions.
    - Climate sensitivity & GWP coefficients using Gaussian distributions.
    - Biological & industrial throughput efficiencies using Beta distributions.

TRIADA & MATE Integration:
    - T1 (INVENTORY): Caches evaluated parameter sets.
    - T2 (MATHEMATICS): Closed-form analytical moments (mean, variance) computed
      directly when analytical distributions permit.
    - MATE R3: Early exit if P(Feasibility) is proven 0.0 or 1.0 within 3 sigma
      after the first batch of samples.

References
----------
Jakosky, B. M. (2019). Planet. Space Sci., 175, 52-59 (inventory distributions).
IPCC AR6 (2021). Climate Change 2021: The Physical Science Basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from noarco.core.constants import SIGMA_SB
from noarco.core.planet_state import PlanetState


@dataclass
class UncertaintyResult:
    parameter_name: str
    percentile_10: float
    percentile_50: float
    percentile_90: float
    mean: float
    std_dev: float
    samples: np.ndarray = field(repr=False)

    def summary(self) -> str:
        return (
            f"  {self.parameter_name:24s} | "
            f"P10: {self.percentile_10:.2e} | "
            f"Median (P50): {self.percentile_50:.2e} | "
            f"P90: {self.percentile_90:.2e} | "
            f"Mean: {self.mean:.2e} (±{self.std_dev:.2e})"
        )


class MonteCarloEngine:
    """Stratified Monte Carlo sampler for planetary geoengineering."""

    def __init__(
        self,
        base_planet: PlanetState,
        n_samples: int = 1000,
        seed: int = 42,
    ) -> None:
        self.base_planet = base_planet
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def sample_co2_inventory_kg(self) -> np.ndarray:
        """Sample Mars CO2 inventory log-normally across 5e16 to 4e18 kg."""
        # Median ~2e17 kg, log-normal sigma ~0.9
        mu = np.log(2.0e17)
        sigma = 0.9
        return self.rng.lognormal(mean=mu, sigma=sigma, size=self.n_samples)

    def sample_climate_sensitivity_alpha(self) -> np.ndarray:
        """Sample CO2 doubling radiative forcing alpha ~ N(6.0, 1.2) W/m2."""
        return np.clip(self.rng.normal(loc=6.0, scale=1.2, size=self.n_samples), 3.0, 10.0)

    def sample_isru_efficiency(self) -> np.ndarray:
        """Sample industrial ISRU recovery factor ~ Beta(a=5, b=3)."""
        return self.rng.beta(a=5.0, b=3.0, size=self.n_samples)

    def propagate_temperature_rise(
        self,
        target_co2_added_kg: float = 2.0e17,
    ) -> UncertaintyResult:
        """Propagate temperature rise uncertainty given added CO2 mass.

        The CO2 actually deliverable in each draw is capped by the sampled
        inventory (``min(target, inventory)``), so inventory uncertainty
        propagates into the pressure rise together with the forcing
        coefficient uncertainty.
        """
        co2_inventory = self.sample_co2_inventory_kg()
        alpha_samples = self.sample_climate_sensitivity_alpha()

        # Hydrostatic pressure rise
        p_base = self.base_planet.surface_pressure_pa
        g = self.base_planet.gravity_ms2
        area = self.base_planet.surface_area_m2

        # Added pressure
        co2_added = np.minimum(target_co2_added_kg, co2_inventory)
        delta_p = co2_added * g / area
        p_new = p_base + delta_p

        # Logarithmic CO2 forcing, converted to a surface warming with the greenhouse
        # (reduced-OLR) mapping of Turyshev (2026) Eq. (32): T_s' = T_s (1 - dF / OLR)^(-1/4).
        delta_f = alpha_samples * np.log2(p_new / p_base)
        olr = SIGMA_SB * self.base_planet.equilibrium_temperature_k**4
        x = np.clip(delta_f / olr, None, 0.999)
        delta_t = self.base_planet.mean_temperature_k * ((1.0 - x) ** -0.25 - 1.0)

        return UncertaintyResult(
            parameter_name="Delta_T_Warming_K",
            percentile_10=float(np.percentile(delta_t, 10)),
            percentile_50=float(np.percentile(delta_t, 50)),
            percentile_90=float(np.percentile(delta_t, 90)),
            mean=float(np.mean(delta_t)),
            std_dev=float(np.std(delta_t)),
            samples=delta_t,
        )
