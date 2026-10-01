"""Independent reference benchmark: NOArCO versus the published numbers of Turyshev (2026).

Every expected value below is printed in S. G. Turyshev, "Terraforming Mars: Mass,
Forcing, and Industrial Throughput Constraints", arXiv:2603.00402 (tables and equations
named in each test). They were read from the paper, *not* produced by this code, so these
tests check the implementation against an external source. Tolerances reflect the paper's
rounding and the small differences between its nominal Mars parameters (g = 3.71 m/s2,
S = 589 W/m2) and NOArCO's dataset (g = 3.7207, S = 586.2).
"""

import math

import pytest

from noarco.core.constants import (
    ARMSTRONG_LIMIT_PA,
    WATER_TRIPLE_POINT_PA,
    reversible_h2_energy_j_per_kg,
    reversible_o2_energy_j_per_kg,
)
from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.core.endpoints import MARS_ENDPOINTS
from noarco.data.planets import MARS
from noarco.feasibility import assess, endpoint_requirements
from noarco.inventory.atmosphere import AtmosphericInventory
from noarco.mechanisms import AerosolMaterial, NanoparticleAerosol, OrbitalMirror
from noarco.radiative.balance import RadiativeBalance

# The paper's own nominal parameters (Table II), to isolate formulas from dataset choices.
PAPER_MARS = MARS.model_copy(update={"gravity_ms2": 3.71, "solar_constant_wm2": 589.0, "radius_m": 3.3895e6})
INV = AtmosphericInventory(PAPER_MARS)
RB = RadiativeBalance(PAPER_MARS)


class TestMassBookkeeping:
    def test_eq2_mass_per_pascal_and_mbar(self):
        assert INV.mass_per_pascal() == pytest.approx(3.89e13, rel=2e-3)        # Eq. (2), (19)
        assert INV.mass_per_mbar() == pytest.approx(3.89e15, rel=2e-3)          # Eq. (18)

    def test_noarco_dataset_within_half_percent_of_paper(self):
        assert AtmosphericInventory(MARS).mass_per_pascal() == pytest.approx(3.89e13, rel=5e-3)

    @pytest.mark.parametrize("p_pa,mass_kg", [
        (610.0, 2.37e16), (611.657, 2.38e16), (2.0e3, 7.78e16),                # Table III
        (6.27e3, 2.44e17), (2.0e4, 7.78e17), (1.0e5, 3.89e18),
    ])
    def test_table3_atmospheric_inventories(self, p_pa, mass_kg):
        assert INV.mass_per_pascal() * p_pa == pytest.approx(mass_kg, rel=1e-2)

    def test_eq3_fill_throughput_per_kpa(self):
        from noarco.core.constants import YEAR_S
        mdot = INV.mass_per_pascal() * 1000.0 / (1000.0 * YEAR_S)               # 1 kPa in 10^3 yr
        assert mdot == pytest.approx(1.23e6, rel=1e-2)

    def test_eq18_global_10kpa_and_eq26_regional(self):
        assert INV.mass_per_pascal() * 1.0e4 == pytest.approx(3.9e17, rel=1e-2)  # Table III note (c)
        assert INV.regional_gas_mass_kg(1.0e12, 5.0e4) == pytest.approx(1.35e16, rel=1e-2)  # Eq. (26)

    def test_eq21_buffer_gas_50kpa(self):
        assert INV.mass_per_pascal() * 5.0e4 == pytest.approx(1.9e18, rel=3e-2)

    def test_eq4_h2_in_co2_background(self):
        # Eq. (43): p_tot = 0.5 bar, f_H2 = 0.05, mu_bar = 41.9 g/mol -> M_H2 ~ 4.7e15 kg
        m = INV.species_mass_for_partial_pressure(0.05 * 0.5e5, "H2", 41.9e-3)
        assert m == pytest.approx(4.7e15, rel=2e-2)

    def test_eq83_max_pressurized_area_table10(self):
        assert INV.max_pressurized_area_m2(1e5, 100, 1e4) / 1e6 == pytest.approx(1.17e5, rel=1e-2)
        assert INV.max_pressurized_area_m2(1e6, 100, 5e4) / 1e6 == pytest.approx(2.34e5, rel=1e-2)
        assert INV.max_pressurized_area_m2(1e7, 100, 1e5) / 1e6 == pytest.approx(1.17e6, rel=1e-2)


class TestOxygenEnergetics:
    def test_reversible_energies_eq5_and_table6(self):
        assert reversible_o2_energy_j_per_kg() == pytest.approx(14.8e6, rel=5e-3)   # Eq. (5)
        assert reversible_h2_energy_j_per_kg() == pytest.approx(1.2e8, rel=2e-2)     # Table VI

    def test_eq68_and_eq72_breathable_oxygen(self):
        m_o2 = INV.oxygen_mass_for_partial_pressure(21_000.0)
        assert m_o2 == pytest.approx(8.2e17, rel=1e-2)                               # Eq. (68)
        assert INV.minimum_o2_production_energy_j(m_o2) == pytest.approx(1.2e25, rel=2e-2)  # Eq. (72)

    def test_eq5_energy_per_kpa_of_o2(self):
        e = INV.minimum_o2_production_energy_j(INV.oxygen_mass_for_partial_pressure(1000.0))
        assert e == pytest.approx(5.76e23, rel=1e-2)

    def test_table13_average_power_for_o2(self):
        from noarco.core.constants import YEAR_S
        e = INV.minimum_o2_production_energy_j(8.2e17)
        assert e / (1000 * YEAR_S) / 1e12 == pytest.approx(380, rel=3e-2)            # TW, Table XIII


class TestRadiativeRequirements:
    def test_eq31_direct_forcing_for_30_and_60_kelvin(self):
        assert RB.forcing_needed_for_delta_t(30.0) == pytest.approx(78.0, rel=2e-2)
        assert RB.forcing_needed_for_delta_t(60.0) == pytest.approx(191.0, rel=2e-2)

    def test_eq31_with_paper_T_e0_of_210_kelvin(self):
        sigma = 5.670374419e-8
        assert sigma * (240.0**4 - 210.0**4) == pytest.approx(78.0, rel=1e-2)
        assert RB.forcing_needed_exact(210.0, 270.0) == pytest.approx(191.0, rel=1e-2)

    def test_eq33_required_optical_depth(self):
        assert RB.required_ir_optical_depth(273.0) == pytest.approx(3.1, abs=0.1)
        assert RB.required_ir_optical_depth(250.0) == pytest.approx(2.0, abs=0.1)

    def test_eq32_is_inverse_of_eq33(self):
        for ts in (230.0, 250.0, 273.0, 290.0):
            assert RB.eddington_surface_temperature(RB.required_ir_optical_depth(ts)) == pytest.approx(ts)

    def test_eq57_albedo_lever(self):
        # Reducing A_B by 0.05 provides only ~7 W/m2
        assert RB.albedo_forcing(-0.05) == pytest.approx(7.0, abs=0.5)


class TestMirrors:
    def test_eq56_mirror_area_for_20_wm2(self):
        m = OrbitalMirror()                                                          # eta = 0.7
        assert m.system_efficiency == pytest.approx(0.7)
        assert m._mirror_area_for_global_forcing(PAPER_MARS, 20.0) == pytest.approx(7.0e12, rel=1e-2)

    def test_melt_class_mirror_area_is_9p6_times_larger(self):
        m = OrbitalMirror()
        ratio = (m._mirror_area_for_global_forcing(PAPER_MARS, RB.forcing_needed_exact(210.0, 270.0))
                 / m._mirror_area_for_global_forcing(PAPER_MARS, 20.0))
        assert ratio == pytest.approx(9.6, rel=3e-2)                                 # Sec. VI.D.2

    def test_mirror_uses_exact_not_linearised_forcing(self):
        res = OrbitalMirror().compute(MARS, target_delta_t_k=60.0)
        assert res.delta_forcing_wm2 == pytest.approx(191.0, rel=3e-2)
        assert res.delta_forcing_wm2 > 1.3 * RB.forcing_needed_linearized(60.0)


class TestParticles:
    def test_eq52_injection_mass_rate(self):
        assert 3e3 * 0.03 == pytest.approx(90.0)                                     # rho * Vdot, 30 L/s

    def test_eq53_column_mass_for_residence_times(self):
        area = 4 * math.pi * 3.3895e6**2
        for tau_days, sigma_mg in ((30, 1.6), (100, 5.4)):
            assert 90.0 * tau_days * 86400 / area * 1e6 == pytest.approx(sigma_mg, rel=3e-2)

    def test_nanorod_preset_reproduces_turyshev_benchmark(self):
        r = NanoparticleAerosol(AerosolMaterial.NANOROD_TURYSHEV).compute(MARS, target_delta_t_k=30.0)
        column_mg = r.total_mass_kg / (4 * math.pi * MARS.radius_m**2) * 1e6
        assert 1.6 <= column_mg <= 5.4                                               # within Eq. (53) range
        replenishment = r.total_mass_kg / (r.effect_duration_years * 365.25 * 86400)
        assert replenishment == pytest.approx(90.0, rel=0.15)                        # ~ 30 L/s of 3e3 kg/m3


class TestEndogenousCO2:
    def test_sublimation_energy_of_polar_deposit(self):
        # Sec. V.B / Table XII: sublimating ~2.3e16 kg of CO2 (L ~ 6e5 J/kg) takes ~1e22 J.
        from noarco.core.constants import YEAR_S
        from noarco.mechanisms import CO2Mobilization
        mech = CO2Mobilization(co2_inventory_kg=2.3e16, heat_source_efficiency=1.0)
        energy = mech._power_to_sublimate(2.3e16, 100.0) * 100.0 * YEAR_S
        assert energy == pytest.approx(1.4e22, rel=0.05)

    def test_dataset_polar_inventory_matches_6_mbar(self):
        assert MARS.co2_ice_kg == pytest.approx(0.006e5 * AtmosphericInventory(MARS).mass_per_pascal(), rel=0.05)

    def test_armstrong_limit_is_6p27_kpa_not_19p3(self):
        assert ARMSTRONG_LIMIT_PA == 6270.0
        assert MARS_ENDPOINTS["E3"].min_pressure_pa == pytest.approx(6.27e3)
        assert DesiredState.from_tier(HabitabilityTier.E3_ARMSTRONG).min_surface_pressure_pa == pytest.approx(6.27e3)

    def test_triple_point_thresholds(self):
        assert WATER_TRIPLE_POINT_PA == pytest.approx(611.657)
        assert MARS_ENDPOINTS["E1"].min_temperature_k == pytest.approx(273.16)

    def test_endogenous_co2_cannot_reach_E3_pi_M_below_one(self):
        # Example 1, Eq. (13): Pi_M ~ 20/62.7 = 0.32 (total pressure basis)
        req = endpoint_requirements(PAPER_MARS, DesiredState.from_tier(HabitabilityTier.E3_ARMSTRONG))
        pi = assess(req, available_inventory_kg=INV.mass_per_pascal() * 2000.0)
        # Paper: 20/62.7 = 0.32 on a total-pressure basis; here the increment over 6.1 mbar is used.
        assert pi.pi_mass == pytest.approx(2000.0 / (6270.0 - 610.0), rel=2e-2)
        assert 0.3 < pi.pi_mass < 0.4
        assert pi.feasible is False and pi.binding_constraint == "mass"


class TestEndpointRequirements:
    def test_table4_E3_throughput(self):
        req = endpoint_requirements(PAPER_MARS, DesiredState.from_tier(HabitabilityTier.E3_ARMSTRONG), 1000.0)
        assert req.mean_mass_flow_kg_s == pytest.approx(7.7e6, rel=0.1)              # Table IV (1e3 yr)

    def test_table4_E4a_oxygen_power_floor(self):
        t = DesiredState(habitability_tier=HabitabilityTier.CUSTOM, min_surface_pressure_pa=1.0e5,
                         min_o2_partial_pressure_pa=21_000.0)
        req = endpoint_requirements(PAPER_MARS, t, 1000.0)
        assert req.o2_mass_kg == pytest.approx(8.2e17, rel=2e-2)
        assert req.mean_power_w / 1e12 == pytest.approx(380.0, rel=5e-2)             # E_min over 1e3 yr
