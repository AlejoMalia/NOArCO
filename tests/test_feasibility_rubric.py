"""Requirement bookkeeping and the conjunctive feasibility rubric (Turyshev 2026, Eq. 9-11)."""

import math

import pytest

from noarco.core.constants import YEAR_S
from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.data.planets import MARS
from noarco.feasibility import assess, endpoint_requirements
from noarco.inventory.atmosphere import AtmosphericInventory
from noarco.radiative.balance import RadiativeBalance

K = AtmosphericInventory(MARS).mass_per_pascal()
E1, E2, E3, E4 = (DesiredState.from_tier(t) for t in (
    HabitabilityTier.E1_TRIPLE_POINT, HabitabilityTier.E2_PROTECTED,
    HabitabilityTier.E3_ARMSTRONG, HabitabilityTier.E4_BREATHABLE))


class TestRequirements:
    def test_no_change_requires_nothing(self):
        t = DesiredState(habitability_tier=HabitabilityTier.CUSTOM, min_surface_pressure_pa=MARS.surface_pressure_pa)
        r = endpoint_requirements(MARS, t)
        assert r.atmosphere_mass_kg == 0 and r.o2_mass_kg == 0 and r.delta_forcing_toa_wm2 == 0
        assert r.mean_mass_flow_kg_s == 0 and r.mean_power_w == 0

    def test_pressure_mass_and_flow(self):
        r = endpoint_requirements(MARS, E3, build_time_years=500.0)
        assert r.atmosphere_mass_kg == pytest.approx(K * (6270.0 - 610.0))
        assert r.mean_mass_flow_kg_s == pytest.approx(r.atmosphere_mass_kg / (500.0 * YEAR_S))

    def test_temperature_requirements_are_consistent(self):
        rb = RadiativeBalance(MARS)
        r = endpoint_requirements(MARS, E3)
        assert r.delta_forcing_toa_wm2 == pytest.approx(rb.forcing_needed_for_surface_temperature(250.0))
        assert r.delta_tau_ir == pytest.approx(rb.required_ir_optical_depth(250.0) - rb.required_ir_optical_depth(210.0))

    def test_cooling_targets_need_no_forcing(self):
        t = DesiredState(habitability_tier=HabitabilityTier.CUSTOM, target_mean_temperature_k=150.0)
        assert endpoint_requirements(MARS, t).delta_forcing_toa_wm2 == 0.0

    def test_oxygen_energy_is_the_gibbs_minimum(self):
        r = endpoint_requirements(MARS, E4)
        p_o2_now = MARS.gas_composition.mole_fraction("O2") * MARS.surface_pressure_pa
        assert r.o2_mass_kg == pytest.approx(K * (16_000.0 - p_o2_now))
        assert r.o2_min_energy_j == pytest.approx(r.o2_mass_kg * 14.82e6, rel=1e-3)
        assert r.buffer_gas_mass_kg == pytest.approx(r.atmosphere_mass_kg - r.o2_mass_kg)

    def test_enclosed_target_has_no_global_mass_requirement(self):
        r = endpoint_requirements(MARS, E2)
        assert r.scope == "enclosed" and r.atmosphere_mass_kg == 0.0 and r.notes

    @pytest.mark.parametrize("bad", [0.0, -5.0, math.inf, math.nan])
    def test_invalid_build_time(self, bad):
        with pytest.raises(ValueError):
            endpoint_requirements(MARS, E3, build_time_years=bad)


class TestRubric:
    def test_no_availability_no_verdict(self):
        pi = assess(endpoint_requirements(MARS, E3))
        assert pi.pi_min is None and pi.feasible is None and pi.binding_constraint is None and pi.values == {}

    def test_ratios(self):
        r = endpoint_requirements(MARS, E3)
        pi = assess(r, available_inventory_kg=r.atmosphere_mass_kg / 2, available_mass_flow_kg_s=r.mean_mass_flow_kg_s * 4)
        assert pi.pi_mass == pytest.approx(0.5) and pi.pi_throughput == pytest.approx(4.0)
        assert pi.pi_min == pytest.approx(0.5) and pi.binding_constraint == "mass" and pi.feasible is False

    def test_conjunctive_minimum_and_feasibility(self):
        r = endpoint_requirements(MARS, E3)
        ok = assess(r, available_inventory_kg=2 * r.atmosphere_mass_kg, available_mass_flow_kg_s=3 * r.mean_mass_flow_kg_s,
                    available_forcing_wm2=1.5 * r.delta_forcing_toa_wm2)
        assert ok.feasible is True and ok.pi_min == pytest.approx(1.5) and ok.binding_constraint == "forcing"

    def test_forcing_uses_the_better_of_the_two_routes(self):
        r = endpoint_requirements(MARS, E3)
        pi = assess(r, available_forcing_wm2=0.1 * r.delta_forcing_toa_wm2, available_delta_tau_ir=2 * r.delta_tau_ir)
        assert pi.pi_forcing == pytest.approx(2.0)

    def test_nothing_required_gives_infinite_ratio(self):
        r = endpoint_requirements(MARS, E1)
        pi = assess(r, available_power_w=1e9)                                # E1 needs no O2 -> no power requirement
        assert math.isinf(pi.pi_power)

    def test_stability_ratio_requires_both_rates(self):
        r = endpoint_requirements(MARS, E3)
        assert assess(r, replenishment_kg_s=10.0).pi_stability is None
        assert assess(r, replenishment_kg_s=10.0, loss_kg_s=5.0).pi_stability == pytest.approx(2.0)
        assert math.isinf(assess(r, replenishment_kg_s=1.0, loss_kg_s=0.0).pi_stability)

    def test_paper_conclusions_on_mars(self):
        """Endogenous CO2 cannot reach E3; breathable E4 is power- and inventory-starved."""
        e3 = assess(endpoint_requirements(MARS, E3), available_inventory_kg=7.78e16)
        e4 = assess(endpoint_requirements(MARS, E4), available_inventory_kg=7.78e16, available_power_w=1e13)
        assert e3.feasible is False and e3.pi_mass < 0.5
        assert e4.feasible is False and e4.pi_min < 0.05
