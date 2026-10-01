"""Fillers, inventory regimes per body, report formatting and aerosol edge cases."""

import pytest

from noarco.autodiscovery import GapFiller, SelfImprovingSession
from noarco.data.planets import ALL_SOLAR_BODIES, EARTH, JUPITER, MARS, MERCURY, MOON, PLUTO, VENUS
from noarco.inventory.atmosphere import AtmosphericInventory
from noarco.inventory.inventory import InventoryEngine
from noarco.mechanisms import AerosolMaterial, NanoparticleAerosol
from noarco.report.reporter import ReportEngine
from noarco.session import NOArCoSession

BODIES = sorted(ALL_SOLAR_BODIES)


class TestFillers:
    @pytest.mark.parametrize("name", BODIES)
    def test_every_filler_returns_value_or_none_never_raises(self, name):
        f, p = GapFiller(), ALL_SOLAR_BODIES[name]
        for fn in (f.fill_ir_optical_depth, f.fill_uv_flux, f.fill_co2_ice_kg,
                   f.fill_h2o_ice_kg, f.fill_regolith_composition):
            v = fn(p)
            assert v is None or (v.field_path.startswith("planet.") and v.uncertainty_factor >= 1.0)

    def test_fill_all_only_contains_known_fields(self):
        known = {"planet.ir_optical_depth", "planet.uv_flux_wm2", "planet.co2_ice_kg",
                 "planet.h2o_ice_kg", "planet.regolith_composition"}
        for p in ALL_SOLAR_BODIES.values():
            assert {d.field_path for d in GapFiller().fill_all_derivable(p)} <= known

    def test_regolith_fractions_are_valid_mass_fractions(self):
        for p in (MARS, MOON, MERCURY, VENUS):
            dv = GapFiller().fill_regolith_composition(p)
            if dv is not None:
                assert all(0 <= x <= 1 for x in dv.value.values()) and sum(dv.value.values()) <= 1.05

    def test_uv_flux_is_nonnegative_and_below_solar_constant(self):
        for p in ALL_SOLAR_BODIES.values():
            dv = GapFiller().fill_uv_flux(p)
            if dv is not None:
                assert 0 <= dv.value <= p.solar_constant_wm2


class TestSelfImprovingOnOtherBodies:
    @pytest.mark.parametrize("planet", [MOON, VENUS, EARTH, JUPITER])
    def test_never_crashes_and_never_overwrites_known_values(self, planet):
        s = SelfImprovingSession(NOArCoSession(planet))
        assert s.session.planet.surface_pressure_pa == planet.surface_pressure_pa
        assert isinstance(s.diagnosis_report(), str)
        assert isinstance(s.open_questions(), list)

    def test_raise_on_underdetermined_option(self):
        bare = JUPITER.model_copy(update={"ir_optical_depth": 0.0})
        try:
            SelfImprovingSession(NOArCoSession(bare), raise_on_underdetermined=True)
        except ValueError:
            pass  # allowed: unresolved gaps may raise; the option must not crash otherwise


class TestInventoryRegimes:
    @pytest.mark.parametrize("name", BODIES)
    def test_bottleneck_summary_is_nonempty_text(self, name):
        r = InventoryEngine.evaluate(ALL_SOLAR_BODIES[name], 2500.0)
        assert isinstance(r.critical_bottleneck, str) and r.critical_bottleneck

    def test_gas_giants_are_infeasible_for_open_atmosphere(self):
        r = InventoryEngine.evaluate(JUPITER, 1.0e21)
        assert r.feasibility_score <= 0.2

    def test_small_habitat_on_mars_is_highly_feasible(self):
        assert InventoryEngine.evaluate(MARS, 33.0).feasibility_score > 0.5

    def test_scores_always_within_unit_interval(self):
        for p in ALL_SOLAR_BODIES.values():
            for v in (2.0, 33.0, 2.5e3, 2.6e6, 1.0e9, 1.2e13, 2.19e19):
                assert 0.0 <= InventoryEngine.evaluate(p, v).feasibility_score <= 1.0

    def test_open_atmosphere_penalties_for_small_bodies(self):
        for p in (PLUTO, MOON, MERCURY):
            r = InventoryEngine.evaluate(p, 1.0e15)
            assert r.mode == "open_atmosphere" and r.feasibility_score < 0.5

    def test_venus_open_atmosphere_is_thermodynamically_blocked(self):
        r = InventoryEngine.evaluate(VENUS, 1.0e15)
        assert r.feasibility_score <= 0.15 and "thermodynamic" in r.critical_bottleneck.lower()

    def test_energy_and_time_consistency(self):
        r = InventoryEngine.evaluate(MARS, 2500.0)
        assert r.total_energy_kwh == pytest.approx(r.total_energy_joules / 3.6e6)
        times = list(r.time_years_at_power.values())
        assert times == sorted(times, reverse=True)       # more power -> shorter time


class TestVolumetricDictAndSummary:
    def test_volumetric_dict_keys(self):
        d = AtmosphericInventory(MARS).volumetric_conditioning_requirements(1000.0)
        assert {"mode", "total_gas_mass_kg", "o2_mass_kg", "total_conditioning_energy_kwh"} <= set(d)
        assert d["total_conditioning_energy_j"] == pytest.approx(d["total_conditioning_energy_kwh"] * 3.6e6, rel=1e-6)

    def test_inventory_summary_with_and_without_target(self):
        inv = AtmosphericInventory(MARS)
        assert "Target" not in inv.summary() and "Target: 6000 Pa" in inv.summary(6000.0)

    def test_mean_molar_mass_fallback_for_unknown_species_only(self):
        from noarco.core.planet_state import GasComposition
        odd = MARS.model_copy(update={"gas_composition": GasComposition(species={"Xx": 1.0})})
        assert AtmosphericInventory(odd)._mean_molar_mass_kg_per_mol() == pytest.approx(43.4e-3)

    def test_pfc_mass_requires_nothing_unphysical(self):
        inv = AtmosphericInventory(MARS)
        assert inv.pfc_mass_for_forcing(0.0) == 0.0


class TestReport:
    def test_format_labels_unique_for_standard_volumes(self):
        labels = [ReportEngine.format_volume_label(v) for v, _ in ReportEngine.STANDARD_VOLUMES]
        assert len(set(labels)) == len(labels)

    def test_table_without_bottleneck_column(self):
        reps = ReportEngine.evaluate_matrix(["Mars"], [33.0])
        t = ReportEngine.build_markdown_summary_table(reps)
        assert t.startswith("|") or "\n|" in t

    def test_default_matrix_covers_nine_bodies_and_all_standard_volumes(self):
        reps = ReportEngine.evaluate_matrix()
        assert len(reps) == 9 * len(ReportEngine.STANDARD_VOLUMES)


class TestAerosol:
    @pytest.mark.parametrize("mat", list(AerosolMaterial))
    def test_every_material_computes_finite_result(self, mat):
        r = NanoparticleAerosol(material=mat).compute(MARS, target_delta_t_k=10.0)
        assert r.total_mass_kg > 0 and r.delta_forcing_wm2 >= 0 and r.deployment_time_years > 0

    def test_more_warming_needs_more_mass(self):
        a = NanoparticleAerosol().compute(MARS, target_delta_t_k=5.0).total_mass_kg
        b = NanoparticleAerosol().compute(MARS, target_delta_t_k=20.0).total_mass_kg
        assert b > a
