"""
tests.test_extratools
=====================
Unit tests for the NOArCO-Extratools suite:
- NOArCO-Habitat
- NOArCO-ISRU
- NOArCO-Logistics
- NOArCO-Exoplanet
- NOArCO-Agro
- NOArCO-CLI
"""

import pytest

from noarco.data.planets import MARS, VENUS
from noarco.extratools import (
    AstroAgroPlanner,
    ExoplanetClassifier,
    ExoplanetProfile,
    HabitatDimensioner,
    HabitatSpecification,
    ISRUEvaluator,
    ISRURequirement,
    ISRUSiteProfile,
    SoilConditioningSpec,
    VolatileLogisticsPlanner,
    run_exoplanet_quick_demo,
    run_habitat_quick_demo,
)


class TestNOArCOExtratools:
    def test_habitat_dimensioner_mars(self):
        spec = HabitatSpecification(volume_m3=2500.0, crew_size=6, mission_days=365)
        report = HabitatDimensioner.dimension(MARS, spec)

        assert report.volume_m3 == 2500.0
        assert report.crew_size == 6
        assert report.total_gas_mass_kg > 2000.0  # ~3000 kg a 1 atm
        assert report.o2_mass_kg > 500.0
        assert report.daily_o2_consumption_kg == pytest.approx(6 * 0.84, 1e-3)
        assert report.continuous_power_kw > 0.0
        assert report.displacement_work_kwh == 0.0  # Interior pressure > Martian exterior

    def test_habitat_dimensioner_venus_displacement_work(self):
        spec = HabitatSpecification(volume_m3=500.0, crew_size=4)
        report = HabitatDimensioner.dimension(VENUS, spec)

        # On Venus the exterior pressure is 92 bar, so purging or maintaining the habitat requires work against the exterior
        assert report.displacement_work_kwh > 0.0
        assert any("Hyperbaric" in r for r in report.recommendations)

    def test_isru_evaluator_mars_site(self):
        evaluator = ISRUEvaluator()
        jezero_site = ISRUSiteProfile(
            name="Jezero Crater Delta",
            planet_name="Mars",
            temperature_k=210.0,
            pressure_pa=610.0,
            mineralogy={"SiO2": 0.45, "Fe2O3": 0.18, "H2O": 0.03, "Cl": 0.005},
        )
        req = ISRURequirement(o2_kg_day=50.0, water_kg_day=20.0, metal_kg_day=10.0)
        report = evaluator.evaluate_site(jezero_site, req)

        assert report.daily_o2_kg == 50.0
        assert report.daily_regolith_mined_kg > 50.0
        assert report.reactor_energy_kwh_day > 0.0
        assert report.continuous_power_kw > 0.0
        assert report.selected_mechanism_id != ""

    def test_volatile_logistics_mars_nitrogen(self):
        # Nitrogen deficit on Mars: ~3e18 kg
        report = VolatileLogisticsPlanner.plan_import(
            target_planet=MARS,
            species="N2",
            required_net_mass_kg=1.0e16,  # 10^16 kg scale test
            source_key="Kuiper_Belt_Comets",
        )

        assert report.target_species == "N2"
        assert report.delta_v_km_s > 0.0
        assert report.bodies_required_count > 0
        assert report.atmospheric_retention_fraction > 0.50
        assert report.kinetic_energy_impact_joules > 0.0

    def test_exoplanet_classifier_trappist_1e(self):
        trappist_1e = ExoplanetProfile(
            name="TRAPPIST-1e",
            mass_earth=0.692,
            radius_earth=0.920,
            semi_major_axis_au=0.029,
            stellar_luminosity_solar=0.000553,
            stellar_teff_k=2566.0,
            albedo=0.25,
            has_magnetic_field=True,
        )
        report = ExoplanetClassifier.classify(trappist_1e)

        assert report.name == "TRAPPIST-1e"
        assert report.surface_gravity_g > 0.70  # ~0.82 g
        assert report.escape_velocity_km_s > 9.0
        assert report.is_in_habitable_zone
        assert report.tfi_score_pct > 70.0
        assert "N2" in report.jeans_parameters
        assert report.jeans_parameters["N2"] > 6.0  # Stable nitrogen retention

    def test_astro_agro_soil_conversion(self):
        spec = SoilConditioningSpec(
            greenhouse_area_m2=500.0,
            soil_depth_m=0.30,
            perchlorate_fraction=0.005,
        )
        report = AstroAgroPlanner.plan_soil_conversion(spec)

        assert report.total_soil_volume_m3 == 150.0
        assert report.total_soil_mass_kg == 150.0 * 1500.0
        assert report.perchlorate_mass_to_remove_kg > 0.0
        assert report.oxygen_liberated_from_detox_kg > 0.0
        assert report.detoxification_energy_kwh > 0.0
        assert report.water_required_hydration_kg > 0.0
        assert report.cyanobacteria_conditioning_days > 30.0
        assert report.annual_food_yield_kg == 500.0 * 25.0

    def test_cli_demos_execution(self, capsys):
        run_habitat_quick_demo("mars", crew=4, volume_m3=1000)
        captured = capsys.readouterr()
        assert "NOArCO-HABITAT" in captured.out

        run_exoplanet_quick_demo("TRAPPIST-1e")
        captured = capsys.readouterr()
        assert "NOArCO-EXOPLANET" in captured.out
        assert "TERRAFORMING FEASIBILITY INDEX" in captured.out
