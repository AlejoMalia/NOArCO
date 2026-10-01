"""
tests.test_analyzers_and_engines
================================
Unit tests for the new AIR + SOIL analyzers and engines:
- StateEngine
- AtmosphereEngine
- SoilEngine
- InventoryEngine
- ImpactEngine
- PathEngine
- TimelineEngine
- ReportEngine
"""

from noarco.analyzers.atmosphere import AtmosphereEngine
from noarco.analyzers.soil import SoilEngine
from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.data.planets import JUPITER, MARS, PLUTO, VENUS
from noarco.engines.pathfinder import PathEngine
from noarco.engines.timeline import TimelineEngine
from noarco.inventory.inventory import InventoryEngine
from noarco.report.reporter import ReportEngine


class TestAtmosphereEngine:
    def test_mars_atmosphere_analysis(self):
        analysis = AtmosphereEngine.analyze(MARS)
        assert analysis.body_name == "Mars"
        assert analysis.surface_pressure_pa == 610.0
        assert analysis.scale_height_m > 10_000.0  # ~11 km
        assert analysis.total_atmospheric_mass_kg > 2.0e16
        assert bool(analysis.jeans_retention_stable) is True

    def test_closed_volume_distinction(self):
        # 2 m3 closed habitat on Mars
        req_mars = AtmosphereEngine.compute_volume_conditioning(MARS, volume_m3=2.0)
        assert req_mars.mode == "closed_volume"
        assert req_mars.total_gas_mass_kg > 2.0
        assert req_mars.o2_mass_kg > 0.4
        assert req_mars.net_mass_delta_kg > 0

        # 2 m3 on Venus: Ambient CO2 is ~92 bar (~130 kg in 2 m3)
        req_venus = AtmosphereEngine.compute_volume_conditioning(VENUS, volume_m3=2.0)
        assert req_venus.mode == "closed_volume"
        # Must purge ambient dense CO2, resulting in massive negative delta vs ambient mass
        assert req_venus.net_mass_delta_kg < 0
        assert req_venus.compression_work_joules > 1.0e7

    def test_open_atmosphere_venus_removal(self):
        # Large scale on Venus requires massive removal of ~4.7e20 kg CO2
        req = AtmosphereEngine.compute_volume_conditioning(VENUS, volume_m3=1.08e21)
        assert req.mode == "open_atmosphere"
        assert req.net_mass_delta_kg < -1.0e20
        assert "REMOVAL" in req.regime_description

    def test_open_atmosphere_gas_giant_infeasible(self):
        # Jupiter open atmosphere must be flagged
        req = AtmosphereEngine.compute_volume_conditioning(JUPITER, volume_m3=1.08e21)
        assert req.mode == "open_atmosphere"
        assert "INFEASIBLE" in req.regime_description

    def test_pluto_jeans_escape_warning(self):
        req = AtmosphereEngine.compute_volume_conditioning(PLUTO, volume_m3=1.08e21)
        assert req.jeans_blowoff_warning is True


class TestSoilEngine:
    def test_mars_regolith_analysis(self):
        soil = SoilEngine.analyze(MARS)
        assert soil.has_solid_surface is True
        assert soil.extractable_o2_from_regolith_pct > 30.0
        assert "Fe" in soil.elemental_mass_fractions
        assert "Si" in soil.elemental_mass_fractions
        assert soil.perchlorate_toxicity_present is True

    def test_oxygen_extraction_water_vs_mre(self):
        # Extract 100 kg O2 from Mars water ice
        plan_water = SoilEngine.compute_oxygen_extraction(MARS, o2_target_kg=100.0, preferred_source="water_ice")
        assert plan_water.primary_source == "water_ice"
        assert plan_water.water_ice_consumed_kg > 100.0
        assert plan_water.extraction_energy_joules > 0

        # Extract 100 kg O2 from Mars regolith via Molten Regolith Electrolysis
        plan_mre = SoilEngine.compute_oxygen_extraction(MARS, o2_target_kg=100.0, preferred_source="molten_regolith_electrolysis")
        assert plan_mre.primary_source == "molten_regolith_electrolysis"
        assert plan_mre.regolith_mined_kg > 200.0
        assert plan_mre.metals_coproduced_kg["Fe"] > 0
        assert plan_mre.metals_coproduced_kg["Si"] > 0
        # MRE requires higher specific energy than water ice electrolysis
        assert plan_mre.extraction_energy_joules > plan_water.extraction_energy_joules


class TestInventoryAndPathEngines:
    def test_inventory_evaluate(self):
        rep = InventoryEngine.evaluate(MARS, volume_m3=33.0)
        assert rep.volume_m3 == 33.0
        assert rep.o2_needed_kg > 0
        assert rep.total_energy_kwh > 0
        assert len(rep.time_years_at_power) > 0

    def test_pathfinder_and_timeline(self):
        desired = DesiredState(
            habitability_tier=HabitabilityTier.E1_TRIPLE_POINT,
            min_surface_pressure_pa=10_000.0,
            target_mean_temperature_k=240.0,
        )
        pathway = PathEngine.optimize_pathway(MARS, desired, available_power_w=1.0e10)
        assert pathway.body_name == "Mars"
        assert len(pathway.steps) >= 2
        assert pathway.total_energy_joules > 0

        schedule = TimelineEngine.generate_schedule(pathway, power_installed_gw=10.0)
        assert len(schedule.milestones) == len(pathway.steps)


class TestReportEngine:
    def test_evaluate_matrix_and_table(self):
        # Test subset of matrix
        reports = ReportEngine.evaluate_matrix(bodies=["Mars", "Venus"], volumes=[2.0, 1.0e9])
        assert len(reports) == 4
        table = ReportEngine.build_markdown_summary_table(reports)
        assert "| Planet / Body |" in table
        assert "Mars" in table
        assert "Venus" in table
