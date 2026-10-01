"""Tests for noarco.assurance (V&V layer) and the regressions found while integrating SFSA."""

import math

import numpy as np
import pytest

from noarco.assurance import (
    Severity,
    UncertainValue,
    UnitError,
    assure,
    build_manifest,
    convert,
    parse_unit,
    propagate,
)
from noarco.assurance.invariants import check_mass_balance, check_planet_invariants
from noarco.assurance.uncertainty import coverage_factor
from noarco.assurance.units import convert_temperature
from noarco.assurance.validity import check_forcing_request, check_mechanism_result
from noarco.data.planets import ALL_SOLAR_BODIES, MARS, VENUS
from noarco.extratools import ExoplanetProfile, HabitatSpecification, ISRURequirement
from noarco.inventory.atmosphere import AtmosphericInventory
from noarco.radiative.balance import RadiativeBalance
from noarco.session import NOArCoSession
from noarco.uncertainty.montecarlo import MonteCarloEngine


# ---- regressions ---------------------------------------------------------------------------
class TestInventoryRegression:
    def test_delta_mass_returns_number(self):
        m = AtmosphericInventory(MARS).delta_mass_for_pressure(6000.0)
        assert m == pytest.approx((6000.0 - 610.0) * 4 * math.pi * MARS.radius_m**2 / MARS.gravity_ms2)
        assert 2.0e17 < m < 2.2e17

    def test_gap_analysis_and_session_feasibility_work(self):
        gap = AtmosphericInventory(MARS).co2_inventory_gap_kg(6000.0)
        assert gap["gap_kg"] > 0 and not gap["isru_feasible"]
        s = NOArCoSession.for_mars()
        assert s.co2_feasibility.value["feasible"] in (True, False)

    def test_unknown_species_and_bad_target_rejected(self):
        inv = AtmosphericInventory(MARS)
        with pytest.raises(ValueError):
            inv.delta_mass_for_pressure(6000.0, species="Unobtainium")
        with pytest.raises(ValueError):
            inv.delta_mass_for_pressure(-1.0)

    def test_pfc_mass_uses_planet_composition(self):
        assert AtmosphericInventory(VENUS).pfc_mass_for_forcing(10.0) != \
            AtmosphericInventory(MARS).pfc_mass_for_forcing(10.0)


class TestRadiativeExact:
    def test_exact_matches_sigma_t4(self):
        rb = RadiativeBalance(MARS)
        assert rb.forcing_needed_exact(210.0, 273.0) == pytest.approx(
            5.670374419e-8 * (273.0**4 - 210.0**4))

    def test_exact_reduces_to_linear_for_small_dt(self):
        rb = RadiativeBalance(MARS)
        t = rb.equilibrium_temperature_k()
        assert rb.forcing_needed_exact(t, t + 0.01) == pytest.approx(
            rb.forcing_needed_for_delta_t(0.01), rel=1e-3)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            RadiativeBalance(MARS).forcing_needed_exact(0.0, 273.0)


class TestMonteCarloInventoryPropagates:
    def test_inventory_uncertainty_widens_distribution(self):
        r = MonteCarloEngine(MARS, n_samples=4000, seed=1).propagate_temperature_rise(2.0e17)
        assert r.std_dev > 0 and r.percentile_10 < r.percentile_50 < r.percentile_90

    def test_reproducible_with_seed(self):
        a = MonteCarloEngine(MARS, seed=7).propagate_temperature_rise().mean
        b = MonteCarloEngine(MARS, seed=7).propagate_temperature_rise().mean
        assert a == b


class TestInputValidation:
    def test_habitat(self):
        with pytest.raises(ValueError):
            HabitatSpecification(volume_m3=-1.0, crew_size=4)
        with pytest.raises(ValueError):
            HabitatSpecification(volume_m3=10.0, crew_size=-1)
        with pytest.raises(ValueError):
            HabitatSpecification(volume_m3=10.0, crew_size=2, o2_fraction=0.6, n2_fraction=0.6)

    def test_exoplanet_and_isru(self):
        with pytest.raises(ValueError):
            ExoplanetProfile("x", 1.0, -1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            ISRURequirement(o2_kg_day=-5.0)


# ---- units ---------------------------------------------------------------------------------
class TestUnits:
    def test_pressure_conversions(self):
        assert convert(1.0, "bar", "Pa") == pytest.approx(1e5)
        assert convert(6.1, "mbar", "Pa") == pytest.approx(610.0)
        assert convert(1.0, "atm", "kPa") == pytest.approx(101.325)

    def test_compound_units(self):
        assert parse_unit("W m^-2 K^-4")[0] == parse_unit("kg s^-3 K^-4")[0]
        assert convert(3.89e15, "kg/mbar", "kg/Pa") == pytest.approx(3.89e13)
        assert convert(1.0, "kW/m^2", "W/m^2") == pytest.approx(1000.0)

    def test_unknown_unit_raises_not_silent(self):
        with pytest.raises(UnitError):
            parse_unit("furlongs")

    def test_mismatch_raises(self):
        with pytest.raises(UnitError):
            convert(1.0, "Pa", "J")

    def test_temperature(self):
        assert convert_temperature(0.0, "degC", "K") == pytest.approx(273.15)
        with pytest.raises(UnitError):
            convert_temperature(-300.0, "degC", "K")


# ---- uncertainty ---------------------------------------------------------------------------
class TestUncertainty:
    def test_coverage_factor(self):
        assert coverage_factor(0.95) == pytest.approx(1.959964, rel=1e-5)
        assert coverage_factor(0.6827) == pytest.approx(1.0, rel=1e-2)

    def test_product_relative_errors_add_in_quadrature(self):
        ins = {"a": UncertainValue(10.0, 0.1), "b": UncertainValue(5.0, 0.05)}
        r = propagate(lambda p: p["a"] * p["b"], ins)
        assert r.output.relative == pytest.approx(math.hypot(0.01, 0.01), rel=1e-4)

    def test_correlation_changes_result(self):
        ins = {"a": UncertainValue(1.0, 0.1), "b": UncertainValue(1.0, 0.1)}
        f = lambda p: p["a"] - p["b"]
        indep = propagate(f, ins).output.sigma
        corr = propagate(f, ins, {("a", "b"): 1.0}).output.sigma
        assert indep == pytest.approx(0.1 * math.sqrt(2), rel=1e-4)
        assert corr == pytest.approx(0.0, abs=1e-6)

    def test_mc_flags_nonlinearity(self):
        ins = {"x": UncertainValue(0.5, 0.45)}
        r = propagate(lambda p: math.exp(8 * p["x"]), ins, mc_samples=4000, seed=3)
        assert r.linearity_ok is False

    def test_mc_confirms_linear_model(self):
        ins = {"x": UncertainValue(10.0, 0.1)}
        assert propagate(lambda p: 3 * p["x"], ins, mc_samples=4000, seed=3).linearity_ok

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            UncertainValue(1.0, -0.1)
        with pytest.raises(ValueError):
            propagate(lambda p: p["a"], {"a": UncertainValue(1.0, 0.1)}, {("a", "a"): 2.0})
        with pytest.raises(KeyError):
            propagate(lambda p: p["a"], {"a": UncertainValue(1.0, 0.1)}, {("a", "z"): 0.5})


# ---- invariants / validity -----------------------------------------------------------------
class TestInvariantsAndValidity:
    def test_all_bodies_have_no_fail_invariants(self):
        for name, p in ALL_SOLAR_BODIES.items():
            assert all(f.severity < Severity.FAIL for f in check_planet_invariants(p)), name

    def test_inconsistent_gravity_fails(self):
        bad = MARS.model_copy(update={"gravity_ms2": 9.81})
        assert any(f.code == "INV-GRAVITY" and f.severity == Severity.FAIL
                   for f in check_planet_invariants(bad))

    def test_mass_balance(self):
        assert not check_mass_balance(10.0, 20.0, 10.0)
        assert check_mass_balance(-1.0, 1.0, 1.0)[0].severity == Severity.FAIL

    def test_co2_log_forcing_ratio_envelope(self):
        assert not check_forcing_request(MARS, 610.0, 6000.0)
        assert [f.code for f in check_forcing_request(MARS, 610.0, 610.0 * 5000)] == ["VAL-CO2-LOG-RATIO"]
        assert not check_forcing_request(MARS)                     # nothing requested, nothing to flag

    def test_mechanism_credibility_is_surfaced(self):
        from noarco.mechanisms import (
            CO2Mobilization,
            Electrolysis,
            NanoparticleAerosol,
            SilicaAerogel,
        )
        codes = lambda r: {f.code for f in check_mechanism_result(r)}
        assert codes(SilicaAerogel().compute(MARS)) == {"MECH-UPPER-BOUND", "MECH-REGIONAL"}
        assert "MECH-ENGINEERING" in codes(CO2Mobilization().compute(MARS, target_delta_t_k=1.0))
        assert codes(NanoparticleAerosol().compute(MARS, target_delta_t_k=5.0)) == {"MECH-SCALING"}
        assert "MECH-UNREACHABLE" in codes(NanoparticleAerosol().compute(MARS, target_delta_f_wm2=1e6))
        assert codes(Electrolysis().compute(MARS)) == set()         # first-principles, global


# ---- end to end ----------------------------------------------------------------------------
class TestAssure:
    def test_mars_passes_without_findings(self):
        r = assure(MARS)
        assert r.passed and r.verdict == "PASS" and set(r.outputs) >= {"equilibrium_temperature", "atmosphere_mass_per_pascal"}
        assert r.outputs["equilibrium_temperature"].output.nominal == pytest.approx(209.83, abs=0.05)

    def test_venus_gray_model_out_of_regime(self):
        r = assure(VENUS)
        assert not r.passed and any(f.code == "VAL-GRAY-TAU" for f in r.findings)

    def test_uncertainty_scales_with_epistemic_status(self):
        assert assure(MARS, rel_sigma=0.10, mc_samples=0).outputs["equilibrium_temperature"].output.sigma > \
            assure(MARS, rel_sigma=0.01, mc_samples=0).outputs["equilibrium_temperature"].output.sigma

    def test_deterministic_content_hash_and_tamper_evidence(self):
        a, b = assure(MARS, seed=5), assure(MARS, seed=5)
        assert a.manifest.content_hash == b.manifest.content_hash
        assert a.manifest.verify()
        a.manifest.inputs["rel_sigma"] = 0.0
        assert not a.manifest.verify()
        assert assure(MARS, seed=6).manifest.content_hash != b.manifest.content_hash

    def test_report_serialisation(self):
        r = NOArCoSession.for_mars().assure(mc_samples=200)
        d = r.to_dict()
        assert d["status"] in {"PASS", "CAUTION", "FAIL"} and "| Output |" in r.to_markdown()

    def test_manifest_standalone(self):
        m = build_manifest("x", {"a": np.float64(1.5)}, seed=1)
        assert m.verify() and m.versions.get("numpy")

    def test_negative_sigma_rejected(self):
        with pytest.raises(ValueError):
            assure(MARS, rel_sigma=-0.1)


def test_sfsa_bridge_optional():
    sfsa = pytest.importorskip("sfsa")  # noqa: F841
    from noarco.assurance.sfsa_bridge import open_sfsa_session
    s = open_sfsa_session(MARS)
    assert s.mre.assess_risk({"ir_optical_depth": 30.0}).verdict == "ABORT_INVALID_REGIME"


class TestPathwayInvariants:
    def test_generated_pathways_satisfy_all_invariants(self):
        from noarco.assurance.invariants import check_pathway_invariants
        from noarco.core.desired_state import DesiredState, HabitabilityTier
        from noarco.engines.pathfinder import PathEngine
        for t in (HabitabilityTier.E1_TRIPLE_POINT, HabitabilityTier.E3_ARMSTRONG, HabitabilityTier.E4_BREATHABLE):
            assert check_pathway_invariants(PathEngine.optimize_pathway(MARS, DesiredState.from_tier(t))) == []

    def test_violations_are_detected(self):
        from noarco.assurance.invariants import check_pathway_invariants
        from noarco.core.desired_state import DesiredState, HabitabilityTier
        from noarco.engines.pathfinder import PathEngine
        pw = PathEngine.optimize_pathway(MARS, DesiredState.from_tier(HabitabilityTier.E4_BREATHABLE))
        o2 = next(s for s in pw.steps if s.target_metric == "oxygen")
        o2.energy_joules = 0.5 * o2.total_mass_kg * 14.8e6            # below the Gibbs floor
        codes = {f.code for f in check_pathway_invariants(pw)}
        assert {"INV-GIBBS-FLOOR", "INV-SUM-ENERGY"} <= codes
        o2.time_years = -1.0
        assert "INV-STEP-SIGN" in {f.code for f in check_pathway_invariants(pw)}
