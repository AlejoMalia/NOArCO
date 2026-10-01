"""
1000-case calculation battery
=============================

1000 distinct situations (different bodies, radii, gravities, enclosure sizes from 1 m^3 to 10^6 m^3,
pressures, temperatures, build times, flows, inventories). Every case is checked against a formula
*re-derived inside this file from first principles* (hydrostatics, ideal gas, Gibbs energy,
Stefan-Boltzmann, Eddington grey atmosphere, min-of-ratios), not against the code under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from noarco.core.desired_state import DesiredState
from noarco.data.planets import ALL_SOLAR_BODIES, MARS, MERCURY, MOON, PLUTO, VENUS
from noarco.extratools.habitat import HabitatDimensioner, HabitatSpecification
from noarco.inventory.atmosphere import AtmosphericInventory
from noarco.mechanisms import CO2Mobilization
from noarco.radiative.balance import RadiativeBalance
from noarco.radiative.literature import mars_co2_warming_k, mars_pressure_for_warming
from noarco.verdict import verdict

# Independent constants (CODATA / SI / NASA BVAD), typed here on purpose.
SIG = 5.670374419e-8
RGAS = 8.314462618
YEAR = 365.25 * 86400.0
DG_H2O = 237_129.0                       # J/mol, Gibbs energy of liquid-water formation
MU_O2 = 0.031998
E_O2 = 2.0 * DG_H2O / MU_O2              # J per kg O2 (2 H2O -> 2 H2 + O2), ~14.82 MJ/kg
RHO = {"O2": 0.031998, "N2": 0.028014, "Ar": 0.039948}

BODIES5 = [MARS, MOON, MERCURY, PLUTO, VENUS]
BODIES10 = list(ALL_SOLAR_BODIES.values())


def k_of(p):
    return 4.0 * math.pi * p.radius_m**2 / p.gravity_ms2


def te_of(p):
    return (p.solar_constant_wm2 * (1.0 - p.surface_albedo) / (4.0 * SIG)) ** 0.25


# A. hydrostatic kg per Pa over radius x gravity (100) -------------------------------------------
A = [(r, g) for r in np.geomspace(5e5, 7e7, 10) for g in np.geomspace(0.6, 25.0, 10)]


@pytest.mark.parametrize("radius,g", A)
def test_a_mass_per_pascal_hydrostatic(radius, g):
    p = MARS.model_copy(update={"radius_m": float(radius), "gravity_ms2": float(g)})
    inv = AtmosphericInventory(p)
    k = 4.0 * math.pi * radius**2 / g
    assert inv.mass_per_pascal() == pytest.approx(k, rel=1e-12)
    assert inv.mass_per_mbar() == pytest.approx(100.0 * k, rel=1e-12)
    assert inv.target_atmospheric_mass_kg(1234.5) == pytest.approx(1234.5 * k, rel=1e-12)


# B. regional gas mass A P / g over 10 areas (1 m2 .. 2 km2 .. 200 ha .. 2000 km2) x 4 P x 3 bodies = 120
AREAS = [1.0, 36.0, 1e2, 1e3, 1e4, 1e5, 2e6, 2e7, 2e8, 2e9]            # m2 (1e4 = 1 ha, 2e6 = 200 ha)
PRESS = [1e3, 2.5e4, 7e4, 1.013e5]
B = [(a, p, b) for a in AREAS for p in PRESS for b in (MARS, MOON, VENUS)]


@pytest.mark.parametrize("area,press,body", B)
def test_b_regional_gas_mass(area, press, body):
    inv = AtmosphericInventory(body)
    assert inv.regional_gas_mass_kg(area, press) == pytest.approx(area * press / body.gravity_ms2, rel=1e-12)


# C. pressurised area <-> mass round trip: 8 flows x 5 times x 2 pressures = 80 -------------------
C = [(f, t, p) for f in np.geomspace(1e-3, 1e6, 8) for t in (0.1, 1, 10, 100, 1000) for p in (2e4, 1.013e5)]


@pytest.mark.parametrize("flow,years,press", C)
def test_c_max_area_round_trip(flow, years, press):
    inv = AtmosphericInventory(MARS)
    area = inv.max_pressurized_area_m2(flow, years, press)
    assert area == pytest.approx(MARS.gravity_ms2 * flow * years * YEAR / press, rel=1e-12)
    assert inv.regional_gas_mass_kg(area, press) == pytest.approx(flow * years * YEAR, rel=1e-9)


# D. minimum O2 work: 50 masses (1 g .. 1e18 kg) ---------------------------------------------------
D = list(np.geomspace(1e-3, 1e18, 50))


@pytest.mark.parametrize("m", D)
def test_d_o2_energy_floor(m):
    assert AtmosphericInventory.minimum_o2_production_energy_j(float(m)) == pytest.approx(m * E_O2, rel=2e-4)
    assert E_O2 == pytest.approx(14.82e6, rel=1e-3)


# E. Eddington grey mapping: 10 bodies x 10 target temperatures = 100 -----------------------------
E = [(b, f) for b in BODIES10 for f in np.linspace(1.0, 2.2, 10)]


@pytest.mark.parametrize("body,factor", E)
def test_e_eddington_mapping(body, factor):
    rb = RadiativeBalance(body)
    te = te_of(body)
    ts = factor * te
    assert rb.equilibrium_temperature_k() == pytest.approx(te, rel=1e-9)
    tau = 4.0 / 3.0 * (ts / te) ** 4 - 2.0 / 3.0
    assert rb.required_ir_optical_depth(ts) == pytest.approx(tau, rel=1e-9, abs=1e-12)
    if tau >= 0:
        assert rb.eddington_surface_temperature(tau) == pytest.approx(ts, rel=1e-9)


# F. exact Stefan-Boltzmann forcing: 100 temperature pairs ---------------------------------------
_rng = np.random.default_rng(20260101)
F = [(float(a), float(b), float(fb)) for a, b, fb in zip(
    _rng.uniform(30, 900, 100), _rng.uniform(30, 900, 100), _rng.uniform(0.5, 3.0, 100), strict=True)]


@pytest.mark.parametrize("t1,t2,fb", F)
def test_f_exact_forcing(t1, t2, fb):
    rb = RadiativeBalance(MARS)
    expect = SIG * (t2**4 - t1**4) / fb
    assert rb.forcing_needed_exact(t1, t2, fb) == pytest.approx(expect, rel=1e-12, abs=1e-12)
    assert rb.forcing_needed_exact(t1, t2, fb) == pytest.approx(-rb.forcing_needed_exact(t2, t1, fb), rel=1e-12, abs=1e-12)


# G. habitat gas mass, O2/N2 split, crew rates: 10 volumes x 4 (P, T) x 2 bodies = 80 ------------
VOLS = [1.0, 6.0, 30.0, 150.0, 1e3, 1e4, 1e5, 1e6, 2.5e3, 40.0]
PT = [(101_325.0, 293.15), (70_300.0, 295.0), (50_000.0, 288.0), (20_000.0, 300.0)]
G = [(v, pt, b, i) for i, (v, pt, b) in enumerate((v, pt, b) for v in VOLS for pt in PT for b in (MARS, VENUS))]


@pytest.mark.parametrize("vol,pt,body,i", G)
def test_g_habitat_gas_and_crew(vol, pt, body, i):
    p, t = pt
    crew = i % 9
    rep = HabitatDimensioner.dimension(body, HabitatSpecification(volume_m3=vol, crew_size=crew, interior_pressure_pa=p,
                                                                 interior_temperature_k=t))
    mu = 0.21 * RHO["O2"] + 0.78 * RHO["N2"] + 0.01 * RHO["Ar"]
    mass = p * vol * mu / (RGAS * t)
    assert rep.total_gas_mass_kg == pytest.approx(mass, rel=2e-3)
    assert rep.o2_mass_kg == pytest.approx(mass * 0.21 * RHO["O2"] / mu, rel=2e-3)
    assert rep.o2_mass_kg + rep.n2_mass_kg <= rep.total_gas_mass_kg * (1 + 1e-9)
    assert rep.daily_o2_consumption_kg == pytest.approx(0.84 * crew, rel=1e-9, abs=1e-12)
    assert rep.daily_co2_scrubbed_kg == pytest.approx(1.00 * crew, rel=1e-9, abs=1e-12)
    assert rep.continuous_power_kw >= 0 and rep.leak_rate_annual_kg >= 0


# H. verdict: mass axis from first principles: 5 bodies x 10 inventories x 2 deltas = 100 ----------
H = [(b, s, d) for b in BODIES5 for s in np.geomspace(0.05, 20.0, 10) for d in (2e3, 5e4)]


@pytest.mark.parametrize("body,scale,dp", H)
def test_h_verdict_mass_axis(body, scale, dp):
    target = DesiredState(min_surface_pressure_pa=body.surface_pressure_pa + dp)
    req_mass = k_of(body) * dp
    avail = scale * req_mass
    v = verdict(body, target, available_inventory_kg=avail)
    assert v.pi["mass"] == pytest.approx(scale, rel=1e-9)
    assert v.requirements["atmosphere_mass_kg"] == pytest.approx(req_mass, rel=1e-9)
    assert v.viable is (scale >= 1.0)
    assert v.limiting_axis == "mass"


# I. conjunctive Pi_min = min(ratios) with random availabilities: 100 -----------------------------
_r = np.random.default_rng(7)
I = [(float(m), float(f), float(w), float(t), float(dp)) for m, f, w, t, dp in zip(
    _r.lognormal(0, 1.5, 100), _r.lognormal(0, 1.5, 100), _r.lognormal(0, 1.5, 100),
    _r.choice([1.0, 10.0, 100.0, 1000.0], 100), _r.uniform(5e3, 8e4, 100), strict=True)]


@pytest.mark.parametrize("sm,sf,sw,years,dp", I)
def test_i_conjunctive_min(sm, sf, sw, years, dp):
    target = DesiredState(min_surface_pressure_pa=MARS.surface_pressure_pa + dp, min_o2_partial_pressure_pa=3000.0)
    x_o2 = 0.00174 * MARS.surface_pressure_pa
    k = k_of(MARS)
    mass = k * dp
    o2 = k * max(3000.0 - x_o2, 0.0)
    flow = mass / (years * YEAR)
    power = o2 * E_O2 / (years * YEAR)
    v = verdict(MARS, target, build_time_years=years, available_inventory_kg=sm * mass,
                available_mass_flow_kg_s=sf * flow, available_power_w=sw * power)
    ratios = {"mass": sm, "throughput": sf, "power": sw}
    assert v.pi["mass"] == pytest.approx(sm, rel=2e-3)
    assert v.pi["throughput"] == pytest.approx(sf, rel=2e-3)
    assert v.pi["power"] == pytest.approx(sw, rel=2e-3)
    assert v.pi_min == pytest.approx(min(ratios.values()), rel=2e-3)
    assert v.limiting_axis == min(ratios, key=ratios.get)
    assert v.viable is (min(ratios.values()) >= 1.0)


# J. reproducibility and hash sensitivity: 20 -----------------------------------------------------
J = list(np.geomspace(1e12, 1e19, 20))


@pytest.mark.parametrize("inv", J)
def test_j_reproducible_and_sensitive(inv):
    t = DesiredState.from_tier(__import__("noarco.core.desired_state", fromlist=["x"]).HabitabilityTier.E3_ARMSTRONG)
    a, b = verdict(MARS, t, available_inventory_kg=float(inv)), verdict(MARS, t, available_inventory_kg=float(inv))
    c = verdict(MARS, t, available_inventory_kg=float(inv) * 1.001)
    assert a.input_hash == b.input_hash and a.to_json() == b.to_json()
    assert a.input_hash != c.input_hash


# K. CO2 mobilisation power and throughput: 10 inventories x 5 times x 2 efficiencies = 100 -------
K = [(m, t, e) for m in np.geomspace(1e12, 7.78e16, 10) for t in (10.0, 50.0, 100.0, 200.0, 1000.0) for e in (0.7, 1.0)]


@pytest.mark.parametrize("inv,years,eff", K)
def test_k_co2_power_and_throughput(inv, years, eff):
    r = CO2Mobilization(co2_inventory_kg=float(inv), heat_source_efficiency=eff).compute(MARS, target_delta_t_k=1e3,
                                                                                         deployment_years=years)
    assert r.total_mass_kg == pytest.approx(inv, rel=1e-12)
    assert r.power_w == pytest.approx(inv * 5.9e5 / (years * YEAR * eff), rel=1e-9)
    assert r.throughput_kg_s == pytest.approx(inv / (years * YEAR), rel=1e-9)


# L. literature curve: monotone, bounded, inverse round trip: 50 ------------------------------------
L = list(np.geomspace(650.0, 1.0e5, 50))


@pytest.mark.parametrize("p", L)
def test_l_literature_curve(p):
    lo, mid, hi = mars_co2_warming_k(float(p))
    assert 0.0 <= lo <= mid <= hi <= 70.0 + 1e-9
    assert mars_co2_warming_k(float(p) * 1.05)[1] >= mid if p * 1.05 <= 1.5e5 else True
    assert mars_pressure_for_warming(mid) == pytest.approx(p, rel=1e-4) if mid > 0 else True
