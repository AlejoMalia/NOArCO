# NOArCO

<p align="center">
  <img src="docs/banner.png" alt="NOArCO banner" width="100%">
</p>

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/tests-1799%20passed%20%7C%201%20skipped-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen.svg)](#testing-and-quality)
[![Lint](https://img.shields.io/badge/ruff-passing-brightgreen.svg)](pyproject.toml)
[![Types](https://img.shields.io/badge/mypy-passing-brightgreen.svg)](pyproject.toml)
[![Engines](https://img.shields.io/badge/engines-14%20(9%20core%20%2B%205%20extratools)-blue.svg)](#engine-catalog)
[![V&V](https://img.shields.io/badge/V%26V-assurance%20layer-orange.svg)](#verification--validation-layer-noarcoassurance)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)

**NOArCO** is an open-source Python framework for **pre-simulation feasibility screening** of planetary-engineering, paraterraforming and habitat/ISRU trades. Given declared body conditions, target states, resource budgets and build time, it returns a conjunctive viability verdict and the limiting axis, without claiming to replace climate models or flight software.

> **Honest accuracy statement.** Inventory and throughput anchors match the reference paper to within a few percent. Absolute greenhouse warming under the simple radiative model remains biased low versus GCMs (documented); those ΔT figures are not used as hard feasibility gates unless a literature-calibrated radiative mode is selected (`radiative="literature_calibrated"`, Mars only).

**What it gives you**

* **Fast feasibility screening:** one call returns a conjunctive viability verdict (`Π_min = min(available / required)`) and the **limiting axis** (mass, power, throughput, forcing or stability), with headroom, shortfall and the build time that would clear it.
* **ISRU, habitat and logistics in one budget:** stoichiometric mass ledgers, thermodynamic energy floors, pressure-shell mass, Hohmann delta-v and a habitat + ISRU conjunctive budget (`size_base`).
* **Traceable by construction:** every constant carries a source and class, every result carries an input hash, a constants hash and a code hash, and every record is citeable JSON that `scripts/verify_record.py` can replay and check.

The framework replaces heavy numerical simulation with analytical stoichiometric and thermodynamic closed forms wherever they are valid (the **TRIADA** protocol), and makes the validity of each closed form explicit and checkable (see the [V&V layer](#verification--validation-layer-noarcoassurance)). A complete planetary inventory evaluation takes tens of microseconds on commodity hardware (see [Performance](#performance)).

> **Read first:** [What NOArCO is - and is not](docs/WHAT_THIS_IS.md) (not a GCM, not flight software, no clouds) · [Constant provenance](docs/CONSTANTS.md) · [Model credibility](docs/MODEL_CREDIBILITY.md).

> **Credibility.** Every model is classified as verified, closed-form, engineering estimate or heuristic in [`docs/MODEL_CREDIBILITY.md`](docs/MODEL_CREDIBILITY.md). Results carry `model_class` and warnings; nothing here is flight-qualified.

> **Scope and status.** NOArCO is an early-stage (0.1.0) research tool for *first-order feasibility and trade-study analysis*. It is not a flight-qualified simulator and is not a substitute for high-fidelity climate or mission models (LMD-PCM, MarsWRF, GMAT, …). Every model carries an explicit validity envelope; results outside it are reported as findings, not returned silently.

---

## The product in four lines

```python
from noarco.verdict import verdict
v = verdict("Mars", DesiredState.from_tier(HabitabilityTier.E3_ARMSTRONG), available_inventory_kg=7.78e16, available_power_w=1e12)
print(v.cite())   # NOArCO 0.1.0 verdict ee6aea6aa0d96... : Mars -> E3_armstrong: not viable, limited by mass.
print(v.to_json())  # schema noarco-result/1.1: verdict, Pi per axis, requirements, inputs, input_hash, model_class, constants_hash
```

* **One conjunctive verdict** (`noarco.verdict`): viable / not viable / undetermined and the **limiting axis** (`mass`, `power`, `forcing`, `throughput`, `stability`); an axis is evaluated only if you supply its availability, nothing is assumed.
* **Closed reference case** (`python scripts/reproduce_reference.py`): prints published value vs NOArCO for 22 anchor cases (kg/mbar, forcing, minimum O₂ work, mirror area, …) and exits non-zero on any mismatch; the same cases are green tests.
* **Provenance of every constant** ([`docs/CONSTANTS.md`](docs/CONSTANTS.md)): value → unit → source → `exact / measured / derived / literature / calibrated / estimated`; `verify_registry()` checks the table against the code in CI.
* **Citeable result**: a JSON record that is a pure function of its inputs (hash, version, model class, constants hash; no timestamps).

## Validation against published GCM results

Compared with the MarsWRF 3-D steady states of Richardson et al. (2025, Table S3) and the climate-model statements quoted by Jakosky & Edwards (2018). Calibrations are labelled as such; out-of-sample tests use leave-one-out; every discrepancy is listed.

| Case | Source | Quantity | GCM / published | NOArCO | Rel. error | Kind | Status | dT abs. gate? |
|---|---|---|---|---|---|---|---|---|
| RICH25-KAPPA-Al | Richardson et al. 2025 (MarsWRF, Table S3) | Al rods: fitted kappa vs preset in NOArCO | 3.151e+04 m2/kg | 3.15e+04 | 0.0% | calibration | PASS | no |
| RICH25-LOO-Al | Richardson et al. 2025 (MarsWRF, Table S3) | Al rods: leave-one-out warming, median |rel. error| (n=7) | 0 relative | 0.1707 | 17.1% | out_of_sample | PASS | no |
| RICH25-KAPPA-C | Richardson et al. 2025 (MarsWRF, Table S3) | graphene: fitted kappa vs preset in NOArCO | 7.084e+04 m2/kg | 7.08e+04 | 0.1% | calibration | PASS | no |
| RICH25-LOO-C | Richardson et al. 2025 (MarsWRF, Table S3) | graphene: leave-one-out warming, median |rel. error| (n=10) | 0 relative | 0.102 | 10.2% | out_of_sample | PASS | no |
| RICH25-LIFETIME | Richardson et al. 2025 (MarsWRF, Table S3) | Al-rod residence time from the column/release balance (mean of runs) | 1.155 yr | 1.15 | 0.5% | calibration | PASS | no |
| RICH25-VS-TURYSHEV | Richardson et al. 2025 (MarsWRF, Table S3) vs Turyshev 2026 Eq. 53 | column for ~+30 K: GCM vs Turyshev scaling | 3.23e-05 kg/m2 | 3.464e-06 | 89.3% | documented_discrepancy | DISCREPANCY | no |
| RICH25-AL160 | Richardson et al. 2025 (MarsWRF, Table S3) | Al 160 nm particle (Ansari 2024 geometry), 45 L/s: warming | 15.76 K | 31.84 | 102.0% | documented_discrepancy | DISCREPANCY | no |
| JE18-20MBAR | Jakosky & Edwards 2018 (climate-model statements) | warming from ~20 mbar CO2 (published: < 10 K) | 10 K | 6.676 | 0.0% | bound | PASS | no |
| JE18-1BAR | Jakosky & Edwards 2018 (climate-model statements) | warming from ~1 bar CO2 (published: ~60 K, close to melting), simple mode | 60 K | 29.21 | 51.3% | documented_discrepancy | KNOWN_LIMITATION | no |
| JE18-1BAR-CAL | Jakosky & Edwards 2018 (climate-model statements) | warming from ~1 bar CO2, literature_calibrated mode | 60 K | 60 | 0.0% | calibration | PASS | yes |
| RICH25-GH6MBAR | Richardson et al. 2025 (MarsWRF, Table S3) intro | present greenhouse effect of the 6 mbar atmosphere (published: ~+5 K) | 5 K | 1.299 | 74.0% | documented_discrepancy | KNOWN_LIMITATION | no |

7 pass, 2 documented discrepancies, 2 known radiative limitations (separate gate), 0 fail.

The radiative cases (`gate = radiative`) are a **separate gate** from the inventory/throughput anchors: they are reported as `KNOWN_LIMITATION` (`use_for_Π_min_mass = yes`, `use_for_ΔT_absolute = no`), not as a failed or passed GCM test; the present +5 K greenhouse is a diagnostic only.

Findings that matter: (1) the grey-mapping with a single fitted mass absorption coefficient reproduces the GCM warming of held-out runs to a median **17 %** (Al rods) and **10 %** (graphene), but only for the particle it was fitted to (a different geometry is over-predicted ×2); (2) **Turyshev's aerosol scaling needs ~9× less column than the GCM**, so the default `NANOROD` preset follows the GCM and the paper's scaling is kept as `NANOROD_TURYSHEV`; (3) the CO₂ log-law is within the published bound at 20 mbar but **under-predicts the 1-bar case by ~2×** (no water-vapour feedback; the `literature_calibrated` mode states the published ~60 K with its source instead) and the 1-layer greenhouse under-states the present +5 K. These are reflected in the structural-error factors of `noarco.projection`.

---

## Six things this engine does

1. **Any world, not only planets.** Describe a planet, moon, asteroid, exoplanet, habitat or lab chamber by a *list of conditions* (`{"gravity_ms2": 1.6, "pressure_pa": 200, ...}`), a solar-system name, an `ExoplanetProfile` or a `PlanetState`. Missing quantities are derived from physics (`g = GM/R²`, `S = L/4πd²`, `T = T_eq`) or recorded as explicit assumptions ([`noarco.bodies`](noarco/bodies.py)).
2. **Projection with a stated confidence.** `project()` returns the probability that a transition is feasible and 95 % (or any level) intervals for mass, energy, time, warming and aerosol demand, in ~10 ms, calibrated against analytic results.
3. **Physics anchored to a reference.** Endpoints, forcing, inventories, energies, mirrors and particle scalings reproduce the numbers printed in Turyshev (2026) within 1-3 % (`tests/test_reference_turyshev2026.py`), with AR5, Kopparapu 2013 and Wordsworth 2019 checks.
4. **End-to-end assessment with compute savings.** `run_assessment()` chains requirements, the feasibility rubric, pathway allocation, the path graph, V&V and a reproducibility manifest; TRIADA caches repeats and MATE projection skips work when infeasibility is already certain.
5. **Honest by construction.** Every result carries `model_class` and warnings; physical invariants (mass conservation, Gibbs/enthalpy floors, sum rules) are enforced; nothing is invented silently (unknown bodies give open questions, not made-up data).
6. **Accuracy by scale and world.** `accuracy_table()` computes the expected accuracy from a 1 m box to a 200 ha dome on any body, with and without measured inputs (see [Accuracy by scale](#accuracy-by-scale-and-world)).

---

## Engine Catalog

NOArCO is composed of **14 specialised analytical engines**: the operational core (9 **Core** engines) and a decoupled satellite suite (5 **Extratools / plugins**).

### 1. Core Engines

| # | Engine | Main class | Purpose |
|---|---|---|---|
| **1** | **StateEngine** | `StateEngine` / `UnifiedPlanetState` | Physical consistency, validation of planetary states and verification of habitable endpoints (E0 to E4). |
| **2** | **AtmosphereEngine** | `AtmosphereEngine` | Dual modelling: closed-volume conditioning (ideal gas + back-pressure purge) and open hydrostatic column ($M \approx \frac{P \cdot A}{g}$). |
| **3** | **SoilEngine** | `SoilEngine` | Mineralogical analysis, elemental assays and dynamic selection of $\text{O}_2$ extraction routes (perchlorates, $\text{H}_2\text{O}$, $\text{Fe}_2\text{O}_3$, MRE). |
| **4** | **InventoryEngine** | `InventoryEngine` | Mass balances, detection of in-situ volatile gaps and a **heuristic viability index** (ranking only; physical feasibility is in `noarco.feasibility`). |
| **5** | **MechanismKnowledgeEngine (v2)** | `MechanismKnowledgeEngine` | Catalog of chemical/physical processes with real-time matching, $T$ and $P$ scoring, continuous energy penalty and efficiency degradation. |
| **6** | **ImpactEngine** | `ImpactEngine` | Biophysical impact assessment, radiative forcings and environmental perturbations. |
| **7** | **PathEngine** | `PathEngine` | Multi-objective Pareto optimization and synthesis of transition routes between planetary states. |
| **8** | **TimelineEngine** | `TimelineEngine` | Industrial deployment schedules, technology scale-up rates and decadal/centennial milestones. |
| **9** | **ReportEngine** | `ReportEngine` | Structured analytical export of energy balances and multi-body volumetric matrices. |

### 2. Satellite Engines and Plugins (5 Extratools)

Located in the decoupled module [`noarco.extratools`](noarco/extratools/):

| # | Extratool | Main class | Purpose |
|---|---|---|---|
| **10** | **NOArCO-Habitat** | `HabitatDimensioner` | Sizing of domes, bases and life support (ECLSS): gas mass, NASA-STD-3001 metabolic rates, HVAC and mechanical purge work. |
| **11** | **NOArCO-ISRU** | `ISRUEvaluator` | Evaluation of mining sites (*Jezero*, *Gale*, lunar poles): daily regolith tonnage, power and metallic by-products ($\text{Fe}, \text{Al}$). |
| **12** | **NOArCO-Logistics** | `VolatileLogisticsPlanner` | Astronautical supply chain to resolve macroscopic bottlenecks (Martian $\text{N}_2$, Mercury volatiles) via $\Delta v$ and cometary redirection. |
| **13** | **NOArCO-Exoplanet** | `ExoplanetClassifier` | Exoplanet astrophysics: Kopparapu (2013) habitability limits, Jeans thermal escape ($\lambda$) and the **Terraforming Feasibility Index (TFI)**. |
| **14** | **NOArCO-Agro** | `AstroAgroPlanner` | Space bioremediation: catalytic perchlorate detoxification, organic amendments, cyanobacteria inoculation and food production. |

---

## Verification & Validation Layer (`noarco.assurance`)

A hardened adaptation of the most agency-relevant engines of the companion [SFSA — Standard Framework for Scientific Advancement](#optional-sfsa-integration), aimed at review processes in the style of NASA-STD-7009 and ECSS:

| Module | SFSA origin | Guarantee |
|---|---|---|
| `units` | UDE | Strict dimensional analysis with compound units (`W m^-2 K^-4`, `kg/mbar`); an unknown unit **raises an error**, never a silent "dimensionless". |
| `uncertainty` | UQE | First-order propagation with full input covariance (correlated inputs) and a seeded Monte-Carlo cross-check that flags when the linear approximation is inadequate. |
| `validity` | AIE / MRE | Validity envelopes for every analytical model (e.g. the 1-layer gray greenhouse is not valid for Venus, τ = 100). |
| `invariants` | TRIADA T3 | Physical invariants: g = GM/R², composition closure, hydrostatic round trip, energy and mass balance. |
| `manifest` | RME | Reproducibility manifest: input hash + dependency versions + seed; tamper-evident seal. |
| `audit.assure` | ORE | One-call PASS / CAUTION / FAIL report, usable as a CI gate. |

```python
from noarco.session import NOArCoSession

report = NOArCoSession.for_mars().assure()
print(report.verdict)          # PASS | CAUTION | FAIL
print(report.to_markdown())    # uncertainty table, findings and manifest seal
```

Default 1-σ relative uncertainties are assigned from each state's epistemic status (`measured` 1 %, `modeled` 5 %, `estimated` 20 %, `assumed` 50 %). These are conservative engineering defaults, **not measurements**: override them with `rel_sigma` when a real error budget exists.

### Optional SFSA integration

SFSA is **optional**. With it installed (`pip install -e "<path-to>/SFSA/python"`), `noarco.assurance.sfsa_bridge.open_sfsa_session(planet)` opens an SFSA session whose model-risk engine uses the same validity envelopes that `assure()` enforces.

---

## Role: screening filter and interface, not a climate model

NOArCO is meant to sit **in front of** a full climate model: it removes scenarios whose inventory, power, throughput or forcing requirements already fail the feasibility rubric, and hands the survivors on. It does not compete with GCMs and does not simulate clouds, water, CO₂ collapse or aerosol dispersal; its only dynamics (`noarco.sim`) are a labelled **reduced-order 0-D** energy/mass balance.

* **Real input data, with provenance** (`noarco.data_io`): load JSON/CSV/NetCDF with T–P profiles, composition, albedo, insolation and simplified topography (mean surface pressure derived hydrostatically). Every scalar carries a unit (strictly converted) and `data_source = measured | literature | assumed`; the dataset has a SHA-256 hash and a provenance report that separates fact from conjecture, carried into `assure()`/`run_assessment()` as findings.
* **Margins that include model error** (`noarco.projection`): each projection reports `margin_parametric` and `margin_model_structural` (grey-atmosphere optical depth ×1.4, CO₂ log-law ×1.5), the probability with and without model error, and a documented list of what is *not* covered. `sensitivity()` ranks the inputs that control Π_min; `forcing_robustness()` answers "if the required forcing is off by ±X %, does Π_min still pass?".
* **External validation** (`noarco.validation`): 22 cases whose expected values are printed in Turyshev (2026), IPCC AR5, Kopparapu (2013) and Wordsworth (2019); run it in CI and attach it to reviews:

| Case | Source | Quantity | Published | NOArCO | Rel. error | Tol. | Pass |
|---|---|---|---|---|---|---|---|
| T-EQ2 | Turyshev 2026 Eq. 2/19 | mass per pascal | 3.89e+13 kg/Pa | 3.891e+13 | 0.04% | 0.30% | yes |
| T-TAB3-1BAR | Turyshev 2026 Table III | atmosphere mass at 1 bar | 3.89e+18 kg | 3.891e+18 | 0.04% | 1.00% | yes |
| T-TAB3-E3 | Turyshev 2026 Table III | atmosphere mass at 6.27 kPa | 2.44e+17 kg | 2.44e+17 | 0.00% | 1.00% | yes |
| T-EQ26 | Turyshev 2026 Eq. 26 | regional gas mass (1e12 m2, 50 kPa) | 1.35e+16 kg | 1.348e+16 | 0.17% | 1.00% | yes |
| T-EQ43 | Turyshev 2026 Eq. 43 | H2 inventory (0.5 bar, f=0.05, CO2 bg) | 4.7e+15 kg | 4.681e+15 | 0.41% | 2.00% | yes |
| T-EQ5 | Turyshev 2026 Eq. 5 | reversible O2 energy | 1.48e+07 J/kg | 1.482e+07 | 0.14% | 0.50% | yes |
| T-TAB6 | Turyshev 2026 Table VI | reversible H2 energy | 1.2e+08 J/kg | 1.176e+08 | 1.97% | 2.00% | yes |
| T-EQ72 | Turyshev 2026 Eq. 72 | minimum energy for p_O2 = 21 kPa | 1.2e+25 J | 1.211e+25 | 0.93% | 2.00% | yes |
| T-EQ31-30K | Turyshev 2026 Eq. 31 | forcing for +30 K | 78 W/m2 | 77.93 | 0.09% | 2.00% | yes |
| T-EQ31-60K | Turyshev 2026 Eq. 31 | forcing for +60 K | 191 W/m2 | 191.2 | 0.13% | 2.00% | yes |
| T-EQ33-273 | Turyshev 2026 Eq. 33 | tau_IR for 273 K | 3.1 1 | 3.136 | 1.16% | 3.00% | yes |
| T-EQ33-250 | Turyshev 2026 Eq. 33 | tau_IR for 250 K | 2 1 | 2.008 | 0.38% | 3.00% | yes |
| T-EQ56 | Turyshev 2026 Eq. 56 | mirror area for 20 W/m2 | 7e+12 m2 | 7.003e+12 | 0.05% | 1.00% | yes |
| T-EQ57 | Turyshev 2026 Eq. 57 | albedo lever, dA = -0.05 | 7 W/m2 | 7.363 | 5.18% | 6.00% | yes |
| T-EQ52 | Turyshev 2026 Eq. 52 | particle mass flow (30 L/s, 3e3 kg/m3) | 90 kg/s | 90 | 0.00% | 0.00% | yes |
| T-EQ83 | Turyshev 2026 Eq. 83 / Table X | pressurised area (1e5 kg/s, 100 yr, 10 kPa) | 1.17e+05 km2 | 1.171e+05 | 0.07% | 1.00% | yes |
| T-REPL | Turyshev 2026 Eq. 52-53 | aerosol replenishment of the +30 K nanorod case | 90 kg/s | 89.92 | 0.09% | 15.00% | yes |
| AR5-C2F6-RE | IPCC AR5 Table 8.A.1 | C2F6 radiative efficiency | 0.25 W/m2/ppb | 0.25 | 0.00% | 0.00% | yes |
| AR5-SF6-GWP | IPCC AR5 Table 8.A.1 | SF6 GWP100 | 2.35e+04 1 | 2.35e+04 | 0.00% | 0.00% | yes |
| K13-IHZ | Kopparapu et al. 2013 Sec. 4 | Sun moist-greenhouse inner edge | 0.99 AU | 0.9931 | 0.31% | 1.00% | yes |
| K13-OHZ | Kopparapu et al. 2013 Sec. 4 | Sun maximum-greenhouse outer edge | 1.7 AU | 1.705 | 0.32% | 1.00% | yes |
| W19-LAB | Wordsworth et al. 2019 (lab bound) | warming under 3 cm particle layer at 150 W/m2 (> 45 K) | 45 K (lower bound) | 45 | 0.00% | 1000.00% | yes |

22/22 cases pass; max relative error 5.18%.

* **Hand-off to a GCM** (`noarco.exchange`): `export_candidates(body, targets, dir)` writes the scenarios that survive screening as `noarco-scenario/1.0` JSON (initial state, target state, absorbed-flux forcing, added optical depth, Π numbers, provenance, limitations) plus an `index.csv`. It deliberately does not emit model-specific namelists: variable names differ between models and must be mapped explicitly and reviewed once.

---

## Fast probabilistic projection (`noarco.projection`)

`project(planet, target, confidence=0.95)` returns, in ~10 ms, the probability that a transition is feasible and confidence intervals (default 95 %, any level 50-99.9 %) for the rubric numbers, atmosphere mass, O2 energy, build time at the available power, CO2 warming and the aerosol mass/replenishment needed for the rest. It samples the uncertain inputs from documented priors (`default_priors`) and evaluates the closed forms vectorised.

```python
from noarco.projection import project
p = project(MARS, DesiredState.from_tier(HabitabilityTier.E3_ARMSTRONG), confidence=0.95)
print(p.statement())   # Mars: LIKELY INFEASIBLE (P(infeasible) = 96.0%); Pi_min 0.35 [0.11, 1.14] @ 95%; ...
```

Calibration is tested against analytic results (lognormal ratios: probability within 0.3 %, quantiles within 2 %, empirical coverage of the 95 % interval = 95 ± 1 %). **The interval is parametric**: it reflects the listed input uncertainties, not model-form error (grey atmosphere, log-law CO2 forcing, no feedbacks). A narrow interval means "insensitive to the listed inputs", not "certain". Tighten intervals by supplying better priors; `run_assessment` includes the projection by default.

## Screening and traceability (what a verdict gives you)

* **Headroom, not just yes/no:** every verdict carries `required` (availability each axis needs), `shortfall`, `min_build_time_years` (the build time at which flow and power axes pass; `inf` when the inventory is the limit, because time does not create mass) and `explain()`.
* **Trades in one call** (`noarco.screening`): `rank()` (deterministic, viable first), `sweep()` + `breakeven()` over one axis, `what_if()` (did viability or the limiting axis flip?), `required_availability()`.
* **Regional targets:** an enclosed target with `region_area_m2` gets a real gas and O₂ requirement (A·P/g, Turyshev Eq. 80).
* **Citeable and replayable record** (`noarco-result/1.1`, [JSON Schema](docs/noarco-result-1.1.schema.json)): inputs, `input_hash`, `constants_hash`, `code_hash` (SHA-256 of the whole package), constants actually used with source and class, strict JSON (no `NaN`/`Infinity`). `python scripts/verify_record.py record.json` validates the schema and **replays the record from its own inputs**; tampering with an input or a result is detected. `Verdict.bibtex()` and [`CITATION.cff`](CITATION.cff) give the citation.
* **Habitat + ISRU + logistics, closed:** `size_base()` sizes the habitat and its ISRU plant together and applies the same conjunctive rule on power, ISRU O₂ capacity and cargo. ISRU keeps a stoichiometric mass ledger, never reports energy below the reversible (Gibbs) floor, and sizes the solar array; the habitat has a thin-wall pressure-shell mass and a mission ledger; logistics derives delta-v from a heliocentric Hohmann transfer (the Moon no longer inherits Mars' number) and the rocket-equation propellant. These are engineering estimates with their assumptions registered in [`docs/CONSTANTS.md`](docs/CONSTANTS.md), not validated against flight hardware.

## Radiative modes and verdict confidence

| `radiative=` | What it does | Use for | Not for |
|---|---|---|---|
| `"simple"` (default) | log-law CO₂ forcing + grey-atmosphere mapping; optional bounded water-vapour factor `water_vapor_feedback_fraction` f_w ∈ [0, 1] (`ΔT_eff = ΔT_dry (1 + f_w)`, class *estimated*) | sweeps, inventory, Π_min mass/power/throughput | absolute ΔT as a hard gate (biased ~51 % low at 1 bar) |
| `"literature_calibrated"` | interpolates a **published** ΔT(P_CO₂) curve (Jakosky & Edwards 2018; Mars only; range assumed ±10 K at 1 bar) | stating a published warming with provenance | other worlds; discovering new physics |

Verdict axis confidence (in every JSON record, `axis_confidence`): **mass / power / throughput = high** (closed-form, anchors within 1–3 %), **forcing = low** in `simple` and **medium** in `literature_calibrated`, **stability = medium**. A `warnings` entry is added when an absolute ΔT target is requested in `simple` mode. The mode currently sets confidence labels and warnings only: Π mass / power / throughput dominate unless forcing availability is supplied, and `literature_calibrated` does not make the simple model a GCM. The default preset is strict inventory; the radiative axis is never sold as a prediction.

**Engineering metrics** (process quality, *not* climate accuracy; `noarco.metrics.engineering_metrics()`):

| Engineering metric (process quality, not climate accuracy) | Value |
|---|---|
| Constants with a named source | 71 % of 56 |
| `verify_registry` (values match the code) | green |
| External anchors reproduced | 22/22 |
| Same inputs -> same JSON | yes |
| Missing availability -> `indeterminado` (`viable = None`) | yes |

## What "accuracy" means here

NOArCO does not claim 100 % exact predictions, and no terraforming tool can: the physics of planetary-scale interventions is not known to that precision. What it guarantees is narrower and checkable:

* **Exact where physics is exact:** mass/pressure, stoichiometry (every catalogue reaction conserves mass and its yields are derived from atomic weights), Gibbs/enthalpy energy floors, Stefan-Boltzmann balances, unit conversions. Tests compare them with independent formulas at rounding-level tolerance.
* **Reproduces published benchmarks:** every number in `tests/test_reference_turyshev2026.py` is printed in the reference paper (within ~1-3 %).
* **Never silent about uncertainty:** each result carries `model_class` and warnings; uncertain inputs are listed in `docs/MODEL_CREDIBILITY.md`; `assure()` propagates input uncertainty; pathways are checked against physical invariants (`check_pathway_invariants`).
* **Scenario-consistent:** results respond correctly to conditions, time, materials and elements (power, build time, reservoirs, regolith assay, composition), which the suite checks via scaling laws and limiting cases.

## Feasibility and end-to-end assessment

`noarco.feasibility` implements the requirement bookkeeping and the conjunctive rubric of Turyshev (2026): `Pi_min = min(available / required)` over mass, forcing/optical depth, throughput, power and stability. `noarco.workflow.run_assessment(planet, target)` runs requirements, rubric, pathway allocation, path graph, V&V and credibility findings in one call, with TRIADA compute savings: results are cached by input hash (T1), and if the rubric already proves infeasibility the path graph is skipped and the verdict is certified as a MATE projection.

```python
from noarco.data.planets import MARS
from noarco.core.desired_state import DesiredState, HabitabilityTier
from noarco.workflow import run_assessment

report = run_assessment(MARS, DesiredState.from_tier(HabitabilityTier.E4_BREATHABLE))
print(report.verdict, report.status.value)   # INFEASIBLE projected
```

---

## Performance

Measured with the reproducible script [`benchmarks/benchmark_throughput.py`](benchmarks/benchmark_throughput.py) (median of 5 repeats, Apple M4, Python 3.12, arm64):

| Case | Median per call | Calls / s |
|---|---|---|
| `InventoryEngine.evaluate` (Mars, 2 500 m³) — full coupled air + soil evaluation | ≈ 13-26 µs | ≈ 40 000-80 000 |
| `MechanismKnowledgeEngine.recommend_for_planet` (full catalog) | ≈ 44-86 µs | ≈ 12 000-23 000 |
| `NOArCoSession.co2_feasibility` (analytical) | ≈ 2-3 µs | ≈ 350 000-630 000 |
| `NOArCoSession.assure` (invariants + validity + 3 first-order UQ outputs) | ≈ 3-4.4 ms | ≈ 230-340 |

Numbers depend on hardware; run the script on yours. No speed-up factor against external numerical solvers is claimed: compare against your own reference solver on identical inputs if you need a ratio.

---

## Installation and Quick Start

```bash
git clone https://github.com/alejomalia/NOArCO.git
cd NOArCO
pip install -e ".[dev]"

# Full checks
pytest tests/
ruff check .
mypy noarco
```

### Example 1: Quick query with the Mechanism Knowledge Engine (v2)

```python
from noarco.data.planets import MARS
from noarco.engines.knowledge import MechanismKnowledgeEngine

engine = MechanismKnowledgeEngine()
# Instantly recommends chemical routes for T = 210 K and P = 610 Pa
recommendations = engine.recommend_for_planet(MARS, goals=["oxygen"], top_n=3, verbose=False)
```

### Example 2: Size a habitat on Mars (NOArCO-Habitat)

```python
from noarco.data.planets import MARS
from noarco.extratools.habitat import HabitatDimensioner, HabitatSpecification

spec = HabitatSpecification(volume_m3=2500.0, crew_size=6, mission_days=365)
report = HabitatDimensioner.dimension(MARS, spec)
print(f"O₂ mass: {report.o2_mass_kg:.1f} kg | Continuous power: {report.continuous_power_kw:.2f} kW")
```

### Example 3: Classify an exoplanet and compute its TFI (NOArCO-Exoplanet)

```python
from noarco.extratools.exoplanet import ExoplanetClassifier, ExoplanetProfile

trappist_1e = ExoplanetProfile(
    name="TRAPPIST-1e",
    mass_earth=0.692,
    radius_earth=0.920,
    semi_major_axis_au=0.029,
    stellar_luminosity_solar=0.000553,
    stellar_teff_k=2566.0,
    albedo=0.25,
)
report = ExoplanetClassifier.classify(trappist_1e)
print(f"TFI score: {report.tfi_score_pct:.1f}% | Habitable zone: {report.is_in_habitable_zone}")
```

### Example 4: Verification & validation of a scenario

```python
from noarco.assurance import assure
from noarco.data.planets import VENUS

report = assure(VENUS)
print(report.verdict)                      # FAIL
for finding in report.findings:
    print(finding)                         # VAL-GRAY-TAU: IR optical depth outside validity range
```

---

## Accuracy by scale and world

What share of plausible worlds is the engine's habitat/enclosure calculation *within the tolerance*? Computed by `noarco.accuracy` (Monte Carlo over documented priors; `n = 20 000`, seed fixed): **hit rate = P(|true/nominal − 1| ≤ tolerance)**.

**Mars, tolerance ±10 %, default (engineering) inputs**

| Enclosure | Crew | Gas mass | O₂ use | Leakage | Power | Power margin (95 %) |
|---|---|---|---|---|---|---|
| 1 m box (1 m³) | 0 | 99 % | n/a | 11 % | 17 % | ±98 % |
| 6 m module (216 m³) | 4 | 100 % | 68 % | 11 % | 25 % | ±64 % |
| 100 m² pad × 3 m | 1 | 100 % | 68 % | 11 % | 18 % | ±88 % |
| 1 ha dome × 5 m | 100 | 100 % | 68 % | 11 % | 20 % | ±82 % |
| 10 ha dome × 10 m | 1 000 | 100 % | 68 % | 11 % | 21 % | ±76 % |
| 50 ha dome × 15 m | 5 000 | 100 % | 68 % | 11 % | 23 % | ±72 % |
| 200 ha dome × 20 m | 20 000 | 100 % | 68 % | 11 % | 24 % | ±68 % |

**Same, with measured inputs** (U-values, leak test, site temperatures, crew logs, electrolyser data supplied by the user)

| Enclosure | Crew | Gas mass | O₂ use | Leakage | Power | Power margin (95 %) |
|---|---|---|---|---|---|---|
| 1 m box (1 m³) | 0 | 99 % | n/a | 38 % | 52 % | ±28 % |
| 6 m module (216 m³) | 4 | 100 % | 100 % | 38 % | 71 % | ±19 % |
| 100 m² pad × 3 m | 1 | 100 % | 100 % | 38 % | 56 % | ±26 % |
| 1 ha dome × 5 m | 100 | 100 % | 100 % | 38 % | 60 % | ±24 % |
| 10 ha dome × 10 m | 1 000 | 100 % | 100 % | 38 % | 63 % | ±22 % |
| 50 ha dome × 15 m | 5 000 | 100 % | 100 % | 38 % | 66 % | ±21 % |
| 200 ha dome × 20 m | 20 000 | 100 % | 100 % | 38 % | 69 % | ±20 % |

**Continuous power, tolerance ±25 %, across worlds** (default inputs / measured inputs)

| World | 1 m box | 6 m module | 1 ha dome | 200 ha dome |
|---|---|---|---|---|
| Mars | 41 % / 92 % | 58 % / 99 % | 48 % / 96 % | 56 % / 99 % |
| Moon | 24 % / 70 % | 50 % / 96 % | 33 % / 84 % | 44 % / 94 % |
| Earth | 17 % / 35 % | 69 % / 100 % | 52 % / 95 % | 69 % / 100 % |
| Venus | 56 % / 96 % | 61 % / 98 % | 58 % / 97 % | 60 % / 98 % |
| TRAPPIST-1e | 30 % / 83 % | 51 % / 98 % | 38 % / 92 % | 48 % / 97 % |

How to read it: gas mass (ideal gas) is ≥ 99 % at every scale; the rest is limited by inputs, not by the equations: leakage and thermal load depend on the envelope and the site, so they become accurate only when those are measured. Small enclosures are the least certain for gas mass (construction tolerance, 3 % of volume at 1 m) and for thermal power on worlds with large site-to-site temperature contrasts (Moon, Mars, thin-atmosphere worlds). These are **parametric** accuracies under the stated priors, **not validated against measured habitats**, and exclude model-form error (conduction-only thermal model, no solar load, no structural dynamics); peak power (diurnal) is not covered. Reproduce with `accuracy_table(body, quality="default" | "measured", tolerance=...)`.

---

## Testing and Quality

| Check | Command | Status |
|---|---|---|
| Unit, regression, property and reference-benchmark tests | `pytest tests/` | 1799 passed, 1 skipped (needs optional SFSA) |
| Line coverage of `noarco/` | `pytest --cov=noarco` | 96 % |
| Static lint | `ruff check .` | passing |
| Static typing | `mypy noarco` | passing |

What the suite covers (test modules, incl. `test_reference_turyshev2026.py`, which checks the code against numbers printed in the reference paper):

| Area | Module(s) | Focus |
|---|---|---|
| Screening, traceability, ISRU/habitat/logistics | `test_screening_traceability` | headroom and min build time, ranking/sweeps, replay + tamper detection, schema, hashes, ISRU ledger and Gibbs floor, hull formula, Hohmann delta-v, integrated base budget |
| 1000-case calculation battery | `test_battery_1000` | 1000 situations (radius × gravity, enclosures 1 m² to 2000 km², pressures, build times, flows, inventories, 10 bodies) checked against formulas re-derived inside the test: hydrostatics, ideal gas, Gibbs O₂ work, Stefan-Boltzmann, Eddington mapping, conjunctive min-of-ratios, hash reproducibility |
| Core models & datasets | `test_core_models` | validation of every field, idempotent composition normalisation, consistency of all 10 solar-system bodies, endpoints, constraints, `StateEngine` |
| Physics laws & scaling | `test_physics_properties` | hydrostatic round trips, T_eq closed form and scaling laws, linear vs. exact forcing, ideal-gas scaling, seeded random-planet sweep |
| TRIADA/MATE core | `test_mate_core` | LRU semantics, key determinism, thread-safety under contention, decorator protocol (T1/T2/T3), projection certification |
| Session | `test_session_reactive` | validated updates (out-of-range and misspelled fields rejected), reactive invalidation, audit history, feasibility projections |
| Pathfinding & simulation | `test_paths_and_simulation` | cascade graph contiguity, Pareto-front non-domination (randomised), temporal simulator invariants and determinism |
| Extratools & engines | `test_extratools_extended`, `test_extratools` | logistics, habitat, ISRU, agro, exoplanet, pathway, closed-form timeline (checked against numerical integration), report, CLI, autodiscovery |
| Mechanisms | `test_aerogel`, `test_electrolysis`, `test_greenhouse_gas`, `test_mirrors`, `test_co2_mobilization`, `test_coverage_gaps` | per-mechanism physics, edge cases, every material/regime |
| Knowledge & analyzers | `test_knowledge_engine`, `test_analyzers_and_engines` | mechanism matching, T/P scoring, atmosphere/soil analysis |
| V&V layer | `test_assurance` | units, covariance UQ, validity envelopes, invariants, reproducibility manifest, `assure()` |

The suite includes regression tests for every defect fixed so far (see [CHANGELOG](CHANGELOG.md)).

---

## Scientific References

- **Turyshev, S. G. (2026).** *Terraforming Mars: Mass, Forcing, and Industrial Throughput Constraints*. arXiv:2603.00402.
- **Ansari et al. (2024).** *Feasibility of keeping Mars warm with nanoparticles*. Science Advances, 10, eadn4650.
- **DeBenedictis et al. (2025).** *The case for Mars terraforming research*. Nature Astronomy, 9, 634-639.
- **Wordsworth et al. (2019).** *Enabling Martian habitability with silica aerogel*. Nature Astronomy, 3, 898-903.
- **Kopparapu et al. (2013).** *Habitable zones around main-sequence stars: new estimates*. The Astrophysical Journal, 765(2), 131.
- **McKay, Toon & Kasting (1991).** *Making Mars habitable*. Nature, 352, 489-496.

Literature values embedded in the data tables cite their source inline; please verify them against the original publications before relying on them for decisions.

---

## License

This framework and its content are licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
See [`LICENSE`](LICENSE) for the full terms.

Copyright (c) 2026 Alejo Malia.
