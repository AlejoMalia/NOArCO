# Changelog

## 0.1.0

First release.

### Added
- 14 analytical engines: 9 core (State, Atmosphere, Soil, Inventory, MechanismKnowledge v2, Impact, Path,
  Timeline, Report) and 5 extratools (Habitat, ISRU, Logistics, Exoplanet, Agro).
- `noarco.assurance`: verification & validation layer (strict units, covariance uncertainty propagation with
  Monte-Carlo linearity check, model validity envelopes, physical invariants, reproducibility manifest,
  `assure()` PASS/CAUTION/FAIL report), adapted from SFSA; `NOArCoSession.assure()`; optional
  `noarco.assurance.sfsa_bridge`.
- `RadiativeBalance.forcing_needed_exact` (non-linear Stefan-Boltzmann forcing).
- `benchmarks/benchmark_throughput.py`: reproducible throughput benchmark.
- Input validation for `HabitatSpecification`, `ExoplanetProfile`, `ISRURequirement`, `ISRUSiteProfile`.
- `verbose` flag on `MechanismKnowledgeEngine.recommend` / `recommend_for_planet`.

### Scientific audit (pre-release)
- Rebuilt against Turyshev (2026): end states (Armstrong limit 6.27 kPa, not 19.3 kPa), exact forcing,
  grey optical-depth mapping, feasibility rubric, PathEngine/PathFinder (computed, no fixed 0.82),
  0-D temporal simulator, aerosols (no Q_abs double count), aerogel (no invented cap, zero global
  forcing), mirrors (Eq. 55), electrolysis (no double efficiency), PFC (AR5 values; ppt/ppb x1000 fix),
  CO2 reservoirs (~20 mbar accessible), Kopparapu 2013 habitable zone, true Jeans parameter.
- `noarco.workflow.run_assessment` with TRIADA cache and MATE projection; `docs/MODEL_CREDIBILITY.md`.
- Habitat, agro and logistics assumptions exposed as parameters; `InventoryEngine` score labelled heuristic.

- Soil: exact oxygen bound in the regolith assay (formula parser, atomic weights), perchlorate flag from the
  assay; catalogue perchlorate yield corrected to 0.6435; stoichiometry/thermodynamic-floor tests; pathway
  invariant checker wired into the workflow.

- `noarco.projection.project`: fast probabilistic projection (P(feasible), confidence intervals, binding-constraint shares),
  calibrated against analytic lognormal results; integrated in `run_assessment`.

- `noarco.bodies.resolve_body`: any world from a list of conditions (derivations + recorded assumptions); accepted by
  `run_assessment`, `project`, `accuracy_*` and `NOArCoSession.from_conditions`.
- `noarco.accuracy`: hit-rate/margin tables by enclosure scale (1 m to 200 ha) and world, default vs measured inputs.
- Habitat: leakage now scales with surface/volume ratio; optional `envelope_area_m2`; `crew_size >= 0`.

- `noarco.data_io` (JSON/CSV/NetCDF inputs with units, `data_source` provenance, topography, input hash),
  structural model-error margins + `sensitivity()` + `forcing_robustness()` in `noarco.projection`,
  `noarco.validation` (22 external benchmark cases), `noarco.exchange` (neutral GCM hand-off), and a
  `reduced_order_0D` label on the temporal simulator.

- Validation against published GCMs: `run_gcm_validation()` (MarsWRF Table S3, Jakosky & Edwards 2018) with
  calibration / leave-one-out / bound / documented-discrepancy labels. It found that Turyshev's aerosol scaling
  is ~9x more optimistic than MarsWRF: default `NANOROD` is now GCM-calibrated (kappa 3.15e4 m2/kg, residence
  1.15 yr); the paper's scaling is `NANOROD_TURYSHEV`.
- `noarco.verdict` (single conjunctive verdict + limiting axis + citeable JSON), `noarco.constants_registry`
  (+ `docs/CONSTANTS.md`), `scripts/reproduce_reference.py`, `docs/WHAT_THIS_IS.md`.

- Radiative modes: `CO2Mobilization(radiative_mode="simple" | "literature_calibrated", water_vapor_feedback_fraction=f_w)`
  (`noarco.radiative.literature`: published Mars dT(P) anchors, Mars only); `run_gcm_validation` separates the radiative gate
  (`KNOWN_LIMITATION`, `use_for_dt_absolute`, `error_radiative`, band check) from inventory anchors; `verdict` carries
  `radiative_mode`, `axis_confidence` and `warnings`; `noarco.metrics.engineering_metrics`; README honest-accuracy statement;
  README intro rewritten as a screening framework; constants registry now 40 entries.

- Screening and traceability: verdict headroom (`required`, `shortfall`, `min_build_time_years`, `explain()`), `noarco.screening`
  (`rank`, `sweep`, `breakeven`, `what_if`), regional/enclosed requirements with `region_area_m2`; record schema `noarco-result/1.1`
  (JSON Schema, `code_hash`, `constants_used`, strict JSON), `replay()` / `validate_record()` / `scripts/verify_record.py`,
  `Verdict.bibtex()`, `CITATION.cff`; constants registry grown to 56 entries.
- Habitat/ISRU/logistics: ISRU mass ledger, Gibbs-floor check, water as one stream, solar-array sizing, no invented mineral
  fractions; habitat pressure-shell mass and mission ledger; logistics Hohmann delta-v and rocket equation (Moon no longer uses
  Mars' delta-v); agro crew support; `noarco.extratools.base_budget.size_base` (habitat + ISRU conjunctive budget).

### Test suite
- 1799 tests (+1 optional; includes the 1000-case calculation battery `tests/test_battery_1000.py`), 96 % line coverage: new modules for the TRIADA/MATE core, the reactive session,
  pathfinding and simulation, core models/datasets, physical-law properties, extratools/engines/CLI/autodiscovery.

### Fixed (found during the pre-release audit)
- `NOArCoSession.update` / `update_target` / `update_constraints` used `model_copy(update=...)`, which skips
  validation: out-of-range values (albedo = 5) and misspelled fields were accepted silently. Updates are now
  validated and unknown fields rejected.
- `TimelineEngine.generate_schedule` ignored `power_installed_gw` and `annual_growth_rate`; it now integrates
  the power curve in closed form (growth default 0).
- `NOArCoCache` claimed to be thread-safe but was not; it now uses a lock, validates `maxsize` and keys with SHA-256.
- `GapFiller.fill_ir_optical_depth` extrapolated a Mars scaling to Earth (0.004 vs 0.78) and Venus (5 vs 100);
  it now returns `None` outside its validity envelope. The planet-state marker used by `GapDetector` (which mutated
  the shared global datasets) was replaced by an explicit `has_mechanisms` argument.
- `VolatileLogisticsPlanner.plan_import` silently invented a 5 % mass fraction for unknown species; it now raises.
- `GasComposition` re-normalised already-valid compositions on every validation (non-idempotent) and forced partial
  compositions to sum to 1; only excesses above 1 are now renormalised.
- `AtmosphericInventory.delta_mass_for_pressure` returned `None` (the return statement was inside a comment),
  breaking `co2_inventory_gap_kg` and `NOArCoSession.co2_feasibility`.
- `NOArCoSession.co2_feasibility` omitted the documented `feasible` key on the non-projected path.
- `MonteCarloEngine.propagate_temperature_rise` sampled the CO2 inventory but ignored it; it now caps the
  delivered CO2 by the sampled inventory.
- `pfc_mass_for_forcing` hard-coded Mars' mean molar mass; it is now derived from the planet's composition.
- Documentation errors: Mars 6 kPa CO2 requirement (2.1e19 -> 2.1e17 kg); 210 -> 273 K forcing
  (52.3 -> 132 W/m² linearized without feedbacks, 205 W/m² exact).
- Removed unverified performance claims (900x speed-up) from the README; replaced by a reproducible benchmark.
- Packaging metadata: license (CC-BY-4.0), version single-sourced from `noarco.__version__`, supported Python
  versions, mypy target; removed "All rights reserved" contradicting CC BY 4.0 in `LICENSE`.
- All ruff and mypy findings resolved; all source, tests, examples and documentation are in English.
