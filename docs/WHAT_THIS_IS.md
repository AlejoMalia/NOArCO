# What NOArCO is - and is not

## What it is

* A **screening filter** for terraforming / paraterraforming / planetary-engineering scenarios. It
  answers, in milliseconds and with an audit trail: *is this transition viable, and which axis
  limits it* (mass, power, forcing, throughput or stability), using the conjunctive rubric
  `Pi_min = min(available / required)` of Turyshev (2026).
* A **requirements calculator** anchored to published numbers (22 anchor cases + published-GCM
  comparisons; run `python scripts/reproduce_reference.py`).
* A **probabilistic projector** with confidence intervals that separates parametric from structural
  (model-form) uncertainty, and a sensitivity tool.
* An **interface**: it exports the scenarios that survive screening, in a neutral format, for a full
  climate model to simulate.
* **Reproducible and citeable**: every verdict is a JSON record with an input hash, package version,
  model class and the hash of the constants registry (`docs/CONSTANTS.md`).

## What it is not

* **Not a GCM.** It does not simulate clouds, the water cycle, CO2 condensation/collapse, aerosol
  microphysics or dispersal, or any 3-D dynamics. Its only dynamics are a labelled reduced-order 0-D
  energy/mass balance.
* **Not flight software** and not qualified for mission use; no independent verification and
  validation has been performed.
* **Not a prediction of reality.** Results are lower bounds and discriminators. Parametric intervals
  do not include everything that can be wrong (see `MODEL_CREDIBILITY.md`, "Not covered").
* **Not validated against measured habitats** or experiments; accuracy tables are parametric.
* **Not free of known discrepancies with GCMs**: the CO2 log-law under-predicts the 1-bar warming by
  ~2x, the grey greenhouse under-states the present +5 K, and Turyshev's aerosol scaling is ~9x more
  optimistic than MarsWRF (all listed by `run_gcm_validation()`).

Use it to decide what to simulate, not to replace the simulation.

## Radiative modes: what they change

* `radiative="literature_calibrated"` changes how a ΔT is *stated* (a published, sourced number, Mars only); it does not turn the simple model into a GCM.
* The radiative mode currently sets the **forcing confidence label** (low in `simple`, medium in `literature_calibrated`) and the warnings. It does not move Π in the whole pipeline: Π mass / power / throughput dominate unless forcing availability is supplied.
* The accuracy-by-scale tables are parametric, not validated against real habitats; they are not a headline claim.

## What the verdict gives you beyond yes/no

Headroom per axis (`required`, `shortfall`), the build time that would clear flow and power (`inf` when mass limits), a ranked comparison of candidates, one-axis sweeps with exact break-even, and a replayable record (`scripts/verify_record.py`). The habitat/ISRU/logistics tools are engineering estimates with registered assumptions; they are closed (mass ledger, thermodynamic floor, shell stress) but not validated against flight hardware.

