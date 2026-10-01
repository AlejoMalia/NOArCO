# Model credibility register

Status of every model in NOArCO 0.1.0, in the spirit of NASA-STD-7009. **Verified** means the
implementation was checked against an external published source that is named; it does not
mean the physics is a prediction. Nothing here is flight-qualified.

Classes used in code (`MechanismResult.model_class`): `first_principles`, `reference_scaling`,
`engineering_estimate`, `screening_upper_bound`.

## Verified against external sources

| Model | Source checked | What was checked |
|---|---|---|
| Mass per pressure `K = 4πR²/g`, Table III inventories | Turyshev 2026 (arXiv:2603.00402) Eq. 2, 17-19, Table III | 3.89e13 kg/Pa; 2.37e16 … 3.89e18 kg |
| Regional gas mass, max pressurised area | Eq. 26, 80-83, Table X | 1.35e16 kg; 1.17e5 km² |
| Reversible O2 energy | Eq. 5, 67-72, Table XIII | 14.8 MJ/kg; 1.2e25 J for 21 kPa; ~380 TW over 1e3 yr |
| Direct forcing (Stefan-Boltzmann) | Eq. 31 | 78 W/m² (30 K), 191 W/m² (60 K) |
| Grey optical depth | Eq. 32-33 | τ = 3.1 (273 K), 2.0 (250 K) |
| Mirror area | Eq. 55-56 | 7.0e12 m² for 20 W/m² at η = 0.7 |
| Particle injection and column | Eq. 52-53 | 90 kg/s; 1.6-5.4 mg/m² |
| End states E0-E4, Armstrong limit 6.27 kPa | Sec. II.A, Table IV | thresholds |
| PFC radiative efficiency, lifetime, GWP100 | IPCC AR5 Table 8.A.1 (downloaded) | CF4, C2F6, C3F8, SF6 |
| Habitable-zone fit | Kopparapu et al. 2013 Table 3 (downloaded) | coefficients; Sun: 0.99 / 1.70 AU |
| Aerogel conductivity and lab warming bound | Wordsworth et al. 2019 (downloaded) | k = 0.01 W/m/K; > 45 K at 150 W/m² |
| Reference citations | DeBenedictis 2025, Richardson 2025, Jakosky 2019 | titles, volumes, pages |

| Catalogue reactions, oxygen yields, CO2 removal | atomic weights (IUPAC) | mass balance and derived yields, all catalogue mechanisms |
| Energy floors | NIST formation data | Fe2O3 reduction >= 824.2 kJ/mol; electrolysis >= Gibbs 14.8 MJ/kg O2 |

## Closed-form physics (correct by derivation, tested against independent formulas)

Hydrostatic mass, ideal-gas enclosure mass, radiative balance and the greenhouse/direct
forcing inversions, water-electrolysis Gibbs work, aerogel slab energy balance (upper bound),
TRIADA/MATE cache semantics, uncertainty propagation, units.

## Engineering estimates (parametrised; validate before relying on them)

| Item | Why it is uncertain |
|---|---|
| Aerosol mass absorption coefficient κ = 3.15e4 m²/kg (Al rods) | calibrated to MarsWRF (Richardson 2025 Table S3), rms 18 %, particle-specific; Turyshev's scaling (2.7e5) is ~9x more optimistic and kept as `NANOROD_TURYSHEV` |
| Sphere `Q_abs_ir` presets (iron, alumina, …) | assumed order-unity values, not Mie calculations |
| CO2 log-forcing coefficient α = 6 W/m² per doubling | Mars thin-atmosphere forcing is not log-law; paper cites < 10 K for 20 mbar |
| PFC forcing with terrestrial radiative efficiency | defined for small Earth perturbations; flagged above 1000 ppb |
| Aerogel `k_eff`, `T_vis`, `T_IR`, steady state | upper bound; no diurnal storage, latent heat or side losses |
| Synthesis / manufacturing energies (PFC 30 MJ/kg, aerogel 100 MJ/kg, particles 3.5 MJ/kg) | placeholders |
| MRE 38-40 MJ/kg O2 (floor-checked only), perchlorate detox 6 MJ/kg O2, mining 1.5 kWh/t, regolith bulk densities by body | unverified figures; MRE respects the enthalpy floor |
| Habitat U-values, COP, leak fraction, electrical loads | assumptions exposed as `HabitatSpecification` fields |
| Agro fixation rate, food yield | assumptions exposed as `SoilConditioningSpec` fields |
| Logistics retention thresholds, deflection rate, source catalogue | unverified; rate is a parameter |
| Knowledge-engine catalogue (T/P windows, energies, yields) | author-set values; the electrolysis energy follows Gibbs/0.70, the rest are unverified |

## Heuristics (rank, do not quantify)

* `InventoryEngine.feasibility_score`: product of fixed factors with body-name rules. Not a
  probability, not the rubric. Use `noarco.feasibility` (`Pi_min`) for physical feasibility.
* Terraforming Feasibility Index (exoplanet): fixed weights.
* `SelfImprovingSession` fillers: literature lookups only for Mars/Venus/Earth; other bodies
  yield open questions, never invented values.

## Not modelled (and therefore not predicted)

Latent-heat and ice reservoirs, CO2 condensation/collapse, water-vapour/cloud/albedo
feedbacks, escape and geochemical sinks, latitudinal/diurnal structure, aerosol microphysics and
dispersal, mirror control and degradation, O2 sinks (FeO oxidation), import logistics beyond the
kinetic benchmark. The temporal simulator is a quasi-equilibrium 0-D bookkeeping tool.

## Not replicated

Wordsworth et al. (2019) full radiative-thermal results and Ansari et al. (2024) simulations were
not reproduced numerically; only the published bounds and benchmark quantities were used.

## Projection intervals

`noarco.projection` intervals are **parametric**: priors are documented in `default_priors` (inventory factor 1.8, aerosol kappa factor 3, residence 1.7, CO2 forcing coefficient 6 +/- 1.5 W/m2, electrolysis efficiency 0.70 +/- 0.08, planet parameters by epistemic status). Model-form error is excluded and must be assessed separately.

## Accuracy tables

`noarco.accuracy` tables are parametric (priors in the module docstring: volume tolerance 3 sigma_L/L + 2 %, leak factor 2, U factor 1.3, site temperature spread 15 % on thin-atmosphere worlds, crew rates +/-10 %). Not validated against measured habitats; model-form error excluded.

## Structural (model-form) error

| Model class | Factor (1-sigma, multiplicative) | Basis |
|---|---|---|
| Grey-atmosphere optical-depth mapping | 1.4 | Turyshev 2026 Sec. IV.B: +/-30-50 % on tau_IR |
| CO2 logarithmic forcing | 1.5 | log law in a thin, band-saturated atmosphere (engineering factor) |
| Hydrostatic mass, Gibbs work, Stefan-Boltzmann | 1.0 | exact closed forms |

Projections report `margin_parametric` and `margin_model_structural` separately. **Not covered even there:** feedbacks (water vapour, clouds, albedo), CO2 condensation/collapse, latent-heat reservoirs, aerosol microphysics/dispersal, non-grey spectral structure beyond the factor above, escape and sinks, latitudinal/diurnal structure, input correlations, and errors in the reference scalings or reservoir estimates themselves.

## Accuracy by block (what "% error" means for each part)

| Block | Metric | Reference | Size of error |
|---|---|---|---|
| Inventory / throughput (kg per mbar, O₂ work, mirror area, column) | % vs paper anchor | Turyshev 2026 (22 anchor cases) | ~1–3 % |
| Radiative, simple mode | % vs GCM / literature | MarsWRF, Jakosky & Edwards 2018 | large / limited (documented) |
| Radiative, `literature_calibrated` | within the published band | Jakosky & Edwards 2018 | by construction inside the band (traceability, not a test) |
| Verdict Π_min | logical consistency | conjunctive rubric (Eq. 9–11) | not a physical % |

## What each result may be used for

| Quantity | vs reference | Action in the verdict |
|---|---|---|
| kg per mbar | ~1–3 % | **hard constraint** |
| O₂ energy floor | ~1–3 % | **hard constraint** |
| ΔT from CO₂ at 1 bar (simple) | ~51 % low (`error_radiative = |ΔT_model − ΔT_ref| / ΔT_ref`; `status = known_limitation`) | do **not** use as a hard ΔT gate (`use_for_ΔT_absolute = no`, `use_for_Π_min_mass = yes`) |
| Present greenhouse effect (simple) | ~74 % low | diagnostic only |

The radiative cases are gated separately from inventory: inventory/Turyshev anchors need a strict match, whereas the radiative gate asks "inside the reduced-model band, or a documented discrepancy" (published inter-model spread). The present +5 K (MarsWRF) versus +1.3 K (1-layer grey) is **not** treated as a passed GCM test. Radiative modes: `simple` (default) and `literature_calibrated` (Mars only; published curve, not invented by NOArCO). The optional water-vapour factor f_w is bounded to [0, 1] and registered as *estimated*; the 1-bar case implies f_w ≈ 1.

## Habitat, ISRU, logistics (engineering estimates, closed ledgers)

| Part | What is first-principles | What is an assumption (registered in CONSTANTS.md) |
|---|---|---|
| ISRU | stoichiometry (reactant = O2 + co-product, from atomic weights), Gibbs floor (water 14.8 MJ/kg O2, Fe2O3 15.5 MJ/kg O2), regolith mass from the assay | water recovery, excavation energy, array efficiency/derate, catalogue process energy |
| Habitat | ideal-gas fill, crew rates (NASA BVAD), thin-wall shell t = dP r / 2 sigma, mission ledger | U-values, leak fraction, hull alloy allowable (buckling under external pressure is flagged, not designed) |
| Logistics | heliocentric Hohmann delta-v, rocket equation, impact kinetic energy | v_inf, Isp, retention step function (Melosh & Vickery, qualitative), campaign rate |
| Agro | perchlorate stoichiometry, calorie arithmetic | crop yield, nitrogen fixation, kcal per kg |

None of these is validated against flight hardware; they feed the base budget as screening numbers.

## Data provenance

Inputs loaded through `noarco.data_io` carry `data_source = measured | literature | assumed`; assumed inputs raise `DATA-ASSUMED` findings and downgrade the planet's epistemic class (and therefore its default uncertainty). The example dataset `examples/data/mars_baseline.json` is copied from the framework's own Mars values, not a new measurement.

## External validation

`python -m noarco.validation` lists every published value used and the relative error. Passing means agreement with the published number within its rounding, not that the physics predicts reality. GCM comparisons are in `run_gcm_validation()` (MarsWRF Table S3: calibration, leave-one-out median error 17 % Al rods / 10 % graphene; Jakosky & Edwards 2018 bounds) with documented discrepancies and known radiative limitations. What remains undone: comparing Pi_min itself against GCM-derived verdicts (published GCM runs do not report Pi), and validation of non-Mars worlds.
