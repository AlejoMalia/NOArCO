| Constant | Value | Unit | Source | Class |
|---|---|---|---|---|
| SIGMA_SB | 5.67037e-08 | W m-2 K-4 | CODATA 2018 (exact in SI 2019) | **exact** |
| G_NEWTON | 6.6743e-11 | m3 kg-1 s-2 | CODATA 2018 | **measured** |
| R_GAS | 8.31446 | J mol-1 K-1 | N_A * k_B (exact) | **exact** |
| K_BOLTZMANN | 1.38065e-23 | J K-1 | SI 2019 | **exact** |
| AU | 1.49598e+11 | m | IAU 2012 | **exact** |
| YEAR_S (Julian) | 3.15576e+07 | s | IAU | **exact** |
| SOLAR_CONSTANT_1AU | 1361 | W m-2 | Kopp & Lean 2011 (TSI) | **measured** |
| DELTA_G_H2O | 237129 | J mol-1 | NIST/CRC standard Gibbs energy of formation, H2O(l) | **measured** |
| DELTA_H_H2O | 285830 | J mol-1 | NIST/CRC standard enthalpy of formation, H2O(l) | **measured** |
| E_O2_reversible | 1.482e+07 | J kg-1 | 2 * DELTA_G_H2O / M_O2 (Turyshev 2026 Eq. 5: 14.8 MJ/kg) | **derived** |
| CO2_SUBLIMATION | 590000 | J kg-1 | NIST WebBook ~0.57-0.59 MJ/kg; Turyshev 2026 uses 6e5 | **measured** |
| WATER_TRIPLE_POINT_PA | 611.657 | Pa | IAPWS | **exact** |
| WATER_TRIPLE_POINT_K | 273.16 | K | ITS-90 definition | **exact** |
| ARMSTRONG_LIMIT_PA | 6270 | Pa | water vapour pressure at 37 C (47 mmHg); Turyshev 2026 Sec. II.A | **measured** |
| MARS_CO2_polar_deposit | 2.3e+16 | kg | Turyshev 2026 Sec. V.B (~6 mbar), after Jakosky & Edwards 2018 | **literature** |
| MARS_CO2_accessible_reference | 7.78e+16 | kg | Turyshev 2026 Table III (~20 mbar); Jakosky & Edwards 2018 (~0.02 bar) | **literature** |
| MARS_CO2_crustal_upper_bound | 3.89e+18 | kg | Jakosky 2019 (up to ~1 bar sequestered; NOT accessible) | **literature** |
| CO2_forcing_per_doubling | 6 | W m-2 | log-law coefficient, engineering default (no verified Mars-specific source) | **estimated** |
| WATER_VAPOR_FEEDBACK_MAX | 1 | 1 | upper bound of f_w in dT_eff = dT_dry (1+f_w): a doubling of the dry response; the published 1-bar case implies f_w ~ 1.0 (noarco.validation); optional parameter, default 0 | **estimated** |
| MARS_CO2_WARMING_1BAR_MID | 60 | K | Jakosky & Edwards 2018, Nature Astron. 2, 634 (~1 bar CO2 close to melting); +/-10 K range assumed by NOArCO; used only by radiative='literature_calibrated' | **literature** |
| MARS_CO2_WARMING_20MBAR_MAX | 10 | K | Jakosky & Edwards 2018 (climate models: < 10 K from ~20 mbar); an upper bound | **literature** |
| AEROSOL_kappa_MarsWRF_Al | 31500 | m2 kg-1 | least-squares fit to Richardson et al. 2025 Table S3 (rms 18 %) | **calibrated** |
| AEROSOL_kappa_MarsWRF_graphene | 70800 | m2 kg-1 | fit to Richardson et al. 2025 Table S3 (rms 11 %) | **calibrated** |
| AEROSOL_kappa_Turyshev | 270000 | m2 kg-1 | inferred from Turyshev 2026 Eq. 33+52+53 (+30 K); ~9x the GCM fit | **literature** |
| AEROSOL_residence_Al | 1.15 | yr | column/release mass balance of Richardson et al. 2025 Table S3 | **calibrated** |
| PFC_RE_C2F6 | 0.25 | W m-2 ppb-1 | IPCC AR5 Table 8.A.1 (Earth, small perturbation) | **literature** |
| PFC_RE_CF4 | 0.09 | W m-2 ppb-1 | IPCC AR5 Table 8.A.1 | **literature** |
| PFC_RE_SF6 | 0.57 | W m-2 ppb-1 | IPCC AR5 Table 8.A.1 | **literature** |
| PFC_RE_C3F8 | 0.28 | W m-2 ppb-1 | IPCC AR5 Table 8.A.1 | **literature** |
| PFC_GWP100_SF6 | 23500 | 1 | IPCC AR5 Table 8.A.1 | **literature** |
| PFC_synthesis_energy | 3e+07 | J kg-1 | engineering placeholder | **estimated** |
| AEROGEL_k_eff | 0.01 | W m-1 K-1 | Wordsworth et al. 2019 (silica aerogel at martian pressures) | **literature** |
| AEROGEL_manufacture_energy | 1e+08 | J kg-1 | engineering placeholder | **estimated** |
| KOPPARAPU_moist_Seff_sun | 1.014 | 1 | Kopparapu et al. 2013 Table 3 | **literature** |
| KOPPARAPU_maxgh_Seff_sun | 0.3438 | 1 | Kopparapu et al. 2013 Table 3 | **literature** |
| CREW_O2 | 0.84 | kg person-1 day-1 | NASA BVAD (NASA/TP-2015-218570) | **literature** |
| CREW_CO2 | 1 | kg person-1 day-1 | NASA BVAD | **literature** |
| HABITAT_U_cold | 0.35 | W m-2 K-1 | engineering assumption (HabitatSpecification field) | **estimated** |
| HABITAT_leak_fraction | 0.005 | 1 yr-1 | engineering assumption at reference A/V | **estimated** |
| CREW_H2O | 2.5 | kg person-1 day-1 | NASA BVAD (drinking + food water), NASA/TP-2015-218570 | **literature** |
| CREW_metabolic_heat | 120 | W person-1 | NASA BVAD sensible heat per crew member | **literature** |
| HULL_allowable_stress | 1.38e+08 | Pa | Al 6061-T6 yield 276 MPa (MMPDS) / safety factor 2; engineering default | **estimated** |
| HULL_density | 2700 | kg m-3 | aluminium alloy 6061 (handbook) | **measured** |
| ISRU_dGf_Fe2O3 | 742200 | J mol-1 | NIST-JANAF, hematite, 298 K (magnitude) | **measured** |
| ISRU_M_Fe | 0.055845 | kg mol-1 | IUPAC standard atomic weight | **measured** |
| ISRU_water_recovery | 0.9 | 1 | engineering assumption (heating recovery of regolith H2O) | **estimated** |
| ISRU_excavation_energy | 1.5 | kWh t-1 | engineering assumption (excavation + grinding) | **estimated** |
| SOLAR_array_efficiency | 0.25 | 1 | engineering assumption (space-grade multijunction on the surface) | **estimated** |
| SOLAR_array_derate | 0.8 | 1 | engineering assumption (dust, pointing, degradation) | **estimated** |
| AGRO_food_kcal_per_kg | 800 | kcal kg-1 | USDA, fresh potato ~770 kcal/kg; crop choice is an assumption | **estimated** |
| CREW_kcal | 2500 | kcal person-1 day-1 | NASA-STD-3001 order of magnitude, active adult | **estimated** |
| GM_SUN | 1.32712e+11 | km3 s-2 | IAU 2015 nominal solar mass parameter | **exact** |
| G0_standard_gravity | 9.80665 | m s-2 | CGPM 1901 (exact) | **exact** |
| LOGISTICS_v_infinity | 5 | km s-1 | assumed hyperbolic excess speed at the target (parameter) | **estimated** |
| LOGISTICS_isp | 3000 | s | assumed electric/mass-driver specific impulse (parameter) | **estimated** |
| LOGISTICS_deflections_per_century | 5 | 1 | engineering assumption (parameter) | **estimated** |

3 calibrated, 1 derived, 16 estimated, 9 exact, 18 literature, 9 measured.
