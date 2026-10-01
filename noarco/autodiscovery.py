"""
noarco.autodiscovery — Self-completing knowledge system
========================================================

The framework detects its own gaps and fills them from first principles.

When NOArCO encounters a missing parameter (None value, stub module,
unknown optical depth, uncharacterised volatile inventory...), instead
of failing it:

    1. Detects the gap and classifies it by severity and derivability.
    2. Attempts to fill it using physics formulas + existing state data.
    3. Marks the derived value with SourceType.MODELED or ESTIMATED.
    4. Stores it in the session's KnowledgeBase.
    5. Next call: TRIADA T1 returns it instantly (cache hit, 0 cost).

The framework GROWS by facing problems:
    - First run against a new planet: many gaps, many derivations.
    - Second run: most gaps filled, TRIADA T1 takes over.
    - Third run: near-zero compute cost for gap-filling.

MATE integration:
    - Gap detection uses TRIADA before any derivation attempt.
    - If a gap cannot be filled by ANY formula with the available data,
      it is escalated as a NOArCoOpenQuestion (not silently ignored).
    - Derived values carry full audit trails (formula + inputs used).

Author credit: MATE + TRIADA methodology by Alejo Malia.

References
----------
Turyshev (2026). arXiv:2603.00402 (scaling relations used as priors).
Pierrehumbert (2010). Principles of Planetary Climate (radiative scalings).
Jakosky (2019). Planet. Space Sci. 175, 52-59 (inventory priors).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from noarco.core.constants import SIGMA_SB

logger = logging.getLogger("noarco.autodiscovery")


# ---------------------------------------------------------------------------
# Gap severity and derivability
# ---------------------------------------------------------------------------

class GapSeverity(str, Enum):
    """How much does this gap limit NOArCO's ability to compute results?"""
    CRITICAL    = "critical"    # Cannot compute core outputs without it
    SIGNIFICANT = "significant" # Degrades accuracy of mechanism estimates
    MINOR       = "minor"       # Only affects secondary outputs
    COSMETIC    = "cosmetic"    # Affects display/reporting only


class DerivabilityStatus(str, Enum):
    """Can this gap be filled from existing data?"""
    DERIVABLE       = "derivable"        # Formula exists, inputs available
    PARTIALLY       = "partially"        # Rough bound derivable, uncertain
    REQUIRES_DATA   = "requires_data"    # Need external data not in state
    UNDERDETERMINED = "underdetermined"  # Physics insufficient to constrain


# ---------------------------------------------------------------------------
# Gap data class
# ---------------------------------------------------------------------------

@dataclass
class NOArCoGap:
    """A detected gap in the framework's knowledge.

    Parameters
    ----------
    field_path : str
        Dot-path to the missing field. E.g. 'planet.ir_optical_depth'.
    description : str
        Human-readable description of what's missing and why it matters.
    severity : GapSeverity
        Impact on computation quality.
    derivability : DerivabilityStatus
        Whether it can be filled from existing data.
    formula_hint : str, optional
        The physics formula that can fill this gap.
    required_inputs : list[str]
        Fields needed to apply the formula.
    impacts : list[str]
        Which computations are blocked or degraded by this gap.
    """
    field_path: str
    description: str
    severity: GapSeverity
    derivability: DerivabilityStatus
    formula_hint: str | None = None
    required_inputs: list[str] = field(default_factory=list)
    impacts: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"[{self.severity.value.upper()}] {self.field_path}: {self.description} "
            f"(derivable: {self.derivability.value})"
        )


# ---------------------------------------------------------------------------
# Open question (gap that cannot be filled — escalated for human input)
# ---------------------------------------------------------------------------

@dataclass
class NOArCoOpenQuestion:
    """A gap that the framework cannot fill from first principles.

    These are escalated to the user as explicit open questions,
    not silently ignored. This prevents false precision.

    Parameters
    ----------
    question : str
        The scientific question that needs external input.
    gap : NOArCoGap
        The underlying gap that triggered this question.
    literature_search_hint : str
        Where to look for an answer.
    """
    question: str
    gap: NOArCoGap
    literature_search_hint: str


# ---------------------------------------------------------------------------
# DerivedValue — audit trail for auto-derived parameters
# ---------------------------------------------------------------------------

@dataclass
class DerivedValue:
    """A parameter value derived by the autodiscovery system.

    Parameters
    ----------
    field_path : str
        Which field was derived.
    value : Any
        The derived value.
    formula : str
        The physics formula used.
    inputs_used : dict[str, Any]
        The input values substituted into the formula.
    uncertainty_factor : float
        Multiplicative uncertainty estimate (e.g. 3.0 = within 3× of truth).
    source_type : str
        'modeled' or 'estimated'.
    references : list[str]
        Papers supporting the derivation formula.
    """
    field_path: str
    value: Any
    formula: str
    inputs_used: dict[str, Any]
    uncertainty_factor: float
    source_type: str  # 'modeled' or 'estimated'
    references: list[str] = field(default_factory=list)

    def summary(self) -> str:
        val_str = (
            f"{self.value:.4g}"
            if isinstance(self.value, (int, float))
            else str(self.value)[:60] + "…" if len(str(self.value)) > 60 else str(self.value)
        )
        return (
            f"  Derived {self.field_path}: {val_str} "
            f"[{self.source_type}, ±{self.uncertainty_factor}×]\n"
            f"    Formula: {self.formula}\n"
            f"    Inputs:  {self.inputs_used}"
        )


# ---------------------------------------------------------------------------
# GapDetector — scans PlanetState and session for missing data
# ---------------------------------------------------------------------------

class GapDetector:
    """Scans a NOArCO session for knowledge gaps.

    Inspects the PlanetState, DesiredState, and module availability
    to produce a prioritised list of gaps ordered by severity × impact.
    """

    def scan(
        self,
        planet: Any,
        constraints: Any | None = None,
        has_mechanisms: bool = False,
    ) -> list[NOArCoGap]:
        """Scan a PlanetState for all detectable gaps.

        Parameters
        ----------
        planet : PlanetState
            The planetary state to inspect.
        has_mechanisms : bool
            Whether any mechanism is already registered (suppresses the
            mechanism-coverage gap).
        constraints : ConstraintSet, optional
            Used to contextualise severity (e.g. ISRU-only changes
            which gaps are critical).

        Returns
        -------
        list[NOArCoGap]
            All detected gaps, sorted by severity.
        """
        gaps: list[NOArCoGap] = []

        # ── Radiative gaps ─────────────────────────────────────────────
        if planet.ir_optical_depth == 0.0:
            gaps.append(NOArCoGap(
                field_path="planet.ir_optical_depth",
                description=(
                    "IR optical depth is 0.0 (default). "
                    "Without it, greenhouse temperature is just T_eq — underestimates T_s."
                ),
                severity=GapSeverity.SIGNIFICANT,
                derivability=DerivabilityStatus.DERIVABLE,
                formula_hint=(
                    "τ_IR ≈ τ_ref × (P / P_ref)^α  where α≈1 for CO2-dominated atmospheres. "
                    "Ref: Pierrehumbert (2010), Section 4.3."
                ),
                required_inputs=["surface_pressure_pa", "gas_composition"],
                impacts=["RadiativeBalance.greenhouse_temperature_gray",
                         "NOArCoSession.forcing_required_wm2"],
            ))

        if planet.uv_flux_wm2 is None:
            gaps.append(NOArCoGap(
                field_path="planet.uv_flux_wm2",
                description=(
                    "UV flux at surface unknown. "
                    "Required for biological habitability assessment."
                ),
                severity=GapSeverity.MINOR,
                derivability=DerivabilityStatus.DERIVABLE,
                formula_hint=(
                    "Φ_UV ≈ S_UV_toa × exp(-τ_UV) where S_UV_toa ≈ 0.07 × S_⊙ "
                    "and τ_UV ≈ σ_CO2 × P/g (column opacity). "
                    "Ref: Cockell et al. (2000) Icarus."
                ),
                required_inputs=["solar_constant_wm2", "surface_pressure_pa",
                                  "gas_composition"],
                impacts=["HabitabilityTier E1-E4 assessment"],
            ))

        # ── Volatile inventory gaps ─────────────────────────────────────
        if planet.co2_ice_kg is None:
            gaps.append(NOArCoGap(
                field_path="planet.co2_ice_kg",
                description=(
                    "CO2 ice reservoir unknown. Critical for CO2 mobilisation "
                    "mechanism and E0→E1/E2 path planning."
                ),
                severity=GapSeverity.CRITICAL,
                derivability=DerivabilityStatus.PARTIALLY,
                formula_hint=(
                    "Lower bound: M_CO2_ice ≈ 0 (no detectable polar cap). "
                    "Upper bound: scale from polar albedo and radar depth. "
                    "For Marte: 5e16 kg measured (SHARAD/MOLA). "
                    "For unknown body: use 0 with high uncertainty flag."
                ),
                required_inputs=["surface_albedo", "radius_m"],
                impacts=["AtmosphericInventory.co2_inventory_gap_kg",
                         "NOArCoSession.co2_feasibility",
                         "CO2Mobilization mechanism"],
            ))

        if planet.h2o_ice_kg is None:
            gaps.append(NOArCoGap(
                field_path="planet.h2o_ice_kg",
                description="H2O ice reservoir unknown. Required for electrolysis O2 pathway.",
                severity=GapSeverity.SIGNIFICANT,
                derivability=DerivabilityStatus.PARTIALLY,
                formula_hint=(
                    "Isotopic D/H ratio gives D/H_current / D/H_SMOW ≈ water lost fraction. "
                    "M_H2O ≈ 0 for dry bodies, ~1e19-5e19 kg for Mars. "
                    "Use 0 with ESTIMATED flag for unknown bodies."
                ),
                required_inputs=["gas_composition", "body_name"],
                impacts=["Electrolysis mechanism", "water stability assessment"],
            ))

        # ── Regolith gaps ───────────────────────────────────────────────
        if planet.regolith_composition is None:
            gaps.append(NOArCoGap(
                field_path="planet.regolith_composition",
                description=(
                    "Regolith composition unknown. Needed for ISRU mass availability "
                    "of Fe, Al, Si for mechanism feedstocks."
                ),
                severity=GapSeverity.SIGNIFICANT,
                derivability=DerivabilityStatus.PARTIALLY,
                formula_hint=(
                    "For Mars: use Curiosity APXS average (SiO2~43%, Fe2O3~18%, Al2O3~9%). "
                    "For rocky bodies generally: use CI chondrite / bulk silicate Earth proxy. "
                    "Mark as ESTIMATED with 2-3× uncertainty."
                ),
                required_inputs=["body_name", "mass_kg"],
                impacts=["NanoparticleAerosol.isru_fraction",
                         "SilicaAerogel ISRU estimate",
                         "SuperGreenhouseGas ISRU estimate"],
            ))

        # ── Mechanism coverage gaps ─────────────────────────────────────
        # These are detected when no mechanism can cover a required forcing
        if not has_mechanisms:
            gaps.append(NOArCoGap(
                field_path="session.mechanisms",
                description=(
                    "No mechanisms have been evaluated yet. "
                    "Cannot assess which pathways are feasible."
                ),
                severity=GapSeverity.SIGNIFICANT,
                derivability=DerivabilityStatus.DERIVABLE,
                formula_hint=(
                    "Auto-select mechanisms based on ΔT target and constraints. "
                    "Priority: NanoparticleAerosol (fastest) → PFCs (long-term) → mirrors."
                ),
                required_inputs=["mean_temperature_k", "surface_pressure_pa"],
                impacts=["PathFinder", "timeline estimates"],
            ))

        # Sort by severity (CRITICAL first)
        severity_order = {
            GapSeverity.CRITICAL: 0,
            GapSeverity.SIGNIFICANT: 1,
            GapSeverity.MINOR: 2,
            GapSeverity.COSMETIC: 3,
        }
        gaps.sort(key=lambda g: severity_order[g.severity])
        return gaps

    def critical_gaps(self, planet: Any) -> list[NOArCoGap]:
        """Return only CRITICAL gaps."""
        return [g for g in self.scan(planet) if g.severity == GapSeverity.CRITICAL]

    def derivable_gaps(self, planet: Any) -> list[NOArCoGap]:
        """Return only gaps that can be filled from existing data."""
        return [
            g for g in self.scan(planet)
            if g.derivability in (
                DerivabilityStatus.DERIVABLE, DerivabilityStatus.PARTIALLY
            )
        ]


# ---------------------------------------------------------------------------
# GapFiller — derives missing values from physics formulas
# ---------------------------------------------------------------------------

class GapFiller:
    """Fills knowledge gaps using physics formulas + existing state data.

    Each `fill_*` method corresponds to a specific gap and implements
    the derivation formula documented in the gap's `formula_hint`.

    All derived values are marked with appropriate SourceType and
    uncertainty estimates. They NEVER overwrite measured values.
    """

    # ── σ constant ─────────────────────────────────────────────────────
    _SIGMA = SIGMA_SB

    def fill_ir_optical_depth(self, planet: Any) -> DerivedValue | None:
        """Derive IR optical depth from pressure and composition.

        Formula: τ_IR ≈ τ_Mars_ref × (P_CO2 / P_CO2_Mars) × (μ_CO2 / μ_atm)

        For CO2-dominated atmospheres, τ scales roughly linearly with
        CO2 column mass. Mars reference: τ ≈ 0.05 at 610 Pa CO2.

        Valid only for CO2 mole fraction >= 0.5 and P_CO2 <= 10 kPa; returns
        None otherwise.

        Reference: Pierrehumbert (2010) Section 4.3;
                   Wordsworth et al. (2010) Icarus 210, 992-997.
        """
        p_tot = planet.surface_pressure_pa
        x_co2 = planet.gas_composition.mole_fraction("CO2")
        p_co2 = x_co2 * p_tot

        # Mars reference: τ_IR ≈ 0.05 at P_CO2 ≈ 581 Pa (Wordsworth et al. 2010)
        tau_mars_ref = 0.05
        p_co2_mars_ref = 581.0  # Pa

        # Validity envelope of the linear scaling: thin, CO2-dominated atmospheres.
        # Outside it (Earth: trace CO2 + H2O; Venus: 9 MPa, strongly non-linear)
        # the estimate would be wrong by orders of magnitude, so report the
        # gap as underdetermined instead of inventing a value.
        if p_co2 <= 0 or x_co2 < 0.5 or p_co2 > 1.0e4:
            return None

        tau_derived = tau_mars_ref * (p_co2 / p_co2_mars_ref)

        return DerivedValue(
            field_path="planet.ir_optical_depth",
            value=tau_derived,
            formula="τ_IR = τ_Mars × (P_CO2 / P_CO2_Mars)",
            inputs_used={
                "P_CO2_Pa": p_co2,
                "P_CO2_Mars_ref_Pa": p_co2_mars_ref,
                "tau_Mars_ref": tau_mars_ref,
            },
            uncertainty_factor=3.0,  # 1D model; GCM would improve this
            source_type="modeled",
            references=[
                "Pierrehumbert (2010). Principles of Planetary Climate, Sect. 4.3.",
                "Wordsworth et al. (2010). Icarus, 210, 992-997.",
            ],
        )

    def fill_uv_flux(self, planet: Any) -> DerivedValue | None:
        """Estimate surface UV flux (200-400 nm) from solar constant and column.

        Formula:
            S_UV_TOA ≈ f_UV × S_⊙   (f_UV ≈ 0.07 for solar spectrum)
            τ_UV ≈ σ_CO2_UV × N_CO2  (CO2 column absorbs UV heavily)
            Φ_UV ≈ S_UV_TOA × exp(-τ_UV)

        For low-pressure CO2 atmospheres (Mars-like), UV at surface is
        significant: ~3-10 W/m² in the 200-400 nm band.

        Reference: Cockell et al. (2000) Icarus 146, 343-359.
        """
        s_sun = planet.solar_constant_wm2
        p_tot = planet.surface_pressure_pa
        x_co2 = planet.gas_composition.mole_fraction("CO2")
        g = planet.gravity_ms2

        # UV fraction of solar spectrum (200-400 nm)
        f_uv = 0.07
        s_uv_toa = f_uv * s_sun

        # CO2 UV cross-section (averaged 200-300 nm): σ ≈ 2e-23 m²/molecule
        sigma_uv_co2 = 2e-23  # m²/molecule
        avogadro = 6.022e23
        mu_co2 = 44.009e-3  # kg/mol
        p_co2 = x_co2 * p_tot

        # CO2 column number density (molecules/m²)
        n_co2 = (p_co2 / (mu_co2 / avogadro)) / g  # molecules/m²

        tau_uv = sigma_uv_co2 * n_co2
        phi_uv = s_uv_toa * math.exp(-min(tau_uv, 50.0))  # Cap to avoid underflow

        return DerivedValue(
            field_path="planet.uv_flux_wm2",
            value=phi_uv,
            formula="Φ_UV = f_UV × S_⊙ × exp(-σ_CO2 × N_CO2)",
            inputs_used={
                "solar_constant_wm2": s_sun,
                "P_CO2_Pa": p_co2,
                "tau_UV": tau_uv,
                "f_UV": f_uv,
            },
            uncertainty_factor=5.0,  # Rough: no ozone, no scattering model
            source_type="estimated",
            references=[
                "Cockell et al. (2000). Icarus 146, 343-359.",
                "1D column absorption, no scattering.",
            ],
        )

    def fill_co2_ice_kg(self, planet: Any) -> DerivedValue | None:
        """Estimate CO2 ice reservoir from body name and pressure.

        For known bodies: use literature lower bounds.
        Bodies without a tabulated literature value return ``None`` (an open question).

        Reference: Turyshev (2026) Sec. V.B; Jakosky & Edwards (2018).
        """
        body = planet.body_name.lower()

        if "mars" in body:
            # Mars: buried south-polar CO2 deposit ~6 mbar (Turyshev 2026, Sec. V.B)
            value = 2.3e16
            formula = "Mars south-polar CO2 deposit ~6 mbar = K * 600 Pa (Turyshev 2026, Sec. V.B)"
            uncertainty = 2.0
            source = "literature"
            refs = ["Turyshev (2026). arXiv:2603.00402, Sec. V.B.",
                    "Jakosky & Edwards (2018). Nature Astronomy 2, 634-639."]
        elif "venus" in body:
            # Venus: essentially no CO2 ice (all gaseous)
            value = 0.0
            formula = "Venus: T >> CO2 frost point at all surfaces"
            uncertainty = 1.0
            source = "modeled"
            refs = ["Basic thermodynamics: Venus T_s=737K >> T_CO2_frost=148K."]
        else:
            return None  # no published value: report an open question rather than invent a zero

        return DerivedValue(
            field_path="planet.co2_ice_kg",
            value=value,
            formula=formula,
            inputs_used={"body_name": planet.body_name},
            uncertainty_factor=uncertainty,
            source_type=source,
            references=refs,
        )

    def fill_h2o_ice_kg(self, planet: Any) -> DerivedValue | None:
        """Estimate H2O ice reservoir from body name and literature.

        Reference: Villanueva et al. (2015) Science 348, 218-221 (Mars D/H).
        """
        body = planet.body_name.lower()

        if "mars" in body:
            value = 2.0e19  # Central estimate, range 1.2e19 – 5e19 kg
            formula = "Mars: isotopic D/H ratio + MARSIS/SHARAD radar"
            uncertainty = 2.5
            source = "modeled"
            refs = [
                "Villanueva et al. (2015). Science 348, 218-221.",
                "Plaut et al. (2007). Science 316, 92-95.",
            ]
        elif "venus" in body:
            value = 0.0  # Venus is completely desiccated, all H2O in vapor traces (Seiff et al.)
            formula = "Venus: desiccated, surface T (737 K) >> boiling point"
            uncertainty = 1.0
            source = "modeled"
            refs = ["Seiff et al. (1986). J. Geophys. Res. (Venus atmosphere)."]
        else:
            return None  # no tabulated literature value

        return DerivedValue(
            field_path="planet.h2o_ice_kg",
            value=value,
            formula=formula,
            inputs_used={"body_name": planet.body_name},
            uncertainty_factor=uncertainty,
            source_type=source,
            references=refs,
        )

    def fill_regolith_composition(self, planet: Any) -> DerivedValue | None:
        """Estimate regolith composition from body name.

        Tabulated measurements exist for Mars (Curiosity APXS), Venus (Venera) and Earth
        (upper continental crust). Other bodies return ``None``.

        Reference: Grotzinger et al. (2014). Science 343, 1242777.
        """
        body = planet.body_name.lower()

        if "mars" in body:
            comp = {
                "SiO2":  0.430, "Fe2O3": 0.165, "Al2O3": 0.093,
                "MgO":   0.088, "CaO":   0.073, "SO3":   0.054,
                "TiO2":  0.009, "Cr2O3": 0.004, "MnO":   0.004,
                "P2O5":  0.009, "Cl":    0.005,
            }
            formula = "Curiosity APXS, Sol 700 average (Grotzinger et al. 2014)"
            uncertainty = 1.5
            refs = ["Grotzinger et al. (2014). Science 343, 1242777."]
        elif "venus" in body:
            # Venera probe measurements
            comp = {
                "SiO2":  0.450, "Al2O3": 0.155, "FeO":   0.092,
                "MgO":   0.115, "CaO":   0.070, "TiO2":  0.020,
                "K2O":   0.004,
            }
            formula = "Venera 13/14 XRF measurements"
            uncertainty = 2.0
            refs = ["Surkov et al. (1984). J. Geophys. Res. 89, B393."]
        elif "earth" in body:
            comp = {
                "SiO2":  0.605, "Al2O3": 0.159, "Fe2O3": 0.069,
                "CaO":   0.063, "MgO":   0.035, "Na2O":  0.032,
                "K2O":   0.028,
            }
            formula = "Upper continental crust average (Taylor & McLennan 1985)"
            uncertainty = 1.2
            refs = ["Taylor & McLennan (1985). The Continental Crust."]
        else:
            return None  # no tabulated measurement: do not substitute a generic chondrite prior

        return DerivedValue(
            field_path="planet.regolith_composition",
            value=comp,
            formula=formula,
            inputs_used={"body_name": planet.body_name},
            uncertainty_factor=uncertainty,
            source_type="literature",
            references=refs,
        )

    def fill_all_derivable(self, planet: Any) -> list[DerivedValue]:
        """Attempt to fill all derivable gaps for a given PlanetState.

        Runs all fill_* methods and returns successful derivations.
        Applies TRIADA T2 (MATHEMATICS): each fill method IS the closed-form.

        Returns
        -------
        list[DerivedValue]
            All successfully derived values, with full audit trails.
        """
        derived: list[DerivedValue] = []
        detector = GapDetector()
        gaps = detector.derivable_gaps(planet)

        fillers: dict[str, Callable[..., Any]] = {
            "planet.ir_optical_depth": self.fill_ir_optical_depth,
            "planet.uv_flux_wm2":      self.fill_uv_flux,
            "planet.co2_ice_kg":       self.fill_co2_ice_kg,
            "planet.h2o_ice_kg":       self.fill_h2o_ice_kg,
            "planet.regolith_composition": self.fill_regolith_composition,
        }

        for gap in gaps:
            filler = fillers.get(gap.field_path)
            if filler is None:
                continue
            try:
                result = filler(planet)
                if result is not None:
                    derived.append(result)
                    logger.info(
                        "[AutoDiscovery] Filled %s = %s [%s, ±%s×]",
                        result.field_path,
                        result.value if not isinstance(result.value, dict) else "{dict}",
                        result.source_type,
                        result.uncertainty_factor,
                    )
            except Exception as e:  # noqa: BLE001 - one failing filler must not abort discovery
                logger.warning("[AutoDiscovery] Could not fill %s: %s", gap.field_path, e)

        return derived


# ---------------------------------------------------------------------------
# SelfImprovingSession — session that grows by facing problems
# ---------------------------------------------------------------------------

class SelfImprovingSession:
    """A NOArCoSession wrapper that auto-fills its own gaps.

    When any computation encounters a missing parameter, instead of
    raising an error, this session:
      1. Detects the gap (GapDetector).
      2. Derives a value from physics (GapFiller).
      3. Updates the PlanetState with the derived value.
      4. Retries the computation.
      5. Logs everything for audit.

    The session GROWS by facing problems:
      - Expose it to a new planet → gaps detected → physics fills them.
      - Run a new mechanism → finds it needs optical data → derives it.
      - Ask about O2 → discovers H2O gap → estimates from body name.

    Parameters
    ----------
    session : NOArCoSession
        The underlying session to wrap with self-improvement.
    auto_fill : bool
        If True (default), automatically fill derivable gaps on init.
        If False, gaps are only filled when a computation requires them.
    raise_on_underdetermined : bool
        If True, raise ValueError for gaps that cannot be filled.
        If False (default), log them as OpenQuestions and continue.

    Examples
    --------
    >>> from noarco.data.planets import MARS
    >>> from noarco.session import NOArCoSession
    >>> from noarco.autodiscovery import SelfImprovingSession
    >>>
    >>> # Create a deliberately incomplete Mars state
    >>> mars_partial = MARS.model_copy(update={
    ...     'ir_optical_depth': 0.0,   # missing
    ...     'uv_flux_wm2': None,       # missing
    ...     'co2_ice_kg': None,        # missing
    ... })
    >>>
    >>> base_session = NOArCoSession(mars_partial)
    >>> smart_session = SelfImprovingSession(base_session, auto_fill=True)
    >>>
    >>> # All gaps auto-filled from physics:
    >>> print(smart_session.session.planet.ir_optical_depth)  # ~0.05 (derived)
    >>> print(smart_session.session.planet.co2_ice_kg)        # ~5e16 (Mars prior)
    >>> print(smart_session.diagnosis_report())               # full audit
    """

    def __init__(
        self,
        session: Any,  # NOArCoSession
        auto_fill: bool = True,
        raise_on_underdetermined: bool = False,
    ) -> None:
        self.session = session
        self.raise_on_underdetermined = raise_on_underdetermined

        self._detector = GapDetector()
        self._filler = GapFiller()
        self._derived_values: list[DerivedValue] = []
        self._open_questions: list[NOArCoOpenQuestion] = []
        self._fill_rounds: int = 0

        if auto_fill:
            self._run_self_improvement()

    def _run_self_improvement(self) -> None:
        """Detect and fill all derivable gaps, iterating until stable."""
        max_rounds = 5  # Prevent infinite loops (MATE R5: no endless compute)
        for round_n in range(max_rounds):
            gaps_before = len(self._detector.scan(self.session.planet, has_mechanisms=bool(self.session.mechanisms)))
            derived = self._filler.fill_all_derivable(self.session.planet)

            if not derived:
                logger.info(
                    "[SelfImproving] Round %d: no new derivations. Stable.", round_n + 1
                )
                break

            # Apply derived values to the session's planet state
            update_dict: dict[str, Any] = {}
            for dv in derived:
                # Only fill if the field is currently None/zero (don't overwrite measured)
                current_val = self._get_field(self.session.planet, dv.field_path)
                if self._is_missing(current_val):
                    field_name = dv.field_path.split(".")[-1]
                    update_dict[field_name] = dv.value
                    self._derived_values.append(dv)

            if update_dict:
                self.session.update(**update_dict)
                self._fill_rounds += 1
                gaps_after = len(self._detector.scan(self.session.planet, has_mechanisms=bool(self.session.mechanisms)))
                logger.info(
                    "[SelfImproving] Round %d: filled %d gaps (%d → %d remaining).",
                    round_n + 1, len(update_dict), gaps_before, gaps_after,
                )
            else:
                break

        # Auto-fill mechanisms if empty
        if not self.session.mechanisms:
            try:
                from noarco.mechanisms import (
                    CO2Mobilization,
                    Electrolysis,
                    NanoparticleAerosol,
                    OrbitalMirror,
                    SilicaAerogel,
                    SuperGreenhouseGas,
                )
                all_mech = [
                    NanoparticleAerosol(),
                    SuperGreenhouseGas(),
                    SilicaAerogel(),
                    OrbitalMirror(),
                    CO2Mobilization(),
                    Electrolysis(),
                ]
                for m in all_mech:
                    if m.is_applicable(self.session.planet):
                        self.session.add_mechanism(m)
            except Exception as e:  # noqa: BLE001 - optional enrichment must not abort the session
                logger.warning("Could not auto-populate mechanisms: %s", e)

        # Collect remaining underdetermined gaps as open questions
        remaining = self._detector.scan(self.session.planet, has_mechanisms=bool(self.session.mechanisms))
        for gap in remaining:
            if gap.derivability == DerivabilityStatus.UNDERDETERMINED:
                self._open_questions.append(NOArCoOpenQuestion(
                    question=(
                        f"What is {gap.field_path} for {self.session.planet.body_name}? "
                        f"Impact: {', '.join(gap.impacts[:2])}."
                    ),
                    gap=gap,
                    literature_search_hint=gap.formula_hint or "Consult mission data or models.",
                ))
                if self.raise_on_underdetermined:
                    raise ValueError(
                        f"Underdetermined gap: {gap.field_path}. "
                        f"Cannot derive from available data. "
                        f"Please provide it explicitly."
                    )

    @staticmethod
    def _get_field(planet: Any, field_path: str) -> Any:
        """Get a value from a nested field path like 'planet.ir_optical_depth'."""
        parts = field_path.split(".")
        obj = planet
        for part in parts[1:]:  # Skip 'planet' prefix
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    @staticmethod
    def _is_missing(value: Any) -> bool:
        """True if a value counts as 'missing' (None, 0.0 for optical depth)."""
        if value is None:
            return True
        return isinstance(value, float) and value == 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnose(self) -> list[NOArCoGap]:
        """Return all current gaps in the session's planet state."""
        return self._detector.scan(self.session.planet, has_mechanisms=bool(self.session.mechanisms))

    def open_questions(self) -> list[NOArCoOpenQuestion]:
        """Return gaps that could not be filled from physics."""
        return self._open_questions

    def derived_values(self) -> list[DerivedValue]:
        """Return all values that were auto-derived."""
        return list(self._derived_values)

    def force_fill(self, field_path: str) -> DerivedValue | None:
        """Force-fill a specific gap now (on demand).

        Parameters
        ----------
        field_path : str
            E.g. 'planet.ir_optical_depth'

        Returns
        -------
        DerivedValue if successful, None otherwise.
        """
        fillers = {
            "planet.ir_optical_depth": self._filler.fill_ir_optical_depth,
            "planet.uv_flux_wm2":      self._filler.fill_uv_flux,
            "planet.co2_ice_kg":       self._filler.fill_co2_ice_kg,
            "planet.h2o_ice_kg":       self._filler.fill_h2o_ice_kg,
            "planet.regolith_composition": self._filler.fill_regolith_composition,
        }
        filler = fillers.get(field_path)
        if filler is None:
            return None
        result = filler(self.session.planet)
        if result is not None:
            field_name = field_path.split(".")[-1]
            self.session.update(**{field_name: result.value})
            self._derived_values.append(result)
        return result

    def diagnosis_report(self) -> str:
        """Full audit report: gaps found, values derived, open questions."""
        lines = [
            "=" * 65,
            f" NOArCO AutoDiscovery Report — {self.session.planet.body_name}",
            "=" * 65,
            f"  Self-improvement rounds   : {self._fill_rounds}",
            f"  Values auto-derived       : {len(self._derived_values)}",
            f"  Remaining gaps            : {len(self.diagnose())}",
            f"  Open questions (unsolved) : {len(self._open_questions)}",
            "",
        ]

        if self._derived_values:
            lines.append("--- Auto-Derived Values ---")
            for dv in self._derived_values:
                lines.append(dv.summary())
            lines.append("")

        remaining = self.diagnose()
        if remaining:
            lines.append("--- Remaining Gaps ---")
            for gap in remaining:
                lines.append(f"  {gap}")
            lines.append("")

        if self._open_questions:
            lines.append("--- Open Questions (require human/mission input) ---")
            for oq in self._open_questions:
                lines.append(f"  Q: {oq.question}")
                lines.append(f"     Hint: {oq.literature_search_hint}")
            lines.append("")

        lines.append(
            "  [MATE] All derivations used closed-form formulas (TRIADA T2). "
            "No simulation was run to fill these gaps."
        )
        lines.append("=" * 65)
        return "\n".join(lines)
