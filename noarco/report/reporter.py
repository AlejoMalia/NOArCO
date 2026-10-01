"""
noarco.report.reporter — ReportEngine
====================================
Generates comprehensive reports, formatted markdown tables, and comparative
matrices for multi-planetary volumetric and terraforming analyses.
"""

from __future__ import annotations

from typing import ClassVar

from noarco.data.planets import ALL_SOLAR_BODIES
from noarco.inventory.inventory import ComprehensiveInventoryReport, InventoryEngine


class ReportEngine:
    """Engine that formats and outputs human-readable terraforming reports."""

    STANDARD_VOLUMES: ClassVar[list[tuple[float, str]]] = [
        (2.0, "2 m³ (EVA / Single-person pod)"),
        (33.0, "33 m³ (Habitat module / Rover)"),
        (2500.0, "2,500 m³ (Modular base / Station)"),
        (2.6e6, "2,600,000 m³ (Biodome / Settlement)"),
        (1.0e9, "10⁹ m³ (Megacity / Enclosed canyon)"),
        (1.2e13, "1.2 × 10¹³ m³ (Regional paraterraforming)"),
        (2.19e19, "2.19 × 10¹⁹ m³ (Sub-atmospheric scale)"),
        (1.0832e21, "1.08 × 10²¹ m³ (Total planetary scale)"),
    ]

    @classmethod
    def evaluate_matrix(
        cls,
        bodies: list[str] | None = None,
        volumes: list[float] | None = None,
    ) -> list[ComprehensiveInventoryReport]:
        """Compute the full matrix of reports for the specified bodies and volumes."""
        if bodies is None:
            bodies = [
                "Mercury",
                "Venus",
                "Earth",
                "Mars",
                "Jupiter",
                "Saturn",
                "Uranus",
                "Neptune",
                "Pluto",
            ]

        if volumes is None:
            volumes = [v[0] for v in cls.STANDARD_VOLUMES]

        results = []
        for body_name in bodies:
            planet = ALL_SOLAR_BODIES[body_name]
            for vol in volumes:
                rep = InventoryEngine.evaluate(planet, volume_m3=vol)
                results.append(rep)
        return results

    @classmethod
    def format_volume_label(cls, v: float) -> str:
        """Format volume in human-readable scientific or engineering format."""
        if v == 2.0:
            return "2 m³"
        elif v == 33.0:
            return "33 m³"
        elif v == 2500.0:
            return "2.500 m³"
        elif v == 2.6e6:
            return "2,6 × 10⁶ m³"
        elif v == 1.0e9:
            return "10⁹ m³"
        elif v == 1.2e13:
            return "1,2 × 10¹³ m³"
        elif v == 2.19e19:
            return "2,19 × 10¹⁹ m³"
        elif v >= 1.0e21:
            return "1,08 × 10²¹ m³"
        return f"{v:.1e} m³"

    @classmethod
    def build_markdown_summary_table(
        cls,
        reports: list[ComprehensiveInventoryReport],
        include_bottleneck: bool = True,
    ) -> str:
        """Construct a clean, professional markdown table for the multi-body analysis."""
        if include_bottleneck:
            lines = [
                "| Planet / Body | Volume (m³) | Regime / Mode | Target Gas Mass (kg) | Net ΔM (kg) | O₂ Mass (kg) | O₂ / Soil Source | Total Energy (kWh) | Estimated Time | Feasibility | Physical Bottleneck |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        else:
            lines = [
                "| Planet / Body | Volume (m³) | Regime / Mode | Target Gas Mass (kg) | Net ΔM (kg) | O₂ Mass (kg) | O₂ / Soil Source | Total Energy (kWh) | Estimated Time | Feasibility |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]

        for r in reports:
            vol_str = cls.format_volume_label(r.volume_m3)
            mode_str = r.mode.replace("_", " ").title()

            # Net mass delta string
            if r.volumetric_req.net_mass_delta_kg < 0:
                delta_str = f"**-{abs(r.volumetric_req.net_mass_delta_kg):.2e}** (Removal)"
            else:
                delta_str = f"+{r.volumetric_req.net_mass_delta_kg:.2e}"

            # Primary source
            src = r.o2_plan.primary_source.replace("_", " ").title()
            if src == "None":
                src = "Infeasible (Gaseous)"

            # Time string
            t_first = next(iter(r.time_years_at_power.items())) if r.time_years_at_power else ("N/A", 0)
            if t_first[1] == float("inf") or t_first[1] == 0:
                time_str = "Infeasible / N/A"
            elif t_first[1] < (1.0 / 365.25):
                time_str = f"{t_first[1]*8760:.1f} h ({t_first[0]})"
            elif t_first[1] < 1.0:
                time_str = f"{t_first[1]*365.25:.1f} days ({t_first[0]})"
            elif t_first[1] < 1000.0:
                time_str = f"{t_first[1]:.1f} years ({t_first[0]})"
            else:
                time_str = f"{t_first[1]:.1e} years ({t_first[0]})"

            feas_pct = f"{r.feasibility_score * 100:.0f}%"
            bottleneck_text = (r.critical_bottleneck or "Nominal").replace("|", "/")

            if include_bottleneck:
                line = (
                    f"| **{r.body_name}** | {vol_str} | {mode_str} | "
                    f"{r.total_gas_mass_kg:.2e} | {delta_str} | {r.o2_needed_kg:.2e} | "
                    f"{src} | {r.total_energy_kwh:.2e} | {time_str} | {feas_pct} | {bottleneck_text} |"
                )
            else:
                line = (
                    f"| **{r.body_name}** | {vol_str} | {mode_str} | "
                    f"{r.total_gas_mass_kg:.2e} | {delta_str} | {r.o2_needed_kg:.2e} | "
                    f"{src} | {r.total_energy_kwh:.2e} | {time_str} | {feas_pct} |"
                )
            lines.append(line)

        return "\n".join(lines)
