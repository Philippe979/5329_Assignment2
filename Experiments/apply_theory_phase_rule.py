import csv
from pathlib import Path

from phase_rule_experiments import (
    assign_slope_plateau_rule,
    load_grouped_rows,
    plot_timeline,
    summarize,
    write_csv,
)


THEORY_RULE = {
    "name": "theory_slope_plateau",
    "clean_threshold": 0.30,
    "drop_abs": 0.02,
    "drop_rel": 0.20,
    "gap_growth": 0.05,
    "tau_ra": 0.05,
    "smooth_window": 5,
    "plateau_ratio": 0.85,
    "fallback_ratio": 0.90,
    "stable_slope": 0.15,
    "negative_slope": 0.25,
    "consecutive": 3,
}


def read_selected_protocol(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as file:
        return {row["model"]: row for row in csv.DictReader(file)}


def main() -> None:
    results_dir = Path("Results/adaptive_protocol/final")
    output_dir = Path("Results/adaptive_protocol/theory_phase_rule")
    grouped = load_grouped_rows(results_dir)
    selected = read_selected_protocol(Path("Results/adaptive_protocol/protocol/selected_model_protocol.csv"))

    all_rows = []
    summary_rows = []
    grouped_phased = {}

    for model_name, rows in grouped.items():
        phased = assign_slope_plateau_rule(rows, THEORY_RULE)
        grouped_phased[model_name] = phased

        summary = summarize(model_name, THEORY_RULE["name"], phased)
        protocol = selected.get(model_name, {})
        summary["epsilon_label"] = protocol.get("epsilon_label", "")
        summary["selected_dynamic_type"] = protocol.get("dynamic_type", "")
        summary_rows.append(summary)

        for row in phased:
            new_row = dict(row)
            new_row["scheme"] = THEORY_RULE["name"]
            new_row["epsilon_label"] = protocol.get("epsilon_label", "")
            new_row["selected_dynamic_type"] = protocol.get("dynamic_type", "")
            all_rows.append(new_row)

    write_csv(all_rows, output_dir / "theory_slope_plateau_checkpoint_metrics_phased.csv")
    write_csv(summary_rows, output_dir / "theory_slope_plateau_occupancy_summary.csv")
    plot_timeline(
        grouped_phased,
        "Theory-Based Dynamic Robustness Phase Timeline",
        output_dir / "theory_slope_plateau_phase_timeline.png",
    )
    write_markdown(summary_rows, output_dir / "THEORY_PHASE_RULE_SUMMARY.md")
    print(f"Saved theory-based phase rule outputs to {output_dir}")


def write_markdown(summary_rows, path: Path) -> None:
    summary_rows = sorted(summary_rows, key=lambda row: row["model"])
    phase_counts = {}
    for row in summary_rows:
        phase_counts[row["dominant_phase"]] = phase_counts.get(row["dominant_phase"], 0) + 1

    lines = [
        "# Theory-Based Phase Rule Summary",
        "",
        "Operational definition:",
        "",
        "```text",
        "Pre-learning: CA(p) < tau_clean",
        "Robustness Formation: RA is below the plateau region and robust performance is still forming",
        "Robustness Stabilization / Peak: smoothed RA has low local slope and remains near the robust peak",
        "Late Degradation: RA decreases materially after the plateau/peak",
        "Late Divergence: clean-robust gap expands materially without strong RA collapse",
        "Late Stabilization: neither RA nor gap changes materially after the plateau/peak",
        "```",
        "",
        "Parameters:",
        "",
        "```text",
        f"tau_clean = {THEORY_RULE['clean_threshold']}",
        f"tau_RA = {THEORY_RULE['tau_ra']}",
        f"smooth_window = {THEORY_RULE['smooth_window']}",
        f"plateau_ratio = {THEORY_RULE['plateau_ratio']}",
        f"stable_slope = {THEORY_RULE['stable_slope']}",
        f"negative_slope = {THEORY_RULE['negative_slope']}",
        f"drop_abs = {THEORY_RULE['drop_abs']}",
        f"drop_rel = {THEORY_RULE['drop_rel']}",
        f"gap_growth = {THEORY_RULE['gap_growth']}",
        "```",
        "",
        "Dominant phase counts:",
        "",
    ]
    for phase, count in sorted(phase_counts.items()):
        lines.append(f"- {phase}: {count}")
    lines.extend([
        "",
        "| model | epsilon | dominant phase | phase completeness | RA_peak | p_peak | RA_final |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(
            "| {model} | {epsilon_label} | {dominant_phase} | {phase_completeness} | "
            "{RA_peak:.4f} | {p_peak} | {RA_final:.4f} |".format(
                model=row["model"],
                epsilon_label=row["epsilon_label"],
                dominant_phase=row["dominant_phase"],
                phase_completeness=row["phase_completeness"],
                RA_peak=float(row["RA_peak"]),
                p_peak=row["p_peak"],
                RA_final=float(row["RA_final"]),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
