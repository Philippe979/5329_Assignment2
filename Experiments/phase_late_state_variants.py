import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from phase_rule_experiments import (
    PHASE_COLORS,
    first_consecutive,
    load_grouped_rows,
    moving_average,
    phase_widths,
    plot_timeline,
    slopes,
    valid_start_index,
    write_csv,
)


def late_state_label(rows: Sequence[Dict], peak_index: int, index: int, scheme: Dict) -> str:
    ra_peak = float(rows[peak_index]["test_robust_acc"])
    ra_current = float(rows[index]["test_robust_acc"])
    gap_peak = float(rows[peak_index]["gap"])
    gap_current = float(rows[index]["gap"])
    drop_abs = ra_peak - ra_current
    drop_rel = drop_abs / ra_peak if ra_peak > 0 else 0.0
    if drop_abs >= scheme["drop_abs"] and drop_rel >= scheme["drop_rel"]:
        return "Late Degradation"
    if gap_current - gap_peak >= scheme["gap_growth"]:
        return "Late Divergence"
    return "Late Stabilization"


def find_plateau_start(rows: Sequence[Dict], scheme: Dict) -> tuple[int, int, List[float]]:
    valid_start = valid_start_index(rows, scheme["clean_threshold"])
    if valid_start is None:
        return -1, -1, []

    progress = [float(row["progress"]) for row in rows]
    robust = [float(row["test_robust_acc"]) for row in rows]
    smooth = moving_average(robust, scheme["smooth_window"])
    slope = slopes(progress, smooth)
    valid_indices = list(range(valid_start, len(rows)))
    peak_index = max(valid_indices, key=lambda index: smooth[index])
    ra_peak = smooth[peak_index]

    plateau_start = first_consecutive(
        valid_indices,
        lambda index: (
            index <= peak_index
            and smooth[index] >= scheme["plateau_ratio"] * ra_peak
            and abs(slope[index]) <= scheme["stable_slope"]
        ),
        scheme["consecutive"],
    )
    if plateau_start is None:
        plateau_start = first_consecutive(
            valid_indices,
            lambda index: index <= peak_index and smooth[index] >= scheme["fallback_ratio"] * ra_peak,
            scheme["consecutive"],
        )
    if plateau_start is None:
        plateau_start = peak_index
    return plateau_start, peak_index, smooth


def assign_late_after_peak(rows: Sequence[Dict], scheme: Dict) -> List[Dict]:
    rows = [dict(row) for row in rows]
    valid_start = valid_start_index(rows, scheme["clean_threshold"])
    for row in rows:
        row["phase"] = "Pre-learning"
        row["valid_for_analysis"] = str(float(row["test_clean_acc"]) >= scheme["clean_threshold"])
    if valid_start is None:
        return rows

    plateau_start, peak_index, _ = find_plateau_start(rows, scheme)
    for index, row in enumerate(rows):
        if index < valid_start:
            row["phase"] = "Pre-learning"
        elif index < plateau_start:
            row["phase"] = "Robustness Formation"
        elif index <= peak_index:
            row["phase"] = "Robustness Stabilization / Peak"
        else:
            row["phase"] = late_state_label(rows, peak_index, index, scheme)
    return rows


def assign_late_after_plateau_width(rows: Sequence[Dict], scheme: Dict) -> List[Dict]:
    rows = [dict(row) for row in rows]
    valid_start = valid_start_index(rows, scheme["clean_threshold"])
    for row in rows:
        row["phase"] = "Pre-learning"
        row["valid_for_analysis"] = str(float(row["test_clean_acc"]) >= scheme["clean_threshold"])
    if valid_start is None:
        return rows

    plateau_start, peak_index, _ = find_plateau_start(rows, scheme)
    progress = [float(row["progress"]) for row in rows]
    min_late_progress = progress[plateau_start] + scheme["min_plateau_width"]
    late_start = next(
        (index for index in range(peak_index + 1, len(rows)) if progress[index] >= min_late_progress),
        len(rows),
    )

    for index, row in enumerate(rows):
        if index < valid_start:
            row["phase"] = "Pre-learning"
        elif index < plateau_start:
            row["phase"] = "Robustness Formation"
        elif index < late_start:
            row["phase"] = "Robustness Stabilization / Peak"
        else:
            row["phase"] = late_state_label(rows, peak_index, index, scheme)
    return rows


def assign_late_training_window(rows: Sequence[Dict], scheme: Dict) -> List[Dict]:
    rows = [dict(row) for row in rows]
    valid_start = valid_start_index(rows, scheme["clean_threshold"])
    for row in rows:
        row["phase"] = "Pre-learning"
        row["valid_for_analysis"] = str(float(row["test_clean_acc"]) >= scheme["clean_threshold"])
    if valid_start is None:
        return rows

    plateau_start, peak_index, _ = find_plateau_start(rows, scheme)
    progress = [float(row["progress"]) for row in rows]
    late_start = next(
        (index for index in range(max(plateau_start, peak_index), len(rows)) if progress[index] >= scheme["late_progress"]),
        len(rows),
    )

    for index, row in enumerate(rows):
        if index < valid_start:
            row["phase"] = "Pre-learning"
        elif index < plateau_start:
            row["phase"] = "Robustness Formation"
        elif index < late_start:
            row["phase"] = "Robustness Stabilization / Peak"
        else:
            row["phase"] = late_state_label(rows, peak_index, index, scheme)
    return rows


def summarize(model: str, scheme_name: str, rows: Sequence[Dict]) -> Dict:
    widths = phase_widths(rows)
    occupancy = {phase: 0.0 for phase in PHASE_COLORS}
    for row, width in zip(rows, widths):
        occupancy[row["phase"]] = occupancy.get(row["phase"], 0.0) + width
    total = sum(widths) or 1.0
    normalized = {phase: value / total for phase, value in occupancy.items()}
    robust = [float(row["test_robust_acc"]) for row in rows]
    peak_index = max(range(len(rows)), key=lambda index: robust[index])
    return {
        "scheme": scheme_name,
        "model": model,
        "RA_peak": robust[peak_index],
        "p_peak": rows[peak_index]["progress"],
        "RA_final": robust[-1],
        "dominant_phase": max(normalized, key=lambda phase: normalized[phase]),
        "phase_completeness": sum(1 for value in normalized.values() if value >= 0.05),
        **{f"occ_{phase}": round(value, 6) for phase, value in normalized.items()},
    }


def write_markdown(summary_rows: Sequence[Dict], path: Path) -> None:
    schemes = sorted({row["scheme"] for row in summary_rows})
    lines = [
        "# Late-State Phase Variants",
        "",
        "These variants treat late phase as a post-plateau or post-peak training state, then classify it as stabilization, degradation, or divergence.",
        "",
        "| scheme | dominant formation | dominant stabilization | dominant late | avg completeness |",
        "|---|---:|---:|---:|---:|",
    ]
    for scheme in schemes:
        rows = [row for row in summary_rows if row["scheme"] == scheme]
        formation = sum(1 for row in rows if row["dominant_phase"] == "Robustness Formation")
        stabilization = sum(1 for row in rows if row["dominant_phase"] == "Robustness Stabilization / Peak")
        late = sum(1 for row in rows if row["dominant_phase"].startswith("Late"))
        avg = sum(int(row["phase_completeness"]) for row in rows) / len(rows)
        lines.append(f"| {scheme} | {formation} | {stabilization} | {late} | {avg:.2f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results_dir = Path("Results/adaptive_protocol/final")
    output_dir = Path("Results/adaptive_protocol/late_state_variants")
    grouped = load_grouped_rows(results_dir)
    base = {
        "clean_threshold": 0.30,
        "drop_abs": 0.02,
        "drop_rel": 0.20,
        "gap_growth": 0.05,
        "smooth_window": 5,
        "plateau_ratio": 0.85,
        "fallback_ratio": 0.90,
        "stable_slope": 0.15,
        "consecutive": 3,
    }
    schemes = [
        {"name": "late_after_peak", "assign": assign_late_after_peak, **base},
        {"name": "late_after_plateau_width_010", "assign": assign_late_after_plateau_width, **base, "min_plateau_width": 0.10},
        {"name": "late_after_plateau_width_020", "assign": assign_late_after_plateau_width, **base, "min_plateau_width": 0.20},
        {"name": "late_training_window_080", "assign": assign_late_training_window, **base, "late_progress": 0.80},
        {"name": "late_training_window_070", "assign": assign_late_training_window, **base, "late_progress": 0.70},
    ]

    summary_rows = []
    all_rows = []
    for scheme in schemes:
        grouped_phased = {}
        for model_name, rows in grouped.items():
            phased = scheme["assign"](rows, scheme)
            grouped_phased[model_name] = phased
            summary_rows.append(summarize(model_name, scheme["name"], phased))
            for row in phased:
                new_row = dict(row)
                new_row["scheme"] = scheme["name"]
                all_rows.append(new_row)
        scheme_dir = output_dir / scheme["name"]
        write_csv(
            [row for row in all_rows if row["scheme"] == scheme["name"]],
            scheme_dir / f"{scheme['name']}_checkpoint_metrics_phased.csv",
        )
        plot_timeline(
            grouped_phased,
            f"Late-State Phase Timeline - {scheme['name']}",
            scheme_dir / f"{scheme['name']}_phase_timeline.png",
        )

    write_csv(summary_rows, output_dir / "late_state_variant_occupancy_summary.csv")
    write_csv(all_rows, output_dir / "all_late_state_variant_checkpoint_metrics.csv")
    write_markdown(summary_rows, output_dir / "LATE_STATE_VARIANTS_SUMMARY.md")
    print(f"Saved late-state variants to {output_dir}")


if __name__ == "__main__":
    main()
