import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PHASE_COLORS = {
    "Pre-learning": "#9d9d9d",
    "Robustness Formation": "#4c78a8",
    "Robustness Maturation": "#72b7b2",
    "Robustness Stabilization / Peak": "#59a14f",
    "Early Robustness Decay": "#8cd17d",
    "Late Degradation": "#e15759",
    "Late Divergence": "#b07aa1",
    "Late Stabilization": "#f28e2b",
}


def read_rows(path: Path) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return sorted(rows, key=lambda row: float(row["progress"]))


def write_csv(rows: Iterable[Dict], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_grouped_rows(results_dir: Path) -> Dict[str, List[Dict]]:
    grouped = {}
    for path in sorted(results_dir.glob("*_dynamic/metrics/*_checkpoint_metrics.csv")):
        rows = read_rows(path)
        if rows:
            grouped[rows[0]["model"]] = rows
    return grouped


def phase_widths(rows: Sequence[Dict]) -> List[float]:
    widths = []
    for index, row in enumerate(rows):
        start = float(row["progress"])
        end = float(rows[index + 1]["progress"]) if index < len(rows) - 1 else 1.0
        widths.append(max(0.0, end - start))
    return widths


def valid_start_index(rows: Sequence[Dict], clean_threshold: float) -> int | None:
    for index, row in enumerate(rows):
        if float(row["test_clean_acc"]) >= clean_threshold:
            return index
    return None


def contiguous_interval(values: Sequence[float], center: int, threshold: float) -> tuple[int, int]:
    start = center
    while start > 0 and values[start - 1] >= threshold:
        start -= 1
    end = center
    while end < len(values) - 1 and values[end + 1] >= threshold:
        end += 1
    return start, end


def moving_average(values: Sequence[float], window: int) -> List[float]:
    half = window // 2
    output = []
    for index in range(len(values)):
        start = max(0, index - half)
        end = min(len(values), index + half + 1)
        output.append(sum(values[start:end]) / (end - start))
    return output


def slopes(progress: Sequence[float], values: Sequence[float]) -> List[float]:
    output = [0.0]
    for index in range(1, len(values)):
        width = max(progress[index] - progress[index - 1], 1e-8)
        output.append((values[index] - values[index - 1]) / width)
    return output


def late_label(rows: Sequence[Dict], peak_index: int, end_index: int, scheme: Dict) -> str:
    ra_peak = float(rows[peak_index]["test_robust_acc"])
    ra_final = float(rows[-1]["test_robust_acc"])
    gap_final = float(rows[-1]["gap"])
    gap_end = float(rows[end_index]["gap"])
    drop_abs = ra_peak - ra_final
    drop_rel = drop_abs / ra_peak if ra_peak > 0 else 0.0
    gap_growth = gap_final - gap_end
    if drop_abs >= scheme["drop_abs"] and drop_rel >= scheme["drop_rel"]:
        return "Late Degradation"
    if gap_growth >= scheme["gap_growth"]:
        return "Late Divergence"
    return "Late Stabilization"


def assign_peak_rule(rows: Sequence[Dict], scheme: Dict) -> List[Dict]:
    rows = [dict(row) for row in rows]
    valid_start = valid_start_index(rows, scheme["clean_threshold"])
    for row in rows:
        row["phase"] = "Pre-learning"
        row["valid_for_analysis"] = str(float(row["test_clean_acc"]) >= scheme["clean_threshold"])
    if valid_start is None:
        return rows

    robust = [float(row["test_robust_acc"]) for row in rows]
    valid_indices = list(range(valid_start, len(rows)))
    peak_index = max(valid_indices, key=lambda index: robust[index])
    threshold = scheme["peak_ratio"] * robust[peak_index]
    local_values = [robust[index] for index in valid_indices]
    local_peak = valid_indices.index(peak_index)
    local_start, local_end = contiguous_interval(local_values, local_peak, threshold)
    peak_start = valid_indices[local_start]
    peak_end = valid_indices[local_end]
    label_after_peak = late_label(rows, peak_index, peak_end, scheme)

    for index, row in enumerate(rows):
        if index < valid_start:
            row["phase"] = "Pre-learning"
        elif index < peak_start:
            row["phase"] = "Robustness Formation"
        elif index <= peak_end:
            row["phase"] = "Robustness Stabilization / Peak"
        else:
            row["phase"] = label_after_peak
    return rows


def first_consecutive(indices: Sequence[int], predicate, count: int) -> int | None:
    run = 0
    start = None
    for index in indices:
        if predicate(index):
            if run == 0:
                start = index
            run += 1
            if run >= count:
                return start
        else:
            run = 0
            start = None
    return None


def assign_slope_plateau_rule(rows: Sequence[Dict], scheme: Dict) -> List[Dict]:
    rows = [dict(row) for row in rows]
    valid_start = valid_start_index(rows, scheme["clean_threshold"])
    for row in rows:
        row["phase"] = "Pre-learning"
        row["valid_for_analysis"] = str(float(row["test_clean_acc"]) >= scheme["clean_threshold"])
    if valid_start is None:
        return rows

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

    late_start = first_consecutive(
        range(max(plateau_start + 1, peak_index + 1), len(rows)),
        lambda index: (
            smooth[index] <= ra_peak - scheme["drop_abs"]
            or (slope[index] < -scheme["negative_slope"] and float(rows[index]["gap"]) > float(rows[index - 1]["gap"]))
        ),
        scheme["consecutive"],
    )

    if late_start is None:
        plateau_end = len(rows) - 1
        label_after_peak = "Late Stabilization"
    else:
        plateau_end = max(plateau_start, late_start - 1)
        label_after_peak = late_label(rows, peak_index, plateau_end, scheme)

    for index, row in enumerate(rows):
        if index < valid_start:
            row["phase"] = "Pre-learning"
        elif index < plateau_start:
            row["phase"] = "Robustness Formation"
        elif index <= plateau_end:
            row["phase"] = "Robustness Stabilization / Peak"
        else:
            row["phase"] = label_after_peak
    return rows


def assign_maturation_rule(rows: Sequence[Dict], scheme: Dict) -> List[Dict]:
    rows = [dict(row) for row in rows]
    valid_start = valid_start_index(rows, scheme["clean_threshold"])
    for row in rows:
        row["phase"] = "Pre-learning"
        row["valid_for_analysis"] = str(float(row["test_clean_acc"]) >= scheme["clean_threshold"])
    if valid_start is None:
        return rows

    robust = [float(row["test_robust_acc"]) for row in rows]
    valid_indices = list(range(valid_start, len(rows)))
    peak_index = max(valid_indices, key=lambda index: robust[index])
    ra_peak = robust[peak_index]
    formation_cutoff = scheme["formation_ratio"] * ra_peak
    maturation_cutoff = scheme["maturation_ratio"] * ra_peak

    stabilization_start = next(
        (index for index in valid_indices if index <= peak_index and robust[index] >= maturation_cutoff),
        peak_index,
    )
    late_cutoff = peak_index
    label_after_peak = late_label(rows, peak_index, peak_index, scheme)

    for index, row in enumerate(rows):
        if index < valid_start:
            row["phase"] = "Pre-learning"
        elif index <= peak_index and robust[index] < formation_cutoff:
            row["phase"] = "Robustness Formation"
        elif index < stabilization_start:
            row["phase"] = "Robustness Maturation"
        elif index <= late_cutoff:
            row["phase"] = "Robustness Stabilization / Peak"
        else:
            row["phase"] = label_after_peak
    return rows


def assign_gap_late_rule(rows: Sequence[Dict], scheme: Dict) -> List[Dict]:
    rows = assign_peak_rule(rows, scheme)
    valid_start = next((i for i, row in enumerate(rows) if row["phase"] != "Pre-learning"), None)
    if valid_start is None:
        return rows
    robust = [float(row["test_robust_acc"]) for row in rows]
    gaps = [float(row["gap"]) for row in rows]
    valid_indices = list(range(valid_start, len(rows)))
    peak_index = max(valid_indices, key=lambda index: robust[index])

    late_start = first_consecutive(
        range(valid_start + 1, len(rows)),
        lambda index: (
            gaps[index] - gaps[max(valid_start, index - scheme["gap_window"])] >= scheme["gap_window_growth"]
            and robust[index] >= scheme["tau_ra"]
        ),
        scheme["consecutive"],
    )
    if late_start is not None and late_start > peak_index:
        label = late_label(rows, peak_index, late_start - 1, scheme)
        for index in range(late_start, len(rows)):
            rows[index]["phase"] = label
    return rows


def summarize(model: str, scheme_name: str, rows: Sequence[Dict]) -> Dict:
    widths = phase_widths(rows)
    occupancy = {phase: 0.0 for phase in PHASE_COLORS}
    for row, width in zip(rows, widths):
        occupancy[row["phase"]] = occupancy.get(row["phase"], 0.0) + width
    total = sum(widths) or 1.0
    normalized = {phase: value / total for phase, value in occupancy.items()}
    dominant_phase = max(normalized, key=lambda phase: normalized[phase])
    meaningful = sum(1 for value in normalized.values() if value >= 0.05)
    robust = [float(row["test_robust_acc"]) for row in rows]
    peak_index = max(range(len(rows)), key=lambda index: robust[index])
    return {
        "scheme": scheme_name,
        "model": model,
        "RA_peak": robust[peak_index],
        "p_peak": rows[peak_index]["progress"],
        "RA_final": robust[-1],
        "dominant_phase": dominant_phase,
        "phase_completeness": meaningful,
        **{f"occ_{phase}": round(value, 6) for phase, value in normalized.items()},
    }


def plot_timeline(grouped_rows: Dict[str, List[Dict]], title: str, output_path: Path) -> None:
    model_names = list(grouped_rows.keys())
    fig_height = max(5.2, 0.48 * len(model_names) + 1.8)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    for y_index, model_name in enumerate(model_names):
        rows = grouped_rows[model_name]
        for index, row in enumerate(rows):
            start = float(row["progress"])
            end = float(rows[index + 1]["progress"]) if index < len(rows) - 1 else 1.0
            ax.barh(
                y_index,
                max(0.0, end - start),
                left=start,
                height=0.55,
                color=PHASE_COLORS.get(row["phase"], "0.7"),
                edgecolor="white",
                linewidth=0.5,
            )

    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Training Progress")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    handles = [mpatches.Patch(color=color, label=phase) for phase, color in PHASE_COLORS.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results_dir = Path("Results/adaptive_protocol/final")
    output_dir = Path("Results/adaptive_protocol/phase_rule_experiments")
    grouped = load_grouped_rows(results_dir)
    if not grouped:
        raise FileNotFoundError(f"No checkpoint metrics found under {results_dir}")

    base = {
        "clean_threshold": 0.30,
        "drop_abs": 0.02,
        "drop_rel": 0.20,
        "gap_growth": 0.05,
        "tau_ra": 0.05,
    }
    schemes = [
        {"name": "peak_ratio_095", "kind": "peak", **base, "peak_ratio": 0.95},
        {"name": "peak_ratio_090", "kind": "peak", **base, "peak_ratio": 0.90},
        {"name": "peak_ratio_085", "kind": "peak", **base, "peak_ratio": 0.85},
        {
            "name": "slope_plateau_balanced",
            "kind": "slope",
            **base,
            "smooth_window": 5,
            "plateau_ratio": 0.85,
            "fallback_ratio": 0.90,
            "stable_slope": 0.15,
            "negative_slope": 0.25,
            "consecutive": 3,
        },
        {
            "name": "slope_plateau_loose",
            "kind": "slope",
            **base,
            "smooth_window": 7,
            "plateau_ratio": 0.80,
            "fallback_ratio": 0.85,
            "stable_slope": 0.25,
            "negative_slope": 0.20,
            "consecutive": 3,
        },
        {
            "name": "maturation_60_85",
            "kind": "maturation",
            **base,
            "formation_ratio": 0.60,
            "maturation_ratio": 0.85,
        },
        {
            "name": "maturation_70_90",
            "kind": "maturation",
            **base,
            "formation_ratio": 0.70,
            "maturation_ratio": 0.90,
        },
        {
            "name": "gap_late_peak090",
            "kind": "gap_late",
            **base,
            "peak_ratio": 0.90,
            "gap_window": 5,
            "gap_window_growth": 0.04,
            "consecutive": 3,
        },
    ]

    summary_rows = []
    all_rows = []
    for scheme in schemes:
        grouped_phased = {}
        for model_name, rows in grouped.items():
            if scheme["kind"] == "peak":
                phased = assign_peak_rule(rows, scheme)
            elif scheme["kind"] == "slope":
                phased = assign_slope_plateau_rule(rows, scheme)
            elif scheme["kind"] == "maturation":
                phased = assign_maturation_rule(rows, scheme)
            elif scheme["kind"] == "gap_late":
                phased = assign_gap_late_rule(rows, scheme)
            else:
                raise ValueError(f"Unknown scheme kind: {scheme['kind']}")

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
            f"Phase Timeline - {scheme['name']}",
            scheme_dir / f"{scheme['name']}_phase_timeline.png",
        )

    write_csv(summary_rows, output_dir / "phase_rule_occupancy_summary.csv")
    write_csv(all_rows, output_dir / "all_phase_rule_checkpoint_metrics.csv")
    write_markdown_summary(summary_rows, output_dir / "PHASE_RULE_EXPERIMENTS_SUMMARY.md")
    print(f"Saved phase rule experiments to {output_dir}")


def write_markdown_summary(summary_rows: Sequence[Dict], path: Path) -> None:
    schemes = sorted({row["scheme"] for row in summary_rows})
    lines = ["# Phase Rule Experiments Summary", ""]
    lines.append("The table below reports dominant phase counts across 18 final adaptive-protocol models.")
    lines.append("")
    lines.append("| scheme | dominant formation | dominant stabilization | dominant late | avg phase completeness |")
    lines.append("|---|---:|---:|---:|---:|")
    for scheme in schemes:
        rows = [row for row in summary_rows if row["scheme"] == scheme]
        formation = sum(1 for row in rows if row["dominant_phase"] == "Robustness Formation")
        stabilization = sum(1 for row in rows if row["dominant_phase"] == "Robustness Stabilization / Peak")
        late = sum(1 for row in rows if row["dominant_phase"].startswith("Late"))
        avg_completeness = sum(int(row["phase_completeness"]) for row in rows) / len(rows)
        lines.append(f"| {scheme} | {formation} | {stabilization} | {late} | {avg_completeness:.2f} |")
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("")
    lines.append("- `peak_ratio_*` tests whether the original peak-centered rule is too strict.")
    lines.append("- `slope_plateau_*` treats low robust-accuracy slope as stabilization, even before the final peak.")
    lines.append("- `maturation_*` splits long formation into early formation and robustness maturation.")
    lines.append("- `gap_late_peak090` keeps a peak rule but lets gap expansion trigger late-state diagnosis.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
