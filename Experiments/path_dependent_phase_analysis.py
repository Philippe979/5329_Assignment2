import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Evaluation import plot_phase_timeline


PATH_PHASES = [
    "Pre-learning",
    "Robustness Formation",
    "Robustness Maturation",
    "Robustness Stabilization / Peak",
    "Early Robustness Decay",
    "Late Degradation",
    "Late Divergence",
    "Late Stabilization",
]


CONFIG = {
    "clean_threshold": 0.30,
    "tau_ra": 0.05,
    "smooth_window": 5,
    "formation_ratio": 0.60,
    "maturation_ratio": 0.85,
    "plateau_ratio": 0.90,
    "early_peak_progress": 0.15,
    "min_late_width": 0.05,
    "drop_abs": 0.02,
    "drop_rel": 0.20,
    "gap_growth": 0.05,
    "stable_slope": 0.15,
    "growth_slope": 0.20,
    "decay_slope": 0.20,
    "turning_slope": 0.15,
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


def read_selected_protocol(path: Path) -> Dict[str, Dict]:
    with open(path, newline="", encoding="utf-8") as file:
        return {row["model"]: row for row in csv.DictReader(file)}


def first_index(indices: Sequence[int], predicate, fallback: int) -> int:
    for index in indices:
        if predicate(index):
            return index
    return fallback


def count_turning_points(values: Sequence[float], threshold: float) -> int:
    signs = []
    for index in range(1, len(values)):
        delta = values[index] - values[index - 1]
        if delta > threshold:
            signs.append(1)
        elif delta < -threshold:
            signs.append(-1)
        else:
            signs.append(0)
    compact = [sign for sign in signs if sign != 0]
    return sum(1 for prev, curr in zip(compact, compact[1:]) if prev != curr)


def late_label(ra_peak: float, ra_value: float, gap_peak: float, gap_value: float) -> str:
    drop_abs = ra_peak - ra_value
    drop_rel = drop_abs / ra_peak if ra_peak > 0 else 0.0
    if drop_abs >= CONFIG["drop_abs"] and drop_rel >= CONFIG["drop_rel"]:
        return "Late Degradation"
    if gap_value - gap_peak >= CONFIG["gap_growth"]:
        return "Late Divergence"
    return "Late Stabilization"


def compute_path_descriptor(model_name: str, rows: Sequence[Dict], protocol: Dict) -> Dict:
    progress = [float(row["progress"]) for row in rows]
    clean = [float(row["test_clean_acc"]) for row in rows]
    robust = [float(row["test_robust_acc"]) for row in rows]
    gap = [float(row["gap"]) for row in rows]
    smooth = moving_average(robust, CONFIG["smooth_window"])
    slope = slopes(progress, smooth)
    valid_start = valid_start_index(rows, CONFIG["clean_threshold"])

    if valid_start is None:
        return {
            "model": model_name,
            "epsilon_label": protocol.get("epsilon_label", ""),
            "path_type": "clean_invalid",
            "p_valid": "",
            "p_peak": "",
            "p_form_end": "",
            "p_maturation_end": "",
            "p_late_start": "",
            "RA_peak": max(robust),
            "RA_final": robust[-1],
            "CA_final": clean[-1],
            "Delta_final": gap[-1],
            "turning_points": count_turning_points(smooth, CONFIG["turning_slope"] * 0.01),
            "diagnosis": "clean accuracy does not enter valid analysis window",
        }

    valid_indices = list(range(valid_start, len(rows)))
    peak_index = max(valid_indices, key=lambda index: smooth[index])
    ra_peak = smooth[peak_index]
    ra_final = smooth[-1]
    gap_peak = gap[peak_index]
    gap_final = gap[-1]

    form_end = first_index(
        [index for index in valid_indices if index <= peak_index],
        lambda index: smooth[index] >= CONFIG["formation_ratio"] * ra_peak,
        peak_index,
    )
    maturation_end = first_index(
        [index for index in valid_indices if index <= peak_index],
        lambda index: smooth[index] >= CONFIG["maturation_ratio"] * ra_peak,
        peak_index,
    )
    plateau_start = first_index(
        [index for index in valid_indices if index <= peak_index],
        lambda index: (
            smooth[index] >= CONFIG["plateau_ratio"] * ra_peak
            and abs(slope[index]) <= CONFIG["stable_slope"]
        ),
        maturation_end,
    )
    late_start = next(
        (
            index for index in range(peak_index + 1, len(rows))
            if progress[index] - progress[peak_index] >= CONFIG["min_late_width"]
        ),
        len(rows),
    )

    early_peak = progress[peak_index] <= CONFIG["early_peak_progress"]
    robust_drop = (ra_peak - ra_final) >= CONFIG["drop_abs"]
    robust_drop_rel = ((ra_peak - ra_final) / ra_peak) if ra_peak > 0 else 0.0
    gap_growth = gap_final - gap_peak

    if early_peak and robust_drop:
        path_type = "early_robustness_decay"
        diagnosis = "robustness peaks near the valid-window start and then declines"
    elif progress[peak_index] >= 0.85 and not robust_drop:
        path_type = "delayed_maturation"
        diagnosis = "robustness continues forming until late training; limited post-peak window"
    elif robust_drop and robust_drop_rel >= CONFIG["drop_rel"]:
        path_type = "late_degradation"
        diagnosis = "robustness forms then materially drops after peak"
    elif gap_growth >= CONFIG["gap_growth"]:
        path_type = "clean_robust_divergence"
        diagnosis = "clean-robust gap expands after robust peak"
    else:
        path_type = "stable_trajectory"
        diagnosis = "robustness forms and remains stable near final checkpoint"

    return {
        "model": model_name,
        "epsilon_label": protocol.get("epsilon_label", ""),
        "selected_dynamic_type": protocol.get("dynamic_type", ""),
        "path_type": path_type,
        "p_valid": progress[valid_start],
        "p_peak": progress[peak_index],
        "p_form_end": progress[form_end],
        "p_maturation_end": progress[maturation_end],
        "p_plateau_start": progress[plateau_start],
        "p_late_start": progress[late_start] if late_start < len(rows) else "",
        "RA_peak": robust[peak_index],
        "RA_peak_smooth": ra_peak,
        "RA_final": robust[-1],
        "RA_final_smooth": ra_final,
        "RA_drop_smooth": ra_peak - ra_final,
        "CA_final": clean[-1],
        "Delta_peak": gap_peak,
        "Delta_final": gap_final,
        "Delta_growth_after_peak": gap_growth,
        "turning_points": count_turning_points(smooth, CONFIG["turning_slope"] * 0.01),
        "diagnosis": diagnosis,
    }


def assign_path_phases(rows: Sequence[Dict], descriptor: Dict) -> List[Dict]:
    rows = [dict(row) for row in rows]
    if descriptor["path_type"] == "clean_invalid":
        for row in rows:
            row["phase"] = "Pre-learning"
            row["path_type"] = descriptor["path_type"]
        return rows

    p_valid = float(descriptor["p_valid"])
    p_form_end = float(descriptor["p_form_end"])
    p_maturation_end = float(descriptor["p_maturation_end"])
    p_plateau_start = float(descriptor["p_plateau_start"])
    p_peak = float(descriptor["p_peak"])
    p_late_start = float(descriptor["p_late_start"]) if descriptor["p_late_start"] != "" else 1.01
    ra_peak = float(descriptor["RA_peak_smooth"])
    gap_peak = float(descriptor["Delta_peak"])

    if descriptor["path_type"] == "early_robustness_decay":
        for row in rows:
            progress = float(row["progress"])
            row["path_type"] = descriptor["path_type"]
            if progress < p_valid:
                row["phase"] = "Pre-learning"
            elif progress <= p_peak:
                row["phase"] = "Robustness Stabilization / Peak"
            else:
                row["phase"] = "Early Robustness Decay"
        return rows

    for row in rows:
        progress = float(row["progress"])
        row["path_type"] = descriptor["path_type"]
        if progress < p_valid:
            phase = "Pre-learning"
        elif progress < p_form_end:
            phase = "Robustness Formation"
        elif progress < p_maturation_end:
            phase = "Robustness Maturation"
        elif progress < p_late_start:
            phase = "Robustness Stabilization / Peak"
        else:
            phase = late_label(
                ra_peak=ra_peak,
                ra_value=float(row["test_robust_acc"]),
                gap_peak=gap_peak,
                gap_value=float(row["gap"]),
            )
        row["phase"] = phase
    return rows


def summarize_phases(model: str, rows: Sequence[Dict], descriptor: Dict) -> Dict:
    widths = phase_widths(rows)
    occupancy = {phase: 0.0 for phase in PATH_PHASES}
    for row, width in zip(rows, widths):
        occupancy[row["phase"]] = occupancy.get(row["phase"], 0.0) + width
    total = sum(widths) or 1.0
    normalized = {phase: value / total for phase, value in occupancy.items()}
    return {
        "model": model,
        "epsilon_label": descriptor.get("epsilon_label", ""),
        "path_type": descriptor["path_type"],
        "dominant_phase": max(normalized, key=lambda phase: normalized[phase]),
        "phase_completeness": sum(1 for value in normalized.values() if value >= 0.05),
        **{f"occ_{phase}": round(value, 6) for phase, value in normalized.items()},
    }


def write_markdown(descriptors: Sequence[Dict], phase_summary: Sequence[Dict], path: Path) -> None:
    path_counts = {}
    for row in descriptors:
        path_counts[row["path_type"]] = path_counts.get(row["path_type"], 0) + 1
    lines = [
        "# Path-Dependent Phase Analysis",
        "",
        "This analysis treats dynamic robustness as a trajectory-state problem rather than a point-wise thresholding problem.",
        "",
        "Path type counts:",
        "",
    ]
    for path_type, count in sorted(path_counts.items()):
        lines.append(f"- {path_type}: {count}")
    lines.extend([
        "",
        "| model | epsilon | path type | p_peak | RA_peak | RA_final | diagnosis |",
        "|---|---|---|---:|---:|---:|---|",
    ])
    for row in sorted(descriptors, key=lambda item: item["model"]):
        lines.append(
            "| {model} | {epsilon_label} | {path_type} | {p_peak} | {RA_peak:.4f} | {RA_final:.4f} | {diagnosis} |".format(
                model=row["model"],
                epsilon_label=row.get("epsilon_label", ""),
                path_type=row["path_type"],
                p_peak=row["p_peak"],
                RA_peak=float(row["RA_peak"]),
                RA_final=float(row["RA_final"]),
                diagnosis=row["diagnosis"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results_dir = Path("Results/adaptive_protocol/final")
    output_dir = Path("Results/adaptive_protocol/path_dependent_phase")
    grouped = load_grouped_rows(results_dir)
    protocol = read_selected_protocol(Path("Results/adaptive_protocol/protocol/selected_model_protocol.csv"))

    descriptors = []
    phase_summaries = []
    all_rows = []
    grouped_phased = {}

    for model_name, rows in grouped.items():
        descriptor = compute_path_descriptor(model_name, rows, protocol.get(model_name, {}))
        phased = assign_path_phases(rows, descriptor)
        descriptors.append(descriptor)
        phase_summaries.append(summarize_phases(model_name, phased, descriptor))
        grouped_phased[model_name] = phased
        for row in phased:
            new_row = dict(row)
            new_row["epsilon_label"] = descriptor.get("epsilon_label", "")
            all_rows.append(new_row)

    write_csv(descriptors, output_dir / "path_descriptors.csv")
    write_csv(phase_summaries, output_dir / "path_phase_occupancy_summary.csv")
    write_csv(all_rows, output_dir / "path_dependent_checkpoint_metrics_phased.csv")
    plot_phase_timeline(
        grouped_phased,
        "Path-Dependent Dynamic Robustness Phase Timeline",
        str(output_dir / "path_dependent_phase_timeline.png"),
    )
    write_markdown(descriptors, phase_summaries, output_dir / "PATH_DEPENDENT_PHASE_SUMMARY.md")
    print(f"Saved path-dependent phase outputs to {output_dir}")


if __name__ == "__main__":
    main()
