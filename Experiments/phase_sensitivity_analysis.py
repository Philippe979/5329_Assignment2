import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


PHASE_COLORS = {
    "Pre-learning": "#9d9d9d",
    "Robustness Formation": "#4c78a8",
    "Weak Robustness Formation": "#a6bddb",
    "Robustness Stabilization / Peak": "#59a14f",
    "Weak Robustness Peak": "#b8e186",
    "Late Degradation": "#e15759",
    "Late Divergence": "#b07aa1",
    "Late Stabilization": "#f28e2b",
}


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return sorted(rows, key=lambda row: float(row["progress"]))


def load_grouped_rows(results_dir):
    grouped = {}
    for path in sorted(Path(results_dir).glob("*_dynamic/metrics/*_checkpoint_metrics.csv")):
        rows = read_rows(path)
        if rows:
            grouped[rows[0]["model"]] = rows
    return grouped


def phase_widths(rows):
    widths = []
    for index, row in enumerate(rows):
        start = float(row["progress"])
        end = float(rows[index + 1]["progress"]) if index < len(rows) - 1 else 1.0
        widths.append(max(0.0, end - start))
    return widths


def find_valid_start(rows, clean_threshold):
    for index, row in enumerate(rows):
        if float(row["test_clean_acc"]) >= clean_threshold:
            return index
    return None


def contiguous_peak_interval(rows, valid_indices, peak_local_index, threshold):
    robust = [float(rows[index]["test_robust_acc"]) for index in valid_indices]
    start = peak_local_index
    while start > 0 and robust[start - 1] >= threshold:
        start -= 1
    end = peak_local_index
    while end < len(robust) - 1 and robust[end + 1] >= threshold:
        end += 1
    return valid_indices[start], valid_indices[end]


def late_label(ra_peak, ra_final, delta_final, delta_peak_end, scheme):
    ra_drop_abs = ra_peak - ra_final
    ra_drop_rel = ra_drop_abs / ra_peak if ra_peak > 0 else 0.0
    delta_growth = delta_final - delta_peak_end

    if scheme["drop_mode"] == "legacy":
        if ra_drop_abs > scheme["epsilon_ra_abs"]:
            return "Late Degradation"
    else:
        if (
            ra_drop_abs >= scheme["epsilon_ra_abs"]
            and ra_drop_rel >= scheme["epsilon_ra_rel"]
        ):
            return "Late Degradation"

    if delta_growth > scheme["epsilon_delta"]:
        return "Late Divergence"
    return "Late Stabilization"


def assign_scheme(rows, scheme):
    rows = [dict(row) for row in rows]
    widths = phase_widths(rows)
    valid_start = find_valid_start(rows, scheme["clean_threshold"])
    for row in rows:
        row["phase"] = "Pre-learning"
        row["valid_for_analysis"] = "False"

    if valid_start is None:
        return rows, {
            "valid_start": "",
            "RA_peak": 0.0,
            "p_peak": "",
            "dominant_phase": "Pre-learning",
            "phase_completeness": 1,
        }

    valid_indices = list(range(valid_start, len(rows)))
    robust = [float(rows[index]["test_robust_acc"]) for index in valid_indices]
    peak_local_index = max(range(len(valid_indices)), key=lambda idx: robust[idx])
    peak_index = valid_indices[peak_local_index]
    ra_peak = float(rows[peak_index]["test_robust_acc"])
    ra_final = float(rows[-1]["test_robust_acc"])
    delta_final = float(rows[-1]["gap"])

    if scheme["peak_mode"] == "relative":
        threshold = scheme["alpha"] * ra_peak
        meaningful_peak = True
    else:
        threshold = max(scheme["alpha"] * ra_peak, scheme["tau_ra"])
        meaningful_peak = ra_peak >= scheme["tau_ra"]

    if scheme["peak_mode"] == "ordered":
        phases = assign_ordered_interval_phases(rows, valid_start, scheme)
        return phases, summarize(rows, phases, widths, valid_start, ra_peak, peak_index)

    if scheme["peak_mode"] == "trend":
        phases = assign_trend_phases(rows, valid_start, scheme)
        return phases, summarize(rows, phases, widths, valid_start, ra_peak, peak_index)

    peak_start, peak_end = contiguous_peak_interval(rows, valid_indices, peak_local_index, threshold)
    peak_start_progress = float(rows[peak_start]["progress"])
    valid_progress = float(rows[valid_start]["progress"])
    formation_growth = float(rows[peak_start]["test_robust_acc"]) - float(rows[valid_start]["test_robust_acc"])
    meaningful_formation = formation_growth >= scheme["tau_growth"]

    label_after_peak = late_label(
        ra_peak=ra_peak,
        ra_final=ra_final,
        delta_final=delta_final,
        delta_peak_end=float(rows[peak_end]["gap"]),
        scheme=scheme,
    )

    for index, row in enumerate(rows):
        progress = float(row["progress"])
        clean_acc = float(row["test_clean_acc"])
        row["valid_for_analysis"] = str(clean_acc >= scheme["clean_threshold"])
        if clean_acc < scheme["clean_threshold"]:
            row["phase"] = "Pre-learning"
        elif index < peak_start:
            row["phase"] = (
                "Robustness Formation" if meaningful_formation else "Weak Robustness Formation"
            )
        elif peak_start <= index <= peak_end:
            row["phase"] = (
                "Robustness Stabilization / Peak" if meaningful_peak else "Weak Robustness Peak"
            )
        elif progress > float(rows[peak_end]["progress"]):
            row["phase"] = label_after_peak
        else:
            row["phase"] = label_after_peak

    return rows, summarize(rows, rows, widths, valid_start, ra_peak, peak_index)


def assign_ordered_interval_phases(rows, valid_start, scheme):
    rows = [dict(row) for row in rows]
    valid_indices = list(range(valid_start, len(rows)))
    robust = [float(rows[index]["test_robust_acc"]) for index in valid_indices]
    peak_local_index = max(range(len(valid_indices)), key=lambda idx: robust[idx])
    peak_index = valid_indices[peak_local_index]
    ra_peak = float(rows[peak_index]["test_robust_acc"])
    ra_final = float(rows[-1]["test_robust_acc"])
    threshold = max(scheme["alpha"] * ra_peak, scheme["tau_ra"])
    meaningful_peak = ra_peak >= scheme["tau_ra"]

    if meaningful_peak:
        peak_start, peak_end = contiguous_peak_interval(rows, valid_indices, peak_local_index, threshold)
    else:
        peak_start = peak_index
        peak_end = peak_index

    formation_growth = float(rows[peak_start]["test_robust_acc"]) - float(rows[valid_start]["test_robust_acc"])
    meaningful_formation = formation_growth >= scheme["tau_growth"]
    label_after_peak = late_label(
        ra_peak=ra_peak,
        ra_final=ra_final,
        delta_final=float(rows[-1]["gap"]),
        delta_peak_end=float(rows[peak_end]["gap"]),
        scheme=scheme,
    )

    for index, row in enumerate(rows):
        clean_acc = float(row["test_clean_acc"])
        row["valid_for_analysis"] = str(clean_acc >= scheme["clean_threshold"])
        if index < valid_start:
            row["phase"] = "Pre-learning"
        elif index < peak_start:
            row["phase"] = (
                "Robustness Formation" if meaningful_formation else "Weak Robustness Formation"
            )
        elif index <= peak_end:
            row["phase"] = (
                "Robustness Stabilization / Peak" if meaningful_peak else "Weak Robustness Peak"
            )
        else:
            row["phase"] = label_after_peak
    return rows


def assign_trend_phases(rows, valid_start, scheme):
    rows = [dict(row) for row in rows]
    valid_indices = list(range(valid_start, len(rows)))
    robust_values = [float(row["test_robust_acc"]) for row in rows]
    peak_index = max(valid_indices, key=lambda idx: robust_values[idx])
    ra_peak = robust_values[peak_index]
    meaningful_peak = ra_peak >= scheme["tau_ra"]

    for index, row in enumerate(rows):
        clean_acc = float(row["test_clean_acc"])
        row["valid_for_analysis"] = str(clean_acc >= scheme["clean_threshold"])
        if clean_acc < scheme["clean_threshold"]:
            row["phase"] = "Pre-learning"
            continue
        if index == valid_start:
            row["phase"] = "Robustness Formation" if meaningful_peak else "Weak Robustness Formation"
            continue
        dra = robust_values[index] - robust_values[index - 1]
        ddelta = float(rows[index]["gap"]) - float(rows[index - 1]["gap"])
        if index <= peak_index and dra > scheme["eta_ra"]:
            row["phase"] = "Robustness Formation" if meaningful_peak else "Weak Robustness Formation"
        elif abs(dra) <= scheme["eta_stable"] and robust_values[index] >= scheme["tau_ra"]:
            row["phase"] = "Robustness Stabilization / Peak"
        elif dra < -scheme["eta_ra"] and ddelta > scheme["eta_delta"]:
            row["phase"] = "Late Degradation"
        elif ddelta > scheme["eta_delta"]:
            row["phase"] = "Late Divergence"
        else:
            row["phase"] = "Late Stabilization"
    return rows


def summarize(original_rows, phased_rows, widths, valid_start, ra_peak, peak_index):
    occupancy = {phase: 0.0 for phase in PHASE_COLORS}
    for row, width in zip(phased_rows, widths):
        occupancy[row["phase"]] = occupancy.get(row["phase"], 0.0) + width
    total = sum(widths) or 1.0
    normalized = {phase: value / total for phase, value in occupancy.items()}
    dominant_phase = max(normalized, key=lambda phase: normalized[phase])
    meaningful_phases = sum(1 for value in normalized.values() if value >= 0.05)
    return {
        "valid_start": original_rows[valid_start]["progress"] if valid_start is not None else "",
        "RA_peak": ra_peak,
        "p_peak": original_rows[peak_index]["progress"] if peak_index is not None else "",
        "dominant_phase": dominant_phase,
        "phase_completeness": meaningful_phases,
        **{f"occ_{phase}": value for phase, value in normalized.items()},
    }


def write_csv(rows, path):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_timeline(grouped_phased, title, output_path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    model_names = list(grouped_phased.keys())
    fig_height = max(5.2, 0.48 * len(model_names) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    for y_index, model_name in enumerate(model_names):
        rows = grouped_phased[model_name]
        for index, row in enumerate(rows):
            start = float(row["progress"])
            end = float(rows[index + 1]["progress"]) if index < len(rows) - 1 else 1.0
            width = max(end - start, 0.012)
            phase = row["phase"]
            ax.barh(
                y_index,
                width,
                left=start,
                height=0.55,
                color=PHASE_COLORS.get(phase, "0.7"),
                edgecolor="white",
                linewidth=0.7,
            )

    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Training Progress")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    handles = [mpatches.Patch(color=color, label=phase) for phase, color in PHASE_COLORS.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    results_dir = PROJECT_ROOT / "Results"
    output_dir = results_dir / "phase_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = load_grouped_rows(results_dir)
    schemes = [
        {
            "name": "A_legacy_relative_peak",
            "clean_threshold": 0.30,
            "peak_mode": "relative",
            "alpha": 0.95,
            "tau_ra": 0.0,
            "tau_growth": 0.0,
            "drop_mode": "legacy",
            "epsilon_ra_abs": 0.01,
            "epsilon_ra_rel": 0.0,
            "epsilon_delta": 0.05,
        },
        *[
            {
                "name": f"B_literature_tauRA_{str(tau).replace('.', '')}",
                "clean_threshold": 0.30,
                "peak_mode": "absolute",
                "alpha": 0.95,
                "tau_ra": tau,
                "tau_growth": 0.02,
                "drop_mode": "absolute_relative",
                "epsilon_ra_abs": 0.02,
                "epsilon_ra_rel": 0.20,
                "epsilon_delta": 0.05,
            }
            for tau in (0.03, 0.05, 0.08, 0.10)
        ],
        {
            "name": "C_trend_based",
            "clean_threshold": 0.30,
            "peak_mode": "trend",
            "alpha": 0.95,
            "tau_ra": 0.05,
            "tau_growth": 0.02,
            "drop_mode": "absolute_relative",
            "epsilon_ra_abs": 0.02,
            "epsilon_ra_rel": 0.20,
            "epsilon_delta": 0.05,
            "eta_ra": 0.005,
            "eta_stable": 0.005,
            "eta_delta": 0.01,
        },
        {
            "name": "D_ordered_interval_tauRA_005",
            "clean_threshold": 0.30,
            "peak_mode": "ordered",
            "alpha": 0.95,
            "tau_ra": 0.05,
            "tau_growth": 0.02,
            "drop_mode": "absolute_relative",
            "epsilon_ra_abs": 0.02,
            "epsilon_ra_rel": 0.20,
            "epsilon_delta": 0.05,
        },
    ]

    summary_rows = []
    all_phased_rows = []

    for scheme in schemes:
        grouped_phased = {}
        for model_name, rows in grouped.items():
            phased, summary = assign_scheme(rows, scheme)
            grouped_phased[model_name] = phased
            for row in phased:
                new_row = dict(row)
                new_row["scheme"] = scheme["name"]
                all_phased_rows.append(new_row)
            summary_rows.append({"scheme": scheme["name"], "model": model_name, **summary})

        write_csv(
            [row for row in all_phased_rows if row["scheme"] == scheme["name"]],
            output_dir / f"{scheme['name']}_checkpoint_metrics_phased.csv",
        )
        plot_timeline(
            grouped_phased,
            title=f"Phase Sensitivity Timeline - {scheme['name']}",
            output_path=output_dir / f"{scheme['name']}_phase_timeline.png",
        )

    write_csv(summary_rows, output_dir / "phase_occupancy_by_scheme.csv")
    print(f"Saved phase sensitivity outputs to {output_dir}")


if __name__ == "__main__":
    main()
