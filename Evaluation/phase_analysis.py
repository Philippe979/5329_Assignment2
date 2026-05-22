from typing import Dict, Iterable, List, Sequence


PRE_LEARNING = "Pre-learning"
FORMATION = "Robustness Formation"
STABILIZATION = "Robustness Stabilization / Peak"
LATE_DEGRADATION = "Late Degradation"
LATE_DIVERGENCE = "Late Divergence"
LATE_STABILIZATION = "Late Stabilization"
NO_VALID_WINDOW = "No Valid Observation Window"


def _sorted_rows(rows: Iterable[Dict]) -> List[Dict]:
    sorted_rows = sorted(rows, key=lambda row: float(row["progress"]))
    if not sorted_rows:
        raise ValueError("At least one trajectory row is required.")
    return sorted_rows


def _trapezoid_area(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    area = 0.0
    for i in range(len(xs) - 1):
        width = xs[i + 1] - xs[i]
        area += 0.5 * (ys[i] + ys[i + 1]) * width
    return area


def _valid_indices(rows: Sequence[Dict], clean_threshold: float) -> List[int]:
    return [
        idx for idx, row in enumerate(rows)
        if float(row["test_clean_acc"]) >= clean_threshold
    ]


def _near_peak_interval(
    robust_acc: Sequence[float],
    peak_index: int,
    peak_threshold: float,
) -> tuple[int, int]:
    start = peak_index
    while start > 0 and robust_acc[start - 1] >= peak_threshold:
        start -= 1

    end = peak_index
    while end < len(robust_acc) - 1 and robust_acc[end + 1] >= peak_threshold:
        end += 1

    return start, end


def _late_phase_label(
    ra_drop: float,
    delta_growth: float,
    collapse_rate: float,
    ra_tolerance: float,
    delta_tolerance: float,
    collapse_tolerance: float,
) -> str:
    if ra_drop > ra_tolerance or collapse_rate > collapse_tolerance:
        return LATE_DEGRADATION
    if delta_growth > delta_tolerance:
        return LATE_DIVERGENCE
    return LATE_STABILIZATION


def compute_dynamic_descriptors(
    rows: Iterable[Dict],
    peak_ratio: float = 0.95,
    clean_analysis_threshold: float = 0.30,
    ra_tolerance: float = 0.02,
    delta_tolerance: float = 0.05,
    collapse_tolerance: float = 0.05,
) -> Dict:
    """
    Compute dynamic robustness descriptors from checkpoint metric rows.

    Expected row fields:
        progress, test_clean_acc, test_robust_acc, gap
    """
    rows = _sorted_rows(rows)
    progress = [float(row["progress"]) for row in rows]
    clean_acc = [float(row["test_clean_acc"]) for row in rows]
    robust_acc = [float(row["test_robust_acc"]) for row in rows]
    gaps = [float(row["gap"]) for row in rows]
    valid_indices = _valid_indices(rows, clean_analysis_threshold)

    ca_final = clean_acc[-1]
    ra_final = robust_acc[-1]
    delta_final = gaps[-1]

    if not valid_indices:
        return {
            "model": rows[0].get("model", ""),
            "level": rows[0].get("level", ""),
            "clean_analysis_threshold": clean_analysis_threshold,
            "valid_start": "",
            "valid_checkpoint_count": 0,
            "RA_peak": 0.0,
            "p_peak": "",
            "RA_final": ra_final,
            "RA_drop": 0.0,
            "Delta_final": delta_final,
            "AURT": 0.0,
            "AUG": 0.0,
            "CA_final": ca_final,
            "formation_speed": 0.0,
            "stability_width": 0.0,
            "collapse_rate": 0.0,
            "peak_start": "",
            "peak_end": "",
            "delta_growth_after_peak": 0.0,
            "phase3_label": NO_VALID_WINDOW,
            "trajectory_pattern": "no_valid_clean_learning",
        }

    valid_start_index = valid_indices[0]
    valid_progress = [progress[idx] for idx in valid_indices]
    valid_robust_acc = [robust_acc[idx] for idx in valid_indices]
    valid_gaps = [gaps[idx] for idx in valid_indices]

    peak_local_index = max(range(len(valid_indices)), key=lambda idx: valid_robust_acc[idx])
    peak_index = valid_indices[peak_local_index]
    ra_peak = robust_acc[peak_index]
    p_peak = progress[peak_index]
    ra_drop = ra_peak - ra_final

    peak_threshold = peak_ratio * ra_peak
    local_peak_start, local_peak_end = _near_peak_interval(
        valid_robust_acc,
        peak_local_index,
        peak_threshold,
    )
    peak_start_index = valid_indices[local_peak_start]
    peak_end_index = valid_indices[local_peak_end]
    peak_start = progress[peak_start_index]
    peak_end = progress[peak_end_index]

    if peak_start > progress[valid_start_index]:
        formation_speed = (
            robust_acc[peak_start_index] - robust_acc[valid_start_index]
        ) / (peak_start - progress[valid_start_index])
    else:
        formation_speed = 0.0

    if progress[-1] > peak_end:
        collapse_rate = max(0.0, ra_peak - ra_final) / (progress[-1] - peak_end)
    else:
        collapse_rate = 0.0

    delta_growth = delta_final - gaps[peak_end_index]
    late_label = _late_phase_label(
        ra_drop=ra_drop,
        delta_growth=delta_growth,
        collapse_rate=collapse_rate,
        ra_tolerance=ra_tolerance,
        delta_tolerance=delta_tolerance,
        collapse_tolerance=collapse_tolerance,
    )

    if ra_peak <= ra_tolerance:
        trajectory_pattern = "no_valid_robustness_formation"
    elif peak_start == progress[valid_start_index] and late_label == LATE_DEGRADATION:
        trajectory_pattern = "early_peak_late_collapse"
    elif formation_speed <= 0.0 and peak_start > progress[valid_start_index]:
        trajectory_pattern = "delayed_or_weak_formation"
    elif late_label == LATE_STABILIZATION:
        trajectory_pattern = "stable_robustness"
    elif late_label == LATE_DIVERGENCE:
        trajectory_pattern = "clean_robust_divergence"
    else:
        trajectory_pattern = "expected_dynamic_pattern"

    return {
        "model": rows[0].get("model", ""),
        "level": rows[0].get("level", ""),
        "clean_analysis_threshold": clean_analysis_threshold,
        "valid_start": progress[valid_start_index],
        "valid_checkpoint_count": len(valid_indices),
        "RA_peak": ra_peak,
        "p_peak": p_peak,
        "RA_final": ra_final,
        "RA_drop": ra_drop,
        "Delta_final": delta_final,
        "AURT": _trapezoid_area(valid_progress, valid_robust_acc),
        "AUG": _trapezoid_area(valid_progress, valid_gaps),
        "CA_final": ca_final,
        "formation_speed": formation_speed,
        "stability_width": peak_end - peak_start,
        "collapse_rate": collapse_rate,
        "peak_start": peak_start,
        "peak_end": peak_end,
        "delta_growth_after_peak": delta_growth,
        "phase3_label": late_label,
        "trajectory_pattern": trajectory_pattern,
    }


def assign_phases(
    rows: Iterable[Dict],
    peak_ratio: float = 0.95,
    clean_analysis_threshold: float = 0.30,
    ra_tolerance: float = 0.02,
    delta_tolerance: float = 0.05,
    collapse_tolerance: float = 0.05,
) -> List[Dict]:
    """
    Add a phase label to each checkpoint row.
    """
    rows = _sorted_rows(rows)
    descriptor = compute_dynamic_descriptors(
        rows,
        peak_ratio=peak_ratio,
        clean_analysis_threshold=clean_analysis_threshold,
        ra_tolerance=ra_tolerance,
        delta_tolerance=delta_tolerance,
        collapse_tolerance=collapse_tolerance,
    )
    peak_start = descriptor["peak_start"]
    peak_end = descriptor["peak_end"]
    late_label = descriptor["phase3_label"]
    valid_start = descriptor["valid_start"]

    phased_rows = []
    for row in rows:
        progress = float(row["progress"])
        clean_acc = float(row["test_clean_acc"])
        new_row = dict(row)
        new_row["valid_for_analysis"] = clean_acc >= clean_analysis_threshold
        if not new_row["valid_for_analysis"] or peak_start == "":
            phase = PRE_LEARNING
        elif progress < float(peak_start):
            phase = FORMATION
        elif progress <= float(peak_end):
            phase = STABILIZATION
        elif progress > float(peak_end):
            phase = late_label
        elif progress < float(valid_start):
            phase = PRE_LEARNING
        else:
            phase = FORMATION
        new_row["phase"] = phase
        phased_rows.append(new_row)
    return phased_rows


def detect_collapse_segments(rows: Iterable[Dict]) -> List[Dict]:
    """
    Identify adjacent checkpoint intervals where clean accuracy increases,
    robust accuracy decreases, and the clean-robust gap expands.
    """
    rows = _sorted_rows(rows)
    segments = []

    for previous, current in zip(rows, rows[1:]):
        ca_change = float(current["test_clean_acc"]) - float(previous["test_clean_acc"])
        ra_change = float(current["test_robust_acc"]) - float(previous["test_robust_acc"])
        gap_change = float(current["gap"]) - float(previous["gap"])

        if ca_change >= 0.0 and ra_change < 0.0 and gap_change > 0.0:
            segments.append({
                "model": current.get("model", previous.get("model", "")),
                "from_progress": float(previous["progress"]),
                "to_progress": float(current["progress"]),
                "ca_change": ca_change,
                "ra_change": ra_change,
                "gap_change": gap_change,
            })

    return segments


def group_rows_by_model(rows: Iterable[Dict]) -> Dict[str, List[Dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row.get("model", "unknown")), []).append(row)
    return {model: _sorted_rows(model_rows) for model, model_rows in grouped.items()}
