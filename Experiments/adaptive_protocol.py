import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_CONFIG_PATH = Path(__file__).with_name("adaptive_protocol_config.json")


def load_protocol_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def epsilon_label(epsilon: float) -> str:
    return f"eps_{int(round(float(epsilon) * 255)):02d}_255"


def read_csv(path: str | Path) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(rows: Iterable[Dict], path: str | Path) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _float(row: Dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return float(value)


def score_descriptor(row: Dict, config: Dict) -> Dict:
    weights = config["scoring_weights"]
    tau_clean = _float(row, "clean_threshold", default=float(config["tau_clean"]))
    tau_ra = float(config["tau_ra"])
    tau_formation_width = float(config["tau_formation_width"])
    tau_late_width = float(config["tau_late_width"])
    min_peak_progress = float(config["min_peak_progress"])
    zero_ra = float(config["zero_robustness_threshold"])

    ca_final = _float(row, "CA_final")
    ra_peak = _float(row, "RA_peak")
    ra_final = _float(row, "RA_final")
    p_peak = _float(row, "p_peak")
    valid_start = _float(row, "valid_start", default=1.0)
    stability_width = _float(row, "stability_width")
    epsilon = _float(row, "epsilon")
    max_epsilon = max(float(value) for value in config["epsilon_candidates"])
    min_epsilon = min(float(value) for value in config["epsilon_candidates"])

    formation_width = max(0.0, p_peak - valid_start)
    late_width = max(0.0, 1.0 - p_peak)
    normalized_ra_peak = min(1.0, ra_peak / max(tau_ra, 1e-8))

    clean_valid = ca_final >= tau_clean
    robust_observable = ra_peak >= tau_ra
    formation_observable = formation_width >= tau_formation_width
    late_observable = late_width >= tau_late_width
    not_zero_final = ra_final > zero_ra
    not_too_early_peak = p_peak >= min_peak_progress

    score = 0.0
    score += weights["clean_valid"] if clean_valid else 0.0
    score += weights["robust_observable"] * normalized_ra_peak
    score += weights["formation_width"] * min(1.0, formation_width / tau_formation_width)
    score += weights["late_width"] * min(1.0, late_width / tau_late_width)
    score += weights["stability_width"] * min(1.0, stability_width / tau_formation_width)
    score += weights["not_zero_final"] if not_zero_final else 0.0
    score += weights["not_too_early_peak"] if not_too_early_peak else 0.0

    if min_epsilon < epsilon < max_epsilon:
        score += weights["mid_strength_bonus"]
    if clean_valid and robust_observable and _float(row, "RA_drop") < 0.01 and epsilon == min_epsilon:
        score -= weights["too_weak_penalty"]
    if clean_valid and ra_final <= zero_ra:
        score -= weights["too_strong_penalty"]

    return {
        "observable_score": round(score, 6),
        "clean_valid": clean_valid,
        "robust_observable": robust_observable,
        "formation_width": round(formation_width, 6),
        "late_width": round(late_width, 6),
        "not_zero_final": not_zero_final,
        "not_too_early_peak": not_too_early_peak,
    }


def classify_dynamic_type(row: Dict, score_info: Dict, config: Dict) -> str:
    tau_ra = float(config["tau_ra"])
    tau_formation_width = float(config["tau_formation_width"])

    if not score_info["clean_valid"]:
        return "Type E - not clean-valid"
    if _float(row, "RA_peak") < tau_ra:
        return "Type D - weak robust learning"
    if score_info["formation_width"] >= tau_formation_width and _float(row, "stability_width") >= tau_formation_width:
        return "Type A - near-complete trajectory"
    if score_info["formation_width"] >= tau_formation_width:
        return "Type B - formation visible, short stabilization"
    if _float(row, "RA_drop") >= 0.02:
        return "Type C - peak then degradation"
    return "Type D - partial dynamic trajectory"


def select_protocol_rows(descriptor_rows: Iterable[Dict], config: Dict) -> List[Dict]:
    candidates_by_model: Dict[str, List[Dict]] = {}
    for row in descriptor_rows:
        scored = dict(row)
        score_info = score_descriptor(scored, config)
        scored.update(score_info)
        scored["dynamic_type"] = classify_dynamic_type(scored, score_info, config)
        candidates_by_model.setdefault(scored["model"], []).append(scored)

    selected_rows = []
    for model_name, candidates in sorted(candidates_by_model.items()):
        candidates.sort(
            key=lambda item: (
                float(item["observable_score"]),
                float(item.get("RA_peak", 0.0)),
                -float(item.get("epsilon", 0.0)),
            ),
            reverse=True,
        )
        best = dict(candidates[0])
        best["selected"] = True
        best["selection_rank"] = 1
        selected_rows.append(best)

    return selected_rows


def build_protocol_json(selected_rows: Iterable[Dict], config: Dict) -> Dict:
    models = {}
    for row in selected_rows:
        models[row["model"]] = {
            "epsilon": float(row["epsilon"]),
            "epsilon_label": row["epsilon_label"],
            "observable_score": float(row["observable_score"]),
            "dynamic_type": row["dynamic_type"],
            "checkpoint_grid_size": int(config["final_checkpoint_grid_size"]),
        }
    return {
        "protocol": "adaptive_dynamic_robustness",
        "epsilon_candidates": config["epsilon_candidates"],
        "analysis_clean_threshold": config["analysis_clean_threshold"],
        "peak_ratio": config["peak_ratio"],
        "selection_rule": "maximize ObservableScore over epsilon candidates per model",
        "models": models,
    }
