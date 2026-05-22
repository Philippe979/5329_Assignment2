import csv
import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Evaluation import (
    group_rows_by_model,
    plot_multi_model_robustness,
    plot_phase_timeline,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate summary figures from phased checkpoint metrics.")
    parser.add_argument("--results-dir", default="Results")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    metric_files = sorted(results_dir.glob("*/metrics/*_checkpoint_metrics_phased.csv"))
    if not metric_files:
        raise FileNotFoundError(
            "No phased checkpoint metrics found. Run experiments before generating summary figures."
        )

    rows = []
    for path in metric_files:
        rows.extend(read_rows(path))

    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "summary_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_rows = group_rows_by_model(rows)

    model_count = len(grouped_rows)

    plot_phase_timeline(
        grouped_rows,
        title=f"Robustness Phase Timeline Across {model_count} CIFAR-10 Models",
        output_path=str(output_dir / "all_models_phase_timeline.png"),
    )
    plot_multi_model_robustness(
        grouped_rows,
        title=f"FGSM Robustness Trajectories Across {model_count} CIFAR-10 Models",
        output_path=str(output_dir / "all_models_robustness_trajectories.png"),
    )

    print(f"Saved summary figures to {output_dir}")


if __name__ == "__main__":
    main()
