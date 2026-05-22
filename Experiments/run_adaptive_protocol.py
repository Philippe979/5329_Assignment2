import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.adaptive_protocol import (
    build_protocol_json,
    load_protocol_config,
    read_csv,
    select_protocol_rows,
    write_csv,
    write_json,
)
from Models import MODEL_REGISTRY, models_by_group


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run adaptive CIFAR-10 dynamic robustness calibration and final experiments."
    )
    parser.add_argument("--stage", choices=["calibration", "select", "final", "all"], default="all")
    parser.add_argument("--group", default="all")
    parser.add_argument("--config", default=str(Path("Experiments") / "adaptive_protocol_config.json"))
    parser.add_argument("--results-dir", default="Results")
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--calibration-eval-subset", type=int, default=2000)
    parser.add_argument("--final-eval-subset", type=int, default=0)
    parser.add_argument("--sweep-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="./Data")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def run_logged(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(str(part) for part in cmd), flush=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(" ".join(str(part) for part in cmd) + "\n\n")
        log_file.flush()
        subprocess.run(
            [str(part) for part in cmd],
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_calibration(args, config, model_names, root_dir):
    calibration_dir = root_dir / "calibration"
    logs_dir = root_dir / "logs" / "calibration"
    calibration_epsilon = float(config["calibration_epsilon"])
    checkpoint_grid_size = int(config["calibration_checkpoint_grid_size"])

    for index, model_name in enumerate(model_names, start=1):
        model_config = MODEL_REGISTRY[model_name]
        output_dir = calibration_dir / f"{model_name}_dynamic"
        descriptor_path = output_dir / "metrics" / f"{model_name}_descriptors.csv"
        if args.resume and descriptor_path.exists():
            print(f"[calibration {index}/{len(model_names)}] skip {model_name}", flush=True)
            continue

        cmd = [
            sys.executable,
            "-u",
            "Experiments/run_dynamic_pipeline.py",
            "--model-name",
            model_name,
            "--epochs",
            str(model_config.default_epochs),
            "--batch-size",
            str(model_config.default_batch_size),
            "--train-subset",
            str(args.train_subset),
            "--eval-subset",
            str(args.calibration_eval_subset),
            "--num-workers",
            str(args.num_workers),
            "--epsilon",
            str(calibration_epsilon),
            "--analysis-clean-threshold",
            str(config["analysis_clean_threshold"]),
            "--peak-ratio",
            str(config["peak_ratio"]),
            "--checkpoint-grid-size",
            str(checkpoint_grid_size),
            "--model-only-checkpoints",
            "--seed",
            str(args.seed),
            "--data-dir",
            args.data_dir,
            "--output-dir",
            str(output_dir),
        ]
        print(f"[calibration {index}/{len(model_names)}] running {model_name}", flush=True)
        run_logged(cmd, logs_dir / f"{index:02d}_{model_name}.log")

    sweep_cmd = [
        sys.executable,
        "-u",
        "Experiments/epsilon_sweep_existing_checkpoints.py",
        "--models",
        *model_names,
        "--epsilons",
        *[str(value) for value in config["epsilon_candidates"]],
        "--eval-subset",
        str(args.calibration_eval_subset),
        "--batch-size",
        str(args.sweep_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--analysis-clean-threshold",
        str(config["analysis_clean_threshold"]),
        "--peak-ratio",
        str(config["peak_ratio"]),
        "--seed",
        str(args.seed),
        "--data-dir",
        args.data_dir,
        "--results-dir",
        str(calibration_dir),
    ]
    print("[calibration] running epsilon sweep", flush=True)
    run_logged(sweep_cmd, root_dir / "logs" / "epsilon_sweep.log")


def run_selection(config, root_dir):
    summary_path = root_dir / "calibration" / "epsilon_sweep" / "epsilon_sweep_descriptors.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Calibration epsilon sweep summary not found: {summary_path}")

    descriptor_rows = read_csv(summary_path)
    selected_rows = select_protocol_rows(descriptor_rows, config)
    selected_csv = root_dir / "protocol" / "selected_model_protocol.csv"
    selected_json = root_dir / "protocol" / "selected_model_protocol.json"

    write_csv(selected_rows, selected_csv)
    write_json(build_protocol_json(selected_rows, config), selected_json)
    print(f"[select] saved {selected_csv}", flush=True)
    print(f"[select] saved {selected_json}", flush=True)


def run_final(args, config, model_names, root_dir):
    selected_path = root_dir / "protocol" / "selected_model_protocol.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Selected protocol not found: {selected_path}")

    selected_by_model = {row["model"]: row for row in read_csv(selected_path)}
    final_dir = root_dir / "final"
    logs_dir = root_dir / "logs" / "final"

    for index, model_name in enumerate(model_names, start=1):
        if model_name not in selected_by_model:
            raise KeyError(f"No selected protocol row for model: {model_name}")
        model_config = MODEL_REGISTRY[model_name]
        selected = selected_by_model[model_name]
        epsilon = float(selected["epsilon"])
        output_dir = final_dir / f"{model_name}_dynamic"
        descriptor_path = output_dir / "metrics" / f"{model_name}_descriptors.csv"
        if args.resume and descriptor_path.exists():
            print(f"[final {index}/{len(model_names)}] skip {model_name}", flush=True)
            continue

        cmd = [
            sys.executable,
            "-u",
            "Experiments/run_dynamic_pipeline.py",
            "--model-name",
            model_name,
            "--epochs",
            str(model_config.default_epochs),
            "--batch-size",
            str(model_config.default_batch_size),
            "--train-subset",
            str(args.train_subset),
            "--eval-subset",
            str(args.final_eval_subset),
            "--num-workers",
            str(args.num_workers),
            "--epsilon",
            str(epsilon),
            "--analysis-clean-threshold",
            str(config["analysis_clean_threshold"]),
            "--peak-ratio",
            str(config["peak_ratio"]),
            "--checkpoint-grid-size",
            str(config["final_checkpoint_grid_size"]),
            "--model-only-checkpoints",
            "--seed",
            str(args.seed),
            "--data-dir",
            args.data_dir,
            "--output-dir",
            str(output_dir),
        ]
        print(
            f"[final {index}/{len(model_names)}] running {model_name} "
            f"epsilon={epsilon:.6f} type={selected['dynamic_type']}",
            flush=True,
        )
        run_logged(cmd, logs_dir / f"{index:02d}_{model_name}.log")

    if not args.skip_summary:
        summary_cmd = [
            sys.executable,
            "-u",
            "Experiments/generate_summary_figures.py",
            "--results-dir",
            str(final_dir),
            "--output-dir",
            str(root_dir / "summary_figures"),
        ]
        print("[final] generating adaptive summary figures", flush=True)
        run_logged(summary_cmd, root_dir / "logs" / "summary.log")

        path_cmd = [
            sys.executable,
            "-u",
            "Experiments/path_dependent_phase_analysis.py",
        ]
        print("[final] generating path-dependent phase analysis", flush=True)
        run_logged(path_cmd, root_dir / "logs" / "path_dependent_phase.log")


def main():
    args = parse_args()
    config = load_protocol_config(args.config)
    root_dir = Path(args.results_dir) / "adaptive_protocol"
    model_names = models_by_group(args.group)
    root_dir.mkdir(parents=True, exist_ok=True)
    write_json({"models": model_names, "args": vars(args), "protocol_config": config}, root_dir / "run_config.json")

    print(f"Adaptive protocol root: {root_dir}", flush=True)
    print(f"Models ({len(model_names)}): {model_names}", flush=True)

    if args.stage in {"calibration", "all"}:
        run_calibration(args, config, model_names, root_dir)
    if args.stage in {"select", "all"}:
        run_selection(config, root_dir)
    if args.stage in {"final", "all"}:
        run_final(args, config, model_names, root_dir)


if __name__ == "__main__":
    main()
