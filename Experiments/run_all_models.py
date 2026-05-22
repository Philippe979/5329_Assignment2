import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models import MODEL_REGISTRY, models_by_group


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full CIFAR-10 dynamic robustness benchmark.")
    parser.add_argument("--group", default="all")
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--eval-subset", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--analysis-clean-threshold", type=float, default=0.30)
    parser.add_argument("--peak-ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="Results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    model_names = models_by_group(args.group)
    print(f"Running {len(model_names)} models: {model_names}", flush=True)

    for index, model_name in enumerate(model_names, start=1):
        config = MODEL_REGISTRY[model_name]
        output_dir = results_dir / f"{model_name}_dynamic"
        descriptor_path = output_dir / "metrics" / f"{model_name}_descriptors.csv"
        if args.resume and descriptor_path.exists():
            print(f"\n[{index}/{len(model_names)}] Skipping {model_name}; descriptor exists.", flush=True)
            continue
        log_path = logs_dir / f"{index:02d}_{model_name}.log"
        cmd = [
            sys.executable,
            "-u",
            "Experiments/run_dynamic_pipeline.py",
            "--model-name",
            model_name,
            "--epochs",
            str(config.default_epochs),
            "--batch-size",
            str(config.default_batch_size),
            "--train-subset",
            str(args.train_subset),
            "--eval-subset",
            str(args.eval_subset),
            "--num-workers",
            str(args.num_workers),
            "--epsilon",
            str(args.epsilon),
            "--analysis-clean-threshold",
            str(args.analysis_clean_threshold),
            "--peak-ratio",
            str(args.peak_ratio),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(output_dir),
        ]
        print(f"\n[{index}/{len(model_names)}] Running {model_name}", flush=True)
        print(" ".join(cmd), flush=True)
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(" ".join(cmd) + "\n\n")
            log_file.flush()
            subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"[{index}/{len(model_names)}] Finished {model_name}; log: {log_path}", flush=True)

    if not args.skip_summary:
        summary_cmd = [sys.executable, "-u", "Experiments/generate_summary_figures.py"]
        print("\nGenerating summary figures", flush=True)
        with open(logs_dir / "summary.log", "w", encoding="utf-8") as log_file:
            subprocess.run(
                summary_cmd,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print("Summary figures generated.", flush=True)


if __name__ == "__main__":
    main()
