import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Attack_Method import fgsm_attack
from Data_Loader.cifar_loader import get_cifar10_datasets
from Evaluation import assign_phases, compute_dynamic_descriptors, evaluate_checkpoint, plot_trajectory, save_metrics_csv
from Models import build_model, get_model_config


def parse_args():
    parser = argparse.ArgumentParser(description="Re-evaluate saved checkpoints under multiple FGSM epsilons.")
    parser.add_argument("--models", nargs="+", default=["vgg19_bn_cifar", "resnet20", "wrn_16_8", "mlp_mixer_base"])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[2 / 255, 4 / 255, 8 / 255])
    parser.add_argument("--eval-subset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--analysis-clean-threshold", type=float, default=0.30)
    parser.add_argument("--peak-ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="./Data")
    parser.add_argument("--results-dir", default="Results")
    return parser.parse_args()


def subset_dataset(dataset, subset_size: int, seed: int):
    if subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def read_checkpoint_rows(model_name, results_dir):
    path = Path(results_dir) / f"{model_name}_dynamic" / "metrics" / f"{model_name}_checkpoint_metrics.csv"
    with open(path, newline="", encoding="utf-8") as file:
        return sorted(csv.DictReader(file), key=lambda row: float(row["progress"]))


def write_descriptor_csv(descriptor, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(descriptor.keys()))
        writer.writeheader()
        writer.writerow(descriptor)


def epsilon_label(epsilon):
    return f"eps_{int(round(epsilon * 255)):02d}_255"


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for epsilon sweep.")

    _, _, test_dataset = get_cifar10_datasets(
        data_dir=args.data_dir,
        val_ratio=0.1,
        use_augmentation=False,
        seed=args.seed,
    )
    eval_dataset = subset_dataset(test_dataset, args.eval_subset, args.seed + 1)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    output_root = Path(args.results_dir) / "epsilon_sweep"
    output_root.mkdir(parents=True, exist_ok=True)
    all_descriptors = []

    for model_name in args.models:
        config = get_model_config(model_name)
        base_rows = read_checkpoint_rows(model_name, args.results_dir)

        for epsilon in args.epsilons:
            label = epsilon_label(epsilon)
            output_dir = output_root / model_name / label
            metrics_dir = output_dir / "metrics"
            figures_dir = output_dir / "figures"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            figures_dir.mkdir(parents=True, exist_ok=True)

            print(f"Evaluating {model_name} at epsilon={epsilon:.6f}", flush=True)
            metric_rows = []
            model = build_model(model_name).to("cuda")

            for source_row in base_rows:
                checkpoint_path = source_row["checkpoint_path"]
                payload = torch.load(checkpoint_path, map_location="cuda")
                model.load_state_dict(payload["model_state_dict"])
                row = evaluate_checkpoint(
                    model=model,
                    dataloader=eval_loader,
                    attack_fn=fgsm_attack,
                    model_name=model_name,
                    level=config.capacity_level,
                    progress=float(source_row["progress"]),
                    epoch=int(source_row["epoch"]),
                    global_step=int(source_row["global_step"]),
                    train_loss=float(source_row["train_loss"]),
                    learning_rate=float(source_row["learning_rate"]),
                    checkpoint_path=checkpoint_path,
                    attack_params={"epsilon": epsilon},
                    device="cuda",
                )
                metric_rows.append(row)

            phased_rows = assign_phases(
                metric_rows,
                peak_ratio=args.peak_ratio,
                clean_analysis_threshold=args.analysis_clean_threshold,
            )
            descriptor = compute_dynamic_descriptors(
                phased_rows,
                peak_ratio=args.peak_ratio,
                clean_analysis_threshold=args.analysis_clean_threshold,
            )
            descriptor["model"] = model_name
            descriptor["epsilon"] = epsilon
            descriptor["epsilon_label"] = label
            descriptor["design_type"] = config.design_type
            descriptor["family"] = config.family
            descriptor["clean_threshold"] = config.clean_threshold
            all_descriptors.append(descriptor)

            save_metrics_csv(metric_rows, str(metrics_dir / f"{model_name}_{label}_checkpoint_metrics.csv"))
            save_metrics_csv(phased_rows, str(metrics_dir / f"{model_name}_{label}_checkpoint_metrics_phased.csv"))
            write_descriptor_csv(descriptor, metrics_dir / f"{model_name}_{label}_descriptors.csv")
            plot_trajectory(
                phased_rows,
                title=f"{model_name} FGSM {label}",
                output_path=str(figures_dir / f"{model_name}_{label}_trajectory.png"),
            )

    if all_descriptors:
        summary_path = output_root / "epsilon_sweep_descriptors.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(all_descriptors[0].keys()))
            writer.writeheader()
            writer.writerows(all_descriptors)
        with open(output_root / "epsilon_sweep_config.json", "w", encoding="utf-8") as file:
            json.dump(vars(args), file, indent=2)
        print(f"Saved epsilon sweep summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
