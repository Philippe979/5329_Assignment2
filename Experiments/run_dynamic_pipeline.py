import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Attack_Method import fgsm_attack
from Data_Loader.cifar_loader import get_cifar10_datasets
from Evaluation import (
    assign_phases,
    compute_dynamic_descriptors,
    detect_collapse_segments,
    evaluate_checkpoint,
    plot_gravity_collapse,
    plot_trajectory,
    save_metrics_csv,
)
from Models import build_model, get_model_config, list_models
from Training import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Run a CIFAR-10 dynamic robustness pipeline.")
    parser.add_argument("--model-name", choices=list_models(), required=True)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--eval-subset", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--analysis-clean-threshold", type=float, default=0.30)
    parser.add_argument("--peak-ratio", type=float, default=0.95)
    parser.add_argument("--checkpoint-grid-size", type=int, default=0)
    parser.add_argument("--model-only-checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", default="./Data")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def subset_dataset(dataset, subset_size: int, seed: int):
    if subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def write_descriptor_csv(descriptor, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(descriptor.keys()))
        writer.writeheader()
        writer.writerow(descriptor)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")

    seed_everything(args.seed)
    config = get_model_config(args.model_name)
    batch_size = args.batch_size or config.default_batch_size
    epochs = args.epochs or config.default_epochs
    learning_rate = args.lr if args.lr > 0 else config.default_lr
    output_dir = Path(args.output_dir or f"./Results/{args.model_name}_dynamic_cuda")
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    for directory in [checkpoint_dir, metrics_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_name}")
    print(f"Level: {config.capacity_level}")
    print(f"Design type: {config.design_type}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}, batch_size: {batch_size}")

    run_config = {
        "model_name": args.model_name,
        "capacity_level": config.capacity_level,
        "design_type": config.design_type,
        "family": config.family,
        "reference": config.reference,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "epsilon": args.epsilon,
        "analysis_clean_threshold": args.analysis_clean_threshold,
        "peak_ratio": args.peak_ratio,
        "checkpoint_grid_size": args.checkpoint_grid_size,
        "model_only_checkpoints": args.model_only_checkpoints,
        "seed": args.seed,
        "train_subset": args.train_subset,
        "eval_subset": args.eval_subset,
        "clean_threshold": config.clean_threshold,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)

    train_dataset, _, test_dataset = get_cifar10_datasets(
        data_dir=args.data_dir,
        val_ratio=0.1,
        use_augmentation=True,
        seed=args.seed,
    )
    train_dataset = subset_dataset(train_dataset, args.train_subset, args.seed)
    eval_dataset = subset_dataset(test_dataset, args.eval_subset, args.seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.model_name).to("cuda")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    total_steps = epochs * len(train_loader)
    if args.checkpoint_grid_size > 0:
        progress_points = [
            round(index / args.checkpoint_grid_size, 4)
            for index in range(1, args.checkpoint_grid_size + 1)
        ]
    else:
        progress_points = [
            0.01, 0.02, 0.03, 0.05, 0.07,
            0.10, 0.15, 0.20, 0.25, 0.30,
            0.40, 0.50, 0.60, 0.70, 0.80,
            0.90, 1.00,
        ]
    target_steps = {
        max(1, min(total_steps, round(point * total_steps))): point
        for point in progress_points
    }

    metric_rows = []
    global_step = 0
    latest_loss = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images = images.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            latest_loss = float(loss.item())
            global_step += 1

            if global_step in target_steps:
                progress = target_steps[global_step]
                checkpoint_path = checkpoint_dir / f"{args.model_name}_p{progress:.2f}.pt"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    progress=progress,
                    output_path=str(checkpoint_path),
                    model_only=args.model_only_checkpoints,
                )

                row = evaluate_checkpoint(
                    model=model,
                    dataloader=eval_loader,
                    attack_fn=fgsm_attack,
                    model_name=args.model_name,
                    level=config.capacity_level,
                    progress=progress,
                    epoch=epoch,
                    global_step=global_step,
                    train_loss=latest_loss,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    checkpoint_path=str(checkpoint_path),
                    attack_params={"epsilon": args.epsilon},
                    device="cuda",
                )
                metric_rows.append(row)
                print(
                    f"p={progress:.2f} epoch={epoch} step={global_step} "
                    f"loss={latest_loss:.4f} CA={row['test_clean_acc']:.4f} "
                    f"RA={row['test_robust_acc']:.4f} gap={row['gap']:.4f}",
                    flush=True,
                )
                model.train()

        scheduler.step()

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
    descriptor["design_type"] = config.design_type
    descriptor["family"] = config.family
    descriptor["epsilon"] = args.epsilon
    descriptor["epsilon_label"] = f"eps_{int(round(args.epsilon * 255)):02d}_255"
    descriptor["clean_threshold"] = config.clean_threshold
    descriptor["validity_status"] = (
        "valid" if descriptor["CA_final"] >= config.clean_threshold else "below_clean_threshold"
    )
    collapse_segments = detect_collapse_segments(phased_rows)

    metrics_path = metrics_dir / f"{args.model_name}_checkpoint_metrics.csv"
    phased_metrics_path = metrics_dir / f"{args.model_name}_checkpoint_metrics_phased.csv"
    descriptor_path = metrics_dir / f"{args.model_name}_descriptors.csv"
    collapse_path = metrics_dir / f"{args.model_name}_collapse_segments.json"

    save_metrics_csv(metric_rows, str(metrics_path))
    save_metrics_csv(phased_rows, str(phased_metrics_path))
    write_descriptor_csv(descriptor, str(descriptor_path))
    with open(collapse_path, "w", encoding="utf-8") as file:
        json.dump(collapse_segments, file, indent=2)

    plot_trajectory(
        phased_rows,
        title=f"Dynamic Robustness Trajectory - {args.model_name}",
        output_path=str(figures_dir / f"{args.model_name}_trajectory.png"),
    )
    plot_gravity_collapse(
        phased_rows,
        title=f"Gravity Collapse Plot - {args.model_name}",
        output_path=str(figures_dir / f"{args.model_name}_gravity_collapse.png"),
    )

    print("\nDescriptor:")
    print(json.dumps(descriptor, indent=2))
    print("\nCollapse segments:")
    print(json.dumps(collapse_segments, indent=2))
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved descriptor: {descriptor_path}")
    print(f"Saved figures: {figures_dir}")


if __name__ == "__main__":
    main()
