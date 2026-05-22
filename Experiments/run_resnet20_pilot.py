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
from Mid_Model import ResNet20
from Training import save_checkpoint, train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Run a CUDA ResNet20 dynamic robustness pilot.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--eval-subset", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--analysis-clean-threshold", type=float, default=0.30)
    parser.add_argument("--peak-ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", default="./Data")
    parser.add_argument("--output-dir", default="./Results/resnet20_pilot")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        raise RuntimeError("CUDA is required for this pilot run, but torch.cuda.is_available() is False.")

    seed_everything(args.seed)
    device = "cuda"
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    for directory in [checkpoint_dir, metrics_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Output: {output_dir}")

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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ResNet20().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    total_steps = args.epochs * len(train_loader)
    progress_points = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
    target_steps = {
        max(1, min(total_steps, round(point * total_steps))): point
        for point in progress_points
    }

    metric_rows = []
    global_step = 0
    latest_loss = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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
                checkpoint_path = checkpoint_dir / f"resnet20_p{progress:.2f}.pt"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    progress=progress,
                    output_path=str(checkpoint_path),
                )

                row = evaluate_checkpoint(
                    model=model,
                    dataloader=eval_loader,
                    attack_fn=fgsm_attack,
                    model_name="resnet20",
                    level="Level 2 - Residual CNN",
                    progress=progress,
                    epoch=epoch,
                    global_step=global_step,
                    train_loss=latest_loss,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    checkpoint_path=str(checkpoint_path),
                    attack_params={"epsilon": args.epsilon},
                    device=device,
                )
                metric_rows.append(row)
                print(
                    f"p={progress:.2f} epoch={epoch} step={global_step} "
                    f"loss={latest_loss:.4f} CA={row['test_clean_acc']:.4f} "
                    f"RA={row['test_robust_acc']:.4f} gap={row['gap']:.4f}"
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
    collapse_segments = detect_collapse_segments(phased_rows)

    metrics_path = metrics_dir / "resnet20_checkpoint_metrics.csv"
    phased_metrics_path = metrics_dir / "resnet20_checkpoint_metrics_phased.csv"
    descriptor_path = metrics_dir / "resnet20_descriptors.csv"
    collapse_path = metrics_dir / "resnet20_collapse_segments.json"

    save_metrics_csv(metric_rows, str(metrics_path))
    save_metrics_csv(phased_rows, str(phased_metrics_path))
    write_descriptor_csv(descriptor, str(descriptor_path))
    with open(collapse_path, "w", encoding="utf-8") as file:
        json.dump(collapse_segments, file, indent=2)

    plot_trajectory(
        phased_rows,
        title="Dynamic Robustness Trajectory - ResNet20 Pilot",
        output_path=str(figures_dir / "resnet20_trajectory.png"),
    )
    plot_gravity_collapse(
        phased_rows,
        title="Gravity Collapse Plot - ResNet20 Pilot",
        output_path=str(figures_dir / "resnet20_gravity_collapse.png"),
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
