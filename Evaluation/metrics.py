import csv
import os
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


@torch.no_grad()
def evaluate_clean(
    model: torch.nn.Module,
    dataloader,
    device: str = "cuda",
) -> float:
    """
    Compute clean accuracy.
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


def evaluate_robust(
    model: torch.nn.Module,
    dataloader,
    attack_fn,
    attack_params: Optional[Dict] = None,
    device: str = "cuda",
) -> float:
    """
    Compute robust accuracy under a given attack.
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    attack_params = attack_params or {}

    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        adv_images = attack_fn(
            model=model,
            images=images,
            labels=labels,
            device=device,
            **attack_params,
        )

        outputs = model(adv_images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


def evaluate_both(
    model: torch.nn.Module,
    dataloader,
    attack_fn,
    attack_params: Optional[Dict] = None,
    device: str = "cuda",
):
    """
    Return clean accuracy, robust accuracy, and robustness gap.
    """
    clean_acc = evaluate_clean(model, dataloader, device=device)
    robust_acc = evaluate_robust(
        model,
        dataloader,
        attack_fn=attack_fn,
        attack_params=attack_params,
        device=device,
    )
    gap = clean_acc - robust_acc

    return {
        "clean_acc": clean_acc,
        "robust_acc": robust_acc,
        "robust_gap": gap,
    }


def evaluate_checkpoint(
    model: torch.nn.Module,
    dataloader,
    attack_fn,
    model_name: str,
    level: str,
    progress: float,
    epoch: int,
    global_step: int,
    train_loss: Optional[float] = None,
    learning_rate: Optional[float] = None,
    checkpoint_path: str = "",
    attack_params: Optional[Dict] = None,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate one training checkpoint and return a CSV-ready metric row.
    """
    attack_params = attack_params or {}
    metrics = evaluate_both(
        model=model,
        dataloader=dataloader,
        attack_fn=attack_fn,
        attack_params=attack_params,
        device=device,
    )

    return {
        "model": model_name,
        "level": level,
        "progress": progress,
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "test_clean_acc": metrics["clean_acc"],
        "test_robust_acc": metrics["robust_acc"],
        "gap": metrics["robust_gap"],
        "epsilon": attack_params.get("epsilon"),
        "learning_rate": learning_rate,
        "checkpoint_path": checkpoint_path,
    }


def save_metrics_csv(rows: Iterable[Dict], output_path: str) -> None:
    """
    Save per-checkpoint metric rows to a CSV file.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("rows must contain at least one metric row.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
