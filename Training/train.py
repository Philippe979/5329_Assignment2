from typing import Optional, Tuple

import torch
import torch.nn as nn


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    start_global_step: int = 0,
    max_batches: Optional[int] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, int]:
    """
    Train for one epoch and return average loss plus updated global step.
    """
    model.train()
    running_loss = 0.0
    total_batches = 0
    global_step = start_global_step
    use_amp = scaler is not None and str(device).startswith("cuda")

    for batch_index, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        total_batches += 1
        global_step += 1

    avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss, global_step
