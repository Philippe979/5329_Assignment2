import os
from typing import Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    progress: float,
    output_path: str,
    scheduler: Optional[object] = None,
    model_only: bool = False,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "progress": progress,
    }
    if not model_only:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and not model_only:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, output_path)
