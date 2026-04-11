from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_checkpoint(
    save_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
):
    """
    保存完整训练检查点
    """
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "config": config,
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if extra is not None:
        state.update(extra)

    torch.save(state, save_path)

def load_checkpoint(
    checkpoint_path: str, model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Step: {checkpoint['global_step']}, Loss: {checkpoint['best_val_loss']:.4f}")

    return checkpoint
