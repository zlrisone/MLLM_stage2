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

def _remap_state_dict_keys(state_dict: Dict[str, torch.Tensor],
                           model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    自动适配新旧 vision_encoder 结构之间的 key 差异。

    - 旧版: self.vision_model = SiglipVisionModel(...)
      state_dict key = vision_encoder.vision_model.vision_model.<...>
    - 新版: self.vision_model = SiglipVisionModel(...).vision_model
      state_dict key = vision_encoder.vision_model.<...>

    若发现 checkpoint 与当前模型在 `vision_encoder.vision_model` 这一段
    的嵌套层数不一致, 则按需增删一层 `vision_model.` 前缀。
    """
    model_keys = set(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if not ckpt_keys:
        return state_dict

    sample_model_key = next((k for k in model_keys
                             if k.startswith("vision_encoder.vision_model.")), None)
    sample_ckpt_key = next((k for k in ckpt_keys
                            if k.startswith("vision_encoder.vision_model.")), None)
    if sample_model_key is None or sample_ckpt_key is None:
        return state_dict

    ckpt_has_extra = sample_ckpt_key.startswith("vision_encoder.vision_model.vision_model.")
    model_has_extra = sample_model_key.startswith("vision_encoder.vision_model.vision_model.")

    if ckpt_has_extra == model_has_extra:
        return state_dict

    new_state_dict: Dict[str, torch.Tensor] = {}
    if ckpt_has_extra and not model_has_extra:
        old_prefix = "vision_encoder.vision_model.vision_model."
        new_prefix = "vision_encoder.vision_model."
    else:
        old_prefix = "vision_encoder.vision_model."
        new_prefix = "vision_encoder.vision_model.vision_model."

    remapped = 0
    for k, v in state_dict.items():
        if k.startswith(old_prefix):
            new_state_dict[new_prefix + k[len(old_prefix):]] = v
            remapped += 1
        else:
            new_state_dict[k] = v

    if remapped:
        print(f"[load_checkpoint] 已自动重映射 {remapped} 个 vision_encoder 的 state_dict key "
              f"({old_prefix!r} -> {new_prefix!r})")
    return new_state_dict


def load_checkpoint(
    checkpoint_path: str, model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True) -> Dict[str, Any]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        parent = ckpt_path.parent if ckpt_path.parent.exists() else Path('.')
        candidates = sorted([str(p) for p in parent.glob('*.pt')])
        hint = f"\n可选的 .pt 文件: {candidates}" if candidates else ""
        raise FileNotFoundError(
            f"Checkpoint 文件不存在: {checkpoint_path}{hint}"
        )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint['model_state_dict']
    state_dict = _remap_state_dict_keys(state_dict, model)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_checkpoint] Missing keys ({len(missing)}): 前 5 个示例: {missing[:5]}")
    if unexpected:
        print(f"[load_checkpoint] Unexpected keys ({len(unexpected)}): 前 5 个示例: {unexpected[:5]}")
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"加载 state_dict 失败: missing={len(missing)}, unexpected={len(unexpected)}。"
            f"如果你确认只有视觉编码器/LM 等冻结/预训练部分不一致, 可把 strict=False 传入。"
        )

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Step: {checkpoint['global_step']}, "
          f"Loss: {checkpoint['best_val_loss']:.4f}")

    return checkpoint
