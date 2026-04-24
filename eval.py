import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.LM_metrics import evaluate_caption
from models.multimodal_model import create_multimodal_model
from data.caption_dataset import build_dataloders
from utils.checkpoint import ensure_dir, save_checkpoint, load_checkpoint

import argparse
import yaml
import os


_DTYPE_MAP = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    dtype = _DTYPE_MAP.get(str(name).lower(), torch.float32)
    # 半精度只在 GPU 上有意义
    if device.type != "cuda" and dtype != torch.float32:
        return torch.float32
    return dtype


@torch.inference_mode()
def evaluate_generation(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 64,
    num_beams: int = 1,
    do_sample: bool = False,
    save_json_path: Optional[str] = None,
    log_to_wandb: bool = False,
    prefix: str = "test",
    amp_dtype: Optional[torch.dtype] = None,
    limit_batches: int = 0,
) -> Dict[str, Any]:
    """
    基于生成结果做 caption 评估。

    Args:
        model: 多模态模型
        dataloader: 验证/测试 dataloader
        tokenizer: 文本 tokenizer，用于 decode
        device: 设备
        max_new_tokens: 生成最大长度
        num_beams: beam search 宽度
        do_sample: 是否采样生成
        save_json_path: 可选，保存预测明细
        log_to_wandb: 是否记录到 wandb
        prefix: wandb / 输出前缀，比如 val/test

    Returns:
        {
            "BLEU-1": ...,
            "BLEU-2": ...,
            "BLEU-3": ...,
            "BLEU-4": ...,
            "ROUGE-L": ...,
            "CIDEr": ...,
            "METEOR": ...,
            "per_sample": ...,
            "predictions": ...,
            "references": ...
        }
    """
    model.eval()

    all_predictions: List[str] = []
    all_references: List[Union[str, List[str]]] = []
    prediction_records: List[Dict[str, Any]] = []

    pbar = tqdm(
        dataloader,
        desc=f"Evaluate {prefix}",
        leave=False,
        dynamic_ncols=True,
    )

    # 仅在 GPU + 半精度时启用 autocast
    use_amp = (device.type == "cuda") and (amp_dtype is not None) and (amp_dtype != torch.float32)
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    for batch_idx, batch in enumerate(pbar):
        if limit_batches and batch_idx >= limit_batches:
            break

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with amp_ctx:
            generated_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                use_cache=True,
            )

        # 用 inputs_embeds 调 generate 时，HF 返回的就是新生成部分，直接 decode 即可
        pred_texts = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        pred_texts = [x.strip() for x in pred_texts]
        if "references" in batch:
            refs = batch["references"]
        else:
            raise KeyError(
                "Batch 中没有找到参考文本字段。请确保 batch 包含 "
                "'references' / 'reference' / 'captions' / 'raw_texts' 之一。"
            )
        refs = list(refs)

        if len(pred_texts) != len(refs):
            raise ValueError(
                f"Prediction size mismatch: len(pred_texts)={len(pred_texts)}, len(refs)={len(refs)}"
            )

        all_predictions.extend(pred_texts)
        all_references.extend(refs)

        for i, (pred, ref) in enumerate(zip(pred_texts, refs)):
            record = {
                "sample_id": len(prediction_records),
                "prediction": pred,
                "reference": ref,
            }
            prediction_records.append(record)
    metrics = evaluate_caption(
        predictions=all_predictions,
        references=all_references,
    )

    metrics["predictions"] = all_predictions
    metrics["references"] = all_references

    # 保存明细
    if save_json_path is not None:
        save_path = Path(save_json_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_obj = {
            "metrics": {k: v for k, v in metrics.items() if k != "per_sample" and k not in ["predictions", "references"]},
            "predictions": prediction_records,
            "per_sample": metrics.get("per_sample", {}),
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_obj, f, ensure_ascii=False, indent=2)

        print(f"[Evaluation Results Saved] {save_json_path}")

    # wandb 记录
    if log_to_wandb:
        import wandb

        log_dict = {}
        for key, value in metrics.items():
            if key in ["per_sample", "predictions", "references"]:
                continue
            log_dict[f"{prefix}/{key}"] = value

        wandb.log(log_dict)

    return metrics
    
def print_caption_metrics(metrics: Dict[str, Any], prefix: str = "Eval"):
    keys = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]

    msg = [f"[{prefix}]"]
    for k in keys:
        if k in metrics:
            msg.append(f"{k}={metrics[k]:.4f}")
    print(" | ".join(msg))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Evaluation")
    parser.add_argument("--config", type=str, default="./config/training_stage2.yaml",
                       help="Path to config file")
    parser.add_argument("--resume", type=str, default="",
                       help="Resume from checkpoint")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖 eval batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="覆盖 dataloader num_workers")
    parser.add_argument("--dtype", type=str, default=None, choices=list(_DTYPE_MAP.keys()) + [None],
                       help="推理精度: fp32 / fp16 / bf16")
    parser.add_argument("--limit_batches", type=int, default=None,
                       help="只跑前 N 个 batch，用于快速验证")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get("eval", {}) or {}
    batch_size = args.batch_size if args.batch_size is not None else eval_cfg.get("batch_size", config["dataset"]["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else eval_cfg.get("num_workers", config["dataset"]["num_workers"])
    dtype_name = args.dtype if args.dtype is not None else eval_cfg.get("dtype", "fp32")
    limit_batches = args.limit_batches if args.limit_batches is not None else int(eval_cfg.get("limit_batches", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    amp_dtype = _resolve_dtype(dtype_name, device)
    print(f"Eval config | batch_size={batch_size} num_workers={num_workers} dtype={amp_dtype} limit_batches={limit_batches}")

    model = create_multimodal_model(config)
    model.to(device)
    model.eval()

    train_loader, val_loader, test_loader = build_dataloders(
        vision_model_name=config["dataset"]["vision_model_name"],
        qwen_model_name=config["dataset"]["qwen_model_name"],
        chat_json_path=config["dataset"]["chat_json_path"],
        image_root=config["dataset"]["image_root"],
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=config["dataset"]["max_length"],
        train_ratio=config["dataset"]["train_ratio"],
        val_ratio=config["dataset"]["val_ratio"],
        test_ratio=config["dataset"]["test_ratio"],
    )
    checkpoint = load_checkpoint(args.resume, model)

    tokenizer = AutoTokenizer.from_pretrained(config["dataset"]["qwen_model_name"], use_fast=True)

    test_metrics = evaluate_generation(
        model=model,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=config.get("generation", {}).get("max_new_tokens", 64),
        num_beams=config.get("generation", {}).get("num_beams", 1),
        do_sample=config.get("generation", {}).get("do_sample", False),
        save_json_path=os.path.join(config.get("save_dir", "./outputs"), "test_predictions.json"),
        log_to_wandb=config.get("use_wandb", True),
        prefix="test",
        amp_dtype=amp_dtype,
        limit_batches=limit_batches,
    )
    print_caption_metrics(test_metrics, prefix="Test")