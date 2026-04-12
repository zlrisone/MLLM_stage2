import torch
from tqdm import tqdm
from typing import Optional
import argparse
import yaml
from transformers import AutoTokenizer

from utils.checkpoint import load_checkpoint
from models.multimodal_model import create_multimodal_model
from data.caption_dataset import build_dataloders


@torch.no_grad()
def demo_caption_generation(
    model,
    dataloader,
    tokenizer,
    device: torch.device,
    max_samples: int = 10,
    max_new_tokens: int = 64,
    num_beams: int = 1,
    do_sample: bool = False,
):
    """
    人工查看 caption 生成结果的 demo

    Args:
        model: 你的 MultimodalModel
        dataloader: val_loader 或 test_loader
        tokenizer: LLM tokenizer
        device: cuda / cpu
        max_samples: 最多展示多少条样本
        max_new_tokens: generate 最大生成长度
        num_beams: beam search 宽度
        do_sample: 是否采样
    """
    model.eval()

    shown = 0

    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        generated_ids,prompt_len = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            return_prompt_length=True,
        )
        gen_only_ids = generated_ids[:, prompt_len:]
        pred_texts = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        pred_texts = [x.strip() for x in pred_texts]
        question = tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ) 
        # 兼容常见参考字段
        if "references" in batch:
            refs = batch["references"]
        else:
            refs = ["<No reference text in batch>"] * len(pred_texts)

        refs = list(refs)

        bs = len(pred_texts)
        for i in range(bs):
            print("=" * 80)
            print(f"Sample #{shown}")
            print("Question:")
            print(question[i])
            print("-" * 80)
            print("Reference:")
            print(refs[i])
            print("-" * 80)
            print("Prediction:")
            print(pred_texts[i])
            print("=" * 80)
            print()

            shown += 1
            if shown >= max_samples:
                return
def main():
    parser = argparse.ArgumentParser(description="Multimodal caption demo inference")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint .pt",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which dataloader split to use",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="How many samples to print",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Beam search width",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling for generation",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_multimodal_model(config)
    model.to(device)

    load_checkpoint(args.checkpoint, model)

    train_loader, val_loader, test_loader = build_dataloders(
        vision_model_name=config["dataset"]["vision_model_name"],
        qwen_model_name=config["dataset"]["qwen_model_name"],
        chat_json_path=config["dataset"]["chat_json_path"],
        image_root=config["dataset"]["image_root"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        max_length=config["dataset"]["max_length"],
        train_ratio=config["dataset"]["train_ratio"],
        val_ratio=config["dataset"]["val_ratio"],
        test_ratio=config["dataset"]["test_ratio"],
    )

    if args.split == "train":
        dataloader = train_loader
    elif args.split == "val":
        dataloader = val_loader
    else:
        dataloader = test_loader

    tokenizer = AutoTokenizer.from_pretrained(config["dataset"]["qwen_model_name"], use_fast=True)

    demo_caption_generation(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        device=device,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
    )


if __name__ == "__main__":
    main()