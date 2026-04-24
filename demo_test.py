import argparse
import os

import torch
import yaml
from PIL import Image
from transformers import AutoTokenizer

from models.multimodal_model import create_multimodal_model
from utils.checkpoint import load_checkpoint


def build_prompt(user_prompt: str) -> str:
    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    user_text = f"{user_prompt}<|vision_start|><|image_pad|><|vision_end|>"
    return f"{system_prompt}<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def infer_single_image(
    model,
    tokenizer,
    image_path: str,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 128,
    num_beams: int = 1,
    do_sample: bool = False,
):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    image = Image.open(image_path).convert("RGB")
    pixel_values = model.vision_encoder.preprocess(image).to(device)

    full_prompt = build_prompt(prompt)
    tokenized = tokenizer(full_prompt, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    generated_ids = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
    )
    pred_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    print("=" * 80)
    print(f"Image: {image_path}")
    print("-" * 80)
    print("Prompt:")
    print(prompt)
    print("-" * 80)
    print("Prediction:")
    print(pred_text)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Local image demo inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--image_path", type=str, required=True, help="Local image path")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image.",
        help="Text prompt for image understanding",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens for generation")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search width")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_multimodal_model(config).to(device)
    model.eval()
    load_checkpoint(args.checkpoint, model)

    tokenizer = AutoTokenizer.from_pretrained(config["dataset"]["qwen_model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    infer_single_image(
        model=model,
        tokenizer=tokenizer,
        image_path=args.image_path,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
    )


if __name__ == "__main__":
    main()
"""
python demo_test.py \
  --config config/training_stage2.yaml \
  --checkpoint outputs/best_checkpoint.pt \
  --image_path "/Users/lris/Pictures/卡片机/P1040634.JPG" \
  --prompt "Please describe the image in detail."

"""