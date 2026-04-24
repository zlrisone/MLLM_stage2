"""
VLM-as-a-Judge 评测脚本
=======================

使用阿里 DashScope 的 Qwen-VL (OpenAI 兼容接口) 作为 judge，
对当前模型在测试集上的 caption 输出做 pointwise 多维打分。

两种使用模式：

1. 跑模型推理 + judge（需要 checkpoint）：
    python eval_judge.py \
        --config config/training_stage2.yaml \
        --checkpoint outputs/best_checkpoint.pt \
        --judge_model qwen-vl-max \
        --output_json outputs/judge_results.json \
        --limit 100

2. 只对已有的预测 JSON 做 judge（不跑推理）：
    python eval_judge.py \
        --predictions_json outputs/test_predictions.json \
        --image_root ../llava_cc3m_raw/images \
        --judge_model qwen-vl-max \
        --output_json outputs/judge_results.json

API key：通过 --api_key 或环境变量 DASHSCOPE_API_KEY 提供。
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from data.caption_dataset import build_dataloders
from models.multimodal_model import create_multimodal_model
from utils.checkpoint import load_checkpoint


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

JUDGE_DIMENSIONS = ["accuracy", "detail", "relevance", "fluency", "overall"]

JUDGE_SYSTEM_PROMPT = (
    "You are a strict and fair multimodal evaluator. Given an image, an optional "
    "reference caption and a candidate caption produced by a vision-language model, "
    "you will score the candidate caption against what the image actually shows. "
    "Always respond with a single JSON object, no extra text."
)

JUDGE_USER_TEMPLATE = """请作为一个严谨的视觉-语言模型评测员，根据下面这张图片，对候选描述（candidate）进行打分。

[参考描述]（可能为空，仅供参考，不必完全一致；以图片为准）：
{reference}

[候选描述]：
{candidate}

请从以下 5 个维度对候选描述进行 1-5 分的整数打分（5 分最好，1 分最差）：
- accuracy : 候选描述与图片内容的事实一致性，有无幻觉/错误
- detail   : 是否涵盖图片中重要的物体/属性/场景细节
- relevance: 是否切题、没有无关信息
- fluency  : 语言是否流畅、无语法错误
- overall  : 综合总分

严格输出一个 JSON 对象，不要任何额外解释，不要 markdown 代码块，格式：
{{"accuracy": int, "detail": int, "relevance": int, "fluency": int, "overall": int, "reason": "一句话中文解释"}}
"""


# ---------------------------------------------------------------------------
# Judge 调用（DashScope OpenAI 兼容）
# ---------------------------------------------------------------------------
def encode_image_to_data_uri(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
        "bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def call_qwen_vl_judge(
    client,
    model_name: str,
    image_path: str,
    prediction: str,
    reference: Optional[str],
    max_retries: int = 3,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    data_uri = encode_image_to_data_uri(image_path)
    user_text = JUDGE_USER_TEMPLATE.format(
        reference=reference if reference else "（无参考）",
        candidate=prediction if prediction else "（空）",
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
            )
            content = resp.choices[0].message.content or ""
            parsed = _extract_json(content)
            if parsed is None:
                raise ValueError(f"Judge response is not valid JSON: {content!r}")

            for k in JUDGE_DIMENSIONS:
                if k in parsed:
                    try:
                        parsed[k] = int(parsed[k])
                    except Exception:
                        pass
            parsed["_raw"] = content
            return parsed
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 10))

    return {"error": last_err}


# ---------------------------------------------------------------------------
# 模式 1：用当前 checkpoint 跑推理
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_predictions(
    model,
    tokenizer,
    test_dataset,
    device: torch.device,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    model.eval()
    total = len(test_dataset)
    if limit is not None:
        total = min(limit, total)

    records: List[Dict[str, Any]] = []
    for i in tqdm(range(total), desc="Inference"):
        sample = test_dataset[i]

        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

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

        records.append(
            {
                "sample_id": i,
                "image_path": sample["image_path"],
                "reference": sample["reference"],
                "prediction": pred_text,
            }
        )
    return records


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------
def aggregate_scores(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    sums = {k: 0.0 for k in JUDGE_DIMENSIONS}
    counts = {k: 0 for k in JUDGE_DIMENSIONS}
    num_errors = 0

    for r in results:
        j = r.get("judge", {}) or {}
        if "error" in j:
            num_errors += 1
            continue
        for k in JUDGE_DIMENSIONS:
            v = j.get(k)
            if isinstance(v, (int, float)):
                sums[k] += float(v)
                counts[k] += 1

    avg = {k: (sums[k] / counts[k] if counts[k] else 0.0) for k in JUDGE_DIMENSIONS}
    return {
        "average": avg,
        "valid_counts": counts,
        "num_samples": len(results),
        "num_errors": num_errors,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def load_predictions_from_json(
    path: str,
    image_root: str = "",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        raw_preds = data
    else:
        raw_preds = data.get("predictions", [])

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(raw_preds):
        if not isinstance(item, dict):
            continue
        image_path = item.get("image_path") or item.get("image") or ""
        if image_path and not os.path.isabs(image_path) and image_root:
            image_path = os.path.join(image_root, image_path)

        if not image_path or not os.path.isfile(image_path):
            print(f"[warn] 样本 {i} 无效 image_path: {image_path!r}，跳过")
            continue

        out.append(
            {
                "sample_id": item.get("sample_id", i),
                "image_path": image_path,
                "reference": item.get("reference", ""),
                "prediction": item.get("prediction", ""),
            }
        )

    if limit is not None:
        out = out[:limit]
    return out


def build_predictions_from_inference(
    config_path: str,
    checkpoint_path: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Using device: {device}")

    model = create_multimodal_model(config).to(device)
    model.eval()
    load_checkpoint(checkpoint_path, model)

    tokenizer = AutoTokenizer.from_pretrained(
        config["dataset"]["qwen_model_name"], use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, _, test_loader = build_dataloders(
        vision_model_name=config["dataset"]["vision_model_name"],
        qwen_model_name=config["dataset"]["qwen_model_name"],
        chat_json_path=config["dataset"]["chat_json_path"],
        image_root=config["dataset"]["image_root"],
        batch_size=1,
        num_workers=0,
        max_length=config["dataset"]["max_length"],
        train_ratio=config["dataset"]["train_ratio"],
        val_ratio=config["dataset"]["val_ratio"],
        test_ratio=config["dataset"]["test_ratio"],
    )
    test_dataset = test_loader.dataset

    gen_cfg = config.get("generation", {})
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        device=device,
        max_new_tokens=gen_cfg.get("max_new_tokens", 64),
        num_beams=gen_cfg.get("num_beams", 1),
        do_sample=gen_cfg.get("do_sample", False),
        limit=limit,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="VLM-as-a-Judge evaluation using DashScope Qwen-VL"
    )
    parser.add_argument("--config", type=str, default="./config/training_stage2.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="checkpoint 路径；若提供 --predictions_json 则忽略",
    )
    parser.add_argument(
        "--predictions_json",
        type=str,
        default="",
        help="已有预测 JSON 文件；提供后不再跑模型推理",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="",
        help="predictions_json 中 image_path 为相对路径时使用",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="qwen-vl-max",
        help="DashScope 模型名，如 qwen-vl-max / qwen-vl-plus / qwen2.5-vl-72b-instruct",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="DashScope API key；不填则读环境变量 DASHSCOPE_API_KEY",
    )
    parser.add_argument(
        "--output_json", type=str, default="outputs/judge_results.json"
    )
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条样本")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--save_every",
        type=int,
        default=20,
        help="每 N 条样本写一次中间结果，便于断点续看",
    )
    args = parser.parse_args()

    use_json = bool(args.predictions_json)
    if not use_json and not args.checkpoint:
        raise ValueError(
            "请提供 --checkpoint 跑推理，或提供 --predictions_json 直接评测。"
        )

    if use_json:
        print(f"[Mode] 从 JSON 读取预测: {args.predictions_json}")
        predictions = load_predictions_from_json(
            args.predictions_json, image_root=args.image_root, limit=args.limit
        )
    else:
        print(f"[Mode] 跑模型推理: {args.checkpoint}")
        predictions = build_predictions_from_inference(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            limit=args.limit,
        )

    if not predictions:
        raise RuntimeError("没有可评测的样本，请检查输入。")
    print(f"[Judge] 共 {len(predictions)} 条样本需要评测")

    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("请先安装 openai: pip install openai") from e

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "请通过 --api_key 或环境变量 DASHSCOPE_API_KEY 提供 DashScope API key"
        )

    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    judged: List[Dict[str, Any]] = []
    pbar = tqdm(predictions, desc=f"Judging({args.judge_model})", dynamic_ncols=True)
    running_sum = 0.0
    running_n = 0

    for idx, item in enumerate(pbar):
        judge_result = call_qwen_vl_judge(
            client=client,
            model_name=args.judge_model,
            image_path=item["image_path"],
            prediction=item["prediction"],
            reference=item.get("reference", ""),
            max_retries=args.max_retries,
            temperature=args.temperature,
        )
        out = dict(item)
        out["judge"] = judge_result
        judged.append(out)

        overall = judge_result.get("overall")
        if isinstance(overall, (int, float)):
            running_sum += float(overall)
            running_n += 1
            pbar.set_postfix(
                {"overall": f"{overall}", "avg": f"{running_sum / running_n:.2f}"}
            )

        if args.save_every > 0 and (idx + 1) % args.save_every == 0:
            _save(out_path, args.judge_model, judged)

    metrics = aggregate_scores(judged)
    _save(out_path, args.judge_model, judged, metrics=metrics)

    print("=" * 60)
    print(f"[Judge Done] model={args.judge_model}  n={metrics['num_samples']}  "
          f"errors={metrics['num_errors']}")
    for k, v in metrics["average"].items():
        print(f"  {k:<9}: {v:.3f}  (valid={metrics['valid_counts'][k]})")
    print(f"Saved -> {out_path}")


def _save(
    out_path: Path,
    judge_model: str,
    judged: List[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]] = None,
):
    if metrics is None:
        metrics = aggregate_scores(judged)
    save_obj = {
        "judge_model": judge_model,
        "aggregate": metrics,
        "results": judged,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
