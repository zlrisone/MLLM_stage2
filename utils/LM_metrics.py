import json
from typing import List, Union, Dict, Any

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
def _validate_inputs(
    predictions: List[str],
    references: List[Union[str, List[str]]]
) -> None:
    if not isinstance(predictions, list) or not isinstance(references, list):
        raise TypeError("predictions and references must both be lists.")

    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: len(predictions)={len(predictions)} "
            f"!= len(references)={len(references)}"
        )

    for i, pred in enumerate(predictions):
        if not isinstance(pred, str):
            raise TypeError(f"predictions[{i}] must be str, got {type(pred)}")

    for i, ref in enumerate(references):
        if isinstance(ref, str):
            continue
        if isinstance(ref, list):
            if len(ref) == 0:
                raise ValueError(f"references[{i}] is an empty list.")
            for j, r in enumerate(ref):
                if not isinstance(r, str):
                    raise TypeError(
                        f"references[{i}][{j}] must be str, got {type(r)}"
                    )
        else:
            raise TypeError(
                f"references[{i}] must be str or list[str], got {type(ref)}"
            )
def _to_coco_format(
    predictions: List[str],
    references: List[Union[str, List[str]]]
):
    """
    转成 pycocoevalcap 所需格式:
      refs:  {id: [ref1, ref2, ...]}
      cands: {id: [pred]}
    """
    refs = {}
    cands = {}

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if isinstance(ref, str):
            ref = [ref]

        refs[i] = ref
        cands[i] = [pred]

    return refs, cands
def _simple_tokenize(text: str) -> List[str]:
    return text.strip().lower().split()

def _compute_meteor(
    predictions: List[str],
    references: List[Union[str, List[str]]]
):
    meteor_scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = _simple_tokenize(pred)

        if isinstance(ref, str):
            ref_tokens = [_simple_tokenize(ref)]
        else:
            ref_tokens = [_simple_tokenize(r) for r in ref]

        score = meteor_score(ref_tokens, pred_tokens)
        meteor_scores.append(float(score))

    meteor_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    return meteor_avg, meteor_scores

def evaluate_caption(
    predictions: List[str],
    references: List[Union[str, List[str]]]
) -> Dict[str, Any]:
    """
    计算 caption 指标。

    Args:
        predictions: 生成结果列表，长度为 N，每个元素是一个字符串
        references: 参考答案列表，长度为 N
                    - 单参考: list[str]
                    - 多参考: list[list[str]]

    Returns:
        {
            "BLEU-1": float,
            "BLEU-2": float,
            "BLEU-3": float,
            "BLEU-4": float,
            "METEOR": float,
            "ROUGE-L": float,
            "CIDEr": float,
            "per_sample": {
                "BLEU": ...,
                "METEOR": ...,
                "ROUGE-L": ...,
                "CIDEr": ...
            }
        }
    """
    _validate_inputs(predictions, references)
    refs, cands = _to_coco_format(predictions, references)

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr"),
    ]

    results: Dict[str, Any] = {}
    per_sample: Dict[str, Any] = {}

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, cands)

        if isinstance(method, list):
            for m, s, ss in zip(method, score, scores):
                results[m] = float(s)
                per_sample[m] = [float(x) for x in ss]
        else:
            results[method] = float(score)
            per_sample[method] = [float(x) for x in scores]
    meteor_avg, meteor_scores = _compute_meteor(predictions, references)
    results["METEOR"] = meteor_avg
    per_sample["METEOR"] = meteor_scores
    results["per_sample"] = per_sample
    return results

if __name__ == "__main__":
    predictions = [
  "a cat sitting on a sofa",
  "a dog running in the grass"
]
    references = [
  "a cat is sitting on the couch",
  "a dog runs through a grassy field"
]   
    results = evaluate_caption(predictions, references)

    print(json.dumps(
        {k: v for k, v in results.items() if k != "per_sample"},
        ensure_ascii=False,
        indent=2
    ))

    save_path = "metrics.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to: {save_path}")