# metrics_eval.py
import os
from typing import Any, Dict

from multimodal_eval.evaluation.schema import Sample
from multimodal_eval.evaluation.evaluator import Evaluator
from multimodal_eval.evaluation.composite_score import compute_composite

TASK_METRICS = {
    "captioning": ["cider", "clip_score", "semantic_similarity"],
    "vqa": ["clip_score", "semantic_similarity", "contextual_relevance"],
    "contextual_relevance": ["clip_score", "cider", "semantic_similarity"],
}

def _map_metric_name(raw_name: str) -> str:
    c = "".join(ch.lower() for ch in str(raw_name) if ch.isalpha())
    if "cider" in c: return "cider"
    if "clip" in c: return "clip_score"
    if "semantic" in c: return "semantic_similarity"
    if "context" in c: return "contextual_relevance"
    if "composite" in c: return "composite_score"
    return raw_name

def _round_all(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: float(f"{float(v):.3f}") if isinstance(v, (int, float)) else v for k, v in d.items()}

def evaluate_golden(sample: Sample, task_type: str) -> dict:
    metrics = TASK_METRICS.get(task_type, [])
    evaluator = Evaluator(metrics=metrics, model_name="llava")
    result = evaluator.evaluate([sample], per_sample=True)

    per_sample = result.get("per_sample", {})
    out = {}
    for metric_name, rows in per_sample.items():
        for row in rows:
            if row.get("id") == sample.id:
                out[_map_metric_name(metric_name)] = float(row.get("score", 0.0) or 0.0)
                break

    if task_type != "hallucination":
        if task_type == "captioning":
            comp_in = {
                "clip_score": out.get("clip_score", 0.0),
                "semantic_similarity": out.get("semantic_similarity", 0.0),
                "cider": out.get("cider", 0.0),
            }
        elif task_type == "vqa":
            comp_in = {
                "clip_score": out.get("clip_score", 0.0),
                "semantic_similarity": out.get("semantic_similarity", 0.0),
                "contextual_relevance": out.get("contextual_relevance", 0.0),
            }
        else:
            comp_in = {
                "clip_score": out.get("clip_score", 0.0),
                "cider": out.get("cider", 0.0),
                "semantic_similarity": out.get("semantic_similarity", 0.0),
            }
        out["composite_score"] = compute_composite(comp_in, task_type=task_type)

    return _round_all(out)
