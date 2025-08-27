"""
📏 Metric registry for multimodal evaluation.

Provides:
- Access to metric functions by name
- Listing of all registered metrics
- evaluate_sample() logic based on task_type
"""

from typing import Callable, Dict, Union, List

from multimodal_eval.evaluation.clip_score import compute_clip_score
from multimodal_eval.evaluation.contextual_relevance import compute_contextual_relevance
from multimodal_eval.evaluation.hallucination_metric import compute_hallucination_metric
from multimodal_eval.evaluation.cider import compute_cider
from multimodal_eval.evaluation.semantic_similarity import compute_semantic_similarity
from multimodal_eval.evaluation.composite_score import compute_composite
from multimodal_eval.evaluation.schema import Sample

#  Registered metric functions
_METRICS: Dict[str, Callable] = {
    "clip_score": compute_clip_score,  # Visual grounding (CLIP)
    "hallucination_metric": compute_hallucination_metric,  # Object hallucination
    "contextual_relevance": compute_contextual_relevance,  # Relevance to prompt/context
    "semantic_similarity": compute_semantic_similarity,  # Semantic similarity (SBERT)
    "cider": compute_cider,  # Lexical similarity (CIDEr)
}

# Metrics associated with each task type
TASK_METRICS: Dict[str, List[str]] = {
    "captioning": ["cider", "clip_score", "semantic_similarity"],
    "hallucination": ["hallucination_metric", "semantic_similarity"],
    "vqa": ["clip_score", "semantic_similarity", "contextual_relevance"],
    "contextual_relevance": ["clip_score", "cider", "semantic_similarity"],
}


def get_metric(name: str) -> Callable:
    """
    Returns a metric function by its name.

    :param name: Name of the metric
    :return: Callable function that computes the metric
    """
    if name not in _METRICS:
        raise ValueError(
            f"[❌] Metric '{name}' is not registered.\n"
            f"🔍 Available metrics: {list(_METRICS.keys())}"
        )
    return _METRICS[name]


def list_available() -> list:
    """Returns the list of all available metrics."""
    return list(_METRICS.keys())


def evaluate_sample(
        sample: Union[Sample, List[Sample]],
        task_type: str = "captioning",
        model_name: str = "llava"
) -> dict:
    """
    Evaluates a sample or list of samples using appropriate metrics for the given task.

    :param sample: Sample or list of Samples to evaluate
    :param task_type: Task type (e.g., captioning, vqa, hallucination, etc.)
    :param model_name: Name of the model used (passed for compatibility, not used here)
    :return: Dict of evaluation results per sample ID
    """
    samples = [sample] if isinstance(sample, Sample) else sample
    metrics_to_use = TASK_METRICS.get(task_type, [])

    results = {}
    for s in samples:
        sample_metrics = {}

        for metric_name in metrics_to_use:
            metric_func = get_metric(metric_name)
            result = metric_func(s)

            # Handle result (single score or dict of scores)
            if isinstance(result, dict):
                sample_metrics.update(result)
            else:
                sample_metrics[metric_name] = result

        if "hallucination_metric" in sample_metrics and "hallucination_score" not in sample_metrics:
            sample_metrics["hallucination_score"] = sample_metrics["hallucination_metric"]

        sample_metrics["composite_score"] = compute_composite(sample_metrics, task_type)
        results[s.id] = sample_metrics

    return results
