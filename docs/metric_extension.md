# Metric Extension Guide

## Overview
The framework provides a **metric registry** to manage evaluation metrics.  
To add a new metric, you must:  
1. Implement the metric function.  
2. Register it in **`metrics_registry.py`**.  
3. Ensure it is associated with relevant tasks.

---

## 1. Metric Registry

Defined in **`metrics_registry.py`**:

```python
# Registered metrics
_METRICS: Dict[str, Callable] = {
    "clip_score": compute_clip_score,
    "hallucination_metric": compute_hallucination_metric,
    "contextual_relevance": compute_contextual_relevance,
    "semantic_similarity": compute_semantic_similarity,
    "cider": compute_cider,
}

# Metrics per task
TASK_METRICS: Dict[str, List[str]] = {
    "captioning": ["cider", "clip_score", "semantic_similarity"],
    "hallucination": ["hallucination_metric", "semantic_similarity"],
    "vqa": ["clip_score", "semantic_similarity", "contextual_relevance"],
    "contextual_relevance": ["clip_score", "cider", "semantic_similarity"],
}
```

Helper functions:
- **`get_metric(name)`** → returns a metric function by name.  
- **`list_available()`** → lists all available metrics.  
- **`evaluate_sample(sample, task_type)`** → applies relevant metrics and computes composite score.

---

## 2. Example: CLIP Score

Defined in **`clip_score.py`**.

```python
def compute_clip_score(sample: Sample) -> float:
    """
    Cosine similarity between image and generated caption using CLIP.
    Returns value in [0.0, 1.0].
    """
    if not getattr(sample, "image", None) or not getattr(sample, "generated_answer", None):
        return 0.0

    _ensure_clip_loaded()
    image = load_image_any(sample.image)
    caption = clean_and_truncate_caption(sample.generated_answer)

    # Encode with CLIP
    inputs = _CLIP_PROC(text=[caption], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        img_f = _CLIP_MODEL.get_image_features(pixel_values=inputs["pixel_values"].to(_DEVICE))
        txt_f = _CLIP_MODEL.get_text_features(
            input_ids=inputs["input_ids"].to(_DEVICE),
            attention_mask=inputs["attention_mask"].to(_DEVICE),
        )
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        sim = float((img_f @ txt_f.T).squeeze().item())

    return round((sim + 1.0) / 2.0, 4)
```

---

## 3. Adding a New Metric

### Step 1 — Implement the function
Create `evaluation/my_metric.py`:

```python
def compute_my_metric(sample: Sample) -> float:
    """
    Example custom metric:
    Compare the length of generated vs expected answer.
    """
    if not sample.generated_answer or not sample.expected_answer:
        return 0.0
    return min(len(sample.generated_answer), len(sample.expected_answer)) / max(
        len(sample.generated_answer), len(sample.expected_answer)
    )
```

### Step 2 — Register in `metrics_registry.py`

```python
from multimodal_eval.evaluation.my_metric import compute_my_metric

_METRICS = {
    **_METRICS,
    "my_metric": compute_my_metric,
}

TASK_METRICS["captioning"].append("my_metric")
```

### Step 3 — Use in evaluation

```python
from multimodal_eval.evaluation.metrics_registry import evaluate_sample

result = evaluate_sample(sample, task_type="captioning")
print(result["my_metric"])
```

---

## 4. Summary

- Each metric is a **function**: `(Sample) -> float | dict`.  
- Register metrics in `_METRICS` and link to tasks in `TASK_METRICS`.  
- Use `evaluate_sample()` to automatically include new metrics.  
- Example: `clip_score` demonstrates a vision-language similarity metric.  
