# Composite Score Guide

## Overview
The **composite score** provides a single aggregated number that summarizes model performance on a task.  
It combines multiple metrics with task-specific weights, ensuring balanced evaluation.

---

## 1. Normalization

The **CIDEr score** is normalized to `[0, 1]` before aggregation:

```python
def normalize_cider(cider: float, max_cider: float = 1.0) -> float:
    if not isinstance(cider, (float, int)) or cider < 0:
        return 0.0
    return min(cider / max_cider, 1.0)
```

This ensures comparability between CIDEr and other metrics.

---

## 2. Composite Score Formula

Implemented in **`compute_composite()`**.  
The formula depends on the task type:

### Captioning
```text
composite = 0.25 * clip_score
          + 0.50 * semantic_similarity
          + 0.25 * normalized_cider
```

### VQA
```text
composite = 0.10 * clip_score
          + 0.40 * semantic_similarity
          + 0.50 * contextual_relevance
```

### Contextual Relevance
```text
composite = 0.40 * clip_score
          + 0.30 * normalized_cider
          + 0.30 * semantic_similarity
```

### Hallucination
Handled separately in **`hallucination_metric.py`**, not inside `composite_score.py`.

---

## 3. Task-Specific Metrics

Defined in **`tasks_config.py`**:

- **Captioning:** `cider`, `clip_score`, `semantic_similarity`  
- **Hallucination:** `hallucination_metric`  
- **VQA:** `clip_score`, `semantic_similarity`, `contextual_relevance`  
- **Contextual relevance:** `clip_score`, `cider`, `semantic_similarity`  

---

## 4. Example

**Input metrics (captioning):**
```json
{
  "clip_score": 0.72,
  "semantic_similarity": 0.81,
  "cider": 0.67
}
```

**Composite score:**
```
0.25 * 0.72 + 0.50 * 0.81 + 0.25 * 0.67 = 0.75
```

---

## 5. Summary

- Composite score is **task-dependent**.  
- CIDEr is **normalized** before aggregation.  
- Hallucination uses a **dedicated metric**.  
- Weights reflect relative importance of semantic vs. lexical vs. contextual correctness.  
