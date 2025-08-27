from typing import Dict

def compute_composite(metrics: Dict, task_type: str = "captioning") -> float:

    cider      = float(metrics.get("cider", 0.0) or 0.0)                  # [0..1]
    clip_score = float(metrics.get("clip_score", 0.0) or 0.0)             # [0..1]
    semantic   = float(metrics.get("semantic_similarity", 0.0) or 0.0)    # [0..1]
    contextual = float(metrics.get("contextual_relevance", 0.0) or 0.0)   # [0..1]

    if task_type == "captioning":
        composite = 0.25 * clip_score + 0.50 * semantic + 0.25 * cider
    elif task_type == "vqa":
        composite = 0.10 * clip_score + 0.40 * semantic + 0.50 * contextual
    elif task_type == "contextual_relevance":
        composite = 0.40 * clip_score + 0.30 * cider + 0.30 * semantic
    else:
        composite = 0.0

    # на всякий случай — жёстко в [0..1] и округление
    return round(max(0.0, min(1.0, composite)), 4)
