from __future__ import annotations
from typing import Callable, Dict, Any
from multimodal_eval.llm_eval.tasks.captioning import score_captioning
from multimodal_eval.llm_eval.tasks.contextual_relevance import score_contextual_relevance
from multimodal_eval.llm_eval.tasks.hallucination import score_hallucination

CompositeFn = Callable[..., Any]

REGISTRY: Dict[str, CompositeFn] = {
    "captioning": score_captioning,
    "contextual_relevance": score_contextual_relevance,
    "hallucination": score_hallucination,
}


def compute(task: str, **kwargs):
    try:
        fn = REGISTRY[task]
    except KeyError as e:
        raise ValueError(f"Unknown LLM composite task: {task}") from e
    return fn(**kwargs)
