from __future__ import annotations
from typing import Optional

from multimodal_eval.llm_eval.tasks.schemas import ContextualScores, Sample
from multimodal_eval.llm_eval.tasks.weight_utils import weighted_sum
from multimodal_eval.llm_eval.utils.logger import print_scores_with_reason
from multimodal_eval.llm_eval.utils.prompts import build_contextual_prompt

# 💡 Weights for the contextual relevance score
_CONTEXTUAL_WEIGHTS = {
    "image_grounding": 0.40,
    "prompt_alignment": 0.40,
    "hallucination_ok": 0.20,
}


def score_contextual_relevance(
    *,
    image_grounding: Optional[float] = None,
    prompt_alignment: Optional[float] = None,
    hallucination_raw: Optional[float] = None,
    hallucination_ok: Optional[float] = None,
    strict: bool = False,
    reason: Optional[str] = None,
    model_output: Optional[str] = None,
    sample: Optional[Sample] = None
) -> ContextualScores:
    """
    Computes contextual relevance score from LLM judgments.
    """
    if sample:
        model_output = model_output or sample.model_output

    # Auto-convert hallucination_raw → hallucination_ok
    if hallucination_ok is None and hallucination_raw is not None:
        hallucination_ok = max(0.0, min(1.0, 1.0 - hallucination_raw))

    parts = {
        "image_grounding": image_grounding,
        "prompt_alignment": prompt_alignment,
        "hallucination_ok": hallucination_ok,
    }

    composite = weighted_sum(parts, _CONTEXTUAL_WEIGHTS, strict=strict)

    result = ContextualScores(
        image_grounding=image_grounding,
        prompt_alignment=prompt_alignment,
        hallucination_ok=hallucination_ok,
        composite=composite,
        reason=reason
    )

    if sample:
        print_scores_with_reason(
            label="CONTEXTUAL RELEVANCE (unlabeled)",
            scores=result,
            model_output=model_output,
            sample=sample
        )

    return result
