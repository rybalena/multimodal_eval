from __future__ import annotations
import json
from typing import Dict, Any, Optional, List

from multimodal_eval.llm_eval.tasks.schemas import HallucinationScores
from multimodal_eval.llm_eval.wrappers.openai_judge import call_llm_judge
from multimodal_eval.llm_eval.tasks.weight_utils import weighted_sum
from multimodal_eval.llm_eval.utils.logger import print_scores_with_reason
from multimodal_eval.llm_eval.utils.prompts import build_hallucination_prompt

# Weights for final composite hallucination score
_HALLUC_WEIGHTS = {
    "hallucination_ok": 0.80,
    "auto_hallucination_ok": 0.20,
}


def score_hallucination(
    *,
    image: str,
    caption: str,
    objects: Optional[List[str]] = None,
    auto_hallucination_ok: Optional[float] = None,
    debug: bool = False
) -> HallucinationScores:

    prompt = build_hallucination_prompt(
        caption=caption,
        ground_truth_objects=objects or [],
    )

    try:
        gpt_response: Dict[str, Any] = call_llm_judge(
            task="hallucination",
            image_path=image,
            prompt=prompt
        )

        if isinstance(gpt_response, str):
            gpt_response = json.loads(gpt_response)

    except Exception as e:
        gpt_response = {
            "hallucinated_objects": [],
            "missing_objects": [],
            "semantic_issues": [],
            "hallucination_score": 1.0,
            "reason": f"Evaluation failed: {str(e)}"
        }

    hallucination_score = float(gpt_response.get("hallucination_score", 1.0))
    hallucination_ok = max(0.0, min(1.0, 1.0 - hallucination_score))
    reason = gpt_response.get("reason", "")

    hallucinated_objects = gpt_response.get("hallucinated_objects", [])
    missing_objects = gpt_response.get("missing_objects", [])
    semantic_issues = gpt_response.get("semantic_issues", [])

    parts = {
        "hallucination_ok": hallucination_ok,
        "auto_hallucination_ok": auto_hallucination_ok,
    }
    composite = weighted_sum(parts, _HALLUC_WEIGHTS)

    result = HallucinationScores(
        hallucination_ok=hallucination_ok,
        auto_hallucination_ok=auto_hallucination_ok,
        composite=composite,
        reason=reason
    )

    if debug:
        print("\n🔍 HALLUCINATION DEBUG INFO")
        print(f"🎯 ground_truth_objects: {objects}")
        print(f"📛 hallucinated_objects: {hallucinated_objects}")
        print(f"📉 missing_objects: {missing_objects}")
        print(f"🧠 semantic_issues: {semantic_issues}")
        print(f"📈 hallucination_score: {hallucination_score}")
        print(f"✅ hallucination_ok: {hallucination_ok:.3f}")
        print(f"🧮 composite: {composite:.3f}")

    print_scores_with_reason("HALLUCINATION (unlabeled)", result, caption)

    return result
