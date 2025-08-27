from typing import Dict, Any, Optional
import json

from multimodal_eval.llm_eval.tasks.schemas import CaptionScores
from multimodal_eval.llm_eval.wrappers.openai_judge import call_llm_judge
from multimodal_eval.llm_eval.tasks.weight_utils import weighted_sum
from multimodal_eval.llm_eval.utils.logger import print_scores_with_reason
from multimodal_eval.llm_eval.utils.prompts import build_caption_eval_prompt

#  Weights for the final composite score
_CAPTION_WEIGHTS = {
    "image_grounding": 0.35,
    "coverage": 0.25,
    "hallucination_ok": 0.20,
    "fluency": 0.20,
}


def score_captioning(sample, hallucination_ok: Optional[float] = None, strict: bool = False) -> CaptionScores:
    """
    Evaluates a model-generated caption using LLM scoring.
    """
    prompt = build_caption_eval_prompt(sample.model_output)

    try:
        gpt_response: Dict[str, Any] = call_llm_judge(
            task="captioning",
            image_path=sample.image,
            prompt=prompt,
            system_prompt="You are a helpful assistant that evaluates captions for images."
        )
        if isinstance(gpt_response, str):
            gpt_response = json.loads(gpt_response)
    except Exception as e:
        print(f"[Judge Error] {e}")
        gpt_response = {
            "image_grounding": 0.0,
            "coverage": 0.0,
            "fluency": 0.0,
            "hallucination_raw": 1.0,
            "reason": f"Evaluation failed: {str(e)}"
        }

    image_grounding = float(gpt_response.get("image_grounding", 0.0))
    coverage = float(gpt_response.get("coverage", 0.0))
    fluency = float(gpt_response.get("fluency", 0.0))
    hallucination_raw = float(gpt_response.get("hallucination_raw", 0.0))
    hallucination_ok = float(gpt_response.get("hallucination_ok", 1.0)) if hallucination_ok is None else hallucination_ok
    reason = gpt_response.get("reason", "")


    if hallucination_ok is None and hallucination_raw is not None:
        hallucination_ok = max(0.0, min(1.0, 1.0 - hallucination_raw))

    parts = {
        "image_grounding": image_grounding,
        "coverage": coverage,
        "hallucination_ok": hallucination_ok,
        "fluency": fluency,
    }

    composite = weighted_sum(parts, _CAPTION_WEIGHTS, strict=strict)

    #  Debug output
    print("\n🧪 Task: CAPTIONING (unlabeled)")
    print(f"🖼️ Image: {sample.image}")
    print(f"🤖 Model Output: {sample.model_output.strip()}\n")
    print("📊 Individual Scores:")
    print(f"  • image_grounding   : {image_grounding:.2f}")
    print(f"  • coverage          : {coverage:.2f}")
    print(f"  • hallucination_ok  : {hallucination_ok:.2f}")
    print(f"  • fluency           : {fluency:.2f}")
    print(f"   composite_score   : {composite:.4f}")

    result = CaptionScores(
        image_grounding=image_grounding,
        coverage=coverage,
        hallucination_ok=hallucination_ok,
        fluency=fluency,
        composite=composite,
        reason=reason
    )

    print_scores_with_reason(
        label="CAPTIONING (unlabeled)",
        scores=result,
        model_output=sample.model_output,
        sample=sample
    )

    return result
