from __future__ import annotations

import sys
import os
from typing import Optional

# Ensure root path is added for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from multimodal_eval.llm_eval.wrappers.openai_judge import call_llm_judge
from multimodal_eval.llm_eval.tasks.captioning import (
    score_captioning,
    build_caption_eval_prompt,
)
from multimodal_eval.llm_eval.tasks.contextual_relevance import (
    score_contextual_relevance,
    build_contextual_prompt,
)
from multimodal_eval.llm_eval.tasks.hallucination import (
    score_hallucination,
    build_hallucination_prompt,
)
from multimodal_eval.llm_eval.tasks.schemas import Sample


def call_judge(
    task: str,
    image_path: str,
    *,
    caption: Optional[str] = None,
    prompt: Optional[str] = None,
    response: Optional[str] = None,
    objects: Optional[list[str]] = None
):
    """
    Unified interface for scoring captioning, contextual_relevance, and hallucination tasks.
    """
    if task == "captioning":
        assert caption, "Caption must be provided for captioning task"
        prompt_str, system_prompt = build_caption_eval_prompt(caption)

        llm_response = call_llm_judge(
            task=task,
            image_path=image_path,
            prompt=prompt_str,
            system_prompt=system_prompt
        )

        sample = Sample(image=image_path, model_output=caption)

        result = score_captioning(
            sample=sample,
            hallucination_ok=1.0 - llm_response.get("hallucination_raw", 0.0),
        )
        return result

    elif task == "contextual_relevance":
        assert prompt and response, "Prompt and response must be provided for contextual_relevance"
        prompt_str, system_prompt = build_contextual_prompt(prompt, response)

        llm_response = call_llm_judge(
            task=task,
            image_path=image_path,
            prompt=prompt_str,
            system_prompt=system_prompt
        )

        sample = Sample(image=image_path, model_output=response)

        result = score_contextual_relevance(
            sample=sample,
            image_grounding=llm_response.get("image_grounding", 0.0),
            prompt_alignment=llm_response.get("prompt_alignment", 0.0),
            hallucination_raw=llm_response.get("hallucination_raw", 0.0),
            reason=llm_response.get("reason", "")
        )
        return result

    elif task == "hallucination":
        assert caption and objects is not None, "Caption and objects must be provided for hallucination"

        result = score_hallucination(
            image=image_path,
            caption=caption,
            objects=objects
        )
        return result

    else:
        raise ValueError(f"Unsupported task: {task}")
