from pydantic import BaseModel
from typing import Optional, Literal, List

class Sample(BaseModel):
    """
    Universal data model for multimodal evaluation.
    Covers tasks like captioning, VQA, hallucination, contextual relevance, and more.
    """

    id: str
    task: Literal["captioning", "vqa", "hallucination", "contextual_relevance"]

    image: str  # Image URL (automatically inserted by the loader)
    generated_answer: Optional[str] = None  # Model output (caption, answer, etc.)

    # For VQA
    question: Optional[str] = None
    expected_answer: Optional[str] = None

    # For captioning/hallucination
    prompt_template: Optional[str] = None
    caption: Optional[str] = None  # Labeled caption (ground truth)

    # For contextual relevance
    prompt: Optional[str] = None
    expected_behavior: Optional[str] = None
    expected_score: Optional[float] = None
    relevant: Optional[bool] = None

    # Additional fields
    generated_caption: Optional[str] = None
    hallucination_expected: Optional[str] = None
    model_output: Optional[str] = None

    # Fields for UNLABELED task
    judged_output: Optional[str] = None
    judge_notes: Optional[str] = None
