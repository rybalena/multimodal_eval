from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import math

# Helper function: clips float values to [0.0, 1.0]
def _clip01(x: Optional[float]) -> Optional[float]:
    if x is None or math.isnan(x):
        return None
    return max(0.0, min(1.0, float(x)))

# Scores for captioning tasks
class CaptionScores(BaseModel):
    image_grounding: Optional[float] = Field(default=None)
    coverage: Optional[float] = Field(default=None)
    hallucination_ok: Optional[float] = Field(default=None)
    fluency: Optional[float] = Field(default=None)
    composite: Optional[float] = Field(default=None)
    reason: Optional[str] = Field(default=None)

    @field_validator("image_grounding", "coverage", "hallucination_ok", "fluency", mode="before")
    def _val01(cls, v):
        return _clip01(v)

# Scores for contextual relevance tasks
class ContextualScores(BaseModel):
    image_grounding: Optional[float] = None
    prompt_alignment: Optional[float] = None
    hallucination_ok: Optional[float] = None
    composite: Optional[float] = None
    reason: Optional[str] = Field(default=None)

    @field_validator("image_grounding", "prompt_alignment", "hallucination_ok", mode="before")
    def _val01(cls, v):
        return _clip01(v)

# Scores for hallucination tasks
class HallucinationScores(BaseModel):
    hallucination_ok: Optional[float] = None
    auto_hallucination_ok: Optional[float] = None
    composite: Optional[float] = None
    reason: Optional[str] = Field(default=None)

    @field_validator("hallucination_ok", "auto_hallucination_ok", mode="before")
    def _val01(cls, v):
        return _clip01(v)

# Sample input to evaluate
class Sample(BaseModel):
    id: Optional[str] = Field(default="unknown")
    image: str
    model_output: str
    objects: List[str] = Field(default_factory=list)
