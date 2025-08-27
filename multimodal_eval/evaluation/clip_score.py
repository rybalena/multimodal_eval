import os
import re
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from multimodal_eval.evaluation.schema import Sample
from multimodal_eval.utils.io import load_image_any


_MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-large-patch14-336")


_CLIP_MODEL: Optional[CLIPModel] = None
_CLIP_PROC:  Optional[CLIPProcessor] = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _ensure_clip_loaded():
    global _CLIP_MODEL, _CLIP_PROC
    if _CLIP_MODEL is None or _CLIP_PROC is None:
        _CLIP_MODEL = CLIPModel.from_pretrained(_MODEL_ID).to(_DEVICE)
        _CLIP_MODEL.eval()
        _CLIP_PROC  = CLIPProcessor.from_pretrained(_MODEL_ID)

def clean_and_truncate_caption(text: str, max_words: int = 32) -> str:

    t = str(text or "").strip()
    # remove common LLM prefixes
    t = re.sub(r"^(answer|caption)\s*:\s*", "", t, flags=re.I)

    t = re.sub(r"[^\w\s.,!?%\-:;()/#]+", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    return " ".join(t.split()[:max_words])

def compute_clip_score(sample: Sample) -> float:
    """
    Cosine similarity (CLIP) between image and text.
    Returns a value in [0.0, 1.0].
    """
    if not getattr(sample, "image", None) or not getattr(sample, "generated_answer", None):
        return 0.0

    try:
        _ensure_clip_loaded()
        if _CLIP_MODEL is None or _CLIP_PROC is None:
            return 0.0

        # Load image from a local path, file://, or http(s)://
        image: Image.Image = load_image_any(sample.image)

        caption = clean_and_truncate_caption(sample.generated_answer)
        if not caption:
            return 0.0

        inputs = _CLIP_PROC(
            text=[caption],
            images=image,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            img_f = _CLIP_MODEL.get_image_features(pixel_values=inputs["pixel_values"].to(_DEVICE))
            txt_f = _CLIP_MODEL.get_text_features(
                input_ids=inputs["input_ids"].to(_DEVICE),
                attention_mask=inputs["attention_mask"].to(_DEVICE),
            )

            # L2-normalization and cosine
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            sim = float((img_f @ txt_f.T).squeeze().item())

        # [-1, 1] → [0, 1]
        clip_score = (sim + 1.0) / 2.0
        # clamp and round
        clip_score = max(0.0, min(1.0, clip_score))
        return round(clip_score, 4)

    except Exception as e:
        print(f"[CLIP Score Error] sample_id={getattr(sample,'id',None)}: {e}")
        return 0.0
