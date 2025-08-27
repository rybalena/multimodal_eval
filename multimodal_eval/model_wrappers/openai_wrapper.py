import os
import base64
from typing import Optional
import openai

from dotenv import load_dotenv
from multimodal_eval.evaluation.schema import Sample
from multimodal_eval.model_wrappers.base_wrapper import BaseModelWrapper

# Load environment variables from .env
load_dotenv()


class OpenAIWrapper(BaseModelWrapper):
    """
    Wrapper for OpenAI GPT-4o family. Supports both text and image inputs.
    API: Chat Completions (openai.chat.completions.create).
    """

    _printed = False

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError(
                "❌ OpenAI API key not found. Set OPENAI_API_KEY env variable or use a .env file."
            )

    def ensure_ready(self):
        """Print a one-time 'ready' message; safe to call multiple times."""
        if not OpenAIWrapper._printed:
            print(f"✅ OpenAI model '{self.model}' is ready (cloud API)")
            OpenAIWrapper._printed = True

    # ── public API used by orchestrator ─────────────────────────────────────────
    def run(self, sample: Sample, task_type: str = None, image_path: Optional[str] = None) -> str:
        """
        Auto-detects task type (vqa|captioning) by presence of sample.question and calls generate().

        """
        if task_type is None:
            if getattr(sample, "question", None) and str(sample.question).strip():
                task_type = "vqa"
            else:
                task_type = "captioning"

        if task_type == "vqa":
            prompt = (
                "You are a visual question answering assistant. "
                "Answer briefly and clearly based on the image. "
                f"Question: {sample.question}"
            )
            short_response = True
        else:
            prompt = (
                "You are an image caption assistant. "
                "Describe this image using 2 sentences. "
                "Mention objects, relationships, background. Now describe the image:"
            )
            short_response = False

        return self.generate(
            prompt=prompt,
            image_path=image_path or sample.image,
            temperature=0.2,
            max_tokens=150,
            top_p=0.9,
            short_response=short_response,
        )

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generates a response from OpenAI's GPT model, optionally with an image.
        Supports vision when model supports images (gpt-4o family).
        """
        messages = [{"role": "user", "content": prompt}]

        if self.supports_images() and image_path:
            try:
                with open(image_path, "rb") as img_file:
                    image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

                messages[0]["content"] = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ]
            except Exception as e:
                return f"[ERROR] Failed to load image: {e}"

        try:
            resp = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024),
                top_p=kwargs.get("top_p", 1.0),
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"[ERROR] OpenAI API error: {str(e)}"

        short_response = bool(kwargs.get("short_response", False))
        if short_response:
            dot = text.find(".")
            text = text[: dot + 1].strip() if dot != -1 else text.strip()
        else:
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            text = ". ".join(sentences[:3])
            if text and not text.endswith("."):
                text += "."

        return text

    def supports_images(self) -> bool:
        """
        Returns True if the model supports vision input (GPT-4o family).
        """
        m = self.model.lower()
        return any(tag in m for tag in ["gpt-4o", "gpt-4.1", "omni", "4o-mini", "o-mini"])
