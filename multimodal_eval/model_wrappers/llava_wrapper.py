import requests
import base64
from multimodal_eval.evaluation.schema import Sample
from multimodal_eval.model_wrappers.base_wrapper import BaseModelWrapper


class LlavaWrapper(BaseModelWrapper):
    """
    Wrapper for LLaVA model via Ollama API.
    Supports both captioning and VQA based on the sample content.
    """

    _printed = False

    def __init__(self, host="http://localhost:11434", default_task="captioning"):
        self.url = f"{host}/api/generate"
        self.default_task = default_task


    def ensure_ready(self):
        """Print a one-time 'ready' message; safe to call multiple times."""
        if not LlavaWrapper._printed:
            print(f"✅ LLaVA model loaded via Ollama API at {self.url}")
            LlavaWrapper._printed = True

    def generate(
        self,
        prompt: str,
        image_path: str,
        temperature: float = 0.3,
        max_tokens: int = 150,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        task_type: str = None,
        short_response: bool = False
    ) -> str:
        """Generates text using LLaVA through Ollama API."""
        try:
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            return f"[ERROR] Failed to read image: {e}"

        payload = {
            "model": "llava",
            "prompt": prompt.strip(),
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
                "repeat_penalty": repetition_penalty,
                "task": task_type or self.default_task
            }
        }

        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            data = response.json()
            result = data.get("response", "").strip()
        except Exception as e:
            return f"[ERROR] LLaVA API error: {e}"

        # Postprocessing
        if short_response:
            dot_index = result.find(".")
            result = result[:dot_index + 1].strip() if dot_index != -1 else result.strip()
        else:
            sentences = result.split(".")
            clean_sentences = [s.strip() for s in sentences if s.strip()]
            result = ". ".join(clean_sentences[:3])
            if result and not result.endswith("."):
                result += "."

        return result

    def run(self, sample: Sample, task_type: str = None, image_path: str = None) -> str:
        """Auto-detects VQA or captioning and generates a response."""
        self.ensure_ready()

        if task_type is None:
            if (
                hasattr(sample, "question")
                and isinstance(sample.question, str)
                and sample.question.strip()
            ):
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
            repetition_penalty=1.1,
            task_type=task_type,
            short_response=short_response
        )
