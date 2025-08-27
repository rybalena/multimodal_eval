import os
import time
import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor

try:
    # transformers >= 5
    from transformers import AutoModelForImageTextToText as _AutoVLModel
except Exception:
    # transformers < 5
    from transformers import AutoModelForVision2Seq as _AutoVLModel

from multimodal_eval.model_wrappers.base_wrapper import BaseModelWrapper


# ────────────────────────────────────────────────────────────────────────────────
# Offline / local model config
# ────────────────────────────────────────────────────────────────────────────────
MODEL_REPO = "Qwen/Qwen2-VL-2B-Instruct"
HF_HOME = Path.home() / "hf_models"
LOCAL_MODELS_DIR = HF_HOME / "local_models"
LOCAL_MODEL_DIR = LOCAL_MODELS_DIR / MODEL_REPO.split("/")[-1]
READY_FLAG = LOCAL_MODEL_DIR / ".ready"

FORCE_CPU = True            # Force CPU to avoid MPS issues
ALLOW_MPS = False           # Enable manually if you want to try MPS
EXIT_AFTER_DOWNLOAD = False  # Exit once model is fully downloaded (clean offline relaunch)

ALLOW_PATTERNS = [
    "*.json", "*.safetensors", "*.bin", "*.model", "*.txt",
    "tokenizer*", "*.py", "*.merges", "*.vocab", "*.tiktoken", "*.png"
]

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


# ────────────────────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────────────────────
class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): return False

class _online_env:
    _keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    def __enter__(self):
        self.prev = {k: os.environ.get(k) for k in self._keys}
        for k in self._keys:
            os.environ.pop(k, None)
    def __exit__(self, *args):
        for k, v in self.prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

def _ensure_dirs():
    HF_HOME.mkdir(parents=True, exist_ok=True)
    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _set_offline_env():
    for k, v in OFFLINE_ENV.items():
        os.environ[k] = v

def _all_needed_shards_present(model_dir: Path) -> bool:
    idx = model_dir / "model.safetensors.index.json"
    if not model_dir.exists() or not idx.exists() or not (model_dir / "config.json").exists():
        return False
    try:
        data = json.loads(idx.read_text())
        needed = set(data.get("weight_map", {}).values())
        present = {p.name for p in model_dir.glob("model-*-of-*.safetensors")}
        return needed.issubset(present)
    except Exception:
        return False

def _local_model_ready() -> bool:
    return _all_needed_shards_present(LOCAL_MODEL_DIR)

def _download_until_complete():
    _ensure_dirs()
    if _local_model_ready():
        return
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    backoff = 2.0
    attempt = 0
    while True:
        attempt += 1
        try:
            with _online_env():
                snapshot_download(
                    repo_id=MODEL_REPO,
                    local_dir=str(LOCAL_MODEL_DIR),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    allow_patterns=ALLOW_PATTERNS,
                    max_workers=8,
                )
            if _all_needed_shards_present(LOCAL_MODEL_DIR):
                READY_FLAG.write_text("ok")
                _set_offline_env()
                print(f"✅ Qwen fully downloaded to {LOCAL_MODEL_DIR}")
                if EXIT_AFTER_DOWNLOAD:
                    raise SystemExit("Qwen download finished. Please relaunch the app.")
                return
            else:
                raise RuntimeError("download incomplete: some shards still missing")
        except SystemExit:
            raise
        except Exception as e:
            print(f"⚠️ attempt {attempt} failed: {e}; retrying in {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def _clip_sentences(t: str, n: int) -> str:
    import re
    s = (t or "").strip()
    if not s:
        return s
    parts = re.split(r'(?<=[\.\!\?…。！？])\s+', s)
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:n])


# ────────────────────────────────────────────────────────────────────────────────
# Qwen2-VL wrapper
# ────────────────────────────────────────────────────────────────────────────────
class QwenVLWrapper(BaseModelWrapper):
    """
    Unified wrapper used by the orchestrator through .run(...).
    """

    _printed = False

    def __init__(self, model_name: Optional[str] = None):
        repo = (model_name or MODEL_REPO).strip()
        if repo != MODEL_REPO:
            global LOCAL_MODEL_DIR, READY_FLAG
            LOCAL_MODEL_DIR = LOCAL_MODELS_DIR / repo.split("/")[-1]
            READY_FLAG = LOCAL_MODEL_DIR / ".ready"

        _download_until_complete()

        # Select device
        if (not FORCE_CPU) and torch.cuda.is_available():
            self.device = torch.device("cuda"); dtype = torch.float16
        elif (not FORCE_CPU) and torch.backends.mps.is_available() and ALLOW_MPS:
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
            self.device = torch.device("mps"); dtype = torch.float16
        else:
            self.device = torch.device("cpu"); dtype = torch.float32

        # Load local model + processor
        self.processor = AutoProcessor.from_pretrained(
            str(LOCAL_MODEL_DIR),
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,  # stability over speed
        )
        self.model = _AutoVLModel.from_pretrained(
            str(LOCAL_MODEL_DIR),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
        )
        self.model.to(self.device).eval()

    def ensure_ready(self):
        """Print a one-time 'ready' message; safe to call multiple times."""
        if not QwenVLWrapper._printed:
            print(f"✅ Qwen2-VL Model loaded from {LOCAL_MODEL_DIR} on {self.device.type.upper()}")
            QwenVLWrapper._printed = True

    # Required by the framework base class
    def supports_images(self) -> bool:
        return True

    # Orchestrator calls this
    def run(self, sample, task_type: str, image_path: str) -> str:
        """
        Unified entry point for all tasks.
        """
        # self.ensure_ready()

        # prompts
        if task_type == "vqa":
            q = getattr(sample, "question", None) or "Answer the question about this image as briefly as possible."
            prompt = (
                "You are a visual question answering assistant. "
                "Answer briefly and clearly based on the image. "
                f"Question: {q}"
            )
        elif task_type == "captioning":
            prompt = (
                "You are an image caption assistant. "
                "Describe this image using 2 sentences. "
                "Mention objects, relationships, background. Now describe the image:"
            )
        elif task_type == "contextual_relevance":
            p = getattr(sample, "prompt", None) or getattr(sample, "question", None)
            prompt = p or "Provide a brief, relevant description of the image content."
        elif task_type == "hallucination":
            prompt = (
                "Write a factual, neutral caption for this image in 1–2 sentences. "
                "Avoid guessing unseen details; name only what is clearly visible."
            )
        else:
            prompt = "Describe the image briefly and factually in one sentence."

        text = self.generate(
            prompt=prompt,
            image_path=image_path,
            max_tokens=128,
            do_sample=False,
            repetition_penalty=1.1,
        )
        text = (text or "").strip()

        if task_type == "vqa":
            text = _clip_sentences(text, 1)
        elif task_type == "captioning":
            text = _clip_sentences(text, 2)
        elif task_type == "contextual_relevance":
            text = _clip_sentences(text, 2)

        return text

    # Low-level generation used by .run(...)
    def generate(self, prompt: str, image_path: Optional[str] = None, **kwargs) -> str:
        if not image_path or not os.path.exists(image_path):
            return "[ERROR] Image path not found or file does not exist."

        # Open image
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((768, 768))
        except Exception as e:
            return f"[ERROR] Failed to open image: {e}"

        if not prompt:
            prompt = (
                "Describe exactly what is in this image in 1–2 concise sentences. "
                "Mention key objects, actions, colors, counts, relations, and any legible text."
            )

        do_sample = bool(kwargs.get("do_sample", False))
        gen_kwargs = {
            "max_new_tokens": int(kwargs.get("max_tokens", 80)),
            "repetition_penalty": float(kwargs.get("repetition_penalty", 1.1)),
        }
        if do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": float(kwargs.get("temperature", 0.7)),
                "top_p": float(kwargs.get("top_p", 0.9)),
            })

        try:
            # Qwen chat-style multimodal input
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
            templated = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=[templated],
                images=[image],
                return_tensors="pt",
            ).to(self.device)

            prompt_len = inputs["input_ids"].shape[-1]

            use_fp16 = (self.device.type in ("cuda", "mps"))
            ctx = torch.autocast(device_type=self.device.type, dtype=torch.float16) if use_fp16 else _nullcontext()

            try:
                with torch.no_grad():
                    with ctx:
                        output_ids = self.model.generate(**inputs, **gen_kwargs)
            except Exception as e:
                # Fallback MPS → CPU if needed
                if self.device.type == "mps" and ("NDArray > 2**32" in str(e) or "MPSNDArray" in str(e)):
                    self.model.to("cpu")
                    self.device = torch.device("cpu")
                    inputs = inputs.to("cpu")
                    with torch.no_grad():
                        output_ids = self.model.generate(**inputs, **gen_kwargs)
                else:
                    raise

            if output_ids.shape[-1] > prompt_len:
                gen_only = output_ids[:, prompt_len:]
            else:
                gen_only = output_ids

            out = self.processor.batch_decode(gen_only, skip_special_tokens=True)[0]
            out = (out or "").strip()

            if out.lower().startswith(("system", "user", "assistant")):
                import re
                out = re.sub(r"^(system|user|assistant)\s*:\s*", "", out, flags=re.IGNORECASE).strip()

            return out

        except Exception as e:
            return f"[ERROR] Failed to generate caption: {e}"

