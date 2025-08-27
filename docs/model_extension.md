# Model Extension Guide

## Overview
The framework supports multiple multimodal models (e.g., Qwen, LLaVA, GPT-4o).  
To add a new model, you must:  
1. Implement a wrapper that extends the **BaseModelWrapper**.  
2. Register the new model inside **`model_registry.py`**.  
3. Ensure its parameters can be passed through CLI and UI.

---

## 1. Base Wrapper

All models inherit from **`BaseModelWrapper`**:

```python
from abc import ABC, abstractmethod
from typing import Optional

class BaseModelWrapper(ABC):
    """Base interface for all LLM and multimodal model wrappers."""

    @abstractmethod
    def generate(self, prompt: str, image_path: Optional[str] = None, **kwargs) -> str:
        """Generates a model response from a prompt (optionally with an image)."""
        pass

    def supports_images(self) -> bool:
        """Return True if the model supports image input."""
        return False
```

Each model wrapper must implement **`.generate()`**.  
Optionally, override **`.supports_images()`** if the model accepts images.

---

## 2. Model Registry

All models are registered in **`model_registry.py`**:

```python
from .qwen_wrapper import QwenVLWrapper
from .llava_wrapper import LlavaWrapper
from .openai_wrapper import OpenAIWrapper

MODEL_REGISTRY = {
    "qwen": QwenVLWrapper,
    "llava": LlavaWrapper,
    "gpt-4o": OpenAIWrapper,
}
```

To fetch a model instance:

```python
from multimodal_eval.model_wrappers.model_registry import get_model_wrapper

model = get_model_wrapper("qwen", temperature=0.7)
```

If the model is not registered, an error will be raised.

---

## 3. Adding a New Model

### Step 1 — Create a wrapper
Example: `blip2_wrapper.py`

```python
from multimodal_eval.model_wrappers.base_wrapper import BaseModelWrapper


class Blip2Wrapper(BaseModelWrapper):
    def __init__(self, model_path: str, **kwargs):
        self.model = load_blip2(model_path, **kwargs)

    def generate(self, prompt: str, image_path: str = None, **kwargs) -> str:
        if image_path:
            return self.model.generate_with_image(prompt, image_path)
        return self.model.generate_text(prompt)

    def supports_images(self) -> bool:
        return True
```

### Step 2 — Register the model
Edit **`model_registry.py`**:

```python
from .blip2_wrapper import Blip2Wrapper

MODEL_REGISTRY = {
    "qwen": QwenVLWrapper,
    "llava": LlavaWrapper,
    "gpt-4o": OpenAIWrapper,
    "blip2": Blip2Wrapper,   # ← new entry
}
```

### Step 3 — Expose in CLI/UI
- **CLI (`cli/main.py`)**: add `"blip2"` to available `--model` choices.  
- **UI (`app_ui/app.py`)**: add `"blip2"` to model selection dropdown.

---

## 4. Example Usage

```python
model = get_model_wrapper("blip2", model_path="path/to/weights")
output = model.generate("Describe the picture", image_path="dog.jpg")
print(output)
```

---

## 5. Summary

- Implement `.generate()` in a subclass of `BaseModelWrapper`.  
- Register the wrapper in `model_registry.py`.  
- Expose the new model to CLI/UI for usability.  
- Optional: override `.supports_images()` if the model supports multimodal input.  
