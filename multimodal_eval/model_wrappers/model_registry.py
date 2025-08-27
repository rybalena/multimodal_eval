from .qwen_wrapper import QwenVLWrapper
from .llava_wrapper import LlavaWrapper
from .openai_wrapper import OpenAIWrapper

MODEL_REGISTRY = {
    "qwen": QwenVLWrapper,
    "llava": LlavaWrapper,
    "gpt-4o": OpenAIWrapper,
}

def get_model_wrapper(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is not registered. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
