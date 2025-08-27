from abc import ABC, abstractmethod
from typing import Optional

class BaseModelWrapper(ABC):
    """
    Base interface for all LLM and multimodal model wrappers.
    """

    @abstractmethod
    def generate(self, prompt: str, image_path: Optional[str] = None, **kwargs) -> str:
        """
        Generates a model response based on a text prompt and (optionally) an image.
        """
        pass

    def supports_images(self) -> bool:
        """
        Returns True if the model supports image input.
        """
        return False
