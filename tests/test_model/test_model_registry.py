import pytest
from multimodal_eval.model_wrappers.model_registry import get_model_wrapper
from multimodal_eval.model_wrappers.base_wrapper import BaseModelWrapper
from multimodal_eval.model_wrappers.llava_wrapper import LlavaWrapper


def test_get_model_wrapper_returns_base_instance():
    """Ensure the wrapper returned is a subclass of BaseModelWrapper."""
    model = get_model_wrapper("llava")
    assert isinstance(model, BaseModelWrapper), "LlavaWrapper must inherit from BaseModelWrapper"


def test_get_model_wrapper_returns_llava_wrapper():
    """Ensure get_model_wrapper('llava') returns an instance of LlavaWrapper."""
    model = get_model_wrapper("llava")
    assert isinstance(model, LlavaWrapper), "Expected LlavaWrapper instance"


def test_get_model_wrapper_invalid_name_raises():
    """Ensure that providing an unknown model name raises a ValueError."""
    with pytest.raises(ValueError, match=r"Model 'unknown_model' is not registered"):
        get_model_wrapper("unknown_model")


def test_model_wrapper_has_run_method():
    """Ensure that the model wrapper implements 'run' method."""
    model = get_model_wrapper("llava")
    assert hasattr(model, "run") and callable(model.run), "'run' method not found or not callable"


def test_model_wrapper_default_task_is_captioning():
    """Ensure default task is set to 'captioning'."""
    model = get_model_wrapper("llava")
    assert model.default_task == "captioning", "Default task should be 'captioning'"

