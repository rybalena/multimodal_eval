import pytest
from multimodal_eval.evaluation.evaluator import Evaluator
from multimodal_eval.evaluation.schema import Sample


def test_evaluator_per_sample_metrics(sample_captioning):
    evaluator = Evaluator(metrics=["clip_score", "semantic_similarity", "cider"])
    result = evaluator.evaluate([sample_captioning], per_sample=True)

    assert "summary" in result
    assert "per_sample" in result
    for metric in ["clip_score", "semantic_similarity", "cider"]:
        assert metric in result["summary"]
        assert isinstance(result["summary"][metric], float)
        assert isinstance(result["per_sample"][metric][0]["score"], float)


def test_evaluator_empty_input():
    evaluator = Evaluator(metrics=["clip_score"])
    with pytest.raises(ValueError):
        evaluator.evaluate([])


def test_evaluator_skips_missing_output(sample_image_path):
    sample = Sample(
        id="missing_output",
        image=sample_image_path,
        caption="A cat playing piano",
        model_output=None,
        task="captioning"
    )
    evaluator = Evaluator(metrics=["clip_score"])
    result = evaluator.evaluate([sample], per_sample=True)

    assert result["summary"]["clip_score"] in [0.0, None]
