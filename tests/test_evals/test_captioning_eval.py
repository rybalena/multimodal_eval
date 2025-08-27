import pytest
from multimodal_eval.evaluation.metrics_registry import evaluate_sample
from multimodal_eval.model_wrappers.model_registry import get_model_wrapper
from multimodal_eval.evaluation.schema import Sample

def test_model_generates_caption(sample_captioning):
    model = get_model_wrapper("llava")
    caption = model.run(sample_captioning)
    assert isinstance(caption, str)
    assert len(caption) > 5
    sample_captioning.model_output = caption

def test_captioning_metrics_exist(sample_captioning):
    results = evaluate_sample(sample_captioning)
    assert sample_captioning.id in results
    scores = results[sample_captioning.id]

    for metric in ["clip_score", "semantic_similarity", "cider", "composite_score"]:
        assert metric in scores
        assert isinstance(scores[metric], float)
        assert scores[metric] >= 0.0

def test_captioning_clip_score_range(sample_captioning):
    results = evaluate_sample(sample_captioning)
    clip_score = results[sample_captioning.id].get("clip_score", 0)
    assert 0.0 <= clip_score <= 1.0

def test_captioning_composite_calculation(sample_captioning):
    results = evaluate_sample(sample_captioning)
    composite_score = results[sample_captioning.id].get("composite_score")
    assert composite_score is not None
    assert isinstance(composite_score, float)
    assert composite_score >= 0.0

def test_captioning_missing_image_graceful():
    sample = Sample(
        id="bad_id",
        image="non_existent.jpg",
        caption="Caption",
        model_output="Generated caption",
        task="captioning"
    )

    result = evaluate_sample(sample)
    assert result is None or isinstance(result, dict)

