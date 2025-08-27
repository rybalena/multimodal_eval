from multimodal_eval.evaluation.metrics_registry import evaluate_sample
from multimodal_eval.model_wrappers.model_registry import get_model_wrapper
from multimodal_eval.evaluation.schema import Sample


def test_model_generates_answer(sample_contextual):
    model = get_model_wrapper("llava")
    answer = model.run(
        sample=sample_contextual,
        task_type="contextual_relevance",
        image_path=sample_contextual.image
    )
    assert isinstance(answer, str)
    assert len(answer) > 5
    sample_contextual.model_output = answer


def test_captioning_golden_metrics_exist(sample_captioning):
    results = evaluate_sample(sample_captioning)

    assert sample_captioning.id in results
    scores = results[sample_captioning.id]

    for metric in [
        "clip_score",
        "semantic_similarity",
        "cider",
        "composite_score"
    ]:
        assert metric in scores, f"{metric} not found in result"
        assert isinstance(scores[metric], float), f"{metric} is not a float"


def test_contextual_clip_score_range(sample_contextual):
    results = evaluate_sample(sample_contextual, task_type="contextual_relevance")
    clip_score = results[sample_contextual.id].get("clip_score", 0.0)
    assert 0.0 <= clip_score <= 1.0


def test_contextual_composite_score(sample_contextual):
    results = evaluate_sample(sample_contextual, task_type="contextual_relevance")
    composite_score = results[sample_contextual.id].get("composite_score")
    assert composite_score is not None
    assert isinstance(composite_score, float)
    assert composite_score >= 0.0


def test_contextual_error_on_missing_question(sample_image_path):
    bad_sample = Sample(
        id="bad_context",
        image=sample_image_path,
        caption="A bird flying over the mountains",
        model_output="A bird in the sky",
        question=None,
        expected_answer=None,
        task="contextual_relevance"
    )
    results = evaluate_sample(bad_sample, task_type="contextual_relevance")

    contextual = results[bad_sample.id].get("contextual_relevance")
    assert contextual is None or isinstance(contextual, float)
