from multimodal_eval.evaluation.metrics_registry import evaluate_sample
from multimodal_eval.model_wrappers.model_registry import get_model_wrapper
from multimodal_eval.evaluation.schema import Sample


def test_model_generates_answer(sample_vqa):
    model = get_model_wrapper("llava")
    answer = model.run(sample=sample_vqa, task_type="vqa", image_path=sample_vqa.image)
    assert isinstance(answer, str)
    assert len(answer) > 5
    sample_vqa.model_output = answer


def test_vqa_metrics_exist(sample_vqa):
    results = evaluate_sample(sample_vqa, task_type="vqa")
    assert sample_vqa.id in results
    scores = results[sample_vqa.id]

    for metric in ["clip_score", "semantic_similarity", "contextual_relevance", "composite_score"]:
        assert metric in scores
        assert isinstance(scores[metric], float)


def test_vqa_clip_score_range(sample_vqa):
    results = evaluate_sample(sample_vqa, task_type="vqa")
    clip_score = results[sample_vqa.id].get("clip_score", 0.0)
    assert 0.0 <= clip_score <= 1.0


def test_vqa_semantic_similarity(sample_vqa):
    results = evaluate_sample(sample_vqa, task_type="vqa")
    sim_score = results[sample_vqa.id].get("semantic_similarity", 0.0)
    assert isinstance(sim_score, float)
    assert sim_score >= 0.0


def test_vqa_empty_model_output(sample_image_path):
    sample = Sample(
        id="vqa_test_02",
        image=sample_image_path,
        question="What is he doing?",
        expected_answer="Riding",
        model_output="",
        task="vqa"
    )
    results = evaluate_sample(sample, task_type="vqa")
    composite = results[sample.id]["composite_score"]
    assert composite <= 0.1


def test_vqa_missing_question(sample_image_path):
    sample = Sample(
        id="vqa_test_03",
        image=sample_image_path,
        question=None,
        expected_answer="Riding",
        model_output="He is riding a bike",
        task="vqa"
    )
    results = evaluate_sample(sample, task_type="vqa")
    assert "contextual_relevance" in results[sample.id]
    assert isinstance(results[sample.id]["contextual_relevance"], float)
