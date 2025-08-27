from multimodal_eval.evaluation.metrics_registry import evaluate_sample
from multimodal_eval.model_wrappers.model_registry import get_model_wrapper
from multimodal_eval.evaluation.schema import Sample




def test_model_generates_caption(sample_hallucination):
    model = get_model_wrapper("llava")
    output = model.run(sample=sample_hallucination, task_type="hallucination", image_path=sample_hallucination.image)
    assert isinstance(output, str)
    assert len(output) > 5
    sample_hallucination.model_output = output


def test_hallucination_metrics_exist(sample_hallucination):
    results = evaluate_sample(sample_hallucination, task_type="hallucination")
    assert sample_hallucination.id in results
    scores = results[sample_hallucination.id]

    for metric in ["hallucination_score", "semantic_similarity", "composite_score"]:
        assert metric in scores
        assert isinstance(scores[metric], float)
        assert 0.0 <= scores[metric] <= 1.0


def test_hallucination_detects_extra_objects(sample_image_path):
    sample = Sample(
        id="hallucination_test_02",
        image=sample_image_path,
        caption="A dog playing with a ball in the pool",
        model_output="A dog, a cat, and a duck are playing in a swimming pool",
        objects=["dog", "pool"],
        task="hallucination"
    )
    results = evaluate_sample(sample, task_type="hallucination")
    halluc_score = results[sample.id]["hallucination_score"]
    assert halluc_score < 1.0


def test_hallucination_no_objects_list(sample_image_path):
    sample = Sample(
        id="hallucination_test_03",
        image=sample_image_path,
        caption="A dog in a pool",
        model_output="A dog in a pool",
        objects=[],
        task="hallucination"
    )
    results = evaluate_sample(sample, task_type="hallucination")
    halluc_score = results[sample.id]["hallucination_score"]
    assert isinstance(halluc_score, float)



def test_hallucination_empty_output(sample_image_path):
    sample = Sample(
        id="hallucination_test_04",
        image=sample_image_path,
        caption="A dog in a pool",
        model_output="",
        objects=["dog", "pool"],
        task="hallucination",
    )
    sample.generated_answer = ""

    results = evaluate_sample(sample, task_type="hallucination")
    score = results[sample.id]["hallucination_score"]

    assert isinstance(score, float)
    assert score >= 0.9
