from multimodal_eval.evaluation.schema import Sample
from multimodal_eval.evaluation.hallucination_metric import compute_hallucination_metric


def test_hallucination_metric_valid(sample_hallucination):
    result = compute_hallucination_metric(sample_hallucination)
    assert "hallucination_score" in result
    assert "semantic_similarity" in result
    assert 0.0 <= result["hallucination_score"] <= 1.0
    assert 0.0 <= result["semantic_similarity"] <= 1.0


def test_hallucination_metric_extra_objects(sample_image_path):
    sample = Sample(
        id="hallucination_extra_obj",
        image=sample_image_path,
        caption="A dog is in a pool",
        model_output="A dog, a cat, and a flamingo are in a swimming pool",
        objects=["dog", "pool"],
        task="hallucination"
    )
    result = compute_hallucination_metric(sample)
    assert result["hallucination_score"] < 1.0


def test_hallucination_metric_no_objects(sample_image_path):
    sample = Sample(
        id="hallucination_no_obj",
        image=sample_image_path,
        caption="Some caption",
        model_output="Some output",
        objects=[],
        task="hallucination"
    )
    result = compute_hallucination_metric(sample)
    assert isinstance(result["hallucination_score"], float)



def test_hallucination_metric_empty_output(sample_image_path):
    sample = Sample(
        id="hallucination_empty_output",
        image=sample_image_path,
        caption="Dog in pool",
        model_output="",
        objects=["dog", "pool"],
        task="hallucination",
    )

    result = compute_hallucination_metric(sample)

    assert "hallucination_score" in result
    assert isinstance(result["hallucination_score"], float)
    assert result["hallucination_score"] >= 0.9


def test_hallucination_metric_none_fields(sample_image_path):
    sample = Sample(
        id="hallucination_none_fields",
        image=sample_image_path,
        caption=None,
        model_output=None,
        objects=None,
        task="hallucination",
    )
    result = compute_hallucination_metric(sample)

    assert "hallucination_score" in result
    assert isinstance(result["hallucination_score"], float)
    assert result["hallucination_score"] == 1.0
    assert result["hallucination_raw"] == 0.0
    assert result["semantic_similarity"] == 0.0
    assert result["composite_score"] == 0.5
