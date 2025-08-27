from multimodal_eval.evaluation.clip_score import compute_clip_score
from multimodal_eval.evaluation.cider import compute_cider
from multimodal_eval.evaluation.semantic_similarity import compute_semantic_similarity
from multimodal_eval.evaluation.schema import Sample


def test_clip_score_range(sample_captioning):
    score = compute_clip_score(sample_captioning)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_semantic_similarity_range(sample_captioning):
    score = compute_semantic_similarity(sample_captioning)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_cider_score_positive(sample_captioning):
    score = compute_cider(sample_captioning)
    assert isinstance(score, float)
    assert score >= 0.0


def test_clip_score_empty_input(sample_image_path):
    empty_sample = Sample(
        id="empty_clip",
        image=sample_image_path,
        caption="",
        model_output="",
        task="captioning"
    )
    score = compute_clip_score(empty_sample)
    assert score == 0.0


def test_semantic_similarity_none_fields(sample_image_path):
    sample = Sample(
        id="sem_none",
        image=sample_image_path,
        caption=None,
        model_output=None,
        task="captioning"
    )
    score = compute_semantic_similarity(sample)
    assert score == 0.0


def test_cider_missing_caption_or_output(sample_image_path):
    sample = Sample(
        id="bad_cider",
        image=sample_image_path,
        caption=None,
        model_output="Some text",
        task="captioning"
    )
    score = compute_cider(sample)
    assert score == 0.0
