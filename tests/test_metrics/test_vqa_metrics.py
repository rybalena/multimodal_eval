from multimodal_eval.evaluation.schema import Sample
from multimodal_eval.evaluation.clip_score import compute_clip_score
from multimodal_eval.evaluation.semantic_similarity import compute_semantic_similarity
from multimodal_eval.evaluation.contextual_relevance import compute_contextual_relevance


def test_clip_score_for_vqa(sample_vqa):
    score = compute_clip_score(sample_vqa)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_semantic_similarity_for_vqa(sample_vqa):
    score = compute_semantic_similarity(sample_vqa)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_contextual_relevance_for_vqa(sample_vqa):
    score = compute_contextual_relevance(sample_vqa)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_contextual_relevance_with_empty_question(sample_image_path):
    sample = Sample(
        id="vqa_empty_q",
        image=sample_image_path,
        question="",
        model_output="He is riding",
        task="vqa"
    )
    score = compute_contextual_relevance(sample)
    assert isinstance(score, float)
    assert score >= 0.0


def test_contextual_relevance_with_none_fields(sample_image_path):
    sample = Sample(
        id="vqa_none_fields",
        image=sample_image_path,
        question=None,
        model_output=None,
        task="vqa"
    )
    score = compute_contextual_relevance(sample)
    assert score == 0.0
