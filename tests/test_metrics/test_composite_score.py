from multimodal_eval.evaluation.composite_score import compute_composite, normalize_cider

def test_compute_captioning_composite():
    metrics = {
        "clip_score": 0.8,
        "semantic_similarity": 0.9,
        "cider": 0.6
    }
    score = compute_composite(metrics, task_type="captioning")
    assert round(score, 4) == round(0.25 * 0.8 + 0.5 * 0.9 + 0.25 * 0.6, 4)



def test_compute_vqa_composite():
    metrics = {"clip_score": 0.5, "semantic_similarity": 0.6, "contextual_relevance": 0.7}
    score = compute_composite(metrics, task_type="vqa")

    #  0.10*clip + 0.40*semantic + 0.50*contextual
    expected = (
        0.10 * metrics["clip_score"]
        + 0.40 * metrics["semantic_similarity"]
        + 0.50 * metrics["contextual_relevance"]
    )

    assert round(score, 6) == round(expected, 6)

def test_compute_contextual_composite():
    metrics = {
        "clip_score": 0.7,
        "semantic_similarity": 0.6,
        "cider": 0.8
    }
    score = compute_composite(metrics, task_type="contextual_relevance")
    expected = round(0.4 * 0.7 + 0.3 * normalize_cider(0.8) + 0.3 * 0.6, 4)
    assert round(score, 4) == expected


def test_normalize_cider_values():
    assert normalize_cider(0.5) == 0.5
    assert normalize_cider(1.2) == 1.0
    assert normalize_cider(-0.1) == 0.0
    assert normalize_cider("bad") == 0.0


def test_compute_composite_handles_missing_metrics():
    metrics = {}
    score = compute_composite(metrics, task_type="captioning")
    assert score == 0.0

