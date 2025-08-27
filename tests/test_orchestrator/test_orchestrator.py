import multimodal_eval.orchestrator as orch
from multimodal_eval.evaluation.schema import Sample


def test_model_name_is_non_empty():
    assert isinstance(orch.MODEL_NAME, str) and orch.MODEL_NAME.strip() != ""


def test_run_task_no_datasets_quiet_exit(monkeypatch, capsys, stub_model, stub_savers):
    monkeypatch.setattr(orch, "load_dataset", lambda *a, **k: [])

    orch.run_task(
        task="captioning",
        task_config={"labeled": "path/doesnt_matter.csv"},
        dataset_type="labeled",
        sources_mode="user",
    )
    out = capsys.readouterr().out
    assert "No datasets" in out or "Loaded 0 samples" in out

def _cap_sample(sample_id="1"):
    return Sample(
        id=sample_id,
        image="img.jpg",
        caption="a cat on a box",
        model_output=None,
        generated_answer=None,
    )

def _unlab_sample(sample_id="u1"):
    return Sample(
        id=sample_id,
        image="img.jpg",
        caption=None,
        model_output=None,
        generated_answer=None,
    )

def _hallu_sample(sample_id="h1"):
    return Sample(
        id=sample_id,
        image="img.jpg",
        caption="a zebra in the kitchen",
        model_output=None,
        generated_answer=None,
    )

def test_smoke_no_datasets(monkeypatch, stub_model, stub_resolve_image_path, stub_savers):

    monkeypatch.setattr(
        orch,
        "load_sources_for",
        lambda *a, **k: [("user_golden", [], "dummy.json")]
    )

    orch.run_task(task="captioning", task_config={}, dataset_type="labeled", sources_mode="user")


def test_smoke_golden_captioning(monkeypatch, stub_model, stub_resolve_image_path, stub_savers):

    s = Sample(
        id="1",
        image="img.jpg",
        caption="a cat on a box",
        model_output=None,
        generated_answer=None,
        task="captioning",
    )

    monkeypatch.setattr(
        orch,
        "load_sources_for",
        lambda *a, **k: [("user_golden", [s], "dummy.json")]
    )

    class _Eval:
        def __init__(self, metrics): self.metrics = metrics
        def evaluate(self, _samples, per_sample=False):
            per_sample = {m: [{"id": "1", "score": 0.8}] for m in self.metrics}
            return {"per_sample": per_sample}
    monkeypatch.setattr(orch, "Evaluator", _Eval)
    monkeypatch.setattr(orch, "compute_composite", lambda *_a, **_k: 0.7777)
    monkeypatch.setattr(orch, "print_scores_with_reason", lambda *a, **k: None)

    orch.run_task(task="captioning", task_config={}, dataset_type="labeled", sources_mode="user")


def test_smoke_unlabeled_skip_judge_on_empty(monkeypatch, stub_resolve_image_path, stub_savers):

    s = Sample(
        id="u42",
        image="img.jpg",
        caption=None,
        model_output=None,
        generated_answer=None,
        task="captioning",
    )

    monkeypatch.setattr(
        orch,
        "load_sources_for",
        lambda *a, **k: [("user_unlabeled", [s], "dummy.json")]
    )


    class _EmptyModel:
        def run(self, *a, **k): return ""
    monkeypatch.setattr(orch, "model", _EmptyModel())

    def _judge_should_not_be_called(*a, **k):
        raise AssertionError("call_llm_judge must NOT be called for empty generated_answer")
    monkeypatch.setattr(orch, "call_llm_judge", _judge_should_not_be_called)

    monkeypatch.setattr(orch, "print_scores_with_reason", lambda *a, **k: None)
    monkeypatch.setattr(orch, "build_caption_eval_prompt", lambda *a, **k: ("u", "s"))

    orch.run_task(task="captioning", task_config={}, dataset_type="unlabeled", sources_mode="user")


def test_sources_mode_required_signature(monkeypatch, stub_model, stub_savers):
    monkeypatch.setattr(orch, "load_dataset", lambda *a, **k: [])

    orch.run_task(
        task="captioning",
        task_config={"labeled": "nope.csv"},
        dataset_type="labeled",
        sources_mode="standard",
    )
