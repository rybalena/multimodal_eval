# tests/conftest.py
import pytest
from pathlib import Path
from typing import List
import multimodal_eval.orchestrator as orch
from multimodal_eval.evaluation.schema import Sample


@pytest.fixture(scope="session")
def sample_image_path() -> str:
    return str(Path("tests/resources/woman in a wedding dress.jpg"))

@pytest.fixture
def sample_caption() -> str:
    return "A boy in a red jersey is playing football on a green field."

@pytest.fixture
def sample_model_output() -> str:
    return "A young boy is kicking a soccer ball on the field."

@pytest.fixture
def sample_question() -> str:
    return "What sport is the boy playing?"

@pytest.fixture
def sample_expected_answer() -> str:
    return "Football"

@pytest.fixture
def sample_objects() -> List[str]:
    return ["boy", "field", "ball"]

@pytest.fixture
def sample_captioning(sample_image_path, sample_caption, sample_model_output) -> Sample:
    return Sample(
        id="captioning_sample",
        image=sample_image_path,
        caption=sample_caption,
        model_output=sample_model_output,
        task="captioning"
    )

@pytest.fixture
def sample_hallucination(sample_image_path, sample_caption, sample_objects) -> Sample:
    return Sample(
        id="hallucination_sample",
        image=sample_image_path,
        caption=sample_caption,
        model_output="A cat, flamingo, and boy playing football",
        objects=sample_objects,
        task="hallucination"
    )

@pytest.fixture
def sample_vqa(sample_image_path, sample_question, sample_expected_answer, sample_model_output) -> Sample:
    return Sample(
        id="vqa_sample",
        image=sample_image_path,
        question=sample_question,
        expected_answer=sample_expected_answer,
        model_output=sample_model_output,
        task="vqa"
    )

@pytest.fixture
def sample_generic(sample_image_path, sample_caption, sample_model_output, sample_question, sample_expected_answer) -> Sample:
    return Sample(
        id="generic_sample",
        image=sample_image_path,
        caption=sample_caption,
        model_output=sample_model_output,
        question=sample_question,
        expected_answer=sample_expected_answer,
        objects=["boy", "field", "ball"],
        task=None
    )

@pytest.fixture
def sample_contextual(sample_image_path, sample_caption, sample_model_output, sample_question, sample_expected_answer) -> Sample:
    return Sample(
        id="contextual_sample",
        image=sample_image_path,
        caption=sample_caption,
        model_output=sample_model_output,
        question=sample_question,
        expected_answer=sample_expected_answer,
        objects=["boy", "field", "ball"],
        task="contextual_relevance"
    )


class StubModelWrapper:

    def run(self, sample: Sample, task_type: str, image_path: str):
        if task_type == "vqa":
            return "stub answer"
        return "stub caption"

@pytest.fixture
def stub_model(monkeypatch):
    monkeypatch.setattr(orch, "model", StubModelWrapper())
    return orch.model

@pytest.fixture
def stub_resolve_image_path(monkeypatch):
    monkeypatch.setattr(orch, "resolve_image_path", lambda p: "/tmp/fake.jpg")

@pytest.fixture
def stub_savers(monkeypatch):
    monkeypatch.setattr(orch, "save_aggregated_metrics", lambda *a, **k: None)
    monkeypatch.setattr(orch, "save_results_by_type", lambda *a, **k: None)

def _stub_evaluator_per_sample(metric_names, sample_id="1", score=0.9):
    per_sample = {}
    for m in metric_names:
        per_sample[m] = [{"id": sample_id, "score": score}]
    return {"per_sample": per_sample}

@pytest.fixture
def stub_evaluator(monkeypatch):
    class _FakeEvaluator:
        def __init__(self, metrics): self.metrics = metrics
        def evaluate(self, _samples, per_sample=False):
            return _stub_evaluator_per_sample(self.metrics, sample_id="1", score=0.8)
    monkeypatch.setattr(orch, "Evaluator", _FakeEvaluator)

@pytest.fixture
def stub_composite(monkeypatch):
    monkeypatch.setattr(orch, "compute_composite", lambda *_a, **_k: 0.7777)
