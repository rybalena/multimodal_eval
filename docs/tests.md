# Testing Guide 

This document explains how to run and extend tests in **MultimodalEval**.

---

## ✅ Requirements

- Python ≥ 3.9  
- Install dev dependencies:  
  ```bash
  pip install -r requirements-dev.txt
  ```
  or if using Poetry:
  ```bash
  poetry install --with dev
  ```

---

## ▶️ Running Tests

We use **pytest** as the testing framework.

Run all tests:
```bash
pytest
```

Run a specific test file:
```bash
pytest tests/test_metrics.py
```

Run a single test case:
```bash
pytest tests/test_metrics.py::test_clip_score
```

Show detailed output:
```bash
pytest -v
```

Generate a coverage report:
```bash
pytest --cov=MultimodalEval
```

---

## 📂 Test Structure

```
tests/
 ├─ test_metrics.py          # Unit tests for evaluation metrics
 ├─ test_tasks.py            # Validation of task schemas and configs
 ├─ test_datasets.py         # Dataset loading and schema compliance
 ├─ test_orchestrator.py     # CLI orchestration and pipeline flow
 ├─ test_model_wrappers.py   # Mocked tests for model integration (Qwen, LLaVA, GPT-4o)
 └─ conftest.py              # Shared pytest fixtures
```

---

## 🧩 Writing New Tests

1. Place the new test file in `tests/` with prefix `test_`.  
2. Use **pytest** functions starting with `test_`. Example:

```python
# tests/test_example.py

from multimodal_eval.metrics import clip_score


def test_clip_score_valid_input():
    score = clip_score("a cat sitting on a mat", "an image of a cat on a mat")
    assert 0.0 <= score <= 1.0
```

3. For new tasks or metrics:  
   - Add fixture data under `tests/resources/`  
   - Test edge cases (empty input, invalid schema, extreme values)

---

## 🔍 Linting and Type Checking

We recommend running these checks before submitting code:

```bash
black .          # auto-formatting
flake8 .         # style linting
mypy MultimodalEval/   # type checking
```

---

## 📝 Best Practices

- Keep tests **isolated** (no external API calls unless mocked).  
- Cover **both success and failure** cases.  
- Use **fixtures** in `conftest.py` for reusable data.  
- Aim for **high coverage**, but prioritize meaningful tests over numbers.  

---

## 🚀 Continuous Integration (Optional)

In GitHub Actions, add this step:

```yaml
- name: Run tests
  run: pytest --cov=MultimodalEval
```

This ensures all PRs are tested automatically.
