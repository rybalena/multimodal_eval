# Architecture Guide

## Overview
The evaluation framework follows a **layered architecture**, connecting datasets, models, evaluation metrics, and reporting. This document explains how these layers interact and where the lifecycle of a `Sample` flows through the system.

---

## 1. Core Entities

### `Sample`
Defined in **`evaluation/schema.py`**, `Sample` is the universal data model.  
It supports all tasks: captioning, VQA, hallucination, contextual relevance.

**Key fields:**
- `id`: unique sample ID  
- `task`: task type (`captioning`, `vqa`, `hallucination`, `contextual_relevance`)  
- `image`: image URL/path  
- `question`, `expected_answer` (VQA)  
- `caption` (captioning / hallucination)  
- `prompt`, `expected_behavior` (contextual relevance)  
- `generated_answer`, `model_output`, `judge_notes` (filled after model + evaluation)

This object flows across **all layers** of the framework.

---

## 2. Model Layer

### Registry
Models are defined in **`model_wrappers/model_registry.py`**:

```python
MODEL_REGISTRY = {
    "qwen": QwenVLWrapper,
    "llava": LlavaWrapper,
    "gpt-4o": OpenAIWrapper,
}
```

Accessed via:

```python
from multimodal_eval.model_wrappers.model_registry import get_model_wrapper

model = get_model_wrapper("qwen")
```

Each wrapper implements `.generate()` or `.run()` to produce outputs given an image and task type.

---

## 3. Evaluation Layer

### Metrics
Metrics are centralized in **`evaluation/metrics_registry.py`**:  

- `clip_score` – visual grounding  
- `semantic_similarity` – semantic overlap  
- `cider` – lexical similarity  
- `contextual_relevance` – prompt/image consistency  
- `hallucination_metric` – hallucination detection  

Each metric is a function taking a `Sample` and returning a score.  
`metrics_registry` also provides `TASK_METRICS` → mapping from task type to applicable metrics.

### Evaluator
Defined in **`evaluation/evaluator.py`**.  

- Generates missing outputs if needed (`maybe_generate_output`)  
- Iterates over samples  
- Applies metrics  
- Returns `summary` (averages) + optional `per_sample` scores  

---

## 4. Orchestration Layer

The **`orchestrator.py`** module coordinates the entire evaluation run:

1. **Load datasets** → via `load_sources_for()` (standard + user datasets)  
2. **Generate outputs** → calls `model.run()`  
3. **Evaluate metrics** → calls `Evaluator` for labeled datasets  
4. **Aggregate results** → computes per-sample + composite scores  
5. **Judge unlabeled outputs** → sends to LLM judge if ground truth missing  
6. **Save results** → JSON, CSV, HTML via `saver`  

This is the main entry for **CLI** execution (`run_task()`).

---

## 5. Composite Scores

Composite scores are computed in **`evaluation/composite_score.py`**.  
They aggregate multiple metrics into a single number, task-specific.  
Example for captioning:  
```
0.25 * clip_score + 0.50 * semantic_similarity + 0.25 * cider
```

---

## 6. Lifecycle of a `Sample`

1. **Creation** → Loaded from JSON (dataset).  
2. **Model Inference** → `model.run(sample, task_type, image_path)` populates `generated_answer`.  
3. **Evaluation** → `Evaluator.evaluate([sample])` applies metrics.  
4. **Composite Score** → added via `compute_composite`.  
5. **Results Saving** → stored in results folder (JSON/CSV).  

---

## 7. Interaction Between Layers

- **Datasets** provide structured input (`Sample`).  
- **Models** consume `Sample` and generate predictions.  
- **Metrics** assess predictions against ground truth.  
- **Evaluator** orchestrates metrics per task.  
- **Orchestrator** drives the whole pipeline (load → generate → evaluate → save).  

---

## 8. Diagram (Conceptual)

```
Dataset → Sample → Model Registry → Model Wrapper
          ↓
       Generated Answer
          ↓
     Evaluator → Metrics Registry → Metric Functions
          ↓
     Composite Score
          ↓
       Saver / Reports
```
