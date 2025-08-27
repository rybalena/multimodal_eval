# Interpreting Results Guide

## Overview
This guide explains how to interpret the evaluation outputs produced by the framework.  
Each run generates three main artifact types:

1. **Model Outputs** – raw predictions for each sample.  
2. **Results** – per-sample metrics with metadata.  
3. **Summaries** – aggregated metrics across the dataset.  

All artifacts are saved automatically by the **orchestrator** and **saver** modules.

---

## 1. File Locations

Results are saved under the project root in the `results/` folder.

- **Labeled datasets:**  
  ```
  results/labeled_data/<task>/
  ```
- **Unlabeled datasets:**  
  ```
  results/unlabeled_data/<task>/
  ```
- **User datasets:**  
  ```
  results/user_dataset/<labeled_data|unlabeled_data>/<task>/
  ```

Each directory contains JSON, and CSV artifacts for results and summaries.

---

## 2. Model Outputs

**File:** `model_outputs/[labeled_data|unlabeled_data]/<task>/*.json`

- Contains only essential fields (`id`, `image`, `caption`, `question`, `expected_answer`, `model_output`).  
- Used to trace back raw predictions without additional metadata.  

**Example:**
```json
[
  {
    "id": "img_001",
    "image": "http://example.com/dog.png",
    "caption": "A dog running in the park.",
    "model_output": "A brown dog is running outside."
  }
]
```

---

## 3. Results Files

**Files:**  
- JSON → `<task>_results.json`  
- CSV → `<task>_results.csv`

These contain per-sample evaluation results with **model metadata**:

- `id`, `image`, `caption` / `question` / `expected_answer`, `model_output`  
- `model_name`, `run_timestamp`, `run_timestamp_utc`  
- Metric scores (e.g. `clip_score`, `semantic_similarity`, `cider`, `contextual_relevance`, `hallucination_score`)  
- `composite_score` (task-specific weighted score)

**Example (JSON entry):**
```json
{
  "id": "img_045",
  "image": "http://example.com/cat.png",
  "caption": "A cat sitting on a chair.",
  "model_output": "A cat sitting on a chair with a cup of coffee.",
  "model_name": "qwen",
  "run_timestamp": "2025-08-18T09:00:00",
  "clip_score": 0.72,
  "semantic_similarity": 0.81,
  "cider": 0.67,
  "composite_score": 0.75
}
```

---

## 4. Summary Files

**Files:**  
- JSON → `<task>_summary.json`  
- CSV → `<task>_summary.csv`  
- HTML → `<task>_summary.html`

Summaries aggregate metrics across the entire dataset:  

- Task name and dataset type  
- Model name  
- Run timestamps  
- Number of samples evaluated  
- Mean values of all numeric metrics  

**Example (JSON summary):**
```json
{
  "task": "captioning",
  "dataset_type": "labeled_data",
  "model_name": "qwen",
  "run_timestamp": "2025-08-18T09:00:00",
  "num_samples": 100,
  "mean_clip_score": 0.74,
  "mean_semantic_similarity": 0.82,
  "mean_cider": 0.69,
  "mean_composite_score": 0.77
}
```


---

## 5. Composite Score Calculation

The **`composite_score`** is a weighted aggregation of metrics:

- **Captioning:** `0.25 * clip_score + 0.50 * semantic_similarity + 0.25 * cider`  
- **VQA:** `0.10 * clip_score + 0.40 * semantic_similarity + 0.50 * contextual_relevance`  
- **Contextual Relevance:** `0.40 * clip_score + 0.30 * cider + 0.30 * semantic_similarity`  
- **Hallucination:** handled separately inside `hallucination_metric.py`  

This provides a single number summarizing model quality per task.

---

## 6. Using Results in Practice

- **Debugging models:** check **per-sample results** to identify weak points (e.g., low semantic similarity).  
- **Benchmarking:** compare **summary scores** between models.  
- **Reporting:** use **HTML summaries** for presentations.  
