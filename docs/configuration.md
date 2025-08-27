# Configuration Guide 

This document explains how to configure tasks, datasets, model outputs, and metrics.

---

## 📂 Location
Task configuration lives in:
```
multimodal_eval/configs/tasks_config.py
```

---

## 📖 Terminology

- **Labeled** — datasets with ground truth annotations (reference answers).  
- **Unlabeled** — datasets without ground truth, used for inference and LLM-as-a-Judge.

---

## 🔧 Task Configuration

Each task is defined under the `CONFIG` dictionary.  
Example (excerpt for **captioning**):

```python
"captioning": {
    "labeled": "data_sets/labeled_data/captioning/captioning_set.json",
    "unlabeled": "data_sets/unlabeled_data/captioning/captioning_set.json",

    "model_output_labeled": "model_outputs/labeled_data/captioning/labeled_captioning_predictions.json",
    "model_output_unlabeled": "model_outputs/unlabeled_data/captioning/captioning_predictions.json",

    "metrics": [
        "cider",
        "clip_score",
        "semantic_similarity"
    ]
}
```

---

## 🗂 Key Fields

- **`labeled`** → path to **Labeled** dataset JSON  
- **`unlabeled`** → path to unlabeled dataset JSON  
- **`model_output_labeled` / `model_output_unlabeled`** → where predictions are saved  
- **`metrics`** → list of metrics used for this task  

---

## ✅ Supported Tasks

- **Captioning**  
  - Labeled + Unlabeled  
  - Metrics: `cider`, `clip_score`, `semantic_similarity`  

- **Hallucination**  
  - Labeled + Unlabeled  
  - Metrics: `hallucination_metric`  

- **VQA**  
  - Labeled only  
  - Metrics: `clip_score`, `semantic_similarity`, `contextual_relevance`  

- **Contextual Relevance**  
  - Labeled + Unlabeled  
  - Metrics: `clip_score`, `cider`, `semantic_similarity`  

---

## 🔑 Environment Variables

Some models require keys:
```
OPENAI_API_KEY=sk-xxxx
```
Keep them in `.env` at project root.

---

## 🛠 How to Modify

- **Add a new metric** → extend `metrics` list.  
- **Change dataset** → update `labeled` / `unlabeled` path.  
- **Redirect outputs** → update `model_output_*` path.  
- **Add new task** → add a new entry in `CONFIG`.  

---
