# Task Addition Guide

## Overview
The framework supports multiple evaluation tasks (captioning, hallucination, VQA, contextual relevance).  
Adding a new task requires changes in **configs**, **CLI**, and **UI**.

---

## 1. Define the Task in Config

Tasks are described in **`configs/tasks_config.py`**.  
Each entry defines:
- paths to datasets (`labeled`, `unlabeled`)  
- paths for model outputs  
- list of metrics  

Example (`captioning`):
```python
CONFIG = {
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
    },
    ...
}
```

👉 To add a new task (e.g., `image_retrieval`):
```python
"image_retrieval": {
    "labeled": "data_sets/labeled_data/image_retrieval/set.json",
    "unlabeled": "data_sets/unlabeled_data/image_retrieval/set.json",

    "model_output_golden": "model_outputs/labeled_data/image_retrieval/predictions.json",
    "model_output_unlabeled": "model_outputs/unlabeled_data/image_retrieval/predictions.json",

    "metrics": ["clip_score", "semantic_similarity"]
}
```

---

## 2. Expose the Task in CLI

Tasks are parsed in **`cli/main.py`**.  

```python
parser.add_argument(
    "--task",
    choices=["captioning", "hallucination", "vqa", "contextual_relevance", "all"],
    help="Evaluation task to run"
)
```

👉 Add the new task to `choices`:
```python
choices=["captioning", "hallucination", "vqa", "contextual_relevance", "image_retrieval", "all"]
```

Now the new task can be launched via:
```bash
python -m multimodal_eval.cli.main --task image_retrieval --dataset_type labeled
```

---

## 3. Expose the Task in UI

In **`app_ui/streamlit_ui/streamlit_app.py`**, tasks are usually defined inside a `selectbox` or task dropdown.  
Example:
```python
task = st.selectbox("Select task", ["captioning", "hallucination", "vqa", "contextual_relevance"])
```

👉 Add your new task:
```python
task = st.selectbox("Select task", ["captioning", "hallucination", "vqa", "contextual_relevance", "image_retrieval"])
```

---

## 4. Verify Metric Mapping

Ensure that `metrics_registry.py` supports the metrics listed for your task.  
If you introduce a new metric, follow the guide in **extending-metrics.md**.

---

## 5. Test the Workflow

1. Add samples for the new task in `data_sets/labeled_data/...` and/or `data_sets/unlabeled_data/...`.  
2. Run from CLI:
   ```bash
   python -m multimodal_eval.cli.main --task image_retrieval --dataset_type labeled --sources standard
   ```
3. Open UI and confirm task availability:
   ```bash
   streamlit run app_ui/streamlit_ui/streamlit_app.py
   ```

---

✅ That’s it — your new task is now integrated into **configs**, **CLI**, and **UI**.
