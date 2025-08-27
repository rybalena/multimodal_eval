# ⚡ Quickstart Guide — MultimodalEval

This Quickstart shows how to run evaluations using the **CLI** or **Streamlit UI**.  
In just a few steps, you will learn how to install models, launch tasks, add a dataset, and select which model to evaluate.

---

## 0. Install Models (Required)

Before running evaluations, you must install at least one model backend.

- **LLaVA via Ollama**  
  ```bash
  curl -fsSL https://ollama.com/download | sh
  ollama pull llava:7b
  ollama pull llava:14b
  ```

- **Qwen2-VL**  
  ```bash
  pip install transformers accelerate
  ```

- **OpenAI GPT-4o**  
  ```bash
  pip install openai
  echo "OPENAI_API_KEY=sk-..." > .env
  ```

⚠️ Do not close Ollama when running LLaVA or Qwen — the models must stay active.

---

## 1. Launching the UI

The framework provides a **Streamlit-based UI**.  
Start it with:

```bash
streamlit run app_ui/streamlit_ui/streamlit_app.py
```

This will open a browser window where you can select **task**, **model**, and **dataset**, and view results interactively.

---

## 2. Running from the CLI

The CLI entry point is **`multimodal_eval/cli/main.py`**.  

Check available options:

```bash
python -m multimodal_eval.cli.main --help
```

### Run a specific task
```bash
python -m multimodal_eval.cli.main --task captioning --dataset_type labeled
```

### Run all tasks
```bash
python -m multimodal_eval.cli.main --task all --dataset_type labeled
```

### Run with user datasets
```bash
python -m multimodal_eval.cli.main --sources user
```

---

## 3. Adding a Custom Dataset

You can add your own dataset under `user_dataset/`.  

Example run:
```bash
python -m multimodal_eval.cli.main --task captioning --dataset_type labeled --sources user
```

Or programmatically:
```python
from multimodal_eval.utils.add_dataset import add_dataset

add_dataset("my_dataset.json", dataset_type="labeled", task="captioning")
```

---

## 4. Selecting a Model

By default, the framework uses `gpt-4o`.  
To change the model, open `multimodal_eval/orchestrator.py` and edit:

```python
MODEL_NAME = "gpt-4o"  # ⬅️ ⬅️❗ Change this to "llava" or "qwen"
```

You can also choose the model directly in the Streamlit UI.

---

## 5. Example Dataset

We provide a **mini labeled dataset** in `datasets/labeled_data/captioning/captioning_set.json`:

```json
{
  "id": "kitten_box",
  "image": "kitten_in_a_box.jpg",
  "caption": "A small white kitten peeking out of a cardboard box."
}
```

Run quickly with:
```bash
python -m multimodal_eval.cli.main --task captioning --dataset_type labeled
```

---

## ✅ Summary

- **Install models** (LLaVA, Qwen, GPT-4o)  
- **UI:** `streamlit run app_ui/streamlit_ui/streamlit_app.py`  
- **CLI (single task):** `python -m multimodal_eval.cli.main --task captioning --dataset_type labeled`  
- **CLI (all tasks):** `python -m multimodal_eval.cli.main --task all --dataset_type labeled`  
- **Add dataset:** `--sources user` or `add_dataset()`  
- **Change model:** edit `MODEL_NAME` in `orchestrator.py`  
- **Test dataset:** `captioning_set.json`  

---
