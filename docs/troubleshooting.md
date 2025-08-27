# Troubleshooting Guide 

This guide lists common issues and fixes when running the framework.

---

## ⚙️ Installation

- **Error: `ModuleNotFoundError`**  
  → Ensure installation was done correctly:  
  ```bash
  pip install -e .
  ```

- **Wrong Python version**  
  → Requires Python ≥ 3.9.  
  Check with:  
  ```bash
  python --version
  ```

---

## 🤖 Ollama & Models

- **`ollama: command not found`**  
  → Install Ollama from [ollama.com/download](https://ollama.com/download).  
  → Make sure the Ollama server is running.

- **Model not found (LLaVA / Qwen)**  
  → Pull the model manually:  
  ```bash
  ollama run llava:13b
  ollama run qwen2-vl:7b
  ```

---

## 🔑 OpenAI API

- **Error: `OPENAI_API_KEY not found`**  
  → Add to `.env` file at project root:  
  ```bash
  OPENAI_API_KEY=sk-xxxx
  ```

- **Rate limit exceeded**  
  → Check your OpenAI account quota or switch to Ollama models.

---

## 📂 Datasets

- **`FileNotFoundError: dataset not found`**  
  → Verify that dataset files exist in:  
  ```
  data_sets/labeled_data/<task>/...
  data_sets/unlabeled_data/<task>/...
  ```  
  → Labeled (Ground Truth) datasets require ground truth JSON.  
  → Unlabeled datasets can be evaluated only with LLM-as-a-Judge.

- **Invalid JSON format**  
  → Validate with:  
  ```bash
  python -m json.tool your_dataset.json
  ```

- **Mismatch between task and dataset type**  
  → Example: VQA only supports **Labeled** datasets, not Unlabeled.  

---

## 📊 Results

- **`results/all_runs.csv` is empty**  
  → Likely the model produced no predictions.  
  → Check task-specific logs in `results/<labeled_data|unlabeled_data>/<task>/`.

- **Metrics missing from results**  
  → Ensure dataset type supports metrics:  
    - **Labeled** → full metrics (CIDEr, CLIP, Semantic, etc.)  
    - **Unlabeled** → LLM judge only.

---

## 🎛 Streamlit

- **Error: `No module named streamlit`**  
  → Install with:  
  ```bash
  pip install streamlit
  ```

- **UI does not launch**  
  → Correct command:  
  ```bash
  streamlit run app_ui/streamlit_ui/streamlit_app.py
  ```

---

## 📝 Tips

- Always activate your virtual environment:  
  ```bash
  source .venv/bin/activate
  ```

- For debugging, print help:  
  ```bash
  python -m multimodal_eval.cli.main --help
  ```

- If GPU is unavailable, Torch falls back to CPU (slower).  
- Check dataset JSON format against [docs/user-guide/tasks.md](user-guide/tasks.md).  

---
