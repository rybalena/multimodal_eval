# AI MultimodalEval  
📊 **Evaluation & QA Framework for Multimodal AI (Vision–Language Models)**  

---

## 🔍 What is MultimodalEval?

**MultimodalEvalCore** is an open-source framework for **evaluating multimodal AI models** (image → text).  
It is designed for **AI QA engineers, ML researchers, and practitioners** who need **reliable, reproducible, and extensible** benchmarks.  

Supports **Captioning, VQA, Hallucination Detection, and Contextual Relevance**, combining **classic metrics** (CIDEr, CLIPScore, SBERT) with **LLM-as-a-Judge** scoring.  

---

## ⚙️ Features

- 🖼 **Tasks**: Captioning · VQA · Hallucination Detection · Contextual Relevance  
- 📊 **Metrics**: CLIPScore · CIDEr · Semantic Similarity · Hybrid Hallucination · Composite Score  
- 🔌 **Models**: LLaVA (via Ollama) · Qwen2-VL · OpenAI GPT-4o  
- 🖥 **Interfaces**:  
  - CLI (`cli/main.py`, `cli/run_all.py`)  
  - Streamlit UI (`app_ui/streamlit_ui/streamlit_app.py`)  
  - Results Viewer (`app_ui/results_viewer/results_viewer.py`)  
  - Python API (`Evaluator`, `evaluate_sample`)  
- 📂 **Results**: JSON + CSV tables with metrics and composite scores  
- 🧪 **Testing**: PyTest, fixtures, coverage reports  
- 📥 **Custom datasets**: Add your own datasets via `--sources user` or `add_dataset`  
- 🛠 **Docs**: Full user guides for installation, usage, tasks, metrics, results, tests, troubleshooting, roadmap  

---

## 📁 Project Structure

```
multimodal_eval/
├── cli/                 # CLI entrypoints (main.py, run_all.py)
├── datasets/            # Labeled & Unlabeled datasets
├── evaluation/          # Metrics (clip, cider, hallucination, composite, etc.)
├── model_wrappers/      # LLaVA, Qwen, GPT-4o integration
├── app_ui/              # Streamlit UI + Results Viewer
├── configs/             # Task configs, prompts
├── results/             # Saved JSON/CSV evaluation outputs
├── tests/               # Unit & integration tests
└── docs/                # Full documentation
```

---

## 🚀 Quickstart

### 1. Install the framework
```bash
git clone https://github.com/rybalena/multimodal_eval.git
cd multimodal_eval
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. Install model backends (⚠️ Required)

MultimodalEvalCore will not work without at least one model installed.  
See the full [Installation Models Guide](docs/installation_models.md), or use quick steps below:

- **LLaVA via Ollama**  
  ```bash
  curl -fsSL https://ollama.com/download | sh
  ollama pull llava:7b    # smaller baseline
  ollama pull llava:14b   # larger benchmark model
  ```

- **Qwen2-VL-2B-Instruct**  
  ```bash
  pip install transformers accelerate
  ```

- **OpenAI GPT-4o**  
  ```bash
  pip install openai
  echo "OPENAI_API_KEY=sk-..." > .env
  ```

⚡ Do not close Ollama after launching a model (`ollama run llava:7b`) — it must stay active while running evaluations.

### 3. Run a task (CLI)
```bash
python -m multimodal_eval.cli.main --task captioning --dataset_type labeled
```

### 4. Run all tasks
```bash
python -m multimodal_eval.cli.run_all
```

### 5. Launch UI
```bash
streamlit run app_ui/streamlit_ui/streamlit_app.py
```

---

## 📊 Example Output

Results stored in `results/` as JSON/CSV tables.  

**Captioning summary (JSON):**
```json
{
  "task": "captioning",
  "dataset_type": "labeled_data",
  "model_name": "qwen",
  "num_samples": 100,
  "mean_clip_score": 0.74,
  "mean_semantic_similarity": 0.82,
  "mean_cider": 0.69,
  "mean_composite_score": 0.77
}
```

---

## 📊 Benchmarking & Results Viewer

The framework provides **two ways to benchmark models**:

### 1. Interactive Benchmark (Streamlit UI)
Run:
```bash
streamlit run app_ui/streamlit_ui/streamlit_app.py
```
Features:
- Side-by-side **model comparison** (e.g., Qwen vs LLaVA, GPT-4o vs LLaVA)  
- Choose **task** (captioning, VQA, hallucination, contextual relevance) and **dataset type** (Labeled / Unlabeled)  
- **Tabular benchmark view**: metrics + composite scores directly in the dashboard  
- Immediate feedback for experiments — no static HTML needed  

### 2. Results Viewer (History & Comparison)
Run:
```bash
streamlit run app_ui/results_viewer/results_viewer.py
```
Features:
- Loads aggregated runs from `results/all_runs.csv`  
- Displays a **comparison table** of models across multiple runs  
- Useful for **benchmark tracking** and reporting performance trends over time  

Together, these tools turn MultimodalEval into a **full benchmarking suite** for multimodal QA.

---

## 📥 Adding Custom Datasets

You can evaluate your own datasets by adding them via the **`user_dataset/`** path.  

### CLI with user datasets
```bash
python -m multimodal_eval.cli.main --task captioning --dataset_type labeled --sources user
```

### Programmatic API

```python
from multimodal_eval.utils.add_dataset import add_dataset

# Add custom dataset
add_dataset("my_dataset.json", dataset_type="labeled", task="captioning")

# Then run evaluation on it
```

This allows benchmarking not only on provided **Labeled** and **Unlabeled** datasets, but also on **your own data**.

---

## 📘 Documentation

- [Quickstart](docs/quickstart.md)  
- [Usage Guide](docs/usage.md)  
- [Running Guide](docs/running.md)  
- [Tasks Guide](docs/tasks.md)  
- [Metrics Guide](docs/metrics.md)  
- [Interpreting Results](docs/interpreting_results.md)  
- [Installation Models](docs/installation_models.md)  
- [Tests Guide](docs/tests.md)  
- [Troubleshooting](docs/troubleshooting.md)  
- [Roadmap](docs/roadmap.md)  

---

## 🔮 Roadmap Highlights

- **Adversarial Red Teaming** for robustness  
- **Failure-case loop**: fine-tuning on hallucination & irrelevant outputs  
- **Explainability (XAI)** with SHAP/LIME  
- **More models**: BLIP-2, MiniGPT-4, InstructBLIP, Mistral  

---

## 🤝 Contributing

Contributions are welcome: add new metrics, tasks, or model wrappers.  
- Fork → branch → PR  
- For large changes, open an issue first  

---

## 📄 License

MIT — see [LICENSE](LICENSE.md)

---

## 👩‍💻 Author

**Elena Rybina**  
AI QA Engineer · Computer Vision Enthusiast  
- [LinkedIn](https://www.linkedin.com/in/elena-rybina-5222bb118)  
- [GitHub](https://github.com/rybalena)  
