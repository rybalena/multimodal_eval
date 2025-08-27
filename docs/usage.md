# Usage Guide 

MultimodalEval is a framework for evaluating multimodal AI models (captioning, hallucination detection, VQA, contextual relevance).  
This guide will help you get started quickly.

---

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rybalena/multimodal_eval.git
   cd multimodal_eval
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

4. Install Ollama (for LLaVA and Qwen models):  
   [Ollama Download](https://ollama.com/download)

---

## 🚀 Running

### CLI

Run evaluations with:
```bash
python -m multimodal_eval.cli.main --task <TASK> --dataset_type <TYPE> --sources <SRC>
```

**Arguments:**
- `--task` — one of:
  - `captioning`
  - `hallucination`
  - `vqa`
  - `contextual_relevance`
  - `all`
- `--dataset_type` — `labeled` or `unlabeled`
- `--sources` — `standard`, `user`, `both`

Example:
```bash
python -m multimodal_eval.cli.main --task captioning --dataset_type labeled --sources user
```

---
### 🔄 Example: Model Selection

Open `multimodal_eval/orchestrator.py`, find the variable `MODEL_NAME`, and select the model you want to use `"llava", "qwen" or "gpt-4o"`

```
MODEL_NAME = "gpt-4o" # ⬅️ ⬅️ ❗ CHANGE MODEL HERE — select "llava", "qwen", or "gpt-4o".
model = get_model_wrapper(MODEL_NAME)
```
---
### Streamlit UI

Start the interactive dashboard:
```bash
streamlit run app_ui/streamlit_ui/streamlit_app.py
```

Features:
- Compare multiple models
- Visualize metrics
- View results by task
- Compare Labeled vs Unlabeled data

---

## 📊 Metrics

| Task                  | Metrics                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| Captioning            | CIDEr, CLIPScore, Semantic Similarity                                   |
| Hallucination         | Hybrid Hallucination Score                                              |
| VQA                   | CLIPScore, Semantic Similarity, Contextual Relevance                    |
| Contextual Relevance  | CIDEr, CLIPScore, Semantic Similarity                                   |

> **Composite Score** is calculated for every task.

---

## 📂 Results

All results are saved in:
```
results/[user_dataset|labeled_data|unlabeled_data]/<task>/
```

- `all_runs.csv` — summary of all runs  
- JSON files — detailed results per sample  
- Streamlit UI — interactive charts and comparisons  

---

## 📘 Examples

1. **Captioning on labeled dataset:**
   ```bash
   python -m multimodal_eval.cli.main --task captioning --dataset_type labeled
   ```

2. **Hallucination on unlabeled dataset:**
   ```bash
   python -m multimodal_eval.cli.main --task hallucination --dataset_type unlabeled
   ```

3. **Run all tasks:**
   ```bash
   python -m multimodal_eval.cli.main --task all --dataset_type labeled
   ```

---

## 🛠 Supported Models

- **LLaVA** (via Ollama)  
- **Qwen2-VL (2B / 7B)** (via Ollama)  
- **OpenAI GPT-4o** (via OpenAI API)  

---

## ⚡ Tips

- Do **not** close Ollama when running LLaVA or Qwen models — the models must remain active.  
- Labeled datasets are required for quality metrics; unlabeled datasets are used with LLM-as-a-Judge.  
- Composite Score is the main indicator for comparing models.  
- Running `streamlit_app.py` is recommended for visual analysis.  

---

## 📞 Contact

Author: **Elena Rybina**  
GitHub: [rybalena](https://github.com/rybalena)  
LinkedIn: [Rybina Elena](https://www.linkedin.com/in/elena-rybina-5222bb118/)

---
