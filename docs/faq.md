# ❓ Frequently Asked Questions (FAQ) 

This page collects answers to common questions when using **MultimodalEval**.

---

## 📖 Datasets

**Q: What is the difference between Labeled (Ground Truth) and Unlabeled datasets?**  
- **Labeled:** contains ground truth annotations (expected captions, answers, objects). Used for metric-based evaluation (CIDEr, CLIPScore, Semantic Similarity, etc.).  
- **Unlabeled:** has only input data (images, questions) without ground truth. Evaluation is performed with **LLM-as-a-Judge**.  

**Q: Can I add my own dataset?**  
Yes. Place your JSON under:  
```
data_sets/user_dataset/<task>/
```
Follow the schema in [Tasks Guide](tasks.md).

---

## ⚙️ Models

**Q: Which models are supported?**  
Currently:  
- **LLaVA** (via Ollama)  
- **Qwen2-VL (2B / 7B)** (via Ollama)  
- **OpenAI GPT-4o** (via API)  

More models (BLIP-2, Mistral-based MLLMs) are planned — see [Roadmap](roadmap.md).

**Q: Do I need to keep Ollama open?**  
Yes. LLaVA and Qwen run inside Ollama. Keep Ollama running in the background while using MultimodalEval.

---

## 📊 Results

**Q: Where are results stored?**  
All runs are stored under:
```
results/[labeled_data|unlabeled_data]/<task>/
```
The file `results/all_runs.csv` provides a summary across all tasks.

**Q: Why is `all_runs.csv` empty?**  
- The model produced no predictions (check logs).  
- Wrong dataset type chosen (e.g., VQA requires Labeled, not Unlabeled).  

---

## 🚀 Running

**Q: How do I run a quick test?**  
Use these commands:  
```bash
# Captioning on Labeled dataset
python -m multimodal_eval.cli.main --task captioning --dataset_type labeled

# Contextual Relevance on Unlabeled dataset
python -m multimodal_eval.cli.main --task contextual_relevance --dataset_type unlabeled
```

**Q: How do I launch the UI?**  
```bash
streamlit run app_ui/streamlit_ui/streamlit_app.py
```

---

## 🛠 Troubleshooting

**Q: `ollama: command not found`**  
Install Ollama: [ollama.com/download](https://ollama.com/download).

**Q: `OPENAI_API_KEY not found`**  
Add your API key to `.env`:
```
OPENAI_API_KEY=sk-xxxx
```

**Q: Metrics are missing in my results**  
- Labeled datasets → all metrics available.  
- Unlabeled datasets → only LLM-as-a-Judge metrics.

---

## 📬 Contributing

**Q: How can I contribute?**  
See [Contributing](../contributing.md). We welcome new metrics, tasks, and model integrations!

---
