# Examples Guide — MultimodalEval

⚠️ Before running these examples, complete installation steps in [Model Installation Guide](model_installation.md)

---

## 1. Captioning on Labeled Dataset
Run captioning evaluation using the Labeled dataset:

```bash
python -m multimodal_eval.cli.main --task captioning --dataset_type labeled --sources standard
```

---

## 2. Contextual Relevance on Unlabeled Dataset
Run contextual relevance evaluation on an unlabeled dataset:

```bash
python -m multimodal_eval.cli.main --task contextual_relevance --dataset_type unlabeled --sources standard
```

---

## 3. Run All Tasks on Labeled Dataset
Run all available tasks on the Labeled dataset:

```bash
python -m multimodal_eval.cli.run_all
```

---

## 4. Launch Streamlit UI
Start the interactive UI:

```bash
streamlit run app_ui/streamlit_ui/streamlit_app.py
```

Features:
- Select dataset (Labeled or Unlabeled)  
- Choose task (captioning, hallucination, VQA, contextual relevance)  
- Compare models (LLaVA, Qwen, GPT-4o)  
- View evaluation summary and composite scores  

---

## 📂 Results
After running, results will be saved in:
```
results/[labeled_data|unlabeled_data]/<task>/
```

Use `results/all_runs.csv` for a global summary of all runs.

---
