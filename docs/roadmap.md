# 🗺️ Roadmap — MultimodalEval

This document outlines the current capabilities and planned enhancements for the MultimodalEval framework.  
The goal is to provide a clear view of development progress and upcoming milestones.

---

## ✅ Already Implemented

- **Multimodal tasks:** captioning, hallucination detection, VQA, contextual relevance  
- **Dataset support:** Labeled (Ground Truth) and Unlabeled datasets (Hugging Face integration)  
- **Core evaluation metrics:** CLIPScore, Semantic Similarity, CIDEr, Hybrid Hallucination  
- **LLM-as-a-Judge scoring:** automatic evaluation for unlabeled datasets  
- **User interfaces:** Streamlit UI for interactive exploration and CLI for orchestration  
- **Multi-model benchmarking:** support for Qwen, LLaVA, GPT-4o  

---

## 🚧 In Progress / Planned

- **Adversarial Red Teaming**  
  Develop a systematic evaluation pipeline to test robustness of multimodal models against adversarial and failure-inducing prompts.  
  - Stress-testing models with edge cases and perturbations  
  - Identifying vulnerabilities in reasoning and grounding  

- **Robustness Training on Failure Cases**  
  Introduce a loop where models are fine-tuned or reinforced using “difficult” examples discovered during evaluation.  
  - Curating a dataset of failure cases (hallucinations, irrelevant answers, misleading captions)  
  - Using these cases to improve stability and reliability of LLMs under diverse real-world conditions  

- **Extended Metrics and XAI**  
  - Add Explainable AI tooling (e.g., SHAP, LIME) for model decision transparency  
  - Introduce factuality and grounding metrics beyond current scores  

- **Expanded Model Coverage**  
  - Support additional multimodal models (e.g., BLIP-2, Mistral-based MLLMs)  
  - Benchmark across both open-source and proprietary APIs  

---

## 🎯 Long-Term Vision

MultimodalEval aims to become an industry-grade evaluation suite for multimodal AI models, combining:  
- Standardized benchmarking datasets  
- Robustness and adversarial stress testing  
- Human + LLM hybrid evaluation loops  
- Continuous integration for multimodal QA pipelines  

---
