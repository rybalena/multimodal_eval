# Metrics Guide

## Overview
This document describes all evaluation metrics available in the framework.  
Metrics are registered in **`metrics_registry.py`** and are applied per task type (`captioning`, `hallucination`, `vqa`, `contextual_relevance`).  

Each metric returns a score normalized to `[0.0 – 1.0]` (unless stated otherwise), where **higher = better**.  

---

## 1. CIDEr (Consensus-based Image Description Evaluation)

**File:** `evaluation/cider.py`  

- **Purpose:** Measures lexical overlap between generated and reference captions.  
- **Logic:**  
  - Uses term frequency (TF) and inverse document frequency (IDF) weighting.  
  - For captioning → compares `caption` vs. `generated_answer`.  
  - For VQA → compares `expected_answer` vs. `generated_answer`.  
- **Output:** float score (0–10, normalized via `composite_score` to [0–1]).  

**Example result:**
```json
{ "cider": 0.82 }
```

---

## 2. CLIP Score

**File:** `evaluation/clip_score.py`  

- **Purpose:** Visual grounding – checks if the generated text matches the image.  
- **Logic:**  
  - Uses OpenAI’s **CLIP** model (`openai/clip-vit-large-patch14-336` by default).  
  - Computes cosine similarity between image and text embeddings.  
  - Normalized to `[0.0 – 1.0]`.  

**Example result:**
```json
{ "clip_score": 0.73 }
```

---

## 3. Semantic Similarity

**File:** `evaluation/semantic_similarity.py`  

- **Purpose:** Checks whether generated text is semantically close to the expected one.  
- **Logic:**  
  - Uses **SBERT** (`all-MiniLM-L6-v2`).  
  - Compares embeddings of `(expected_answer or caption)` vs. `generated_answer`.  
  - Returns cosine similarity in `[0.0 – 1.0]`.  

**Example result:**
```json
{ "semantic_similarity": 0.88 }
```

---

## 4. Contextual Relevance

**File:** `evaluation/contextual_relevance.py`  

- **Purpose:** Measures whether generated text fits the **prompt / context / expected behavior**.  
- **Logic:**  
  - Builds context from `prompt_template`, `prompt`, `question`, `expected_answer`.  
  - Uses SBERT similarity between context and `generated_answer`.  
  - Normalized to `[0.0 – 1.0]`.  
- **Fallback:** If no context is found, falls back to comparing `expected_answer` vs. `generated_answer`.  

**Example result:**
```json
{ "contextual_relevance": 0.64 }
```

---

## 5. Hallucination Metric (Hybrid Hallucination)

**File:** `evaluation/hallucination_metric.py`  

- **Purpose:** Detects **hallucinated objects** in generated captions (objects not in ground truth).  
- **Logic:**  
  - Extracts tokens using **spaCy** + POS tagging.  
  - Weighs tokens (NOUN=1.0, VERB=0.8, ADJ=0.3).  
  - Checks similarity to neutral words (to ignore generic fillers).  
  - Computes:  
    - `hallucinated_objects` → list of extra objects  
    - `hallucination_raw` → ratio of mismatched tokens  
    - `hallucination_score`  
    - `semantic_similarity` → SBERT similarity of full texts  
    - `composite_score` 
- **Output:** dict with detailed breakdown.  

**Example result:**
```json
{
  "hallucinated_objects": ["coffee"],
  "hallucination_raw": 0.25,
  "hallucination_score": 0.75,
  "semantic_similarity": 0.82,
  "composite_score": 0.80
}
```

---

## 6. Composite Score

**File:** `evaluation/composite_score.py`  

- **Purpose:** Produces a single weighted score per task.  
- **Task-specific weights:**  
  - **Captioning:** `0.25 * clip_score + 0.50 * semantic + 0.25 * cider`  
  - **VQA:** `0.10 * clip_score + 0.40 * semantic + 0.50 * contextual`  
  - **Contextual Relevance:** `0.40 * clip_score + 0.30 * cider + 0.30 * semantic`  
  - **Hallucination:** handled separately inside `hallucination_metric.py`.  

**Example result:**
```json
{ "composite_score": 0.77 }
```

---

## 7. Metric Registry

**File:** `evaluation/metrics_registry.py`  

- Defines available metrics:  
  - `clip_score`  
  - `hallucination_metric`  
  - `contextual_relevance`  
  - `semantic_similarity`  
  - `cider`  
- Maps metrics to tasks:  
  - Captioning → `cider`, `clip_score`, `semantic_similarity`  
  - Hallucination → `hallucination_metric`, `semantic_similarity`  
  - VQA → `clip_score`, `semantic_similarity`, `contextual_relevance`  
  - Contextual Relevance → `clip_score`, `cider`, `semantic_similarity`  

---

## 📊 Summary Table

| Metric                 | Purpose                                    | Range        | Used in Tasks                        |
|-------------------------|--------------------------------------------|-------------|--------------------------------------|
| **CIDEr**              | Lexical overlap (TF-IDF)                   | 0–10 (→0–1) | Captioning, VQA, Contextual Relevance |
| **CLIP Score**         | Image-text alignment (CLIP)                | 0–1         | Captioning, VQA, Contextual Relevance |
| **Semantic Similarity**| Embedding similarity (SBERT)               | 0–1         | Captioning, VQA, Contextual Relevance, Hallucination |
| **Contextual Relevance**| Fit to prompt/expected behavior (SBERT)   | 0–1         | VQA, Contextual Relevance             |
| **Hallucination Metric**| Detects extra objects not in ground truth | 0–1 (+ list)| Hallucination                         |
| **Composite Score**    | Weighted overall task score                | 0–1         | All tasks                             |
