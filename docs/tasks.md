# Tasks Guide

## Overview
This guide describes the supported **evaluation tasks** in the framework, their configuration, and the expected input format. Each task has specific required fields (defined in the `Sample` schema) and uses its own evaluation metrics.

The main tasks currently supported are:  

- **Captioning**  
- **Hallucination Detection**  
- **Visual Question Answering (VQA)**  
- **Contextual Relevance**  

---

## 1. Captioning

### Description
Generates image captions and evaluates them against labeled (ground truth) captions.  

### Configuration
Defined in `tasks_config.py` under `CONFIG["captioning"]`.  
- **Datasets:**  
  - Labeled: `data_sets/labeled_data/captioning/captioning_set.json`  
  - Unlabeled: `data_sets/unlabeled_data/captioning/captioning_set.json`  
- **Outputs:**  
  - Labeled: `model_outputs/labeled_data/captioning/labeled_captioning_predictions.json`  
  - Unlabeled: `model_outputs/unlabeled_data/captioning/captioning_predictions.json`  
- **Metrics:** `cider`, `clip_score`, `semantic_similarity`  

### Required Fields (`Sample`)
- `id` (str)  
- `task` = `"captioning"`  
- `image` (str, image URL)  
- `caption` (str, labeled reference)  
- `model_output` (str, generated caption)  

### Example (JSON)
```json
{
  "id": "img_001",
  "task": "captioning",
  "image": "http://example.com/dog.png",
  "caption": "A dog running in the park.",
  "model_output": "A brown dog is running outside."
}
```

---

## 2. Hallucination Detection

### Description
Detects hallucinated content in generated captions relative to labeled references.  

### Configuration
Defined in `tasks_config.py` under `CONFIG["hallucination"]`.  
- **Datasets:**  
  - Labeled: `data_sets/labeled_data/hallucination/hallucination_set.json`  
  - Unlabeled: `data_sets/unlabeled_data/hallucination/hallucination_set.json`  
- **Outputs:**  
  - Labeled: `model_outputs/labeled_data/hallucination/labeled_hallucination_predictions.json`  
  - Unlabeled: `model_outputs/unlabeled_data/hallucination/hallucination_predictions.json`  
- **Metrics:** `hallucination_metric`  

### Required Fields (`Sample`)
- `id` (str)  
- `task` = `"hallucination"`  
- `image` (str)  
- `caption` (str, golden reference)  
- `model_output` (str, generated caption)  
- `hallucination_expected` (str, optional explanation/label)  

### Example (JSON)
```json
{
  "id": "img_045",
  "task": "hallucination",
  "image": "http://example.com/cat.png",
  "caption": "A cat sitting on a chair.",
  "model_output": "A cat sitting on a chair with a cup of coffee.",
  "hallucination_expected": "cup of coffee"
}
```

---

## 3. Visual Question Answering (VQA)

### Description
Answers natural language questions about an image.  

### Configuration
Defined in `tasks_config.py` under `CONFIG["vqa"]`.  
- **Datasets:**  
  - Labeled: `data_sets/labeled_data/vqa/vqa_set.json`  
- **Outputs:**  
  - Labeled: `model_outputs/labeled_data/vqa/labeled_vqa_predictions.json`  
- **Metrics:** `clip_score`, `semantic_similarity`, `contextual_relevance`  

### Required Fields (`Sample`)
- `id` (str)  
- `task` = `"vqa"`  
- `image` (str)  
- `question` (str)  
- `expected_answer` (str)  
- `model_output` (str, predicted answer)  

### Example (JSON)
```json
{
  "id": "img_101",
  "task": "vqa",
  "image": "http://example.com/bike.png",
  "question": "What color is the bicycle?",
  "expected_answer": "red",
  "model_output": "red"
}
```

---

## 4. Contextual Relevance

### Description
Evaluates whether a generated output is relevant to a given context, prompt, or expected behavior.  

### Configuration
Defined in `tasks_config.py` under `CONFIG["contextual_relevance"]`.  
- **Datasets:**  
  - Labeled: `data_sets/labeled_data/contextual_relevance/contextual_relevance_set.json`  
  - Unlabeled: `data_sets/unlabeled_data/contextual_relevance/contextual_relevance_set.json`  
- **Outputs:**  
  - Labeled: `model_outputs/labeled_data/contextual_relevance/contextual_relevance_predictions.json`  
  - Unlabeled: `model_outputs/unlabeled_data/contextual_relevance/contextual_relevance_predictions.json`  
- **Metrics:** `clip_score`, `cider`, `semantic_similarity`  

### Required Fields (`Sample`)
- `id` (str)  
- `task` = `"contextual_relevance"`  
- `image` (str)  
- `prompt` (str)  
- `expected_behavior` (str)  
- `expected_score` (float, optional)  
- `relevant` (bool, ground truth)  
- `model_output` (str, predicted relevance/caption)  

### Example (JSON)
```json
{
  "id": "img_210",
  "task": "contextual_relevance",
  "image": "http://example.com/book.png",
  "prompt": "Describe the book cover.",
  "expected_behavior": "Focus on the main illustration, not background.",
  "expected_score": 1.0,
  "relevant": true,
  "model_output": "A fantasy-style cover with a dragon illustration."
}
```

---

## 📊 Summary Table

| Task                  | Required Fields                                                                                 | Metrics                                |
|-----------------------|-------------------------------------------------------------------------------------------------|----------------------------------------|
| **Captioning**        | `id`, `task`, `image`, `caption`, `model_output`                                               | `cider`, `clip_score`, `semantic_similarity` |
| **Hallucination**     | `id`, `task`, `image`, `caption`, `model_output`, `hallucination_expected`                      | `hallucination_metric`                  |
| **VQA**               | `id`, `task`, `image`, `question`, `expected_answer`, `model_output`                           | `clip_score`, `semantic_similarity`, `contextual_relevance` |
| **Contextual Relevance** | `id`, `task`, `image`, `prompt`, `expected_behavior`, `expected_score`, `relevant`, `model_output` | `clip_score`, `cider`, `semantic_similarity` |
