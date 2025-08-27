# Dataset Guide

## Overview
This guide describes how datasets are structured, loaded, and extended in the framework.  
Datasets are categorized into:  

- **Labeled datasets** → contain ground truth labels (used for evaluation).  
- **Unlabeled datasets** → contain only inputs (used for inference).  
- **User datasets** → custom data provided by users (stored under `data_sets/user_dataset/`).  

---

## 1. Dataset Structure

Datasets are stored under the `data_sets/` directory:

```
data_sets/
├── labeled_data/             # curated datasets with ground truth
│   ├── captioning/
│   ├── hallucination/
│   ├── vqa/
│   └── contextual_relevance/
├── unlabeled_data/           # inputs without ground truth
│   ├── captioning/
│   ├── hallucination/
│   └── contextual_relevance/
└── user_dataset/             # user-provided datasets
    ├── labeled/
    │   ├── captioning/
    │   ├── hallucination/
    │   ├── vqa/
    │   └── contextual_relevance/
    └── unlabeled/
        ├── captioning/
        ├── hallucination/
        └── contextual_relevance/
```

Each dataset file is a JSON list of samples, with fields defined by the **task type**.

---

## 2. Sample Formats

### Captioning (labeled)
```json
{
  "id": "kitten_box",
  "image": "kitten_in_a_box.jpg",
  "caption": "A small white kitten peeking out of a cardboard box."
}
```

### Captioning (unlabeled)
```json
{
  "id": "kitten_box",
  "image": "kitten_in_a_box.jpg"
}
```

---

### VQA (labeled)
```json
{
  "id": "img_01",
  "image": "bicycle.jpg",
  "question": "What color is the bicycle?",
  "expected_answer": "red"
}
```

---

### Hallucination (labeled)
```json
{
  "id": "img_02",
  "image": "cat.png",
  "ground_truth": "A cat sitting on a chair."
}
```

### Hallucination (unlabeled)
```json
{
  "id": "img_03",
  "image": "cat.png"
}
```

---

### Contextual Relevance (labeled)
```json
{
  "id": "img_04",
  "image": "book.png",
  "caption": "A fantasy-style book cover with a dragon."
}
```

### Contextual Relevance (unlabeled)
```json
{
  "id": "img_05",
  "image": "book.png"
}
```

---

## 3. Loading Datasets

Datasets are loaded via **`data_sets/loader.py`**:

```python
from multimodal_eval.data_sets.loader import load_dataset

samples = load_dataset("captioning", "captioning_set.json", source="labeled")
```

Supported `source` values:
- `labeled`  
- `unlabeled`  
- `user_labeled`  
- `user_unlabeled`

The loader automatically:
- Converts image filenames to HuggingFace URLs if needed.  
- Normalizes fields depending on the task type.  
- Returns a list of `Sample` objects.

---

## 4. Adding Custom Datasets

You can add new datasets interactively using **`tools/add_dataset.py`**:

```bash
python -m multimodal_eval.tools.add_dataset
```

The script will:
1. Ask you to choose dataset type (`labeled` / `unlabeled`).  
2. Ask you to choose a task (captioning, hallucination, vqa, contextual_relevance).  
3. Prompt you for required fields (id, image, caption, etc.).  
4. Save your sample into:
   ```
   data_sets/user_dataset/<labeled|unlabeled>/<task>/<task>_dataset.json
   ```

**Example console flow:**
```
Choose dataset type [labeled / unlabeled]: labeled
Choose task ['captioning', 'contextual_relevance', 'hallucination', 'vqa']: captioning
Enter image name (e.g., tree.png): kitten_in_a_box.jpg
Enter id (leave empty to auto-generate): kitten_box
Enter caption: A small white kitten peeking out of a cardboard box.

✅ Saved to: data_sets/user_dataset/labeled/captioning/captioning_dataset.json
⚠️ Remember to place your image file in: resources/images/kitten_in_a_box.jpg
```

---

## 5. Image Paths

- **Local images must be placed in the `resources/images/` directory.**  
  Example:  
  ```
  resources/images/kitten_in_a_box.jpg
  ```
  and referenced in the dataset as:  
  ```json
  { "image": "resources/images/kitten_in_a_box.jpg" }
  ```


**Important:**  
👉 When adding user datasets, always copy your images into **`resources/images/`** so they can be accessed during evaluation.

---

## 6. Summary

- **Labeled datasets**: labeled for evaluation.  
- **Unlabeled datasets**: only inputs.  
- **User datasets**: extend the framework with your own data.  
- Use **`loader.py`** to load datasets programmatically.  
- Use **`add_dataset.py`** to add custom samples interactively.  
- **Always store your own images in `resources/images/`** for proper loading.  
