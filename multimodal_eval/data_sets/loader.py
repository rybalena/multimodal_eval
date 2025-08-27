import json
from pathlib import Path
from typing import List
from multimodal_eval.evaluation.schema import Sample

HF_BASE_URL = "https://huggingface.co/datasets/rybalena/aiqa_images/resolve/main"

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_ROOT = ROOT_DIR / "data_sets"

def get_image_url(task: str, filename: str) -> str:
    return f"{HF_BASE_URL}/{task}/{filename}"

def _is_local_project_path(p: str) -> bool:

    return ("/" in p) or p.startswith("resources/") or p.startswith("./") or p.startswith("../")

def load_dataset(task: str, filename: str, source: str = "labeled") -> List[Sample]:
    """
    Loads dataset from:
      - data_sets/labeled_data/<task>/<filename>
      - data_sets/unlabeled_data/<task>/<filename>
      - data_sets/user_dataset/labeled/<task>/<filename>
      - data_sets/user_dataset/unlabeled/<task>/<filename>
    """
    source_folder = {
        "labeled": "labeled_data",
        "unlabeled": "unlabeled_data",
        "user_golden": "user_dataset/labeled",
        "user_unlabeled": "user_dataset/unlabeled",
    }.get(source)

    if source_folder is None:
        raise ValueError(f"[❌] Unknown dataset source: {source}")

    path = DATASETS_ROOT / source_folder / task / filename
    if not path.exists():
        raise FileNotFoundError(f"[❌] Dataset file not found: {path}")

    with path.open(encoding="utf-8") as f:
        raw_data = json.load(f)

    is_golden = source in ("labeled", "user_golden")

    for item in raw_data:
        item.setdefault("task", task)

        # Transform only BASE names into HF URLs.
        # Leave local paths such as "resources/images/xxx.png" untouched.
        if "image" in item and item["image"] and not item["image"].startswith("http"):
            img = item["image"]
            if _is_local_project_path(img):
                # leave as is (resources/images/...)
                pass
            else:

                item["id"] = img
                item["image"] = get_image_url(task, img)

        if is_golden:
            if task == "captioning":
                item["caption"] = item.get("caption")
            elif task == "vqa":
                item["expected_answer"] = item.get("expected_answer")
                item["question"] = item.get("question")
            elif task == "hallucination":
                item["expected_answer"] = "No hallucination detected."
                item["caption"] = item.get("ground_truth")
                item["prompt"] = item.get("prompt_template")
            elif task == "contextual_relevance":
                item["expected_answer"] = "relevant" if item.get("relevant") else "irrelevant"
                item["caption"] = item.get("caption")
        else:
            # unlabeled
            if task == "hallucination":
                item["prompt"] = item.get("prompt_template", "")
            elif task == "contextual_relevance":
                item["prompt"] = item.get("caption", "")

    return [Sample(**item) for item in raw_data]
