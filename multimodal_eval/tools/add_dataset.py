import os
import json
from pathlib import Path

# =========================
# Auto-search for roots
# =========================

def _find_ancestor_with(subpath: Path) -> Path:
    """
    Search for an ancestor directory that contains the relative path `subpath`.
    """
    cur = Path(__file__).resolve().parent
    for _ in range(60):
        if (cur / subpath).is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    cur = Path.cwd().resolve()
    for _ in range(60):
        if (cur / subpath).is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    raise FileNotFoundError(f"Directory '{subpath.as_posix()}' not found higher in the tree.")

# Project root for images
PROJECT_ROOT_FOR_ASSETS = _find_ancestor_with(Path("resources/images"))
IMAGES_DIR = PROJECT_ROOT_FOR_ASSETS / "resources" / "images"

# Core root for datasets
CORE_ROOT_FOR_DATASETS = _find_ancestor_with(Path("data_sets"))
DATASETS_DIR = CORE_ROOT_FOR_DATASETS / "data_sets"

# =========================
# Task/field config
# =========================

GOLDEN_TASKS = ["captioning", "contextual_relevance", "hallucination", "vqa"]
UNLABELED_TASKS = ["captioning", "contextual_relevance", "hallucination"]

FIELDS_BY_TYPE_TASK = {
    ("labeled", "captioning"):              ["id", "image", "caption"],
    ("labeled", "contextual_relevance"):    ["id", "image", "caption"],
    ("labeled", "hallucination"):           ["id", "image", "ground_truth"],
    ("labeled", "vqa"):                     ["id", "image", "question", "expected_answer"],
    ("unlabeled", "captioning"):           ["id", "image"],
    ("unlabeled", "contextual_relevance"): ["id", "image"],
    ("unlabeled", "hallucination"):        ["id", "image"],
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

# =========================
# Utilities
# =========================

def ask_with_attempts(prompt: str, validate_fn=None, error_message: str = "Invalid input.") -> str:
    """
    Ask user for input up to 5 times. If validate_fn is provided, it must return True for valid input.
    """
    attempts = 0
    while attempts < 5:
        value = input(prompt).strip()
        if not validate_fn or validate_fn(value):
            return value
        attempts += 1
        print(f"❌ {error_message}")
    print("\n❌ Too many invalid attempts. Exiting.")
    raise SystemExit(1)

def choose_dataset_type() -> str:
    return ask_with_attempts(
        "Choose dataset type [labeled / unlabeled]: ",
        lambda v: v.lower() in ["labeled", "unlabeled"],
        "Invalid dataset type."
    ).lower()

def choose_task(dataset_type: str) -> str:
    allowed = GOLDEN_TASKS if dataset_type == "labeled" else UNLABELED_TASKS
    return ask_with_attempts(
        f"Choose task {allowed}: ",
        lambda v: v.lower() in allowed,
        f"Task is not allowed for '{dataset_type}'."
    ).lower()

def list_images(images_dir: Path) -> list[str]:
    if not images_dir.exists():
        return []
    return sorted([
        name for name in os.listdir(images_dir)
        if (images_dir / name).is_file() and Path(name).suffix.lower() in IMAGE_EXTS
    ])

def find_image_by_name(images_dir: Path, user_input: str) -> Path:
    """
    Accepts 'tree' or 'tree.png' (case-insensitive).
    Searches ONLY in images_dir.
    """
    user_input = user_input.strip()
    if " " in user_input and Path(user_input).suffix == "":
        user_input = user_input.split()[0]

    full = images_dir / user_input
    if full.exists():
        return full

    base = Path(user_input).stem.lower()
    files = list_images(images_dir)
    for f in files:
        if Path(f).stem.lower() == base:
            return images_dir / f
    for f in files:
        if Path(f).stem.lower().startswith(base):
            return images_dir / f

    return None

def dataset_json_path(dataset_type: str, task: str) -> Path:
    base = DATASETS_DIR / "user_dataset" / dataset_type / task
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{task}_dataset.json"

def load_dataset(path: Path):
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_dataset(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# =========================
# Collect sample
# =========================

def prompt_sample(fields, images_dir: Path):
    sample = {}
    last_image_filename = None

    imgs = list_images(images_dir)
    if imgs:
        print("Available images:", ", ".join(imgs))

    for field in fields:
        if field == "image":
            def validate_image(name):
                return find_image_by_name(images_dir, name) is not None
            user_input = ask_with_attempts(
                "Enter image name (e.g., tree or tree.png): ",
                validate_image,
                "Image not found in the images directory."
            )
            file_path = find_image_by_name(images_dir, user_input)
            filename = file_path.name
            sample["image"] = f"resources/images/{filename}"
            last_image_filename = filename

        elif field == "id":
            user_id = ask_with_attempts(
                "Enter id (leave empty to auto-generate from image filename): ",
                lambda v: True
            )
            sample["id"] = user_id

        else:
            value = ask_with_attempts(
                f"Enter {field}: ",
                lambda v: bool(v.strip())
            )
            sample[field] = value

    if "id" in fields and not sample.get("id"):
        sample["id"] = Path(last_image_filename).stem if last_image_filename else "sample"

    return sample

# =========================
# Main
# =========================

def main():
    print("\nSample Creator")
    print("==============")
    try:
        dataset_type = choose_dataset_type()
        task = choose_task(dataset_type)
        fields = FIELDS_BY_TYPE_TASK[(dataset_type, task)]

        sample = prompt_sample(fields, IMAGES_DIR)

        save_path = dataset_json_path(dataset_type, task)
        data = load_dataset(save_path)
        data.append(sample)
        save_dataset(save_path, data)

        print(f"\n✅ Saved to: {save_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
