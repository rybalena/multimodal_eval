import os
from pathlib import Path
from typing import List, Tuple
import re  # >>> added

from dotenv import load_dotenv
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*encoder_attention_mask.*BertSdpaSelfAttention\.forward.*"
)

from multimodal_eval.data_sets import load_dataset
from multimodal_eval.evaluation.schema import Sample
from multimodal_eval.evaluation.evaluator import Evaluator
from multimodal_eval.model_wrappers.model_registry import get_model_wrapper
from multimodal_eval.evaluation.composite_score import compute_composite
from multimodal_eval.llm_eval.wrappers.openai_judge import call_llm_judge
from multimodal_eval.configs.tasks_config import CONFIG
from multimodal_eval.utils.downloader import download_image_to_tmp
from multimodal_eval.utils.saver import save_results_by_type
from multimodal_eval.llm_eval.utils.logger import print_scores_with_reason
from multimodal_eval.utils.saver import save_aggregated_metrics
from multimodal_eval.evaluation.hallucination_metric import compute_hallucination_metric
from multimodal_eval.llm_eval.utils.prompts import (
    build_caption_eval_prompt,
    build_contextual_prompt,
    build_hallucination_prompt,
)

# ────────────────────────────────────────────────────────────────────────────────
# env & model
# ────────────────────────────────────────────────────────────────────────────────

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_NAME = "gpt-4o" # ⬅️ ⬅️❗ CHANGE MODEL HERE — select "llava", "qwen", or "gpt-4o".
model = get_model_wrapper(MODEL_NAME)



if hasattr(model, "ensure_ready"):
    model.ensure_ready()


# ────────────────────────────────────────────────────────────────────────────────
# task → metrics mapping (for Labeled datasets)
# ────────────────────────────────────────────────────────────────────────────────
TASK_METRICS = {
    "hallucination": ["hallucination_metric"],
    "captioning": ["cider", "clip_score", "semantic_similarity"],
    "vqa": ["clip_score", "semantic_similarity", "contextual_relevance"],
    "contextual_relevance": ["clip_score", "cider", "semantic_similarity"],
}

# ─────────────────
# path helpers
# ─────────────────
def _find_ancestor_with(subpath: Path) -> Path:
    cur = Path(__file__).resolve().parent
    for _ in range(60):
        if (cur / subpath).exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    cur = Path.cwd().resolve()
    for _ in range(60):
        if (cur / subpath).exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError(f"Ancestor with '{subpath.as_posix()}' not found.")

PROJECT_ROOT_FOR_ASSETS = _find_ancestor_with(Path("resources/images"))

def resolve_image_path(image_str: str) -> str:
    s = str(image_str)
    if s.startswith("http"):
        return download_image_to_tmp(s)
    p = Path(s)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {p}")
        return str(p)
    abs_path = (PROJECT_ROOT_FOR_ASSETS / s).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"Image file not found: {abs_path}")
    return str(abs_path)

# ────────────────────────────────────────────────────────────────────────────────
# evaluation helpers
# ────────────────────────────────────────────────────────────────────────────────
def evaluate_sample(sample: Sample, metrics: List[str]) -> dict:
    evaluator = Evaluator(metrics=metrics)
    try:
        per_sample_data = evaluator.evaluate([sample], per_sample=True)["per_sample"]
        # Flatten {metric: [{id, score, ...}, ...]} -> {metric: score_for_this_sample}
        result = {}
        for metric_name, rows in per_sample_data.items():
            for v in rows:
                if v.get("id") == sample.id:
                    result[metric_name] = v.get("score", 0.0)
                    break
        return result
    except Exception as e:
        print(f"[⚠️ Metric Error] {e}")
        return {}

def print_custom_metadata(s: Sample, task: str, dataset_type: str, source_label: str):
    print(f"\n🧪 Task: {task.upper()} | Type: {source_label}")
    print(f"🖼️ Image: {s.image}")
    if getattr(s, "question", None):
        print(f"❓ Question: {s.question}")
    if getattr(s, "caption", None):
        print(f"📝 Caption: {s.caption}")
    if task == "vqa" and getattr(s, "expected_answer", None):
        print(f"✅ Expected Answer: {s.expected_answer}")
    print(f"🤖 Model_Output: {s.generated_answer}")


def _to_float_safe(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            x = float(v)
            if x != x or x in (float("inf"), float("-inf")):
                return None
            return x
        except Exception:
            return None
    s = str(v).strip()
    if not s:
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except Exception:
            return None
    m = re.match(r"^\s*[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        x = float(m.group(0))
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return x
    except Exception:
        return None


def _load_one_source(task: str, dataset_type: str, source_label: str, dataset_file: str, *, silent: bool) -> Tuple[str, List[Sample], str]:

    try:
        samples = load_dataset(task, dataset_file, source=source_label)
        if not silent:
            if source_label not in ("labeled", "unlabeled"):
                print(f"📄 Loading dataset file: {dataset_file}  (source={source_label})")
            print(f"📊 Loaded {len(samples)} samples from {source_label}.")
        return source_label, samples, dataset_file
    except FileNotFoundError:
        if not silent and source_label not in ("user_labeled", "user_unlabeled"):
            print(f"[ℹ️ Skip] Standard dataset for '{task}' ({dataset_type}) not found.")
        return source_label, [], dataset_file
    except Exception as e:
        if not silent:
            print(f"[❌ Error] {source_label}: {e}")
        return source_label, [], dataset_file

def load_sources_for(task: str, dataset_type: str, task_config: dict, sources_mode: str, *, silent: bool) -> List[Tuple[str, List[Sample], str]]:
    """
    sources_mode: 'standard' | 'user' | 'both'
    Returns list of tuples: (source_label, samples, dataset_file)
    """
    pairs: List[Tuple[str, List[Sample], str]] = []

    if sources_mode in ("standard", "both"):
        try:
            cfg_path = Path(task_config[dataset_type]).name
            pairs.append(_load_one_source(task, dataset_type, dataset_type, cfg_path, silent=silent))
        except KeyError:
            if not silent:
                print(f"[ℹ️ Info] No standard {dataset_type} file in CONFIG for task '{task}'")

    if sources_mode in ("user", "both"):
        user_label = "user_labeled" if dataset_type == "labeled" else "user_unlabeled"
        user_filename = f"{task}_dataset.json"
        pairs.append(_load_one_source(task, dataset_type, user_label, user_filename, silent=silent))

    return pairs

# ────────────────────────────────────────────────────────────────────────────────
# main entry used by CLI
# ────────────────────────────────────────────────────────────────────────────────
def run_task(task: str, task_config: dict, dataset_type: str, sources_mode: str):
    # 1) Silent probe
    probe_pairs = load_sources_for(task, dataset_type, task_config, sources_mode, silent=True)

    # 2) No datasets at all → one-line info and return
    if all((not samples) for (_, samples, _) in probe_pairs):
        print(f"[ℹ️ Info] No datasets for task='{task}', type='{dataset_type}', sources='{sources_mode}'.")
        return

    # 3) Header
    if not (dataset_type in ("labeled", "unlabeled") and sources_mode == "standard"):
        print(f"\n📂 [START] Task: {task.upper()} | Type: {dataset_type.upper()} | Sources: {sources_mode}")
    else:
        print(f"\n📂 [START] Task: {task.upper()} | Type: {dataset_type.upper()}")

    # 4) Load (non-silent)
    pairs = load_sources_for(task, dataset_type, task_config, sources_mode, silent=False)

    all_results: List[dict] = []

    for source_label, samples, _dataset_file in pairs:
        if not samples:
            continue

        metrics = TASK_METRICS.get(task, []) if dataset_type == "labeled" else []

        for i, s in enumerate(samples, 1):
            print(f"\n🔄 Processing image {i}/{len(samples)}: {getattr(s, 'id', 'no-id')}")
            try:
                image_path = resolve_image_path(s.image)
                s.generated_answer = model.run(sample=s, task_type=task, image_path=image_path)
                s.model_output = s.generated_answer
            except Exception as e:
                print(f"[⚠️ Generation Error] Sample {getattr(s, 'id', 'no-id')}: {e}")
                continue

            print_custom_metadata(s, task=task, dataset_type=dataset_type, source_label=source_label)

            if dataset_type == "labeled":
                score_dict = {}
                composite_input = {}

                if task == "hallucination":
                    print("🧠 Computing hallucination metric...")
                    halluc_scores = compute_hallucination_metric(s)
                    score_dict.update(halluc_scores)
                    composite_input = {
                        "hallucination_score": halluc_scores["hallucination_score"],
                        "semantic_similarity": halluc_scores["semantic_similarity"],
                    }
                else:
                    print("📏 Evaluating metrics...")
                    score_dict = evaluate_sample(s, metrics)

                    if task == "captioning":
                        composite_input = {
                            "clip_score": score_dict.get("clip_score", 0.0),
                            "semantic_similarity": score_dict.get("semantic_similarity", 0.0),
                            "cider": score_dict.get("cider", 0.0),
                        }
                    elif task == "vqa":
                        composite_input = {
                            "clip_score": score_dict.get("clip_score", 0.0),
                            "semantic_similarity": score_dict.get("semantic_similarity", 0.0),
                            "contextual_relevance": score_dict.get("contextual_relevance", 0.0),
                        }
                    elif task == "contextual_relevance":
                        composite_input = {
                            "clip_score": score_dict.get("clip_score", 0.0),
                            "cider": score_dict.get("cider", 0.0),
                            "semantic_similarity": score_dict.get("semantic_similarity", 0.0),
                        }

                # By design you skipped composite for hallucination. Keep it as-is.
                if task != "hallucination":
                    score_dict["composite_score"] = compute_composite(composite_input, task_type=task)

                entry = {
                    "id": getattr(s, "id", ""),
                    "image": s.image,
                    "model_name": MODEL_NAME,
                    "model_output": s.generated_answer,
                }

                if task == "vqa":
                    entry["question"] = getattr(s, "question", "")
                    entry["expected_answer"] = getattr(s, "expected_answer", "")
                elif task in ["captioning", "hallucination", "contextual_relevance"]:
                    entry["caption"] = getattr(s, "caption", "")

                if task == "hallucination":
                    entry.update({
                        "hallucination_raw": round(score_dict.get("hallucination_raw", 0.0), 4),
                        "hallucination_score": round(score_dict.get("hallucination_score", 0.0), 4),
                        "semantic_similarity": round(score_dict.get("semantic_similarity", 0.0), 4),
                        "composite_score": round(score_dict.get("composite_score", 0.0), 4),
                    })

                for m in TASK_METRICS.get(task, []):
                    if m in score_dict:
                        entry[m] = round(score_dict[m], 4)

                if task != "hallucination":
                    entry["composite_score"] = round(score_dict.get("composite_score", 0.0), 4)

                all_results.append(entry)

                if task != "hallucination":
                    for m in TASK_METRICS.get(task, []):
                        if m in score_dict:
                            print(f"📏 {m}: {score_dict[m]:.4f}")
                    print(f"📊 composite_score: {entry['composite_score']:.4f}")

            else:
                # UNLABELED: send to LLM judge
                try:
                    if not getattr(s, "generated_answer", None):
                        print(f"[⚠️ Warning] No generated answer for sample {getattr(s, 'id', 'no-id')}, skipping LLM judge.")
                        s.judge_notes = {"warning": "Empty generated answer"}
                    else:
                        print(f"🔍 Sending to LLM judge: {task}")
                        if task == "captioning":
                            prompt, system_prompt = build_caption_eval_prompt(s.generated_answer)
                            s.judge_notes = call_llm_judge(task, image_path=image_path, prompt=prompt,
                                                           system_prompt=system_prompt)
                        elif task == "contextual_relevance":
                            prompt = getattr(s, "prompt", None) or getattr(s, "question", None) or "Describe the image."
                            response = s.generated_answer
                            user_prompt, system_prompt = build_contextual_prompt(prompt, response)
                            s.judge_notes = call_llm_judge(task, image_path=image_path, prompt=user_prompt,
                                                           system_prompt=system_prompt)
                        elif task == "hallucination":
                            prompt, system_prompt = build_hallucination_prompt(
                                caption=s.generated_answer,
                                ground_truth_objects=getattr(s, "objects", []),
                            )
                            s.judge_notes = call_llm_judge(task, image_path=image_path, prompt=prompt,
                                                           system_prompt=system_prompt)
                        else:
                            raise ValueError(f"Unsupported task: {task}")

                        print_scores_with_reason(task, s.judge_notes, s.generated_answer, sample=s)

                except Exception as e:
                    print(f"[Judge Error] {e}")
                    s.judge_notes = {"error": "Judge failed", "exception": str(e)}

                judge_notes = getattr(s, "judge_notes", {}) or {}
                entry = {
                    "id": getattr(s, "id", ""),
                    "image": s.image,
                    "model_name": MODEL_NAME,
                    "model_output": s.generated_answer,
                    "judge_notes": judge_notes,
                }

                NUMERIC_KEYS = {
                    "image_grounding",
                    "coverage",
                    "fluency",
                    "hallucination_raw",
                    "prompt_alignment",
                    "hall_score",
                    "semantic_score",
                    "composite_score",
                }

                for k, v in judge_notes.items():
                    if k in NUMERIC_KEYS:
                        fv = _to_float_safe(v)
                        if fv is not None:
                            entry[k] = round(fv, 4)
                    # {"score": ...}
                    if isinstance(v, dict) and "score" in v and k in NUMERIC_KEYS:
                        fv = _to_float_safe(v.get("score"))
                        if fv is not None:
                            entry[k] = round(fv, 4)

                all_results.append(entry)

    print("\n💾 Saving results...")
    is_user_mode = (sources_mode == "user")

    save_results_by_type(
        task, dataset_type, CONFIG, all_results,
        is_user=is_user_mode, model_name=MODEL_NAME
    )

    print(f"✅ Done! {len(all_results)} total samples processed across sources.\n")

