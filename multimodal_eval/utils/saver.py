from __future__ import annotations

import os
import json
import csv
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable, Dict, Any, List
from zoneinfo import ZoneInfo  # py>=3.9


EXCLUDE_MEAN_KEYS = {"mean_hallucination_raw"}

EXCLUDE_METRIC_KEYS = {"hallucination_raw"}


# ────────────────────────────────────────────────────────────────────────────────
# Project root
# ────────────────────────────────────────────────────────────────────────────────

def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for cur in [here.parent, *here.parents]:
        if (cur / ".git").exists() or (cur / "pyproject.toml").exists() or (cur / "requirements.txt").exists():
            return cur
        if (cur / "MultimodalEval").is_dir() and cur != here.parent:
            return cur
    return here.parents[2]

PROJECT_ROOT: Path = _find_project_root()


# ────────────────────────────────────────────────────────────────────────────────
# TZ helpers
# ────────────────────────────────────────────────────────────────────────────────

def _resolve_local_tz() -> ZoneInfo | datetime.tzinfo:
    tz_name = os.environ.get("RUN_TZ") or os.environ.get("TZ")
    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    try:
        return datetime.now().astimezone().tzinfo
    except Exception:
        return ZoneInfo("UTC")

def _now_stamps() -> tuple[str, str]:
    local_tz = _resolve_local_tz()
    now_local = datetime.now(local_tz).replace(microsecond=0)
    now_utc   = datetime.now(ZoneInfo("UTC")).replace(microsecond=0)
    return now_local.isoformat(), now_utc.isoformat()


# ────────────────────────────────────────────────────────────────────────────────
# FS helpers
# ────────────────────────────────────────────────────────────────────────────────

def _insert_segment_after(path: Path, anchor: str, segment: str) -> Path:
    parts = path.parts
    try:
        i = parts.index(anchor)
    except ValueError:
        return path
    return Path(*parts[:i + 1], segment, *parts[i + 1:])

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _folder_name_for_dataset_type(dataset_type: str) -> str:
    return "labeled_data" if dataset_type == "labeled" else "unlabeled_data"

def _results_dir(dataset_type: str, task: str, is_user: bool) -> Path:
    """
      results/<labeled_data|unlabeled_data>/<task>/...
      results/user_dataset/<labeled_data|unlabeled_data>/<task>/...
    """
    base = PROJECT_ROOT / "results"
    ds_folder = _folder_name_for_dataset_type(dataset_type)
    if is_user:
        base = base / "user_dataset" / ds_folder
    else:
        base = base / ds_folder
    return base / task


# ────────────────────────────────────────────────────────────────────────────────
# Runs registry (CSV/JSONL)
# ────────────────────────────────────────────────────────────────────────────────

def _runs_registry_dir() -> Path:
    return PROJECT_ROOT / "results"

def _append_csv_row(csv_path: Path, row: Dict[str, Any], base_order: List[str]) -> None:
    _ensure_dir(csv_path.parent)
    row_keys = set(row.keys())

    if not csv_path.exists():

        metric_keys = sorted([
            k for k in row
            if k.startswith("mean_") and k not in ("mean_score",) and k not in EXCLUDE_MEAN_KEYS
        ])
        fieldnames = base_order + ["mean_score"] + metric_keys
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow({k: row.get(k, "") for k in fieldnames})
        return


    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        old_fieldnames = list(r.fieldnames or [])
        old_rows = list(r)


    new_fields_set = set(old_fieldnames) | row_keys
    mean_fields = sorted([
        k for k in new_fields_set
        if k.startswith("mean_") and k not in ("mean_score",) and k not in EXCLUDE_MEAN_KEYS
    ])
    other_fields = sorted([
        k for k in new_fields_set
        if (k not in base_order and k not in mean_fields and k != "mean_score")
    ])
    fieldnames = base_order + ["mean_score"] + mean_fields + other_fields


    if fieldnames == old_fieldnames:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow({k: row.get(k, "") for k in fieldnames})
        return


    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in old_rows:
            w.writerow({k: rec.get(k, "") for k in fieldnames})
        w.writerow({k: row.get(k, "") for k in fieldnames})

def _append_runs_registry(agg_summary: Dict[str, Any]) -> None:
    """
      - CSV:   results/all_runs.csv
      - JSONL: results/all_runs.jsonl
    """
    reg_dir = _runs_registry_dir()
    _ensure_dir(reg_dir)
    csv_path = reg_dir / "all_runs.csv"
    jsonl_path = reg_dir / "all_runs.jsonl"

    base_order = ["run_timestamp", "dataset_type", "task", "model_name", "num_samples"]


    row: Dict[str, Any] = {
        "run_timestamp": agg_summary.get("run_timestamp", ""),
        "dataset_type":  agg_summary.get("dataset_type", ""),
        "task":          agg_summary.get("task", ""),
        "model_name":    agg_summary.get("model_name", ""),
        "num_samples":   agg_summary.get("num_samples", ""),
    }


    mean_comp = agg_summary.get("mean_composite_score", None)
    if isinstance(mean_comp, (int, float)):
        row["mean_score"] = round(float(mean_comp), 4)
    else:
        mean_vals = [
            float(v) for k, v in agg_summary.items()
            if k.startswith("mean_") and k not in EXCLUDE_MEAN_KEYS and isinstance(v, (int, float))
        ]
        row["mean_score"] = round(mean(mean_vals), 4) if mean_vals else ""


    for k, v in agg_summary.items():
        if k.startswith("mean_") and k not in EXCLUDE_MEAN_KEYS:
            row[k] = v

    # JSONL append
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # CSV append
    _append_csv_row(csv_path, row, base_order)


# ────────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────────

def save_results_by_type(
    task: str,
    dataset_type: str,
    config: Dict[str, Any],
    results: Iterable[Dict[str, Any]],
    *,
    is_user: bool = False,
    model_name: str | None = None,
) -> None:
    # 0) model name
    model_name = (
        model_name
        or (config.get(task, {}) or {}).get("model_name")
        or os.environ.get("MODEL_NAME")
        or "unknown_model"
    )

    # 1) model_outputs.json
    key = f"model_output_{dataset_type}"
    try:
        cfg_rel = Path(config[task][key])  #  'model_outputs/...'
    except KeyError:
        # print(f"[❌ Config Error] No key '{key}' found for task '{task}'")
        return
    model_out_path = cfg_rel if cfg_rel.is_absolute() else (PROJECT_ROOT / cfg_rel)
    if "model_outputs" not in model_out_path.parts:
        model_out_path = PROJECT_ROOT / "model_outputs" / _folder_name_for_dataset_type(dataset_type) / model_out_path.name
    if is_user:
        model_out_path = _insert_segment_after(model_out_path, "model_outputs", "user_dataset")
    _ensure_dir(model_out_path.parent)

    # 2) results paths
    out_dir = _results_dir(dataset_type, task, is_user)
    _ensure_dir(out_dir)
    results_json_path = out_dir / f"{task}_results.json"
    results_csv_path  = out_dir / f"{task}_results.csv"

    # 3) list
    results_list: List[Dict[str, Any]] = list(results)

    # 4) model_outputs (trimmed JSON)
    main_keys = ["id", "image", "caption", "question", "expected_answer", "model_output"]
    trimmed = [{k: r[k] for k in main_keys if k in r} for r in results_list]
    with model_out_path.open("w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2, ensure_ascii=False)
    print(f"💾 MODEL OUTPUTS saved to {model_out_path}")

    # 5) results JSON add timestamp
    run_ts_local, run_ts_utc = _now_stamps()
    ordered_results: List[OrderedDict] = []
    for r in results_list:
        head = [(k, r[k]) for k in main_keys if k in r]
        meta = [("model_name", model_name), ("run_timestamp", run_ts_local)]

        exclude = {*(k for k, _ in head), "model_name", "run_timestamp", "run_timestamp_utc"}
        tail = [(k, v) for k, v in r.items() if k not in exclude]
        ordered_results.append(OrderedDict(head + meta + tail))

    with results_json_path.open("w", encoding="utf-8") as f:
        json.dump(ordered_results, f, indent=2, ensure_ascii=False)
    print(f"📊 RESULTS saved to {results_json_path}")

    # 6) results CSV
    base_csv_fields = [
        "id", "image", "caption", "question", "expected_answer", "model_output",
        "model_name", "run_timestamp",
    ]
    non_metric = set(base_csv_fields)
    dynamic_metrics: List[str] = []
    for rec in ordered_results:
        for k, v in rec.items():
            if k in non_metric:
                continue
            try:
                float(v)
                if k not in dynamic_metrics:
                    dynamic_metrics.append(k)
            except (TypeError, ValueError):
                pass
    csv_fields = base_csv_fields + dynamic_metrics

    with results_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for rec in ordered_results:
            row = {k: rec.get(k, "") for k in csv_fields}
            w.writerow(row)
    # print(f"📄 RESULTS CSV saved to {results_csv_path}")

    # 7) summary (JSON + CSV + HTML) out_dir
    _save_summary_files(ordered_results, task, dataset_type, out_dir)


def save_aggregated_metrics(results, task, dataset_type, *, is_user: bool = False):
    """
    Public wrapper (for compatibility with the orchestrator).
    Writes summary into a SINGLE folder: results/[user_dataset/]<labeled_data|unlabeled_data>/<task>/
    """
    out_dir = _results_dir(dataset_type, task, is_user)
    _ensure_dir(out_dir)
    _save_summary_files(list(results), task, dataset_type, out_dir)


# ────────────────────────────────────────────────────────────────────────────────
# Internal: summary writers
# ────────────────────────────────────────────────────────────────────────────────

def _save_summary_files(results_list: List[Dict[str, Any]], task: str, dataset_type: str, out_dir: Path) -> None:
    NON_METRIC_KEYS = {
        "id","image","caption","question","expected_answer","model_output",
        "model_name","run_timestamp",
        "task","dataset_type","judge_notes","objects","ground_truth","prompt"
    }

    model_name = (results_list[0].get("model_name") if results_list else "") or ""
    run_ts_local = (results_list[0].get("run_timestamp") if results_list else "") or ""

    all_keys = set().union(*(r.keys() for r in results_list)) if results_list else set()


    numeric_metric_keys: List[str] = []
    for k in all_keys - NON_METRIC_KEYS:
        if k in EXCLUDE_METRIC_KEYS:
            continue
        vals = []
        for r in results_list:
            v = r.get(k, None)
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
        if vals:
            numeric_metric_keys.append(k)

    preferred = ["clip_score","semantic_similarity","cider","contextual_relevance","hallucination_score","composite_score"]
    ordered_metric_keys = [k for k in preferred if k in numeric_metric_keys] + sorted(
        [k for k in numeric_metric_keys if k not in preferred]
    )

    agg: Dict[str, Any] = {
        "task": task,
        "dataset_type": _folder_name_for_dataset_type(dataset_type),
        "model_name": model_name,
        "run_timestamp": run_ts_local,
        "num_samples": len(results_list),
    }


    for m in ordered_metric_keys:
        if m in EXCLUDE_METRIC_KEYS:
            continue
        vals = []
        for r in results_list:
            v = r.get(m)
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
        if vals:
            key = f"mean_{m}"
            if key in EXCLUDE_MEAN_KEYS:
                continue
            agg[key] = round(mean(vals), 4)

    # dirs
    sum_json = out_dir / f"{task}_summary.json"
    sum_csv  = out_dir / f"{task}_summary.csv"
    sum_html = out_dir / f"{task}_summary.html"

    # JSON
    with sum_json.open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    # print(f"🧾 SUMMARY JSON saved to {sum_json}")

    # CSV
    csv_fields = list(agg.keys())
    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        w.writerow(agg)
    # print(f"🧾 SUMMARY CSV saved to {sum_csv}")

    # (results/all_runs.*)
    _append_runs_registry(agg)
