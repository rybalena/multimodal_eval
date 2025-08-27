import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

PREFERRED_METRICS_ORDER = [
    "mean_composite_score",
    "mean_image_grounding",
    "mean_coverage",
    "mean_fluency",
    "mean_hallucination_raw",
    "mean_clip_score",
    "mean_semantic_similarity",
    "mean_cider",
]

DATASET_MAP = {
    "labeled": "labeled_data",
    "unlabeled": "unlabeled_data",
}

def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"[ERROR] CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r)

def _norm(s: Optional[str]) -> str:
    return (s or "").strip()

def _norm_lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _flt(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _pick_row(
    rows: List[Dict[str, str]],
    *,
    model_name: str,
    dataset_type: str,
    task: str,
    strategy: str,
    run_ts: Optional[str] = None,
) -> Dict[str, str]:
    cand = [
        r for r in rows
        if _norm_lower(r.get("model_name")) == _norm_lower(model_name)
        and _norm_lower(r.get("dataset_type")) == _norm_lower(dataset_type)
        and _norm_lower(r.get("task")) == _norm_lower(task)
    ]
    if not cand:
        raise SystemExit(f"[ERROR] No runs for model='{model_name}', dataset='{dataset_type}', task='{task}'")

    if strategy == "ts":
        if not run_ts:
            raise SystemExit("--ts1/--ts2 required when --pick ts")
        for r in cand:
            if _norm(r.get("run_timestamp")) == _norm(run_ts):
                return r
        raise SystemExit(f"[ERROR] run_timestamp='{run_ts}' not found for model='{model_name}'")
    elif strategy == "best":
        def key_best(r):
            ms = _flt(r.get("mean_score"))
            if ms is None:
                ms = _flt(r.get("mean_composite_score"))

            return (-(ms if ms is not None else -1e9), _norm(r.get("run_timestamp")))
        cand_sorted = sorted(cand, key=key_best)
        return cand_sorted[0]
    else:
        # latest по run_timestamp (ISO)
        cand_sorted = sorted(cand, key=lambda r: _norm(r.get("run_timestamp")), reverse=True)
        return cand_sorted[0]

def _collect_metric_columns(a: Dict[str, str], b: Dict[str, str]) -> List[str]:
    keys = {k for k in a.keys() | b.keys() if k.startswith("mean_")}
    keys.discard("mean_score")
    ordered = [k for k in PREFERRED_METRICS_ORDER if k in keys]
    rest = sorted(keys - set(ordered))
    return ordered + rest

def _fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if isinstance(v, float) else ""

def _print_table(title: str, headers: List[str], rows: List[List[str]], markdown: bool = False) -> None:
    if markdown:
        print(f"\n### {title}\n")
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["-" * len(h) for h in headers]) + " |")
        for r in rows:
            print("| " + " | ".join(r) + " |")
        return

    print(f"\n{title}")
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths) + "-" * (3 * (len(widths) - 1)))
    for r in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(r, widths)))

def _diff_rows(metrics: List[str], a: Dict[str, str], b: Dict[str, str]) -> List[List[str]]:
    out: List[List[str]] = []
    for m in ["mean_score"] + metrics:
        va = _flt(a.get(m))
        vb = _flt(b.get(m))
        delta = (vb - va) if (va is not None and vb is not None) else None
        out.append([m, _fmt(va), _fmt(vb), _fmt(delta) if delta is not None else ""])
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Compare two models using aggregated runs in results/all_runs.csv (no model execution)."
    )
    ap.add_argument("--csv", default="results/all_runs.csv", help="Path to runs registry CSV")
    ap.add_argument("--dataset", required=True, choices=["labeled", "unlabeled"], help="Dataset type to filter")
    ap.add_argument("--task", required=True, choices=["captioning", "vqa", "contextual_relevance", "hallucination"])
    ap.add_argument("--models", nargs=2, required=True, metavar=("MODEL_A", "MODEL_B"), help="Two model names as saved in registry")
    ap.add_argument("--pick", choices=["latest", "best", "ts"], default="latest",
                    help="Which run to pick per model: latest (by timestamp), best (by mean_score), ts (exact timestamp)")
    ap.add_argument("--ts1", help="Exact run_timestamp for MODEL_A if --pick ts")
    ap.add_argument("--ts2", help="Exact run_timestamp for MODEL_B if --pick ts")
    ap.add_argument("--markdown", action="store_true", help="Print tables in Markdown")
    ap.add_argument("--out-json", help="Optional path to save chosen rows + diff as JSON")
    args = ap.parse_args()

    rows = _read_csv(Path(args.csv))
    ds_folder = DATASET_MAP[args.dataset]
    model_a, model_b = args.models[0], args.models[1]

    row_a = _pick_row(rows, model_name=model_a, dataset_type=ds_folder, task=args.task,
                      strategy=args.pick, run_ts=args.ts1)
    row_b = _pick_row(rows, model_name=model_b, dataset_type=ds_folder, task=args.task,
                      strategy=args.pick, run_ts=args.ts2)

    metric_cols = _collect_metric_columns(row_a, row_b)

    hdr1 = ["model_name", "run_timestamp", "dataset_type", "task", "num_samples", "mean_score"] + metric_cols
    r1a = [row_a.get("model_name",""), row_a.get("run_timestamp",""), row_a.get("dataset_type",""),
           row_a.get("task",""), row_a.get("num_samples",""), _fmt(_flt(row_a.get("mean_score")))]
    r1a += [_fmt(_flt(row_a.get(k))) for k in metric_cols]
    r1b = [row_b.get("model_name",""), row_b.get("run_timestamp",""), row_b.get("dataset_type",""),
           row_b.get("task",""), row_b.get("num_samples",""), _fmt(_flt(row_b.get("mean_score")))]
    r1b += [_fmt(_flt(row_b.get(k))) for k in metric_cols]
    _print_table("Selected runs", hdr1, [r1a, r1b], markdown=args.markdown)

    hdr2 = ["metric", row_a.get("model_name","A"), row_b.get("model_name","B"), "delta(B−A)"]
    diff_rows = _diff_rows(metric_cols, row_a, row_b)
    _print_table("Metric differences", hdr2, diff_rows, markdown=args.markdown)

    if args.out_json:
        out_payload = {
            "filter": {
                "csv": str(Path(args.csv)),
                "dataset_type": ds_folder,
                "task": args.task,
                "pick": args.pick,
                "ts1": args.ts1,
                "ts2": args.ts2,
            },
            "selected": {"A": row_a, "B": row_b},
            "metrics_shown": ["mean_score"] + metric_cols,
            "diff": [{"metric": r[0], "A": r[1], "B": r[2], "delta_B_minus_A": r[3]} for r in diff_rows],
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON: {args.out_json}")

if __name__ == "__main__":
    main()
