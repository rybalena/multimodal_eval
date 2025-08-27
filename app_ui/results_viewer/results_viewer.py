import re
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import date

REGISTRY_CSV = Path("results/all_runs.csv")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
TASK_METRICS: Dict[str, List[str]] = {
    "captioning": ["cider", "clip_score", "semantic_similarity"],
    "hallucination": ["hallucination_metric"],
    "vqa": ["clip_score", "semantic_similarity", "contextual_relevance"],
    "contextual_relevance": ["clip_score", "cider", "semantic_similarity"],
}
METRIC_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "cider": ["mean_cider", "mean_CIDEr", "cider_mean"],
    "clip_score": ["mean_clip_score", "clip_mean", "mean_CLIP"],
    "semantic_similarity": ["mean_semantic_similarity", "semantic_mean"],
    "contextual_relevance": ["mean_contextual_relevance", "context_mean"],
    "hallucination_metric": ["mean_hallucination_score", "mean_hallucination_raw", "hallucination_mean"],

    "image_grounding": ["mean_image_grounding"],
    "coverage": ["mean_coverage"],
    "fluency": ["mean_fluency"],
    "composite": ["mean_composite_score"],
}

TASK_DISPLAY_MAP = {
    "captioning": "Captioning",
    "hallucination": "Hallucination",
    "vqa": "VQA",
    "contextual_relevance": "Contextual Relevance",
}
TASK_DISPLAY_TO_RAW = {v: k for k, v in TASK_DISPLAY_MAP.items()}

DATASET_DISPLAY_MAP = {"labeled_data": "Labeled data", "unlabeled_data": "Unlabeled data"}
DATASET_DISPLAY_TO_RAW = {v: k for k, v in DATASET_DISPLAY_MAP.items()}

# ──────────────────────────────────────────────────────────────────────────────
# ALIASES: group models in UI
# ──────────────────────────────────────────────────────────────────────────────
def model_alias(raw: str) -> str:
    low = str(raw).strip().lower()
    if low.startswith(("openai", "gpt-", "gpt", "o1", "o3")):
        return "OpenAI"
    if low.startswith("llava"):
        return "LLaVA"
    if low.startswith("qwen"):
        return "Qwen"
    return str(raw)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def __m4(val):
    if isinstance(val, (float, np.floating)):
        s = str(val)
        if '.' not in s:
            return s
        int_part, dec_part = s.split('.', 1)
        dec_part = dec_part[:4].rstrip('0')
        return int_part if dec_part == "" else f"{int_part}.{dec_part}"
    if isinstance(val, (int, np.integer)):
        return str(val)
    return val

# timestamp formatting
_TSDATE_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2}(?:\.\d+)?)')
_TZTAIL_RE = re.compile(r'(?:\s*(?:[+-]\d{2}:\d{2}|Z))+\s*$')

def _fmt_ts(x: str) -> str:
    s = str(x) if x is not None else ""
    if not s:
        return s
    s = _TZTAIL_RE.sub('', s)
    s = _TSDATE_RE.sub(r'\1   \2', s)
    return s

def _left_align(styler):
    return (
        styler.set_properties(**{"text-align": "left"})
              .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}])
    )

def _style_winner(df_display: pd.DataFrame, base_sel: pd.DataFrame):
    """Soft green highlight of the winner by Composite Score (only if exactly 2 models are selected)."""
    styler = df_display.style
    styler = _left_align(styler)
    if len(base_sel) == 2 and "mean_composite_score" in base_sel.columns:
        try:
            c0 = float(base_sel.loc[0, "mean_composite_score"])
            c1 = float(base_sel.loc[1, "mean_composite_score"])
        except Exception:
            return styler
        if pd.notna(c0) and pd.notna(c1) and c0 != c1:
            winner_idx = 0 if c0 > c1 else 1
            def _hl_row(row):
                if row.name == winner_idx:
                    return ['background-color: rgba(76, 175, 80, 0.12)'] * len(row)
                return [''] * len(row)
            styler = styler.apply(_hl_row, axis=1)
    return styler

@st.cache_data(show_spinner=False)
def load_registry(csv_path: Path) -> pd.DataFrame:

    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "run_timestamp" in df.columns:
        df["run_timestamp"] = df["run_timestamp"].astype(str)
    # Numeric fields
    for c in [
        "num_samples", "mean_image_grounding", "mean_coverage", "mean_fluency",
        "mean_hallucination_raw", "mean_hallucination_score",
        "mean_contextual_relevance", "mean_composite_score",
        "mean_cider", "mean_clip_score", "mean_semantic_similarity",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def resolve_metric_columns(task: str, df_columns: List[str]) -> List[Tuple[str, str]]:
    """Return only (metric_key, csv_column) for metrics of the selected task that actually exist in df."""
    out: List[Tuple[str, str]] = []
    for m in TASK_METRICS.get(task, []):
        col = next((c for c in METRIC_COLUMN_CANDIDATES.get(m, []) if c in df_columns), None)
        if col:
            out.append((m, col))
    return out

def _humanize(df_like: pd.DataFrame) -> pd.DataFrame:

    out = df_like.copy()
    if "run_timestamp" in out.columns:
        out["run_timestamp"] = out["run_timestamp"].map(_fmt_ts)
    if "dataset_type" in out.columns:
        out["dataset_type"] = out["dataset_type"].map(lambda x: DATASET_DISPLAY_MAP.get(str(x), str(x)))
    if "task" in out.columns:
        out["task"] = out["task"].map(lambda x: TASK_DISPLAY_MAP.get(str(x), str(x).title()))
    if "model_name" in out.columns:
        out["model_name"] = out["model_name"].map(model_alias)

    if "mean_score" in out.columns:
        out = out.drop(columns=["mean_score"])

    if "mean_composite_score" in out.columns:
        if "Composite Score" in out.columns:
            comp_col = "Composite Score"
        else:
            out = out.rename(columns={"mean_composite_score": "Composite Score"})
            comp_col = "Composite Score"
        cols = [c for c in out.columns if c != comp_col] + [comp_col]
        out = out[cols]

    return out

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Run History", layout="wide")
st.title("Run History")

df = load_registry(REGISTRY_CSV)
if df.empty:
    st.warning("No results found: results/all_runs.csv is missing or empty.")
    st.stop()

# Dataset values + defaults
if "dataset_type" in df.columns:
    dataset_raw_vals = sorted(df["dataset_type"].dropna().unique().tolist())
else:
    dataset_raw_vals = []
dataset_display_vals = ["All"] + [DATASET_DISPLAY_MAP.get(d, d) for d in dataset_raw_vals]
dataset_default = "Labeled data" if "labeled_data" in dataset_raw_vals else ("All" if not dataset_raw_vals else DATASET_DISPLAY_MAP.get(dataset_raw_vals[0], dataset_raw_vals[0]))

# Task values + defaults
if "task" in df.columns:
    task_raw_vals = sorted(df["task"].dropna().unique().tolist())
else:
    task_raw_vals = []
task_display_vals = ["All"] + [TASK_DISPLAY_MAP.get(t, t.title()) for t in task_raw_vals]
task_default = "Captioning" if "captioning" in task_raw_vals else ("All" if not task_raw_vals else TASK_DISPLAY_MAP.get(task_raw_vals[0], task_raw_vals[0].title()))

# Models (alias)
all_raw_models = sorted(df["model_name"].dropna().unique().tolist()) if "model_name" in df.columns else []
alias_to_raw: Dict[str, List[str]] = {}
for raw in all_raw_models:
    a = model_alias(raw)
    alias_to_raw.setdefault(a, []).append(raw)
alias_options = sorted(alias_to_raw.keys())
default_aliases = [alias_options[0]] if alias_options else []

# Dates presence
def _extract_dates(series: pd.Series):
    ds = pd.to_datetime(series.astype(str).str.slice(0, 10), errors="coerce")
    return ds.dt.date

if "run_timestamp" in df.columns and not df["run_timestamp"].isna().all():
    dates_series = _extract_dates(df["run_timestamp"])
    if dates_series.notna().any():
        min_date, max_date = dates_series.min(), dates_series.max()
        has_date = True
    else:
        min_date = max_date = date.today()
        has_date = False
else:
    min_date = max_date = date.today()
    has_date = False

# Controls
c1, c2, c3, c4, c5 = st.columns([1, 1, 1.2, 1.2, 2.6])
with c1:
    dataset_display = st.selectbox("Dataset", options=dataset_display_vals, index=dataset_display_vals.index(dataset_default))
    dataset_raw = None if dataset_display == "All" else DATASET_DISPLAY_TO_RAW.get(dataset_display, dataset_display)
with c2:
    task_display = st.selectbox("Task", options=task_display_vals, index=task_display_vals.index(task_default))
    task_raw = None if task_display == "All" else TASK_DISPLAY_TO_RAW.get(task_display, task_display.lower())
with c3:
    if has_date:
        date_from = st.date_input("From", value=min_date)
    else:
        st.text("No date available")
        date_from = None
with c4:
    if has_date:
        date_to = st.date_input("To", value=max_date)
    else:
        st.text("No date available")
        date_to = None
with c5:
    models_selected_alias = st.multiselect(
        "Pick model (1 or 2 for comparison)",
        options=alias_options,
        default=default_aliases,
    )

# Map selected aliases → raw model names
selected_raw_models: List[str] = []
for a in models_selected_alias:
    selected_raw_models.extend(alias_to_raw.get(a, []))

# ────────────
# Filtering
# ────────────
base = df.copy()
if dataset_raw and "dataset_type" in base.columns:
    base = base[base["dataset_type"] == dataset_raw]
if task_raw and "task" in base.columns:
    base = base[base["task"] == task_raw]

# date filter
if "run_timestamp" in base.columns and not base["run_timestamp"].isna().all():
    if date_from is not None and date_to is not None:
        base_dates = _extract_dates(base["run_timestamp"])
        mask = (base_dates >= date_from) & (base_dates <= date_to)
        base = base[mask]
else:
    st.caption("No date available")

# model filter (aliases → raw)
if selected_raw_models:
    base = base[base["model_name"].isin(selected_raw_models)]

if base.empty:
    st.warning("No runs found for the selected filters.")
    st.stop()

if dataset_display == "All" or task_display == "All":
    st.subheader("Filtered Runs (all matches)")
    show_all_tbl = _humanize(base).applymap(__m4)
    st.dataframe(_left_align(show_all_tbl.style), use_container_width=True, hide_index=True)
    st.stop()


if len(models_selected_alias) == 1:
    alias = models_selected_alias[0]
    raws = alias_to_raw.get(alias, [])
    one_df = base[base["model_name"].isin(raws)].sort_values("run_timestamp", ascending=False)

    metric_pairs = resolve_metric_columns(task_raw, one_df.columns.tolist()) if task_raw else []
    task_metric_cols = [col for _, col in metric_pairs]

    base_cols = ["run_timestamp", "dataset_type", "task", "model_name", "num_samples"]
    display_cols = [c for c in base_cols if c in one_df.columns] + task_metric_cols

    if "mean_composite_score" in one_df.columns:
        display_cols.append("mean_composite_score")

    rename_map = {col: mk.replace("_", " ").title() for mk, col in metric_pairs}

    one_tbl = one_df[display_cols].rename(columns=rename_map)
    one_tbl = _humanize(one_tbl).applymap(__m4)
    st.subheader(f"Runs Summary")
    st.dataframe(_left_align(one_tbl.style), use_container_width=True, hide_index=True)
    st.stop()

# ────────────────────────────────────
# Two models: compare the latest runs
# ────────────────────────────────────
if len(models_selected_alias) > 2 or len(models_selected_alias) == 0:
    st.info("Please select 1 or 2 models.")
    st.stop()

chosen_rows = []
for a in models_selected_alias:
    raws = alias_to_raw.get(a, [])
    sub = base[base["model_name"].isin(raws)].sort_values("run_timestamp")
    if sub.empty:
        st.error(f"No runs for model {a}.")
        st.stop()
    chosen_rows.append(sub.iloc[-1])

sel_df = pd.DataFrame(chosen_rows).reset_index(drop=True)

def _winner_banner(df_sel: pd.DataFrame):
    if len(df_sel) != 2 or "mean_composite_score" not in df_sel.columns:
        return
    try:
        a0 = model_alias(df_sel.loc[0, "model_name"])
        a1 = model_alias(df_sel.loc[1, "model_name"])
        c0 = float(df_sel.loc[0, "mean_composite_score"])
        c1 = float(df_sel.loc[1, "mean_composite_score"])
    except Exception:
        return
    if pd.notna(c0) and pd.notna(c1):
        if c0 > c1:
            st.success(f"Higher Score: {a0}  (Composite Score {c0:.4f} vs {c1:.4f})")
        elif c1 > c0:
            st.success(f"Higher Score: {a1}  (Composite Score {c1:.4f} vs {c0:.4f})")
        else:
            st.info(f"Equal Score: {a0} = {a1}  (Composite Score {c0:.4f})")

if len(sel_df) == 2:
    _winner_banner(sel_df)

st.subheader("Runs Summary")

metric_pairs = resolve_metric_columns(task_raw, sel_df.columns.tolist())
task_metric_cols = [col for _, col in metric_pairs]

base_cols = ["run_timestamp", "dataset_type", "task", "model_name", "num_samples"]
display_cols = [c for c in base_cols if c in sel_df.columns] + task_metric_cols

if "mean_composite_score" in sel_df.columns:
    display_cols.append("mean_composite_score")

rename_map = {col: mk.replace("_", " ").title() for mk, col in metric_pairs}

pretty_sel = sel_df[display_cols].rename(columns=rename_map)
pretty_sel = _humanize(pretty_sel).applymap(__m4)
styler_summary = _style_winner(pretty_sel, sel_df)
st.dataframe(styler_summary, use_container_width=True, hide_index=True)

