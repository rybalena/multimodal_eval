# app.py
import os
import json
from datetime import datetime
from typing import List, Optional

import streamlit as st
from PIL import Image

from multimodal_eval.evaluation.schema import Sample
from multimodal_eval.model_wrappers.model_registry import get_model_wrapper

from ui_styles import (
    inject_global_labels_css, inject_tooltip_css, inject_sidebar_css,
    inject_compare_css
)
from metrics_eval import TASK_METRICS, evaluate_golden
from judge_logic import build_prompt, enforce_n_sentences, judge_unlabeled
from renderers import render_summary_minimal, render_judge_summary

# ──────────────────────────
# UI
# ──────────────────────────
st.set_page_config(page_title="🧪 Multimodal AI Evaluator", layout="wide")
st.title("🧪 Multimodal AI Evaluator")

inject_global_labels_css()
inject_tooltip_css()

MODEL_LABELS = ["LLaVA", "Qwen", "OpenAI"]
MODEL_KEY_BY_LABEL = {"LLaVA": "llava", "Qwen": "qwen", "OpenAI": "gpt-4o"}

DATASET_DISPLAY_TO_VALUE = {"Labeled": "LABELED", "Unlabeled": "UNLABELED"}
DATASET_VALUE_TO_DISPLAY = {v: k for k, v in DATASET_DISPLAY_TO_VALUE.items()}

TASK_DISPLAY_TO_VALUE = {
    "Captioning": "captioning",
    "VQA": "vqa",
    "Contextual Relevance": "contextual_relevance",
    "Hallucination": "hallucination",
}
TASK_VALUE_TO_DISPLAY = {v: k for k, v in TASK_DISPLAY_TO_VALUE.items()}

# ──────────────────────────
# Sidebar
# ──────────────────────────
with st.sidebar:
    inject_sidebar_css()

    st.markdown("<h2 class='controls-title'>📊 Dashboard</h2>", unsafe_allow_html=True)

    st.markdown("<h2 class='section-title'>📂 Dataset</h2>", unsafe_allow_html=True)
    dataset_display = st.selectbox(
        label="", options=["Labeled", "Unlabeled"], index=0,
        label_visibility="collapsed", key="dataset_display",
    )
    dataset = DATASET_DISPLAY_TO_VALUE[dataset_display]

    st.markdown("<h2 class='section-title'>📌 Task</h2>", unsafe_allow_html=True)
    TASKS_GOLDEN_VALUES = ["captioning", "vqa", "contextual_relevance"]
    TASKS_UNLABELED_VALUES = ["captioning", "contextual_relevance", "hallucination"]
    task_values = TASKS_GOLDEN_VALUES if dataset == "LABELED" else TASKS_UNLABELED_VALUES
    task_display_options = [TASK_VALUE_TO_DISPLAY[v] for v in task_values]
    task_display = st.selectbox(
        label="", options=task_display_options, index=0,
        label_visibility="collapsed", key="task_display",
    )
    task = TASK_DISPLAY_TO_VALUE[task_display]

    st.markdown("<h2 class='section-title'>🧠 Model</h2>", unsafe_allow_html=True)
    _prior_compare = st.session_state.get("compare_models", [])
    model_label = st.selectbox(
        label="", options=MODEL_LABELS, index=0,
        label_visibility="collapsed", key="model_label",
        disabled=bool(_prior_compare),
    )
    model_key = MODEL_KEY_BY_LABEL[model_label]

    st.markdown("<h2 class='section-title'>⚖️ Model Comparison</h2>", unsafe_allow_html=True)
    selected_now: List[str] = list(st.session_state.get("compare_models", []))
    options_for_widget = MODEL_LABELS if len(selected_now) < 2 else selected_now
    st.multiselect(
        label="", options=options_for_widget, default=selected_now,
        key="compare_models", label_visibility="collapsed",
        placeholder="Select 2 models",
    )
    if len(st.session_state.get("compare_models", [])) > 2:
        st.session_state.compare_models = st.session_state.compare_models[:2]

# ──────────────────────────
# Controls change detector
# ──────────────────────────
def _controls_signature():
    compare_now = tuple(sorted(st.session_state.get("compare_models", [])))
    return {"dataset": dataset, "task": task, "compare": compare_now}

_cur_sig = _controls_signature()
_prev_sig = st.session_state.get("_controls_prev_sig")
if _prev_sig != _cur_sig:
    st.session_state["_controls_prev_sig"] = _cur_sig
    st.session_state.pop("last_run", None)
    st.session_state.pop("_last_metrics", None)

# ──────────────────────────
# Upload
# ──────────────────────────
TMP_DIR = "tmp_uploads"
os.makedirs(TMP_DIR, exist_ok=True)

uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])
image_path_abs: Optional[str] = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    tmp_path = os.path.join(TMP_DIR, uploaded_file.name)
    image.save(tmp_path)
    image_path_abs = os.path.abspath(tmp_path)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    new_sig = (os.path.basename(image_path_abs), os.path.getsize(image_path_abs))
    prev_sig = st.session_state.get("_image_sig")
    if prev_sig != new_sig:
        st.session_state["_image_sig"] = new_sig
        st.session_state.pop("last_run", None)
        st.session_state.pop("_last_metrics", None)
else:
    if st.session_state.get("_image_sig") is not None:
        st.session_state["_image_sig"] = None
        st.session_state.pop("last_run", None)
        st.session_state.pop("_last_metrics", None)


st.markdown("---")

# ──────────────────────────
# Labeled inputs
# ──────────────────────────
caption: Optional[str] = None
question: Optional[str] = None
expected_answer: Optional[str] = None

if dataset == "LABELED":
    if task == "captioning":
        st.markdown("<div class='x-big-label'>📝 Ground truth caption</div>", unsafe_allow_html=True)
        caption = st.text_area(
            label="",
            placeholder=(
                "Describe the image in detail, including all visible objects, people, background, and context "
                "(e.g., A brown dog running on green grass with a red ball). This ensures metrics are accurate."
            ),
            label_visibility="collapsed"
        )
    elif task == "vqa":
        st.markdown("<div class='x-big-label'>❓ Question</div>", unsafe_allow_html=True)
        question = st.text_input(label="", placeholder="e.g., What is the boy doing?", label_visibility="collapsed")
        st.markdown("<div class='x-big-label'>✅ Expected answer</div>", unsafe_allow_html=True)
        expected_answer = st.text_input(label="", placeholder="e.g., Running", label_visibility="collapsed")
    elif task == "contextual_relevance":
        st.markdown("<div class='x-big-label'>📝 Ground truth caption</div>", unsafe_allow_html=True)
        caption = st.text_area(
            label="",
            placeholder=(
                "Describe the image in detail, including all visible objects, people, background, and context "
                "(e.g., A brown dog running on green grass with a red ball). This ensures metrics are accurate."
            ),
            label_visibility="collapsed"
        )
        question = caption

st.markdown("---")

# ──────────────────────────
# Param models
# ──────────────────────────
temperature = 0.4
max_tokens = 120
top_p = 0.9

def _load_model_wrapper_safe(mkey: str, lab: str):
    try:
        if mkey == "qwen":
            with st.spinner("Preparing Qwen (auto-install if missing)…"):
                return get_model_wrapper(mkey)
        return get_model_wrapper(mkey)
    except Exception as e:
        st.error(f"Failed to initialize {lab}: {e}", icon="⚠️")
        raise

_rendered_in_this_run = False

def _score_for_unlabeled_compare(judge_notes: dict) -> float:
    def _to01(x, default=None):
        try:
            v = float(x)
            v = max(0.0, min(1.0, v))
            return float(f"{v:.3f}")
        except:
            return default

    if not isinstance(judge_notes, dict):
        return 0.0

    comp_explicit = _to01(judge_notes.get("composite_score"), None)
    if comp_explicit is not None:
        return comp_explicit

    ig = _to01(judge_notes.get("image_grounding"), 0.0)
    cv = _to01(judge_notes.get("coverage"), 0.0)
    fl = _to01(judge_notes.get("fluency"), 0.0)
    hr = _to01(judge_notes.get("hallucination_raw"), 1.0)
    comp = 0.40*ig + 0.30*cv + 0.20*fl + 0.10*hr
    comp = max(0.0, min(1.0, comp))
    return float(f"{comp:.3f}")

# ──────────────────────────
# Start
# ──────────────────────────
if st.button("Run Evaluation", type="primary"):
    if not image_path_abs:
        st.error("Please upload an image."); st.stop()
    if dataset == "LABELED":
        if task == "captioning" and not caption: st.error("Please provide a reference caption."); st.stop()
        if task == "vqa" and (not question or not expected_answer): st.error("Please provide both question and expected answer."); st.stop()
        if task == "contextual_relevance" and not caption: st.error("Please provide a reference caption."); st.stop()

    compare_sel = list(st.session_state.get("compare_models", []))
    if len(compare_sel) == 1:
        st.error("Please choose 2 models in Model Comparison to run side-by-side.", icon="⚠️")
        st.stop()

    active_labels = compare_sel if len(compare_sel) == 2 else [model_label]
    active_models = [(lab, MODEL_KEY_BY_LABEL[lab]) for lab in active_labels]
    cols = st.columns(len(active_models)) if len(active_models) == 2 else [st.container()]

    results_to_save: List[dict] = []
    golden_metrics_per_col: List[dict] = []
    judge_payloads_per_col: List[dict] = []

    for (lab, mkey), col in zip(active_models, cols):
        with col:
            sample = Sample(
                id=f"user_input_{mkey}",
                task=task,
                image=image_path_abs,
                generated_answer=None,
                question=question,
                expected_answer=expected_answer,
                caption=caption,
            )

            # 1) Model
            try:
                model = _load_model_wrapper_safe(mkey, lab)
            except Exception as e:
                st.error(f"{lab} init failed: {e}", icon="⚠️")
                results_to_save.append({"model": lab, "output": "", "error": str(e)})
                if dataset == "LABELED":
                    golden_metrics_per_col.append({m: 0.0 for m in TASK_METRICS.get(task, [])} | {"composite_score": 0.0})
                else:
                    judge_payloads_per_col.append({"judge_notes": {"error": str(e)}})
                continue

            # 2) Generator
            prompt = build_prompt(task, question, mkey)
            try:
                with st.spinner(f"Running {lab}..."):
                    model_output_raw = model.generate(
                        prompt=prompt,
                        image_path=image_path_abs,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                sample.generated_answer = sample.generated_caption = sample.model_output = model_output_raw

                display_output = enforce_n_sentences(model_output_raw, 1 if task == "vqa" else 2)
                st.markdown(f"### 🤖 {lab} Output")
                if str(display_output).strip():
                    st.success(display_output)
                else:
                    st.info("(empty output)")

            except Exception as e:
                st.error(f"{lab} generation failed: {e}", icon="⚠️")
                sample.generated_answer = sample.generated_caption = sample.model_output = ""
                results_to_save.append({
                    "model": lab,
                    "output": "",
                    "error": f"generation failed: {e}",
                })
                if dataset == "LABELED":
                    golden_metrics_per_col.append({m: 0.0 for m in TASK_METRICS.get(task, [])} | {"composite_score": 0.0})
                else:
                    judge_payloads_per_col.append({"judge_notes": {"error": str(e)}})
                continue

            # 3) Judge
            if dataset == "LABELED":
                try:
                    with st.spinner("Scoring..."):
                        metrics_dict = evaluate_golden(sample, task_type=task)
                    golden_metrics_per_col.append(metrics_dict)
                    results_to_save.append({
                        "model": lab,
                        "output": sample.model_output,
                        "metrics": metrics_dict,
                    })
                except Exception as e:
                    st.error(f"Scoring failed: {e}", icon="⚠️")
                    metrics_dict = {m: 0.0 for m in TASK_METRICS.get(task, [])}
                    metrics_dict["composite_score"] = 0.0
                    golden_metrics_per_col.append(metrics_dict)
                    results_to_save.append({
                        "model": lab,
                        "output": sample.model_output,
                        "metrics": metrics_dict,
                        "error": f"scoring failed: {e}",
                    })
            else:
                try:
                    with st.spinner("Judging with LLM..."):
                        judge = judge_unlabeled(sample, task_type=task, image_path_for_judge=image_path_abs)
                    judge_payloads_per_col.append(judge)
                    results_to_save.append({
                        "model": lab,
                        "output": sample.model_output,
                        "judge": judge,
                    })
                except Exception as e:
                    st.error(f"Judge failed: {e}", icon="⚠️")
                    judge_payloads_per_col.append({"judge_notes": {"error": str(e)}})
                    results_to_save.append({
                        "model": lab,
                        "output": sample.model_output,
                        "judge": {"error": str(e)},
                    })

    # 4) Summary / Judge Summary
    from ui_styles import inject_compare_css
    inject_compare_css()

    if dataset == "LABELED":
        cols_summary = st.columns(len(active_models)) if len(active_models) == 2 else [st.container()]
        for idx, col in enumerate(cols_summary):
            with col:
                data = (golden_metrics_per_col[idx] if idx < len(golden_metrics_per_col) else
                        {m: 0.0 for m in TASK_METRICS.get(task, [])} | {"composite_score": 0.0})
                render_summary_minimal(task, data)
        if len(active_models) == 2 and len(golden_metrics_per_col) >= 2:
            name0, name1 = active_models[0][0], active_models[1][0]
            s0 = float(golden_metrics_per_col[0].get("composite_score", 0.0))
            s1 = float(golden_metrics_per_col[1].get("composite_score", 0.0))
            if s0 > s1:
                html = (f"<div class='compare-result'><span class='title'>🏆 Comparison Result: {name0}</span>"
                        f"<span class='rest'>(Composite score: <b>{s0:.3f}</b> vs <b>{s1:.3f}</b>)</span></div>")
            elif s1 > s0:
                html = (f"<div class='compare-result'><span class='title'>🏆 Comparison Result: {name1}</span>"
                        f"<span class='rest'>(Composite score: <b>{s1:.3f}</b> vs <b>{s0:.3f}</b>)</span></div>")
            else:
                html = (f"<div class='compare-result'><span class='title'>🏆 Comparison Result: {name0} = {name1}</span>"
                        f"<span class='rest'>(Composite score: <b>{s0:.3f}</b>)</span></div>")
            st.markdown(html, unsafe_allow_html=True)
    else:
        cols_judge = st.columns(len(active_models)) if len(active_models) == 2 else [st.container()]
        for idx, col in enumerate(cols_judge):
            with col:
                judge_data = judge_payloads_per_col[idx] if idx < len(judge_payloads_per_col) else {}
                align_compare = (len(active_models) == 2)
                show_h = (task == "hallucination")
                render_judge_summary(judge_data, align_compare=align_compare, show_hallucinated=show_h)

        if len(active_models) == 2 and len(judge_payloads_per_col) >= 2 and task in ("captioning", "contextual_relevance", "hallucination"):
            name0, name1 = active_models[0][0], active_models[1][0]
            j0 = (judge_payloads_per_col[0].get("judge_notes") or {})
            j1 = (judge_payloads_per_col[1].get("judge_notes") or {})
            s0 = _score_for_unlabeled_compare(j0)
            s1 = _score_for_unlabeled_compare(j1)
            if s0 > s1:
                html = (f"<div class='compare-result'><span class='title'>🏆 Comparison Result: {name0}</span>"
                        f"<span class='rest'>(Composite score: <b>{s0:.3f}</b> vs <b>{s1:.3f}</b>)</span></div>")
            elif s1 > s0:
                html = (f"<div class='compare-result'><span class='title'>🏆 Comparison Result: {name1}</span>"
                        f"<span class='rest'>(Composite score: <b>{s1:.3f}</b> vs <b>{s0:.3f}</b>)</span></div>")
            else:
                html = (f"<div class='compare-result'><span class='title'>🏆 Comparison Result: {name0} = {name1}</span>"
                        f"<span class='rest'>(Composite score: <b>{s0:.3f}</b>)</span></div>")
            st.markdown(html, unsafe_allow_html=True)

    st.session_state["last_run"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "task": task,
        "selected_models": [lab for lab, _ in active_models],
        "compare": list(st.session_state.get("compare_models", [])),
        "image_path": image_path_abs,
        "results": results_to_save,
    }
    _rendered_in_this_run = True

# ──────────────────────────
# RE-RENDER last results on rerun
# ──────────────────────────
_last = st.session_state.get("last_run")
if _last and not _rendered_in_this_run:
    same_dataset = (_last.get("dataset") == dataset)
    same_task = (_last.get("task") == task)
    last_models = set(_last.get("selected_models", []))
    cur_models  = set(st.session_state.get("compare_models", [])) if st.session_state.get("compare_models") else {MODEL_LABELS[0]}

    same_image = (_last.get("image_path") == image_path_abs)
    if same_dataset and same_task and last_models == cur_models and same_image:

        _dataset = _last.get("dataset")
        _task = _last.get("task")
        _results = _last.get("results", [])

        cols = st.columns(len(_results)) if len(_results) == 2 else [st.container()]
        for res, col in zip(_results, cols):
            with col:
                mdl = res.get("model") or "Model"
                out = res.get("output", "")
                st.markdown(f"### 🤖 {mdl} Output")
                if str(out).strip():
                    shown = enforce_n_sentences(out, 1 if _task == "vqa" else 2)
                    st.success(shown)
                else:
                    st.info("(empty output)")

        if _dataset == "LABELED":
            cols_summary = st.columns(len(_results)) if len(_results) == 2 else [st.container()]
            for res, col in zip(_results, cols_summary):
                with col:
                    render_summary_minimal(_task, res.get("metrics", {}))
            if len(_results) == 2:
                m0, m1 = _results[0], _results[1]
                name0 = m0.get("model", "Model A")
                name1 = m1.get("model", "Model B")
                s0 = float((m0.get("metrics") or {}).get("composite_score", 0.0))
                s1 = float((m1.get("metrics") or {}).get("composite_score", 0.0))
                if s0 > s1:
                    _html = (f"<div class='compare-result'><span class='title'>🏆 Comparison Result: {name0}</span>"
                             f"<span class='rest'>(Composite score: <b>{s0:.3f}</b> vs <b>{s1:.3f}</b>)</span></div>")
                elif s1 > s0:
                    _html = (f"<div class='compare-result'><span class='title'>🏆 Comparison Result: {name1}</span>"
                             f"<span class='rest'>(Composite score: <b>{s1:.3f}</b> vs <b>{s0:.3f}</b>)</span></div>")
                else:
                    _html = (f"<div class='compare-result'><span class='title'>🏆 Comparison Result: {name0} = {name1}</span>"
                             f"<span class='rest'>(Composite score: <b>{s0:.3f}</b>)</span></div>")
                st.markdown(_html, unsafe_allow_html=True)
        else:
            cols_judge = st.columns(len(_results)) if len(_results) == 2 else [st.container()]
            for res, col in zip(_results, cols_judge):
                with col:
                    align_compare = (len(_results) == 2)
                    show_h = (_task == "hallucination")
                    render_judge_summary(res.get("judge", {}), align_compare=align_compare, show_hallucinated=show_h)

            if len(_results) == 2 and _task in ("captioning", "contextual_relevance", "hallucination"):
                m0, m1 = _results[0], _results[1]
                name0 = m0.get("model", "Model A")
                name1 = m1.get("model", "Model B")
                j0 = (m0.get("judge") or {}).get("judge_notes", {})
                j1 = (m1.get("judge") or {}).get("judge_notes", {})
                s0 = (lambda d: float(d.get("composite_score", 0.0)) if isinstance(d, dict) else 0.0)(j0)
                s1 = (lambda d: float(d.get("composite_score", 0.0)) if isinstance(d, dict) else 0.0)(j1)
                if s0 > s1:
                    _html = (f"<div class='compare-result'><span class='title'>🏆 Comparison Result: {name0}</span>"
                             f"<span class='rest'>(Composite score: <b>{s0:.3f}</b> vs <b>{s1:.3f}</b>)</span></div>")
                elif s1 > s0:
                    _html = (f"<div class='compare-result'><span class='title'>🏆 Comparison Result: {name1}</span>"
                             f"<span class='rest'>(Composite score: <b>{s1:.3f}</b> vs <b>{s0:.3f}</b>)</span></div>")
                else:
                    _html = (f"<div class='compare-result'><span class='title'>🏆 Comparison Result: {name0} = {name1}</span>"
                             f"<span class='rest'>(Composite score: <b>{s0:.3f}</b>)</span></div>")
                st.markdown(_html, unsafe_allow_html=True)

# ──────────────────────────
# SAVE
# ──────────────────────────
last_run = st.session_state.get("last_run")
if last_run:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"eval_{last_run['dataset'].lower()}_{last_run['task']}_{ts}.json"
    json_bytes = json.dumps(last_run, ensure_ascii=False, indent=2).encode("utf-8")

    st.markdown("""
    <style>
      div[data-testid="stDownloadButton"]{
        margin-top: 24px !important;
        margin-bottom: 100px !important;
      }
    </style>
    """, unsafe_allow_html=True)

    st.download_button(
        "Save results",
        data=json_bytes,
        file_name=default_name,
        mime="application/json",
        key="save_results_btn"
    )

    st.markdown("<div style='height:100px'></div>", unsafe_allow_html=True)

