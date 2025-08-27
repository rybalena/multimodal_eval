# renderers.py
import numbers
import streamlit as st
from typing import Dict

from ui_styles import metric_label_with_tooltip, bar_line

def val_for_ui(metric_key: str, val: float) -> float:
    try:
        v = float(val or 0.0)
    except Exception:
        v = 0.0
    if metric_key == "cider":
        return round(min(v / 10.0, 1.0), 3) if v > 1.0 else round(v, 3)
    return round(v, 3)

def render_summary_minimal(task_type: str, data: dict):
    st.markdown("<h3>📈 Evaluation Summary</h3>", unsafe_allow_html=True)
    if task_type == "captioning":
        fields = [("CIDEr","cider"), ("CLIP","clip_score"), ("Semantic","semantic_similarity"), ("Composite score","composite_score")]
    elif task_type == "vqa":
        fields = [("CLIP","clip_score"), ("Semantic","semantic_similarity"), ("Context","contextual_relevance"), ("Composite score","composite_score")]
    else:
        fields = [("CLIP","clip_score"), ("CIDEr","cider"), ("Semantic","semantic_similarity"), ("Composite score","composite_score")]
    for nice, key in fields:
        raw_v = data.get(key, 0.0)
        ui_v  = val_for_ui(key, raw_v)
        label_html = metric_label_with_tooltip(nice, key)
        bar_line(label_html, ui_v)

def _reason_box(text: str, align_compare: bool) -> str:
    import html
    safe = html.escape(text)
    inner_style = "background:#f3f4f6;color:#111827;border:none;border-radius:12px;padding:12px 14px;font-size:15px;line-height:1.5;margin-top:18px;"
    if align_compare:
        return (
            "<div style='max-width:720px;margin:0;'>"
            "  <div style='width:76%;max-width:420px;'>"
            f"    <div style='{inner_style}'><b>Reason:</b> {safe}</div>"
            "  </div>"
            "</div>"
        )
    return f"<div style='{inner_style}'><b>Reason:</b> {safe}</div>"

def _flatten(d, parent_key="", sep="."):
    items = []
    for k, v in (d or {}).items():
        nk = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(_flatten(v, nk, sep=sep))
        else:
            items.append((nk, v))
    return items

def _extract_reason(notes):
    if isinstance(notes, str):
        return notes.strip() or None
    if not isinstance(notes, dict):
        return None
    for key in ("reason","Reason","REASON","analysis","rationale","explanation","justification"):
        v = notes.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k, v in _flatten(notes):
        lk = k.lower()
        if any(tok in lk for tok in ("reason","analysis","rationale","explanation","justification")):
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None

def render_judge_summary(judge: dict, align_compare: bool = False, show_hallucinated: bool = False):
    st.markdown("<h3>🧑‍⚖️ Judge Summary</h3>", unsafe_allow_html=True)
    notes = judge.get("judge_notes", judge)

    if show_hallucinated and isinstance(notes, dict) and any(k in notes for k in ("visual_alignment","object_agreement","attribute_accuracy","extraneous_penalty","composite_score")):
        fields = [
            ("Visual Alignment", "visual_alignment"),
            ("Object Agreement", "object_agreement"),
            ("Attribute Accuracy", "attribute_accuracy"),
            ("Ungrounded Share", "extraneous_penalty"),
            ("Composite score", "composite_score"),
        ]
        for nice, key in fields:
            v = notes.get(key, 0.0)
            label_html = metric_label_with_tooltip(nice, key)
            bar_line(label_html, val_for_ui(key, v))

        reason_text = _extract_reason(notes)
        if isinstance(reason_text, str) and reason_text.strip():
            from judge_logic import enforce_n_sentences
            reason_text = enforce_n_sentences(reason_text, 2)
            st.markdown(_reason_box(reason_text, align_compare), unsafe_allow_html=True)
        return

    if isinstance(notes, dict):
        ordered_keys = None
        if any(k in notes for k in ("coverage","fluency","hallucination_raw")):
            ordered_keys = []
            if "image_grounding" in notes:
                ordered_keys.append("image_grounding")
            ordered_keys += ["coverage", "fluency", "hallucination_raw"]
            if "composite_score" in notes:
                ordered_keys.append("composite_score")

        if ordered_keys:
            for key in ordered_keys:
                if key in notes and isinstance(notes[key], numbers.Number):
                    nice = key.replace("_"," ").title()
                    label_html = metric_label_with_tooltip(nice, key)
                    bar_line(label_html, val_for_ui(key, notes[key]))
        else:
            pairs = []
            if isinstance(notes, dict):
                for k, v in _flatten(notes):
                    if isinstance(v, numbers.Number):
                        f = float(v);  f = f/100.0 if (1.0 < f <= 100.0) else f
                        if 0.0 <= f <= 1.0:
                            tail = k.split(".")[-1].lower()
                            if tail == "score":
                                continue
                            title = tail.replace("_"," ").title()
                            pairs.append((title, round(f, 3), tail))
            for nice, val, key in pairs[:6]:
                label_html = metric_label_with_tooltip(nice, key)
                bar_line(label_html, val)

    reason_text = _extract_reason(notes)
    if isinstance(reason_text, str) and reason_text.strip():
        from judge_logic import enforce_n_sentences
        reason_text = enforce_n_sentences(reason_text, 2)
        st.markdown(_reason_box(reason_text, align_compare), unsafe_allow_html=True)
