# ui_styles.py
import html
import streamlit as st

# ──────────────────────────
# CSS
# ──────────────────────────
def inject_global_labels_css():
    st.markdown(
        """
<style>
.x-big-label{
  font-size: 18px !important;
  font-weight: 400 !important;
  line-height: 1.28 !important;
  letter-spacing: .1px !important;
  margin: 2px 0 6px 0 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

# ──────────────────────────
# Tooltip CSS
# ──────────────────────────
def inject_tooltip_css():
    st.markdown("""
<style>
.metric-label-wrap{ position: relative; display: inline-block; }
.metric-title{ font-size:15px; color:#0f172a; padding-right: 16px; }
.metric-label-wrap .tooltip-q{
  position:absolute; right:0; top:4px; width:12px; height:12px;
  display:inline-flex; align-items:center; justify-content:center;
  border-radius:999px; font-size:9px; font-weight:700; line-height:1;
  background:#eef2ff; color:#4338ca; border:1px solid #c7d2fe;
  cursor:default; user-select:none;
}
.metric-label-wrap .tooltip-box{
  position:absolute; top:-8px; left:calc(100% + 8px);
  min-width:220px; max-width:320px; background:#111827; color:#f9fafb;
  font-size:12px; line-height:1.35; border-radius:10px; padding:10px 12px;
  box-shadow:0 8px 24px rgba(0,0,0,.22); border:1px solid rgba(255,255,255,.08);
  opacity:0; pointer-events:none; transform:translateY(4px);
  transition:opacity .12s ease, transform .12s ease; z-index:9999;
}
.metric-label-wrap:hover .tooltip-box{ opacity:1; pointer-events:auto; transform:translateY(0); }
@media (max-width: 640px){ .metric-label-wrap .tooltip-box{ left:0; top:22px; } }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────
# Sidebar CSS
# ──────────────────────────
def inject_sidebar_css():
    st.markdown(
        """
<style>
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap:0!important; margin:0!important; padding:0!important; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]{ margin:0!important; padding:0!important; }
section[data-testid="stSidebar"] h2.controls-title{ font-size:20px!important; font-weight:600!important; margin:0!important; line-height:1.15!important; }
section[data-testid="stSidebar"] h2.section-title{ font-size:16px!important; font-weight:500!important; margin:0 0 0px 0!important; line-height:1.2!important; }
section[data-testid="stSidebar"] .element-container,
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div,
section[data-testid="stSidebar"] [data-testid^="stWidget"]{ margin:0!important; padding:0!important; }
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stMultiSelect{ margin:0!important; padding:0!important; }
section[data-testid="stSidebar"] div[data-baseweb="select"]{ margin:0!important; }
section[data-testid="stSidebar"] .stSelectbox > label,
section[data-testid="stSidebar"] .stMultiSelect > label{ display:none!important; height:0!important; margin:0!important; padding:0!important; }
section[data-testid="stSidebar"] .stMultiSelect [aria-label="Clear all"]{ display:none!important; }
div[data-testid="stAppViewContainer"] .stTextArea > label,
div[data-testid="stAppViewContainer"] .stTextInput > label{ font-size:16px!important; font-weight:600!important; }
</style>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────
# Metric tooltips + label builder
# ──────────────────────────

METRIC_TOOLTIPS = {
    # LABELED metrics
    "cider": "Similarity to reference captions using TF-IDF n-grams (higher is better).",
    "clip_score": "Image–text alignment score from CLIP-like models (0–1).",
    "semantic_similarity": "Semantic similarity to the reference text (0–1).",
    "contextual_relevance": "Answer relevance to the image and question (0–1).",
    "composite_score": "Task-specific weighted blend of component metrics (0–1).",

    # UNLABELED captioning / contextual
    "image_grounding": "How well the caption’s mentions are grounded in visible evidence (0–1).",
    "coverage": "Share of salient elements correctly mentioned in the caption (0–1).",
    "fluency": "Grammar and readability of the caption (0–1).",
    "hallucination_raw": "Consistency proxy: 1 − fraction of ungrounded/incorrect mentions (0–1).",

    # UNLABELED hallucination
    "visual_alignment": "Overall agreement of the caption with the image (0–1).",
    "object_agreement": "Whether mentioned objects are actually visible (0–1).",
    "attribute_accuracy": "Correctness of attributes: color, count, shape, material (0–1).",
    "extraneous_penalty": "Ungrounded share inverse: higher means fewer unsupported mentions (0–1).",

    # Helpers
    "prompt_alignment": "How closely the output follows the instruction constraints (0–1).",
    "instruction_following": "Adherence to the instruction/prompt (0–1).",
    "adherence": "General instruction adherence (0–1).",
    "no_hallucination": "Judge’s raw estimate of being hallucination-free (0–1).",

    # Verification
    "present": "Phrases from the caption that are confirmed as visible in the image.",
    "absent": "Phrases from the caption that are NOT visible in the image.",
    "typos": "Misspellings vs visible on-screen UI/text.",
}

def metric_label_with_tooltip(nice_label: str, metric_key: str) -> str:
    tip = METRIC_TOOLTIPS.get(metric_key, "")
    safe_nice = html.escape(nice_label)
    safe_tip  = html.escape(tip)
    return (
        f"<span class='metric-label-wrap'>"
        f"<span class='metric-title'>📏 {safe_nice}</span>"
        f"<span class='tooltip-q' aria-label='help' title=''>?</span>"
        f"<span class='tooltip-box' role='tooltip'>{safe_tip}</span>"
        f"</span>"
    )

# ──────────────────────────
# Compare-result CSS
# ──────────────────────────
def inject_compare_css():
    st.markdown("""
<style>
.compare-result{ margin-top:48px; margin-bottom:8px; }
.compare-result .title{ font-size:18px; font-weight:700; color:#0f172a; }
.compare-result .rest{ font-size:16px; font-weight:500; color:#0f172a; margin-left:6px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────
# Bars (colors/render)
# ──────────────────────────
def _status_and_colors(val):
    v = float(val) if isinstance(val, (int, float)) else 0.0
    if v < 0.40:
        return ("poor", "linear-gradient(90deg,#ff5252,#ff1744)", "0 0 8px rgba(255,23,68,.20)")
    if v <= 0.60:
        return ("moderate", "linear-gradient(90deg,#b2ebf2,#80deea)", "0 0 8px rgba(128,222,234,.25)")
    if v <= 0.75:
        return ("good", "linear-gradient(90deg,#ffcc80,#ff9800)", "0 0 8px rgba(255,152,0,.18)")
    return ("excellent", "linear-gradient(90deg,#00c853,#00e676)", "0 0 8px rgba(0,230,118,.20)")

def _badge_style(status: str) -> str:
    base = "padding:2px 8px;border-radius:999px;font-size:12px;"
    if status == "excellent":
        return base + "color:#065f46;background:#d1fae5;border:1px solid #99f6e4;"
    if status == "good":
        return base + "color:#92400e;background:#fef3c7;border:1px solid #fde68a;"
    if status == "moderate":
        return base + "color:#0f4c5c;background:#e0f7fa;border:1px solid #b2ebf2;"
    return base + "color:#991b1b;background:#fee2e2;border:1px solid #fecaca;"

def bar_line(label_html: str, value):
    v = float(value) if isinstance(value, (int, float)) else 0.0
    status, gradient, glow = _status_and_colors(v)
    fill_w = max(0, min(100, int(round(v * 100))))
    shown = f"{v:.3f}"
    percent_txt = f"{fill_w}%"
    st.markdown(
        f"<div style='max-width:720px;margin:0;'>"
        f"{label_html}"
        f"<div style='display:flex;align-items:center;gap:6px;margin-top:2px;'>"
        f"<div style='width:76%;max-width:420px;'>"
        f"<div style='width:100%;height:6px;background:#eef0f3;border:1px solid #e5e7eb;border-radius:999px;'>"
        f"<div style='width:{fill_w}%;height:100%;background:{gradient};box-shadow:{glow};border-radius:999px;'></div>"
        f"</div></div>"
        f"<div style='font-size:14px;color:#111827;font-weight:600;'>{percent_txt}</div></div>"
        f"<div style='display:flex;align-items:center;gap:8px;margin-top:1px;'>"
        f"<span style='font-size:15px;color:#0f172a;'>{shown}</span>"
        f"<span style='{_badge_style(status)}'>{status}</span></div></div>",
        unsafe_allow_html=True
    )
