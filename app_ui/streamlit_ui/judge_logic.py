# judge_logic.py
import re
import json
from typing import List, Optional, Any, Dict

from multimodal_eval.llm_eval.wrappers.openai_judge import call_llm_judge
from multimodal_eval.llm_eval.utils.prompts import (
    build_caption_eval_prompt,
    build_contextual_prompt,
    build_hallucination_prompt,
)

from multimodal_eval.evaluation.schema import Sample

# ──────────────────────────
# Prompt builder + sentence utils
# ──────────────────────────
def build_prompt(task: str, question: Optional[str], model_key: str) -> str:
    if task == "vqa":
        return (
            (question or "Answer the visual question concisely.") + "\n"
            "Constraints:\n- Answer in EXACTLY ONE sentence.\n- Refer only to visible evidence."
        )
    return (
        "You are a meticulous image evaluator.\n"
        "Describe exactly what is visible in the image.\n"
        "Requirements:\n"
        "- Use exactly TWO sentences.\n"
        "- Be specific about objects, attributes, colors, counts, relations, and any legible text.\n"
        "- No preamble, no speculation, no reasoning steps.\n"
        "Return exactly two sentences."
    )


_SENT_SPLIT_RE = re.compile(r'(?<=[\.\!\?…。！？])\s+')
def enforce_n_sentences(text: str, n: int) -> str:
    t = re.sub(r"\s+", " ", str(text or "").strip())
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(t) if p.strip()] if t else []
    return " ".join(parts[:n]) if len(parts) > n else (t or "")

# ──────────────────────────
# Rubrics
# ──────────────────────────
STRICT_RUBRIC = """
You are a strict grader. Calibrate scores conservatively.

Scoring policy (continuous, 0–1 with EXACTLY three decimals, e.g., 0.731):
- 0.95–1.000: flawless, complete, no hallucinations, covers all salient details explicitly grounded in the image.
- 0.85–0.94: very good; at most one minor omission; no contradictions to the image.
- 0.70–0.84: adequate; some omissions/vagueness; minor mistakes or unverified claims.
- 0.50–0.69: mediocre; notable missing details, generic phrasing, or partial mismatch.
- 0.20–0.49: poor; several errors, speculative or weak grounding.
- 0.00–0.19: fail; off-topic or hallucinated.

Guidance:
- Default to 0.70–0.84 for “good but not perfect”; DO NOT default to ≥0.85.
- Penalize generic language, hedging, and any claim not clearly verifiable from the image.
- Avoid round numbers like 0.90 or 0.95; choose a precise value with three decimals.
Output JSON with fields:
  {"score": 0.xxx, "reason": "<1–2 concise sentences explaining the score>"}
"""

CAPTION_JUDGE_RUBRIC = """
You are a strict captioning judge. Return JSON with EXACTLY three decimals.

Fields (all required):
- "image_grounding": 0.xxx
- "coverage": 0.xxx
- "fluency": 0.xxx
- "hallucination_raw": 0.xxx
- "score": 0.xxx
- "reason": "<1–2 sentences>"

Rules:
- Compare named strings literally. Penalize misspellings strictly.
- Prefer granular scores (no 0.90/0.95 anchors).
- image_grounding ~ (# correct exact items) / (# mentioned items)
- coverage ~ (# correct items mentioned) / (estimated # salient items)
- hallucination_raw ~ 1 - ( # wrong or non-existent items / max(1, # mentioned items) )

Also return:
- "objects_mentioned": [...]
"""

FREEFORM_REASON_JSON_RUBRIC = """
You are an assistant writing a concise qualitative summary of a caption's quality based on:
(1) the IMAGE, (2) the model CAPTION text, and (3) METRICS.

Write 1–2 sentences (≤ ~40 words), neutral-professional tone, no numbers or lists.
Focus on grounding, coverage, fluency, and factual consistency implied by the metrics and visible evidence.
Mention 1–2 concrete strengths and (if applicable) one shortcoming.
Avoid templated phrasing.

Return ONLY valid JSON with a single key:
{"reason": "<your 1–2 sentence summary>"}

Rules:
- The value of "reason" must be plain text (no markdown, no code fences).
- No extra keys, no trailing text before/after the JSON.
"""

# ──────────────────────────
# Normalization / Parsing and Helpers
# ──────────────────────────
def _flatten(d, parent_key="", sep="."):
    items = []
    for k, v in (d or {}).items():
        nk = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(_flatten(v, nk, sep=sep))
        else:
            items.append((nk, v))
    return items

def _find_val(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    if not isinstance(d, dict):
        return None
    lk = {k.lower(): k for k in d.keys()}
    for want in keys:
        w = want.lower()
        if w in lk:
            v = d[lk[w]]
            try:
                f = float(v)
                f = max(0.0, min(1.0, f))
                return float(f"{f:.3f}")
            except:
                try:
                    if isinstance(v, dict):
                        for sub in ("score", "value"):
                            if sub in v:
                                f = float(v[sub])
                                f = max(0.0, min(1.0, f))
                                return float(f"{f:.3f}")
                except:
                    pass
    for k, v in _flatten(d):
        tail = k.split(".")[-1].lower()
        if tail in [s.lower() for s in keys]:
            try:
                f = float(v)
                f = max(0.0, min(1.0, f))
                return float(f"{f:.3f}")
            except:
                pass
    return None

def _normalize_to_list(v) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()]
    if isinstance(v, str):
        parts = re.split(r"[;,]|(?:\s{2,})", v)
        return [p.strip() for p in parts if p and p.strip()]
    return []

def _extract_reason(notes) -> Optional[str]:
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

def _extract_hallucinated_objects(notes) -> List[str]:
    if not isinstance(notes, dict):
        return []
    for key in ("hallucinated_objects", "hallucinations", "ungrounded_objects", "objects"):
        if key in notes:
            lst = _normalize_to_list(notes.get(key))
            if lst:
                return lst[:8]
    for k, v in _flatten(notes):
        if any(tok in k.lower() for tok in ("hallucinated_objects", "hallucinations", "ungrounded_objects")):
            lst = _normalize_to_list(v)
            if lst:
                return lst[:8]
    return []

def _estimate_objects_from_text(model_text: str) -> List[str]:
    if not model_text:
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", model_text.lower())
    stop = {
        "the","a","an","and","or","with","on","in","at","of","to","from","for","by",
        "is","are","was","were","be","being","been","this","that","these","those",
        "woman","man","person","people","someone","something","it","she","he","they",
        "room","area","scene","view","background","effect"
    }
    candidates = [t for t in tokens if len(t) > 2 and t not in stop]
    uniq = []
    for t in candidates:
        if t not in uniq:
            uniq.append(t)
        if len(uniq) >= 15:
            break
    return uniq

def _filter_hallucinations_to_caption(hall: List[str], model_text: str) -> List[str]:
    cap = (model_text or "").lower()
    out = []
    for h in hall or []:
        s = str(h).strip()
        if not s:
            continue
        if s.lower() in cap:
            out.append(s)
    seen, uniq = set(), []
    for x in out:
        xl = x.lower()
        if xl not in seen:
            uniq.append(x)
            seen.add(xl)
    return uniq

def _clamp3(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except:
        v = default
    v = max(0.0, min(1.0, v))
    return float(f"{v:.3f}")

MIN_HR = 0.30
def _soft_floor(x: float, floor: float = MIN_HR) -> float:
    try:
        v = float(x)
    except:
        v = floor
    v = max(floor, min(1.0, v))
    return float(f"{v:.3f}")

def _get_ep_goodness(notes: dict, hall_list: List[str], model_text: str) -> float:
    objs = notes.get("objects_mentioned")
    if isinstance(objs, list) and objs:
        denom = max(1, len({str(x).strip().lower() for x in objs if str(x).strip()}))
        num   = len({str(x).strip().lower() for x in (hall_list or []) if str(x).strip()})
        score = 1.0 - (num / denom)
        score = max(0.0, min(1.0, score))
        return _soft_floor(score)
    est = _estimate_objects_from_text(model_text or "")
    denom = max(1, len(est))
    num   = len({str(x).strip().lower() for x in (hall_list or []) if str(x).strip()})
    score = 1.0 - (num / denom)
    score = max(0.0, min(1.0, score))
    return _soft_floor(score)

def _normalize_hallucination_notes(raw_notes, model_text: str):
    notes = raw_notes if isinstance(raw_notes, dict) else {"reason": str(raw_notes or "").strip()}
    hall_raw = _extract_hallucinated_objects(notes)
    hall = _filter_hallucinations_to_caption(hall_raw, model_text or "")
    reason = _extract_reason(notes) or ""
    judge_score = _find_val(notes if isinstance(notes, dict) else {}, ["score","overall_score","judge_score"])

    EP = _get_ep_goodness(notes if isinstance(notes, dict) else {}, hall, model_text or "")

    VA = _find_val(notes, ["visual_alignment", "visual alignment", "visual-alignment", "visualAlignment"])
    OA = _find_val(notes, ["object_agreement", "object agreement", "object-agreement", "objectAgreement"])
    AA = _find_val(notes, ["attribute_accuracy", "attribute accuracy", "attribute-accuracy", "attributeAccuracy"])

    base = judge_score if isinstance(judge_score, (int, float)) else 0.5
    if VA is None:
        VA = 0.6*EP + 0.4*base
    if OA is None:
        OA = 0.7*EP + 0.3*(VA if isinstance(VA, (int, float)) else 0.5)
    if AA is None:
        AA = 0.5*(VA if isinstance(VA, (int, float)) else 0.5) + 0.5*EP

    VA = _clamp3(VA, 0.5)
    OA = _clamp3(OA, 0.5)
    AA = _clamp3(AA, 0.5)
    EP = _soft_floor(EP)

    composite = 0.30*VA + 0.30*OA + 0.20*AA + 0.20*EP
    composite = _clamp3(composite)

    out = dict(notes) if isinstance(notes, dict) else {}
    out.update({
        "visual_alignment": VA,
        "object_agreement": OA,
        "attribute_accuracy": AA,
        "extraneous_penalty": EP,
        "score": judge_score if isinstance(judge_score, (int,float)) else composite,
        "reason": reason,
        "hallucinated_objects": hall,
        "objects_mentioned": out.get("objects_mentioned", notes.get("objects_mentioned", []))
    })
    return out

def _compute_fluency_proxy(text: str) -> float:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return 0.30
    letters = sum(ch.isalpha() for ch in t)
    ratio_letters = letters / max(1, len(t))
    tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", t)
    avg_len = (sum(len(x) for x in tokens) / max(1, len(tokens))) if tokens else 0
    bad_punct = 1 if re.search(r"[!?]{3,}|\.{4,}|[,;:]{3,}", t) else 0
    score = 0.55 * min(1.0, max(0.0, ratio_letters)) + 0.35 * min(1.0, avg_len / 7.0) - 0.20 * bad_punct
    score = max(0.0, min(1.0, score))
    return float(f"{score:.3f}")

def _normalize_contextual_relations(notes: dict, model_text: str) -> dict:
    out = dict(notes) if isinstance(notes, dict) else {}

    ig = _find_val(out, ["image_grounding", "imageGrounding", "grounding"])
    cv = _find_val(out, ["coverage"])
    fl = _find_val(out, ["fluency"])
    hr = _find_val(out, ["hallucination_raw", "no_hallucination", "noHallucination"])

    pa_key_candidates = [
        "prompt_alignment",
        "instruction_following",
        "prompt adherence",
        "adherence",
        "promptFollowing",
        "prompt-following",
    ]
    pa = _find_val(out, pa_key_candidates)

    if ig is not None:
        out["image_grounding"] = _clamp3(ig)
    if cv is not None:
        out["coverage"] = _clamp3(cv)
    if fl is not None:
        out["fluency"] = _clamp3(fl)
    if hr is not None:
        out["hallucination_raw"] = _clamp3(hr)

    if "coverage" not in out and pa is not None:
        out["coverage"] = _clamp3(pa)

    if "fluency" not in out:
        out["fluency"] = _compute_fluency_proxy(model_text or "")

    for k in pa_key_candidates:
        if k in out:
            out.pop(k, None)

    return out

VERIFICATION_RUBRIC = """
You are an image FACT VERIFIER. Compare the given CAPTION (model output) to the IMAGE.
Return ONLY JSON with these fields:

{
  "present": ["<caption phrases that ARE visible on screen>", ...],
  "absent":  ["<caption phrases that are NOT visible on screen>", ...],
  "typos":   ["<phrases that are misspelled vs visible UI text>", ...]
}
"""

def _second_pass_verify(image_path: str, model_text: str, task_type: str) -> dict:
    try:
        user_prompt = (
            "CAPTION:\n" + str(model_text) +
            "\n\nExtract literal UI words/labels/names mentioned in the caption and verify them on the image."
        )
        sys_prompt = VERIFICATION_RUBRIC
        notes = call_llm_judge("hallucination", image_path=image_path, prompt=user_prompt, system_prompt=sys_prompt)
        if isinstance(notes, dict):
            return {
                "present": _normalize_to_list(notes.get("present", []))[:10],
                "absent":  _normalize_to_list(notes.get("absent",  []))[:10],
                "typos":   _normalize_to_list(notes.get("typos",   []))[:10],
            }
        return {}
    except Exception:
        return {}

def generate_freeform_reason(
    image_path: str,
    model_text: str,
    metrics: Dict[str, float],
    correct_mentions: Optional[List[str]] = None,
) -> str:
    safe_metrics = {k: float(f"{float(v):.3f}") for k, v in (metrics or {}).items() if isinstance(v, (int, float))}
    present = ", ".join([m for m in (correct_mentions or []) if str(m).strip()][:6])

    user_prompt = (
        "CAPTION (model output):\n"
        f"{(model_text or '').strip()}\n\n"
        "METRICS (0..1):\n"
        f"{json.dumps(safe_metrics, ensure_ascii=False)}\n"
        f"{'PRESENT MENTIONS: ' + present + '\\n' if present else ''}"
        "Return ONLY the JSON object as specified."
    )

    try:
        out = call_llm_judge(
            "captioning",
            image_path=image_path,
            prompt=user_prompt,
            system_prompt=FREEFORM_REASON_JSON_RUBRIC,
        )
        if isinstance(out, dict) and isinstance(out.get("reason"), str):
            text = out["reason"]
        elif isinstance(out, str):
            m = re.search(r"\{.*\}", out, flags=re.DOTALL)
            if m:
                try:
                    jt = json.loads(m.group(0))
                    text = jt.get("reason", out)
                except Exception:
                    text = out
            else:
                text = out
        else:
            text = ""

        text = re.sub(r"\s+", " ", str(text or "").strip())
        return enforce_n_sentences(text, 2)

    except Exception:
        parts = []
        if safe_metrics.get("image_grounding", 0) >= 0.7: parts.append("The caption stays close to what is visible")
        if safe_metrics.get("coverage", 0) >= 0.7: parts.append("and covers the key elements")
        if safe_metrics.get("fluency", 0) >= 0.7: parts.append("while reading smoothly")
        s1 = " ".join(parts) or "The caption clearly describes the scene"
        downs = []
        if safe_metrics.get("image_grounding", 1) < 0.6: downs.append("some details may not fully match the image")
        if safe_metrics.get("coverage", 1) < 0.6: downs.append("important elements could be elaborated")
        if downs:
            s2 = "However, " + " and ".join(downs) + "."
            return enforce_n_sentences(f"{s1}. {s2}", 2)
        return enforce_n_sentences(f"{s1}.", 2)

def jitter_close_vals(x: float, eps: float = 0.004) -> float:
    try:
        v = float(x)
    except:
        return x
    anchors = [0.850, 0.900, 0.950]
    for a in anchors:
        if abs(v - a) < 0.006:
            v = a + (eps if v >= a else -eps)
    v = max(0.0, min(1.0, v))
    return float(f"{v:.3f}")

def promote_and_strip_score(x):
    if isinstance(x, dict):
        if "score" in x and "composite_score" not in x:
            try:
                x["composite_score"] = float(x["score"])
            except Exception:
                pass
        x.pop("score", None)
        for k in list(x.keys()):
            promote_and_strip_score(x[k])
    elif isinstance(x, list):
        for e in x:
            promote_and_strip_score(e)

def judge_unlabeled(sample: Sample, task_type: str, image_path_for_judge: str) -> dict:
    try:
        if not getattr(sample, "generated_answer", None):
            return {"warning": "Empty generated answer — nothing to judge."}

        if task_type == "captioning":
            up, _ = build_caption_eval_prompt(sample.generated_answer)
            sp = CAPTION_JUDGE_RUBRIC
            notes = call_llm_judge(task_type, image_path=image_path_for_judge, prompt=up, system_prompt=sp)

            if isinstance(notes, dict):
                notes = _normalize_contextual_relations(notes, sample.generated_answer)

                hall_raw = _extract_hallucinated_objects(notes)
                hall = _filter_hallucinations_to_caption(hall_raw, sample.generated_answer)
                notes["hallucinated_objects"] = hall
                hr_base = _get_ep_goodness(notes, hall, sample.generated_answer)
                notes["hallucination_raw"] = _soft_floor(hr_base)

                for fld in ("image_grounding", "coverage", "fluency", "hallucination_raw"):
                    if isinstance(notes.get(fld), (int, float)):
                        notes[fld] = jitter_close_vals(notes[fld])

                ver = _second_pass_verify(image_path_for_judge, sample.generated_answer, "captioning")
                try:
                    hr_now = float(notes.get("hallucination_raw", 1.0))
                except Exception:
                    hr_now = 1.0
                absent = (ver or {}).get("absent", []) if isinstance(ver, dict) else []
                typos  = (ver or {}).get("typos", [])  if isinstance(ver, dict) else []
                violations = len([x for x in (absent or []) if str(x).strip()]) + len([x for x in (typos or []) if str(x).strip()])
                denom = max(1, len(notes.get("objects_mentioned", [])) or 0)
                hr_from_ver = 1.0 - (violations / denom)
                hr_from_ver = max(0.0, min(1.0, hr_from_ver))
                rtext_prev = _extract_reason(notes) or ""
                mismatch_hint = any(tok in (rtext_prev.lower()) for tok in ["not visible","mismatch","wrong","absent","unsupported","hallucination"])
                hr_blend = 0.7*hr_now + 0.3*hr_from_ver - (0.05 if mismatch_hint else 0.0)
                notes["hallucination_raw"] = _soft_floor(hr_blend)

                def _to01(x, default=0.0):
                    try:
                        v = float(x); v = max(0.0, min(1.0, v)); return float(f"{v:.3f}")
                    except:
                        return default
                ig = _to01(notes.get("image_grounding"), 0.0)
                cv = _to01(notes.get("coverage"), 0.0)
                fl = _to01(notes.get("fluency"), 0.0)
                hr = _to01(notes.get("hallucination_raw"), 1.0)

                comp = 0.40*ig + 0.30*cv + 0.20*fl + 0.10*hr
                comp = max(0.0, min(1.0, comp))
                notes["composite_score"] = float(f"{comp:.3f}")

                ver = ver or {}
                correct_mentions = ver.get("present") if isinstance(ver, dict) else []
                metrics_for_reason = {
                    "composite_score": notes.get("composite_score", comp),
                    "image_grounding": ig,
                    "coverage": cv,
                    "fluency": fl,
                    "hallucination_raw": hr,
                }
                notes["reason"] = generate_freeform_reason(
                    image_path_for_judge, sample.generated_answer, metrics_for_reason, correct_mentions=correct_mentions
                )

                promote_and_strip_score(notes)

            return {"judge_notes": notes}

        elif task_type == "contextual_relevance":
            up, sp0 = build_contextual_prompt("Describe the image.", sample.generated_answer)
            sp = (sp0 or "") + "\n\n" + STRICT_RUBRIC
            notes = call_llm_judge(task_type, image_path=image_path_for_judge, prompt=up, system_prompt=sp)

            if isinstance(notes, dict):
                notes = _normalize_contextual_relations(notes, sample.generated_answer)

                ver = _second_pass_verify(image_path_for_judge, sample.generated_answer, "contextual_relevance")
                if "hallucination_raw" in notes:
                    try:
                        hr_now = float(notes.get("hallucination_raw", 1.0))
                    except Exception:
                        hr_now = 1.0
                    absent = (ver or {}).get("absent", []) if isinstance(ver, dict) else []
                    typos  = (ver or {}).get("typos", [])  if isinstance(ver, dict) else []
                    violations = len([x for x in (absent or []) if str(x).strip()]) + len([x for x in (typos or []) if str(x).strip()])
                    denom = max(1, len(notes.get("objects_mentioned", [])) or 0)
                    hr_from_ver = 1.0 - (violations / denom)
                    hr_from_ver = max(0.0, min(1.0, hr_from_ver))
                    rtext_prev = _extract_reason(notes) or ""
                    mismatch_hint = any(tok in (rtext_prev.lower()) for tok in ["not visible","mismatch","wrong","absent","unsupported","hallucination"])
                    hr_blend = 0.7*hr_now + 0.3*hr_from_ver - (0.05 if mismatch_hint else 0.0)
                    notes["hallucination_raw"] = _clamp3(hr_blend)

                if isinstance(notes.get("coverage"), (int,float)) and isinstance(notes.get("fluency"), (int,float)):
                    if abs(float(notes["coverage"]) - float(notes["fluency"])) < 1e-6:
                        notes["fluency"] = _clamp3(float(notes["fluency"]) - 0.003)

                def _to01(x, default=0.0):
                    try:
                        v = float(x); v = max(0.0, min(1.0, v)); return float(f"{v:.3f}")
                    except:
                        return default
                ig = _to01(notes.get("image_grounding"), 0.0)
                cv = _to01(notes.get("coverage"), 0.0)
                fl = _to01(notes.get("fluency"), 0.0)
                hr = _to01(notes.get("hallucination_raw"), 1.0)
                comp = 0.40*ig + 0.30*cv + 0.20*fl + 0.10*hr
                comp = max(0.0, min(1.0, comp))
                notes["composite_score"] = float(f"{comp:.3f}")

                ver = ver or {}
                correct_mentions = ver.get("present") if isinstance(ver, dict) else []
                metrics_for_reason = {
                    "composite_score": notes.get("composite_score", comp),
                    "image_grounding": ig,
                    "coverage": cv,
                    "fluency": fl,
                    "hallucination_raw": hr,
                }
                notes["reason"] = generate_freeform_reason(
                    image_path_for_judge, sample.generated_answer, metrics_for_reason, correct_mentions=correct_mentions
                )

                promote_and_strip_score(notes)

            return {"judge_notes": notes}

        elif task_type == "hallucination":
            up, sp0 = build_hallucination_prompt(sample.generated_answer, ground_truth_objects=[])
            sp = (sp0 or "") + "\n\n" + STRICT_RUBRIC
            raw = call_llm_judge(task_type, image_path=image_path_for_judge, prompt=up, system_prompt=sp)
            notes = _normalize_hallucination_notes(raw, sample.generated_answer)

            required = {"visual_alignment","object_agreement","attribute_accuracy","extraneous_penalty","hallucinated_objects"}
            if not (isinstance(notes, dict) and required.issubset(set(notes.keys()))):
                raw2 = call_llm_judge(task_type, image_path_for_judge, prompt=up, system_prompt=sp)
                notes = _normalize_hallucination_notes(raw2, sample.generated_answer)

            ver = _second_pass_verify(image_path_for_judge, sample.generated_answer, "hallucination")
            try:
                ep_now = float(notes.get("extraneous_penalty", 1.0))
            except Exception:
                ep_now = 1.0
            absent = (ver or {}).get("absent", []) if isinstance(ver, dict) else []
            typos  = (ver or {}).get("typos", [])  if isinstance(ver, dict) else []
            violations = len([x for x in (absent or []) if str(x).strip()]) + len([x for x in (typos or []) if str(x).strip()])
            denom = max(1, len(notes.get("objects_mentioned", [])) or 0)
            ep_from_ver = 1.0 - (violations / denom)
            ep_from_ver = max(0.0, min(1.0, ep_from_ver))
            rtext_prev = _extract_reason(notes) or ""
            mismatch_hint = any(tok in (rtext_prev.lower()) for tok in ["not visible","mismatch","wrong","absent","unsupported","hallucination"])
            ep_blend = 0.7*ep_now + 0.3*ep_from_ver - (0.05 if mismatch_hint else 0.0)
            notes["extraneous_penalty"] = _soft_floor(ep_blend)

            def _to01(x, default=0.0):
                try:
                    v = float(x); v = max(0.0, min(1.0, v)); return float(f"{v:.3f}")
                except:
                    return default
            VA = _to01(notes.get("visual_alignment"), 0.5)
            OA = _to01(notes.get("object_agreement"), 0.5)
            AA = _to01(notes.get("attribute_accuracy"), 0.5)
            EP = _to01(notes.get("extraneous_penalty"), 0.5)
            comp = 0.30*VA + 0.30*OA + 0.20*AA + 0.20*EP
            comp = max(0.0, min(1.0, comp))
            notes["composite_score"] = float(f"{comp:.3f}")

            ver = ver or {}
            correct_mentions = ver.get("present") if isinstance(ver, dict) else []
            metrics_for_reason = {
                "composite_score": notes.get("composite_score", comp),
                "image_grounding": VA,
                "coverage": OA,
                "fluency": AA,
                "hallucination_raw": EP,
            }
            notes["reason"] = generate_freeform_reason(
                image_path_for_judge, sample.generated_answer, metrics_for_reason, correct_mentions=correct_mentions
            )

            promote_and_strip_score(notes)

            return {"judge_notes": notes}

        else:
            return {"judge_notes": {"info": "Task not supported for UNLABELED."}}

    except Exception as e:
        return {"error": f"Judge failed: {e}"}