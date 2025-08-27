from collections import Counter
from typing import List, Dict
import math
import re
from multimodal_eval.evaluation.schema import Sample

_TOKEN_RE = re.compile(r"\w+")

def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())

def _ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if n <= len(tokens) else []

def _tf(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values()) or 1.0
    return {k: v / total for k, v in counter.items()}

def _cosine(u: Dict[str, float], v: Dict[str, float], idf: Dict[str, float]) -> float:
    num = 0.0
    for k in u.keys() & v.keys():
        w = idf.get(k, 1.0)
        num += (u[k] * w) * (v[k] * w)

    def norm(x: Dict[str, float]) -> float:
        s = 0.0
        for k, val in x.items():
            w = idf.get(k, 1.0)
            s += (val * w) ** 2
        return math.sqrt(s) or 1e-8

    den = norm(u) * norm(v)
    return num / den if den > 0 else 0.0

def compute_cider(sample: Sample) -> float:

    reference = sample.expected_answer if sample.task == "vqa" else sample.caption
    hypothesis = sample.generated_answer
    if not reference or not hypothesis:
        return 0.0

    ref_toks = _tokens(reference)
    hyp_toks = _tokens(hypothesis)

    base_weights = {1: 0.90, 2: 0.10, 3: 0.00, 4: 0.00}
    if min(len(ref_toks), len(hyp_toks)) < 4:
        base_weights = {1: 0.95, 2: 0.05, 3: 0.00, 4: 0.00}


    k_smooth   = 0.8
    idf_alpha  = 0.4

    gamma = 0.55
    floor = 0.15

    token_overlap_bonus = 0.10
    char_mix_weight     = 0.25

    sims = []
    used_weights = []

    for n in (1, 2, 3, 4):
        w_n = base_weights.get(n, 0.0)
        if w_n <= 0.0:
            continue

        ref_ngr = Counter(_ngrams(ref_toks, n))
        hyp_ngr = Counter(_ngrams(hyp_toks, n))
        if not ref_ngr or not hyp_ngr:
            continue

        for g in list(ref_ngr.keys()):
            ref_ngr[g] += k_smooth
        for g in list(hyp_ngr.keys()):
            hyp_ngr[g] += k_smooth

        tf_ref = _tf(ref_ngr)
        tf_hyp = _tf(hyp_ngr)

        idf: Dict[str, float] = {}
        total_ref = float(sum(ref_ngr.values())) or 1.0
        keys = set(ref_ngr) | set(hyp_ngr)
        for g in keys:
            freq = float(ref_ngr.get(g, 0.0)) / total_ref
            idf[g] = 1.0 / (1.0 + idf_alpha * freq)

        sims.append(_cosine(tf_ref, tf_hyp, idf))
        used_weights.append(w_n)

    if sims:
        wsum = sum(used_weights) or 1.0
        norm_weights = [w / wsum for w in used_weights]
        ngram_sim = sum(s * w for s, w in zip(sims, norm_weights))  # [0..1]
    else:
        ngram_sim = 0.0

    ref_set = set(ref_toks)
    hyp_set = set(hyp_toks)
    union = len(ref_set | hyp_set) or 1
    iou_tokens = len(ref_set & hyp_set) / union  # [0..1]

    boosted_ngram = min(1.0, ngram_sim + token_overlap_bonus * iou_tokens)

    ref_s = "".join(ref_toks)
    hyp_s = "".join(hyp_toks)
    if 0 < len(ref_s) and 0 < len(hyp_s) and min(len(ref_s), len(hyp_s)) < 16:

        ref_chars = set(ref_s)
        hyp_chars = set(hyp_s)
        char_union = len(ref_chars | hyp_chars) or 1
        char_iou = len(ref_chars & hyp_chars) / char_union  # [0..1]
        mixed = (1 - char_mix_weight) * boosted_ngram + char_mix_weight * char_iou
    else:
        mixed = boosted_ngram

    raw = max(0.0, min(1.0, mixed))
    boosted = floor + (raw ** gamma) * (1.0 - floor)
    return round(min(1.0, boosted), 4)

