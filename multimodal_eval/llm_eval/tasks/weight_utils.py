from __future__ import annotations
from typing import Dict, Optional
import math


def weighted_sum(parts: Dict[str, Optional[float]], weights: Dict[str, float], *, strict: bool = False) -> Optional[float]:
    avail = {k: v for k, v in parts.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    if not avail:
        return None

    if strict:
        total = sum(weights.values())
        if total <= 0:
            return None
        acc = 0.0
        for k, w in weights.items():
            v = parts.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                v = 0.0
            acc += w * v
        return acc / total
    else:
        sub_w = {k: weights[k] for k in avail}
        total = sum(sub_w.values())
        if total <= 0:
            return None
        acc = sum(sub_w[k] * avail[k] for k in avail)
        return acc / total
