from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple

from nexus_quant_core.contracts import FirFilterParams
from nexus_quant_core.math_utils import clean_finite_series, round_adaptive_price, safe_pct_change


def _clean_weights(weights: Sequence[float]) -> List[float]:
    cleaned = [float(x) for x in weights if not isinstance(x, bool) and math.isfinite(float(x)) and float(x) >= 0.0]
    w_sum = sum(cleaned) if cleaned else 0.0
    if not cleaned or not math.isfinite(w_sum) or w_sum <= 1e-9:
        raise ValueError(f"Invalid FIR filter weights: {weights!r}")
    return cleaned


def calculate_cascaded_fir(series: Any, *, params: FirFilterParams) -> float:
    """Two-stage cascaded FIR filter using explicit FirFilterParams."""
    clean_s = clean_finite_series(series)
    if not clean_s:
        return 0.0

    w1 = _clean_weights(params.stage1_weights)
    w2 = _clean_weights(params.stage2_weights)
    len_w1 = len(w1)
    len_w2 = len(w2)
    req_len = len_w1 + len_w2 - 1

    if len(clean_s) < req_len:
        return clean_s[-1]

    w1_sum = sum(w1)
    stage1_out: List[float] = []
    for i in range(len_w2):
        start_idx = len(clean_s) - (len_w1 + len_w2 - 1 - i)
        end_idx = len(clean_s) - (len_w2 - 1 - i)
        window = clean_s[start_idx:end_idx]
        stage1_out.append(sum(w * x for w, x in zip(w1, window)) / w1_sum)

    w2_sum = sum(w2)
    final_val = sum(w * x for w, x in zip(w2, stage1_out)) / w2_sum
    if not math.isfinite(final_val):
        return clean_s[-1]
    if abs(final_val - clean_s[-1]) < 1e-12:
        return clean_s[-1]
    return final_val


def calculate_cascaded_fir_series(series: Any, *, params: FirFilterParams) -> List[float]:
    """O(N) causal cascaded FIR series across an entire price vector."""
    clean_s = clean_finite_series(series)
    if not clean_s:
        return []
    w1 = _clean_weights(params.stage1_weights)
    w2 = _clean_weights(params.stage2_weights)
    req_len = len(w1) + len(w2) - 1
    out: List[float] = []
    for i in range(len(clean_s)):
        win = clean_s[max(0, i - req_len + 1) : i + 1]
        out.append(round_adaptive_price(calculate_cascaded_fir(win, params=params)))
    return out


def calculate_derivatives(
    curr_val: float,
    prev_val: float,
    prev2_val: float,
    *,
    deadzone_eps: float,
) -> Tuple[float, float]:
    """First-order velocity (%) and second-order acceleration (%) with explicit deadzone."""
    if (
        curr_val is None
        or prev_val is None
        or prev2_val is None
        or isinstance(curr_val, bool)
        or isinstance(prev_val, bool)
        or isinstance(prev2_val, bool)
    ):
        return 0.0, 0.0
    try:
        c_val = float(curr_val)
        p_val = float(prev_val)
        p2_val = float(prev2_val)
    except (ValueError, TypeError):
        return 0.0, 0.0

    if not (math.isfinite(c_val) and math.isfinite(p_val) and math.isfinite(p2_val)):
        return 0.0, 0.0
    if abs(p_val) < 1e-9 or abs(p2_val) < 1e-9:
        return 0.0, 0.0

    eps = max(0.0, float(deadzone_eps))
    vel_curr = safe_pct_change(c_val, p_val)
    vel_prev = safe_pct_change(p_val, p2_val)
    if abs(vel_curr) < eps:
        vel_curr = 0.0
    if abs(vel_prev) < eps:
        vel_prev = 0.0

    accel = vel_curr - vel_prev
    if abs(accel) < eps:
        accel = 0.0
    return vel_curr, accel


def calculate_fir_derivatives(series: Any, *, params: FirFilterParams) -> Tuple[float, float]:
    """Compute (velocity, acceleration) on cascaded FIR output; requires >= (len(w1)+len(w2)+1) samples."""
    clean_s = clean_finite_series(series)
    min_samples = len(params.stage1_weights) + len(params.stage2_weights) + 1
    if len(clean_s) < min_samples:
        return 0.0, 0.0
    fir_curr = calculate_cascaded_fir(clean_s, params=params)
    fir_prev = calculate_cascaded_fir(clean_s[:-1], params=params)
    fir_prev2 = calculate_cascaded_fir(clean_s[:-2], params=params)
    return calculate_derivatives(fir_curr, fir_prev, fir_prev2, deadzone_eps=params.deadzone_eps)
