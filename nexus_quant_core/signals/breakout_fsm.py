from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from nexus_quant_core.contracts import BreakoutFsmParams, FirFilterParams
from nexus_quant_core.indicators.fir import calculate_cascaded_fir
from nexus_quant_core.math_utils import safe_float, safe_pct_change


def evaluate_brk_entry(
    fir_vals: Sequence[float],
    vwap_vals: Sequence[float],
    ma_long_vals: Sequence[float],
    *,
    idx: int,
    params: BreakoutFsmParams,
    search_start_idx: int = 1,
) -> bool:
    """
    [BRK] Sequential 3-State Breakout FSM parameterized by BreakoutFsmParams:
    - Warmup: idx >= params.ma_long_window (e.g. TW=1080, US=1560)
    - State 3 Trigger: FIR crosses up above MA_LONG on bar idx, while VWAP < MA_LONG
    - State 1 -> State 2 Ignition within params.fsm_timeout_bars:
      FIR crossed up above VWAP at k, where the preceding params.ignition_lookback bars
      all had VWAP < MA_LONG, and VWAP stayed < MA_LONG from k through idx.
    """
    if (
        not fir_vals
        or not vwap_vals
        or not ma_long_vals
        or idx < params.ma_long_window
        or idx >= min(len(fir_vals), len(vwap_vals), len(ma_long_vals))
    ):
        return False

    if not (fir_vals[idx - 1] <= ma_long_vals[idx - 1] and fir_vals[idx] > ma_long_vals[idx]):
        return False

    if not (vwap_vals[idx] < ma_long_vals[idx]):
        return False

    min_k = max(1, int(search_start_idx), idx - params.fsm_timeout_bars)
    for k in range(idx, min_k - 1, -1):
        if not (vwap_vals[k] < ma_long_vals[k]):
            return False
        is_ignition = fir_vals[k - 1] <= vwap_vals[k - 1] and fir_vals[k] > vwap_vals[k]
        if is_ignition:
            ign_base_start = max(0, k - (params.ignition_lookback - 1))
            if all(vwap_vals[t] < ma_long_vals[t] for t in range(ign_base_start, k + 1)):
                return True
    return False


def evaluate_mfo_entry(
    fir_vals: Sequence[float],
    vwap_vals: Sequence[float],
    ma_mid_vals: Sequence[float],
    ma_long_vals: Sequence[float],
    *,
    idx: int,
    params: BreakoutFsmParams,
) -> bool:
    """
    [MFO] Bullish Moving-Average Fanout Momentum Entry parameterized by BreakoutFsmParams:
    1. Across params.ignition_lookback bars ending at idx: VWAP > MA_MID > MA_LONG.
    2. FIR crosses up above VWAP on bar idx.
    3. VWAP, MA_MID, and MA_LONG all have positive slope over params.slope_lookback bars.
    """
    if not fir_vals or not vwap_vals or not ma_mid_vals or not ma_long_vals:
        return False

    lb = max(1, int(params.ignition_lookback))
    slope_lb = max(1, int(params.slope_lookback))
    min_req = params.ma_long_window + max(lb, slope_lb)
    min_len = min(len(fir_vals), len(vwap_vals), len(ma_mid_vals), len(ma_long_vals))
    if idx < min_req or idx >= min_len or idx < slope_lb:
        return False

    win_start = max(0, idx - (lb - 1))
    if not all(vwap_vals[k] > ma_mid_vals[k] > ma_long_vals[k] for k in range(win_start, idx + 1)):
        return False

    c_fir = safe_float(fir_vals[idx], 0.0)
    p_fir = safe_float(fir_vals[idx - 1], 0.0)
    c_vwap = safe_float(vwap_vals[idx], 0.0)
    p_vwap = safe_float(vwap_vals[idx - 1], 0.0)
    if not (p_fir <= p_vwap and c_fir > c_vwap):
        return False

    if params.min_submersion_bars > 0:
        below_cnt = 0
        for k in range(idx - 1, max(-1, idx - 1 - params.min_submersion_bars - 15), -1):
            if safe_float(fir_vals[k], 0.0) <= safe_float(vwap_vals[k], 0.0):
                below_cnt += 1
            else:
                break
        if below_cnt < params.min_submersion_bars:
            return False

    if params.min_submersion_pct > 0.0 and c_vwap > 0.0:
        chk_start = max(0, idx - 15)
        min_diff = min(safe_float(fir_vals[k], 0.0) - safe_float(vwap_vals[k], 0.0) for k in range(chk_start, idx))
        if (min_diff / c_vwap) > -params.min_submersion_pct:
            return False

    ref_idx = idx - slope_lb
    if not (
        c_vwap > safe_float(vwap_vals[ref_idx], 0.0)
        and safe_float(ma_mid_vals[idx], 0.0) > safe_float(ma_mid_vals[ref_idx], 0.0)
        and safe_float(ma_long_vals[idx], 0.0) > safe_float(ma_long_vals[ref_idx], 0.0)
    ):
        return False

    return True


def evaluate_btx_brk_exit(
    highs: Sequence[float],
    closes: Sequence[float],
    vwap_vals: Sequence[float],
    *,
    idx: int,
    params: BreakoutFsmParams,
    entry_idx: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    [BTX_BRK] Swing Breakout Exit parameterized by BreakoutFsmParams:
    - Close < max_high(btx_high_lookback or since entry_idx) * btx_drawdown_ratio
    - OR High < VWAP
    """
    if (
        not highs
        or not closes
        or not vwap_vals
        or idx < params.btx_high_lookback
        or idx >= min(len(highs), len(closes), len(vwap_vals))
    ):
        return False, ""

    lb_start = max(0, idx - (params.btx_high_lookback - 1))
    if entry_idx is not None and 0 <= entry_idx <= idx:
        lb_start = max(lb_start, entry_idx)
    max_high = max(safe_float(h, 0.0) for h in highs[lb_start : idx + 1])
    c_close = safe_float(closes[idx], 0.0)
    c_high = safe_float(highs[idx], 0.0)
    c_vwap = safe_float(vwap_vals[idx], 0.0)

    dd_pct = round((1.0 - params.btx_drawdown_ratio) * 100.0, 1)
    if c_close < max_high * params.btx_drawdown_ratio:
        return True, f"BTX_BRK (close {c_close:.2f} < peak*{params.btx_drawdown_ratio:.2f} [-{dd_pct:.0f}%])"
    if c_high < c_vwap:
        return True, f"BTX_BRK (high {c_high:.2f} < VWAP {c_vwap:.2f})"
    return False, ""


def evaluate_btx_mfo_exit(
    highs: Sequence[float],
    closes: Sequence[float],
    vwap_vals: Sequence[float],
    *,
    idx: int,
    params: BreakoutFsmParams,
    fir_params: FirFilterParams,
    entry_idx: Optional[int] = None,
    max_high_since_entry: Optional[float] = None,
    fir_vals: Optional[Sequence[float]] = None,
    fir_velocities: Optional[Sequence[float]] = None,
) -> Tuple[bool, str]:
    """
    [BTX_MFO] Bullish Fanout Exit FSM parameterized by BreakoutFsmParams:
    1. Post-entry peak drawdown breach (< eff_peak * btx_drawdown_ratio)
    2. Underwater FIR < VWAP divergence/velocity watch FSM
    """
    if (
        not highs
        or not closes
        or not vwap_vals
        or idx < 0
        or idx >= min(len(highs), len(closes), len(vwap_vals))
    ):
        return False, ""

    eff_peak = 0.0
    if max_high_since_entry is not None and math.isfinite(max_high_since_entry) and max_high_since_entry > 0.0:
        eff_peak = max(float(max_high_since_entry), safe_float(highs[idx], 0.0))
    if entry_idx is not None and 0 <= entry_idx <= idx:
        eff_peak = max(eff_peak, max(safe_float(h, 0.0) for h in highs[entry_idx : idx + 1]))
    if eff_peak <= 0.0:
        eff_peak = safe_float(highs[idx], 0.0)

    c_close = safe_float(closes[idx], 0.0)
    c_vwap = safe_float(vwap_vals[idx], 0.0)
    if eff_peak > 0.0 and c_close < eff_peak * params.btx_drawdown_ratio:
        return True, f"BTX_MFO (close {c_close:.2f} < entry_peak*{params.btx_drawdown_ratio:.2f})"

    if fir_vals is None:
        fir_list: Sequence[float] = [
            round(calculate_cascaded_fir(closes[max(0, k - 20) : k + 1], params=fir_params), 4)
            for k in range(idx + 1)
        ]
    else:
        fir_list = fir_vals

    c_fir = safe_float(fir_list[idx], 0.0)
    if c_vwap <= 0.0 or c_fir >= c_vwap:
        return False, ""

    if fir_velocities is not None and idx < len(fir_velocities):
        c_vel = safe_float(fir_velocities[idx], 0.0)
    else:
        prev_fir = safe_float(fir_list[idx - 1], c_fir) if idx >= 1 else c_fir
        c_vel = safe_pct_change(c_fir, prev_fir)

    start_bound = entry_idx if (entry_idx is not None and 0 <= entry_idx <= idx) else max(0, idx - 10)
    sub_start = idx
    while sub_start > start_bound:
        pk = sub_start - 1
        if safe_float(vwap_vals[pk], 0.0) > 0.0 and safe_float(fir_list[pk], 0.0) < safe_float(vwap_vals[pk], 0.0):
            sub_start = pk
        else:
            break

    if c_vel < params.btx_mfo_vel_threshold:
        return True, f"BTX_MFO (underwater vel {c_vel:.2f}% < {params.btx_mfo_vel_threshold:.2f}%)"

    init_vwap = safe_float(vwap_vals[sub_start], 0.0)
    init_fir = safe_float(fir_list[sub_start], 0.0)
    entry_gap = max(0.0, (init_vwap - init_fir) / init_vwap) if init_vwap > 0.0 else 0.0
    curr_gap = max(0.0, (c_vwap - c_fir) / c_vwap)
    delta_gap = curr_gap - entry_gap
    if delta_gap >= params.btx_mfo_max_divergence_pct:
        return True, f"BTX_MFO (divergence +{delta_gap * 100:.2f}% >= {params.btx_mfo_max_divergence_pct * 100:.2f}%)"

    return False, ""
