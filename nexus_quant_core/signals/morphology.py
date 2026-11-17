from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nexus_quant_core.contracts import FirFilterParams, MorphologyParams
from nexus_quant_core.indicators.fir import calculate_cascaded_fir_series, calculate_derivatives
from nexus_quant_core.indicators.moving_avg import (
    calculate_rolling_vwap_series,
    calculate_rsi_series,
    calculate_sma_series,
)
from nexus_quant_core.math_utils import safe_float


def _parse_normalized_daily_bar(c: Any) -> Optional[Dict[str, Any]]:
    """Parse pre-normalized daily bar requiring explicit YYYY-MM-DD trading_date (zero TZ guessing)."""
    if c is None or isinstance(c, bool):
        return None
    if isinstance(c, dict):
        dt_str = str(c.get("trading_date") or c.get("dt_str") or "").strip()
        ts = safe_float(c.get("timestamp", c.get("ts", 0.0)), 0.0)
        o = safe_float(c.get("open", c.get("o", 0.0)), 0.0)
        h = safe_float(c.get("high", c.get("h", 0.0)), 0.0)
        l = safe_float(c.get("low", c.get("l", 0.0)), 0.0)
        cl = safe_float(c.get("close", c.get("c", 0.0)), 0.0)
        v = max(0.0, safe_float(c.get("volume", c.get("v", 0.0)), 0.0))
    else:
        dt_str = str(getattr(c, "trading_date", getattr(c, "dt_str", "")) or "").strip()
        ts = safe_float(getattr(c, "timestamp", getattr(c, "ts", 0.0)), 0.0)
        o = safe_float(getattr(c, "open", getattr(c, "o", 0.0)), 0.0)
        h = safe_float(getattr(c, "high", getattr(c, "h", 0.0)), 0.0)
        l = safe_float(getattr(c, "low", getattr(c, "l", 0.0)), 0.0)
        cl = safe_float(getattr(c, "close", getattr(c, "c", 0.0)), 0.0)
        v = max(0.0, safe_float(getattr(c, "volume", getattr(c, "v", 0.0)), 0.0))

    if len(dt_str) != 10 or dt_str[4] != "-" or dt_str[7] != "-" or cl <= 0.0:
        return None
    if o <= 0.0:
        o = cl
    if l <= 0.0:
        l = min(o, cl)
    if h <= 0.0:
        h = max(o, cl)
    if h < l:
        h, l = l, h
    h = max(h, o, cl)
    l = min(l, o, cl)
    return {"dt_str": dt_str, "ts": ts, "open": o, "high": h, "low": l, "close": cl, "volume": v}


def _dedup_daily_bars(bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for b in bars:
        cur = best.get(b["dt_str"])
        if cur is None or b["ts"] >= cur["ts"]:
            best[b["dt_str"]] = b
    return [best[d] for d in sorted(best.keys())]


def _daily_atr_at(highs: List[float], lows: List[float], closes: List[float], j: int, period: int) -> float:
    if j <= 0:
        return abs(highs[0] - lows[0])
    start = max(1, j - period + 1)
    tr_sum = 0.0
    cnt = 0
    for k in range(start, j + 1):
        hk, lk, pc = highs[k], lows[k], closes[k - 1]
        tr_sum += max(hk - lk, abs(hk - pc), abs(lk - pc))
        cnt += 1
    return (tr_sum / float(cnt)) if cnt > 0 else abs(highs[j] - lows[j])


def extract_daily_morphology(
    daily_bars: Sequence[Any],
    *,
    params: MorphologyParams,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """
    Pure daily morphology feature scanner (CAP / DBL / DIP / SQZ / REV) parameterized by MorphologyParams.
    Requires pre-normalized daily_bars carrying canonical 'trading_date' ('YYYY-MM-DD').
    """
    raw_parsed = [b for b in (_parse_normalized_daily_bar(c) for c in daily_bars) if b is not None]
    if not raw_parsed:
        return {}, {}

    parsed = _dedup_daily_bars(raw_parsed)
    closes = [p["close"] for p in parsed]
    highs = [p["high"] for p in parsed]
    lows = [p["low"] for p in parsed]
    vols = [p["volume"] for p in parsed]

    q_len = params.ma_quarter
    m_len = params.ma_month
    v_len = params.vol_ma_period

    ma60_map: Dict[str, float] = {}
    morph_map: Dict[str, Dict[str, Any]] = {}

    for j, p in enumerate(parsed):
        dt_str = p["dt_str"]
        c_close = closes[j]
        c_vol = vols[j]

        if j >= q_len - 1:
            ma_q = sum(closes[j - (q_len - 1) : j + 1]) / float(q_len)
            ma60_map[dt_str] = round(ma_q, 2)
        else:
            ma_q = sum(closes[: j + 1]) / float(j + 1)
            if len(parsed) < q_len:
                ma60_map[dt_str] = round(ma_q, 2)

        d_atr = _daily_atr_at(highs, lows, closes, j, params.atr_period)
        d_atr_pct = (d_atr / c_close) if c_close > 1e-9 else 0.02
        is_prev_down = (closes[j - 1] <= closes[j - 2]) if j >= 2 else False

        if j >= m_len:
            vol_win = vols[max(0, j - v_len) : j]
            vol_ma = (sum(vol_win) / float(len(vol_win))) if vol_win else 1.0
            vol_ratio = (c_vol / vol_ma) if vol_ma > 1e-9 else 1.0

            ma_m = sum(closes[max(0, j - (m_len - 1)) : j + 1]) / float(m_len)
            bias_ma20 = ((c_close - ma_m) / ma_m) if ma_m > 1e-9 else 0.0

            down_days = sum(1 for k in range(max(1, j - 4), j + 1) if closes[k] <= closes[k - 1])
            is_new_low = lows[j] <= min(lows[max(0, j - 5) : j])
            vr_prev = (c_vol / vols[j - 1]) if (j > 0 and vols[j - 1] > 1e-9) else 1.0
            rng = highs[j] - lows[j]
            lower_shadow = ((c_close - lows[j]) / rng) if rng > 1e-9 else 0.0

            ma_q_dyn = sum(closes[max(0, j - (q_len - 1)) : j + 1]) / float(min(q_len, j + 1))
            bias_ma60_dyn = ((c_close - ma_q_dyn) / ma_q_dyn) if ma_q_dyn > 1e-9 else 0.0

            # [CAP] Capitulation Climax
            is_cap_day = (
                down_days >= params.cap_down_days
                and bias_ma20 <= params.cap_bias20_max
                and vol_ratio >= params.cap_vol_ratio_min
                and is_new_low
            )

            # [DBL] Double Bottom / W-Base
            has_base_dbl = False
            for k in range(max(0, j - params.dbl_lookback_max), max(0, j - params.dbl_lookback_min)):
                if lows[k] <= 1e-9:
                    continue
                if abs(lows[j] - lows[k]) / lows[k] <= params.dbl_diff_tol:
                    neck = (max(highs[k:j]) - lows[k]) / lows[k]
                    if neck >= params.dbl_neck_min:
                        has_base_dbl = True
                        break
            in_dbl_bias = params.dbl_bias20_min <= bias_ma20 <= params.dbl_bias20_max
            is_dbl_a = has_base_dbl and vol_ratio >= params.dbl_vol_ratio_min and in_dbl_bias
            is_dbl_b = has_base_dbl and lower_shadow >= params.dbl_shadow_min and in_dbl_bias
            is_dbl_day = is_dbl_a or is_dbl_b

            # [DIP] Quarterly/Monthly Pullback Support
            is_dip_a = (
                abs(bias_ma60_dyn) <= params.dip_bias60_abs_max
                and bias_ma20 <= params.dip_a_bias20_max
                and (lower_shadow >= 0.15 or vol_ratio >= 1.30)
                and down_days >= 3
                and is_new_low
            )
            is_dip_b = (
                bias_ma20 <= params.dip_b_bias20_max
                and lower_shadow >= 0.50
                and vol_ratio >= 0.90
                and down_days >= 3
                and is_new_low
                and not is_dbl_day
                and not is_cap_day
            )
            prior_highs = highs[max(0, j - 8) : j]
            max_h = max(prior_highs) if prior_highs else 0.0
            pullback = ((lows[j] - max_h) / max_h) if max_h > 1e-9 else 0.0
            is_dip_c = (
                pullback <= params.dip_c_pullback_max
                and 0.0 <= bias_ma20 <= 0.08
                and bias_ma60_dyn >= 0.15
                and down_days >= 2
                and (vol_ratio <= 0.85 or vr_prev >= 1.20)
            )
            prior_low_win = lows[max(0, j - 25) : max(0, j - 10)]
            prior_panic_low = min(prior_low_win) if prior_low_win else 0.0
            is_higher_low = (lows[j] >= prior_panic_low * 1.10) if prior_panic_low > 1e-9 else False
            is_dip_d = (
                abs(bias_ma60_dyn) <= params.dip_bias60_abs_max
                and pullback <= params.dip_d_pullback_max
                and vol_ratio <= 0.65
                and lower_shadow >= 0.50
                and is_higher_low
            )
            is_dip_day = is_dip_a or is_dip_b or is_dip_c or is_dip_d

            # [SQZ] 4-MA Squeeze Base
            ma_f = sum(closes[max(0, j - (params.ma_fast - 1)) : j + 1]) / float(min(params.ma_fast, j + 1))
            ma_md = sum(closes[max(0, j - (params.ma_mid - 1)) : j + 1]) / float(min(params.ma_mid, j + 1))
            mas4 = [ma_f, ma_md, ma_m, ma_q_dyn]
            sp4 = ((max(mas4) - min(mas4)) / ma_q_dyn) if ma_q_dyn > 1e-9 else 1.0
            is_sqz_base_day = (
                j >= 30
                and sp4 <= params.sqz_spread4_max
                and abs(bias_ma60_dyn) <= params.sqz_bias60_abs_max
                and bias_ma20 <= 0.0
                and vol_ratio >= params.sqz_vol_ratio_min
            )

            # [REV] Oversold Reversal Base
            is_rev_day = (not is_dip_day) and is_prev_down and (bias_ma20 <= 0.0)

            morph_map[dt_str] = {
                "close": round(c_close, 4),
                "prev_close": round(closes[j - 1], 4),
                "yesterday_low": round(lows[j], 4),
                "vol_ratio": round(max(0.0, vol_ratio), 2),
                "bias_ma20": round(bias_ma20, 4),
                "ma60": round(ma_q, 2),
                "daily_atr": round(d_atr, 4),
                "daily_atr_pct": round(d_atr_pct, 4),
                "is_rev_day": bool(is_rev_day),
                "is_cap_day": bool(is_cap_day),
                "is_dbl_day": bool(is_dbl_day),
                "is_dbl_a": bool(is_dbl_a),
                "is_dbl_b": bool(is_dbl_b),
                "is_dip_day": bool(is_dip_day),
                "is_dip_a": bool(is_dip_a),
                "is_dip_b": bool(is_dip_b),
                "is_dip_c": bool(is_dip_c),
                "is_dip_d": bool(is_dip_d),
                "is_sqz_base_day": bool(is_sqz_base_day),
            }
        elif len(parsed) < m_len:
            ma_sub = sum(closes[: j + 1]) / float(j + 1)
            bias_sub = ((c_close - ma_sub) / ma_sub) if ma_sub > 1e-9 else 0.0
            morph_map[dt_str] = {
                "close": round(c_close, 4),
                "prev_close": round(closes[j - 1] if j > 0 else c_close, 4),
                "yesterday_low": round(lows[j], 4),
                "vol_ratio": 1.0,
                "bias_ma20": round(bias_sub, 4),
                "ma60": round(ma_q, 2),
                "daily_atr": round(d_atr, 4),
                "daily_atr_pct": round(d_atr_pct, 4),
                "is_rev_day": bool(is_prev_down and bias_sub <= 0.0),
                "is_cap_day": False,
                "is_dbl_day": False,
                "is_dbl_a": False,
                "is_dbl_b": False,
                "is_dip_day": False,
                "is_dip_a": False,
                "is_dip_b": False,
                "is_dip_c": False,
                "is_dip_d": False,
                "is_sqz_base_day": False,
            }

    return ma60_map, morph_map


def extract_bond_etf_features(
    daily_bars: Sequence[Any],
    *,
    ma_period: int,
    rsi_period: int,
    breakdown_ratio: float,
) -> Dict[str, Dict[str, Any]]:
    """Pure bond ETF macro cycle & breakdown scanner with explicit parameters."""
    raw_parsed = [b for b in (_parse_normalized_daily_bar(c) for c in daily_bars) if b is not None]
    if not raw_parsed:
        return {}
    parsed = _dedup_daily_bars(raw_parsed)
    closes = [p["close"] for p in parsed]
    opens = [p["open"] for p in parsed]
    lows = [p["low"] for p in parsed]
    rsi_vals = calculate_rsi_series(closes, period=rsi_period)

    out: Dict[str, Dict[str, Any]] = {}
    for i, p in enumerate(parsed):
        w = closes[max(0, i - (ma_period - 1)) : i + 1]
        ma_val = sum(w) / float(len(w))
        bias = ((closes[i] - ma_val) / ma_val * 100.0) if ma_val > 1e-9 else 0.0
        prev_c = closes[i - 1] if i > 0 else opens[i]
        gap_pct = ((opens[i] - prev_c) / prev_c * 100.0) if prev_c > 1e-9 else 0.0
        prior_lows = lows[max(0, i - ma_period) : max(0, i - 10)] if i >= 15 else lows[:i]
        min_base = min(prior_lows) if prior_lows else lows[i]
        out[p["dt_str"]] = {
            "ma60": round(ma_val, 2),
            "bias60": round(bias, 4),
            "rsi": rsi_vals[i],
            "gap_pct": round(gap_pct, 4),
            "is_breakdown": bool(closes[i] < min_base * breakdown_ratio) if min_base > 0 else False,
            "min_base_low": round(min_base, 2),
        }
    return out


def evaluate_masb_signals(
    candles: Sequence[Any],
    *,
    fir_params: FirFilterParams,
    morph_params: MorphologyParams,
    vwap_window: int,
) -> Dict[str, Any]:
    """
    Pure Moving Average Squeeze Breakout (MASB) scanner parameterized by FirFilterParams & MorphologyParams.
    """
    if len(candles) < morph_params.ma_month:
        return {"fir": [], "vwap": [], "ma5": [], "ma10": [], "ma20": [], "spread_pct": [], "buy_indices": []}

    closes = [safe_float(getattr(c, "close", c.get("close") if isinstance(c, dict) else 0.0), 0.0) for c in candles]
    vols = [max(0.0, safe_float(getattr(c, "volume", c.get("volume") if isinstance(c, dict) else 0.0), 0.0)) for c in candles]

    fir_s = calculate_cascaded_fir_series(closes, params=fir_params)
    vwap_s = calculate_rolling_vwap_series(candles, lookback=vwap_window)
    ma5_s = calculate_sma_series(closes, period=morph_params.ma_fast)
    ma10_s = calculate_sma_series(closes, period=morph_params.ma_mid)
    ma20_s = calculate_sma_series(closes, period=morph_params.ma_month)
    vol_ma_s = calculate_sma_series(vols, period=morph_params.vol_ma_period)

    spread_pct: List[float] = []
    for i in range(len(closes)):
        cluster = (fir_s[i], vwap_s[i], ma5_s[i], ma10_s[i], ma20_s[i])
        sp = ((max(cluster) - min(cluster)) / ma20_s[i] * 100.0) if ma20_s[i] > 1e-9 else 999.0
        spread_pct.append(round(sp, 4))

    buy_indices: List[int] = []
    for i in range(max(morph_params.ma_month, 6), len(closes)):
        recent_squeeze = min(spread_pct[max(0, i - 5) : i]) <= morph_params.masb_squeeze_threshold_pct
        vel, acc = calculate_derivatives(
            fir_s[i], fir_s[i - 1], fir_s[i - 2], deadzone_eps=fir_params.deadzone_eps
        )
        cluster_top = max(ma5_s[i], ma10_s[i], ma20_s[i], vwap_s[i])
        vol_ok = vols[i] >= vol_ma_s[i - 1] * morph_params.masb_vol_mult if vol_ma_s[i - 1] > 0 else True
        if (
            recent_squeeze
            and closes[i] > cluster_top
            and closes[i - 1] <= max(ma5_s[i - 1], ma10_s[i - 1], ma20_s[i - 1], vwap_s[i - 1])
            and vel >= morph_params.masb_vel_min
            and acc >= morph_params.masb_acc_min
            and vol_ok
        ):
            buy_indices.append(i)

    return {
        "fir": fir_s,
        "vwap": vwap_s,
        "ma5": ma5_s,
        "ma10": ma10_s,
        "ma20": ma20_s,
        "spread_pct": spread_pct,
        "buy_indices": buy_indices,
    }
