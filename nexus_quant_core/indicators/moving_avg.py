from __future__ import annotations

import math
from typing import Any, List, Sequence

from nexus_quant_core.contracts import SessionCalendarProtocol
from nexus_quant_core.math_utils import round_adaptive_price, safe_float


def _extract_bar_field(bar: Any, *keys: str, default: Any = None) -> Any:
    if isinstance(bar, dict):
        for k in keys:
            if k in bar and bar[k] is not None:
                return bar[k]
        return default
    for k in keys:
        val = getattr(bar, k, None)
        if val is not None:
            return val
    return default


def calculate_sma_series(values: Sequence[float], *, period: int) -> List[float]:
    """O(N) prefix-sum Simple Moving Average series with explicit period."""
    if not values:
        return []
    if period <= 0 or isinstance(period, bool):
        raise ValueError(f"SMA period must be > 0, got {period!r}")
    n = len(values)
    pref = [0.0] * (n + 1)
    for k, v in enumerate(values):
        pref[k + 1] = pref[k] + safe_float(v, 0.0)
    out = [0.0] * n
    for i in range(n):
        start = max(0, i - period + 1)
        cnt = float(i - start + 1)
        out[i] = round_adaptive_price((pref[i + 1] - pref[start]) / cnt)
    return out


def calculate_ema_series(values: Sequence[float], *, period: int) -> List[float]:
    """Causal Exponential Moving Average series with explicit period."""
    if not values:
        return []
    if period <= 0 or isinstance(period, bool):
        raise ValueError(f"EMA period must be > 0, got {period!r}")
    alpha = 2.0 / (float(period) + 1.0)
    out: List[float] = []
    ema = safe_float(values[0], 0.0)
    out.append(round_adaptive_price(ema))
    for v in values[1:]:
        fv = safe_float(v, ema)
        ema = alpha * fv + (1.0 - alpha) * ema
        out.append(round_adaptive_price(ema))
    return out


def calculate_linear_weighted_vwap(candles: Sequence[Any]) -> float:
    """Linearly weighted rolling VWAP (weights 1..N) across pre-normalized candles."""
    if not candles:
        return 0.0
    clean = [c for c in candles if c is not None]
    if not clean:
        return 0.0

    indexed = []
    has_ts = False
    for c in clean:
        raw_ts = _extract_bar_field(c, "timestamp", "ts", "time", "t", default=None)
        if raw_ts is not None:
            if isinstance(raw_ts, bool):
                continue
            ts_f = safe_float(raw_ts, float("nan"))
            if not math.isfinite(ts_f) or ts_f <= 0:
                continue
            has_ts = True
            indexed.append((ts_f, c))
        else:
            indexed.append((0.0, c))

    if not indexed:
        return 0.0
    if has_ts and len(indexed) >= 2:
        indexed.sort(key=lambda x: x[0])
    ordered = [c for _, c in indexed]

    vols = [max(0.0, safe_float(_extract_bar_field(c, "volume", "v", default=0.0), 0.0)) for c in ordered]
    pos_vols = [v for v in vols if v > 0.0]
    avg_pos_vol = (sum(pos_vols) / len(pos_vols)) if (pos_vols and len(pos_vols) < len(vols)) else 0.0

    total_w_vol = 0.0
    total_w_vp = 0.0
    fallback_w_p = 0.0
    total_w = 0.0

    for i, c in enumerate(ordered):
        weight = float(i + 1)
        vol = vols[i]
        if vol <= 0.0 and avg_pos_vol > 0.0:
            vol = avg_pos_vol
        cl = max(0.0, safe_float(_extract_bar_field(c, "close", "c", default=0.0), 0.0))
        h = max(0.0, safe_float(_extract_bar_field(c, "high", "h", default=cl), cl))
        l = max(0.0, safe_float(_extract_bar_field(c, "low", "l", default=cl), cl))
        if h < l:
            h, l = l, h
        h = max(h, cl)
        l = min(l, cl)
        typ = (h + l + cl) / 3.0
        total_w_vol += weight * vol
        total_w_vp += weight * vol * typ
        fallback_w_p += weight * typ
        total_w += weight

    if total_w_vol <= 0.0:
        return max(0.0, fallback_w_p / total_w) if total_w > 0.0 else 0.0
    return max(0.0, total_w_vp / total_w_vol)


def calculate_rolling_vwap_series(candles: Sequence[Any], *, lookback: int) -> List[float]:
    """O(N) prefix-sum linearly weighted rolling VWAP series with explicit lookback."""
    if not candles:
        return []
    if lookback <= 0 or isinstance(lookback, bool):
        raise ValueError(f"VWAP lookback must be > 0, got {lookback!r}")

    n = len(candles)
    pref_v = [0.0] * (n + 1)
    pref_kv = [0.0] * (n + 1)
    pref_vp = [0.0] * (n + 1)
    pref_kvp = [0.0] * (n + 1)
    pref_p = [0.0] * (n + 1)
    pref_kp = [0.0] * (n + 1)

    for k, c in enumerate(candles):
        vol = max(0.0, safe_float(_extract_bar_field(c, "volume", "v", default=0.0), 0.0))
        cl = max(0.0, safe_float(_extract_bar_field(c, "close", "c", default=0.0), 0.0))
        h = max(0.0, safe_float(_extract_bar_field(c, "high", "h", default=cl), cl))
        l = max(0.0, safe_float(_extract_bar_field(c, "low", "l", default=cl), cl))
        if h < l:
            h, l = l, h
        h = max(h, cl)
        l = min(l, cl)
        typ = (h + l + cl) / 3.0
        vp = vol * typ
        kf = float(k)

        pref_v[k + 1] = pref_v[k] + vol
        pref_kv[k + 1] = pref_kv[k] + kf * vol
        pref_vp[k + 1] = pref_vp[k] + vp
        pref_kvp[k + 1] = pref_kvp[k] + kf * vp
        pref_p[k + 1] = pref_p[k] + typ
        pref_kp[k + 1] = pref_kp[k] + kf * typ

    out = [0.0] * n
    for i in range(n):
        a = max(0, i - lookback + 1)
        af = float(a - 1)
        sum_v = pref_v[i + 1] - pref_v[a]
        sum_kv = pref_kv[i + 1] - pref_kv[a]
        total_w_vol = sum_kv - af * sum_v
        if total_w_vol > 0.0:
            sum_vp = pref_vp[i + 1] - pref_vp[a]
            sum_kvp = pref_kvp[i + 1] - pref_kvp[a]
            out[i] = round_adaptive_price(max(0.0, (sum_kvp - af * sum_vp) / total_w_vol))
        else:
            sum_p = pref_p[i + 1] - pref_p[a]
            sum_kp = pref_kp[i + 1] - pref_kp[a]
            length = float(i - a + 1)
            total_w = length * (length + 1.0) / 2.0
            out[i] = round_adaptive_price(max(0.0, (sum_kp - af * sum_p) / total_w)) if total_w > 0.0 else 0.0
    return out


def calculate_session_vwap_series(
    candles: Sequence[Any],
    *,
    calendar: SessionCalendarProtocol,
    symbol: str,
) -> List[float]:
    """Session-anchored cumulative VWAP resetting strictly on calendar.get_trading_day transitions."""
    if not candles:
        return []
    out: List[float] = []
    cur_td = None
    cum_vol = 0.0
    cum_vp = 0.0

    for c in candles:
        if c is None:
            continue
        ts_f = safe_float(_extract_bar_field(c, "timestamp", "ts", "time", default=0.0), 0.0)
        if ts_f > 0.0:
            td = calendar.get_trading_day(ts_f, symbol=symbol)
            if cur_td is None:
                cur_td = td
            elif td != cur_td:
                cur_td = td
                cum_vol = 0.0
                cum_vp = 0.0

        cl = max(0.0, safe_float(_extract_bar_field(c, "close", "c", default=0.0), 0.0))
        h = max(0.0, safe_float(_extract_bar_field(c, "high", "h", default=cl), cl))
        l = max(0.0, safe_float(_extract_bar_field(c, "low", "l", default=cl), cl))
        if h < l:
            h, l = l, h
        h = max(h, cl)
        l = min(l, cl)
        typ = (h + l + cl) / 3.0
        vol = max(0.0, safe_float(_extract_bar_field(c, "volume", "v", default=0.0), 0.0))

        cum_vol += vol
        cum_vp += vol * typ
        out.append(round_adaptive_price((cum_vp / cum_vol) if cum_vol > 0.0 else typ))
    return out


def calculate_atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    *,
    period: int,
) -> float:
    """Average True Range over explicit period."""
    if period <= 0 or isinstance(period, bool):
        raise ValueError(f"ATR period must be > 0, got {period!r}")
    min_len = min(len(highs), len(lows), len(closes))
    if min_len <= 0:
        return 0.0
    if min_len == 1:
        return max(0.0, abs(safe_float(highs[0], 0.0) - safe_float(lows[0], 0.0)))

    trs: List[float] = []
    for i in range(1, min_len):
        h = safe_float(highs[i], 0.0)
        l = safe_float(lows[i], 0.0)
        prev_c = safe_float(closes[i - 1], 0.0)
        if h <= 0.0 or l <= 0.0:
            continue
        if h < l:
            h, l = l, h
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c)) if prev_c > 0.0 else (h - l)
        trs.append(tr)

    if not trs:
        return 0.0
    recent = trs[-period:]
    return sum(recent) / float(len(recent))


def calculate_rsi_series(closes: Sequence[float], *, period: int) -> List[float]:
    """Wilder's smoothed RSI series over explicit period."""
    if not closes:
        return []
    if period <= 0 or isinstance(period, bool):
        raise ValueError(f"RSI period must be > 0, got {period!r}")
    n = len(closes)
    out = [50.0] * n
    if n <= period:
        return out

    diffs = [safe_float(closes[i], 0.0) - safe_float(closes[i - 1], 0.0) for i in range(1, n)]
    gains = [max(d, 0.0) for d in diffs]
    losses = [max(-d, 0.0) for d in diffs]

    avg_gain = sum(gains[:period]) / float(period)
    avg_loss = sum(losses[:period]) / float(period)

    def _to_rsi(ag: float, al: float) -> float:
        if al <= 0.0 and ag <= 0.0:
            return 50.0
        if al <= 0.0:
            return 100.0
        rs = ag / al
        return round(max(0.0, min(100.0, 100.0 - (100.0 / (1.0 + rs)))), 2)

    out[period] = _to_rsi(avg_gain, avg_loss)
    p_minus_1 = float(period - 1)
    pf = float(period)
    for i in range(period, len(diffs)):
        avg_gain = (avg_gain * p_minus_1 + gains[i]) / pf
        avg_loss = (avg_loss * p_minus_1 + losses[i]) / pf
        out[i + 1] = _to_rsi(avg_gain, avg_loss)
    return out
