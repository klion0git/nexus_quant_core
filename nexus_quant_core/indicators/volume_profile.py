from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nexus_quant_core.contracts import SessionCalendarProtocol, VolumeProfileParams
from nexus_quant_core.math_utils import safe_float, safe_int


def _extract_hlvct(c: Any) -> Optional[Tuple[float, float, float, float, int]]:
    if c is None or isinstance(c, bool):
        return None
    if isinstance(c, dict):
        raw_h = c.get("high", c.get("h"))
        raw_l = c.get("low", c.get("l"))
        raw_c = c.get("close", c.get("c"))
        raw_v = c.get("volume", c.get("v", 0.0))
        raw_t = c.get("timestamp", c.get("ts", c.get("time", 0)))
    else:
        raw_h = getattr(c, "high", getattr(c, "h", None))
        raw_l = getattr(c, "low", getattr(c, "l", None))
        raw_c = getattr(c, "close", getattr(c, "c", None))
        raw_v = getattr(c, "volume", getattr(c, "v", 0.0))
        raw_t = getattr(c, "timestamp", getattr(c, "ts", getattr(c, "time", 0)))

    fc = safe_float(raw_c, -1.0)
    fh = safe_float(raw_h if raw_h is not None else fc, fc)
    fl = safe_float(raw_l if raw_l is not None else fc, fc)
    fv = max(0.0, safe_float(raw_v, 0.0))
    ts = safe_int(raw_t, 0)
    if fh >= fl > 0.0 and fc > 0.0:
        return (fh, fl, fv, fc, ts)
    return None


def resolve_bin_size(
    candles: Sequence[Any],
    *,
    params: VolumeProfileParams,
    is_daily_level: bool = False,
) -> float:
    """Resolve price bin size from market-specific tick step table in VolumeProfileParams."""
    parsed = [x for x in (_extract_hlvct(c) for c in candles) if x is not None]
    if not parsed:
        return 1.0
    avg_px = sum(x[3] for x in parsed) / float(len(parsed))
    table = params.daily_tick_steps if is_daily_level else params.intraday_tick_steps
    for threshold_px, step in sorted(table, key=lambda t: t[0], reverse=True):
        if avg_px >= threshold_px:
            return max(1e-8, float(step))
    return max(1e-8, float(table[-1][1])) if table else 1.0


def calculate_volume_profile(
    candles: Sequence[Any],
    *,
    params: VolumeProfileParams,
    bin_size: Optional[float] = None,
    is_daily_level: bool = False,
) -> Dict[str, Any]:
    """Compute POC, VAH, VAL, and histogram bins using explicit VolumeProfileParams."""
    parsed = [x for x in (_extract_hlvct(c) for c in candles) if x is not None]
    if not parsed:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "total_volume": 0.0, "bins": [], "bin_size": 1.0}

    eff_bin = safe_float(bin_size, 0.0)
    if eff_bin <= 0.0:
        eff_bin = resolve_bin_size(candles, params=params, is_daily_level=is_daily_level)

    min_p = min(x[1] for x in parsed)
    max_p = max(x[0] for x in parsed)
    if max_p <= min_p:
        max_p = min_p + eff_bin
    if (max_p - min_p) / eff_bin > 2000.0:
        eff_bin = (max_p - min_p) / 2000.0
    eff_bin = max(1e-8, eff_bin)

    start_bin = math.floor(min_p / eff_bin) * eff_bin
    end_bin = math.ceil(max_p / eff_bin) * eff_bin
    num_bins = max(1, min(2005, int(round((end_bin - start_bin) / eff_bin)) + 1))
    volume_bins = [0.0] * num_bins

    for h, l, v, _, _ in parsed:
        if v <= 0.0 or h < l:
            continue
        idx_low = max(0, min(num_bins - 1, int(math.floor((l - start_bin) / eff_bin + 1e-9))))
        idx_high = max(idx_low, min(num_bins - 1, int(math.floor((h - start_bin) / eff_bin + 1e-9))))
        span = max(1, idx_high - idx_low + 1)
        vol_per_bin = v / float(span)
        for b_idx in range(idx_low, idx_high + 1):
            volume_bins[b_idx] += vol_per_bin

    price_decimals = max(2, min(8, math.ceil(-math.log10(eff_bin)) + 1)) if eff_bin < 1.0 else 2
    total_vol = sum(volume_bins)
    if total_vol <= 0.0:
        return {
            "poc": round((min_p + max_p) / 2.0, price_decimals),
            "vah": round(max_p, price_decimals),
            "val": round(min_p, price_decimals),
            "total_volume": 0.0,
            "bins": [],
            "bin_size": eff_bin,
        }

    poc_idx = max(range(num_bins), key=lambda k: volume_bins[k])
    poc_price = round(start_bin + (poc_idx + 0.5) * eff_bin, price_decimals)

    va_pct = max(0.01, min(1.0, params.value_area_pct))
    target_va = total_vol * va_pct
    cur_va = volume_bins[poc_idx]
    up_idx = poc_idx
    down_idx = poc_idx
    while cur_va < target_va and (up_idx < num_bins - 1 or down_idx > 0):
        next_up = volume_bins[up_idx + 1] if up_idx < num_bins - 1 else 0.0
        next_down = volume_bins[down_idx - 1] if down_idx > 0 else 0.0
        if next_up >= next_down and up_idx < num_bins - 1:
            up_idx += 1
            cur_va += next_up
        elif down_idx > 0:
            down_idx -= 1
            cur_va += next_down
        elif up_idx < num_bins - 1:
            up_idx += 1
            cur_va += next_up
        else:
            break

    val_price = round(start_bin + down_idx * eff_bin, price_decimals)
    vah_price = round(start_bin + (up_idx + 1) * eff_bin, price_decimals)
    max_vol = volume_bins[poc_idx]
    bins_out = []
    for idx, vol in enumerate(volume_bins):
        if vol > 0.0:
            p_lo = round(start_bin + idx * eff_bin, price_decimals)
            p_hi = round(start_bin + (idx + 1) * eff_bin, price_decimals)
            bins_out.append({
                "price_low": p_lo,
                "price_high": p_hi,
                "price_center": round((p_lo + p_hi) / 2.0, price_decimals),
                "volume": round(vol, 4),
                "ratio": round(vol / max_vol, 4) if max_vol > 0.0 else 0.0,
            })

    return {
        "poc": poc_price,
        "vah": vah_price,
        "val": val_price,
        "total_volume": round(total_vol, 4),
        "bins": bins_out,
        "bin_size": eff_bin,
    }


def calculate_rolling_poc_series(
    candles: Sequence[Any],
    *,
    params: VolumeProfileParams,
    calendar: Optional[SessionCalendarProtocol] = None,
    symbol: str = "",
    bin_size: Optional[float] = None,
    is_daily_level: bool = False,
) -> List[Dict[str, Any]]:
    """Sliding-window Point of Control (POC) series with optional SessionCalendarProtocol reset."""
    raw_parsed = [x for x in (_extract_hlvct(c) for c in candles) if x is not None and x[4] > 0]
    if not raw_parsed:
        return []

    dedup: Dict[int, Tuple[float, float, float, float, int]] = {}
    for item in raw_parsed:
        dedup[item[4]] = item
    valid = [dedup[t] for t in sorted(dedup.keys())]

    eff_bin = safe_float(bin_size, 0.0)
    if eff_bin <= 0.0:
        eff_bin = resolve_bin_size(candles, params=params, is_daily_level=is_daily_level)
    eff_bin = max(1e-8, eff_bin)
    price_decimals = max(2, min(8, math.ceil(-math.log10(eff_bin)) + 1)) if eff_bin < 1.0 else 2

    candle_bins: List[List[Tuple[int, float]]] = []
    for h, l, v, _, _ in valid:
        if v <= 0.0 or h < l:
            candle_bins.append([])
            continue
        low_idx = int(math.floor(l / eff_bin + 1e-9))
        high_idx = int(math.floor(h / eff_bin + 1e-9))
        span = max(1, high_idx - low_idx + 1)
        if span > 2000:
            stride = span / 2000.0
            v_per = v / 2000.0
            candle_bins.append([(low_idx + int(k * stride), v_per) for k in range(2000)])
        else:
            v_per = v / float(span)
            candle_bins.append([(b, v_per) for b in range(low_idx, high_idx + 1)])

    counts: Dict[int, float] = {}
    poc_series: List[Dict[str, Any]] = []
    session_start_idx = 0
    cur_sess_key = None
    win = max(1, int(params.rolling_window))

    for i, (_, _, _, cl, ts) in enumerate(valid):
        if params.session_reset and calendar is not None:
            sess_key = (
                calendar.get_trading_day(float(ts), symbol=symbol),
                calendar.get_session_type(float(ts), symbol=symbol),
            )
            if cur_sess_key is not None and sess_key != cur_sess_key:
                counts.clear()
                session_start_idx = i
            cur_sess_key = sess_key

        for b, vb in candle_bins[i]:
            counts[b] = counts.get(b, 0.0) + vb

        if (i - session_start_idx) >= win:
            old_idx = i - win
            for b, vb in candle_bins[old_idx]:
                nv = counts.get(b, 0.0) - vb
                if nv <= 1e-7:
                    counts.pop(b, None)
                else:
                    counts[b] = nv

        if counts:
            best_b = max(counts, key=counts.__getitem__)
            poc_px = round((best_b + 0.5) * eff_bin, price_decimals)
        else:
            poc_px = round(cl, price_decimals)
        poc_series.append({"time": ts, "value": poc_px})

    return poc_series
