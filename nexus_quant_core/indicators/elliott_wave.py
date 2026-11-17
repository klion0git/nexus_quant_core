from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nexus_quant_core.contracts import TimeframeNormalizer, WaveOverlayParams
from nexus_quant_core.math_utils import round_adaptive_price, safe_float, safe_int


def _extract_hlct(c: Any) -> Optional[Tuple[float, float, float, int]]:
    if c is None or isinstance(c, bool):
        return None
    if isinstance(c, (tuple, list)) and len(c) >= 3:
        raw_h, raw_l, raw_c = c[0], c[1], c[2]
        raw_t = c[3] if len(c) >= 4 else 0
    elif isinstance(c, dict):
        raw_h = c.get("high", c.get("h", c.get("High")))
        raw_l = c.get("low", c.get("l", c.get("Low")))
        raw_c = c.get("close", c.get("c", c.get("Close")))
        raw_t = c.get("timestamp", c.get("ts", c.get("time", c.get("t", 0))))
    else:
        raw_h = getattr(c, "high", getattr(c, "h", None))
        raw_l = getattr(c, "low", getattr(c, "l", None))
        raw_c = getattr(c, "close", getattr(c, "c", None))
        raw_t = getattr(c, "timestamp", getattr(c, "ts", getattr(c, "time", 0)))

    fh = safe_float(raw_h, -1.0)
    fl = safe_float(raw_l, -1.0)
    fc = safe_float(raw_c if raw_c is not None else fl, fl)
    ts = safe_int(raw_t, -1)
    if fh >= fl > 0.0 and fc > 0.0 and ts >= 0:
        return (fh, fl, fc, ts)
    return None


def calculate_dynamic_threshold(
    candles: Sequence[Any],
    *,
    timeframe: str,
    params: WaveOverlayParams,
) -> float:
    """Compute adaptive ZigZag swing filter percentage bounded by explicit WaveOverlayParams."""
    tf = TimeframeNormalizer.normalize_chart_timeframe(timeframe)
    base_pct, max_pct = params.threshold_bounds_by_timeframe[tf]
    parsed = [x for x in (_extract_hlct(c) for c in candles) if x is not None]
    if not parsed:
        return base_pct

    recent = parsed[-min(len(parsed), 120) :]
    avg_range = sum(item[0] - item[1] for item in recent) / float(len(recent))
    closes = [item[2] for item in recent if item[2] > 0.0]
    avg_close = sum(closes) / float(len(closes)) if closes else 100.0
    vol_pct = (avg_range / avg_close) * params.volatility_multiplier if avg_close > 0.0 else base_pct
    if not math.isfinite(vol_pct):
        vol_pct = base_pct
    return max(base_pct, min(max_pct, vol_pct))


def find_zigzag_pivots(
    candles: Sequence[Any],
    *,
    timeframe: str,
    params: WaveOverlayParams,
    window: Optional[int] = None,
    pct_filter: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Extract strictly alternating ZigZag High/Low pivots with bilateral window confirmation."""
    tf = TimeframeNormalizer.normalize_chart_timeframe(timeframe)
    parsed = [x for x in (_extract_hlct(c) for c in candles) if x is not None]
    parsed.sort(key=lambda x: x[3])
    n = len(parsed)
    if n < 5:
        return []

    eff_win = int(window) if (window is not None and window > 0) else params.window_by_timeframe[tf]
    eff_win = min(eff_win, max(1, (n - 3) // 2))
    eff_pct = pct_filter if (pct_filter is not None and pct_filter > 0.0) else calculate_dynamic_threshold(
        parsed, timeframe=tf, params=params
    )

    highs = [item[0] for item in parsed]
    lows = [item[1] for item in parsed]
    times = [item[3] for item in parsed]

    raw_pivots: List[Dict[str, Any]] = []
    for i in range(eff_win, n - eff_win):
        is_high = all(highs[i] >= highs[i - w] and highs[i] >= highs[i + w] for w in range(1, eff_win + 1))
        is_low = all(lows[i] <= lows[i - w] and lows[i] <= lows[i + w] for w in range(1, eff_win + 1))
        if is_high and not is_low:
            raw_pivots.append({"index": i, "time": times[i], "price": highs[i], "type": "high"})
        elif is_low and not is_high:
            raw_pivots.append({"index": i, "time": times[i], "price": lows[i], "type": "low"})

    if not raw_pivots:
        return []

    alternating: List[Dict[str, Any]] = [raw_pivots[0]]
    for p in raw_pivots[1:]:
        last = alternating[-1]
        if p["type"] == last["type"]:
            if p["type"] == "high" and p["price"] > last["price"]:
                alternating[-1] = p
            elif p["type"] == "low" and p["price"] < last["price"]:
                alternating[-1] = p
        else:
            change = abs(p["price"] - last["price"]) / max(1e-9, last["price"])
            if change >= eff_pct:
                alternating.append(p)
    return alternating


def identify_elliott_waves(
    pivots: Sequence[Dict[str, Any]],
    *,
    wave4_overlap_tolerance: float,
) -> List[Dict[str, Any]]:
    """Identify 1-2-3-4-5 (+ optional A-B-C) Elliott Wave structures from alternating pivots."""
    waves: List[Dict[str, Any]] = []
    i = 0
    n = len(pivots)
    tol = max(0.0, float(wave4_overlap_tolerance))

    while i <= n - 6:
        pts6 = pivots[i : i + 6]
        types6 = [str(p.get("type", "")).lower() for p in pts6]
        px = [safe_float(p.get("price"), 0.0) for p in pts6]

        if types6 == ["low", "high", "low", "high", "low", "high"]:
            l1, l2, l3, l4, l5 = px[1] - px[0], px[1] - px[2], px[3] - px[2], px[3] - px[4], px[5] - px[4]
            if (
                l1 > 0 and l2 > 0 and l3 > 0 and l4 > 0 and l5 > 0
                and px[2] > px[0]
                and px[3] > px[1]
                and px[4] > px[2]
                and px[4] >= px[1] * (1.0 - tol)
                and px[5] > px[4]
                and not (l3 < l1 and l3 < l5)
            ):
                w_dict: Dict[str, Any] = {
                    "direction": "bullish",
                    "points": {str(k): pts6[k] for k in range(6)},
                }
                step = 5
                if i + 8 < n:
                    pa, pb, pc = pivots[i + 6], pivots[i + 7], pivots[i + 8]
                    if [str(x.get("type", "")).lower() for x in (pa, pb, pc)] == ["low", "high", "low"]:
                        pxa, pxb, pxc = safe_float(pa["price"]), safe_float(pb["price"]), safe_float(pc["price"])
                        if pxa < px[5] and px[5] > pxb > pxa and pxc < pxb:
                            w_dict["points"].update({"A": pa, "B": pb, "C": pc})
                            step = 8
                waves.append(w_dict)
                i += step
                continue

        elif types6 == ["high", "low", "high", "low", "high", "low"]:
            l1, l2, l3, l4, l5 = px[0] - px[1], px[2] - px[1], px[2] - px[3], px[4] - px[3], px[4] - px[5]
            if (
                l1 > 0 and l2 > 0 and l3 > 0 and l4 > 0 and l5 > 0
                and px[2] < px[0]
                and px[3] < px[1]
                and px[4] < px[2]
                and px[4] <= px[1] * (1.0 + tol)
                and px[5] < px[4]
                and not (l3 < l1 and l3 < l5)
            ):
                w_dict = {
                    "direction": "bearish",
                    "points": {str(k): pts6[k] for k in range(6)},
                }
                step = 5
                if i + 8 < n:
                    pa, pb, pc = pivots[i + 6], pivots[i + 7], pivots[i + 8]
                    if [str(x.get("type", "")).lower() for x in (pa, pb, pc)] == ["high", "low", "high"]:
                        pxa, pxb, pxc = safe_float(pa["price"]), safe_float(pb["price"]), safe_float(pc["price"])
                        if pxa > px[5] and px[5] < pxb < pxa and pxc > pxb:
                            w_dict["points"].update({"A": pa, "B": pb, "C": pc})
                            step = 8
                waves.append(w_dict)
                i += step
                continue

        i += 1
    return waves


def calculate_elliott_wave_overlays(
    candles: Sequence[Any],
    *,
    timeframe: str,
    params: WaveOverlayParams,
    pivots: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Produce Lightweight-Charts-compatible Elliott Wave lines and markers."""
    tf = TimeframeNormalizer.normalize_chart_timeframe(timeframe)
    eff_pivots = list(pivots) if pivots is not None else find_zigzag_pivots(candles, timeframe=tf, params=params)
    if not eff_pivots:
        return {"lines": [], "markers": [], "wave_count": 0}

    waves = identify_elliott_waves(eff_pivots, wave4_overlap_tolerance=params.wave4_overlap_tolerance)
    lines: List[Dict[str, Any]] = []
    markers: List[Dict[str, Any]] = []
    used_times = set()

    for w in waves:
        pts = w["points"]
        direction = w["direction"]
        for k in ("0", "1", "2", "3", "4", "5", "A", "B", "C"):
            if k not in pts:
                continue
            pt = pts[k]
            t = safe_int(pt.get("time"), 0)
            p = round_adaptive_price(safe_float(pt.get("price"), 0.0))
            if t <= 0 or p <= 0.0:
                continue
            lines.append({"time": t, "value": p})
            if k == "0" or t in used_times:
                continue
            used_times.add(t)
            is_impulse = k in ("1", "2", "3", "4", "5")
            if direction == "bullish":
                color = "#2563EB" if is_impulse else "#EA580C"
                pos = "aboveBar" if k in ("1", "3", "5", "B") else "belowBar"
            else:
                color = "#DC2626" if is_impulse else "#059669"
                pos = "belowBar" if k in ("1", "3", "5", "B") else "aboveBar"
            markers.append({
                "time": t,
                "position": pos,
                "color": color,
                "shape": "circle" if is_impulse else "square",
                "text": f"({k})",
                "size": 1,
            })

    if not lines and len(eff_pivots) >= 3:
        for pt in eff_pivots:
            t = safe_int(pt.get("time"), 0)
            p = round_adaptive_price(safe_float(pt.get("price"), 0.0))
            if t > 0 and p > 0.0:
                lines.append({"time": t, "value": p})

    dedup_lines: Dict[int, Dict[str, Any]] = {}
    for item in sorted(lines, key=lambda x: x["time"]):
        t = item["time"]
        if t not in dedup_lines or abs(item["value"]) > abs(dedup_lines[t]["value"]):
            dedup_lines[t] = item

    markers.sort(key=lambda x: x["time"])
    return {
        "lines": [dedup_lines[t] for t in sorted(dedup_lines.keys())],
        "markers": markers,
        "wave_count": len(waves),
    }
