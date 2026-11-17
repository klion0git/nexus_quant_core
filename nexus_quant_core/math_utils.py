from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any, List, Optional


_NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely parse finite float; reject None, bool, mocks, NaN, and Inf."""
    if val is None or isinstance(val, bool):
        return default
    if hasattr(val, "_mock_methods") or hasattr(val, "_mock_name") or type(val).__name__ in ("MagicMock", "Mock", "AsyncMock"):
        return default
    if isinstance(val, (int, float)):
        f = float(val)
        return f if math.isfinite(f) else default
    try:
        s = str(val).strip().replace(",", "")
        if not s or (s.startswith("<") and s.endswith(">")):
            return default
        try:
            f = float(s)
            return f if math.isfinite(f) else default
        except ValueError:
            m = _NUM_RE.search(s)
            if m:
                f = float(m.group(0))
                return f if math.isfinite(f) else default
            return default
    except (ValueError, TypeError, OverflowError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    """Safely parse finite integer; reject None, bool, NaN, and Inf."""
    if val is None or isinstance(val, bool):
        return default
    if isinstance(val, int):
        return val
    f = safe_float(val, float("nan"))
    if not math.isfinite(f):
        return default
    try:
        return int(f)
    except (ValueError, OverflowError):
        return default


def safe_pct_change(curr: Optional[float], prev: Optional[float]) -> float:
    """Percentage change ((curr - prev) / abs(prev)) * 100 with zero/NaN protection."""
    if curr is None or prev is None or isinstance(curr, bool) or isinstance(prev, bool):
        return 0.0
    try:
        c = float(curr)
        p = float(prev)
        if not math.isfinite(c) or not math.isfinite(p) or p == 0.0:
            return 0.0
        res = ((c - p) / abs(p)) * 100.0
        return res if math.isfinite(res) else 0.0
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return 0.0


def round_adaptive_price(val: float) -> float:
    """Adaptive rounding preserving sub-dollar and penny precision."""
    f = safe_float(val, 0.0)
    av = abs(f)
    if av == 0.0:
        return 0.0
    if av >= 1.0:
        return round(f, 2)
    if av >= 0.01:
        return round(f, 4)
    if av >= 0.0001:
        return round(f, 6)
    return round(f, 8)


def clean_finite_series(series: Any) -> List[float]:
    """Filter iterable into finite floats, rejecting strings, dicts, and booleans."""
    if series is None or isinstance(series, (str, bytes, bytearray, dict, bool)) or not isinstance(series, Iterable):
        return []
    out: List[float] = []
    for x in series:
        if x is None or isinstance(x, bool):
            continue
        try:
            fx = float(x)
            if math.isfinite(fx):
                out.append(fx)
        except (ValueError, TypeError):
            continue
    return out
