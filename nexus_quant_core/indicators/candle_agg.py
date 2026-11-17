from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from nexus_quant_core.contracts import SessionCalendarProtocol
from nexus_quant_core.indicators.moving_avg import calculate_atr, calculate_linear_weighted_vwap
from nexus_quant_core.math_utils import safe_float


class SessionCandleAggregator:
    """
    Thread-safe real-time candlestick aggregator driven purely by an injected SessionCalendarProtocol.
    Contains ZERO market-name conditionals or wall-clock fallbacks.
    """

    def __init__(
        self,
        calendar: SessionCalendarProtocol,
        *,
        window_seconds: int,
        max_bars: int,
    ) -> None:
        if window_seconds <= 0 or isinstance(window_seconds, bool):
            raise ValueError(f"window_seconds must be > 0, got {window_seconds!r}")
        if max_bars <= 0 or isinstance(max_bars, bool):
            raise ValueError(f"max_bars must be > 0, got {max_bars!r}")
        self.calendar = calendar
        self.window_seconds = int(window_seconds)
        self.max_bars = int(max_bars)
        self.history_bars: Dict[str, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.max_bars)
        )
        self.current_bars: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def update_candle(
        self,
        symbol: str,
        price: float,
        timestamp: float,
        volume: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Update current bar; return sealed previous bar dict when crossing into a new bar label."""
        clean_sym = str(symbol or "").strip().upper()
        px = safe_float(price, 0.0)
        ts = safe_float(timestamp, 0.0)
        vol = max(0.0, safe_float(volume, 0.0))
        if not clean_sym or px <= 0.0 or ts <= 0.0:
            return None

        bar_label = self.calendar.get_session_bar_label(ts, self.window_seconds, symbol=clean_sym)
        td_str = self.calendar.get_trading_day(ts, symbol=clean_sym).isoformat()

        with self._lock:
            cur = self.current_bars.get(clean_sym)
            if cur is None:
                self.current_bars[clean_sym] = {
                    "symbol": clean_sym,
                    "ts": bar_label,
                    "trading_date": td_str,
                    "o": px,
                    "h": px,
                    "l": px,
                    "c": px,
                    "v": vol,
                }
                return None

            # Out-of-order stale tick older than current bar label is ignored
            if bar_label < cur["ts"]:
                return None

            if bar_label == cur["ts"]:
                cur["h"] = max(cur["h"], px)
                cur["l"] = min(cur["l"], px)
                cur["c"] = px
                cur["v"] += vol
                return None

            sealed = dict(cur)
            self.history_bars[clean_sym].append(sealed)
            self.current_bars[clean_sym] = {
                "symbol": clean_sym,
                "ts": bar_label,
                "trading_date": td_str,
                "o": px,
                "h": px,
                "l": px,
                "c": px,
                "v": vol,
            }
            return sealed

    def get_current_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        clean_sym = str(symbol or "").strip().upper()
        with self._lock:
            cur = self.current_bars.get(clean_sym)
            return dict(cur) if cur is not None else None

    def seal_current_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        clean_sym = str(symbol or "").strip().upper()
        with self._lock:
            cur = self.current_bars.pop(clean_sym, None)
            if cur is None:
                return None
            sealed = dict(cur)
            self.history_bars[clean_sym].append(sealed)
            return sealed

    def reset_session(self, symbol: Optional[str] = None) -> None:
        """Clear unsealed bar(s) while preserving historical closed bars."""
        with self._lock:
            if symbol:
                self.current_bars.pop(str(symbol).strip().upper(), None)
            else:
                self.current_bars.clear()

    def get_latest_atr(self, symbol: str, *, period: int) -> float:
        clean_sym = str(symbol or "").strip().upper()
        with self._lock:
            bars = [b for b in self.history_bars.get(clean_sym, []) if not b.get("is_gap_fill", False)]
            cur = self.current_bars.get(clean_sym)
            if cur is not None:
                bars = bars + [cur]
        if len(bars) < 2:
            return 0.0
        highs = [safe_float(b["h"], 0.0) for b in bars]
        lows = [safe_float(b["l"], 0.0) for b in bars]
        closes = [safe_float(b["c"], 0.0) for b in bars]
        return calculate_atr(highs, lows, closes, period=period)

    def get_rolling_vwap(self, symbol: str, *, n_bars: int) -> float:
        clean_sym = str(symbol or "").strip().upper()
        with self._lock:
            bars = [b for b in self.history_bars.get(clean_sym, []) if not b.get("is_gap_fill", False)]
            cur = self.current_bars.get(clean_sym)
            if cur is not None:
                bars = bars + [cur]
        if not bars:
            return 0.0
        return calculate_linear_weighted_vwap(bars[-max(1, int(n_bars)) :])

    def resample_bars(
        self,
        bars: List[Dict[str, Any]],
        *,
        target_window_seconds: int,
        symbol: str,
    ) -> List[Dict[str, Any]]:
        """Aggregate sorted 15m/intraday bars into higher session-aligned bars (e.g. 1h, 4h)."""
        if not bars:
            return []
        grouped: Dict[int, Dict[str, Any]] = {}
        for b in sorted(bars, key=lambda x: int(safe_float(x.get("timestamp", x.get("ts", 0)), 0.0))):
            ts = int(safe_float(b.get("timestamp", b.get("ts", 0)), 0.0))
            if ts <= 0:
                continue
            label = self.calendar.get_session_bar_label(float(ts), target_window_seconds, symbol=symbol)
            td_str = str(b.get("trading_date") or self.calendar.get_trading_day(float(ts), symbol=symbol).isoformat())
            o = safe_float(b.get("open", b.get("o", 0.0)), 0.0)
            h = safe_float(b.get("high", b.get("h", o)), o)
            l = safe_float(b.get("low", b.get("l", o)), o)
            c = safe_float(b.get("close", b.get("c", o)), o)
            v = max(0.0, safe_float(b.get("volume", b.get("v", 0.0)), 0.0))
            if c <= 0.0 or not math.isfinite(c):
                continue
            if label not in grouped:
                grouped[label] = {
                    "symbol": symbol,
                    "timestamp": label,
                    "trading_date": td_str,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                }
            else:
                acc = grouped[label]
                acc["high"] = max(acc["high"], h)
                acc["low"] = min(acc["low"], l)
                acc["close"] = c
                acc["volume"] += v
        return [grouped[k] for k in sorted(grouped.keys())]
