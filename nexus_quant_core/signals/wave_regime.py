from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from nexus_quant_core.contracts import (
    MacroRegimeSnapshot,
    SessionCalendarProtocol,
    TimeframeNormalizer,
    WaveRegimeParams,
)
from nexus_quant_core.math_utils import safe_float


class ElliottWaveRegimeFSM:
    """
    Causal 0-lookahead streaming Elliott Wave macro regime engine.
    Shared by nexus_quant_tw (TXF night/day futures) and nexus_quant_us (ES=F / NQ=F CME Globex futures),
    configured strictly via explicit WaveRegimeParams + SessionCalendarProtocol.
    """

    def __init__(
        self,
        *,
        params: WaveRegimeParams,
        calendar: SessionCalendarProtocol,
        symbol: str,
        timeframe: Optional[str] = None,
    ) -> None:
        if params.swing_threshold_pct <= 0.0 or not math.isfinite(params.swing_threshold_pct):
            raise ValueError(f"Invalid swing_threshold_pct: {params.swing_threshold_pct!r}")
        self.params = params
        self.calendar = calendar
        self.symbol = str(symbol).strip().upper()
        self.timeframe_seconds: Optional[float] = (
            float(TimeframeNormalizer.to_seconds(timeframe, default=int(params.default_bar_seconds)))
            if timeframe is not None
            else None
        )
        self.pivots: List[Dict[str, Any]] = []
        self._pivot_seq: int = 0
        self.cycle_stage_count: int = 0
        self._current_peak_id: Optional[int] = None
        self._wave1_peak_price: Optional[float] = None
        self._wave3_peak_price: Optional[float] = None
        self._last_ts: Optional[float] = None
        self._observed_interval: Optional[float] = None
        self.last_regime = MacroRegimeSnapshot(
            trend="NEUTRAL",
            wave_stage="CONSOLIDATION",
            fib_level=None,
            description="Macro wave engine initialized",
            updated_at_ts=0.0,
        )

    def reset(self) -> None:
        self.pivots.clear()
        self._pivot_seq = 0
        self.cycle_stage_count = 0
        self._current_peak_id = None
        self._wave1_peak_price = None
        self._wave3_peak_price = None
        self._last_ts = None
        self.last_regime = MacroRegimeSnapshot(
            trend="NEUTRAL",
            wave_stage="CONSOLIDATION",
            fib_level=None,
            description="Macro wave engine reset",
            updated_at_ts=0.0,
        )

    def _append_pivot(self, p_type: str, price: float, ts: float) -> None:
        self._pivot_seq += 1
        self.pivots.append({"type": p_type, "price": price, "time": ts, "id": self._pivot_seq})
        if len(self.pivots) > self.params.max_pivots:
            self.pivots = self.pivots[-self.params.max_pivots :]

    def _update_pivots(
        self,
        high: float,
        low: float,
        ts: float,
        open_price: Optional[float],
        close_price: float,
    ) -> bool:
        if not self.pivots:
            is_bearish = bool(open_price is not None and open_price > 0.0 and close_price < open_price)
            if is_bearish:
                self._append_pivot("PEAK", high, ts)
                self._append_pivot("VALLEY", low, ts)
            else:
                self._append_pivot("VALLEY", low, ts)
                self._append_pivot("PEAK", high, ts)
            return True

        last = self.pivots[-1]
        is_peak = last["type"] == "PEAK"
        prev_p = float(last["price"])
        th = self.params.swing_threshold_pct

        if is_peak:
            updated = False
            if high > prev_p:
                last["price"] = high
                last["time"] = max(float(last["time"]), ts)
                prev_p = high
                updated = True
            outside_drop = (
                (not updated)
                or (open_price is not None and close_price < open_price)
                or (close_price < (low + (high - low) * 0.4))
            )
            if outside_drop and prev_p > 0.0 and (((prev_p - low) / prev_p) * 100.0) >= th:
                if updated and len(self.pivots) >= 4:
                    p0, p1, p2, p3 = self.pivots[-4], self.pivots[-3], self.pivots[-2], self.pivots[-1]
                    if p3["type"] == "PEAK" and p1["type"] == "PEAK" and p3["price"] > p1["price"]:
                        self.last_regime = self._evaluate_bullish_impulse(p0, p1, p2, p3, ts)
                self._append_pivot("VALLEY", low, ts)
        else:
            updated = False
            if low < prev_p:
                last["price"] = low
                last["time"] = max(float(last["time"]), ts)
                prev_p = low
                updated = True
            outside_rise = (
                (not updated)
                or (open_price is not None and close_price > open_price)
                or (close_price > (low + (high - low) * 0.6))
            )
            if outside_rise and prev_p > 0.0 and (((high - prev_p) / prev_p) * 100.0) >= th:
                self._append_pivot("PEAK", high, ts)

        return False

    def _evaluate_bullish_impulse(
        self, p0: Dict[str, Any], p1: Dict[str, Any], p2: Dict[str, Any], p3: Dict[str, Any], ts: float
    ) -> MacroRegimeSnapshot:
        p0_p, p1_p, p2_p, p3_p = float(p0["price"]), float(p1["price"]), float(p2["price"]), float(p3["price"])
        w1_h = max(0.0, p1_p - p0_p)
        w2_pb = max(0.0, p1_p - p2_p)
        min_h = max(0.01, p0_p * 0.0001)
        fib_ratio = (w2_pb / w1_h) if w1_h >= min_h else 0.0

        peak_id = p3["id"]
        if self._current_peak_id == peak_id:
            return MacroRegimeSnapshot(
                trend=self.last_regime.trend,
                wave_stage=self.last_regime.wave_stage,
                fib_level=round(fib_ratio, 3),
                description=self.last_regime.description,
                updated_at_ts=ts,
            )

        self._current_peak_id = peak_id
        is_valid_fib = (
            p2_p > p0_p
            and w1_h >= min_h
            and (self.params.fib_min_retracement <= fib_ratio <= self.params.fib_max_retracement)
        )
        if not is_valid_fib:
            self.cycle_stage_count = 0
            self._current_peak_id = None
            self._wave1_peak_price = None
            self._wave3_peak_price = None
            return MacroRegimeSnapshot(
                trend="NEUTRAL",
                wave_stage="CONSOLIDATION",
                fib_level=round(fib_ratio, 3),
                description=f"[{self.symbol}] Invalid Fib retracement ({fib_ratio * 100:.1f}%)",
                updated_at_ts=ts,
            )

        if self.cycle_stage_count < 3 or self.last_regime.trend != "BULLISH":
            self.cycle_stage_count = 3
            self._wave1_peak_price = p1_p
            self._wave3_peak_price = p3_p
            return MacroRegimeSnapshot(
                trend="BULLISH",
                wave_stage="IMPULSE_WAVE_3",
                fib_level=round(fib_ratio, 3),
                description=f"[{self.symbol}] Impulse Wave 3 breakout (Wave 2 Fib {fib_ratio * 100:.1f}%)",
                updated_at_ts=ts,
            )

        if self._wave1_peak_price is not None and p2_p <= self._wave1_peak_price:
            self.cycle_stage_count = 0
            self._current_peak_id = None
            self._wave1_peak_price = None
            self._wave3_peak_price = None
            return MacroRegimeSnapshot(
                trend="NEUTRAL",
                wave_stage="CONSOLIDATION",
                fib_level=round(fib_ratio, 3),
                description=f"[{self.symbol}] Wave 4 overlapped Wave 1 peak; structure reset",
                updated_at_ts=ts,
            )

        if self._wave3_peak_price is not None and p3_p <= self._wave3_peak_price:
            return MacroRegimeSnapshot(
                trend="BULLISH",
                wave_stage="CORRECTIVE_WAVE_4",
                fib_level=round(fib_ratio, 3),
                description=f"[{self.symbol}] Wave 4 consolidation below Wave 3 peak ({self._wave3_peak_price:.2f})",
                updated_at_ts=ts,
            )

        self.cycle_stage_count = 5
        return MacroRegimeSnapshot(
            trend="BULLISH",
            wave_stage="IMPULSE_WAVE_5",
            fib_level=round(fib_ratio, 3),
            description=f"[{self.symbol}] Late Impulse Wave 5 extension",
            updated_at_ts=ts,
        )

    def process_candle(
        self,
        *,
        high: float,
        low: float,
        close: float,
        timestamp: float,
        open_price: Optional[float] = None,
    ) -> MacroRegimeSnapshot:
        ts = safe_float(timestamp, 0.0)
        sh = safe_float(high, 0.0)
        sl = safe_float(low, 0.0)
        sc = safe_float(close, 0.0)
        so = safe_float(open_price, sc) if open_price is not None else None
        if ts <= 0.0 or sh <= 0.0 or sl <= 0.0 or sc <= 0.0:
            return self.last_regime
        if sh < sl:
            sh, sl = sl, sh
        sc = max(sl, min(sh, sc))

        if self._last_ts is not None:
            if ts < self._last_ts:
                return self.last_regime
            gap_sec = ts - self._last_ts
            expected_sec = (
                self.timeframe_seconds
                if self.timeframe_seconds is not None
                else (self._observed_interval or self.params.default_bar_seconds)
            )
            if 0.0 < gap_sec < 86400.0 * 7.0:
                if self._observed_interval is None or gap_sec < self._observed_interval:
                    self._observed_interval = gap_sec

            gap_threshold = max(3600.0, expected_sec * self.params.gap_mult)
            if gap_sec > gap_threshold:
                active_limit = max(900.0, expected_sec * self.params.holiday_active_mult)
                active_sec = self.calendar.get_session_active_seconds(self._last_ts, ts, symbol=self.symbol)
                if active_sec > active_limit:
                    self.reset()

        self._last_ts = ts
        if self._update_pivots(sh, sl, ts, so, sc):
            return self.last_regime

        if len(self.pivots) >= 4:
            p0, p1, p2, p3 = self.pivots[-4], self.pivots[-3], self.pivots[-2], self.pivots[-1]
            if p3["type"] == "PEAK" and p1["type"] == "PEAK" and p3["price"] > p1["price"] and p0["type"] == "VALLEY":
                self.last_regime = self._evaluate_bullish_impulse(p0, p1, p2, p3, ts)
            elif p3["type"] == "VALLEY" and p1["type"] == "VALLEY" and p3["price"] < p1["price"]:
                self.cycle_stage_count = 0
                self._current_peak_id = None
                self._wave1_peak_price = None
                self._wave3_peak_price = None
                self.last_regime = MacroRegimeSnapshot(
                    trend="BEARISH",
                    wave_stage="CORRECTIVE_WAVE_C",
                    fib_level=None,
                    description=f"[{self.symbol}] Bearish Wave C / Impulse Wave 3 breakdown",
                    updated_at_ts=ts,
                )
            elif p3["type"] == "PEAK" and p1["type"] == "PEAK" and p3["price"] <= p1["price"]:
                if self.last_regime.trend == "BULLISH":
                    self.last_regime = MacroRegimeSnapshot(
                        trend=self.last_regime.trend,
                        wave_stage=self.last_regime.wave_stage,
                        fib_level=self.last_regime.fib_level,
                        description=self.last_regime.description,
                        updated_at_ts=ts,
                    )
                else:
                    self.cycle_stage_count = 0
                    trend = "BEARISH" if self.last_regime.trend == "BEARISH" else "NEUTRAL"
                    stage = "CORRECTIVE_WAVE_C" if trend == "BEARISH" else "CONSOLIDATION"
                    self.last_regime = MacroRegimeSnapshot(
                        trend=trend,
                        wave_stage=stage,
                        fib_level=None,
                        description=f"[{self.symbol}] Lower high / resistance hold",
                        updated_at_ts=ts,
                    )
            elif p3["type"] == "VALLEY" and p1["type"] == "VALLEY" and p3["price"] >= p1["price"]:
                if self.last_regime.trend == "BEARISH":
                    self.last_regime = MacroRegimeSnapshot(
                        trend="BEARISH",
                        wave_stage=self.last_regime.wave_stage,
                        fib_level=None,
                        description=self.last_regime.description,
                        updated_at_ts=ts,
                    )
                elif self.cycle_stage_count == 3 and self._wave1_peak_price is not None:
                    if p3["price"] <= self._wave1_peak_price:
                        self.cycle_stage_count = 0
                        self._wave1_peak_price = None
                        self._wave3_peak_price = None
                        self.last_regime = MacroRegimeSnapshot(
                            trend="NEUTRAL",
                            wave_stage="CONSOLIDATION",
                            fib_level=None,
                            description=f"[{self.symbol}] Wave 4 broke Wave 1 peak; reset",
                            updated_at_ts=ts,
                        )
                    else:
                        self.last_regime = MacroRegimeSnapshot(
                            trend="BULLISH",
                            wave_stage="CORRECTIVE_WAVE_4",
                            fib_level=None,
                            description=f"[{self.symbol}] Healthy Wave 4 higher-low pullback",
                            updated_at_ts=ts,
                        )
                else:
                    self.last_regime = MacroRegimeSnapshot(
                        trend="NEUTRAL",
                        wave_stage="CONSOLIDATION",
                        fib_level=None,
                        description=f"[{self.symbol}] Higher-low support consolidation",
                        updated_at_ts=ts,
                    )

        return self.last_regime
