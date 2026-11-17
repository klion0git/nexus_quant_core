from __future__ import annotations

import datetime
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Protocol, Tuple, runtime_checkable


# ==============================================================================
# Canonical 7-Timeframe Specification (Eliminates '1m' vs '1mo' / '1M' Ambiguity)
# ==============================================================================
VALID_CHART_TIMEFRAMES: FrozenSet[str] = frozenset(
    {"15m", "1h", "4h", "1d", "1wk", "1mo", "3mo"}
)
INTRADAY_TIMEFRAMES: FrozenSet[str] = frozenset({"15m", "1h", "4h"})
CALENDAR_TIMEFRAMES: FrozenSet[str] = frozenset({"1d", "1wk", "1mo", "3mo"})


class TimeframeNormalizer:
    """Case-safe timeframe canonicalizer preserving uppercase '1M'->'1mo' & '3M'->'3mo'."""

    _ALIAS_MAP: Dict[str, str] = {
        "15m": "15m",
        "15min": "15m",
        "1h": "1h",
        "60m": "1h",
        "60min": "1h",
        "4h": "4h",
        "240m": "4h",
        "1d": "1d",
        "d": "1d",
        "day": "1d",
        "daily": "1d",
        "1wk": "1wk",
        "1w": "1wk",
        "w": "1wk",
        "week": "1wk",
        "weekly": "1wk",
        "1mo": "1mo",
        "mo": "1mo",
        "month": "1mo",
        "monthly": "1mo",
        "1mth": "1mo",
        "mth": "1mo",
        "3mo": "3mo",
        "3mth": "3mo",
        "quarter": "3mo",
        "quarterly": "3mo",
        "q": "3mo",
        "1q": "3mo",
    }

    _TF_TO_SEC: Dict[str, int] = {
        "60s": 60,
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "1d": 86400,
        "1wk": 604800,
        "1mo": 2592000,
        "3mo": 7776000,
    }

    @classmethod
    def normalize_chart_timeframe(cls, tf: Any) -> str:
        """Strictly normalize into the 7 canonical chart timeframes; raise ValueError otherwise."""
        if tf is None or isinstance(tf, bool) or not isinstance(tf, str):
            raise ValueError(f"Invalid chart timeframe: {tf!r}")
        raw = tf.strip()
        if not raw:
            raise ValueError("Empty chart timeframe")
        if raw in ("1M", "M"):
            return "1mo"
        if raw in ("3M", "Q", "1Q"):
            return "3mo"
        low = raw.lower()
        # Explicitly forbid '1m' and bare 'm' in chart databases to prevent minute/month mix-ups
        if low in ("1m", "m"):
            raise ValueError(
                f"Ambiguous or unsupported chart timeframe {tf!r}; allowed: {sorted(VALID_CHART_TIMEFRAMES)}"
            )
        norm = cls._ALIAS_MAP.get(low, low)
        if norm not in VALID_CHART_TIMEFRAMES:
            raise ValueError(
                f"Unsupported chart timeframe {tf!r} -> {norm!r}; allowed: {sorted(VALID_CHART_TIMEFRAMES)}"
            )
        return norm

    @classmethod
    def to_seconds(cls, tf: Any, *, default: int) -> int:
        """Convert timeframe code or integer seconds to duration in seconds (case-safe for '1M'/'3M')."""
        if tf is None or isinstance(tf, bool):
            return default
        if isinstance(tf, (int, float)):
            return max(1, int(tf)) if (math.isfinite(tf) and tf > 0) else default
        if not isinstance(tf, str):
            return default
        raw = tf.strip()
        if not raw:
            return default
        if raw in ("1M", "M"):
            return 2592000
        if raw in ("3M", "Q", "1Q"):
            return 7776000
        low = raw.lower()
        if low in cls._ALIAS_MAP:
            return cls._TF_TO_SEC[cls._ALIAS_MAP[low]]
        if low in cls._TF_TO_SEC:
            return cls._TF_TO_SEC[low]
        if low.isdigit():
            val = int(low)
            return val if val > 0 else default
        m = re.match(r"^(\d+)\s*(s|sec|m|min|h|hr|d|day|w|wk|mo|mth|month)s?$", low)
        if m:
            val = int(m.group(1))
            if val <= 0:
                return default
            mult = {
                "s": 1,
                "sec": 1,
                "m": 60,
                "min": 60,
                "h": 3600,
                "hr": 3600,
                "d": 86400,
                "day": 86400,
                "w": 604800,
                "wk": 604800,
                "mo": 2592000,
                "mth": 2592000,
                "month": 2592000,
            }[m.group(2)]
            return val * mult
        return default


# ==============================================================================
# Pure Protocols (Injected by nexus_quant_tw or nexus_quant_us)
# ==============================================================================
@runtime_checkable
class ClockProtocol(Protocol):
    def now_epoch(self) -> float: ...
    def now_monotonic(self) -> float: ...


@runtime_checkable
class SessionCalendarProtocol(Protocol):
    """Market-specific calendar & session boundary contract."""

    @property
    def timezone(self) -> datetime.tzinfo: ...

    def get_trading_day(self, utc_ts: float, symbol: str = "") -> datetime.date: ...

    def get_session_type(self, utc_ts: float, symbol: str = "") -> str: ...

    def is_trading_day(self, d: datetime.date, symbol: str = "") -> bool: ...

    def roll_business_day(self, d: datetime.date, step_days: int, symbol: str = "") -> datetime.date: ...

    def trading_days_between(self, older_ts_or_date: Any, newer_ts_or_date: Any, symbol: str = "") -> int: ...

    def get_session_bar_label(self, utc_ts: float, window_seconds: int, symbol: str = "") -> int: ...

    def get_bar_close_ts(self, bar_label_ts: int, window_seconds: int, symbol: str = "") -> int: ...

    def get_session_active_seconds(self, start_ts: float, end_ts: float, symbol: str = "") -> float: ...


@runtime_checkable
class BrokerFeedProtocol(Protocol):
    """Pluggable real-time & historical market feed contract."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def fetch_daily_candles(self, symbol: str, days: int) -> List[CandleBar]: ...
    async def fetch_15m_candles(self, symbol: str, days: int) -> List[CandleBar]: ...
    async def fetch_macro_futures_candles(self, symbol: str, days: int) -> List[CandleBar]: ...


# ==============================================================================
# Pure Market-Agnostic Value Objects
# ==============================================================================
@dataclass(frozen=True, kw_only=True)
class CandleBar:
    symbol: str
    timeframe: str
    timestamp: int          # Canonical UTC epoch seconds (bar start label for intraday, 00:00Z anchor for calendar)
    trading_date: str       # Canonical 'YYYY-MM-DD' discrete trading date in target market calendar
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_gap_fill: bool = False


@dataclass(frozen=True, kw_only=True)
class MarketTick:
    symbol: str
    price: float
    volume: float
    timestamp: float        # Canonical UTC epoch seconds
    bid: float = 0.0
    ask: float = 0.0
    cum_volume: float = 0.0
    cum_value: float = 0.0
    source: str = ""


@dataclass(frozen=True, kw_only=True)
class MacroRegimeSnapshot:
    trend: str              # 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    wave_stage: str         # 'IMPULSE_WAVE_3' | 'CORRECTIVE_WAVE_4' | 'IMPULSE_WAVE_5' | 'CORRECTIVE_WAVE_C' | 'CONSOLIDATION' | 'UNKNOWN'
    fib_level: Optional[float]
    description: str
    updated_at_ts: float


@dataclass(frozen=True, kw_only=True)
class SignalEvent:
    symbol: str
    side: str               # 'BUY' | 'SELL'
    action: str             # 'OPEN' | 'CLOSE'
    price: float
    stop_loss_price: float
    entry_type: str
    entry_subtype: str
    reason: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class OrderInstruction:
    symbol: str
    side: str               # 'BUY' | 'SELL'
    action: str             # 'OPEN' | 'CLOSE'
    quantity: float
    price: float
    stop_loss_price: float
    reason: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Explicit Zero-Default Parameter Specs (Must be supplied by nexus_quant_tw / nexus_quant_us)
# ==============================================================================
@dataclass(frozen=True, kw_only=True)
class FirFilterParams:
    stage1_weights: Tuple[float, ...]
    stage2_weights: Tuple[float, ...]
    deadzone_eps: float


@dataclass(frozen=True, kw_only=True)
class BreakoutFsmParams:
    bars_per_day_15m: int           # TW = 18, US = 26
    vwap_window: int                # TW = 360 (20d), US = 520 (20d)
    ma_short_window: int            # TW = 90  (5d),  US = 130 (5d)
    ma_mid_window: int              # TW = 360 (20d), US = 520 (20d)
    ma_long_window: int             # TW = 1080(60d), US = 1560(60d)
    ignition_lookback: int          # TW = 95  (5d),  US = 130 (5d)
    fsm_timeout_bars: int           # TW = 95  (5d),  US = 130 (5d)
    slope_lookback: int             # TW = 54  (3d),  US = 78  (3d)
    btx_high_lookback: int          # TW = 360 (20d), US = 520 (20d)
    btx_drawdown_ratio: float       # TW = 0.90,      US = 0.88
    btx_mfo_vel_threshold: float    # e.g. -0.30 (TW) / -0.35 (US)
    btx_mfo_max_divergence_pct: float
    bias20_min: float
    bias20_max: float
    min_submersion_bars: int
    min_submersion_pct: float


@dataclass(frozen=True, kw_only=True)
class MorphologyParams:
    atr_period: int
    ma_fast: int
    ma_mid: int
    ma_month: int
    ma_quarter: int
    vol_ma_period: int
    cap_down_days: int
    cap_bias20_max: float
    cap_vol_ratio_min: float
    dbl_lookback_min: int
    dbl_lookback_max: int
    dbl_diff_tol: float
    dbl_neck_min: float
    dbl_vol_ratio_min: float
    dbl_shadow_min: float
    dbl_bias20_min: float
    dbl_bias20_max: float
    dip_bias60_abs_max: float
    dip_a_bias20_max: float
    dip_b_bias20_max: float
    dip_c_pullback_max: float
    dip_d_pullback_max: float
    sqz_spread4_max: float
    sqz_bias60_abs_max: float
    sqz_vol_ratio_min: float
    masb_squeeze_threshold_pct: float
    masb_vel_min: float
    masb_acc_min: float
    masb_vol_mult: float


@dataclass(frozen=True, kw_only=True)
class WaveOverlayParams:
    window_by_timeframe: Dict[str, int]
    threshold_bounds_by_timeframe: Dict[str, Tuple[float, float]]
    volatility_multiplier: float
    wave4_overlap_tolerance: float


@dataclass(frozen=True, kw_only=True)
class WaveRegimeParams:
    swing_threshold_pct: float      # TW TXF = 0.35, US ES/NQ = 0.40
    fib_min_retracement: float      # e.g. 0.20
    fib_max_retracement: float      # e.g. 0.65
    default_bar_seconds: float      # e.g. 60.0 or 900.0
    max_pivots: int                 # e.g. 100
    holiday_active_mult: float      # e.g. 1.2
    gap_mult: float                 # e.g. 1.8


@dataclass(frozen=True, kw_only=True)
class VolumeProfileParams:
    rolling_window: int
    value_area_pct: float
    intraday_tick_steps: Tuple[Tuple[float, float], ...]
    daily_tick_steps: Tuple[Tuple[float, float], ...]
    session_reset: bool


@dataclass(frozen=True, kw_only=True)
class MarketExecutionParams:
    currency: str
    lot_size: int                           # TW = 1000 (board lot) or 1, US = 1
    allow_odd_lot: bool                     # TW = True/False, US = True
    price_decimals: int                     # TW = 2, US = 2
    daily_price_limit_pct: Optional[float]  # TW = 0.10, US = None
    limit_up_buy_guard_ratio: Optional[float]  # TW = 1.095, US = None
    min_stop_buffer_ratio: float
    max_stop_risk_ratio: float              # TW = 0.035, US = 0.050
    atr_stop_mult: float
    loss_cooldown_seconds: float
    signal_cooldown_seconds: float


@dataclass(frozen=True, kw_only=True)
class FrictionParams:
    buy_commission_rate: float      # TW = 0.001425, US = 0.0
    sell_commission_rate: float     # TW = 0.001425, US = 0.0
    sell_tax_rate: float            # TW = 0.003,    US = 0.0
    day_trade_tax_rate: float       # TW = 0.0015,   US = 0.0
    etf_tax_rate: float             # TW = 0.001,    US = 0.0
    min_commission: float           # TW = 20.0 TWD, US = 0.0 USD
    per_share_fee: float            # TW = 0.0,      US = 0.0 or 0.005
    sec_taf_sell_rate: float        # TW = 0.0,      US = 0.0000278
    slippage_bps: float
