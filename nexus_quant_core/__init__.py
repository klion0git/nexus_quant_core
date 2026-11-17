"""
nexus_quant_core: Pure stateless quantitative algorithms & explicit parameter contracts.

Design Invariants:
1. ZERO market-specific branching (no exchange-specific conditionals or timezone constants).
2. ZERO implicit market parameter defaults on strategy/execution dataclasses (`kw_only=True`).
3. ZERO host wall-clock access (`datetime.now`, `time.time`, `date.today` are strictly forbidden).
"""

from nexus_quant_core.contracts import (
    VALID_CHART_TIMEFRAMES,
    CALENDAR_TIMEFRAMES,
    INTRADAY_TIMEFRAMES,
    TimeframeNormalizer,
    ClockProtocol,
    SessionCalendarProtocol,
    BrokerFeedProtocol,
    CandleBar,
    MarketTick,
    MacroRegimeSnapshot,
    SignalEvent,
    OrderInstruction,
    FirFilterParams,
    BreakoutFsmParams,
    MorphologyParams,
    WaveOverlayParams,
    WaveRegimeParams,
    VolumeProfileParams,
    MarketExecutionParams,
    FrictionParams,
)
from nexus_quant_core.math_utils import (
    safe_float,
    safe_int,
    safe_pct_change,
    round_adaptive_price,
    clean_finite_series,
)
from nexus_quant_core.risk_math import (
    compute_structural_stop_price,
    compute_position_size,
    compute_trade_friction,
    is_buy_blocked_by_limit_up,
    is_within_daily_price_limit,
)

__all__ = [
    "VALID_CHART_TIMEFRAMES",
    "CALENDAR_TIMEFRAMES",
    "INTRADAY_TIMEFRAMES",
    "TimeframeNormalizer",
    "ClockProtocol",
    "SessionCalendarProtocol",
    "BrokerFeedProtocol",
    "CandleBar",
    "MarketTick",
    "MacroRegimeSnapshot",
    "SignalEvent",
    "OrderInstruction",
    "FirFilterParams",
    "BreakoutFsmParams",
    "MorphologyParams",
    "WaveOverlayParams",
    "WaveRegimeParams",
    "VolumeProfileParams",
    "MarketExecutionParams",
    "FrictionParams",
    "safe_float",
    "safe_int",
    "safe_pct_change",
    "round_adaptive_price",
    "clean_finite_series",
    "compute_structural_stop_price",
    "compute_position_size",
    "compute_trade_friction",
    "is_buy_blocked_by_limit_up",
    "is_within_daily_price_limit",
]
