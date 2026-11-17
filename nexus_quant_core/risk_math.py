from __future__ import annotations

import math
from typing import Dict, Optional

from nexus_quant_core.contracts import FrictionParams, MarketExecutionParams
from nexus_quant_core.math_utils import safe_float


def compute_structural_stop_price(
    entry_price: float,
    side: str,
    atr: float,
    bar_low: Optional[float],
    bar_high: Optional[float],
    *,
    exec_params: MarketExecutionParams,
) -> float:
    """Compute structural stop-loss price bounded by [min_stop_buffer_ratio, max_stop_risk_ratio]."""
    px = safe_float(entry_price, 0.0)
    if px <= 0.0:
        return 0.0
    safe_atr = max(0.0, safe_float(atr, 0.0))
    min_buf = max(0.001, exec_params.min_stop_buffer_ratio)
    max_risk = max(min_buf, exec_params.max_stop_risk_ratio)
    atr_dist = safe_atr * max(0.1, exec_params.atr_stop_mult) if safe_atr > 0 else px * (min_buf * 2.0)

    is_short = str(side or "BUY").strip().upper() in ("SELL", "SHORT")
    if not is_short:
        cand = px - atr_dist
        if bar_low is not None:
            bl = safe_float(bar_low, 0.0)
            if 0.0 < bl < px:
                cand = min(cand, bl * (1.0 - min_buf * 0.5))
        min_stop = px * (1.0 - max_risk)
        max_stop = px * (1.0 - min_buf)
        stop = max(min_stop, min(max_stop, cand))
    else:
        cand = px + atr_dist
        if bar_high is not None:
            bh = safe_float(bar_high, 0.0)
            if bh > px:
                cand = max(cand, bh * (1.0 + min_buf * 0.5))
        min_stop = px * (1.0 + min_buf)
        max_stop = px * (1.0 + max_risk)
        stop = max(min_stop, min(max_stop, cand))

    return round(stop, exec_params.price_decimals)


def compute_position_size(
    capital_budget: float,
    entry_price: float,
    stop_price: float,
    max_risk_amount: float,
    *,
    exec_params: MarketExecutionParams,
) -> int:
    """Compute integer share/unit quantity respecting risk budget, capital cap, and market lot_size."""
    px = safe_float(entry_price, 0.0)
    cap = safe_float(capital_budget, 0.0)
    risk_cap = safe_float(max_risk_amount, 0.0)
    if px <= 0.0 or cap <= 0.0 or risk_cap <= 0.0:
        return 0

    per_unit_risk = abs(px - safe_float(stop_price, 0.0))
    min_unit_risk = px * max(0.001, exec_params.min_stop_buffer_ratio)
    eff_unit_risk = max(per_unit_risk, min_unit_risk)

    qty_by_risk = math.floor(risk_cap / eff_unit_risk)
    qty_by_cap = math.floor(cap / px)
    raw_qty = max(0, min(qty_by_risk, qty_by_cap))
    if raw_qty <= 0:
        return 0

    lot = max(1, int(exec_params.lot_size))
    if lot > 1 and not exec_params.allow_odd_lot:
        return (raw_qty // lot) * lot
    return raw_qty


def compute_trade_friction(
    side: str,
    price: float,
    quantity: float,
    *,
    friction_params: FrictionParams,
    is_etf: bool = False,
    is_day_trade: bool = False,
) -> Dict[str, float]:
    """Compute commission, transaction/SEC tax, slippage cost, and total friction in native currency."""
    px = max(0.0, safe_float(price, 0.0))
    qty = max(0.0, safe_float(quantity, 0.0))
    notional = px * qty
    if notional <= 0.0:
        return {"notional": 0.0, "commission": 0.0, "tax": 0.0, "slippage": 0.0, "total": 0.0}

    is_sell = str(side or "").strip().upper() in ("SELL", "SHORT")
    comm_rate = friction_params.sell_commission_rate if is_sell else friction_params.buy_commission_rate
    pct_comm = notional * max(0.0, comm_rate)
    share_comm = qty * max(0.0, friction_params.per_share_fee)
    raw_comm = pct_comm + share_comm
    commission = max(friction_params.min_commission, raw_comm) if raw_comm > 0.0 else (
        friction_params.min_commission if comm_rate > 0.0 else 0.0
    )

    tax = 0.0
    if is_sell:
        if is_etf and friction_params.etf_tax_rate > 0.0:
            tax_rate = friction_params.etf_tax_rate
        elif is_day_trade and friction_params.day_trade_tax_rate > 0.0:
            tax_rate = friction_params.day_trade_tax_rate
        else:
            tax_rate = friction_params.sell_tax_rate
        tax = notional * max(0.0, tax_rate) + notional * max(0.0, friction_params.sec_taf_sell_rate)

    slippage = notional * (max(0.0, friction_params.slippage_bps) / 10000.0)
    total = commission + tax + slippage
    return {
        "notional": round(notional, 4),
        "commission": round(commission, 4),
        "tax": round(tax, 4),
        "slippage": round(slippage, 4),
        "total": round(total, 4),
    }


def is_buy_blocked_by_limit_up(
    price: float,
    ref_prev_close: Optional[float],
    *,
    exec_params: MarketExecutionParams,
) -> bool:
    """Return True if target market enforces a limit-up buy guard and price exceeds guard ratio."""
    if exec_params.limit_up_buy_guard_ratio is None:
        return False
    prev_c = safe_float(ref_prev_close, 0.0)
    px = safe_float(price, 0.0)
    if prev_c <= 0.0 or px <= 0.0:
        return False
    return px >= prev_c * exec_params.limit_up_buy_guard_ratio


def is_within_daily_price_limit(
    price: float,
    ref_prev_close: Optional[float],
    *,
    exec_params: MarketExecutionParams,
) -> bool:
    """Return True if target market has no daily limit or price is within ±daily_price_limit_pct."""
    if exec_params.daily_price_limit_pct is None:
        return True
    prev_c = safe_float(ref_prev_close, 0.0)
    px = safe_float(price, 0.0)
    if prev_c <= 0.0 or px <= 0.0:
        return True
    lim = abs(exec_params.daily_price_limit_pct)
    return (prev_c * (1.0 - lim - 1e-6)) <= px <= (prev_c * (1.0 + lim + 1e-6))
