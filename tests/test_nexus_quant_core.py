from __future__ import annotations

import ast
import datetime
import pathlib
import unittest

from nexus_quant_core import (
    BreakoutFsmParams,
    FirFilterParams,
    FrictionParams,
    MarketExecutionParams,
    MorphologyParams,
    TimeframeNormalizer,
    VALID_CHART_TIMEFRAMES,
    VolumeProfileParams,
    WaveOverlayParams,
    WaveRegimeParams,
    compute_position_size,
    compute_structural_stop_price,
    compute_trade_friction,
    is_buy_blocked_by_limit_up,
    is_within_daily_price_limit,
)
from nexus_quant_core.indicators import (
    SessionCandleAggregator,
    calculate_cascaded_fir,
    calculate_elliott_wave_overlays,
    calculate_fir_derivatives,
    calculate_rolling_poc_series,
    calculate_rolling_vwap_series,
    calculate_sma_series,
)
from nexus_quant_core.signals import (
    ElliottWaveRegimeFSM,
    evaluate_brk_entry,
    evaluate_btx_brk_exit,
    evaluate_masb_signals,
    evaluate_mfo_entry,
    extract_daily_morphology,
)


class _DummyUtcCalendar:
    @property
    def timezone(self) -> datetime.tzinfo:
        return datetime.timezone.utc

    def get_trading_day(self, utc_ts: float, symbol: str = "") -> datetime.date:
        return datetime.datetime.fromtimestamp(utc_ts, tz=datetime.timezone.utc).date()

    def get_session_type(self, utc_ts: float, symbol: str = "") -> str:
        return "REGULAR"

    def is_trading_day(self, d: datetime.date, symbol: str = "") -> bool:
        return d.weekday() < 5

    def roll_business_day(self, d: datetime.date, step_days: int, symbol: str = "") -> datetime.date:
        cur = d
        step = 1 if step_days >= 0 else -1
        rem = abs(step_days)
        while rem > 0:
            cur += datetime.timedelta(days=step)
            if cur.weekday() < 5:
                rem -= 1
        return cur

    def trading_days_between(self, older_ts_or_date, newer_ts_or_date, symbol: str = "") -> int:
        return 0

    def get_session_bar_label(self, utc_ts: float, window_seconds: int, symbol: str = "") -> int:
        return int(utc_ts // window_seconds) * window_seconds

    def get_bar_close_ts(self, bar_label_ts: int, window_seconds: int, symbol: str = "") -> int:
        return bar_label_ts + window_seconds

    def get_session_active_seconds(self, start_ts: float, end_ts: float, symbol: str = "") -> float:
        return max(0.0, end_ts - start_ts)


class TestNexusQuantCoreArchitectureAndMath(unittest.TestCase):
    def test_ast_gatekeeper_zero_market_hardcoding_and_zero_wall_clock(self) -> None:
        """nexus_quant_core source files must NEVER contain market names, market timezones, or host wall-clock calls."""
        pkg_dir = pathlib.Path(__file__).resolve().parent.parent / "nexus_quant_core"
        forbidden_strings = {
            "TW_STOCK",
            "TW_FUTURES",
            "US_STOCK",
            "US_FUTURES",
            "CRYPTO",
            "Asia/Taipei",
            "America/New_York",
            "TZ_TW",
            "TZ_US",
        }
        py_files = list(pkg_dir.rglob("*.py"))
        self.assertGreaterEqual(len(py_files), 10)

        for path in py_files:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for bad in forbidden_strings:
                        self.assertNotIn(
                            bad,
                            node.value,
                            f"Forbidden market literal {bad!r} found in {path.name}",
                        )
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(
                        node.func.attr,
                        {"now", "utcnow", "today", "localtime", "ctime"},
                        f"Forbidden wall-clock call .{node.func.attr}() in {path.name}",
                    )
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "time" and node.func.attr == "time":
                        self.fail(f"Forbidden time.time() call in {path.name}")

    def test_zero_default_parameter_contracts_require_explicit_injection(self) -> None:
        """Calling parameter dataclasses without arguments must raise TypeError (zero implicit market defaults)."""
        for cls in (
            FirFilterParams,
            BreakoutFsmParams,
            MorphologyParams,
            WaveOverlayParams,
            WaveRegimeParams,
            VolumeProfileParams,
            MarketExecutionParams,
            FrictionParams,
        ):
            with self.assertRaises(TypeError):
                cls()  # type: ignore[call-arg]

    def test_timeframe_normalizer_strict_7_whitelist_and_uppercase_1M_3M(self) -> None:
        self.assertEqual(len(VALID_CHART_TIMEFRAMES), 7)
        self.assertEqual(TimeframeNormalizer.normalize_chart_timeframe("1M"), "1mo")
        self.assertEqual(TimeframeNormalizer.normalize_chart_timeframe("3M"), "3mo")
        self.assertEqual(TimeframeNormalizer.to_seconds("1M", default=900), 2592000)
        self.assertEqual(TimeframeNormalizer.to_seconds("3M", default=900), 7776000)
        for bad in ("1m", "m", "5m", "30m", "2h"):
            with self.assertRaises(ValueError):
                TimeframeNormalizer.normalize_chart_timeframe(bad)

    def test_same_brk_algorithm_behaves_differently_under_tw_1080_vs_us_1560_params(self) -> None:
        tw_brk = BreakoutFsmParams(
            bars_per_day_15m=18,
            vwap_window=360,
            ma_short_window=90,
            ma_mid_window=360,
            ma_long_window=1080,
            ignition_lookback=95,
            fsm_timeout_bars=95,
            slope_lookback=54,
            btx_high_lookback=360,
            btx_drawdown_ratio=0.90,
            btx_mfo_vel_threshold=-0.30,
            btx_mfo_max_divergence_pct=0.003,
            bias20_min=-0.03,
            bias20_max=0.35,
            min_submersion_bars=0,
            min_submersion_pct=0.0,
        )
        us_brk = BreakoutFsmParams(
            bars_per_day_15m=26,
            vwap_window=520,
            ma_short_window=130,
            ma_mid_window=520,
            ma_long_window=1560,
            ignition_lookback=130,
            fsm_timeout_bars=130,
            slope_lookback=78,
            btx_high_lookback=520,
            btx_drawdown_ratio=0.88,
            btx_mfo_vel_threshold=-0.35,
            btx_mfo_max_divergence_pct=0.004,
            bias20_min=-0.04,
            bias20_max=0.45,
            min_submersion_bars=0,
            min_submersion_pct=0.0,
        )
        n = 1600
        vwap = [95.0] * n
        ma_long = [100.0] * n
        fir = [90.0] * n
        # Ignition at bar 1190, breakout above 100 at bar 1200 (after TW 1080 warmup, BEFORE US 1560 warmup!)
        fir[1190] = 96.0
        for k in range(1191, 1200):
            fir[k] = 98.0
        fir[1200] = 101.0

        self.assertTrue(evaluate_brk_entry(fir, vwap, ma_long, idx=1200, params=tw_brk))
        self.assertFalse(evaluate_brk_entry(fir, vwap, ma_long, idx=1200, params=us_brk))

        # Now trigger at bar 1580 (after US 1560 warmup!)
        fir[1570] = 96.0
        for k in range(1571, 1580):
            fir[k] = 98.0
        fir[1580] = 101.0
        self.assertTrue(evaluate_brk_entry(fir, vwap, ma_long, idx=1580, params=us_brk))

        # BTX drawdown: 11% pullback triggers TW (0.90 = 10%), does NOT trigger US (0.88 = 12%)!
        highs = [100.0] * n
        closes = [89.0] * n
        tw_exit, _ = evaluate_btx_brk_exit(highs, closes, [80.0] * n, idx=1580, params=tw_brk)
        us_exit, _ = evaluate_btx_brk_exit(highs, closes, [80.0] * n, idx=1580, params=us_brk)
        self.assertTrue(tw_exit)
        self.assertFalse(us_exit)

    def test_execution_and_friction_math_tw_vs_us(self) -> None:
        tw_exec = MarketExecutionParams(
            currency="TWD",
            lot_size=1000,
            allow_odd_lot=False,
            price_decimals=2,
            daily_price_limit_pct=0.10,
            limit_up_buy_guard_ratio=1.095,
            min_stop_buffer_ratio=0.005,
            max_stop_risk_ratio=0.035,
            atr_stop_mult=1.5,
            loss_cooldown_seconds=900.0,
            signal_cooldown_seconds=180.0,
        )
        us_exec = MarketExecutionParams(
            currency="USD",
            lot_size=1,
            allow_odd_lot=True,
            price_decimals=2,
            daily_price_limit_pct=None,
            limit_up_buy_guard_ratio=None,
            min_stop_buffer_ratio=0.005,
            max_stop_risk_ratio=0.050,
            atr_stop_mult=1.8,
            loss_cooldown_seconds=900.0,
            signal_cooldown_seconds=180.0,
        )
        # TW board lot rounds 2,600 shares down to 2,000; US keeps exact 2,600 shares
        self.assertEqual(compute_position_size(300_000, 100.0, 98.0, 5_200, exec_params=tw_exec), 2000)
        self.assertEqual(compute_position_size(300_000, 100.0, 98.0, 5_200, exec_params=us_exec), 2600)

        # +12% jump blocks TW buy & daily limit, allowed on US!
        self.assertTrue(is_buy_blocked_by_limit_up(112.0, 100.0, exec_params=tw_exec))
        self.assertFalse(is_within_daily_price_limit(112.0, 100.0, exec_params=tw_exec))
        self.assertFalse(is_buy_blocked_by_limit_up(112.0, 100.0, exec_params=us_exec))
        self.assertTrue(is_within_daily_price_limit(112.0, 100.0, exec_params=us_exec))

        tw_fric = FrictionParams(
            buy_commission_rate=0.001425,
            sell_commission_rate=0.001425,
            sell_tax_rate=0.003,
            day_trade_tax_rate=0.0015,
            etf_tax_rate=0.001,
            min_commission=20.0,
            per_share_fee=0.0,
            sec_taf_sell_rate=0.0,
            slippage_bps=2.0,
        )
        us_fric = FrictionParams(
            buy_commission_rate=0.0,
            sell_commission_rate=0.0,
            sell_tax_rate=0.0,
            day_trade_tax_rate=0.0,
            etf_tax_rate=0.0,
            min_commission=0.0,
            per_share_fee=0.0,
            sec_taf_sell_rate=0.0000278,
            slippage_bps=2.0,
        )
        tw_sell = compute_trade_friction("SELL", 1000.0, 1000, friction_params=tw_fric)
        us_sell = compute_trade_friction("SELL", 200.0, 100, friction_params=us_fric)
        self.assertGreater(tw_sell["tax"], 2900.0)
        self.assertLess(us_sell["tax"], 2.0)

    def test_fir_vwap_poc_wave_and_morphology_end_to_end(self) -> None:
        fir_p = FirFilterParams(stage1_weights=(1.0, 2.0, 3.0), stage2_weights=(1.0, 2.0, 3.0), deadzone_eps=1e-6)
        series = [100.0, 101.0, 102.0, 104.0, 107.0, 111.0, 116.0]
        fir_val = calculate_cascaded_fir(series, params=fir_p)
        vel, acc = calculate_fir_derivatives(series, params=fir_p)
        self.assertGreater(fir_val, 105.0)
        self.assertGreater(vel, 0.0)
        self.assertGreater(acc, 0.0)

        cal = _DummyUtcCalendar()
        agg = SessionCandleAggregator(cal, window_seconds=900, max_bars=50)
        self.assertIsNone(agg.update_candle("AAPL", 200.0, 1763100000.0, 100.0))
        sealed = agg.update_candle("AAPL", 202.0, 1763100905.0, 150.0)
        self.assertIsNotNone(sealed)
        self.assertEqual(sealed["c"], 200.0)

        wave_p = WaveRegimeParams(
            swing_threshold_pct=0.40,
            fib_min_retracement=0.20,
            fib_max_retracement=0.65,
            default_bar_seconds=60.0,
            max_pivots=100,
            holiday_active_mult=1.2,
            gap_mult=1.8,
        )
        fsm = ElliottWaveRegimeFSM(params=wave_p, calendar=cal, symbol="ES=F")
        fsm.process_candle(high=5900.0, low=5800.0, close=5895.0, timestamp=1763100000.0, open_price=5805.0)
        fsm.process_candle(high=5890.0, low=5850.0, close=5855.0, timestamp=1763100060.0, open_price=5888.0)
        reg = fsm.process_candle(high=5960.0, low=5855.0, close=5955.0, timestamp=1763100120.0, open_price=5860.0)
        self.assertEqual(reg.trend, "BULLISH")
        self.assertEqual(reg.wave_stage, "IMPULSE_WAVE_3")

        vp_p = VolumeProfileParams(
            rolling_window=10,
            value_area_pct=0.70,
            intraday_tick_steps=((100.0, 0.5), (0.0, 0.1)),
            daily_tick_steps=((100.0, 1.0), (0.0, 0.25)),
            session_reset=True,
        )
        bars = [
            {"timestamp": 1763100000 + i * 900, "trading_date": "2026-11-14", "open": 150.0, "high": 152.0, "low": 149.0, "close": 151.0, "volume": 1000.0}
            for i in range(12)
        ]
        poc = calculate_rolling_poc_series(bars, params=vp_p, calendar=cal, symbol="AAPL")
        self.assertEqual(len(poc), 12)

        wo_p = WaveOverlayParams(
            window_by_timeframe={"15m": 2, "1h": 2, "4h": 2, "1d": 2, "1wk": 2, "1mo": 2, "3mo": 2},
            threshold_bounds_by_timeframe={tf: (0.005, 0.05) for tf in VALID_CHART_TIMEFRAMES},
            volatility_multiplier=1.5,
            wave4_overlap_tolerance=0.015,
        )
        ov = calculate_elliott_wave_overlays(bars, timeframe="15m", params=wo_p)
        self.assertIn("lines", ov)
        self.assertIn("markers", ov)

        morph_p = MorphologyParams(
            atr_period=14,
            ma_fast=5,
            ma_mid=10,
            ma_month=20,
            ma_quarter=60,
            vol_ma_period=20,
            cap_down_days=4,
            cap_bias20_max=-0.05,
            cap_vol_ratio_min=1.65,
            dbl_lookback_min=15,
            dbl_lookback_max=75,
            dbl_diff_tol=0.025,
            dbl_neck_min=0.08,
            dbl_vol_ratio_min=1.60,
            dbl_shadow_min=0.50,
            dbl_bias20_min=-0.11,
            dbl_bias20_max=-0.03,
            dip_bias60_abs_max=0.08,
            dip_a_bias20_max=-0.05,
            dip_b_bias20_max=-0.08,
            dip_c_pullback_max=-0.15,
            dip_d_pullback_max=-0.12,
            sqz_spread4_max=0.020,
            sqz_bias60_abs_max=0.020,
            sqz_vol_ratio_min=1.20,
            masb_squeeze_threshold_pct=0.35,
            masb_vel_min=0.01,
            masb_acc_min=0.005,
            masb_vol_mult=1.1,
        )
        daily_bars = [
            {
                "trading_date": (datetime.date(2026, 6, 1) + datetime.timedelta(days=i)).isoformat(),
                "timestamp": 1780000000 + i * 86400,
                "open": 100.0 + i * 0.2,
                "high": 101.0 + i * 0.2,
                "low": 99.0 + i * 0.2,
                "close": 100.5 + i * 0.2,
                "volume": 5000.0,
            }
            for i in range(65)
        ]
        ma60_map, morph_map = extract_daily_morphology(daily_bars, params=morph_p)
        self.assertGreater(len(ma60_map), 0)
        self.assertGreater(len(morph_map), 40)
        masb = evaluate_masb_signals(daily_bars, fir_params=fir_p, morph_params=morph_p, vwap_window=20)
        self.assertEqual(len(masb["fir"]), 65)
        self.assertEqual(len(calculate_sma_series([b["close"] for b in daily_bars], period=20)), 65)
        self.assertEqual(len(calculate_rolling_vwap_series(daily_bars, lookback=20)), 65)
        self.assertFalse(evaluate_mfo_entry([100.0] * 10, [99.0] * 10, [98.0] * 10, [97.0] * 10, idx=5, params=BreakoutFsmParams(
            bars_per_day_15m=18, vwap_window=360, ma_short_window=90, ma_mid_window=360, ma_long_window=1080,
            ignition_lookback=95, fsm_timeout_bars=95, slope_lookback=54, btx_high_lookback=360,
            btx_drawdown_ratio=0.90, btx_mfo_vel_threshold=-0.30, btx_mfo_max_divergence_pct=0.003,
            bias20_min=-0.03, bias20_max=0.35, min_submersion_bars=0, min_submersion_pct=0.0,
        )))
        stop = compute_structural_stop_price(200.0, "BUY", 2.0, 198.0, 202.0, exec_params=MarketExecutionParams(
            currency="USD", lot_size=1, allow_odd_lot=True, price_decimals=2, daily_price_limit_pct=None,
            limit_up_buy_guard_ratio=None, min_stop_buffer_ratio=0.005, max_stop_risk_ratio=0.05,
            atr_stop_mult=1.5, loss_cooldown_seconds=900.0, signal_cooldown_seconds=180.0,
        ))
        self.assertLess(stop, 200.0)


if __name__ == "__main__":
    unittest.main()
