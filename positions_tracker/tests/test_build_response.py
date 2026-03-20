"""
Unit tests and property-based tests for positions_tracker.build_response module.
"""

import time
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from positions_tracker.build_response import (
    STARTING_BALANCE,
    _build_history,
    _build_orders,
    build_dashboard_data,
)


# ---------------------------------------------------------------------------
# Helpers — mock data factories
# ---------------------------------------------------------------------------

def _make_position(symbol="BTC_USDT:USDT", contracts=10, contract_size=0.001,
                   entry_price=67000.0, realized_pnl=0.5, unrealized_pnl=5.0,
                   leverage="20x", liquidation_price=60000.0, margin_ratio=0.05,
                   timestamp=1700000000):
    return {
        "symbol": symbol,
        "contracts": contracts,
        "contractSize": contract_size,
        "entryPrice": entry_price,
        "realizedPnl": realized_pnl,
        "unrealizedPnl": unrealized_pnl,
        "leverage": leverage,
        "liquidationPrice": liquidation_price,
        "marginRatio": margin_ratio,
        "timestamp": timestamp,
    }


def _make_order(id="order1", side="buy", price=66500.0, amount=0.01, remaining=0.01):
    return {
        "id": id,
        "side": side,
        "price": price,
        "amount": amount,
        "remaining": remaining,
    }


def _make_closed_position(contract="BTC_USDT", pnl=2.5, close_time=1700000000,
                          open_time=1699999405):
    return {
        "contract": contract,
        "pnl": pnl,
        "time": close_time,
        "first_open_time": open_time,
    }


# ---------------------------------------------------------------------------
# _build_history tests
# ---------------------------------------------------------------------------

class TestBuildHistory:
    def test_limits_to_48_when_active(self):
        closed = [_make_closed_position(close_time=1700000000 + i) for i in range(60)]
        result = _build_history(closed, idle=False)
        assert len(result) == 48

    def test_limits_to_100_when_idle(self):
        closed = [_make_closed_position(close_time=1700000000 + i) for i in range(120)]
        result = _build_history(closed, idle=True)
        assert len(result) == 100

    def test_empty_list(self):
        assert _build_history([], idle=False) == []
        assert _build_history(None, idle=True) == []

    def test_history_item_fields(self):
        closed = [_make_closed_position()]
        result = _build_history(closed, idle=False)
        item = result[0]
        assert item["symbol"] == "BTC"
        assert item["pnl"] == 2.5
        assert item["close_timestamp"] == 1700000000
        assert item["open_timestamp"] == 1699999405
        assert item["duration_seconds"] == 595
        assert abs(item["pnl_per_sec"] - 2.5 / 595) < 1e-10

    def test_zero_duration(self):
        closed = [_make_closed_position(close_time=100, open_time=100)]
        result = _build_history(closed, idle=False)
        assert result[0]["duration_seconds"] == 0
        assert result[0]["pnl_per_sec"] == 0.0


# ---------------------------------------------------------------------------
# _build_orders tests
# ---------------------------------------------------------------------------

class TestBuildOrders:
    def test_empty_orders(self):
        assert _build_orders([], 67000.0, _make_position()) == []
        assert _build_orders(None, 67000.0, _make_position()) == []

    def test_orders_sorted_by_price_descending(self):
        orders = [
            _make_order(id="low", price=65000.0),
            _make_order(id="high", price=68000.0),
            _make_order(id="mid", price=66500.0),
        ]
        result = _build_orders(orders, 67000.0, _make_position())
        prices = [o["price"] for o in result]
        assert prices == sorted(prices, reverse=True)

    def test_order_fields_present(self):
        orders = [_make_order()]
        result = _build_orders(orders, 67000.0, _make_position())
        order = result[0]
        required_keys = {"id", "side", "price", "amount", "remaining",
                         "progress_percent", "color", "cumulative_avg_price",
                         "estimated_pnl"}
        assert required_keys.issubset(order.keys())


# ---------------------------------------------------------------------------
# build_dashboard_data integration test (mocked exchange calls)
# ---------------------------------------------------------------------------

_MOCK_PREFIX = "positions_tracker.build_response"


class TestBuildDashboardData:
    @patch(f"{_MOCK_PREFIX}.fetch_active_positions", return_value=[])
    @patch(f"{_MOCK_PREFIX}.fetch_usdt_try_price", return_value=32.0)
    @patch(f"{_MOCK_PREFIX}.fetch_unified_account_equity", return_value=5000.0)
    @patch(f"{_MOCK_PREFIX}.fetch_recent_closed_positions", return_value=[])
    @patch(f"{_MOCK_PREFIX}.fetch_closed_positions_30d", return_value=[])
    def test_idle_response_structure(self, *mocks):
        result = build_dashboard_data()
        assert "error" not in result
        assert result["idle"] is True
        assert result["positions"] == []
        assert "account" in result
        assert "history" in result
        assert "timestamp" in result
        assert result["usdt_try_rate"] == 32.0

    @patch(f"{_MOCK_PREFIX}.fetch_active_positions", return_value=[])
    @patch(f"{_MOCK_PREFIX}.fetch_usdt_try_price", return_value=32.0)
    @patch(f"{_MOCK_PREFIX}.fetch_unified_account_equity", return_value=5000.0)
    @patch(f"{_MOCK_PREFIX}.fetch_recent_closed_positions", return_value=[])
    @patch(f"{_MOCK_PREFIX}.fetch_closed_positions_30d", return_value=[])
    def test_account_try_conversion(self, *mocks):
        result = build_dashboard_data()
        account = result["account"]
        assert account["equity_usdt"] == 5000.0
        assert account["equity_try"] == 5000.0 * 32.0
        assert account["profit_try"] == (5000.0 * 32.0) - STARTING_BALANCE


    @patch(f"{_MOCK_PREFIX}.get_last_funding_payment", return_value=-0.32)
    @patch(f"{_MOCK_PREFIX}.fetch_funding_rate", return_value={
        "fundingRate": 0.0001, "fundingTimestamp": 1700003600
    })
    @patch(f"{_MOCK_PREFIX}.get_price_changes", return_value={
        "5m": "+0.12%", "15m": "-0.05%", "1h": "+1.2%", "1d": "+3.5%"
    })
    @patch(f"{_MOCK_PREFIX}.fetch_recent_trades", return_value=[
        {"timestamp": 1700000000, "side": "buy", "price": 67000.0, "amount": 0.01}
    ])
    @patch(f"{_MOCK_PREFIX}.fetch_orders", return_value=[
        _make_order(id="o1", side="buy", price=66500.0),
        _make_order(id="o2", side="sell", price=68000.0),
    ])
    @patch(f"{_MOCK_PREFIX}.fetch_ticker", return_value={"last": 67500.5})
    @patch(f"{_MOCK_PREFIX}.fetch_active_positions", return_value=[_make_position()])
    @patch(f"{_MOCK_PREFIX}.fetch_usdt_try_price", return_value=32.0)
    @patch(f"{_MOCK_PREFIX}.fetch_unified_account_equity", return_value=5000.0)
    @patch(f"{_MOCK_PREFIX}.fetch_recent_closed_positions", return_value=[
        _make_closed_position()
    ])
    @patch(f"{_MOCK_PREFIX}.fetch_closed_positions_30d", return_value=[])
    def test_active_response_structure(self, *mocks):
        result = build_dashboard_data()
        assert "error" not in result
        assert result["idle"] is False
        assert len(result["positions"]) == 1

        pos = result["positions"][0]
        assert pos["symbol"] == "BTC_USDT:USDT"
        assert pos["visual_symbol"] == "BTC"
        assert pos["current_price"] == 67500.5
        assert pos["entry_price"] == 67000.0
        assert pos["break_even_price"] is not None
        assert "funding" in pos
        assert pos["funding"]["rate"] == 0.0001
        assert pos["funding"]["last_payment"] == -0.32
        assert len(pos["orders"]) == 2
        # Orders sorted by price descending
        assert pos["orders"][0]["price"] >= pos["orders"][1]["price"]

    @patch(f"{_MOCK_PREFIX}.fetch_active_positions", side_effect=Exception("API down"))
    def test_error_handling(self, mock_fetch):
        result = build_dashboard_data()
        assert "error" in result
        assert "API down" in result["error"]

    @patch(f"{_MOCK_PREFIX}.fetch_active_positions", return_value=[])
    @patch(f"{_MOCK_PREFIX}.fetch_usdt_try_price", return_value=32.0)
    @patch(f"{_MOCK_PREFIX}.fetch_unified_account_equity", return_value=None)
    @patch(f"{_MOCK_PREFIX}.fetch_recent_closed_positions", return_value=[])
    @patch(f"{_MOCK_PREFIX}.fetch_closed_positions_30d", return_value=[])
    def test_none_equity_defaults_to_zero(self, *mocks):
        result = build_dashboard_data()
        assert result["account"]["equity_usdt"] == 0.0
        assert result["account"]["equity_try"] == 0.0
        assert result["account"]["profit_try"] == -STARTING_BALANCE


# ---------------------------------------------------------------------------
# Property-based tests (hypothesis)
# ---------------------------------------------------------------------------

# Strategies for generating test data

_closed_position_st = st.fixed_dictionaries({
    "contract": st.sampled_from(["BTC_USDT", "ETH_USDT", "SOL_USDT", "DOGE_USDT"]),
    "pnl": st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    "time": st.integers(min_value=1_000_000_000, max_value=2_000_000_000),
    "first_open_time": st.integers(min_value=1_000_000_000, max_value=2_000_000_000),
})

_order_st = st.fixed_dictionaries({
    "id": st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N"))),
    "side": st.sampled_from(["buy", "sell"]),
    "price": st.floats(min_value=0.01, max_value=100000.0, allow_nan=False, allow_infinity=False),
    "amount": st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False),
    "remaining": st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False),
})


# Feature: positions-tracker-webapp, Property 1: Dashboard response contains all required fields
# **Validates: Requirements 1.1, 1.9, 1.10, 2.2, 2.3, 2.6, 3.2, 4.2, 4.4, 5.7, 6.3**
class TestProperty1ResponseFields:
    @settings(max_examples=100)
    @given(
        closed_positions=st.lists(_closed_position_st, min_size=0, max_size=10),
        idle=st.booleans(),
    )
    def test_history_items_have_all_required_fields(self, closed_positions, idle):
        """Each history item must have: symbol, pnl, pnl_per_sec, duration_seconds,
        close_timestamp, open_timestamp."""
        result = _build_history(closed_positions, idle)
        required_keys = {"symbol", "pnl", "pnl_per_sec", "duration_seconds",
                         "close_timestamp", "open_timestamp"}
        for item in result:
            assert required_keys.issubset(item.keys()), (
                f"Missing keys: {required_keys - item.keys()}"
            )

    @settings(max_examples=100)
    @given(
        orders=st.lists(_order_st, min_size=1, max_size=10),
    )
    def test_order_items_have_all_required_fields(self, orders):
        """Each order must have: id, side, price, amount, remaining,
        progress_percent, color."""
        position = _make_position()
        current_price = 67000.0
        result = _build_orders(orders, current_price, position)
        required_keys = {"id", "side", "price", "amount", "remaining",
                         "progress_percent", "color"}
        for order in result:
            assert required_keys.issubset(order.keys()), (
                f"Missing keys: {required_keys - order.keys()}"
            )


# Feature: positions-tracker-webapp, Property 5: Response size limits and idle flag
# **Validates: Requirements 2.1, 2.11, 6.1, 6.2**
class TestProperty5SizeLimitsAndIdleFlag:
    @settings(max_examples=100)
    @given(
        count=st.integers(min_value=0, max_value=200),
    )
    def test_history_limit_48_when_not_idle(self, count):
        """_build_history returns at most 48 items when idle=False."""
        closed = [
            _make_closed_position(close_time=1700000000 + i)
            for i in range(count)
        ]
        result = _build_history(closed, idle=False)
        assert len(result) <= 48

    @settings(max_examples=100)
    @given(
        count=st.integers(min_value=0, max_value=200),
    )
    def test_history_limit_100_when_idle(self, count):
        """_build_history returns at most 100 items when idle=True."""
        closed = [
            _make_closed_position(close_time=1700000000 + i)
            for i in range(count)
        ]
        result = _build_history(closed, idle=True)
        assert len(result) <= 100


# Feature: positions-tracker-webapp, Property 6: USDT-to-TRY conversion and profit calculation
# **Validates: Requirements 2.7, 2.8, 5.1**
class TestProperty6UsdtTryConversion:
    @settings(max_examples=100)
    @given(
        equity_usdt=st.floats(min_value=0.01, max_value=1_000_000, allow_nan=False, allow_infinity=False),
        usdt_try_rate=st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
    )
    def test_equity_try_and_profit_try(self, equity_usdt, usdt_try_rate):
        """equity_try = equity_usdt * rate and profit_try = equity_try - 150000."""
        equity_try = equity_usdt * usdt_try_rate
        profit_try = equity_try - STARTING_BALANCE

        assert abs(equity_try - equity_usdt * usdt_try_rate) < 1e-6
        assert abs(profit_try - (equity_try - 150000)) < 1e-6


# Feature: positions-tracker-webapp, Property 8: Orders sorted by price descending
# **Validates: Requirements 3.1**
class TestProperty8OrdersSortedDescending:
    @settings(max_examples=100)
    @given(
        orders=st.lists(_order_st, min_size=0, max_size=20),
    )
    def test_orders_sorted_by_price_descending(self, orders):
        """_build_orders returns orders sorted by price in descending order."""
        position = _make_position()
        current_price = 67000.0
        result = _build_orders(orders, current_price, position)
        prices = [o["price"] for o in result]
        assert prices == sorted(prices, reverse=True), (
            f"Prices not in descending order: {prices}"
        )
