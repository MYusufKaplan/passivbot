"""
Unit tests and property-based tests for positions_tracker.app (FastAPI endpoints).
"""

import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from positions_tracker.app import app

_MOCK_PREFIX = "positions_tracker.app"
_BR_PREFIX = "positions_tracker.build_response"


@pytest.fixture
def client():
    """Create a TestClient that skips the startup event (no real exchange init)."""
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestGetApiData:
    """Unit test: GET /api/data returns 200 with valid JSON structure."""

    @patch(
        f"{_MOCK_PREFIX}.build_dashboard_data",
        return_value={
            "positions": [],
            "account": {
                "equity_usdt": 5000.0,
                "equity_try": 160000.0,
                "profit_try": 10000.0,
                "rate_stats": {},
                "projections": None,
            },
            "history": [],
            "idle": True,
            "timestamp": 1700001000.0,
            "usdt_try_rate": 32.0,
        },
    )
    def test_returns_200_with_valid_json(self, mock_build, client):
        response = client.get("/api/data")
        assert response.status_code == 200
        data = response.json()
        assert "positions" in data
        assert "account" in data
        assert "history" in data
        assert "idle" in data
        assert "timestamp" in data
        assert "usdt_try_rate" in data


class TestRootRoute:
    """Unit test: GET / serves index.html (returns 200 with HTML content)."""

    def test_serves_index_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Positions Tracker" in response.text


# ---------------------------------------------------------------------------
# Property 16: API error resilience
# Feature: positions-tracker-webapp, Property 16: API error resilience
# **Validates: Requirements 9.4**
# ---------------------------------------------------------------------------

# Strategy: generate diverse exception types and messages
_exception_st = st.one_of(
    st.builds(ValueError, st.text(min_size=1, max_size=100)),
    st.builds(RuntimeError, st.text(min_size=1, max_size=100)),
    st.builds(ConnectionError, st.text(min_size=1, max_size=100)),
    st.builds(TimeoutError, st.text(min_size=1, max_size=100)),
    st.builds(KeyError, st.text(min_size=1, max_size=50)),
    st.builds(TypeError, st.text(min_size=1, max_size=100)),
    st.builds(IOError, st.text(min_size=1, max_size=100)),
)


class TestProperty16ApiErrorResilience:
    """Mock build_dashboard_data to raise various exceptions and verify
    the endpoint returns a JSON error response with an 'error' field
    instead of crashing with HTTP 500."""

    @settings(max_examples=100)
    @given(exc=_exception_st)
    def test_error_returns_json_with_error_field(self, exc):
        with patch(f"{_MOCK_PREFIX}.build_dashboard_data", side_effect=exc):
            with TestClient(app, raise_server_exceptions=False) as c:
                response = c.get("/api/data")
                assert response.status_code == 200, (
                    f"Expected 200 but got {response.status_code} for {type(exc).__name__}"
                )
                data = response.json()
                assert "error" in data, (
                    f"Response missing 'error' field for {type(exc).__name__}: {data}"
                )


# ---------------------------------------------------------------------------
# Integration test: full request cycle through /api/data
# Validates: Requirements 1.1, 9.4
# ---------------------------------------------------------------------------


def _mock_position():
    """Realistic active position dict as returned by CCXT fetch_positions."""
    return {
        "symbol": "BTC_USDT:USDT",
        "contracts": 10,
        "contractSize": 0.001,
        "entryPrice": 67000.0,
        "realizedPnl": 0.5,
        "unrealizedPnl": 5.005,
        "leverage": "20x",
        "liquidationPrice": 60000.0,
        "marginRatio": 0.05,
        "timestamp": 1700000000,
    }


def _mock_orders():
    """Two open orders: one buy, one sell."""
    return [
        {"id": "buy1", "side": "buy", "price": 66500.0, "amount": 0.01, "remaining": 0.01},
        {"id": "sell1", "side": "sell", "price": 68000.0, "amount": 0.005, "remaining": 0.005},
    ]


def _mock_ticker():
    return {"last": 67500.5}


def _mock_ohlcv():
    base_ts = 1700000000
    return [
        [base_ts + i * 900, 67000 + i, 67500 + i, 66800 + i, 67200 + i, 100 + i]
        for i in range(5)
    ]


def _mock_recent_trades():
    return [
        {"timestamp": 1700000100, "side": "buy", "price": 67000.0, "amount": 0.01},
        {"timestamp": 1700000200, "side": "sell", "price": 67200.0, "amount": 0.005},
    ]


def _mock_price_changes():
    return {"5m": "+0.12%", "15m": "-0.05%", "1h": "+1.20%", "1d": "+3.50%"}


def _mock_funding_rate():
    return {"fundingRate": 0.0001, "fundingTimestamp": 1700003600000}


def _mock_recent_closed():
    return [
        {
            "contract": "ETH_USDT",
            "pnl": 2.5,
            "time": int(time.time()) - 600,
            "first_open_time": int(time.time()) - 1200,
        },
        {
            "contract": "SOL_USDT",
            "pnl": -1.2,
            "time": int(time.time()) - 3600,
            "first_open_time": int(time.time()) - 7200,
        },
    ]


def _mock_closed_30d():
    now = int(time.time())
    return [
        {"contract": "ETH_USDT", "pnl": "2.5", "time": now - 600, "first_open_time": now - 1200},
        {"contract": "SOL_USDT", "pnl": "-1.2", "time": now - 3600, "first_open_time": now - 7200},
        {"contract": "BTC_USDT", "pnl": "10.0", "time": now - 86000, "first_open_time": now - 86500},
    ]


class TestIntegrationFullRequestCycle:
    """Integration test: mock all exchange APIs, call GET /api/data via
    TestClient, and verify the complete JSON response structure including
    positions, account, history, and idle flag.

    Validates: Requirements 1.1, 9.4
    """

    @patch(f"{_BR_PREFIX}.fetch_closed_positions_30d", return_value=_mock_closed_30d())
    @patch(f"{_BR_PREFIX}.fetch_recent_closed_positions", return_value=_mock_recent_closed())
    @patch(f"{_BR_PREFIX}.fetch_unified_account_equity", return_value=5000.0)
    @patch(f"{_BR_PREFIX}.fetch_usdt_try_price", return_value=32.0)
    @patch(f"{_BR_PREFIX}.get_last_funding_payment", return_value=-0.32)
    @patch(f"{_BR_PREFIX}.fetch_funding_rate", return_value=_mock_funding_rate())
    @patch(f"{_BR_PREFIX}.get_price_changes", return_value=_mock_price_changes())
    @patch(f"{_BR_PREFIX}.fetch_recent_trades", return_value=_mock_recent_trades())
    @patch(f"{_BR_PREFIX}.fetch_orders", return_value=_mock_orders())
    @patch(f"{_BR_PREFIX}.fetch_ticker", return_value=_mock_ticker())
    @patch(f"{_BR_PREFIX}.fetch_active_positions", return_value=[_mock_position()])
    def test_full_request_cycle(self, *mocks):
        """End-to-end: mock exchange APIs → GET /api/data → verify complete
        JSON response structure."""
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/api/data")

        assert response.status_code == 200
        data = response.json()

        # ---- Top-level keys ----
        assert "error" not in data
        for key in ("positions", "account", "history", "idle", "timestamp", "usdt_try_rate"):
            assert key in data, f"Missing top-level key: {key}"

        assert isinstance(data["positions"], list)
        assert isinstance(data["account"], dict)
        assert isinstance(data["history"], list)
        assert isinstance(data["idle"], bool)
        assert isinstance(data["timestamp"], (int, float))
        assert isinstance(data["usdt_try_rate"], (int, float))

        # ---- idle flag is False when positions exist ----
        assert data["idle"] is False
        assert data["usdt_try_rate"] == 32.0

        # ---- Position structure ----
        assert len(data["positions"]) == 1
        pos = data["positions"][0]

        pos_required = [
            "symbol", "contracts", "current_price", "entry_price",
            "unrealized_pnl", "realized_pnl", "price_changes",
            "ohlcv", "orders", "funding",
        ]
        for key in pos_required:
            assert key in pos, f"Missing position key: {key}"

        assert pos["symbol"] == "BTC_USDT:USDT"
        assert pos["contracts"] == 10
        assert pos["current_price"] == 67500.5
        assert pos["entry_price"] == 67000.0

        # ---- Price changes sub-structure ----
        pc = pos["price_changes"]
        for tf in ("5m", "15m", "1h", "1d"):
            assert tf in pc, f"Missing price_changes timeframe: {tf}"

        # ---- OHLCV ----
        assert isinstance(pos["ohlcv"], list)

        # ---- Orders structure ----
        assert isinstance(pos["orders"], list)
        assert len(pos["orders"]) == 2
        for order in pos["orders"]:
            for key in ("side", "price", "amount", "remaining", "progress_percent", "color"):
                assert key in order, f"Missing order key: {key}"
        # Orders sorted by price descending
        assert pos["orders"][0]["price"] >= pos["orders"][1]["price"]

        # ---- Funding structure ----
        funding = pos["funding"]
        for key in ("rate", "expected_payment", "next_funding_timestamp", "last_payment"):
            assert key in funding, f"Missing funding key: {key}"
        assert funding["rate"] == 0.0001
        assert funding["last_payment"] == -0.32
        assert funding["next_funding_timestamp"] == 1700003600

        # ---- Account structure ----
        account = data["account"]
        for key in ("equity_usdt", "equity_try", "profit_try", "rate_stats", "projections"):
            assert key in account, f"Missing account key: {key}"
        assert account["equity_usdt"] == 5000.0
        assert account["equity_try"] == 5000.0 * 32.0
        assert account["profit_try"] == (5000.0 * 32.0) - 150000

        # rate_stats should have 24h/7d/30d windows
        rs = account["rate_stats"]
        assert rs is not None
        for window in ("24h", "7d", "30d"):
            assert window in rs, f"Missing rate_stats window: {window}"

        # projections should exist (equity > 0 and rate_stats present)
        assert account["projections"] is not None

        # ---- History structure ----
        assert len(data["history"]) > 0
        # When not idle, history limited to 48
        assert len(data["history"]) <= 48
        for item in data["history"]:
            for key in ("symbol", "pnl", "pnl_per_sec", "duration_seconds",
                        "close_timestamp", "open_timestamp"):
                assert key in item, f"Missing history key: {key}"
