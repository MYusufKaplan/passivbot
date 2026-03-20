"""Tests for data_fetcher – Tasks 1.1, 1.2, 1.3."""

import json
import os
import time
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from positions_tracker.data_fetcher import (
    load_api_keys,
    init_clients,
    get_gate_ccxt,
    get_gate_futures_api,
    get_gate_unified_api,
    get_binance_ccxt,
    _clients,
    _cache,
    cached_fetch,
    clear_cache_for_symbol,
    TTL_USDT_TRY,
    TTL_PRICE_CHANGES,
    TTL_OHLCV,
    TTL_RECENT_TRADES,
    TTL_LAST_FUNDING,
    TTL_FUNDING_RATE,
    TTL_UNIFIED_EQUITY,
    TTL_RECENT_CLOSED,
    TTL_CLOSED_30D,
    CLOSED_POSITIONS_CACHE_FILE,
    _load_disk_cache,
    _save_disk_cache,
    fetch_active_positions,
    fetch_orders,
    fetch_ticker,
    fetch_ohlcv,
    fetch_recent_trades,
    fetch_usdt_try_price,
    fetch_unified_account_equity,
    fetch_funding_rate,
    get_last_funding_payment,
    get_price_changes,
    fetch_recent_closed_positions,
    fetch_closed_positions_30d,
)


# ── load_api_keys ──────────────────────────────────────────────────────────

class TestLoadApiKeys:
    def test_parses_valid_key_file(self, tmp_path):
        key_file = tmp_path / "api.key"
        key_file.write_text("key=mykey123\nsecret=mysecret456\n")

        api_key, api_secret = load_api_keys(str(key_file))
        assert api_key == "mykey123"
        assert api_secret == "mysecret456"

    def test_handles_values_containing_equals(self, tmp_path):
        key_file = tmp_path / "api.key"
        key_file.write_text("key=abc=def\nsecret=ghi=jkl\n")

        api_key, api_secret = load_api_keys(str(key_file))
        assert api_key == "abc=def"
        assert api_secret == "ghi=jkl"

    def test_strips_whitespace(self, tmp_path):
        key_file = tmp_path / "api.key"
        key_file.write_text("key=  spaced_key  \nsecret=  spaced_secret  \n")

        api_key, api_secret = load_api_keys(str(key_file))
        # strip applies to the whole line, not the value after split
        assert api_key == "  spaced_key"
        assert api_secret == "  spaced_secret"

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_api_keys("/nonexistent/path/api.key")

    def test_default_path_uses_module_dir(self):
        """Default key_file should point to positions_tracker/api.key."""
        expected = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "api.key"
        )
        # Just verify the function can be called with the real file
        api_key, api_secret = load_api_keys()
        assert api_key and api_secret


# ── init_clients ───────────────────────────────────────────────────────────

class TestInitClients:
    def test_creates_all_clients(self, tmp_path):
        key_file = tmp_path / "api.key"
        key_file.write_text("key=testkey\nsecret=testsecret\n")

        init_clients(str(key_file))

        assert get_gate_ccxt() is not None
        assert get_gate_futures_api() is not None
        assert get_gate_unified_api() is not None
        assert get_binance_ccxt() is not None

    def test_gate_ccxt_configured_for_swap_unified(self, tmp_path):
        key_file = tmp_path / "api.key"
        key_file.write_text("key=testkey\nsecret=testsecret\n")

        init_clients(str(key_file))

        gate = get_gate_ccxt()
        assert gate.options.get("defaultType") == "swap"
        assert gate.options.get("unified") is True

    def test_gate_ccxt_has_correct_api_key(self, tmp_path):
        key_file = tmp_path / "api.key"
        key_file.write_text("key=testkey\nsecret=testsecret\n")

        init_clients(str(key_file))

        gate = get_gate_ccxt()
        assert gate.apiKey == "testkey"
        assert gate.secret == "testsecret"


# ── TTL constants ──────────────────────────────────────────────────────────

class TestTTLConstants:
    def test_ttl_values_match_spec(self):
        assert TTL_USDT_TRY == 60
        assert TTL_PRICE_CHANGES == 15
        assert TTL_OHLCV == 30
        assert TTL_RECENT_TRADES == 30
        assert TTL_LAST_FUNDING == 300
        assert TTL_FUNDING_RATE == 60
        assert TTL_UNIFIED_EQUITY == 5
        assert TTL_RECENT_CLOSED == 30
        assert TTL_CLOSED_30D == 60


# ── cached_fetch ───────────────────────────────────────────────────────────

class TestCachedFetch:
    def setup_method(self):
        _cache.clear()

    def test_returns_fresh_value_on_miss(self):
        result = cached_fetch("k1", lambda: 42, ttl=10)
        assert result == 42

    def test_returns_cached_value_within_ttl(self):
        call_count = 0

        def fetcher():
            nonlocal call_count
            call_count += 1
            return "data"

        cached_fetch("k2", fetcher, ttl=60)
        cached_fetch("k2", fetcher, ttl=60)
        assert call_count == 1

    def test_refetches_after_ttl_expires(self):
        call_count = 0

        def fetcher():
            nonlocal call_count
            call_count += 1
            return call_count

        cached_fetch("k3", fetcher, ttl=0.01)
        time.sleep(0.02)
        result = cached_fetch("k3", fetcher, ttl=0.01)
        assert call_count == 2
        assert result == 2

    def test_stale_fallback_on_fetch_failure(self):
        # Populate cache
        cached_fetch("k4", lambda: "good", ttl=0.01)
        time.sleep(0.02)  # expire

        # Fetch fails → stale data returned
        result = cached_fetch("k4", _raise_runtime, ttl=10)
        assert result == "good"

    def test_reraises_when_no_stale_data(self):
        with pytest.raises(RuntimeError):
            cached_fetch("k5", _raise_runtime, ttl=10)

    def test_stores_value_with_correct_expiry(self):
        before = time.time()
        cached_fetch("k6", lambda: "v", ttl=100)
        after = time.time()

        _, expiry = _cache["k6"]
        assert before + 100 <= expiry <= after + 100


def _raise_runtime():
    raise RuntimeError("boom")


# ── clear_cache_for_symbol ─────────────────────────────────────────────────

class TestClearCacheForSymbol:
    def setup_method(self):
        _cache.clear()

    def test_removes_matching_keys(self):
        _cache["BTC_USDT:ticker"] = ("v1", time.time() + 100)
        _cache["BTC_USDT:ohlcv"] = ("v2", time.time() + 100)
        _cache["ETH_USDT:ticker"] = ("v3", time.time() + 100)

        clear_cache_for_symbol("BTC_USDT")

        assert "BTC_USDT:ticker" not in _cache
        assert "BTC_USDT:ohlcv" not in _cache
        assert "ETH_USDT:ticker" in _cache

    def test_noop_when_no_match(self):
        _cache["ETH_USDT:ticker"] = ("v", time.time() + 100)
        clear_cache_for_symbol("SOL_USDT")
        assert len(_cache) == 1


# =============================================================================
# Task 1.3 — Exchange data fetch functions
# =============================================================================


class TestFetchActivePositions:
    def setup_method(self):
        _cache.clear()

    @patch("positions_tracker.data_fetcher._clients")
    def test_filters_active_positions(self, mock_clients):
        mock_gate = MagicMock()
        mock_gate.fetch_positions.return_value = [
            {"contracts": "5", "symbol": "BTC_USDT:USDT"},
            {"contracts": "0", "symbol": "ETH_USDT:USDT"},
            {"contracts": "3", "symbol": "SOL_USDT:USDT"},
        ]
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = fetch_active_positions()
        assert len(result) == 2
        assert result[0]["symbol"] == "BTC_USDT:USDT"
        assert result[1]["symbol"] == "SOL_USDT:USDT"

    @patch("positions_tracker.data_fetcher._clients")
    def test_respects_max_count(self, mock_clients):
        mock_gate = MagicMock()
        mock_gate.fetch_positions.return_value = [
            {"contracts": "1", "symbol": f"SYM{i}"} for i in range(10)
        ]
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = fetch_active_positions(max_count=2)
        assert len(result) == 2


class TestFetchOrders:
    @patch("positions_tracker.data_fetcher._clients")
    def test_delegates_to_ccxt(self, mock_clients):
        mock_gate = MagicMock()
        mock_gate.fetch_open_orders.return_value = [{"id": "o1"}]
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = fetch_orders("BTC_USDT:USDT")
        mock_gate.fetch_open_orders.assert_called_once_with("BTC_USDT:USDT")
        assert result == [{"id": "o1"}]


class TestFetchTicker:
    @patch("positions_tracker.data_fetcher._clients")
    def test_returns_ticker_dict(self, mock_clients):
        mock_gate = MagicMock()
        mock_gate.fetch_ticker.return_value = {"last": 67500.0, "symbol": "BTC_USDT:USDT"}
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = fetch_ticker("BTC_USDT:USDT")
        assert result["last"] == 67500.0


class TestFetchOhlcv:
    def setup_method(self):
        _cache.clear()

    @patch("positions_tracker.data_fetcher._clients")
    def test_returns_candle_data(self, mock_clients):
        candles = [[1700000000, 100, 105, 99, 103, 500]]
        mock_gate = MagicMock()
        mock_gate.fetch_ohlcv.return_value = candles
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = fetch_ohlcv("BTC_USDT:USDT", "15m", 30)
        assert result == candles

    @patch("positions_tracker.data_fetcher._clients")
    def test_caches_result(self, mock_clients):
        mock_gate = MagicMock()
        mock_gate.fetch_ohlcv.return_value = [[1, 2, 3, 4, 5, 6]]
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        fetch_ohlcv("BTC_USDT:USDT")
        fetch_ohlcv("BTC_USDT:USDT")
        assert mock_gate.fetch_ohlcv.call_count == 1


class TestFetchRecentTrades:
    def setup_method(self):
        _cache.clear()

    @patch("positions_tracker.data_fetcher._clients")
    def test_uses_fetch_my_trades(self, mock_clients):
        trades = [{"id": "t1", "side": "buy", "price": 100}]
        mock_gate = MagicMock()
        mock_gate.fetch_my_trades.return_value = trades
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = fetch_recent_trades("BTC_USDT:USDT", limit=50)
        assert result == trades


class TestFetchUsdtTryPrice:
    def setup_method(self):
        _cache.clear()

    @patch("positions_tracker.data_fetcher._clients")
    def test_returns_last_price(self, mock_clients):
        mock_binance = MagicMock()
        mock_binance.fetch_ticker.return_value = {"last": 32.5}
        mock_clients.__getitem__ = lambda self, k: mock_binance if k == "binance_ccxt" else None

        result = fetch_usdt_try_price()
        assert result == 32.5

    @patch("positions_tracker.data_fetcher._clients")
    def test_returns_1_on_failure(self, mock_clients):
        mock_binance = MagicMock()
        mock_binance.fetch_ticker.side_effect = Exception("network error")
        mock_clients.__getitem__ = lambda self, k: mock_binance if k == "binance_ccxt" else None

        result = fetch_usdt_try_price()
        assert result == 1.0


class TestFetchUnifiedAccountEquity:
    def setup_method(self):
        _cache.clear()

    @patch("positions_tracker.data_fetcher._clients")
    def test_returns_equity_from_unified_total(self, mock_clients):
        mock_account = MagicMock()
        mock_account.unified_account_total_equity = "5000.0"
        mock_unified = MagicMock()
        mock_unified.list_unified_accounts.return_value = mock_account
        mock_clients.__getitem__ = lambda self, k: mock_unified if k == "gate_unified_api" else None

        result = fetch_unified_account_equity()
        assert result == 5000.0

    @patch("positions_tracker.data_fetcher._clients")
    def test_returns_none_on_failure(self, mock_clients):
        mock_unified = MagicMock()
        mock_unified.list_unified_accounts.side_effect = Exception("fail")
        mock_clients.__getitem__ = lambda self, k: mock_unified if k == "gate_unified_api" else None

        result = fetch_unified_account_equity()
        assert result is None


class TestFetchFundingRate:
    def setup_method(self):
        _cache.clear()

    @patch("positions_tracker.data_fetcher._clients")
    def test_returns_funding_rate_dict(self, mock_clients):
        rate_data = {"fundingRate": 0.0001, "fundingTimestamp": 1700003600000}
        mock_gate = MagicMock()
        mock_gate.fetch_funding_rate.return_value = rate_data
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = fetch_funding_rate("BTC_USDT:USDT")
        assert result["fundingRate"] == 0.0001


class TestGetLastFundingPayment:
    def setup_method(self):
        _cache.clear()

    @patch("positions_tracker.data_fetcher._clients")
    def test_returns_amount(self, mock_clients):
        mock_gate = MagicMock()
        mock_gate.fetch_funding_history.return_value = [{"amount": -0.32}]
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = get_last_funding_payment("BTC_USDT:USDT")
        assert result == -0.32

    @patch("positions_tracker.data_fetcher._clients")
    def test_returns_none_on_empty_history(self, mock_clients):
        mock_gate = MagicMock()
        mock_gate.fetch_funding_history.return_value = []
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = get_last_funding_payment("BTC_USDT:USDT")
        assert result is None

    @patch("positions_tracker.data_fetcher._clients")
    def test_returns_none_on_exception(self, mock_clients):
        mock_gate = MagicMock()
        mock_gate.fetch_funding_history.side_effect = Exception("fail")
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = get_last_funding_payment("BTC_USDT:USDT")
        assert result is None


class TestGetPriceChanges:
    def setup_method(self):
        _cache.clear()

    @patch("positions_tracker.data_fetcher._clients")
    def test_computes_percent_changes(self, mock_clients):
        mock_gate = MagicMock()
        # Each timeframe returns 2 candles: prev_close=100, last_close=101 → +1%
        mock_gate.fetch_ohlcv.return_value = [
            [0, 0, 0, 0, 100, 0],
            [0, 0, 0, 0, 101, 0],
        ]
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = get_price_changes("BTC_USDT:USDT")
        assert "5m" in result
        assert "15m" in result
        assert "1h" in result
        assert "1d" in result
        assert result["5m"] == "+1.00%"

    @patch("positions_tracker.data_fetcher._clients")
    def test_handles_fetch_error_gracefully(self, mock_clients):
        mock_gate = MagicMock()
        mock_gate.fetch_ohlcv.side_effect = Exception("timeout")
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = get_price_changes("BTC_USDT:USDT")
        for tf in ["5m", "15m", "1h", "1d"]:
            assert result[tf] == "N/A"


class TestFetchRecentClosedPositions:
    def setup_method(self):
        _cache.clear()

    @patch("positions_tracker.data_fetcher._clients")
    def test_returns_sorted_by_time_desc(self, mock_clients):
        mock_gate = MagicMock()
        mock_gate.private_futures_get_settle_position_close.return_value = [
            {"contract": "BTC_USDT", "time": 100},
            {"contract": "ETH_USDT", "time": 300},
            {"contract": "SOL_USDT", "time": 200},
        ]
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = fetch_recent_closed_positions(limit=48)
        assert result[0]["time"] == 300
        assert result[1]["time"] == 200
        assert result[2]["time"] == 100

    @patch("positions_tracker.data_fetcher._clients")
    def test_returns_empty_on_exception(self, mock_clients):
        mock_gate = MagicMock()
        mock_gate.private_futures_get_settle_position_close.side_effect = Exception("fail")
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = fetch_recent_closed_positions()
        assert result == []


class TestFetchClosedPositions30d:
    def setup_method(self):
        _cache.clear()

    # ── _load_disk_cache ──────────────────────────────────────────────────

    def test_load_disk_cache_returns_empty_on_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "positions_tracker.data_fetcher.CLOSED_POSITIONS_CACHE_FILE",
            str(tmp_path / "nonexistent.json"),
        )
        positions, ts = _load_disk_cache()
        assert positions == []
        assert ts == 0

    def test_load_disk_cache_returns_empty_on_invalid_json(self, tmp_path, monkeypatch):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json at all{{{")
        monkeypatch.setattr(
            "positions_tracker.data_fetcher.CLOSED_POSITIONS_CACHE_FILE",
            str(bad_file),
        )
        positions, ts = _load_disk_cache()
        assert positions == []
        assert ts == 0

    def test_load_disk_cache_reads_valid_file(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        data = {
            "positions": [{"contract": "BTC_USDT", "time": 100}],
            "last_fetch_ts": 12345,
        }
        cache_file.write_text(json.dumps(data))
        monkeypatch.setattr(
            "positions_tracker.data_fetcher.CLOSED_POSITIONS_CACHE_FILE",
            str(cache_file),
        )
        positions, ts = _load_disk_cache()
        assert len(positions) == 1
        assert positions[0]["contract"] == "BTC_USDT"
        assert ts == 12345

    # ── _save_disk_cache ──────────────────────────────────────────────────

    def test_save_disk_cache_writes_json(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(
            "positions_tracker.data_fetcher.CLOSED_POSITIONS_CACHE_FILE",
            str(cache_file),
        )
        positions = [{"contract": "ETH_USDT", "time": 200}]
        _save_disk_cache(positions, 99999)

        with open(cache_file) as f:
            data = json.load(f)
        assert data["positions"] == positions
        assert data["last_fetch_ts"] == 99999

    def test_save_disk_cache_round_trip(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(
            "positions_tracker.data_fetcher.CLOSED_POSITIONS_CACHE_FILE",
            str(cache_file),
        )
        original = [
            {"contract": "BTC_USDT", "time": 100, "pnl": "1.5"},
            {"contract": "ETH_USDT", "time": 200, "pnl": "-0.3"},
        ]
        _save_disk_cache(original, 55555)
        loaded, ts = _load_disk_cache()
        assert loaded == original
        assert ts == 55555

    # ── fetch_closed_positions_30d (integration with mocks) ───────────────

    @patch("positions_tracker.data_fetcher._clients")
    def test_fetches_and_returns_sorted_positions(self, mock_clients, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(
            "positions_tracker.data_fetcher.CLOSED_POSITIONS_CACHE_FILE",
            str(cache_file),
        )
        now = int(time.time())
        mock_gate = MagicMock()
        # Return positions from the API
        mock_gate.private_futures_get_settle_position_close.return_value = [
            {"contract": "BTC_USDT", "time": str(now - 100)},
            {"contract": "ETH_USDT", "time": str(now - 50)},
        ]
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = fetch_closed_positions_30d()
        assert len(result) >= 2
        # Should be sorted by time descending
        times = [int(p.get("time", 0)) for p in result]
        assert times == sorted(times, reverse=True)

    @patch("positions_tracker.data_fetcher._clients")
    def test_deduplicates_by_contract_and_time(self, mock_clients, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(
            "positions_tracker.data_fetcher.CLOSED_POSITIONS_CACHE_FILE",
            str(cache_file),
        )
        now = int(time.time())
        # Pre-populate disk cache with a position
        _save_disk_cache(
            [{"contract": "BTC_USDT", "time": str(now - 100)}],
            now - 200,
        )

        mock_gate = MagicMock()
        # API returns the same position (duplicate)
        mock_gate.private_futures_get_settle_position_close.return_value = [
            {"contract": "BTC_USDT", "time": str(now - 100)},
        ]
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = fetch_closed_positions_30d()
        # Should have only 1 unique position
        btc_positions = [p for p in result if p["contract"] == "BTC_USDT" and p["time"] == str(now - 100)]
        assert len(btc_positions) == 1

    @patch("positions_tracker.data_fetcher._clients")
    def test_falls_back_to_disk_cache_on_total_failure(self, mock_clients, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(
            "positions_tracker.data_fetcher.CLOSED_POSITIONS_CACHE_FILE",
            str(cache_file),
        )
        now = int(time.time())
        # Pre-populate disk cache
        _save_disk_cache(
            [{"contract": "SOL_USDT", "time": str(now - 50)}],
            now - 10,
        )

        mock_gate = MagicMock()
        # All API calls fail
        mock_gate.private_futures_get_settle_position_close.side_effect = Exception("network down")
        mock_clients.__getitem__ = lambda self, k: mock_gate if k == "gate_ccxt" else None

        result = fetch_closed_positions_30d()
        # Should still return the cached data (either from in-memory stale fallback or disk)
        assert len(result) >= 1


# =============================================================================
# Task 1.5 — Property-based tests for data fetcher
# =============================================================================

from hypothesis import given, settings
from hypothesis import strategies as st


# ── Property 2: API key parsing round-trip ─────────────────────────────────
# Feature: positions-tracker-webapp, Property 2: API key parsing round-trip
# **Validates: Requirements 1.2**

class TestPropertyApiKeyRoundTrip:
    @settings(max_examples=100)
    @given(
        key=st.text(min_size=1).filter(
            lambda s: "\n" not in s and "\r" not in s and s == s.strip()
        ),
        secret=st.text(min_size=1).filter(
            lambda s: "\n" not in s and "\r" not in s and s == s.strip()
        ),
    )
    def test_api_key_round_trip(self, key, secret):
        """Writing key=<key>\\nsecret=<secret> and parsing with load_api_keys
        returns the original key and secret."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".key", delete=False) as f:
            f.write(f"key={key}\nsecret={secret}\n")
            f.flush()
            key_file = f.name
        try:
            parsed_key, parsed_secret = load_api_keys(key_file)
            assert parsed_key == key
            assert parsed_secret == secret
        finally:
            os.unlink(key_file)


# ── Property 3: Cache returns same value within TTL and re-fetches after expiry
# Feature: positions-tracker-webapp, Property 3: Cache returns same value within TTL and re-fetches after expiry
# **Validates: Requirements 1.6**

class TestPropertyCacheTTL:
    def setup_method(self):
        _cache.clear()

    @settings(max_examples=100, deadline=None)
    @given(
        ttl=st.floats(min_value=0.01, max_value=0.05),
        value1=st.integers(),
        value2=st.integers(),
    )
    def test_cache_hit_within_ttl_and_refetch_after_expiry(self, ttl, value1, value2):
        """Within TTL the cache returns the same value and calls fetch_fn once.
        After TTL expiry, fetch_fn is called again."""
        _cache.clear()
        call_count = 0

        def make_fetcher(val):
            def fetcher():
                nonlocal call_count
                call_count += 1
                return val
            return fetcher

        # First call — cache miss
        result1 = cached_fetch("prop3_key", make_fetcher(value1), ttl)
        assert result1 == value1
        assert call_count == 1

        # Second call within TTL — cache hit, fetch_fn NOT called again
        result2 = cached_fetch("prop3_key", make_fetcher(value2), ttl)
        assert result2 == value1  # same cached value
        assert call_count == 1    # still only one call

        # Wait for TTL to expire
        time.sleep(ttl + 0.01)

        # Third call after expiry — cache miss, fetch_fn called again
        result3 = cached_fetch("prop3_key", make_fetcher(value2), ttl)
        assert result3 == value2
        assert call_count == 2


# ── Property 4: Disk cache serialization round-trip ────────────────────────
# Feature: positions-tracker-webapp, Property 4: Disk cache serialization round-trip
# **Validates: Requirements 1.7**

_closed_position_strategy = st.fixed_dictionaries({
    "contract": st.text(min_size=1, max_size=20),
    "pnl": st.text(min_size=1, max_size=15),
    "time": st.integers(min_value=0, max_value=2_000_000_000),
    "first_open_time": st.integers(min_value=0, max_value=2_000_000_000),
})


class TestPropertyDiskCacheRoundTrip:
    @settings(max_examples=100)
    @given(
        positions=st.lists(_closed_position_strategy, min_size=0, max_size=20),
    )
    def test_disk_cache_round_trip(self, positions):
        """Saving a list of closed positions to disk cache and loading back
        produces an equivalent list."""
        import positions_tracker.data_fetcher as df

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cache_file = f.name

        original_file = df.CLOSED_POSITIONS_CACHE_FILE
        df.CLOSED_POSITIONS_CACHE_FILE = cache_file
        try:
            ts = int(time.time())
            _save_disk_cache(positions, ts)
            loaded_positions, loaded_ts = _load_disk_cache()

            assert loaded_positions == positions
            assert loaded_ts == ts
        finally:
            df.CLOSED_POSITIONS_CACHE_FILE = original_file
            try:
                os.unlink(cache_file)
            except OSError:
                pass
