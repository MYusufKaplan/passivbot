"""
Exchange data layer for the positions tracker web app.

Handles API key loading, exchange client initialization, and cached data fetching
from Gate.io (via CCXT and gate_api SDK) and Binance (via CCXT).
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone

import ccxt
from gate_api import ApiClient, Configuration, FuturesApi, UnifiedApi


# =============================================================================
# CACHING SYSTEM — Reduces API calls by caching slow-changing data
# =============================================================================

_cache: dict[str, tuple[object, float]] = {}  # key -> (value, expiry_timestamp)

# TTL constants (seconds) — match the original positions_tracker.py values
TTL_USDT_TRY = 60          # USDT/TRY exchange rate
TTL_PRICE_CHANGES = 15     # 5m/15m/1h/1d price change percentages
TTL_OHLCV = 30             # OHLCV / sparkline candle data
TTL_RECENT_TRADES = 30     # Recent trade executions
TTL_LAST_FUNDING = 300     # Last funding payment
TTL_FUNDING_RATE = 60      # Current funding rate
TTL_UNIFIED_EQUITY = 5     # Unified account equity
TTL_RECENT_CLOSED = 30     # Recent closed positions
TTL_CLOSED_30D = 60        # 30-day closed position history

CLOSED_POSITIONS_CACHE_FILE = '.closed_positions_cache.json'


def cached_fetch(key: str, fetch_fn, ttl: float):
    """Generic time-based caching wrapper.

    1. If *key* exists in ``_cache`` and hasn't expired, return the cached value.
    2. Otherwise call *fetch_fn()*, store the result with an expiry, and return it.
    3. If *fetch_fn()* raises and stale data exists, return the stale value.
    4. If *fetch_fn()* raises and there is no stale data, re-raise.

    Args:
        key: Cache key string.
        fetch_fn: Zero-argument callable that fetches fresh data.
        ttl: Time-to-live in seconds.

    Returns:
        The (possibly cached) value produced by *fetch_fn*.
    """
    now = time.time()
    if key in _cache:
        value, expires = _cache[key]
        if now < expires:
            return value
    try:
        value = fetch_fn()
        _cache[key] = (value, now + ttl)
        return value
    except Exception as e:
        # If fetch fails but we have stale data, return it
        if key in _cache:
            return _cache[key][0]
        raise e


def clear_cache_for_symbol(symbol: str):
    """Clear all cached entries whose key contains *symbol*.

    Useful when a position closes and stale data should be discarded.
    """
    keys_to_remove = [k for k in _cache if symbol in k]
    for k in keys_to_remove:
        _cache.pop(k, None)


def load_api_keys(key_file=None):
    """Read API key and secret from the api.key file.

    File format:
        line 1: key=<value>
        line 2: secret=<value>

    Args:
        key_file: Path to the key file. Defaults to api.key in the same
                  directory as this module.

    Returns:
        Tuple of (api_key, api_secret).
    """
    if key_file is None:
        key_file = os.path.join(os.path.dirname(__file__), "api.key")

    with open(key_file, "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip().split("=", 1)[1]
        api_secret = lines[1].strip().split("=", 1)[1]
    return api_key, api_secret


# ---------------------------------------------------------------------------
# Client holders – populated by init_clients()
# ---------------------------------------------------------------------------
_clients = {
    "gate_ccxt": None,       # ccxt.gateio – swap/unified
    "gate_futures_api": None, # gate_api FuturesApi
    "gate_unified_api": None, # gate_api UnifiedApi
    "binance_ccxt": None,     # ccxt.binance – for USDT/TRY
}


def init_clients(key_file=None):
    """Initialise all exchange clients and store them in ``_clients``.

    Creates:
    - CCXT Gate.io client configured for swap/unified mode
    - gate_api SDK Configuration, FuturesApi, and UnifiedApi clients
    - CCXT Binance client (used for USDT/TRY price)

    Args:
        key_file: Optional path forwarded to :func:`load_api_keys`.
    """
    api_key, api_secret = load_api_keys(key_file)

    # CCXT Gate.io – swap / unified
    _clients["gate_ccxt"] = ccxt.gateio({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
            "unified": True,
        },
    })

    # gate_api SDK – FuturesApi & UnifiedApi
    configuration = Configuration(key=api_key, secret=api_secret)
    api_client = ApiClient(configuration)
    _clients["gate_futures_api"] = FuturesApi(api_client)
    _clients["gate_unified_api"] = UnifiedApi(api_client)

    # CCXT Binance – public, no auth needed
    _clients["binance_ccxt"] = ccxt.binance()


def get_gate_ccxt():
    """Return the CCXT Gate.io client (swap/unified)."""
    return _clients["gate_ccxt"]


def get_gate_futures_api():
    """Return the gate_api FuturesApi client."""
    return _clients["gate_futures_api"]


def get_gate_unified_api():
    """Return the gate_api UnifiedApi client."""
    return _clients["gate_unified_api"]


def get_binance_ccxt():
    """Return the CCXT Binance client."""
    return _clients["binance_ccxt"]


# =============================================================================
# EXCHANGE DATA FETCH FUNCTIONS
# =============================================================================


def fetch_active_positions(max_count=3):
    """Fetch active futures positions from Gate.io.

    Calls ``gate_ccxt.fetch_positions()``, keeps only those with
    ``contracts > 0``, and returns up to *max_count* results.
    No caching — called fresh each poll.
    """
    positions = get_gate_ccxt().fetch_positions()
    active = [p for p in positions if float(p["contracts"]) > 0]
    return active[:max_count]


def fetch_orders(symbol):
    """Fetch open orders for *symbol* from Gate.io.

    No caching — order state changes frequently.
    """
    return get_gate_ccxt().fetch_open_orders(symbol)


def fetch_ticker(symbol):
    """Fetch the current ticker for *symbol* from Gate.io.

    Returns the full ticker dict.  No caching.
    """
    return get_gate_ccxt().fetch_ticker(symbol)


def fetch_ohlcv(symbol, timeframe="15m", limit=30):
    """Fetch OHLCV candle data for *symbol*.

    Cached with :data:`TTL_OHLCV` (30 s).

    Returns:
        List of ``[timestamp, open, high, low, close, volume]`` lists.
    """
    return cached_fetch(
        f"ohlcv:{symbol}:{timeframe}:{limit}",
        lambda: get_gate_ccxt().fetch_ohlcv(symbol, timeframe=timeframe, limit=limit),
        TTL_OHLCV,
    )


def fetch_recent_trades(symbol, limit=100):
    """Fetch recent trade executions for *symbol*.

    Cached with :data:`TTL_RECENT_TRADES` (30 s).
    Uses ``gate_ccxt.fetch_my_trades()``.
    """
    return cached_fetch(
        f"recent_trades:{symbol}",
        lambda: get_gate_ccxt().fetch_my_trades(symbol, limit=limit),
        TTL_RECENT_TRADES,
    )


def fetch_usdt_try_price():
    """Fetch the USDT/TRY exchange rate from Binance.

    Cached with :data:`TTL_USDT_TRY` (60 s).
    Returns ``1.0`` on any failure to avoid ``None`` propagation.
    """
    def _fetch():
        return get_binance_ccxt().fetch_ticker("USDT/TRY")["last"]

    try:
        return cached_fetch("usdt_try", _fetch, TTL_USDT_TRY)
    except Exception:
        return 1.0


def fetch_unified_account_equity():
    """Fetch total equity from the Gate.io unified account.

    Cached with :data:`TTL_UNIFIED_EQUITY` (5 s).
    Uses ``gate_unified_api.list_unified_accounts()``.

    Returns:
        Float equity value, or ``None`` on failure.
    """
    def _fetch():
        try:
            account = get_gate_unified_api().list_unified_accounts()

            if hasattr(account, "unified_account_total_equity"):
                return float(account.unified_account_total_equity)
            if hasattr(account, "total_equity"):
                return float(account.total_equity)

            # Fallback to USDT balance equity
            if hasattr(account, "balances") and "USDT" in account.balances:
                usdt_balance = account.balances["USDT"]
                if isinstance(usdt_balance, dict):
                    return float(usdt_balance.get("equity", 0))

            return None
        except Exception:
            return None

    return cached_fetch("unified_equity", _fetch, TTL_UNIFIED_EQUITY)


def fetch_funding_rate(symbol):
    """Fetch the current funding rate for *symbol*.

    Cached with :data:`TTL_FUNDING_RATE` (60 s).

    Returns:
        Dict with keys like ``fundingRate``, ``fundingTimestamp``, etc.
    """
    return cached_fetch(
        f"funding_rate:{symbol}",
        lambda: get_gate_ccxt().fetch_funding_rate(symbol),
        TTL_FUNDING_RATE,
    )


def get_last_funding_payment(symbol):
    """Fetch the last funding payment for *symbol*.

    Cached with :data:`TTL_LAST_FUNDING` (300 s).

    Returns:
        Float amount of the last funding payment, or ``None`` if unavailable.
    """
    def _fetch():
        history = get_gate_ccxt().fetch_funding_history(symbol, limit=1)
        if history and len(history) > 0:
            return float(history[0].get("amount", 0))
        return None

    try:
        return cached_fetch(f"last_funding:{symbol}", _fetch, TTL_LAST_FUNDING)
    except Exception:
        return None


def get_price_changes(symbol):
    """Compute price change percentages for multiple timeframes.

    Fetches 2 candles for each of ``[5m, 15m, 1h, 1d]`` and computes the
    percent change between the previous close and the current close.

    Cached with :data:`TTL_PRICE_CHANGES` (15 s).

    Returns:
        Dict like ``{"5m": "+0.12%", "15m": "-0.05%", ...}``.
    """
    def _fetch():
        intervals = ["5m", "15m", "1h", "1d"]
        changes = {}
        gate = get_gate_ccxt()
        for timeframe in intervals:
            try:
                ohlcv = gate.fetch_ohlcv(symbol, timeframe=timeframe, limit=2)
                if len(ohlcv) == 2:
                    prev_close = ohlcv[0][4]
                    last_close = ohlcv[1][4]
                    pct = ((last_close - prev_close) / prev_close) * 100
                    sign = "+" if pct >= 0 else ""
                    changes[timeframe] = f"{sign}{pct:.2f}%"
                else:
                    changes[timeframe] = "N/A"
            except Exception:
                changes[timeframe] = "N/A"
        return changes

    return cached_fetch(f"price_changes:{symbol}", _fetch, TTL_PRICE_CHANGES)


def fetch_recent_closed_positions(limit=48):
    """Fetch recently closed positions from Gate.io.

    Uses ``gate_ccxt.private_futures_get_settle_position_close()``.
    Cached with :data:`TTL_RECENT_CLOSED` (30 s).

    Returns:
        List of closed-position dicts sorted by ``time`` descending.
    """
    def _fetch():
        try:
            response = get_gate_ccxt().private_futures_get_settle_position_close(
                {"settle": "usdt", "limit": limit}
            )
            sorted_positions = sorted(
                response, key=lambda x: x.get("time", 0), reverse=True
            )
            return sorted_positions[:limit]
        except Exception:
            return []

    return cached_fetch("recent_closed_positions", _fetch, TTL_RECENT_CLOSED)


def _load_disk_cache():
    """Load closed positions from disk cache.

    Returns:
        Tuple of (positions_list, last_fetch_ts). Returns ([], 0) on
        FileNotFoundError or JSONDecodeError.
    """
    try:
        with open(CLOSED_POSITIONS_CACHE_FILE, 'r') as f:
            data = json.load(f)
            return data.get('positions', []), data.get('last_fetch_ts', 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return [], 0


def _save_disk_cache(positions, last_fetch_ts):
    """Save closed positions to disk cache.

    Writes JSON with keys ``positions`` and ``last_fetch_ts``.

    Args:
        positions: List of closed position dicts.
        last_fetch_ts: Unix timestamp of the last successful fetch.
    """
    try:
        with open(CLOSED_POSITIONS_CACHE_FILE, 'w') as f:
            json.dump({'positions': positions, 'last_fetch_ts': last_fetch_ts}, f)
    except Exception:
        pass


def fetch_closed_positions_30d():
    """Fetch closed positions for the last 30 days with disk-backed caching.

    Uses :func:`cached_fetch` with :data:`TTL_CLOSED_30D` (60 s). The inner
    fetch function:

    1. Computes a 30-day cutoff timestamp.
    2. Loads the disk cache.
    3. If ``last_fetch_ts`` > cutoff, fetches only the delta since
       ``last_fetch_ts``; otherwise starts fresh.
    4. Fetches in 1-day windows using
       ``gate_ccxt.private_futures_get_settle_position_close()``.
    5. Merges new + cached positions, deduplicates by ``(contract, time)``,
       and filters to the 30-day window.
    6. Sorts by ``time`` descending.
    7. Saves to disk cache.
    8. Falls back to disk cache on total failure.

    Returns:
        List of closed-position dicts sorted by ``time`` descending.
    """
    def _fetch():
        try:
            now = datetime.now(timezone.utc)
            cutoff_30d = int((now - timedelta(days=30)).timestamp())

            # Load existing cache from disk
            cached_positions, last_fetch_ts = _load_disk_cache()

            # Determine where to start fetching from
            if last_fetch_ts > cutoff_30d:
                fetch_start = datetime.fromtimestamp(last_fetch_ts, tz=timezone.utc)
            else:
                fetch_start = now - timedelta(days=30)
                cached_positions = []  # Cache is too old, start fresh

            # Fetch only the delta (new positions since last fetch)
            step = timedelta(days=1)
            new_positions = []
            current = fetch_start

            while current < now:
                window_end = min(current + step, now)
                from_ts = int(current.timestamp())
                to_ts = int(window_end.timestamp())
                try:
                    response = get_gate_ccxt().private_futures_get_settle_position_close({
                        'settle': 'usdt',
                        '_from': from_ts,
                        'to': to_ts,
                        'limit': 1000,
                    })
                    new_positions.extend(response)
                except Exception:
                    pass
                current = window_end

            # Merge: deduplicate by (contract, time) pair
            seen = set()
            merged = []
            for pos in new_positions + cached_positions:
                key = (pos.get('contract', ''), pos.get('time', 0))
                if key not in seen:
                    seen.add(key)
                    # Only keep positions within 30-day window
                    if int(pos.get('time', 0)) >= cutoff_30d:
                        merged.append(pos)

            merged.sort(key=lambda x: x.get('time', 0), reverse=True)

            # Save to disk
            _save_disk_cache(merged, int(now.timestamp()))

            return merged
        except Exception:
            # Try to return disk cache as fallback
            cached, _ = _load_disk_cache()
            return cached

    return cached_fetch('closed_positions_30d', _fetch, TTL_CLOSED_30D)
