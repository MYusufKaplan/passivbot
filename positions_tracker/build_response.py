"""
Response assembly module for the positions tracker web app.

Orchestrates data fetching and calculation into the final JSON response
structure consumed by the frontend dashboard.
"""

import time

from positions_tracker.calculations import (
    calculate_break_even,
    calculate_order_progress,
    calculate_pnl_rate_stats,
    calculate_projections,
)
from positions_tracker.data_fetcher import (
    fetch_active_positions,
    fetch_closed_positions_30d,
    fetch_funding_rate,
    fetch_orders,
    fetch_recent_closed_positions,
    fetch_recent_trades,
    fetch_ticker,
    fetch_unified_account_equity,
    fetch_usdt_try_price,
    get_last_funding_payment,
    get_price_changes,
)

STARTING_BALANCE = 150000


def build_dashboard_data() -> dict:
    """Assemble all dashboard data into a single JSON-serializable dict.

    Fetches active positions, account data, and closed position history,
    then computes derived metrics (break-even, order progress, PnL rates,
    projections) and returns the complete response structure.

    Returns ``{"error": str}`` on any unhandled exception instead of crashing.
    """
    try:
        # --- Active positions ---
        active_positions = fetch_active_positions(max_count=3)
        idle = len(active_positions) == 0

        # --- USDT/TRY rate (needed for conversions) ---
        usdt_try_rate = fetch_usdt_try_price()

        # --- Build position objects ---
        positions = []
        for pos in active_positions:
            positions.append(_build_position(pos, usdt_try_rate))

        # --- Account data ---
        equity_usdt = fetch_unified_account_equity()
        if equity_usdt is None:
            equity_usdt = 0.0
        equity_try = equity_usdt * usdt_try_rate
        profit_try = equity_try - STARTING_BALANCE

        # --- Closed positions & PnL stats ---
        recent_closed = fetch_recent_closed_positions(limit=100)
        closed_30d = fetch_closed_positions_30d()

        rate_stats = calculate_pnl_rate_stats(closed_30d)
        proj = calculate_projections(equity_usdt, rate_stats) if equity_usdt > 0 else None

        account = {
            "equity_usdt": equity_usdt,
            "equity_try": equity_try,
            "profit_try": profit_try,
            "rate_stats": rate_stats,
            "projections": proj,
        }

        # --- History list ---
        history = _build_history(recent_closed, idle)

        return {
            "positions": positions,
            "account": account,
            "history": history,
            "idle": idle,
            "timestamp": time.time(),
            "usdt_try_rate": usdt_try_rate,
        }
    except Exception as e:
        return {"error": str(e)}


def _build_position(pos: dict, usdt_try_rate: float) -> dict:
    """Build a single position dict for the API response.

    Fetches ticker, orders, OHLCV, recent trades, price changes, funding
    rate, and last funding payment for the given position, then computes
    break-even price and order progress.
    """
    symbol = pos.get("symbol", "")
    contracts = float(pos.get("contracts", 0))
    contract_size = float(pos.get("contractSize", 1))
    entry_price = float(pos.get("entryPrice", 0))
    realized_pnl = float(pos.get("realizedPnl", 0))

    # Fetch supplementary data
    ticker = fetch_ticker(symbol)
    current_price = float(ticker.get("last", 0)) if ticker else 0.0

    raw_orders = fetch_orders(symbol)
    trades_raw = fetch_recent_trades(symbol, limit=100)
    price_changes = get_price_changes(symbol)
    funding = fetch_funding_rate(symbol)
    last_payment = get_last_funding_payment(symbol)

    # Break-even price
    break_even_price = calculate_break_even(
        entry_price, realized_pnl, contracts, contract_size
    )

    # Build orders with progress
    orders = _build_orders(raw_orders, current_price, pos)

    # Build trades list
    trades = [
        {
            "timestamp": t.get("timestamp"),
            "side": t.get("side"),
            "price": float(t.get("price", 0)),
            "amount": float(t.get("amount", 0)),
        }
        for t in (trades_raw or [])
    ]

    # Funding info
    funding_rate = float(funding.get("fundingRate", 0)) if funding else 0.0
    next_funding_ts = int(funding.get("fundingTimestamp", 0) / 1000) if funding else 0
    expected_payment = funding_rate * contracts * contract_size * current_price

    # Visual symbol: strip suffix like "_USDT:USDT"
    visual_symbol = symbol.split("_")[0].split("/")[0] if symbol else symbol

    return {
        "symbol": symbol,
        "visual_symbol": visual_symbol,
        "contracts": contracts,
        "contract_size": contract_size,
        "current_price": current_price,
        "entry_price": entry_price,
        "break_even_price": break_even_price,
        "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
        "realized_pnl": realized_pnl,
        "leverage": pos.get("leverage"),
        "liquidation_price": float(pos["liquidationPrice"]) if pos.get("liquidationPrice") else None,
        "margin_ratio": float(pos["marginRatio"]) if pos.get("marginRatio") else None,
        "open_timestamp": int(pos.get("timestamp", 0) / 1000),
        "price_changes": price_changes or {},
        "ohlcv": [],
        "trades": trades,
        "orders": orders,
        "funding": {
            "rate": funding_rate,
            "expected_payment": expected_payment,
            "next_funding_timestamp": next_funding_ts,
            "last_payment": last_payment if last_payment is not None else 0.0,
        },
    }


def _build_orders(raw_orders: list, current_price: float, position: dict) -> list:
    """Build order list with progress data, sorted by price descending."""
    if not raw_orders:
        return []

    orders = []
    for order in raw_orders:
        progress = calculate_order_progress(
            order, current_price, raw_orders, position
        )
        orders.append({
            "id": order.get("id", ""),
            "side": order.get("side", ""),
            "price": float(order.get("price", 0)),
            "amount": float(order.get("amount", 0)),
            "remaining": float(order.get("remaining", order.get("amount", 0))),
            "progress_percent": progress["progress_percent"],
            "color": progress["color"],
            "cumulative_avg_price": progress["cumulative_avg_price"],
            "estimated_pnl": progress["estimated_pnl"],
        })

    # Sort by price descending
    orders.sort(key=lambda o: o["price"], reverse=True)
    return orders


def _build_history(recent_closed: list, idle: bool) -> list:
    """Build the closed position history list.

    Limits to 48 items when active, 100 when idle.
    """
    limit = 100 if idle else 48
    items = []

    for pos in (recent_closed or [])[:limit]:
        close_time = int(pos.get("time", 0))
        open_time = int(pos.get("first_open_time", 0))
        duration = close_time - open_time if close_time > open_time else 0
        pnl = float(pos.get("pnl", 0))
        pnl_per_sec = pnl / duration if duration > 0 else 0.0

        contract = pos.get("contract", "")
        # Extract visual symbol from contract name like "BTC_USDT"
        symbol = contract.split("_")[0] if contract else contract

        items.append({
            "symbol": symbol,
            "pnl": pnl,
            "pnl_per_sec": pnl_per_sec,
            "duration_seconds": duration,
            "close_timestamp": close_time,
            "open_timestamp": open_time,
        })

    return items
