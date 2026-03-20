"""
Pure calculation functions for the positions tracker web app.
Ported from positions_tracker.py — no side effects, no API calls.
"""

import math
import time


def calculate_break_even(entry_price, realized_pnl, contracts, contract_size):
    """Calculate break-even price for a position.

    Returns entry_price - (realized_pnl / (contracts * contract_size)),
    or None if position size is zero.
    """
    position_size = contracts * contract_size
    if position_size == 0:
        return None
    return entry_price - (realized_pnl / position_size)


def calculate_running_avg_buy_orders(buy_orders, position):
    """Compute cumulative average fill price for each buy order.

    Assumes all higher-priced buy orders and the existing position fill first.
    *buy_orders* is a list of dicts with ``price`` and ``remaining`` (or ``amount``) keys.
    *position* is a dict with ``contracts``, ``contractSize``, and ``entryPrice``.

    Returns a dict mapping each order's price → cumulative average price.
    """
    contracts = float(position.get("contracts", 0))
    contract_size = float(position.get("contractSize", 1))
    entry_price = float(position.get("entryPrice", 0))

    cumulative_qty = contracts * contract_size
    cumulative_cost = entry_price * cumulative_qty

    buy_avg_prices = {}

    # Sort buy orders by price descending (highest first)
    sorted_buys = sorted(buy_orders, key=lambda o: float(o["price"]), reverse=True)

    for order in sorted_buys:
        price = float(order["price"])
        qty = float(order.get("remaining", order.get("amount", 0))) * contract_size
        cumulative_cost += price * qty
        cumulative_qty += qty
        avg_price = cumulative_cost / cumulative_qty if cumulative_qty > 0 else price
        buy_avg_prices[price] = avg_price

    return buy_avg_prices


def interpolate_color(percent, side):
    """Smooth CSS color interpolation for order progress bars.

    RED  = (255, 83, 112)
    GREEN = (195, 232, 141)

    For "Buy":  green → red  (green when far, red when close to fill)
    For "Sell": red → green  (red when far, green when close to fill)

    Returns a hex color string like ``"#rrggbb"``.
    """
    RED = (255, 83, 112)
    GREEN = (195, 232, 141)

    if side == "Sell":
        start_rgb = RED
        end_rgb = GREEN
    else:
        start_rgb = GREEN
        end_rgb = RED

    r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * percent / 100)
    g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * percent / 100)
    b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * percent / 100)

    # Clamp to [0, 255]
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return f"#{r:02x}{g:02x}{b:02x}"


def calculate_order_progress(order, current_price, orders, position):
    """Calculate progress bar data for a single order.

    Returns a dict with:
      - progress_percent  (float, 0–100)
      - color             (hex string)
      - cumulative_avg_price  (float for buys, None for sells)
      - estimated_pnl         (float for sells, None for buys)
    """
    price = float(order["price"])
    side = order["side"].capitalize()
    contract_size = float(position.get("contractSize", 1))
    entry_price = float(position.get("entryPrice", 0))
    contracts = float(position.get("contracts", 0))
    realized_pnl = float(position.get("realizedPnl", 0))

    sell_orders = [o for o in orders if o["side"].lower() == "sell"]
    buy_orders = [o for o in orders if o["side"].lower() == "buy"]

    highest_buy = max((float(b["price"]) for b in buy_orders), default=current_price)
    current_sell_price = float(sell_orders[0]["price"]) if sell_orders else current_price

    all_orders = buy_orders + sell_orders

    if side == "Sell":
        bar_min = highest_buy
        bar_max = price
        value = current_price
    else:
        # Buy orders use negative prices for left-to-right visual consistency
        bar_min = -1 * current_sell_price
        bar_max = -1 * price
        value = -1 * current_price

    # Special case: only one order total
    if len(all_orders) < 2:
        break_even_price = calculate_break_even(
            entry_price, realized_pnl, contracts, contract_size
        )
        bar_min = -1 * (break_even_price if break_even_price else 0)
        bar_max = -1 * price
        value = -1 * current_price

    distance = bar_max - bar_min
    if distance != 0:
        percent = ((value - bar_min) / distance) * 100
    else:
        percent = 100.0
    percent = max(0.0, min(100.0, percent))

    color = interpolate_color(percent, side)

    # Cumulative avg price for buy orders
    cumulative_avg_price = None
    if side == "Buy":
        buy_avg_prices = calculate_running_avg_buy_orders(buy_orders, position)
        cumulative_avg_price = buy_avg_prices.get(price)

    # Estimated PnL for sell orders
    estimated_pnl = None
    if side == "Sell":
        qty = float(order.get("remaining", order.get("amount", 0))) * contract_size
        estimated_pnl = (price - entry_price) * qty

    return {
        "progress_percent": percent,
        "color": color,
        "cumulative_avg_price": cumulative_avg_price,
        "estimated_pnl": estimated_pnl,
    }


def calculate_pnl_rate_stats(closed_positions):
    """Calculate ¢/s rates for 24h, 7d, 30d windows from closed positions.

    Each position dict must have ``time`` (unix timestamp) and ``pnl`` keys.
    Returns a dict like::

        {"24h": {"cents_per_sec": float, "total_pnl": float, "count": int}, ...}

    Returns None if *closed_positions* is empty or None.
    """
    if not closed_positions:
        return None

    now = time.time()
    windows = {
        "24h": now - 86400,
        "7d": now - 604800,
        "30d": now - 2592000,
    }

    results = {}
    for label, cutoff in windows.items():
        total_pnl = 0.0
        count = 0
        for pos in closed_positions:
            close_time = int(pos.get("time", 0))
            if close_time >= cutoff:
                total_pnl += float(pos.get("pnl", 0))
                count += 1
        elapsed = now - cutoff
        pnl_per_sec = total_pnl / elapsed if elapsed > 0 else 0
        cents_per_sec = pnl_per_sec * 100
        results[label] = {
            "cents_per_sec": cents_per_sec,
            "total_pnl": total_pnl,
            "count": count,
        }

    return results


def calculate_projections(equity, rate_stats):
    """Compound growth projections based on PnL rate windows.

    Returns a dict with keys ``24h``, ``7d``, ``30d``, each containing:
      - daily_rate
      - 1d, 1w, 1m  (projected equity)
      - 2x, 5x, 10x (days to reach milestone)

    Returns None if equity <= 0 or rate_stats is falsy.
    """
    if not rate_stats or equity <= 0:
        return None

    windows = {
        "24h": {"pnl": rate_stats["24h"]["total_pnl"], "days": 1},
        "7d": {"pnl": rate_stats["7d"]["total_pnl"], "days": 7},
        "30d": {"pnl": rate_stats["30d"]["total_pnl"], "days": 30},
    }

    projections = {}
    for label, w in windows.items():
        daily_rate = (w["pnl"] / equity) / w["days"] if w["days"] > 0 else 0

        if daily_rate <= 0:
            projections[label] = None
            continue

        proj = {"daily_rate": daily_rate}

        # Projected equity at future horizons
        for horizon_days, horizon_label in [(1, "1d"), (7, "1w"), (30, "1m")]:
            proj[horizon_label] = equity * (1 + daily_rate) ** horizon_days

        # Milestone ETAs: days to reach Nx equity
        for mult in [2, 5, 10]:
            proj[f"{mult}x"] = math.log(mult) / math.log(1 + daily_rate)

        projections[label] = proj

    return projections


def format_duration(seconds):
    """Format a duration in seconds as ``Xh Xm Xs``.

    - If hours > 0: ``"Xh XXm XXs"`` (zero-padded minutes and seconds)
    - If hours == 0 and minutes > 0: ``"Xm XXs"`` (zero-padded seconds)
    - Otherwise: ``"Xs"``
    """
    seconds = int(seconds)
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs}h {mins:02d}m {secs:02d}s"
    elif mins > 0:
        return f"{mins}m {secs:02d}s"
    else:
        return f"{secs}s"


def normalize_pnl_gradient(pnl_values):
    """Linear min-max normalization of PnL values to [0.0, 1.0].

    If all values are equal, returns a list of 0.5.
    """
    if not pnl_values:
        return []

    min_val = min(pnl_values)
    max_val = max(pnl_values)
    val_range = max_val - min_val

    if val_range == 0:
        return [0.5] * len(pnl_values)

    return [(v - min_val) / val_range for v in pnl_values]


def normalize_pnl_rate_gradient(pnl_rate_values):
    """Logarithmic normalization of PnL rate values to [0.0, 1.0].

    Shifts values so the minimum maps to a small epsilon, applies log,
    then normalizes to [0, 1].  If all values are equal, returns a list of 0.5.
    """
    if not pnl_rate_values:
        return []

    min_val = min(pnl_rate_values)
    max_val = max(pnl_rate_values)

    if min_val == max_val:
        return [0.5] * len(pnl_rate_values)

    # Shift so minimum maps to a small epsilon, then apply log
    epsilon = 1e-10
    shifted_max = max_val - min_val + epsilon

    results = []
    for v in pnl_rate_values:
        shifted = v - min_val + epsilon
        if shifted > 0 and shifted_max > 0:
            log_norm = math.log(shifted) / math.log(shifted_max)
            log_norm = max(0.0, min(1.0, log_norm))
        else:
            log_norm = 0.5
        results.append(log_norm)

    return results
