import ccxt
import time
import sys
from rich.live import Live
from rich.table import Table
from rich.console import Console, Group
from rich.progress import Progress, BarColumn, TextColumn
from rich.columns import Columns
from rich.panel import Panel
from rich.align import Align
from rich.style import Style
from datetime import datetime,timedelta, timezone
from gate_api import ApiClient, Configuration, FuturesApi
import subprocess
import re
import math

# =============================================================================
# DEBUGGING NOTE:
# If closed positions aren't showing, run: python test_gateio_api.py
# This will test the Gate.io API and show what data is available
# =============================================================================

STARTING_BALANCE = 0

# =============================================================================
# CACHING SYSTEM - Reduces API calls by caching slow-changing data
# =============================================================================
_cache = {}

def cached_fetch(key, fetch_fn, ttl_seconds):
    """Generic time-based caching wrapper"""
    now = time.time()
    if key in _cache:
        value, expires = _cache[key]
        if now < expires:
            return value
    try:
        value = fetch_fn()
        _cache[key] = (value, now + ttl_seconds)
        return value
    except Exception as e:
        # If fetch fails but we have stale data, return it
        if key in _cache:
            return _cache[key][0]
        raise e

def clear_cache_for_symbol(symbol):
    """Clear cached data for a specific symbol (call when position closes)"""
    keys_to_remove = [k for k in _cache if symbol in k]
    for k in keys_to_remove:
        _cache.pop(k, None)

def play_sound(path):
    subprocess.run(["aplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def load_api_keys():
    with open('api.key', 'r') as file:
        lines = file.readlines()
        api_key = lines[0].strip().split('=')[1]
        api_secret = lines[1].strip().split('=')[1]
    return api_key, api_secret

API_KEY, API_SECRET = load_api_keys()

gate = ccxt.gateio({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
        'unified': True  # Enable unified account mode
    }
})


console = Console(force_terminal=True, no_color=False, color_system="truecolor", log_path=False, width=255)

def hide_cursor():
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

def show_cursor():
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()

def interpolate_color(percent, side):
    RED=(255, 83, 112)
    GREEN=(195, 232, 141)
    if side == "Sell":
        start_rgb = RED
        end_rgb = GREEN
    else:
        start_rgb = GREEN
        end_rgb = RED
    r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * percent / 100)
    g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * percent / 100)
    b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * percent / 100)
    return f"#{r:02x}{g:02x}{b:02x}"

def fetch_balance_unified():
    """Fetch balance with unified account support"""
    try:
        x = gate.balance
        y = gate.currencies
        # Try unified account balance first
        balance = gate.fetch_balance(params={'type': 'unified'})
        return balance
    except Exception as e:
        # Fallback to regular futures balance
        try:
            balance = gate.fetch_balance(params={'type': 'swap'})
            return balance
        except Exception as e2:
            # Final fallback to default balance
            return gate.fetch_balance()

def fetch_active_positions(max_count=3):
    positions = gate.fetch_positions()
    active = [p for p in positions if float(p['contracts']) > 0]
    return active[:max_count]


def fetch_orders(symbol):
    return gate.fetch_open_orders(symbol)

class ColoredBarColumn(BarColumn):
    def __init__(self, bar_color: str, **kwargs):
        super().__init__(bar_width=None, complete_style=Style(color=bar_color), **kwargs)

def format_trade_duration(position):
    ts = position.get("timestamp")
    if not ts:
        return "—"
    open_time = datetime.fromtimestamp(ts / 1000)  # Convert ms to datetime
    elapsed = datetime.now() - open_time
    return str(elapsed).split('.')[0]

def calculate_order_volume_sum(order_book, min_price, max_price, side):
    total_volume = 0
    if side == 'buy':
        for order in order_book['bids']:
            price = float(order[0])  # Price of the buy order
            size = float(order[1])   # Size of the asset being bought
            if min_price <= price <= max_price:
                total_volume += price * size  # Convert to USDT
    elif side == 'sell':
        for order in order_book['asks']:
            price = float(order[0])  # Price of the sell order
            size = float(order[1])   # Size of the asset being sold
            if min_price <= price <= max_price:
                total_volume += price * size  # Convert to USDT
    return total_volume

def fetch_usdt_try_price():
    """Fetch USDT/TRY price - cached for 60s (forex doesn't move fast)"""
    def _fetch():
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker('USDT/TRY')
        return ticker['last']
    
    try:
        return cached_fetch('usdt_try', _fetch, ttl_seconds=60)
    except Exception as e:
        print(f"Error fetching USDT/TRY price: {e}")
        return 1.0  # Fallback to avoid None errors

def format_duration(seconds):
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs}h {mins:02d}m {secs:02d}s"
    elif mins > 0:
        return f"{mins}m {secs:02d}s"
    else:
        return f"{secs}s"

def calculate_running_avg_buy_orders(buy_orders):
    """
    Calculate the cumulative average price of all open buy orders,
    assuming all are filled cumulatively.
    buy_orders: list of dicts with 'price' and 'remaining' (qty)
    Returns cumulative average price (float) or None if no buys.
    """
    total_qty = 0
    total_cost = 0

    # Sort buys by price ascending or timestamp if available for consistency
    # Here sorted by price ascending just as a fallback
    sorted_buys = sorted(buy_orders, key=lambda o: float(o['price']))

    for order in sorted_buys:
        price = float(order['price'])
        qty = float(order.get('remaining', order.get('amount', 0)))
        total_cost += price * qty
        total_qty += qty

    if total_qty == 0:
        return None
    return total_cost / total_qty

def fetch_position_history_stats(start_date):
    SETTLE = "usdt"  # or "btc" or "eth" depending on what you're trading
    STEP = timedelta(days=1)
    end_date = datetime.now(timezone.utc)
    configuration = Configuration(key=API_KEY, secret=API_SECRET)
    client = ApiClient(configuration)
    futures_api = FuturesApi(client)

    all_closed_positions = []
    pnls = {}
    win_count = 0
    loss_count = 0
    total_pnl = 0.0

    start_time = start_date

    while start_time < end_date:
        window_end = min(start_time + STEP, end_date)
        from_ts = int(start_time.timestamp())
        to_ts = int(window_end.timestamp())

        try:
            resp = futures_api.list_position_close(
                settle=SETTLE,
                _from=from_ts,
                to=to_ts,
                limit=1000,
                offset=0
            )

            for position in resp:
                symbol = position['contract']
                pnl = float(position['pnl'])
                total_pnl += pnl
                pnls[symbol] = pnls.get(symbol, 0.0) + pnl
                all_closed_positions.append(position)

                if pnl >= 0:
                    win_count += 1
                else:
                    loss_count += 1

        except Exception as e:
            console.log(f"[red]❌ Failed fetching history for {start_time} – {window_end}[/]: {e}")

        start_time = window_end

    total_trades = win_count + loss_count
    avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

    best_symbol = max(pnls.items(), key=lambda x: x[1], default=("N/A", 0))
    worst_symbol = min(pnls.items(), key=lambda x: x[1], default=("N/A", 0))

    stats_summary = {
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "avg_pnl_per_trade": avg_pnl_per_trade,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "best_symbol": best_symbol,
        "worst_symbol": worst_symbol
    }

    return pnls, all_closed_positions, stats_summary

def get_last_funding_payment(symbol):
    """Fetch the last funding payment for a symbol - cached for 5 min (historical data)"""
    def _fetch():
        funding_history = gate.fetch_funding_history(symbol, limit=1)
        if funding_history and len(funding_history) > 0:
            last_payment = float(funding_history[0].get('amount', 0))
            payment_color = "green" if last_payment >= 0 else "red"
            return f"[{payment_color}]{last_payment:+.5f}[/{payment_color}]"
        return "N/A"
    
    try:
        return cached_fetch(f'last_funding_{symbol}', _fetch, ttl_seconds=300)
    except Exception as e:
        console.log(f"[dim red]Funding history error for {symbol}: {e}[/]")
        return "N/A"

def get_funding_info(symbol, position):
    """Fetch next funding rate and time remaining until funding - rate cached for 60s"""
    try:
        # Cache the funding rate info (doesn't change often)
        def _fetch_rate():
            return gate.fetch_funding_rate(symbol)
        
        funding_rate_info = cached_fetch(f'funding_rate_{symbol}', _fetch_rate, ttl_seconds=60)
        
        # Get next funding time (in milliseconds)
        next_funding_time = funding_rate_info.get('fundingTimestamp')
        funding_rate = funding_rate_info.get('fundingRate', 0)
        
        if next_funding_time:
            # Convert to datetime - time calculation is always fresh
            next_funding_dt = datetime.fromtimestamp(next_funding_time / 1000)
            now = datetime.now()
            time_remaining = next_funding_dt - now
            
            # Format time remaining
            total_seconds = int(time_remaining.total_seconds())
            if total_seconds < 0:
                time_str = "Calculating..."
            else:
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                time_str = f"{hours}h {minutes:02d}m {seconds:02d}s"
            
            # Calculate expected funding payment
            contracts = float(position.get('contracts', 0))
            contract_size = float(position.get('contractSize', 1))
            entry_price = float(position.get('entryPrice', 0))
            position_value = contracts * contract_size * entry_price
            expected_funding = -1 * position_value * funding_rate
            
            # Format funding rate as percentage
            funding_rate_pct = funding_rate * 100
            if funding_rate < 0:
                rate_color = "green"
                emoji = "💰"
            else:
                rate_color = "red"
                emoji = "💸"
            
            expected_color = "green" if expected_funding >= 0 else "red"
            rate_str = f"{emoji} [{rate_color}]{funding_rate_pct:+.4f}%[/{rate_color}] ([{expected_color}]{expected_funding:+.5f}[/{expected_color}])"
            
            return rate_str, time_str
        else:
            return "N/A", "N/A"
            
    except Exception as e:
        return "Err", "Err"

def get_price_changes(symbol):
    """Get price changes across timeframes - cached for 15s (historical candle data)"""
    def _fetch():
        intervals = ["5m", "15m", "1h", "1d"]
        changes = {}
        for timeframe in intervals:
            try:
                ohlcv = gate.fetch_ohlcv(symbol, timeframe=timeframe, limit=2)
                if len(ohlcv) == 2:
                    prev_close = ohlcv[0][4]
                    last_close = ohlcv[1][4]
                    percent_change = ((last_close - prev_close) / prev_close) * 100
                    direction = "📈" if percent_change >= 0 else "📉"
                    changes[timeframe] = f"{percent_change:.3f}%{direction}"
                else:
                    changes[timeframe] = "N/A"
            except Exception as e:
                changes[timeframe] = "Err"
        return changes
    
    return cached_fetch(f'price_changes_{symbol}', _fetch, ttl_seconds=15)

def create_sparkline(prices, width=30, execution_data=None):
    """Create a simple ASCII sparkline from price data using bar characters with execution highlighting"""
    if len(prices) < 2:
        return "─" * width, 0, 0
    
    min_price = min(prices)
    max_price = max(prices)
    
    if max_price == min_price:
        return "─" * width, min_price, max_price
    
    # Bar characters for different price levels
    chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    normalized = []
    
    for i, price in enumerate(prices):
        norm = (price - min_price) / (max_price - min_price)
        char_idx = min(int(norm * (len(chars) - 1)), len(chars) - 1)
        char = chars[char_idx]
        
        # Check if this bar has executions and color accordingly
        if execution_data and i in execution_data:
            if execution_data[i] == 'buy':
                char = f"[blue]{char}[/blue]"
            elif execution_data[i] == 'sell':
                char = f"[purple]{char}[/purple]"
        
        normalized.append(char)
    
    # Truncate or pad to desired width
    if len(normalized) > width:
        # Take the most recent data points
        normalized = normalized[-width:]
    elif len(normalized) < width:
        # Pad with the last character (without color tags for padding)
        last_char = normalized[-1] if normalized else "─"
        # Strip color tags for padding
        if "[" in last_char and "]" in last_char:
            # Extract just the character without color tags
            clean_char = re.sub(r'\[.*?\]', '', last_char)
            normalized.extend([clean_char] * (width - len(normalized)))
        else:
            normalized.extend([last_char] * (width - len(normalized)))
    
    chart = "".join(normalized)
    return chart, min_price, max_price

def get_sparkline_data(symbol, timeframe="15m", limit=30):
    """Fetch recent price data for sparkline - cached for 30s (15m candles don't change fast)"""
    def _fetch():
        ohlcv = gate.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if ohlcv:
            prices = [candle[4] for candle in ohlcv]
            timestamps = [candle[0] for candle in ohlcv]
            return prices, timestamps
        return [], []
    
    try:
        return cached_fetch(f'sparkline_{symbol}_{timeframe}', _fetch, ttl_seconds=30)
    except Exception as e:
        return [], []

def fetch_recent_trades(symbol, limit=100):
    """Fetch recent trade history - cached for 30s (historical data)"""
    def _fetch():
        return gate.fetch_my_trades(symbol, limit=limit)
    
    try:
        return cached_fetch(f'recent_trades_{symbol}', _fetch, ttl_seconds=30)
    except Exception as e:
        console.log(f"[red]❌ Failed fetching trades for {symbol}[/]: {e}")
        return []

def map_executions_to_sparkline(trades, timestamps, timeframe_ms=15*60*1000):
    """Map trade executions to sparkline bar indices"""
    execution_data = {}
    
    # Only consider trades from the last 24 hours to match sparkline timeframe
    cutoff_time = timestamps[0] if timestamps else (time.time() * 1000 - 24*60*60*1000)
    
    for trade in trades:
        trade_time = trade['timestamp']
        trade_side = trade['side']  # 'buy' or 'sell'
        
        # Skip trades older than our sparkline data
        if trade_time < cutoff_time:
            continue
            
        # Find which sparkline bar this trade belongs to
        for i, bar_timestamp in enumerate(timestamps):
            # Check if trade falls within this 15-minute bar
            if bar_timestamp <= trade_time < (bar_timestamp + timeframe_ms):
                # If multiple trades in same bar, prioritize sell over buy for visibility
                if i not in execution_data or trade_side == 'sell':
                    execution_data[i] = trade_side
                break
    
    return execution_data


def build_position_panel(visual_symbol, symbol, position, current_price, orders, balance, all_positions=None):
    try_price = fetch_usdt_try_price()
    sell_orders = [o for o in orders if o['side'].lower() == 'sell']
    buy_orders = [o for o in orders if o['side'].lower() == 'buy']

    highest_buy = max((float(b['price']) for b in buy_orders), default=current_price)
    current_sell_price = float(sell_orders[0]['price']) if sell_orders else current_price
    duration_str = format_trade_duration(position)

    all_orders = sorted(buy_orders + sell_orders, key=lambda o: float(o['price']), reverse=True)
    progress_renderables = []
    buys = 0
    sells = len(sell_orders) + 1

    price_str = f"{current_price:.5f}"
    unrealized_pnl = float(position['unrealizedPnl'])
    pnl_color = "green" if unrealized_pnl >= 0 else "red"
    entry_price = float(position.get('entryPrice', 0))
    contracts = float(position.get('contracts', 0)) * float(position.get('contractSize', 0))
    realized_pnl = float(position.get('realizedPnl', 0))
    break_even_price = entry_price - (realized_pnl / contracts) if contracts else None

    # Starting with already filled position quantity and avg price
    position_qty = float(position.get('contracts', 0)) * float(position.get('contractSize', 0))
    entry_price = float(position.get('entryPrice', 0))

    # cumulative cost and qty start from position's filled amount
    cumulative_qty = position_qty
    cumulative_cost = entry_price * position_qty if position_qty else 0.0
    buy_avg_prices = {}

    # Sort buy orders descending price (high to low)
    sorted_buys_desc = sorted(buy_orders, key=lambda o: float(o['price']), reverse=True)

    for order in sorted_buys_desc:
        price = float(order['price'])
        qty = float(order.get('remaining', order.get('amount', 0))) * float(position.get('contractSize', 0))
        cumulative_cost += price * qty
        cumulative_qty += qty
        avg_price = cumulative_cost / cumulative_qty if cumulative_qty > 0 else price
        buy_avg_prices[price] = avg_price

    for order in all_orders:
        price = float(order['price'])
        side = order['side'].capitalize()
        if side == "Sell":
            bar_min = highest_buy
            bar_max = price
            value = current_price
            sells -= 1

            # Add aligned PnL string
            qty = float(order.get('remaining', order.get('amount', 0))) * float(position.get('contractSize', 0))
            if entry_price and qty:
                pnl = (price - entry_price) * qty
                sign = "+" if pnl >= 0 else "-"
                pnl_str = f"({sign}${abs(pnl):10.5f})"
            else:
                pnl_str = " " * 13  # blank if data missing
        else:
            # Buy orders: negative to keep left-to-right progress visually consistent with price drop fill
            bar_min = -1 * current_sell_price
            bar_max = -1 * price
            value = -1 * current_price
            buys += 1

        if len(all_orders) < 2:
            bar_min = -1 * (break_even_price if break_even_price else 0)
            bar_max = -1 * price
            value = -1 * current_price
            buys += 1

        distance = bar_max - bar_min
        progress_value = value - bar_min
        percent = (progress_value / distance) * 100 if distance else 100
        percent = max(0.0, min(100.0, percent))

        color = interpolate_color(percent, side)
        qty = float(order.get('remaining', order.get('amount', 0))) * float(position.get('contractSize', 0))

        # For buy orders, add cumulative average fill price next to the qty@price text
        avg_price_str = ""
        if side == "Buy":
            # Use price key in buy_avg_prices to get cumulative avg for this order
            avg_price = buy_avg_prices.get(price)
            if avg_price is not None:
                avg_price_str = f" (Avg: {avg_price:.5f})"

        progress = Progress(
            TextColumn(
                f"[bold]{f'{side}{sells}' if side == 'Sell' else f'{side}{buys} '}[/] "
                f"{qty:.2f} @ {price:.5f}"
                f"{' ' + pnl_str if side == 'Sell' else avg_price_str}"
            ),
            ColoredBarColumn(bar_color=color),
            TextColumn(f"{percent:>5.2f}%"),
            expand=True
        )

        progress.add_task(
            "", total=100, completed=percent,
            price=price, side=side,
            custom_percent=round(percent, 2)
        )
        progress_renderables.append(progress.get_renderable())

    # Get sparkline data and price changes
    sparkline_prices, sparkline_timestamps = get_sparkline_data(symbol, "15m", 30)
    
    # Fetch recent trades and map to sparkline bars
    execution_data = None
    if sparkline_timestamps:
        recent_trades = fetch_recent_trades(symbol, limit=100)
        execution_data = map_executions_to_sparkline(recent_trades, sparkline_timestamps)
    
    sparkline, low_price, high_price = create_sparkline(sparkline_prices, width=30, execution_data=execution_data)
    price_changes = get_price_changes(symbol)
    
    # Get funding rate info and last funding payment
    funding_rate_str, funding_time_str = get_funding_info(symbol, position)
    last_funding_payment = get_last_funding_payment(symbol)
    
    # Determine sparkline color based on current price position within range
    sparkline_color = "green"
    if low_price != high_price:  # Avoid division by zero
        # Calculate percentile of current price within sparkline range
        price_percentile = (current_price - low_price) / (high_price - low_price)
        price_percentile = max(0.0, min(1.0, price_percentile))  # Clamp to 0-1
        
        # Create smoother color transition with minimum color values to avoid pure colors
        # Red component: high when price is low (add 50 minimum to avoid pure green)
        red_component = int(50 + 205 * (1 - price_percentile))
        # Green component: high when price is high (add 50 minimum to avoid pure red)  
        green_component = int(50 + 205 * price_percentile)
        sparkline_color = f"#{red_component:02x}{green_component:02x}00"

    # Info panel with sparkline
    info = Table.grid(padding=0)
    info.add_row(f"[bold]📈 Symbol:[/] {visual_symbol}")
    info.add_row(f"[bold]🎯 Position:[/] {position['contracts']} contracts")
    info.add_row(f"[bold]💰 Current Price:[/] {price_str}")
    info.add_row(f"[bold]📊 15m Chart:[/] [{sparkline_color}]{sparkline}[/{sparkline_color}] Low: {low_price:.5f} High: {high_price:.5f}")
    info.add_row(f"[bold]📈 Changes:[/] 5m:{price_changes.get('5m', 'N/A')} 15m:{price_changes.get('15m', 'N/A')} 1h:{price_changes.get('1h', 'N/A')} 1d:{price_changes.get('1d', 'N/A')}")
    info.add_row(f"[bold]🧮 Unrealized PnL:[/] [{pnl_color}]{unrealized_pnl:.5f} ({(unrealized_pnl * try_price):.5f}₺)[/{pnl_color}]")
    realized_pnl_color = "green" if realized_pnl >= 0 else "red"
    info.add_row(f"[bold]✅ Realized PnL:[/] [{realized_pnl_color}]{realized_pnl:.5f} ({(realized_pnl * try_price):.5f}₺)[/{realized_pnl_color}]")
    info.add_row(f"[bold]⚖️ Break-Even Price:[/] {break_even_price:.5f}" if break_even_price else "[bold]⚖️ Break-Even Price:[/] —")

    info_panel = Panel(info, title="📋 Position Info", expand=False)

    # Trade Duration Panel
    timer_table = Table.grid(padding=0)
    timer_table.add_row(f"[bold]⏱️ Time in Trade:[/] {duration_str}")
    timer_panel = Panel(timer_table, title="⏱️ Trade Duration", expand=False)

    # Funding Info Panel
    funding_table = Table.grid(padding=0)
    funding_table.add_row(f"[bold]Next Funding:[/] {funding_rate_str} in {funding_time_str}")
    funding_table.add_row(f"[bold]Last Funding:[/] {last_funding_payment}")
    funding_panel = Panel(funding_table, title="💸 Funding", expand=False)

    # Stack timer and funding panels vertically on the right side
    right_side_panels = Group(timer_panel, funding_panel)

    # Combine info panel on left and stacked panels on right
    info_timer_columns = Columns([info_panel, right_side_panels], equal=True, expand=True)

    # Create position group with progress bars and info
    position_content = Group(
        *progress_renderables,
        info_timer_columns
    )

    # Wrap in panel with symbol title
    position_panel = Panel(position_content, title=f"📍 {visual_symbol}", border_style="cyan", expand=True)

    return position_panel

def fetch_unified_account_equity():
    """Fetch total equity from Gate.io unified account - cached for 5s"""
    def _fetch():
        try:
            configuration = Configuration(key=API_KEY, secret=API_SECRET)
            client = ApiClient(configuration)
            
            from gate_api import UnifiedApi
            unified_api = UnifiedApi(client)
            
            account = unified_api.list_unified_accounts()
            
            # Try unified_account_total_equity first
            if hasattr(account, 'unified_account_total_equity'):
                return float(account.unified_account_total_equity)
            elif hasattr(account, 'total_equity'):
                return float(account.total_equity)
            
            # Fallback to USDT balance equity
            if hasattr(account, 'balances') and 'USDT' in account.balances:
                usdt_balance = account.balances['USDT']
                if isinstance(usdt_balance, dict):
                    return float(usdt_balance.get('equity', 0))
            
            return None
        except Exception as e:
            with open('tracker_errors.log', 'a') as f:
                f.write(f"{datetime.now()}: Unified account fetch error: {e}\n")
            return None
    
    return cached_fetch('unified_equity', _fetch, ttl_seconds=5)

def build_account_metrics_panel(balance, all_positions):
    try_price = fetch_usdt_try_price()
    
    # Try to get equity from unified account first (includes borrowed funds calculation)
    equity_usdt = fetch_unified_account_equity()
    
    if equity_usdt is None:
        # Fallback: calculate from futures balance
        balance_usdt = 0.0
        debt_usdt = 0.0
        
        # Check if we have the info array structure (Gate.io unified/swap accounts)
        if 'info' in balance and isinstance(balance['info'], list):
            for item in balance['info']:
                if item.get('currency') == 'USDT':
                    free_usdt = float(item.get('available', 0))
                    order_margin = float(item.get('order_margin', 0))
                    position_initial_margin = float(item.get('position_initial_margin', 0))
                    balance_usdt = free_usdt + order_margin + position_initial_margin
                    break
        else:
            # Fallback to standard CCXT structure
            if 'USDT' in balance.get('total', {}):
                balance_usdt = balance['total']['USDT']
                debt_usdt = balance.get('debt', {}).get('USDT', 0)
        
        # Calculate total unrealized PnL from all active positions
        total_unrealized_pnl = 0.0
        if all_positions:
            for pos in all_positions:
                total_unrealized_pnl += float(pos['unrealizedPnl'])
        
        equity_usdt = balance_usdt - debt_usdt + total_unrealized_pnl
    
    equity_try = equity_usdt * try_price
    profit_try = equity_try - STARTING_BALANCE

    # Create a wider grid layout for better spread
    balance_table = Table.grid(padding=(0, 2))
    balance_table.add_column(justify="left")
    balance_table.add_column(justify="left")
    balance_table.add_column(justify="left")
    
    balance_table.add_row(
        f"[bold]💰 Total Equity:[/] {equity_usdt:.2f} USDT",
        f"[bold]💰 Equity TRY:[/] {equity_try:.2f}₺ (Profit: {profit_try:+.2f}₺)"
    )

    # Add liquidation info if available from any position
    if all_positions:
        liq_info = []
        for pos in all_positions:
            liq_price = pos.get('liquidationPrice')
            leverage = pos.get('leverage')
            margin_ratio = pos.get('marginRatio')
            
            if liq_price:
                liq_info.append(f"🧯 {pos['symbol'].split(':')[0]} Liq: {liq_price:.5f}")
            if leverage:
                liq_info.append(f"🧾 {pos['symbol'].split(':')[0]} Lev: {leverage}")
            if margin_ratio:
                liq_info.append(f"📊 {pos['symbol'].split(':')[0]} Margin: {float(margin_ratio):.2%}")
        
        # Add liquidation info in chunks of 3
        for i in range(0, len(liq_info), 3):
            chunk = liq_info[i:i+3]
            while len(chunk) < 3:
                chunk.append("")  # Fill empty columns
            balance_table.add_row(*chunk)

    return Panel(balance_table, title="💼 Account Metrics", expand=True)

def fetch_recent_closed_positions(limit=48):
    """Fetch recent closed positions - cached for 30s"""
    def _fetch():
        try:
            # Use CCXT's private API method directly
            response = gate.private_futures_get_settle_position_close({
                'settle': 'usdt',
                'limit': limit
            })
            
            # Sort by close time descending (most recent first)
            sorted_positions = sorted(response, key=lambda x: x.get('time', 0), reverse=True)
            return sorted_positions[:limit]
            
        except Exception as e:
            with open('tracker_errors.log', 'a') as f:
                f.write(f"{datetime.now()}: Closed positions fetch error: {e}\n")
            return []
    
    return cached_fetch('recent_closed_positions', _fetch, ttl_seconds=30)

def build_position_history_panel():
    """Build a compact grid showing recent closed positions (8 rows x 6 columns = 48 positions)"""
    closed_positions = fetch_recent_closed_positions(48)
    
    if not closed_positions:
        return Panel("No recent closed positions", title="📜 Recent History", expand=True)
    
    # Find min and max PnL for gradient coloring
    pnls = [float(pos.get('pnl', 0)) for pos in closed_positions]
    min_pnl = min(pnls) if pnls else 0
    max_pnl = max(pnls) if pnls else 0
    pnl_range = max_pnl - min_pnl if max_pnl != min_pnl else 1
    
    # Find min and max PnL/s for independent gradient coloring
    pnl_per_secs = []
    for pos in closed_positions:
        pnl = float(pos.get('pnl', 0))
        close_time = int(pos.get('time', 0))
        open_time = int(pos.get('first_open_time', close_time))
        dur = max(close_time - open_time, 1)
        pnl_per_secs.append(pnl / dur)
    min_pps = min(pnl_per_secs) if pnl_per_secs else 0
    max_pps = max(pnl_per_secs) if pnl_per_secs else 0
    pps_range = max_pps - min_pps if max_pps != min_pps else 1
    
    # Create a grid with 6 columns
    history_table = Table.grid(padding=(0, 1), expand=True)
    for _ in range(6):
        history_table.add_column()
    
    # Prepare all cells first
    cells = []
    for pos in closed_positions:
        symbol = pos.get('contract', '').replace('_USDT', '').replace('_', '')
        pnl = float(pos.get('pnl', 0))
        close_time = int(pos.get('time', 0))
        open_time = int(pos.get('first_open_time', close_time))
        
        # Calculate position duration with detailed time
        duration_seconds = close_time - open_time
        if duration_seconds == 0:
            duration_seconds = 1  # Avoid division by zero
            
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        seconds = duration_seconds % 60
        
        if hours > 0:
            duration = f"{hours}h{minutes}m{seconds}s"
        elif minutes > 0:
            duration = f"{minutes}m{seconds}s"
        else:
            duration = f"{seconds}s"
        
        # Calculate time ago (when it closed) with detailed time
        now = datetime.now().timestamp()
        time_diff = int(now - close_time)
        ago_hours = time_diff // 3600
        ago_minutes = (time_diff % 3600) // 60
        ago_seconds = time_diff % 60
        
        if ago_hours > 0:
            time_ago = f"{ago_hours}h{ago_minutes}m{ago_seconds}s"
        elif ago_minutes > 0:
            time_ago = f"{ago_minutes}m{ago_seconds}s"
        else:
            time_ago = f"{ago_seconds}s"
        
        # Calculate PnL per second
        pnl_per_sec = pnl / duration_seconds
        
        # Calculate gradient color for PnL: yellow (worst) to green (best)
        normalized_pnl = (pnl - min_pnl) / pnl_range if pnl_range > 0 else 0.5
        red = int(255 * (1 - normalized_pnl))
        green = 255
        blue = 0
        pnl_color = f"#{red:02x}{green:02x}{blue:02x}"
        
        # Calculate gradient color for PnL/s: white (worst) to light blue (best)
        # Use logarithmic scale for better distribution
        # White RGB: (255, 255, 255) -> Light Blue RGB: (100, 180, 255)
        
        # Shift values to be positive for log scale
        shifted_pps = pnl_per_sec - min_pps + 1e-10  # Add small epsilon to avoid log(0)
        shifted_max = max_pps - min_pps + 1e-10
        
        # Apply log scale
        if shifted_max > 0 and shifted_pps > 0:
            log_normalized = math.log(shifted_pps) / math.log(shifted_max)
            log_normalized = max(0, min(1, log_normalized))  # Clamp to [0, 1]
        else:
            log_normalized = 0.5
        
        pps_red = int(255 - 155 * log_normalized)  # 255 to 100
        pps_green = int(255 - 75 * log_normalized)  # 255 to 180
        pps_blue = 255  # stays 255
        pps_color = f"#{pps_red:02x}{pps_green:02x}{pps_blue:02x}"
        
        # Format: SYMBOL +/-PNL +/-PNL/S [DURATION] TIME_AGO
        cell = f"[dim]{symbol}[/] [{pnl_color}]{pnl:+.2f}[/{pnl_color}] [{pps_color}]{pnl_per_sec:+.1e}/s[/{pps_color}] [dim]\\[{duration}] {time_ago}[/]"
        cells.append(cell)
    
    # Pad cells to fill the grid
    while len(cells) < 48:
        cells.append("")
    
    # Rearrange: top-to-bottom, then left-to-right (column-major order)
    # We have 8 rows x 6 columns
    num_rows = 8
    num_cols = 6
    
    for row_idx in range(num_rows):
        row = []
        for col_idx in range(num_cols):
            # Column-major indexing: position = col * num_rows + row
            cell_idx = col_idx * num_rows + row_idx
            if cell_idx < len(cells):
                row.append(cells[cell_idx])
            else:
                row.append("")
        history_table.add_row(*row)
    
    return Panel(history_table, title="📜 Recent History (Last 48 Closed)", expand=True)

def build_large_position_history_panel(closed_positions):
    """Build a large grid showing closed positions (50 rows x 6 columns = 300 positions)"""
    
    if not closed_positions:
        return Panel("No recent closed positions", title="📜 Recent History", expand=True)
    
    # Find min and max PnL for gradient coloring
    pnls = [float(pos.get('pnl', 0)) for pos in closed_positions]
    min_pnl = min(pnls) if pnls else 0
    max_pnl = max(pnls) if pnls else 0
    pnl_range = max_pnl - min_pnl if max_pnl != min_pnl else 1
    
    # Find min and max PnL/s for independent gradient coloring
    pnl_per_secs = []
    for pos in closed_positions:
        pnl = float(pos.get('pnl', 0))
        close_time = int(pos.get('time', 0))
        open_time = int(pos.get('first_open_time', close_time))
        dur = max(close_time - open_time, 1)
        pnl_per_secs.append(pnl / dur)
    min_pps = min(pnl_per_secs) if pnl_per_secs else 0
    max_pps = max(pnl_per_secs) if pnl_per_secs else 0
    pps_range = max_pps - min_pps if max_pps != min_pps else 1
    
    # Create a grid with 6 columns
    history_table = Table.grid(padding=(0, 1), expand=True)
    for _ in range(6):
        history_table.add_column()
    
    # Prepare all cells first
    cells = []
    for pos in closed_positions:
        symbol = pos.get('contract', '').replace('_USDT', '').replace('_', '')
        pnl = float(pos.get('pnl', 0))
        close_time = int(pos.get('time', 0))
        open_time = int(pos.get('first_open_time', close_time))
        
        # Calculate position duration with detailed time
        duration_seconds = close_time - open_time
        if duration_seconds == 0:
            duration_seconds = 1
            
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        seconds = duration_seconds % 60
        
        if hours > 0:
            duration = f"{hours}h{minutes}m{seconds}s"
        elif minutes > 0:
            duration = f"{minutes}m{seconds}s"
        else:
            duration = f"{seconds}s"
        
        # Calculate time ago
        now = datetime.now().timestamp()
        time_diff = int(now - close_time)
        ago_hours = time_diff // 3600
        ago_minutes = (time_diff % 3600) // 60
        ago_seconds = time_diff % 60
        
        if ago_hours > 0:
            time_ago = f"{ago_hours}h{ago_minutes}m{ago_seconds}s"
        elif ago_minutes > 0:
            time_ago = f"{ago_minutes}m{ago_seconds}s"
        else:
            time_ago = f"{ago_seconds}s"
        
        # Calculate PnL per second
        pnl_per_sec = pnl / duration_seconds
        
        # Calculate gradient color for PnL: yellow (worst) to green (best)
        normalized_pnl = (pnl - min_pnl) / pnl_range if pnl_range > 0 else 0.5
        red = int(255 * (1 - normalized_pnl))
        green = 255
        blue = 0
        pnl_color = f"#{red:02x}{green:02x}{blue:02x}"
        
        # Calculate gradient color for PnL/s with logarithmic scale
        shifted_pps = pnl_per_sec - min_pps + 1e-10
        shifted_max = max_pps - min_pps + 1e-10
        
        if shifted_max > 0 and shifted_pps > 0:
            log_normalized = math.log(shifted_pps) / math.log(shifted_max)
            log_normalized = max(0, min(1, log_normalized))
        else:
            log_normalized = 0.5
        
        pps_red = int(255 - 155 * log_normalized)
        pps_green = int(255 - 75 * log_normalized)
        pps_blue = 255
        pps_color = f"#{pps_red:02x}{pps_green:02x}{pps_blue:02x}"
        
        # Format: SYMBOL +/-PNL +/-PNL/S [DURATION] TIME_AGO
        cell = f"[dim]{symbol}[/] [{pnl_color}]{pnl:+.2f}[/{pnl_color}] [{pps_color}]{pnl_per_sec:+.1e}/s[/{pps_color}] [dim]\\[{duration}] {time_ago}[/]"
        cells.append(cell)
    
    # Pad cells to fill the grid
    while len(cells) < 300:
        cells.append("")
    
    # Rearrange: top-to-bottom, then left-to-right (column-major order)
    num_rows = 50
    num_cols = 6
    
    for row_idx in range(num_rows):
        row = []
        for col_idx in range(num_cols):
            cell_idx = col_idx * num_rows + row_idx
            if cell_idx < len(cells):
                row.append(cells[cell_idx])
            else:
                row.append("")
        history_table.add_row(*row)
    
    return Panel(history_table, title="📜 Extended History (Last 100 Closed)", expand=True)

def main():
    idle_time = 0
    first_run = True
    previous_positions = {}
    previous_orders = {}
    # Suppress false sell sounds right after a buy fill triggers ladder adjustments
    sell_suppress_until = {}



    with Live(console=console, refresh_per_second=1, screen=False) as live:
        while True:
            try:
                positions = fetch_active_positions()

                if not positions:
                    idle_time += 1
                    idle_seconds = idle_time * 5
                    formatted_time = format_duration(idle_seconds)
                    
                    # Show expanded history when idle
                    balance = fetch_balance_unified()
                    account_panel = build_account_metrics_panel(balance, [])
                    
                    # Fetch 100 positions for 50x6 grid (300 total)
                    def fetch_large_history():
                        try:
                            response = gate.private_futures_get_settle_position_close({
                                'settle': 'usdt',
                                'limit': 100
                            })
                            sorted_positions = sorted(response, key=lambda x: x.get('time', 0), reverse=True)
                            return sorted_positions[:100]
                        except:
                            return []
                    
                    large_history = fetch_large_history()
                    
                    # Build large history panel (100 rows x 6 columns)
                    if large_history:
                        history_panel = build_large_position_history_panel(large_history)
                    else:
                        history_panel = Panel(f"[yellow]No open positions. Waiting... {formatted_time}", title="📭 Idle", border_style="dim")
                    
                    # Create header
                    header = Table.grid(expand=True)
                    header.add_column(justify="center")
                    header.add_row(f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]")
                    
                    layout = Group(header, account_panel, history_panel)
                    live.update(layout)
                    time.sleep(5)
                    continue

                idle_time = 0
                
                # Determine symbol set changes for position start/end
                current_symbols = {p['symbol'] for p in positions}
                previous_symbols = set(previous_positions.keys())
                started_symbols = current_symbols - previous_symbols
                ended_symbols = previous_symbols - current_symbols

                if not first_run:
                    for sym in started_symbols:
                        play_sound("sounds/position_start.wav")
                    for sym in ended_symbols:
                        play_sound("sounds/postion_end.wav")
                        # Cleanup state for ended symbols
                        previous_positions.pop(sym, None)
                        previous_orders.pop(sym, None)
                        sell_suppress_until.pop(sym, None)
                        clear_cache_for_symbol(sym)  # Clear cached data for closed position

                # Process all active positions for order-based sounds
                for pos in positions:
                    pos_symbol = pos['symbol']
                    pos_orders = fetch_orders(pos_symbol)
                    
                    prev_orders = previous_orders.get(pos_symbol, [])

                    if not first_run:
                        # Check for removed buy orders (buy fill)
                        prev_buy_ids = {o["id"] for o in prev_orders if o["side"].lower() == "buy"}
                        new_buy_ids = {o["id"] for o in pos_orders if o["side"].lower() == "buy"}
                        buy_orders_removed = prev_buy_ids - new_buy_ids
                        
                        if buy_orders_removed:
                            play_sound("sounds/buy.wav")
                            # Suppress sell sounds briefly to avoid false positives from ladder adjustments
                            sell_suppress_until[pos_symbol] = time.time() + 3.0

                        # Check for removed sell orders (sell fill)
                        prev_sell_ids = {o["id"] for o in prev_orders if o["side"].lower() == "sell"}
                        new_sell_ids = {o["id"] for o in pos_orders if o["side"].lower() == "sell"}
                        sell_orders_removed = prev_sell_ids - new_sell_ids
                        sell_orders_added = new_sell_ids - prev_sell_ids
 
                        # If a buy order was filled, the bot might adjust the sell ladder.
                        # We should not play a sell sound in that case.
                        suppress_active = time.time() < sell_suppress_until.get(pos_symbol, 0)
                        if not buy_orders_removed and not suppress_active:
                            # A true sell fill is a removed order.
                            # An adjustment (recreate) is a removed and an added order.
                            is_sell_adjustment = sell_orders_removed and sell_orders_added
                            if sell_orders_removed and not is_sell_adjustment:
                                play_sound("sounds/sell.wav")
 
                    # Save state for next loop
                    previous_positions[pos_symbol] = pos
                    previous_orders[pos_symbol] = pos_orders

                # Build position panels
                position_panels = []
                balance = fetch_balance_unified()
                
                for position in positions:
                    symbol = position['symbol']
                    visual_symbol = position['symbol'].split(":")[0]
                    current_price = float(gate.fetch_ticker(symbol)['last'])
                    orders = fetch_orders(symbol)
                    
                    position_panel = build_position_panel(
                        visual_symbol, symbol, position, current_price, orders, balance, positions
                    )
                    position_panels.append(position_panel)

                # Build account metrics panel
                account_panel = build_account_metrics_panel(balance, positions)

                # Build position history panel
                history_panel = build_position_history_panel()

                # Create header with timestamp
                header = Table.grid(expand=True)
                header.add_column(justify="center")
                header.add_row(f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]")

                # Layout - always vertical stacking
                layout_components = [header]
                layout_components.extend(position_panels)
                layout_components.append(account_panel)
                layout_components.append(history_panel)
                
                layout = Group(*layout_components)

                live.update(layout)
                
                first_run = False
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    hide_cursor()
    try:
        # fetch_balance_unified()
        main()
    except KeyboardInterrupt:
        show_cursor()
        print("\nExiting...")
