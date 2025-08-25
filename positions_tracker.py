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
    'options': {'defaultType': 'swap'}
})

console = Console(force_terminal=True, no_color=False, color_system="truecolor", log_path=False, width=191)

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
    exchange = ccxt.binance()
    symbol = 'USDT/TRY'  # Binance has this pair

    try:
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']  # Last traded price
        return price
    except Exception as e:
        print(f"Error fetching {symbol} price: {e}")
        return None

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

def get_price_changes(symbol):
    intervals = [
        "5m",
        "15m",
        "1h",
        "1d"
    ]
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

    # Info panel
    info = Table.grid(padding=0)
    info.add_row(f"[bold]📈 Symbol:[/] {visual_symbol}")
    info.add_row(f"[bold]🎯 Position:[/] {position['contracts']} contracts")
    info.add_row(f"[bold]💰 Current Price:[/] {price_str}")
    info.add_row(f"[bold]🧮 Unrealized PnL:[/] [{pnl_color}]{unrealized_pnl:.5f} ({(unrealized_pnl * try_price):.5f}₺)[/{pnl_color}]")
    info.add_row(f"[bold]✅ Realized PnL:[/] {realized_pnl:.5f} ({(realized_pnl * try_price):.5f}₺)")
    info.add_row(f"[bold]⚖️ Break-Even Price:[/] {break_even_price:.5f}" if break_even_price else "[bold]⚖️ Break-Even Price:[/] —")

    info_panel = Panel(info, title="📋 Position Info", expand=False)

    # Trade Duration Panel
    timer_table = Table.grid(padding=0)
    timer_table.add_row(f"[bold]⏱️ Time in Trade:[/] {duration_str}")
    timer_panel = Panel(timer_table, title="⏱️ Trade Duration", expand=False)

    # Combine info and timer panels side by side
    info_timer_columns = Columns([info_panel, timer_panel], equal=True, expand=True)

    # Create position group with progress bars and info
    position_content = Group(
        *progress_renderables,
        info_timer_columns
    )

    # Wrap in panel with symbol title
    position_panel = Panel(position_content, title=f"📍 {visual_symbol}", border_style="cyan", expand=True)

    return position_panel

def build_account_metrics_panel(balance, all_positions):
    try_price = fetch_usdt_try_price()
    balance_usdt = balance['total']['USDT']
    
    # Calculate total unrealized PnL from all active positions
    total_unrealized_pnl = 0.0
    if all_positions:
        for pos in all_positions:
            total_unrealized_pnl += float(pos['unrealizedPnl'])
    
    equity_usdt = (balance_usdt + total_unrealized_pnl) 
    equity_try = equity_usdt * try_price
    profit_try = equity_try - 209000

    # Create a wider grid layout for better spread
    balance_table = Table.grid(padding=(0, 2))
    balance_table.add_column(justify="left")
    balance_table.add_column(justify="left")
    balance_table.add_column(justify="left")
    
    balance_table.add_row(
        f"[bold]💰 Total Equity:[/] {equity_usdt:.2f} USDT",
        f"[bold]🔓 Free Balance:[/] {balance['free']['USDT']:.2f} USDT",
        f"[bold]📉 Used Margin:[/] {balance['used']['USDT']:.2f} USDT",
        f"[bold]💰 Equity TRY:[/] {equity_try:.2f}₺ (Profit: {profit_try:+.2f}₺)"
    )
    
    # balance_table.add_row(
    #     f"[bold]💰 Equity TRY:[/] {equity_try:.2f}₺ (Profit: {profit_try:+.2f}₺)",
    #     "",  # Empty middle column for this row
    #     ""   # Empty right column for this row
    # )

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

def main():
    idle_time = 0
    first_run = True
    previous_positions = {}
    previous_orders = {}


    with Live(console=console, refresh_per_second=1, screen=False) as live:
        while True:
            try:
                positions = fetch_active_positions()

                if not positions:
                    idle_time += 1
                    idle_seconds = idle_time * 5
                    formatted_time = format_duration(idle_seconds)
                    panel = Panel(f"[yellow]No open positions. Waiting... {formatted_time}", title="📭 Idle", border_style="dim")
                    live.update(panel)
                    time.sleep(5)
                    continue

                idle_time = 0
                
                # Process all positions for sound detection
                for pos in positions:
                    pos_symbol = pos['symbol']
                    pos_orders = fetch_orders(pos_symbol)
                    
                    # Detect state changes — skip sounds on first loop
                    prev_pos = previous_positions.get(pos_symbol)
                    prev_orders = previous_orders.get(pos_symbol, [])

                    if not first_run:
                        # Position start
                        if not prev_pos and pos:
                            play_sound("sounds/position_start.wav")

                        # Position end
                        if prev_pos and not pos:
                            play_sound("sounds/postion_end.wav")

                        # New sell order
                        prev_sell_ids = {o["id"] for o in prev_orders if o["side"].lower() == "sell"}
                        new_sell_ids = {o["id"] for o in pos_orders if o["side"].lower() == "sell"}
                        if new_sell_ids - prev_sell_ids:
                            play_sound("sounds/sell.wav")

                        # Removed buy order
                        prev_buy_ids = {o["id"] for o in prev_orders if o["side"].lower() == "buy"}
                        new_buy_ids = {o["id"] for o in pos_orders if o["side"].lower() == "buy"}
                        if prev_buy_ids - new_buy_ids:
                            play_sound("sounds/buy.wav")

                    # Save state for next loop
                    previous_positions[pos_symbol] = pos
                    previous_orders[pos_symbol] = pos_orders

                # Build position panels
                position_panels = []
                balance = gate.fetch_balance()
                
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

                # Create header with timestamp
                header = Table.grid(expand=True)
                header.add_column(justify="center")
                header.add_row(f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]")

                # Layout based on number of positions
                if len(position_panels) == 1:
                    # Single position - full width
                    layout = Group(
                        header,
                        position_panels[0],
                        account_panel
                    )
                else:
                    # Multiple positions - side by side using table
                    positions_table = Table.grid(padding=1, expand=True)
                    for _ in range(len(position_panels)):
                        positions_table.add_column(ratio=1)
                    positions_table.add_row(*position_panels)
                    
                    layout = Group(
                        header,
                        positions_table,
                        account_panel
                    )

                live.update(layout)
                
                first_run = False
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    hide_cursor()
    try:
        main()
    except KeyboardInterrupt:
        show_cursor()
        print("\nExiting...")
