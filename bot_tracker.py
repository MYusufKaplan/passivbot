import ccxt
import time
import sys
import requests
import hmac
import hashlib
from rich.live import Live
from rich.table import Table
from rich.console import Console, Group
from rich.panel import Panel
from datetime import datetime, timedelta

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
        'unified': True
    }
})

console = Console(force_terminal=True, no_color=False, color_system="truecolor", log_path=False, width=255)

def hide_cursor():
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

def show_cursor():
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()

def fetch_usdt_try_price():
    exchange = ccxt.binance()
    symbol = 'USDT/TRY'
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        return None

def get_bot_account_balance():
    """Fetch the bot trading account balance (quant account) from Gate.io API"""
    try:
        total_balance = gate.privateWalletGetTotalBalance()
        
        if 'details' in total_balance and 'quant' in total_balance['details']:
            quant_balance = total_balance['details']['quant']
            if quant_balance.get('currency') == 'USDT':
                return float(quant_balance.get('amount', 0))
        
        return 0.0
    except Exception as e:
        console.log(f"[dim red]Could not fetch bot account balance: {e}[/]")
        return 0.0

def gen_sign(method, url, query_string, payload_string, timestamp):
    """Generate Gate.io API signature"""
    m = hashlib.sha512()
    m.update((payload_string or "").encode('utf-8'))
    hashed_payload = m.hexdigest()
    s = '%s\n%s\n%s\n%s\n%s' % (method, url, query_string or "", hashed_payload, timestamp)
    sign = hmac.new(API_SECRET.encode('utf-8'), s.encode('utf-8'), hashlib.sha512).hexdigest()
    return sign

def fetch_futures_dca_bots_direct():
    """
    Fetch futures DCA bots using direct API calls.
    
    NOTE: Your API key needs "Earn" permission enabled in Gate.io API settings.
    Go to: Gate.io → API Management → Edit API → Enable "Earn" permission
    """
    base_url = "https://api.gateio.ws"
    
    # Try multiple potential endpoints for DCA/strategy bots
    endpoints_to_try = [
        "/api/v4/earn/uni/lends",  # Unified earn products
        "/api/v4/futures/usdt/plan_orders",  # Plan orders (might include DCA)
        "/api/v4/futures/usdt/price_orders",  # Price-triggered orders
    ]
    
    for endpoint in endpoints_to_try:
        try:
            timestamp = str(int(time.time()))
            sign = gen_sign('GET', endpoint, '', '', timestamp)
            
            headers = {
                'KEY': API_KEY,
                'Timestamp': timestamp,
                'SIGN': sign
            }
            
            response = requests.get(base_url + endpoint, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                console.log(f"[green]✓ Found data at {endpoint}[/]")
                return parse_bot_data(data, endpoint)
            elif response.status_code == 403:
                error_msg = response.json().get('message', '')
                if 'earn permission' in error_msg.lower():
                    console.log(f"[yellow]⚠ API key needs 'Earn' permission enabled[/]")
                    console.log(f"[yellow]  Go to: Gate.io → API Management → Edit API → Enable 'Earn'[/]")
                    return None
        except Exception as e:
            continue
    
    console.log(f"[red]❌ Could not find DCA bot endpoint[/]")
    console.log(f"[yellow]💡 To find the correct endpoint:[/]")
    console.log(f"[yellow]   1. Open Gate.io in browser[/]")
    console.log(f"[yellow]   2. Go to your DCA bots page[/]")
    console.log(f"[yellow]   3. Open DevTools (F12) → Network tab[/]")
    console.log(f"[yellow]   4. Refresh page and look for API calls[/]")
    console.log(f"[yellow]   5. Find the endpoint that returns bot data[/]")
    console.log(f"[yellow]   6. Update this script with the correct endpoint[/]")
    return None

def parse_bot_data(data, endpoint):
    """Parse bot data from API response"""
    bots = []
    
    # This will need to be adjusted based on actual API response structure
    # For now, return empty list
    console.log(f"[cyan]Raw data structure: {list(data.keys()) if isinstance(data, dict) else type(data)}[/]")
    
    return bots

def fetch_futures_dca_bots():
    """
    Fetch futures positions as a fallback to show DCA bot activity.
    """
    try:
        # First try direct API
        bots = fetch_futures_dca_bots_direct()
        if bots is not None:
            return bots
        
        # Fallback: Use positions
        console.log(f"[yellow]Using positions as fallback...[/]")
        positions = gate.fetch_positions()
        
        bots = []
        for pos in positions:
            if float(pos.get('contracts', 0)) == 0:
                continue
            
            symbol = pos['symbol']
            contracts = float(pos.get('contracts', 0))
            contract_size = float(pos.get('contractSize', 1))
            position_size = contracts * contract_size
            entry_price = float(pos.get('entryPrice', 0))
            
            try:
                ticker = gate.fetch_ticker(symbol)
                current_price = float(ticker['last'])
            except:
                current_price = entry_price
            
            unrealized_pnl = float(pos.get('unrealizedPnl', 0))
            realized_pnl = float(pos.get('realizedPnl', 0))
            total_profit = unrealized_pnl + realized_pnl
            
            investment = position_size * entry_price
            roi = (total_profit / investment * 100) if investment > 0 else 0
            
            created_time = None
            run_duration = None
            if pos.get('timestamp'):
                created_time = datetime.fromtimestamp(pos['timestamp'] / 1000)
                run_duration = datetime.now() - created_time
            
            bot_info = {
                'id': pos.get('id', 'N/A'),
                'symbol': symbol.split(':')[0],
                'status': 'running',
                'created_time': created_time,
                'investment': investment,
                'profit': unrealized_pnl,
                'total_profit': total_profit,
                'roi': roi,
                'position_size': position_size,
                'avg_entry_price': entry_price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': realized_pnl,
                'run_duration': run_duration,
                'leverage': pos.get('leverage', 1),
                'side': pos.get('side', 'unknown'),
                # DCA-specific fields (not available from positions)
                'dca_orders_filled': '?',
                'dca_orders_total': '?',
                'take_profit_price': '?',
                'cycles_completed': '?',
            }
            
            bots.append(bot_info)
        
        return bots
    except Exception as e:
        console.log(f"[dim red]Could not fetch futures positions: {e}[/]")
        import traceback
        console.log(f"[dim red]{traceback.format_exc()}[/]")
        return []

def format_duration(td):
    """Format timedelta to human readable string"""
    if not td:
        return "—"
    
    total_seconds = int(td.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"

def build_dca_bots_panel(bots, try_price):
    """Build a panel showing all futures DCA bots"""
    if not bots:
        return Panel("[yellow]No active DCA bots found", title="🤖 Futures DCA Bots")
    
    bot_table = Table(show_header=True, header_style="bold cyan", expand=True)
    bot_table.add_column("Symbol", style="yellow", no_wrap=True)
    bot_table.add_column("Side", justify="center")
    bot_table.add_column("Position", justify="right")
    bot_table.add_column("Leverage", justify="center")
    bot_table.add_column("Entry", justify="right")
    bot_table.add_column("Current", justify="right")
    bot_table.add_column("TP Price", justify="right")
    bot_table.add_column("DCA Orders", justify="center")
    bot_table.add_column("Cycles", justify="center")
    bot_table.add_column("Unrealized", justify="right")
    bot_table.add_column("Realized", justify="right")
    bot_table.add_column("Total", justify="right")
    bot_table.add_column("ROI %", justify="right")
    bot_table.add_column("Runtime", justify="right")
    
    total_investment = 0.0
    total_unrealized = 0.0
    total_realized = 0.0
    total_profit = 0.0
    
    for bot in bots:
        symbol = bot['symbol']
        side = bot['side']
        side_emoji = "🟢" if side == "long" else "🔴" if side == "short" else "⚪"
        position = bot['position_size']
        leverage = bot['leverage']
        entry_price = bot['avg_entry_price']
        current_price = bot['current_price']
        tp_price = bot.get('take_profit_price', '?')
        dca_orders = f"{bot.get('dca_orders_filled', '?')}/{bot.get('dca_orders_total', '?')}"
        cycles = bot.get('cycles_completed', '?')
        unrealized_pnl = bot['unrealized_pnl']
        realized_pnl = bot['realized_pnl']
        total_pnl = bot['total_profit']
        roi = bot['roi']
        runtime = format_duration(bot['run_duration'])
        
        total_investment += bot['investment']
        total_unrealized += unrealized_pnl
        total_realized += realized_pnl
        total_profit += total_pnl
        
        unrealized_color = "green" if unrealized_pnl >= 0 else "red"
        realized_color = "green" if realized_pnl >= 0 else "red"
        profit_color = "green" if total_pnl >= 0 else "red"
        roi_color = "green" if roi >= 0 else "red"
        
        bot_table.add_row(
            symbol,
            f"{side_emoji} {side}",
            f"{position:.4f}",
            f"{leverage}x",
            f"{entry_price:.5f}",
            f"{current_price:.5f}",
            str(tp_price) if tp_price != '?' else '?',
            dca_orders,
            str(cycles),
            f"[{unrealized_color}]{unrealized_pnl:+.5f}[/{unrealized_color}]",
            f"[{realized_color}]{realized_pnl:+.5f}[/{realized_color}]",
            f"[{profit_color}]{total_pnl:+.5f}[/{profit_color}]",
            f"[{roi_color}]{roi:+.2f}%[/{roi_color}]",
            runtime
        )
    
    # Add summary row
    total_roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
    unrealized_color = "green" if total_unrealized >= 0 else "red"
    realized_color = "green" if total_realized >= 0 else "red"
    profit_color = "green" if total_profit >= 0 else "red"
    roi_color = "green" if total_roi >= 0 else "red"
    
    bot_table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        f"[bold {unrealized_color}]{total_unrealized:+.5f}[/bold {unrealized_color}]",
        f"[bold {realized_color}]{total_realized:+.5f}[/bold {realized_color}]",
        f"[bold {profit_color}]{total_profit:+.5f}[/bold {profit_color}]",
        f"[bold {roi_color}]{total_roi:+.2f}%[/bold {roi_color}]",
        ""
    )
    
    return Panel(bot_table, title=f"🤖 Futures DCA Bots ({len(bots)})", border_style="cyan", expand=True)

def build_bot_summary_panel(bot_balance, bots, try_price):
    """Build a panel showing bot account summary"""
    total_investment = sum(bot['investment'] for bot in bots)
    total_unrealized = sum(bot['unrealized_pnl'] for bot in bots)
    total_realized = sum(bot['realized_pnl'] for bot in bots)
    total_profit = sum(bot['total_profit'] for bot in bots)
    total_profit_try = total_profit * try_price
    total_roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
    
    available_balance = bot_balance - total_investment
    total_value = bot_balance + total_profit
    
    unrealized_color = "green" if total_unrealized >= 0 else "red"
    realized_color = "green" if total_realized >= 0 else "red"
    profit_color = "green" if total_profit >= 0 else "red"
    roi_color = "green" if total_roi >= 0 else "red"
    
    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_column(justify="left")
    summary_table.add_column(justify="left")
    summary_table.add_column(justify="left")
    
    summary_table.add_row(
        f"[bold]💰 Bot Balance:[/] {bot_balance:.2f} USDT ({bot_balance * try_price:.2f}₺)",
        f"[bold]💼 Total Investment:[/] {total_investment:.2f} USDT",
        f"[bold]💵 Available:[/] {available_balance:.2f} USDT"
    )
    summary_table.add_row(
        f"[bold]📊 Unrealized PnL:[/] [{unrealized_color}]{total_unrealized:+.5f} USDT ({total_unrealized * try_price:+.2f}₺)[/{unrealized_color}]",
        f"[bold]✅ Realized PnL:[/] [{realized_color}]{total_realized:+.5f} USDT ({total_realized * try_price:+.2f}₺)[/{realized_color}]",
        f"[bold]💎 Total Profit:[/] [{profit_color}]{total_profit:+.5f} USDT ({total_profit_try:+.2f}₺)[/{profit_color}]"
    )
    summary_table.add_row(
        f"[bold]📈 Total ROI:[/] [{roi_color}]{total_roi:+.2f}%[/{roi_color}]",
        f"[bold]💎 Total Value:[/] {total_value:.2f} USDT ({total_value * try_price:.2f}₺)",
        f"[bold]🤖 Active Bots:[/] {len(bots)}"
    )
    
    return Panel(summary_table, title="🤖 DCA Bot Account Summary", border_style="cyan", expand=True)

def main():
    with Live(console=console, refresh_per_second=0.2, screen=False) as live:
        while True:
            try:
                # Fetch bot balance
                bot_balance = get_bot_account_balance()
                
                # Fetch TRY price
                try_price = fetch_usdt_try_price() or 1.0
                
                # Fetch all active DCA bots
                bots = fetch_futures_dca_bots()
                
                if not bots:
                    idle_panel = Panel(
                        "[yellow]No active DCA bots found. Waiting...\n\n"
                        "[dim]Note: Enable 'Earn' permission in your Gate.io API settings to access bot data.[/]",
                        title="🤖 DCA Bot Tracker",
                        border_style="yellow"
                    )
                    live.update(idle_panel)
                    time.sleep(5)
                    continue
                
                # Build panels
                summary_panel = build_bot_summary_panel(bot_balance, bots, try_price)
                dca_bots_panel = build_dca_bots_panel(bots, try_price)
                
                # Create header with timestamp
                header = Table.grid(expand=True)
                header.add_column(justify="center")
                header.add_row(f"[bold cyan]🤖 Gate.io Futures DCA Bot Tracker[/bold cyan]")
                header.add_row(f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]")
                
                # Layout
                layout = Group(
                    header,
                    summary_panel,
                    dca_bots_panel
                )
                
                live.update(layout)
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                error_panel = Panel(f"[red]Error: {e}", title="❌ Error", border_style="red")
                live.update(error_panel)
                time.sleep(5)

if __name__ == "__main__":
    hide_cursor()
    try:
        main()
    except KeyboardInterrupt:
        show_cursor()
        print("\nExiting...")
