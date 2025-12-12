import ccxt
import csv
import os
import time
from datetime import datetime, timezone, timedelta

CSV_PATH = "/home/ubuntu/passivbot/balance.csv"
# CSV_PATH = "/home/myusuf/Projects/passivbot/balance.csv"
TARGET_TRY = 209000
SLEEP_SECONDS = 10

# === LOAD API KEYS ===
def load_api_keys():
    with open("api.key", "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip().split("=")[1]
        api_secret = lines[1].strip().split("=")[1]
    return api_key, api_secret

def fetch_balance_unified(exchange):
    """Fetch balance with unified account support"""
    try:
        # Try unified account balance first
        balance = exchange.fetch_balance(params={'type': 'unified'})
        return balance
    except Exception as e:
        # Fallback to regular futures balance
        try:
            balance = exchange.fetch_balance(params={'type': 'swap'})
            return balance
        except Exception as e2:
            # Final fallback to default balance
            return exchange.fetch_balance()

def extract_balance_from_unified(balance):
    """Extract USDT balance from Gate.io unified account structure"""
    balance_usdt = 0.0
    
    # Handle Gate.io unified/swap account balance structure
    if 'info' in balance and isinstance(balance['info'], list):
        for item in balance['info']:
            if item.get('currency') == 'USDT':
                # For Gate.io, available is the free balance
                free_usdt = float(item.get('available', 0))
                # Position margin + order margin = used balance
                position_margin = float(item.get('position_margin', 0))
                order_margin = float(item.get('order_margin', 0))
                position_initial_margin = float(item.get('position_initial_margin', 0))
                used_usdt = position_margin + order_margin + position_initial_margin
                # Total = available + used
                balance_usdt = free_usdt + used_usdt
                break
    else:
        # Fallback to standard CCXT structure
        if 'USDT' in balance.get('total', {}):
            balance_usdt = balance['total']['USDT'] - balance['debt']['USDT']
    
    # If still 0, try the free balance from CCXT structure (Gate.io unified accounts often have total=0)
    if balance_usdt == 0.0 and 'USDT' in balance.get('free', {}):
        balance_usdt = balance['free']['USDT']
    
    return balance_usdt

def fetch_active_pnl(exchange):
    positions = exchange.fetch_positions()
    total_pnl = 0.0
    for pos in positions:
        if float(pos['contracts']) > 0:
            total_pnl += float(pos.get("realizedPnl", 0)) + float(pos.get("unrealizedPnl", 0))
    return total_pnl

# === GET FUTURES BALANCE USING CCXT ===
def fetch_balance_and_equity(exchange):
    try:
        balance_response = fetch_balance_unified(exchange)
        balance = extract_balance_from_unified(balance_response)
        equity = balance + fetch_active_pnl(exchange)
        return balance, equity
    except Exception as e:
        print(f"❌ Error fetching futures balance: {e}")
        return None, None

# === GET SPOT USDT/TRY PRICE ===
def get_usdt_try(exchange):
    try:
        ticker = exchange.fetch_ticker("USDT/TRY")
        return float(ticker["last"])
    except Exception as e:
        print(f"❌ Error fetching USDT/TRY price: {e}")
        return None

# === WRITE CSV HEADER IF FILE DOESN'T EXIST ===
def ensure_csv_header():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "balance", "equity", "tr_balance", "tr_equity", "profit"])

# === MAIN LOOP ===
def log_loop():
    api_key, api_secret = load_api_keys()
    exchange = ccxt.gateio({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",  # USDT-M futures
            "unified": True  # Enable unified account mode
        },
    })
    binance = ccxt.binance()


    ensure_csv_header()

    while True:
        try:
            balance, equity = fetch_balance_and_equity(exchange)
            if balance is None:
                time.sleep(SLEEP_SECONDS)
                continue

            rate = get_usdt_try(binance)
            if rate is None:
                time.sleep(SLEEP_SECONDS)
                continue

            tr_balance = balance * rate
            tr_equity = equity * rate
            profit = tr_equity - TARGET_TRY

            tz = timezone(timedelta(hours=3))
            timestamp = datetime.now(tz).timestamp()

            row = [timestamp, balance, equity, tr_balance, tr_equity, profit]

            with open(CSV_PATH, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(f"✅ {datetime.fromtimestamp(timestamp)} | ₺{tr_equity:.2f} | Profit: ₺{profit:.2f}")

        except Exception as e:
            print(f"❌ Error in main loop: {e}")

        time.sleep(SLEEP_SECONDS)

# === RUN ===
if __name__ == "__main__":
    log_loop()

