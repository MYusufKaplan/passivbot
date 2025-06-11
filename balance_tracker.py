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

def fetch_active_pnl(exchange):
    positions = exchange.fetch_positions()
    for pos in positions:
        if float(pos['contracts']) > 0:
            return pos["realizedPnl"] + pos["unrealizedPnl"]
    return 0

# === GET FUTURES BALANCE USING CCXT ===
def fetch_balance_and_equity(exchange):
    try:
        balance = exchange.fetch_balance(params={"type": "swap"})["USDT"]["total"]
        equity = balance + fetch_active_pnl(exchange)
        return balance, equity  # balance = equity fallback
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
