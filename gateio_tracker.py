import time
import requests
from datetime import datetime
from gate_api import ApiClient, Configuration, FuturesApi
import json
import os

# === CONFIGURATION ===
SETTLE = "usdt"
POLL_INTERVAL = 10  # seconds
STATE_FILE = "state.json"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    else:
        return {"open_positions": {}, "latest_close_time": 0}

def save_state(open_positions, latest_close_time):
    state = {
        "open_positions": open_positions,
        "latest_close_time": latest_close_time
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


# === TELEGRAM CONFIGURATION FROM FILE ===
def load_telegram_keys():
    with open('telegram.key', 'r') as file:
        lines = file.readlines()
        bot_token = lines[0].strip().split('=')[1]
        chat_id = lines[1].strip().split('=')[1]
    return bot_token, chat_id

BOT_TOKEN, CHAT_ID = load_telegram_keys()

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {'chat_id': CHAT_ID, 'text': message}
    response = requests.post(url, data=data)
    if response.status_code != 200:
        print(f"❌ Telegram error: {response.text}")

# === API KEY LOADING ===
def load_api_keys():
    with open('api.key', 'r') as file:
        lines = file.readlines()
        api_key = lines[0].strip().split('=')[1]
        api_secret = lines[1].strip().split('=')[1]
    return api_key, api_secret

# === INITIALIZE CLIENT ===
API_KEY, API_SECRET = load_api_keys()
configuration = Configuration(key=API_KEY, secret=API_SECRET)
client = ApiClient(configuration)
futures_api = FuturesApi(client)

# === TRACKING STATE ===
state = load_state()
open_positions = state["open_positions"]
latest_close_time = state["latest_close_time"]

# === MAIN LOOP ===
print("🚀 Gate.io Futures Position Tracker Started (polling every 10 seconds)\n")

while True:
    try:
        timestamp = int(time.time())

        # === CHECK CURRENT OPEN POSITIONS ===
        positions = futures_api.list_positions(settle=SETTLE)
        current_positions = {}

        for pos in positions:
            symbol = pos.contract
            size = float(pos.size)
            entry_price = float(pos.entry_price)
            leverage = float(pos.leverage)

            current_positions[symbol] = size

            if symbol not in open_positions and size > 0:
                send_telegram_alert(f"📈 Position Opened: {symbol} | Size: {size} | Entry: {entry_price}")

            # elif symbol in open_positions:
            #     prev_size = open_positions[symbol]
            #     if size < prev_size:
            #         send_telegram_alert(f"🔄 Position Partially Closed: {symbol} | New Size: {size} (Prev: {prev_size})")

        # === CHECK FOR CLOSED POSITIONS ===
        closes = futures_api.list_position_close(SETTLE, limit=10)
        closes_sorted = sorted(closes, key=lambda c: c.time, reverse=True)

        for close in closes_sorted:
            if close.time > latest_close_time:
                realized_pnl = float(close.pnl)
                latest_close_time = close.time
                send_telegram_alert(f"📉 Position Closed: {close.contract} | Realized PnL: {realized_pnl:+.2f} USDT")

        # === UPDATE OPEN POSITIONS STATE ===
        open_positions = current_positions

        print(f"[{datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}] ✅ Positions checked.")

        # Update persistent state after processing everything
        save_state(open_positions, latest_close_time)
    except Exception as e:
        print(f"[ERROR] {e}")
    

    time.sleep(POLL_INTERVAL)

