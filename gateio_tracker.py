import time
import requests
from datetime import datetime
from gate_api import ApiClient, Configuration, FuturesApi
import json
import os
from datetime import datetime, timezone,timedelta
import pickle
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

STEP = timedelta(days=1)  # or use timedelta(hours=1) for tighter slices

start_time = datetime(2025, 3, 5, tzinfo=timezone.utc)
end_date = datetime(2025, 5, 6, tzinfo=timezone.utc)

all_closed_positions = []
pnls = {}

while start_time < end_date:
    # Slide 1 day forward
    window_end = min(start_time + STEP, end_date)
    from_ts = int(start_time.timestamp())
    to_ts = int(window_end.timestamp())

    print(f"🔄 Fetching positions from {start_time} to {window_end}")

    resp = futures_api.list_position_close(
        settle=SETTLE,
        _from=from_ts,
        to=to_ts,
        limit=1000,
        offset=0
    )

    for position in resp:
        symbol = position.contract
        pnl = float(position.pnl)
        pnls[symbol] = pnls.get(symbol, 0) + pnl
        all_closed_positions.append(position)

    start_time = window_end  # move window forward



# for transaction in all_closed_positions:
#     try:
#         pnls[transaction.contract] += float(transaction.pnl)
#     except:
#         pnls[transaction.contract] = float(transaction.pnl)

with open("closed_positions.json", "w", encoding="utf-8") as f:
    json.dump(pnls, f, indent=4, ensure_ascii=False)
    print("✅📁 JSON file saved: closed_positions.json")
# Print PnLs
print("📈 Closed Positions PnL since April 30, 2025:\n")
for pos in all_closed_positions:
    print(f"🔹 Symbol: {pos.contract}, PnL: {pos.realised_point}, Time: {pos.close_time}")