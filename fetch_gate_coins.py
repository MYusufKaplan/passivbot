"""Temporary script to fetch all tradable USDT futures coins from Gate.io."""

import json
import requests

url = "https://api.gateio.ws/api/v4/futures/usdt/contracts"
resp = requests.get(url)
resp.raise_for_status()

contracts = resp.json()
coins = sorted(set(
    c["name"].replace("_USDT", "")
    for c in contracts
    if not c.get("in_delisting", False)
))

with open("configs/gate_coins.json", "w") as f:
    json.dump(coins, f, indent=4)

print(f"Saved {len(coins)} coins to configs/gate_coins.json")
