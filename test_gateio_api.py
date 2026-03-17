#!/usr/bin/env python3
"""
Test script for Gate.io API endpoints using CCXT
Run this to debug various CCXT API calls and see what data is available.

Usage: python test_gateio_api.py
"""

import ccxt
from datetime import datetime, timedelta
import json

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

def test_closed_positions():
    print("=" * 80)
    print("Testing CCXT Gate.io Closed Positions")
    print("=" * 80)
    
    # Test 1: Check if fetch_closed_orders exists
    print("\n[TEST 1] Checking available methods...")
    methods = [m for m in dir(gate) if 'close' in m.lower() or 'position' in m.lower() or 'history' in m.lower()]
    print(f"Available methods with 'close', 'position', or 'history':")
    for method in methods:
        print(f"  - {method}")
    
    # Test 2: Try fetch_my_trades (might include closed positions)
    print("\n[TEST 2] Fetching my trades (last 100)...")
    try:
        # Try a common symbol first
        symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        all_trades = []
        
        for symbol in symbols:
            try:
                trades = gate.fetch_my_trades(symbol, limit=50)
                all_trades.extend(trades)
                print(f"  ✓ {symbol}: {len(trades)} trades")
            except Exception as e:
                print(f"  ✗ {symbol}: {e}")
        
        if all_trades:
            print(f"\nTotal trades found: {len(all_trades)}")
            print(f"Sample trade:")
            print(json.dumps(all_trades[0], indent=2, default=str))
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Try fetch_orders with closed status
    print("\n[TEST 3] Fetching closed orders...")
    try:
        # Try fetching closed orders for a symbol
        symbol = 'BTC/USDT:USDT'
        orders = gate.fetch_closed_orders(symbol, limit=10)
        print(f"✓ Found {len(orders)} closed orders for {symbol}")
        if orders:
            print(f"\nSample order:")
            print(json.dumps(orders[0], indent=2, default=str))
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: Try private API call directly
    print("\n[TEST 4] Trying private API for position history...")
    try:
        # Gate.io specific endpoint
        response = gate.private_futures_get_settle_position_close({
            'settle': 'usdt',
            'limit': 32
        })
        print(f"✓ Found {len(response)} closed positions")
        if response:
            print(f"\nSample position:")
            print(json.dumps(response[0], indent=2, default=str))
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 5: Check gate.has capabilities
    print("\n[TEST 5] Checking CCXT capabilities...")
    relevant_caps = {
        'fetchClosedOrders': gate.has.get('fetchClosedOrders', False),
        'fetchMyTrades': gate.has.get('fetchMyTrades', False),
        'fetchOrders': gate.has.get('fetchOrders', False),
        'fetchPositions': gate.has.get('fetchPositions', False),
        'fetchPositionsHistory': gate.has.get('fetchPositionsHistory', False),
    }
    for cap, available in relevant_caps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {cap}: {available}")
    
    # Test 6: Try fetching all positions (including closed)
    print("\n[TEST 6] Fetching all positions...")
    try:
        positions = gate.fetch_positions()
        print(f"✓ Found {len(positions)} positions")
        closed = [p for p in positions if float(p.get('contracts', 0)) == 0]
        print(f"  Active: {len(positions) - len(closed)}")
        print(f"  Closed: {len(closed)}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_closed_positions()
