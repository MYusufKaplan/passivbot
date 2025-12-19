#!/usr/bin/env python3

import numpy as np
import time
from src.backtest import run_backtest

def create_dummy_data():
    """Create realistic dummy data for profiling"""
    n_timesteps = 10000  # ~7 days of 1min data
    n_coins = 20
    n_features = 6
    
    # Create realistic OHLCV data
    hlcvs = np.zeros((n_timesteps, n_coins, n_features))
    
    for coin_idx in range(n_coins):
        base_price = 100.0 + coin_idx * 50.0
        price = base_price
        
        for t in range(n_timesteps):
            # Random walk with some volatility
            change = np.random.normal(0, 0.002) * price
            price = max(price + change, 0.01)
            
            # OHLC with realistic spreads
            high = price * (1 + abs(np.random.normal(0, 0.001)))
            low = price * (1 - abs(np.random.normal(0, 0.001)))
            close = price
            volume = np.random.uniform(1000, 10000)
            
            hlcvs[t, coin_idx, 0] = high
            hlcvs[t, coin_idx, 1] = low
            hlcvs[t, coin_idx, 2] = close
            hlcvs[t, coin_idx, 3] = price  # open
            hlcvs[t, coin_idx, 4] = volume
            hlcvs[t, coin_idx, 5] = volume
    
    # BTC prices
    btc_prices = np.array([50000.0 + i * 0.1 for i in range(n_timesteps)])
    
    return hlcvs, btc_prices

def create_bot_params():
    """Create realistic bot parameters"""
    return {
        "long": {
            "enabled": True,
            "n_positions": 10,
            "total_wallet_exposure_limit": 1.0,
            "unstuck_loss_allowance_pct": 0.1,
            "unstuck_close_pct": 0.01,
            "auto_unstuck_delay_minutes": 60,
            "auto_unstuck_ema_dist": 0.1,
            "auto_unstuck_wallet_exposure_threshold": 0.1,
            "ema_span_0": 1000,
            "ema_span_1": 2000,
            "ema_span_2": 4000,
            "initial_qty_pct": 0.01,
            "initial_eprice_ema_dist": -0.01,
            "markup_range": 0.02,
            "min_markup": 0.005,
            "n_close_orders": 5,
            "rentry_pprice_dist": 0.02,
            "rentry_pprice_dist_wallet_exposure_weighting": 0.1,
            "wallet_exposure_limit": 0.1
        },
        "short": {
            "enabled": False,
            "n_positions": 0,
            "total_wallet_exposure_limit": 0.0,
            "unstuck_loss_allowance_pct": 0.0,
            "unstuck_close_pct": 0.0,
            "auto_unstuck_delay_minutes": 0,
            "auto_unstuck_ema_dist": 0.0,
            "auto_unstuck_wallet_exposure_threshold": 0.0,
            "ema_span_0": 1000,
            "ema_span_1": 2000,
            "ema_span_2": 4000,
            "initial_qty_pct": 0.0,
            "initial_eprice_ema_dist": 0.0,
            "markup_range": 0.0,
            "min_markup": 0.0,
            "n_close_orders": 0,
            "rentry_pprice_dist": 0.0,
            "rentry_pprice_dist_wallet_exposure_weighting": 0.0,
            "wallet_exposure_limit": 0.0
        }
    }

def create_exchange_params(n_coins):
    """Create exchange parameters"""
    return [{"c_mult": 1.0} for _ in range(n_coins)]

def create_backtest_params(n_coins):
    """Create backtest parameters"""
    return {
        "starting_balance": 10000.0,
        "maker_fee": -0.0002,
        "coins": [f"COIN{i}USDT" for i in range(n_coins)],
        "enable_inactivity_bankruptcy": True,
        "max_days_without_position": 7.0,
        "max_days_with_stale_position": 14.0
    }

def main():
    print("🚀 Creating dummy data for profiling...")
    hlcvs, btc_prices = create_dummy_data()
    n_coins = hlcvs.shape[1]
    
    bot_params = create_bot_params()
    exchange_params = create_exchange_params(n_coins)
    backtest_params = create_backtest_params(n_coins)
    
    print(f"📊 Running backtest with {hlcvs.shape[0]} timesteps, {n_coins} coins")
    print("⏱️  Starting profiled backtest...")
    
    start_time = time.time()
    
    try:
        results = run_backtest(
            hlcvs,
            btc_prices,
            bot_params,
            exchange_params,
            backtest_params
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Backtest completed in {total_time:.2f} seconds")
        print(f"📈 Generated {len(results[0])} fills")
        
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
