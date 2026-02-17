import datetime
import pprint
import json
import re
from collections import OrderedDict
from hashlib import sha256

import numpy as np


__all__ = [
    "safe_filename",
    "numpyize",
    "denumpyize",
    "ts_to_date",
    "config_pretty_str",
    "sort_dict_keys",
    "filter_orders",
    "flatten",
    "floatify",
    "shorten_custom_id",
    "determine_pos_side_ccxt",
    "calc_hash",
    "ensure_millis",
    "multi_replace",
    "str2bool",
    "determine_side_from_order_tuple",
    "remove_OD",
    "log_dict_changes",
]


def safe_filename(symbol: str) -> str:
    """Convert a symbol to a filesystem-safe string."""
    return re.sub(r'[<>:"/\|?*]', "_", symbol)


def numpyize(x):
    if isinstance(x, (list, tuple)):
        return np.array([numpyize(e) for e in x])
    if isinstance(x, dict):
        return {k: numpyize(v) for k, v in x.items()}
    return x


def denumpyize(x):
    if isinstance(x, (np.float64, np.float32, np.float16)):
        return float(x)
    if isinstance(x, (np.int64, np.int32, np.int16, np.int8)):
        return int(x)
    if isinstance(x, np.ndarray):
        return [denumpyize(e) for e in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (dict, OrderedDict)):
        return {k: denumpyize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [denumpyize(z) for z in x]
    if isinstance(x, tuple):
        return tuple(denumpyize(z) for z in x)
    return x


def ts_to_date(timestamp: float) -> str:
    if timestamp > 253402297199:
        dt = datetime.datetime.utcfromtimestamp(timestamp / 1000)
    else:
        dt = datetime.datetime.utcfromtimestamp(timestamp)
    return dt.isoformat().replace(" ", "T")


def config_pretty_str(config: dict) -> str:
    pretty = pprint.pformat(config)
    for before, after in [("'", '"'), ("True", "true"), ("False", "false"), ("None", "null")]:
        pretty = pretty.replace(before, after)
    return pretty


def sort_dict_keys(d):
    if isinstance(d, list):
        return [sort_dict_keys(e) for e in d]
    if not isinstance(d, dict):
        return d
    return {key: sort_dict_keys(d[key]) for key in sorted(d)}


def filter_orders(
    actual_orders,
    ideal_orders,
    keys=("symbol", "side", "qty", "price"),
):
    """Return orders to cancel and to create by comparing actual vs ideal."""

    if not actual_orders:
        return [], ideal_orders
    if not ideal_orders:
        return actual_orders, []

    actual_orders = actual_orders.copy()
    orders_to_create = []
    ideal_cropped = [{k: o[k] for k in keys} for o in ideal_orders]
    actual_cropped = [{k: o[k] for k in keys} for o in actual_orders]

    for cropped, original in zip(ideal_cropped, ideal_orders):
        matches = [(a_c, a_o) for a_c, a_o in zip(actual_cropped, actual_orders) if a_c == cropped]
        if matches:
            actual_orders.remove(matches[0][1])
            actual_cropped.remove(matches[0][0])
        else:
            orders_to_create.append(original)
    return actual_orders, orders_to_create


<<<<<<< HEAD
def get_dummy_settings(config: dict):
    dummy_settings = get_template_live_config()
    dummy_settings.update({k: 1.0 for k in get_xk_keys()})
    dummy_settings.update(
        {
            "user": config["user"],
            "exchange": config["exchange"],
            "symbol": config["symbol"],
            "config_name": "",
            "logging_level": 0,
        }
    )
    return {**config, **dummy_settings}


def flatten(lst: list) -> list:
    return [y for x in lst for y in x]


def get_template_live_config(passivbot_mode="v7"):
    if passivbot_mode == "v7":
        return {
            "backtest": {
                "base_dir": "backtests",
                "combine_ohlcvs": True,
                "compress_cache": True,
                "end_date": "now",
                "exchanges": ["binance", "bybit", "gateio", "bitget"],
                "gap_tolerance_ohlcvs_minutes": 120.0,
                "start_date": "2021-04-01",
                "starting_balance": 100000.0,
                "use_btc_collateral": False,
            },
            "bot": {
                "long": {
                    "close_grid_markup_range": 0.0255,
                    "close_grid_min_markup": 0.0089,
                    "close_grid_qty_pct": 0.125,
                    "close_trailing_grid_ratio": 0.5,
                    "close_trailing_qty_pct": 0.125,
                    "close_trailing_retracement_pct": 0.002,
                    "close_trailing_threshold_pct": 0.008,
                    "ema_span_0": 1318.0,
                    "ema_span_1": 1435.0,
                    "enforce_exposure_limit": True,
                    "entry_grid_double_down_factor": 0.894,
                    "entry_grid_spacing_pct": 0.04,
                    "entry_grid_spacing_weight": 0.697,
                    "entry_initial_ema_dist": -0.00738,
                    "entry_initial_qty_pct": 0.00592,
                    "entry_trailing_grid_ratio": 0.5,
                    "entry_trailing_retracement_pct": 0.01,
                    "entry_trailing_threshold_pct": 0.05,
                    "filter_rolling_window": 60,
                    "filter_relative_volume_clip_pct": 0.95,
                    "n_positions": 10.0,
                    "total_wallet_exposure_limit": 1.7,
                    "unstuck_close_pct": 0.001,
                    "unstuck_ema_dist": 0.0,
                    "unstuck_loss_allowance_pct": 0.03,
                    "unstuck_threshold": 0.916,
                },
                "short": {
                    "close_grid_markup_range": 0.0255,
                    "close_grid_min_markup": 0.0089,
                    "close_grid_qty_pct": 0.125,
                    "close_trailing_grid_ratio": 0.5,
                    "close_trailing_qty_pct": 0.125,
                    "close_trailing_retracement_pct": 0.002,
                    "close_trailing_threshold_pct": 0.008,
                    "ema_span_0": 1318.0,
                    "ema_span_1": 1435.0,
                    "enforce_exposure_limit": True,
                    "entry_grid_double_down_factor": 0.894,
                    "entry_grid_spacing_pct": 0.04,
                    "entry_grid_spacing_weight": 0.697,
                    "entry_initial_ema_dist": -0.00738,
                    "entry_initial_qty_pct": 0.00592,
                    "entry_trailing_grid_ratio": 0.5,
                    "entry_trailing_retracement_pct": 0.01,
                    "entry_trailing_threshold_pct": 0.05,
                    "filter_rolling_window": 60,
                    "filter_relative_volume_clip_pct": 0.95,
                    "n_positions": 10.0,
                    "total_wallet_exposure_limit": 1.7,
                    "unstuck_close_pct": 0.001,
                    "unstuck_ema_dist": 0.0,
                    "unstuck_loss_allowance_pct": 0.03,
                    "unstuck_threshold": 0.916,
                },
            },
            "live": {
                "approved_coins": [],
                "auto_gs": True,
                "coin_flags": {},
                "empty_means_all_approved": False,
                "execution_delay_seconds": 2.0,
                "filter_by_min_effective_cost": True,
                "forced_mode_long": "",
                "forced_mode_short": "",
                "ignored_coins": [],
                "leverage": 10.0,
                "market_orders_allowed": True,
                "max_n_cancellations_per_batch": 5,
                "max_n_creations_per_batch": 3,
                "max_n_restarts_per_day": 10,
                "minimum_coin_age_days": 7.0,
                "ohlcvs_1m_rolling_window_days": 7.0,
                "ohlcvs_1m_update_after_minutes": 10.0,
                "pnls_max_lookback_days": 30.0,
                "price_distance_threshold": 0.002,
                "time_in_force": "good_till_cancelled",
                "user": "bybit_01",
            },
            "optimize": {
                "bounds": {
                    "long_close_grid_markup_range": [0.0, 0.03],
                    "long_close_grid_min_markup": [0.001, 0.03],
                    "long_close_grid_qty_pct": [0.05, 1.0],
                    "long_close_trailing_grid_ratio": [-1.0, 1.0],
                    "long_close_trailing_qty_pct": [0.05, 1.0],
                    "long_close_trailing_retracement_pct": [0.0, 0.1],
                    "long_close_trailing_threshold_pct": [-0.1, 0.1],
                    "long_ema_span_0": [200.0, 1440.0],
                    "long_ema_span_1": [200.0, 1440.0],
                    "long_entry_grid_double_down_factor": [0.1, 3.0],
                    "long_entry_grid_spacing_pct": [0.005, 0.12],
                    "long_entry_grid_spacing_weight": [0.0, 2.0],
                    "long_entry_initial_ema_dist": [-0.1, 0.002],
                    "long_entry_initial_qty_pct": [0.005, 0.1],
                    "long_entry_trailing_grid_ratio": [-1.0, 1.0],
                    "long_entry_trailing_retracement_pct": [0.0, 0.1],
                    "long_entry_trailing_threshold_pct": [-0.1, 0.1],
                    "long_filter_rolling_window": [10.0, 1440.0],
                    "long_filter_relative_volume_clip_pct": [0.0, 1.0],
                    "long_n_positions": [1.0, 20.0],
                    "long_total_wallet_exposure_limit": [0.0, 2.0],
                    "long_unstuck_close_pct": [0.001, 0.1],
                    "long_unstuck_ema_dist": [-0.1, 0.01],
                    "long_unstuck_loss_allowance_pct": [0.001, 0.05],
                    "long_unstuck_threshold": [0.4, 0.95],
                    "short_close_grid_markup_range": [0.0, 0.03],
                    "short_close_grid_min_markup": [0.001, 0.03],
                    "short_close_grid_qty_pct": [0.05, 1.0],
                    "short_close_trailing_grid_ratio": [-1.0, 1.0],
                    "short_close_trailing_qty_pct": [0.05, 1.0],
                    "short_close_trailing_retracement_pct": [0.0, 0.1],
                    "short_close_trailing_threshold_pct": [-0.1, 0.1],
                    "short_ema_span_0": [200.0, 1440.0],
                    "short_ema_span_1": [200.0, 1440.0],
                    "short_entry_grid_double_down_factor": [0.1, 3.0],
                    "short_entry_grid_spacing_pct": [0.005, 0.12],
                    "short_entry_grid_spacing_weight": [0.0, 2.0],
                    "short_entry_initial_ema_dist": [-0.1, 0.002],
                    "short_entry_initial_qty_pct": [0.005, 0.1],
                    "short_entry_trailing_grid_ratio": [-1.0, 1.0],
                    "short_entry_trailing_retracement_pct": [0.0, 0.1],
                    "short_entry_trailing_threshold_pct": [-0.1, 0.1],
                    "short_filter_rolling_window": [10.0, 1440.0],
                    "short_filter_relative_volume_clip_pct": [0.0, 1.0],
                    "short_n_positions": [1.0, 20.0],
                    "short_total_wallet_exposure_limit": [0.0, 2.0],
                    "short_unstuck_close_pct": [0.001, 0.1],
                    "short_unstuck_ema_dist": [-0.1, 0.01],
                    "short_unstuck_loss_allowance_pct": [0.001, 0.05],
                    "short_unstuck_threshold": [0.4, 0.95],
                },
                "compress_results_file": True,
                "crossover_probability": 0.7,
                "enable_overrides": [],
                "iters": 30000,
                "limits": {
                },
                "mutation_probability": 0.2,
                "n_cpus": 5,
                "population_size": 500,
                "scoring": ["adg", "sharpe_ratio"],
            },
        }
    elif passivbot_mode == "multi_hjson":
        return {
            "user": "bybit_01",
            "pnls_max_lookback_days": 30,
            "loss_allowance_pct": 0.005,
            "stuck_threshold": 0.89,
            "unstuck_close_pct": 0.005,
            "execution_delay_seconds": 2,
            "max_n_cancellations_per_batch": 8,
            "max_n_creations_per_batch": 4,
            "price_distance_threshold": 0.002,
            "filter_by_min_effective_cost": False,
            "auto_gs": True,
            "leverage": 10.0,
            "TWE_long": 2.0,
            "TWE_short": 0.1,
            "long_enabled": True,
            "short_enabled": False,
            "approved_symbols": {
                "COIN1": "-lm n -sm gs -lc configs/live/custom/COIN1USDT.json",
                "COIN2": "-lm n -sm gs -sw 0.4",
                "COIN3": "-lm gs -sm n  -lw 0.15 -lev 12",
            },
            "ignored_symbols": ["COIN4", "COIN5"],
            "n_longs": 0,
            "n_shorts": 0,
            "forced_mode_long": "",
            "forced_mode_short": "",
            "minimum_coin_age_days": 60,
            "ohlcv_interval": "15m",
            "relative_volume_filter_clip_pct": 0.1,
            "n_ohlcvs": 100,
            "live_configs_dir": "configs/live/multisymbol/no_AU/",
            "default_config_path": "configs/live/recursive_grid_mode.example.json",
            "universal_live_config": {
                "long": {
                    "ddown_factor": 0.8783,
                    "ema_span_0": 1054.0,
                    "ema_span_1": 1307.0,
                    "initial_eprice_ema_dist": -0.002641,
                    "initial_qty_pct": 0.01151,
                    "markup_range": 0.0008899,
                    "min_markup": 0.007776,
                    "n_close_orders": 3.724,
                    "rentry_pprice_dist": 0.04745,
                    "rentry_pprice_dist_wallet_exposure_weighting": 0.111,
                },
                "short": {
                    "ddown_factor": 0.8783,
                    "ema_span_0": 1054.0,
                    "ema_span_1": 1307.0,
                    "initial_eprice_ema_dist": -0.002641,
                    "initial_qty_pct": 0.01151,
                    "markup_range": 0.0008899,
                    "min_markup": 0.007776,
                    "n_close_orders": 3.724,
                    "rentry_pprice_dist": 0.04745,
                    "rentry_pprice_dist_wallet_exposure_weighting": 0.111,
                },
            },
        }
    elif passivbot_mode == "multi_json":
        return {
            "analysis": {
                "adg": 0.003956322219722023,
                "adg_weighted": 0.0028311504314386944,
                "drawdowns_daily_mean": 0.017813249628287595,
                "loss_profit_ratio": 0.11029048146750035,
                "loss_profit_ratio_long": 0.11029048146750035,
                "loss_profit_ratio_short": 1.0,
                "n_days": 1073,
                "n_iters": 30100,
                "pnl_ratio_long_short": 1.0,
                "pnl_ratios_symbols": {
                    "AVAXUSDT": 0.27468845221661337,
                    "MATICUSDT": 0.2296174818198483,
                    "SOLUSDT": 0.2953751960870163,
                    "SUSHIUSDT": 0.20031886987652528,
                },
                "price_action_distance_mean": 0.04008693098008042,
                "sharpe_ratio": 0.16774229057551596,
                "w_adg_weighted": -0.0028311504314386944,
                "w_drawdowns_daily_mean": 0.017813249628287595,
                "w_loss_profit_ratio": 0.11029048146750035,
                "w_price_action_distance_mean": 0.04008693098008042,
                "w_sharpe_ratio": -0.16774229057551596,
                "worst_drawdown": 0.4964442148721724,
            },
            "args": {
                "end_date": "2024-04-07",
                "exchange": "binance",
                "long_enabled": True,
                "short_enabled": False,
                "start_date": "2021-05-01",
                "starting_balance": 1000000,
                "symbols": ["AVAXUSDT", "MATICUSDT", "SOLUSDT", "SUSHIUSDT"],
                "worst_drawdown_lower_bound": 0.5,
            },
            "live_config": {
                "global": {
                    "TWE_long": 1.5444230850628553,
                    "TWE_short": 9.649688432169954,
                    "loss_allowance_pct": 0.0026679762307641607,
                    "stuck_threshold": 0.8821459931849173,
                    "unstuck_close_pct": 0.0010155575341165876,
                },
                "long": {
                    "ddown_factor": 2.629714810883098,
                    "ema_span_0": 899.2508850110795,
                    "ema_span_1": 421.7063898877953,
                    "enabled": True,
                    "initial_eprice_ema_dist": -0.1,
                    "initial_qty_pct": 0.014476246820125136,
                    "markup_range": 0.0053184619781202315,
                    "min_markup": 0.007118561833656905,
                    "n_close_orders": 1.8921222249558793,
                    "rentry_pprice_dist": 0.053886357819123286,
                    "rentry_pprice_dist_wallet_exposure_weighting": 2.399828941237894,
                    "wallet_exposure_limit": 0.3861057712657138,
                },
                "short": {
                    "ddown_factor": 2.4945922781706855,
                    "ema_span_0": 455.44131691615075,
                    "ema_span_1": 802.61831996626,
                    "enabled": False,
                    "initial_eprice_ema_dist": -0.1,
                    "initial_qty_pct": 0.010939831544335615,
                    "markup_range": 0.003907075073595213,
                    "min_markup": 0.00126517818899668,
                    "n_close_orders": 3.1853269137597926,
                    "rentry_pprice_dist": 0.04288693053869011,
                    "rentry_pprice_dist_wallet_exposure_weighting": 0.48577214018315135,
                    "wallet_exposure_limit": 2.4124221080424886,
                },
            },
        }
    elif passivbot_mode == "recursive_grid":
        return sort_dict_keys(
            {
                "config_name": "recursive_grid_test",
                "logging_level": 0,
                "long": {
                    "enabled": True,
                    "ema_span_0": 1036.4758617491368,
                    "ema_span_1": 1125.5167077975314,
                    "initial_qty_pct": 0.01,
                    "initial_eprice_ema_dist": -0.02,
                    "wallet_exposure_limit": 1.0,
                    "ddown_factor": 0.6,
                    "auto_unstuck_delay_minutes": 300.0,
                    "rentry_pprice_dist": 0.015,
                    "rentry_pprice_dist_wallet_exposure_weighting": 15,
                    "min_markup": 0.02,
                    "markup_range": 0.02,
                    "n_close_orders": 7,
                    "auto_unstuck_qty_pct": 0.04,
                    "auto_unstuck_wallet_exposure_threshold": 0.15,
                    "auto_unstuck_ema_dist": 0.02,
                    "backwards_tp": False,
                },
                "short": {
                    "enabled": False,
                    "ema_span_0": 1036.4758617491368,
                    "ema_span_1": 1125.5167077975314,
                    "initial_qty_pct": 0.01,
                    "initial_eprice_ema_dist": -0.02,
                    "wallet_exposure_limit": 1.0,
                    "ddown_factor": 0.6,
                    "auto_unstuck_delay_minutes": 300.0,
                    "rentry_pprice_dist": 0.015,
                    "rentry_pprice_dist_wallet_exposure_weighting": 15,
                    "min_markup": 0.02,
                    "markup_range": 0.02,
                    "n_close_orders": 7,
                    "auto_unstuck_qty_pct": 0.04,
                    "auto_unstuck_wallet_exposure_threshold": 0.15,
                    "auto_unstuck_ema_dist": 0.02,
                    "backwards_tp": False,
                },
            }
        )
    elif passivbot_mode == "neat_grid":
        return sort_dict_keys(
            {
                "config_name": "neat_template",
                "logging_level": 0,
                "long": {
                    "enabled": True,
                    "ema_span_0": 1440,  # in minutes
                    "ema_span_1": 4320,
                    "grid_span": 0.16,
                    "wallet_exposure_limit": 1.6,
                    "max_n_entry_orders": 10,
                    "initial_qty_pct": 0.01,
                    "initial_eprice_ema_dist": -0.01,  # negative is closer; positive is further away
                    "eqty_exp_base": 1.8,
                    "eprice_exp_base": 1.618034,
                    "min_markup": 0.0045,
                    "markup_range": 0.0075,
                    "n_close_orders": 7,
                    "auto_unstuck_wallet_exposure_threshold": 0.1,  # percentage of wallet_exposure_limit to trigger soft stop.
                    # e.g. wallet_exposure_limit=0.06 and auto_unstuck_wallet_exposure_threshold=0.1: soft stop when wallet_exposure > 0.06 * (1 - 0.1) == 0.054
                    "auto_unstuck_ema_dist": 0.02,
                    "backwards_tp": False,
                    "auto_unstuck_delay_minutes": 300.0,
                    "auto_unstuck_qty_pct": 0.04,
                },
                "short": {
                    "enabled": True,
                    "ema_span_0": 1440,  # in minutes
                    "ema_span_1": 4320,
                    "grid_span": 0.16,
                    "wallet_exposure_limit": 1.6,
                    "max_n_entry_orders": 10,
                    "initial_qty_pct": 0.01,
                    "initial_eprice_ema_dist": -0.01,  # negative is closer; positive is further away
                    "eqty_exp_base": 1.8,
                    "eprice_exp_base": 1.618034,
                    "min_markup": 0.0045,
                    "markup_range": 0.0075,
                    "n_close_orders": 7,
                    "auto_unstuck_wallet_exposure_threshold": 0.1,  # percentage of wallet_exposure_limit to trigger soft stop.
                    # e.g. wallet_exposure_limit=0.06 and auto_unstuck_wallet_exposure_threshold=0.1: soft stop when wallet_exposure > 0.06 * (1 - 0.1) == 0.054
                    "auto_unstuck_ema_dist": 0.02,
                    "backwards_tp": False,
                    "auto_unstuck_delay_minutes": 300.0,
                    "auto_unstuck_qty_pct": 0.04,
                },
            }
        )
    elif passivbot_mode == "clock":
        return sort_dict_keys(
            {
                "config_name": "clock_template",
                "long": {
                    "enabled": True,
                    "wallet_exposure_limit": 1.0,
                    "ema_span_0": 700.0,
                    "ema_span_1": 5300.0,
                    "ema_dist_entry": 0.005,
                    "ema_dist_close": 0.005,
                    "qty_pct_entry": 0.01,
                    "qty_pct_close": 0.01,
                    "we_multiplier_entry": 30.0,
                    "we_multiplier_close": 30.0,
                    "delay_weight_entry": 30.0,
                    "delay_weight_close": 30.0,
                    "delay_between_fills_minutes_entry": 2000.0,
                    "delay_between_fills_minutes_close": 2000.0,
                    "min_markup": 0.0075,
                    "markup_range": 0.03,
                    "n_close_orders": 10,
                    "backwards_tp": True,
                },
                "short": {
                    "enabled": True,
                    "wallet_exposure_limit": 1.0,
                    "ema_span_0": 700.0,
                    "ema_span_1": 5300.0,
                    "ema_dist_entry": 0.0039,
                    "ema_dist_close": 0.0045,
                    "qty_pct_entry": 0.013,
                    "qty_pct_close": 0.03,
                    "we_multiplier_entry": 20.0,
                    "we_multiplier_close": 20.0,
                    "delay_weight_entry": 20.0,
                    "delay_weight_close": 20.0,
                    "delay_between_fills_minutes_entry": 2000.0,
                    "delay_between_fills_minutes_close": 2000.0,
                    "min_markup": 0.0075,
                    "markup_range": 0.03,
                    "n_close_orders": 10,
                    "backwards_tp": True,
                },
            }
        )
    else:
        raise Exception(f"unknown passivbot mode {passivbot_mode}")


def calc_drawdowns(equity_series):
    """
    Calculate the drawdowns of a portfolio of equities over time.

    Parameters:
    equity_series (pandas.Series): A pandas Series containing the portfolio's equity values over time.

    Returns:
    drawdowns (pandas.Series): The drawdowns as a percentage (expressed as a negative value).
    """
    if not isinstance(equity_series, pd.Series):
        equity_series = pd.Series(equity_series)

    # Calculate the cumulative returns of the portfolio
    cumulative_returns = (1 + equity_series.pct_change()).cumprod()

    # Calculate the cumulative maximum value over time
    cumulative_max = cumulative_returns.cummax()

    # Return the drawdown as the percentage decline from the cumulative maximum
    return (cumulative_returns - cumulative_max) / cumulative_max


def calc_max_drawdown(equity_series):
    return calc_drawdowns(equity_series).min()


def calc_sharpe_ratio(equity_series):
    """
    Calculate the Sharpe ratio for a portfolio of equities assuming a zero risk-free rate.

    Parameters:
    equity_series (pandas.Series): A pandas Series containing daily equity values.

    Returns:
    float: The Sharpe ratio.
    """
    if not isinstance(equity_series, pd.Series):
        equity_series = pd.Series(equity_series)

    # Calculate the hourly returns
    returns = equity_series.pct_change().dropna()
    std_dev = returns.std()
    return returns.mean() / std_dev if std_dev != 0.0 else 0.0


def analyze_fills_slim(fills_long: list, fills_short: list, stats: list, config: dict) -> dict:
    sdf = pd.DataFrame(
        stats,
        columns=[
            "timestamp",
            "bkr_price_long",
            "bkr_price_short",
            "psize_long",
            "pprice_long",
            "psize_short",
            "pprice_short",
            "price",
            "closest_bkr_long",
            "closest_bkr_short",
            "balance_long",
            "balance_short",
            "equity_long",
            "equity_short",
        ],
    )
    longs = pd.DataFrame(
        fills_long,
        columns=[
            "trade_id",
            "timestamp",
            "pnl",
            "fee_paid",
            "balance",
            "equity",
            "qty",
            "price",
            "psize",
            "pprice",
            "type",
        ],
    )
    longs.index = longs.timestamp
    shorts = pd.DataFrame(
        fills_short,
        columns=[
            "trade_id",
            "timestamp",
            "pnl",
            "fee_paid",
            "balance",
            "equity",
            "qty",
            "price",
            "psize",
            "pprice",
            "type",
        ],
    )
    shorts.index = shorts.timestamp
    n_days = (sdf.timestamp.iloc[-1] - sdf.timestamp.iloc[0]) / 1000 / 60 / 60 / 24.0
    if config["inverse"]:
        longs.loc[:, "pcost"] = (longs.psize / longs.pprice).abs() * config["c_mult"]
        shorts.loc[:, "pcost"] = (shorts.psize / shorts.pprice).abs() * config["c_mult"]
        sdf.loc[:, "wallet_exposure_long"] = (
            sdf.psize_long / sdf.pprice_long / sdf.balance_long
        ).abs() * config["c_mult"]
        sdf.loc[:, "wallet_exposure_short"] = (
            sdf.psize_short / sdf.pprice_short / sdf.balance_short
        ).abs() * config["c_mult"]
    else:
        longs.loc[:, "pcost"] = (longs.psize * longs.pprice).abs() * config["c_mult"]
        shorts.loc[:, "pcost"] = (shorts.psize * shorts.pprice).abs() * config["c_mult"]
        sdf.loc[:, "wallet_exposure_long"] = (
            sdf.psize_long * sdf.pprice_long / sdf.balance_long
        ).abs() * config["c_mult"]
        sdf.loc[:, "wallet_exposure_short"] = (
            sdf.psize_short * sdf.pprice_short / sdf.balance_short
        ).abs() * config["c_mult"]

    if "adg_n_subdivisions" not in config:
        config["adg_n_subdivisions"] = 1

    if sdf.balance_long.iloc[-1] <= 0.0:
        adg_long = adg_weighted_long = sdf.balance_long.iloc[-1]
    else:
        adgs_long = []
        for i in range(config["adg_n_subdivisions"]):
            idx = round(int(len(sdf) * (1 - 1 / (i + 1))))
            n_days_ = (sdf.timestamp.iloc[-1] - sdf.timestamp.iloc[idx]) / (1000 * 60 * 60 * 24)
            if n_days_ == 0.0 or sdf.balance_long.iloc[idx] == 0.0:
                adgs_long.append(0.0)
            else:
                adgs_long.append(
                    (sdf.balance_long.iloc[-1] / sdf.balance_long.iloc[idx]) ** (1 / n_days_) - 1
                )
        adg_long = adgs_long[0]
        adg_weighted_long = np.mean(adgs_long)
    if sdf.balance_short.iloc[-1] <= 0.0:
        adg_short = adg_weighted_short = sdf.balance_short.iloc[-1]

    else:
        adgs_short = []
        for i in range(config["adg_n_subdivisions"]):
            idx = round(int(len(sdf) * (1 - 1 / (i + 1))))
            n_days_ = (sdf.timestamp.iloc[-1] - sdf.timestamp.iloc[idx]) / (1000 * 60 * 60 * 24)
            if n_days_ == 0.0 or sdf.balance_short.iloc[idx] == 0.0:
                adgs_short.append(0.0)
            else:
                adgs_short.append(
                    (sdf.balance_short.iloc[-1] / sdf.balance_short.iloc[idx]) ** (1 / n_days_) - 1
                )
        adg_short = adgs_short[0]
        adg_weighted_short = np.mean(adgs_short)
    if config["long"]["wallet_exposure_limit"] > 0.0:
        adg_per_exposure_long = adg_long / config["long"]["wallet_exposure_limit"]
        adg_weighted_per_exposure_long = adg_weighted_long / config["long"]["wallet_exposure_limit"]
    else:
        adg_per_exposure_long = adg_weighted_per_exposure_long = 0.0

    if config["short"]["wallet_exposure_limit"] > 0.0:
        adg_per_exposure_short = adg_short / config["short"]["wallet_exposure_limit"]
        adg_weighted_per_exposure_short = (
            adg_weighted_short / config["short"]["wallet_exposure_limit"]
        )
    else:
        adg_per_exposure_short = adg_weighted_per_exposure_short = 0.0

    lpprices = sdf[sdf.psize_long != 0.0]
    spprices = sdf[sdf.psize_short != 0.0]
    pa_dists_long = (
        ((lpprices.pprice_long - lpprices.price).abs() / lpprices.price)
        if len(lpprices) > 0
        else pd.Series([100.0])
    )
    pa_dists_short = (
        ((spprices.pprice_short - spprices.price).abs() / spprices.price)
        if len(spprices) > 0
        else pd.Series([100.0])
    )
    pa_distance_mean_long = pa_dists_long.mean()
    pa_distance_mean_short = pa_dists_short.mean()
    pa_distance_std_long = pa_dists_long.std()
    pa_distance_std_short = pa_dists_short.std()

    ms_diffs_long = longs.timestamp.diff()
    ms_diffs_short = shorts.timestamp.diff()
    hrs_stuck_max_long = max(
        ms_diffs_long.max(),
        (sdf.iloc[-1].timestamp - longs.iloc[-1].timestamp if len(longs) > 0 else 0.0),
    ) / (1000.0 * 60 * 60)
    hrs_stuck_max_short = max(
        ms_diffs_short.max(),
        (sdf.iloc[-1].timestamp - shorts.iloc[-1].timestamp if len(shorts) > 0 else 0.0),
    ) / (1000.0 * 60 * 60)

    profit_sum_long = longs[longs.pnl > 0.0].pnl.sum()
    loss_sum_long = longs[longs.pnl < 0.0].pnl.sum()

    profit_sum_short = shorts[shorts.pnl > 0.0].pnl.sum()
    loss_sum_short = shorts[shorts.pnl < 0.0].pnl.sum()

    exposure_ratios_long = sdf.wallet_exposure_long / config["long"]["wallet_exposure_limit"]
    time_at_max_exposure_long = (
        1.0 if len(sdf) == 0 else (len(exposure_ratios_long[exposure_ratios_long > 0.9]) / len(sdf))
    )
    exposure_ratios_mean_long = exposure_ratios_long.mean()
    exposure_ratios_short = sdf.wallet_exposure_short / config["short"]["wallet_exposure_limit"]
    time_at_max_exposure_short = (
        1.0 if len(sdf) == 0 else (len(exposure_ratios_short[exposure_ratios_short > 0.9]) / len(sdf))
    )
    exposure_ratios_mean_short = exposure_ratios_short.mean()

    drawdowns_long = calc_drawdowns(sdf.equity_long)
    drawdown_max_long = drawdowns_long.min()
    mean_of_10_worst_drawdowns = drawdowns_long.sort_values().iloc[:10].mean()

    drawdowns_long = calc_drawdowns(sdf.equity_long)
    drawdowns_short = calc_drawdowns(sdf.equity_short)

    daily_sdf = sdf.groupby(sdf.timestamp // (1000 * 60 * 60 * 24)).last()
    sharpe_ratio_long = calc_sharpe_ratio(daily_sdf.equity_long)
    sharpe_ratio_short = calc_sharpe_ratio(daily_sdf.equity_short)

    return {
        "adg_weighted_per_exposure_long": adg_weighted_long / config["long"]["wallet_exposure_limit"],
        "adg_weighted_per_exposure_short": adg_weighted_short
        / config["short"]["wallet_exposure_limit"],
        "adg_per_exposure_long": adg_long / config["long"]["wallet_exposure_limit"],
        "adg_per_exposure_short": adg_short / config["short"]["wallet_exposure_limit"],
        "n_days": n_days,
        "starting_balance": sdf.balance_long.iloc[0],
        "pa_distance_mean_long": (
            pa_distance_mean_long if pa_distance_mean_long == pa_distance_mean_long else 1.0
        ),
        "pa_distance_max_long": pa_dists_long.max(),
        "pa_distance_std_long": (
            pa_distance_std_long if pa_distance_std_long == pa_distance_std_long else 1.0
        ),
        "pa_distance_mean_short": (
            pa_distance_mean_short if pa_distance_mean_short == pa_distance_mean_short else 1.0
        ),
        "pa_distance_max_short": pa_dists_short.max(),
        "pa_distance_std_short": (
            pa_distance_std_short if pa_distance_std_short == pa_distance_std_short else 1.0
        ),
        "pa_distance_1pct_worst_mean_long": pa_dists_long.sort_values()
        .iloc[-max(1, len(sdf) // 100) :]
        .mean(),
        "pa_distance_1pct_worst_mean_short": pa_dists_short.sort_values()
        .iloc[-max(1, len(sdf) // 100) :]
        .mean(),
        "hrs_stuck_max_long": hrs_stuck_max_long,
        "hrs_stuck_max_short": hrs_stuck_max_short,
        "loss_profit_ratio_long": (
            abs(loss_sum_long) / profit_sum_long if profit_sum_long > 0.0 else 1.0
        ),
        "loss_profit_ratio_short": (
            abs(loss_sum_short) / profit_sum_short if profit_sum_short > 0.0 else 1.0
        ),
        "exposure_ratios_mean_long": exposure_ratios_mean_long,
        "exposure_ratios_mean_short": exposure_ratios_mean_short,
        "time_at_max_exposure_long": time_at_max_exposure_long,
        "time_at_max_exposure_short": time_at_max_exposure_short,
        "drawdown_max_long": -drawdowns_long.min(),
        "drawdown_max_short": -drawdowns_short.min(),
        "drawdown_1pct_worst_mean_long": -drawdowns_long.sort_values()
        .iloc[: (len(drawdowns_long) // 100)]
        .mean(),
        "drawdown_1pct_worst_mean_short": -drawdowns_short.sort_values()
        .iloc[: (len(drawdowns_short) // 100)]
        .mean(),
        "sharpe_ratio_long": sharpe_ratio_long,
        "sharpe_ratio_short": sharpe_ratio_short,
    }


def analyze_fills(
    fills_long: list, fills_short: list, stats: list, config: dict
) -> (pd.DataFrame, pd.DataFrame, dict):
    sdf = pd.DataFrame(
        stats,
        columns=[
            "timestamp",
            "bkr_price_long",
            "bkr_price_short",
            "psize_long",
            "pprice_long",
            "psize_short",
            "pprice_short",
            "price",
            "closest_bkr_long",
            "closest_bkr_short",
            "balance_long",
            "balance_short",
            "equity_long",
            "equity_short",
        ],
    )
    longs = pd.DataFrame(
        fills_long,
        columns=[
            "trade_id",
            "timestamp",
            "pnl",
            "fee_paid",
            "balance",
            "equity",
            "qty",
            "price",
            "psize",
            "pprice",
            "type",
        ],
    )
    longs.index = longs.timestamp
    shorts = pd.DataFrame(
        fills_short,
        columns=[
            "trade_id",
            "timestamp",
            "pnl",
            "fee_paid",
            "balance",
            "equity",
            "qty",
            "price",
            "psize",
            "pprice",
            "type",
        ],
    )
    shorts.index = shorts.timestamp
    n_days = (sdf.timestamp.iloc[-1] - sdf.timestamp.iloc[0]) / 1000 / 60 / 60 / 24.0
    if config["inverse"]:
        longs.loc[:, "pcost"] = (longs.psize / longs.pprice).abs() * config["c_mult"]
        shorts.loc[:, "pcost"] = (shorts.psize / shorts.pprice).abs() * config["c_mult"]
        sdf.loc[:, "wallet_exposure_long"] = (
            sdf.psize_long / sdf.pprice_long / sdf.balance_long
        ).abs() * config["c_mult"]
        sdf.loc[:, "wallet_exposure_short"] = (
            sdf.psize_short / sdf.pprice_short / sdf.balance_short
        ).abs() * config["c_mult"]
    else:
        longs.loc[:, "pcost"] = (longs.psize * longs.pprice).abs() * config["c_mult"]
        shorts.loc[:, "pcost"] = (shorts.psize * shorts.pprice).abs() * config["c_mult"]
        sdf.loc[:, "wallet_exposure_long"] = (
            sdf.psize_long * sdf.pprice_long / sdf.balance_long
        ).abs() * config["c_mult"]
        sdf.loc[:, "wallet_exposure_short"] = (
            sdf.psize_short * sdf.pprice_short / sdf.balance_short
        ).abs() * config["c_mult"]
    longs.loc[:, "wallet_exposure"] = longs.pcost / longs.balance
    shorts.loc[:, "wallet_exposure"] = shorts.pcost / shorts.balance

    ms_diffs_long = longs.timestamp.diff()
    ms_diffs_short = shorts.timestamp.diff()
    longs.loc[:, "mins_since_prev_fill"] = ms_diffs_long / 1000.0 / 60.0
    shorts.loc[:, "mins_since_prev_fill"] = ms_diffs_short / 1000.0 / 60.0

    profit_sum_long = longs[longs.pnl > 0.0].pnl.sum()
    loss_sum_long = longs[longs.pnl < 0.0].pnl.sum()
    pnl_sum_long = profit_sum_long + loss_sum_long
    gain_long = sdf.balance_long.iloc[-1] / sdf.balance_long.iloc[0] - 1

    profit_sum_short = shorts[shorts.pnl > 0.0].pnl.sum()
    loss_sum_short = shorts[shorts.pnl < 0.0].pnl.sum()
    pnl_sum_short = profit_sum_short + loss_sum_short
    gain_short = sdf.balance_short.iloc[-1] / sdf.balance_short.iloc[0] - 1

    # adgs:
    # adg
    # adg_per_exposure
    # adg_weighted
    # adg_weighted_per_exposure

    if "adg_n_subdivisions" not in config:
        config["adg_n_subdivisions"] = 1

    if sdf.balance_long.iloc[-1] <= 0.0:
        adg_long = adg_weighted_long = sdf.balance_long.iloc[-1]
    else:
        adgs_long = []
        for i in range(config["adg_n_subdivisions"]):
            idx = round(int(len(sdf) * (1 - 1 / (i + 1))))
            n_days_ = (sdf.timestamp.iloc[-1] - sdf.timestamp.iloc[idx]) / (1000 * 60 * 60 * 24)
            if n_days_ == 0.0 or sdf.balance_long.iloc[idx] == 0.0:
                adgs_long.append(0.0)
            else:
                adgs_long.append(
                    (sdf.balance_long.iloc[-1] / sdf.balance_long.iloc[idx]) ** (1 / n_days_) - 1
                )
        adg_long = adgs_long[0]
        adg_weighted_long = np.mean(adgs_long)
    if sdf.balance_short.iloc[-1] <= 0.0:
        adg_short = adg_weighted_short = sdf.balance_short.iloc[-1]

    else:
        adgs_short = []
        for i in range(config["adg_n_subdivisions"]):
            idx = round(int(len(sdf) * (1 - 1 / (i + 1))))
            n_days_ = (sdf.timestamp.iloc[-1] - sdf.timestamp.iloc[idx]) / (1000 * 60 * 60 * 24)
            if n_days_ == 0.0 or sdf.balance_short.iloc[idx] == 0.0:
                adgs_short.append(0.0)
            else:
                adgs_short.append(
                    (sdf.balance_short.iloc[-1] / sdf.balance_short.iloc[idx]) ** (1 / n_days_) - 1
                )
        adg_short = adgs_short[0]
        adg_weighted_short = np.mean(adgs_short)
    if config["long"]["wallet_exposure_limit"] > 0.0:
        adg_per_exposure_long = adg_long / config["long"]["wallet_exposure_limit"]
        adg_weighted_per_exposure_long = adg_weighted_long / config["long"]["wallet_exposure_limit"]
    else:
        adg_per_exposure_long = adg_weighted_per_exposure_long = 0.0
    if config["short"]["wallet_exposure_limit"] > 0.0:
        adg_per_exposure_short = adg_short / config["short"]["wallet_exposure_limit"]
        adg_weighted_per_exposure_short = (
            adg_weighted_short / config["short"]["wallet_exposure_limit"]
        )
    else:
        adg_per_exposure_short = adg_weighted_per_exposure_short = 0.0

    volume_quote_long = longs.pcost.sum()
    volume_quote_short = shorts.pcost.sum()

    lpprices = sdf[sdf.psize_long != 0.0]
    spprices = sdf[sdf.psize_short != 0.0]
    pa_dists_long = (
        ((lpprices.pprice_long - lpprices.price).abs() / lpprices.price)
        if len(lpprices) > 0
        else pd.Series([100.0])
    )
    pa_dists_short = (
        ((spprices.pprice_short - spprices.price).abs() / spprices.price)
        if len(spprices) > 0
        else pd.Series([100.0])
    )
    pa_distance_std_long = pa_dists_long.std()
    pa_distance_std_short = pa_dists_short.std()
    pa_distance_mean_long = pa_dists_long.mean()
    pa_distance_mean_short = pa_dists_short.mean()

    eqbal_ratios_long = longs.equity / longs.balance
    eqbal_ratios_sdf_long = sdf.equity_long / sdf.balance_long
    eqbal_ratio_std_long = eqbal_ratios_sdf_long.std()
    eqbal_ratios_short = shorts.equity / shorts.balance
    eqbal_ratios_sdf_short = sdf.equity_short / sdf.balance_short
    eqbal_ratio_std_short = eqbal_ratios_sdf_short.std()

    exposure_ratios_long = sdf.wallet_exposure_long / config["long"]["wallet_exposure_limit"]
    time_at_max_exposure_long = (
        1.0 if len(sdf) == 0 else (len(exposure_ratios_long[exposure_ratios_long > 0.9]) / len(sdf))
    )
    exposure_ratios_mean_long = exposure_ratios_long.mean()
    exposure_ratios_short = sdf.wallet_exposure_short / config["short"]["wallet_exposure_limit"]
    time_at_max_exposure_short = (
        1.0 if len(sdf) == 0 else (len(exposure_ratios_short[exposure_ratios_short > 0.9]) / len(sdf))
    )
    exposure_ratios_mean_short = exposure_ratios_short.mean()

    drawdowns_long = calc_drawdowns(sdf.equity_long)
    drawdowns_short = calc_drawdowns(sdf.equity_short)

    daily_sdf = sdf.groupby(sdf.timestamp // (1000 * 60 * 60 * 24)).last()
    sharpe_ratio_long = calc_sharpe_ratio(daily_sdf.equity_long)
    sharpe_ratio_short = calc_sharpe_ratio(daily_sdf.equity_short)

    analysis = {
        "exchange": config["exchange"] if "exchange" in config else "unknown",
        "symbol": config["symbol"] if "symbol" in config else "unknown",
        "starting_balance": sdf.balance_long.iloc[0],
        "pa_distance_mean_long": (
            pa_distance_mean_long if pa_distance_mean_long == pa_distance_mean_long else 1.0
        ),
        "pa_distance_max_long": pa_dists_long.max(),
        "pa_distance_std_long": (
            pa_distance_std_long if pa_distance_std_long == pa_distance_std_long else 1.0
        ),
        "pa_distance_mean_short": (
            pa_distance_mean_short if pa_distance_mean_short == pa_distance_mean_short else 1.0
        ),
        "pa_distance_max_short": pa_dists_short.max(),
        "pa_distance_std_short": (
            pa_distance_std_short if pa_distance_std_short == pa_distance_std_short else 1.0
        ),
        "pa_distance_1pct_worst_mean_long": pa_dists_long.sort_values()
        .iloc[-max(1, len(sdf) // 100) :]
        .mean(),
        "pa_distance_1pct_worst_mean_short": pa_dists_short.sort_values()
        .iloc[-max(1, len(sdf) // 100) :]
        .mean(),
        "equity_balance_ratio_mean_long": (sdf.equity_long / sdf.balance_long).mean(),
        "equity_balance_ratio_std_long": eqbal_ratio_std_long,
        "equity_balance_ratio_mean_short": (sdf.equity_short / sdf.balance_short).mean(),
        "equity_balance_ratio_std_short": eqbal_ratio_std_short,
        "gain_long": gain_long,
        "adg_long": adg_long if adg_long == adg_long else -1.0,
        "adg_weighted_long": adg_weighted_long if adg_weighted_long == adg_weighted_long else -1.0,
        "adg_per_exposure_long": adg_per_exposure_long,
        "adg_weighted_per_exposure_long": adg_weighted_per_exposure_long,
        "gain_short": gain_short,
        "adg_short": adg_short if adg_short == adg_short else -1.0,
        "adg_weighted_short": (
            adg_weighted_short if adg_weighted_short == adg_weighted_short else -1.0
        ),
        "adg_per_exposure_short": adg_per_exposure_short,
        "adg_weighted_per_exposure_short": adg_weighted_per_exposure_short,
        "exposure_ratios_mean_long": exposure_ratios_mean_long,
        "exposure_ratios_mean_short": exposure_ratios_mean_short,
        "time_at_max_exposure_long": time_at_max_exposure_long,
        "time_at_max_exposure_short": time_at_max_exposure_short,
        "n_days": n_days,
        "n_fills_long": len(fills_long),
        "n_fills_short": len(fills_short),
        "n_closes_long": len(longs[longs.type.str.contains("close")]),
        "n_closes_short": len(shorts[shorts.type.str.contains("close")]),
        "n_normal_closes_long": len(longs[longs.type.str.contains("nclose")]),
        "n_normal_closes_short": len(shorts[shorts.type.str.contains("nclose")]),
        "n_entries_long": len(longs[longs.type.str.contains("entry")]),
        "n_entries_short": len(shorts[shorts.type.str.contains("entry")]),
        "n_ientries_long": len(longs[longs.type.str.contains("ientry")]),
        "n_ientries_short": len(shorts[shorts.type.str.contains("ientry")]),
        "n_rentries_long": len(longs[longs.type.str.contains("rentry")]),
        "n_rentries_short": len(shorts[shorts.type.str.contains("rentry")]),
        "n_unstuck_closes_long": len(
            longs[longs.type.str.contains("unstuck_close") | longs.type.str.contains("clock_close")]
        ),
        "n_unstuck_closes_short": len(
            shorts[
                shorts.type.str.contains("unstuck_close") | shorts.type.str.contains("clock_close")
            ]
        ),
        "n_unstuck_entries_long": len(
            longs[longs.type.str.contains("unstuck_entry") | longs.type.str.contains("clock_entry")]
        ),
        "n_unstuck_entries_short": len(
            shorts[
                shorts.type.str.contains("unstuck_entry") | shorts.type.str.contains("clock_entry")
            ]
        ),
        "avg_fills_per_day_long": len(longs) / n_days,
        "avg_fills_per_day_short": len(shorts) / n_days,
        "hrs_stuck_max_long": ms_diffs_long.max() / (1000.0 * 60 * 60),
        "hrs_stuck_avg_long": ms_diffs_long.mean() / (1000.0 * 60 * 60),
        "hrs_stuck_max_short": ms_diffs_short.max() / (1000.0 * 60 * 60),
        "hrs_stuck_avg_short": ms_diffs_short.mean() / (1000.0 * 60 * 60),
        "loss_sum_long": loss_sum_long,
        "loss_sum_short": loss_sum_short,
        "profit_sum_long": profit_sum_long,
        "profit_sum_short": profit_sum_short,
        "pnl_sum_long": pnl_sum_long,
        "pnl_sum_short": pnl_sum_short,
        "loss_profit_ratio_long": (abs(loss_sum_long) / profit_sum_long) if profit_sum_long else 1.0,
        "loss_profit_ratio_short": (
            (abs(loss_sum_short) / profit_sum_short) if profit_sum_short else 1.0
        ),
        "fee_sum_long": (fee_sum_long := longs.fee_paid.sum()),
        "fee_sum_short": (fee_sum_short := shorts.fee_paid.sum()),
        "net_pnl_plus_fees_long": pnl_sum_long + fee_sum_long,
        "net_pnl_plus_fees_short": pnl_sum_short + fee_sum_short,
        "final_equity_long": sdf.equity_long.iloc[-1],
        "final_balance_long": sdf.balance_long.iloc[-1],
        "final_equity_short": sdf.equity_short.iloc[-1],
        "final_balance_short": sdf.balance_short.iloc[-1],
        "closest_bkr_long": sdf.closest_bkr_long.min(),
        "closest_bkr_short": sdf.closest_bkr_short.min(),
        "eqbal_ratio_min_long": min(eqbal_ratios_long.min(), eqbal_ratios_sdf_long.min()),
        "eqbal_ratio_mean_of_10_worst_long": eqbal_ratios_sdf_long.sort_values().iloc[:10].mean(),
        "eqbal_ratio_mean_long": eqbal_ratios_sdf_long.mean(),
        "eqbal_ratio_std_long": eqbal_ratio_std_long,
        "eqbal_ratio_min_short": min(eqbal_ratios_short.min(), eqbal_ratios_sdf_short.min()),
        "eqbal_ratio_mean_of_10_worst_short": eqbal_ratios_sdf_short.sort_values().iloc[:10].mean(),
        "eqbal_ratio_mean_short": eqbal_ratios_sdf_short.mean(),
        "eqbal_ratio_std_short": eqbal_ratio_std_short,
        "volume_quote_long": volume_quote_long,
        "volume_quote_short": volume_quote_short,
        "drawdown_max_long": -drawdowns_long.min(),
        "drawdown_max_short": -drawdowns_short.min(),
        "drawdown_1pct_worst_mean_long": -drawdowns_long.sort_values()
        .iloc[: (len(drawdowns_long) // 100)]
        .mean(),
        "drawdown_1pct_worst_mean_short": -drawdowns_short.sort_values()
        .iloc[: (len(drawdowns_short) // 100)]
        .mean(),
        "sharpe_ratio_long": sharpe_ratio_long,
        "sharpe_ratio_short": sharpe_ratio_short,
    }
    return longs, shorts, sdf, sort_dict_keys(analysis)


def get_empty_analysis():
    return {
        "exchange": "unknown",
        "symbol": "unknown",
        "starting_balance": 0.0,
        "pa_distance_mean_long": 1000.0,
        "pa_distance_max_long": 1000.0,
        "pa_distance_std_long": 1000.0,
        "pa_distance_mean_short": 1000.0,
        "pa_distance_max_short": 1000.0,
        "pa_distance_std_short": 1000.0,
        "gain_long": 0.0,
        "adg_long": 0.0,
        "adg_per_exposure_long": 0.0,
        "gain_short": 0.0,
        "adg_short": 0.0,
        "adg_per_exposure_short": 0.0,
        "adg_DGstd_ratio_long": 0.0,
        "adg_DGstd_ratio_short": 0.0,
        "adg_realized_long": 0.0,
        "adg_realized_short": 0.0,
        "adg_realized_per_exposure_long": 0.0,
        "adg_realized_per_exposure_short": 0.0,
        "DGstd_long": 0.0,
        "DGstd_short": 0.0,
        "n_days": 0.0,
        "n_fills_long": 0.0,
        "n_fills_short": 0.0,
        "n_closes_long": 0.0,
        "n_closes_short": 0.0,
        "n_normal_closes_long": 0.0,
        "n_normal_closes_short": 0.0,
        "n_entries_long": 0.0,
        "n_entries_short": 0.0,
        "n_ientries_long": 0.0,
        "n_ientries_short": 0.0,
        "n_rentries_long": 0.0,
        "n_rentries_short": 0.0,
        "n_unstuck_closes_long": 0.0,
        "n_unstuck_closes_short": 0.0,
        "n_unstuck_entries_long": 0.0,
        "n_unstuck_entries_short": 0.0,
        "avg_fills_per_day_long": 0.0,
        "avg_fills_per_day_short": 0.0,
        "hrs_stuck_max_long": 1000.0,
        "hrs_stuck_avg_long": 1000.0,
        "hrs_stuck_max_short": 1000.0,
        "hrs_stuck_avg_short": 1000.0,
        "hrs_stuck_max": 1000.0,
        "hrs_stuck_avg": 1000.0,
        "loss_sum_long": 0.0,
        "loss_sum_short": 0.0,
        "profit_sum_long": 0.0,
        "profit_sum_short": 0.0,
        "pnl_sum_long": 0.0,
        "pnl_sum_short": 0.0,
        "fee_sum_long": 0.0,
        "fee_sum_short": 0.0,
        "net_pnl_plus_fees_long": 0.0,
        "net_pnl_plus_fees_short": 0.0,
        "final_equity_long": 0.0,
        "final_balance_long": 0.0,
        "final_equity_short": 0.0,
        "final_balance_short": 0.0,
        "closest_bkr_long": 0.0,
        "closest_bkr_short": 0.0,
        "eqbal_ratio_min_long": 0.0,
        "eqbal_ratio_mean_long": 0.0,
        "eqbal_ratio_min_short": 0.0,
        "eqbal_ratio_mean_short": 0.0,
        "biggest_psize_long": 0.0,
        "biggest_psize_short": 0.0,
        "biggest_psize_quote_long": 0.0,
        "biggest_psize_quote_short": 0.0,
        "volume_quote_long": 0.0,
        "volume_quote_short": 0.0,
    }


def calc_pprice_from_fills(coin_balance, fills, n_fills_limit=100):
    # assumes fills are sorted old to new
    if coin_balance == 0.0 or len(fills) == 0:
        return 0.0
    relevant_fills = []
    qty_sum = 0.0
    for fill in fills[::-1][:n_fills_limit]:
        abs_qty = fill["qty"]
        if fill["side"] == "buy":
            adjusted_qty = min(abs_qty, coin_balance - qty_sum)
            qty_sum += adjusted_qty
            relevant_fills.append({**fill, **{"qty": adjusted_qty}})
            if qty_sum >= coin_balance * 0.999:
                break
        else:
            qty_sum -= abs_qty
            relevant_fills.append(fill)
    psize, pprice = 0.0, 0.0
    for fill in relevant_fills[::-1]:
        abs_qty = abs(fill["qty"])
        if fill["side"] == "buy":
            new_psize = psize + abs_qty
            pprice = pprice * (psize / new_psize) + fill["price"] * (abs_qty / new_psize)
            psize = new_psize
        else:
            psize -= abs_qty
    return pprice


def get_position_fills(psize_long: float, psize_short: float, fills: [dict]) -> ([dict], [dict]):
    """
    returns fills since and including initial entry
    """
    fills = sorted(fills, key=lambda x: x["timestamp"])  # sort old to new
    psize_long *= 0.999
    psize_short *= 0.999
    long_qty_sum = 0.0
    short_qty_sum = 0.0
    long_done, short_done = psize_long == 0.0, psize_short == 0.0
    if long_done and short_done:
        return [], []
    long_pfills, short_pfills = [], []
    for x in fills[::-1]:
        if x["position_side"] == "long":
            if not long_done:
                long_qty_sum += x["qty"] * (1.0 if x["side"] == "buy" else -1.0)
                long_pfills.append(x)
                long_done = long_qty_sum >= psize_long
        elif x["position_side"] == "short":
            if not short_done:
                short_qty_sum += x["qty"] * (1.0 if x["side"] == "sell" else -1.0)
                short_pfills.append(x)
                short_done = short_qty_sum >= psize_short
    return long_pfills[::-1], short_pfills[::-1]


def calc_pprice_long(psize_long, long_pfills):
    """
    assumes long pfills are sorted old to new
    """
    psize, pprice = 0.0, 0.0
    for fill in long_pfills:
        abs_qty = abs(fill["qty"])
        if fill["side"] == "buy":
            new_psize = psize + abs_qty
            pprice = pprice * (psize / new_psize) + fill["price"] * (abs_qty / new_psize)
            psize = new_psize
        else:
            psize = max(0.0, psize - abs_qty)
    return pprice


def nullify(x):
    if type(x) in [list, tuple]:
        return [nullify(x1) for x1 in x]
    elif type(x) == np.ndarray:
        return numpyize([nullify(x1) for x1 in x])
    elif type(x) == dict:
        return {k: nullify(x[k]) for k in x}
    elif type(x) in [bool, np.bool_]:
        return x
    else:
        return 0.0


def spotify_config(config: dict, nullify_short=True) -> dict:
    spotified = config.copy()

    spotified["spot"] = True
    if "market_type" not in spotified:
        spotified["market_type"] = "spot"
    elif "spot" not in spotified["market_type"]:
        spotified["market_type"] += "_spot"
    spotified["do_long"] = spotified["long"]["enabled"] = config["long"]["enabled"]
    spotified["do_short"] = spotified["short"]["enabled"] = False
    spotified["long"]["wallet_exposure_limit"] = min(1.0, spotified["long"]["wallet_exposure_limit"])
    if nullify_short:
        spotified["short"] = nullify(spotified["short"])
    return spotified


def tuplify(xs, sort=False):
    if type(xs) in [list]:
        if sort:
            return tuple(sorted(tuplify(x, sort=sort) for x in xs))
        return tuple(tuplify(x, sort=sort) for x in xs)
    elif type(xs) in [dict, OrderedDict]:
        if sort:
            return tuple(sorted({k: tuplify(v, sort=sort) for k, v in xs.items()}.items()))
        return tuple({k: tuplify(v, sort=sort) for k, v in xs.items()}.items())
    return xs


def round_values(xs, n: int):
    if type(xs) in [float, np.float64]:
        return pbr.round_dynamic(xs, n)
    if type(xs) == dict:
        return {k: round_values(xs[k], n) for k in xs}
    if type(xs) == list:
        return [round_values(x, n) for x in xs]
    if type(xs) == np.ndarray:
        return numpyize([round_values(x, n) for x in xs])
    if type(xs) == tuple:
        return tuple([round_values(x, n) for x in xs])
    if type(xs) == OrderedDict:
        return OrderedDict([(k, round_values(xs[k], n)) for k in xs])
    return xs
=======
def flatten(nested):
    return [item for sublist in nested for item in sublist]
>>>>>>> upstream/master


def floatify(xs):
    if isinstance(xs, (int, float)):
        return float(xs)
    if isinstance(xs, str):
        try:
            return float(xs)
        except (ValueError, TypeError):
            return xs
    if isinstance(xs, bool):
        return xs
    if isinstance(xs, list):
        return [floatify(x) for x in xs]
    if isinstance(xs, tuple):
        return tuple(floatify(x) for x in xs)
    if isinstance(xs, dict):
        return {k: floatify(v) for k, v in xs.items()}
    return xs


def shorten_custom_id(id_: str) -> str:
    replacements = [
        ("clock", "clk"),
        ("close", "cls"),
        ("entry", "etr"),
        ("_", ""),
        ("normal", "nrml"),
        ("long", "lng"),
        ("short", "shrt"),
        ("primary", "prm"),
        ("unstuck", "ustk"),
        ("partial", "prtl"),
        ("panic", "pnc"),
    ]
    for before, after in replacements:
        id_ = id_.replace(before, after)
    return id_


def determine_pos_side_ccxt(open_order: dict) -> str:
    info = open_order.get("info", open_order)
    if "positionIdx" in info:
        idx = float(info["positionIdx"])
        if idx == 1.0:
            return "long"
        if idx == 2.0:
            return "short"

    keys_map = {key.lower().replace("_", ""): key for key in info}
    for pos_key in ("posside", "positionside"):
        if pos_key in keys_map:
            return info[keys_map[pos_key]].lower()

    if info.get("side", "").lower() == "buy":
        if "reduceonly" in keys_map:
            return "long" if not info[keys_map["reduceonly"]] else "short"
        if "closedsize" in keys_map:
            return "long" if float(info[keys_map["closedsize"]]) != 0.0 else "short"

    for key in ["order_link_id", "clOrdId", "clientOid", "orderLinkId"]:
        if key in info:
            value = info[key].lower()
            if "long" in value or "lng" in value:
                return "long"
            if "short" in value or "shrt" in value:
                return "short"
    return "both"


def calc_hash(data) -> str:
    data_string = json.dumps(data, sort_keys=True)
    return sha256(data_string.encode("utf-8")).hexdigest()


def ensure_millis(timestamp):
    """Normalize various timestamp formats to milliseconds."""
    if not isinstance(timestamp, (int, float)):
        raise TypeError("Timestamp must be an int or float")

    ts = float(timestamp)
    if ts > 1e16:  # nanoseconds
        return ts / 1e6
    if ts > 1e14:  # microseconds
        return ts / 1e3
    if ts > 1e11:  # milliseconds
        return ts
    if ts > 1e9:  # seconds with decimals
        return ts * 1e3
    if ts > 1e6:  # seconds
        return ts * 1e3
    raise ValueError("Timestamp value too small or unrecognized format")


def multi_replace(input_data, replacements):
    if isinstance(input_data, str):
        for old, new in replacements:
            input_data = input_data.replace(old, new)
        return input_data
    if isinstance(input_data, list):
        return [multi_replace(item, replacements) for item in input_data]
    if isinstance(input_data, dict):
        return {key: multi_replace(value, replacements) for key, value in input_data.items()}
    return input_data


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("yes", "true", "t", "y", "1"):
        return True
    if lowered in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError("Boolean value expected.")


def determine_side_from_order_tuple(order_tuple):
    side_info = order_tuple[2]
    if "long" in side_info:
        if "entry" in side_info:
            return "buy"
        if "close" in side_info:
            return "sell"
    elif "short" in side_info:
        if "entry" in side_info:
            return "sell"
        if "close" in side_info:
            return "buy"
    raise ValueError(f"malformed order tuple {order_tuple}")


def remove_OD(d):
    if isinstance(d, dict):
        return {k: remove_OD(v) for k, v in d.items()}
    if isinstance(d, list):
        return [remove_OD(x) for x in d]
    return d


def log_dict_changes(d1, d2, parent_key=""):
    """Return a summary of differences between two nested dictionaries."""

    changes = {"added": [], "removed": [], "changed": []}
    if not d1:
        changes["added"].extend([f"{parent_key}{k}: {v}" for k, v in (d2 or {}).items()])
        return changes
    if not d2:
        changes["removed"].extend([f"{parent_key}{k}: {v}" for k, v in (d1 or {}).items()])
        return changes

    for key in sorted(set(d1.keys()) | set(d2.keys())):
        new_parent = f"{parent_key}{key}." if parent_key else f"{key}."
        if key in d1 and key in d2:
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                nested = log_dict_changes(d1[key], d2[key], new_parent)
                for change_type, items in nested.items():
                    changes[change_type].extend(items)
            elif d1[key] != d2[key]:
                changes["changed"].append(f"{parent_key}{key}: {d1[key]} -> {d2[key]}")
        elif key in d2:
            if isinstance(d2[key], dict):
                nested = log_dict_changes({}, d2[key], new_parent)
                changes["added"].extend(nested["added"])
            else:
                changes["added"].append(f"{parent_key}{key}: {d2[key]}")
        else:
            if isinstance(d1[key], dict):
                nested = log_dict_changes(d1[key], {}, new_parent)
                changes["removed"].extend(nested["removed"])
            else:
                changes["removed"].append(f"{parent_key}{key}: {d1[key]}")
    return changes
