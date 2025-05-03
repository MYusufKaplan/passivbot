import numpy as np
import json
from optimizer_overrides import optimizer_overrides
from pure_funcs import get_template_live_config
from copy import deepcopy
from backtest import (
    prep_backtest_args,
    expand_analysis
    )
import passivbot_rust as pbr
import traceback
import fcntl
import time
from rich.console import Console
from rich.text import Text

console = Console()

def individual_to_config(individual, optimizer_overrides, overrides_list, template=None):
    if template is None:
        template = get_template_live_config("v7")
    keys_ignored = ["enforce_exposure_limit"]
    config = deepcopy(template)
    keys = [k for k in sorted(config["bot"]["long"]) if k not in keys_ignored]
    i = 0
    for pside in ["long", "short"]:
        for key in keys:
            config["bot"][pside][key] = individual[i]
            i += 1
        is_enabled = (
            config["bot"][pside]["total_wallet_exposure_limit"] > 0.0
            and config["bot"][pside]["n_positions"] > 0.0
        )
        if not is_enabled:
            for key in config["bot"][pside]:
                if key in keys_ignored:
                    continue
                bounds = config["optimize"]["bounds"][f"{pside}_{key}"]
                if len(bounds) == 1:
                    bounds = [bounds[0], bounds[0]]
                config["bot"][pside][key] = min(max(bounds[0], 0.0), bounds[1])
        # Call the optimizer overrides
        config = optimizer_overrides(overrides_list, config, pside)
    return config

def calc_fitness(config,analyses_combined,verbose=True):
    modifier = 0.0
    keys = config["optimize"]["limits"]
    i = len(keys) + 1
    # i = 5
    prefix = "btc_" if config["backtest"]["use_btc_collateral"] else ""

    # Step 1: Initialize min/max values
    min_contribution = float('inf')
    max_contribution = float('-inf')
    min_modifier = float('inf')
    max_modifier = float('-inf')

    # Define color codes
    RESET = "\033[0m"
    CYAN = "\033[96m"
    # Define the custom green and red colors
    GREEN_RGB = (195, 232, 141)
    RED_RGB = (255, 83, 112)

    # Store the results for each key to later apply colors
    results = []

    header_aliases = {
        "adg": "ADG",
        "adg_w": "ADG(w)",
        "mdg": "MDG",
        "mdg_w": "MDG(w)",
        "gain": "Gain",
        "positions_held_per_day": "Pos/Day",
        "sharpe_ratio": "Sharpe",
        "sharpe_ratio_w": "Sharpe(w)",
        "drawdown_worst": "DD Worst",
        "drawdown_worst_mean_1pct": "DD 1%",
        "expected_shortfall_1pct": "ES 1%",
        "position_held_hours_mean": "Hrs/Pos",
        "position_unchanged_hours_max": "Unchg Max",
        "loss_profit_ratio": "LPR",
        "loss_profit_ratio_w": "LPR(w)",
        "calmar_ratio": "Calmar",
        "calmar_ratio_w": "Calmar(w)",
        "omega_ratio": "Omega",
        "omega_ratio_w": "Omega(w)",
        "sortino_ratio": "Sortino",
        "sortino_ratio_w": "Sortino(w)",
        "sterling_ratio": "Sterling",
        "sterling_ratio_w": "Sterling(w)",
        "rsquared": "R²",
        "equity_balance_diff_neg_max": "E/B Diff - max",
        "equity_balance_diff_neg_mean": "E/B Diff - mean",
        "equity_balance_diff_pos_max": "E/B Diff + max",
        "equity_balance_diff_pos_mean": "E/B Diff + mean",
    }

    # Step 2: Single pass to process and gather data
    for key in keys:
        keym = key.replace("lower_bound_", "") + "_max"
        myKey = key.replace("lower_bound_", "")
        if keym not in analyses_combined:
            keym = prefix + keym
            assert keym in analyses_combined, f"❌ malformed key: {keym}"

        target = config["optimize"]["limits"][key]
        current = analyses_combined[keym]
        delta = current - target

        # Determine if we expect higher or lower values for the current key
        if "gain" in keym or "rsquared" in keym or "positions_held_per_day" in keym or "mdg" in keym or "sharpe" in keym or "calmar" in keym or "omega" in keym or "sortino" in keym or "sterling" in keym:
            expect_higher = True
        else:
            expect_higher = False

        def normalize_delta(delta, current, target, expect_higher, reward_mode=False):
            eps = 1e-9

            if current < 0:
                return 1.0

            if expect_higher:
                if delta >= 0:
                    if reward_mode:
                        percent_above = delta / max(target, eps)
                        return -0.9 * min(percent_above, 1.0)# Negative = reward
                    return 0.0
                else:
                    norm = abs(delta) / max(target, eps)
                    return 0.9 * min(norm, 1.0) + 0.1
            else:
                if delta <= 0:
                    if reward_mode:
                        percent_below = abs(delta) / max(target, eps)
                        return -0.9 * min(percent_below, 1.0)
                    return 0.0
                else:
                    norm = delta / max(current, eps)
                    return 0.9 * min(norm, 1.0) + 0.1



        # Calculate normalized error based on delta and target
        weight = normalize_delta(delta,current,target,expect_higher)
        contribution = (10 ** i) * weight

        i -= 1
        modifier += contribution

        # Update min/max for contribution and modifier
        min_contribution = min(min_contribution, contribution)
        max_contribution = max(max_contribution, contribution)
        min_modifier = min(min_modifier, modifier)
        max_modifier = max(max_modifier, modifier)

        # Store the result (we'll use this data for printing later)
        results.append({
            'key': header_aliases[myKey],
            'target': target,
            'current': current,
            'delta': delta,
            'contribution': contribution,
            'modifier': modifier,
            'expect_higher': expect_higher
        })

    def value_to_color(value, min_value, max_value):
        eps = 1e-9  # prevent log10(0)

        # Ensure all values are strictly positive for log10
        value = max(value, eps)
        min_value = max(min_value, eps)
        max_value = max(max_value, min_value + eps)

        if max_value == min_value:
            norm_value = 0.5
        else:
            # Log-scale normalization
            log_min = np.log10(min_value)
            log_max = np.log10(max_value)
            log_val = np.log10(value)

            norm_value = (log_val - log_min) / (log_max - log_min)
            norm_value = np.clip(norm_value, 0, 1)

        # Interpolate between green and red
        r = int(GREEN_RGB[0] + norm_value * (RED_RGB[0] - GREEN_RGB[0]))
        g = int(GREEN_RGB[1] + norm_value * (RED_RGB[1] - GREEN_RGB[1]))
        b = int(GREEN_RGB[2] + norm_value * (RED_RGB[2] - GREEN_RGB[2]))

        return f"\033[38;2;{r};{g};{b}m"

    all_zero_contributions = all(r['contribution'] == 0.0 for r in results)

    i = len(keys) + 1

    # Step 4: Print the results with colorized values
    for result in results:
        key = result['key']
        target = result['target']
        current = result['current']
        delta = result['delta']
        # contribution = result['contribution']
        # modifier = result['modifier']
        expect_higher = result['expect_higher']

        if all_zero_contributions:
            if "gain" in key:
                contribution = (10 ** (i - 1)) * normalize_delta(result['delta'], result['current'], result['target'], result['expect_higher'], reward_mode=all_zero_contributions)
            else:
                contribution = (10 ** i) * normalize_delta(result['delta'], result['current'], result['target'], result['expect_higher'], reward_mode=all_zero_contributions)
            modifier += contribution
        else:
            contribution = result['contribution']
            modifier = result['modifier']
        # i -= 1
        # Status determination
        if delta >= 0 and expect_higher:
            status = "✅ above target"
        elif delta <= 0 and not expect_higher:
            status = "✅ below target"
        elif delta < 0 and expect_higher:
            status = "❌ below target"
        elif delta > 0 and not expect_higher:
            status = "❌ above target"

        # Decide the color based on status
        if "✅" in status:
            status_color = GREEN_RGB
        elif "❌" in status:
            status_color = RED_RGB
        else:
            status_color = CYAN  # For other cases, like "🔵 neutral" if any

        # Trim or pad the keym to 30 characters
        keym_display = (key[:12] + '...') if len(key) > 15 else f"{key:<15}"

        # Colorize current, contribution, and modifier
        current_color = CYAN  # Always cyan for current
        contribution_color = value_to_color(contribution, min_contribution, max_contribution)
        modifier_color = value_to_color(modifier, min_modifier, max_modifier)


        # Final colorful, aligned print
        print(f"\033[38;2;{status_color[0]};{status_color[1]};{status_color[2]}m{status:<3}{RESET} "
            f"[{keym_display}] "
            f"Target: {target:>10.5f}, "
            f"Current: {current_color}{current:>10.5f}{RESET}, "
            f"Δ: {delta:+10.5f}, "
            f"Contribution: {contribution_color}{contribution:>12.5e}{RESET}, "
            f"Modifier: {modifier_color}{modifier:>12.5e}{RESET}")


    drawdown = analyses_combined.get(f"{prefix}drawdown_worst_max", 0)
    equity_diff = analyses_combined.get(f"{prefix}equity_balance_diff_neg_max_max", 0)

    if drawdown >= 1.0 or equity_diff >= 1.0:
        if verbose:
            print(f"⚠️ Drawdown or Equity cap hit! Drawdown: {drawdown:.2f}, Equity Diff: {equity_diff:.2f}")
        w_0 = w_1 = modifier
    else:
        scoring_keys = config["optimize"]["scoring"]
        assert len(scoring_keys) == 2, f"❌ Expected 2 fitness scoring keys, got {len(scoring_keys)}"

        scores = []
        for sk in scoring_keys:
            skm = f"{sk}_mean"
            if skm not in analyses_combined:
                skm = prefix + skm
                if skm not in analyses_combined:
                    raise Exception(f"❌ Invalid scoring key: {sk}")

            # score_value = modifier - analyses_combined[skm]
            score_value = modifier
            scores.append(score_value)

            if verbose:
                print(f"🎯 [{skm}] Modifier: {modifier:.5e}, Value: {analyses_combined[skm]:.5f}, Score: {score_value:.5e}")

        # if verbose:
        #     print(f"🥇 Final Scores: {scores[0]:.5f}, {scores[1]:.5f}")

        return scores[0], scores[1]

    if verbose:
        print(f"🥇 Final Equal Scores (penalized): {w_0:.5f}, {w_1:.5f}")
    return w_0, w_1

def evaluate(individual, allParams, overrides_list=[]):
    config = individual_to_config(
        individual, optimizer_overrides, overrides_list, template=allParams["config"]
    )
    analyses = {}
    for exchange in allParams["exchanges"]:
        bot_params, _, _ = prep_backtest_args(
            config,
            [],
            exchange,
            exchange_params=allParams["exchange_params"][exchange],
            backtest_params=allParams["backtest_params"][exchange],
        )
        fills, equities_usd, equities_btc, analysis_usd, analysis_btc = pbr.run_backtest(
            allParams["shared_memory_files"][exchange],
            allParams["hlcvs_shapes"][exchange],
            allParams["hlcvs_dtypes"][exchange].str,
            allParams["btc_usd_shared_memory_files"][exchange],  # Pass BTC/USD shared memory file
            allParams["btc_usd_dtypes"][exchange].str,  # Pass BTC/USD dtype
            bot_params,
            allParams["exchange_params"][exchange],
            allParams["backtest_params"][exchange],
        )
        analyses[exchange] = expand_analysis(analysis_usd, analysis_btc, fills, config)

    analyses_combined = combine_analyses(analyses)
    w_0, w_1 = calc_fitness(config,analyses_combined)
    analyses_combined.update({"w_0": w_0, "w_1": w_1})

    data = {
        **config,
        **{
            "analyses_combined": analyses_combined,
            "analyses": analyses,
        },
    }
    append_dict_to_file_safely_nonblocking(data,"results.txt")
    return w_0, w_1

def append_dict_to_file_safely_nonblocking(data: dict, filepath: str):
    with open(filepath, "a") as f:
        try:
            # Try to acquire the lock without blocking
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            f.write(json.dumps(data) + "\n")
            f.flush()
        except BlockingIOError:
            print("File is locked by another process. Retrying...")
            time.sleep(0.1)  # Wait for a short time and then try again
            append_dict_to_file_safely_nonblocking(data, filepath)
        finally:
            # Release the lock if it was acquired
            fcntl.flock(f, fcntl.LOCK_UN)


def combine_analyses(analyses):
    analyses_combined = {}
    keys = analyses[next(iter(analyses))].keys()
    for key in keys:
        values = [analysis[key] for analysis in analyses.values()]
        if not values or any([x == np.inf for x in values]) or any([x is None for x in values]):
            analyses_combined[f"{key}_mean"] = 0.0
            analyses_combined[f"{key}_min"] = 0.0
            analyses_combined[f"{key}_max"] = 0.0
            analyses_combined[f"{key}_std"] = 0.0
        else:
            try:
                analyses_combined[f"{key}_mean"] = np.mean(values)
                analyses_combined[f"{key}_min"] = np.min(values)
                analyses_combined[f"{key}_max"] = np.max(values)
                analyses_combined[f"{key}_std"] = np.std(values)
            except Exception as e:
                print("\n\n debug\n\n")
                print("key, values", key, values)
                print(e)
                traceback.print_exc()
                raise
    return analyses_combined
