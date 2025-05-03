import os
import shutil
import sys
import passivbot_rust as pbr
import asyncio
import argparse
import multiprocessing
import subprocess
import mmap
from multiprocessing import Queue, Process
from collections import defaultdict
from backtest import (
    prepare_hlcvs_mss,
    prep_backtest_args,
    expand_analysis,
)
from pure_funcs import (
    get_template_live_config,
    symbol_to_coin,
    ts_to_date_utc,
    denumpyize,
    sort_dict_keys,
    calc_hash,
    flatten,
    date_to_ts,
)
from procedures import (
    make_get_filepath,
    utc_ms,
    load_hjson_config,
    load_config,
    format_config,
    add_arguments_recursively,
    update_config_with_args,
)
from downloader import add_all_eligible_coins_to_config
from copy import deepcopy
from main import manage_rust_compilation
import numpy as np
from uuid import uuid4
import logging
import traceback
import json
import pprint
from deap import base, creator, tools, algorithms
from contextlib import contextmanager
import tempfile
import time
import fcntl
from tqdm import tqdm
import dictdiffer
from optimizer_overrides import optimizer_overrides
import fcntl
import dill as pickle

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

def make_json_serializable(obj):
    """
    Recursively convert tuples in the object to lists to make it JSON serializable.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return [make_json_serializable(e) for e in obj]
    elif isinstance(obj, list):
        return [make_json_serializable(e) for e in obj]
    else:
        return obj


def results_writer_process(queue: Queue, results_filename: str, compress=True):
    """
    Manager process that handles writing results to file.
    Runs in a separate process and receives results through a queue.
    Applies diffing to the entire data dictionary.
    """
    prev_data = None  # Initialize previous data as None
    try:
        while True:
            data = queue.get()
            if data == "DONE":  # Sentinel value to signal shutdown
                break
            try:
                if prev_data is None or not compress:
                    # First data entry or compression disabled, write full data
                    output_data = data
                else:
                    # Compute diff of the entire data dictionary
                    diff = list(dictdiffer.diff(prev_data, data))
                    for i in range(len(diff)):
                        if diff[i][0] == "change":
                            diff[i] = [diff[i][1], diff[i][2][1]]
                    output_data = {"diff": make_json_serializable(diff)}

                prev_data = data

                # Write to disk
                with open(results_filename, "a") as f:
                    json.dump(denumpyize(output_data), f)
                    f.write("\n")
            except Exception as e:
                logging.error(f"Error writing results: {e}")
    except Exception as e:
        logging.error(f"Results writer process error: {e}")


def create_shared_memory_file(hlcvs):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    logging.info(f"Creating shared memory file: {temp_file.name}...")
    shared_memory_file = temp_file.name
    temp_file.close()

    try:
        total_size = hlcvs.nbytes
        chunk_size = 1024 * 1024  # 1 MB chunks
        hlcvs_bytes = hlcvs.tobytes()

        with open(shared_memory_file, "wb") as f:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Writing to shared memory"
            ) as pbar:
                for i in range(0, len(hlcvs_bytes), chunk_size):
                    chunk = hlcvs_bytes[i : i + chunk_size]
                    f.write(chunk)
                    pbar.update(len(chunk))

    except IOError as e:
        logging.error(f"Error writing to shared memory file: {e}")
        raise
    logging.info(f"Done creating shared memory file")
    return shared_memory_file


def check_disk_space(path, required_space):
    total, used, free = shutil.disk_usage(path)
    logging.info(
        f"Disk space - Total: {total/(1024**3):.2f} GB, Used: {used/(1024**3):.2f} GB, Free: {free/(1024**3):.2f} GB"
    )
    if free < required_space:
        raise IOError(
            f"Not enough disk space. Required: {required_space/(1024**3):.2f} GB, Available: {free/(1024**3):.2f} GB"
        )


def mutPolynomialBoundedWrapper(individual, eta, low, up, indpb):
    """
    A wrapper around DEAP's mutPolynomialBounded function to pre-process
    bounds and handle the case where lower and upper bounds may be equal.

    Args:
        individual: Sequence individual to be mutated.
        eta: Crowding degree of the mutation.
        low: A value or sequence of values that is the lower bound of the search space.
        up: A value or sequence of values that is the upper bound of the search space.
        indpb: Independent probability for each attribute to be mutated.

    Returns:
        A tuple of one individual, mutated with consideration for equal lower and upper bounds.
    """
    # Convert low and up to numpy arrays for easier manipulation
    low_array = np.array(low)
    up_array = np.array(up)

    # Identify dimensions where lower and upper bounds are equal
    equal_bounds_mask = low_array == up_array

    # Temporarily adjust bounds for those dimensions
    # This adjustment is arbitrary and won't affect the outcome since the mutation
    # won't be effective in these dimensions
    temp_low = np.where(equal_bounds_mask, low_array - 1e-6, low_array)
    temp_up = np.where(equal_bounds_mask, up_array + 1e-6, up_array)

    # Call the original mutPolynomialBounded function with the temporarily adjusted bounds
    tools.mutPolynomialBounded(individual, eta, list(temp_low), list(temp_up), indpb)

    # Reset values in dimensions with originally equal bounds to ensure they remain unchanged
    for i, equal in enumerate(equal_bounds_mask):
        if equal:
            individual[i] = low[i]

    return (individual,)


def cxSimulatedBinaryBoundedWrapper(ind1, ind2, eta, low, up):
    """
    A wrapper around DEAP's cxSimulatedBinaryBounded function to pre-process
    bounds and handle the case where lower and upper bounds may be equal.

    Args:
        ind1: The first individual participating in the crossover.
        ind2: The second individual participating in the crossover.
        eta: Crowding degree of the crossover.
        low: A value or sequence of values that is the lower bound of the search space.
        up: A value or sequence of values that is the upper bound of the search space.

    Returns:
        A tuple of two individuals after crossover operation.
    """
    # Convert low and up to numpy arrays for easier manipulation
    low_array = np.array(low)
    up_array = np.array(up)

    # Identify dimensions where lower and upper bounds are equal
    equal_bounds_mask = low_array == up_array

    # Temporarily adjust bounds for those dimensions to prevent division by zero
    # This adjustment is arbitrary and won't affect the outcome since the crossover
    # won't modify these dimensions
    low_array[equal_bounds_mask] -= 1e-6
    up_array[equal_bounds_mask] += 1e-6

    # Call the original cxSimulatedBinaryBounded function with adjusted bounds
    tools.cxSimulatedBinaryBounded(ind1, ind2, eta, list(low_array), list(up_array))

    # Ensure that values in dimensions with originally equal bounds are reset
    # to the bound value (since they should not be modified)
    for i, equal in enumerate(equal_bounds_mask):
        if equal:
            ind1[i] = low[i]
            ind2[i] = low[i]

    return ind1, ind2


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


def config_to_individual(config, param_bounds):
    individual = []
    keys_ignored = ["enforce_exposure_limit"]
    for pside in ["long", "short"]:
        is_enabled = (
            param_bounds[f"{pside}_n_positions"][1] > 0.0
            and param_bounds[f"{pside}_total_wallet_exposure_limit"][1] > 0.0
        )
        individual += [
            (v if is_enabled else 0.0)
            for k, v in sorted(config["bot"][pside].items())
            if k not in keys_ignored
        ]
    # adjust to bounds
    bounds = [(low, high) for low, high in param_bounds.values()]
    adjusted = [max(min(x, bounds[z][1]), bounds[z][0]) for z, x in enumerate(individual)]
    return adjusted


@contextmanager
def managed_mmap(filename, dtype, shape):
    mmap = None
    try:
        mmap = np.memmap(filename, dtype=dtype, mode="r", shape=shape)
        yield mmap
    except FileNotFoundError:
        if shutdown_event.is_set():
            yield None
        else:
            raise
    finally:
        if mmap is not None:
            del mmap


def validate_array(arr, name):
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains inf values")


class Evaluator:
    def __init__(
        self,
        shared_memory_files,
        hlcvs_shapes,
        hlcvs_dtypes,
        btc_usd_shared_memory_files,
        btc_usd_dtypes,
        msss,
        config,
    ):
        logging.info("Initializing Evaluator...")
        self.shared_memory_files = shared_memory_files
        self.hlcvs_shapes = hlcvs_shapes
        self.hlcvs_dtypes = hlcvs_dtypes
        self.btc_usd_shared_memory_files = btc_usd_shared_memory_files
        self.btc_usd_dtypes = btc_usd_dtypes
        self.msss = msss
        self.exchanges = list(shared_memory_files.keys())

        self.mmap_contexts = {}
        self.shared_hlcvs_np = {}
        self.exchange_params = {}
        self.backtest_params = {}
        for exchange in self.exchanges:
            logging.info(f"Setting up managed_mmap for {exchange}...")
            self.mmap_contexts[exchange] = managed_mmap(
                self.shared_memory_files[exchange],
                self.hlcvs_dtypes[exchange],
                self.hlcvs_shapes[exchange],
            )
            self.shared_hlcvs_np[exchange] = self.mmap_contexts[exchange].__enter__()
            _, self.exchange_params[exchange], self.backtest_params[exchange] = prep_backtest_args(
                config, self.msss[exchange], exchange
            )
            logging.info(f"mmap_context entered successfully for {exchange}.")

        self.config = config
        logging.info("Evaluator initialization complete.")

    def setResultsQueue(self,results_queue):
        self.results_queue = results_queue

    def evaluate(self, individual, overrides_list=[]):
        config = individual_to_config(
            individual, optimizer_overrides, overrides_list, template=self.config
        )
        try:
            with open("evaluator.pkl", "rb") as f:
                allParams = pickle.load(f)
        except:
            allParams = {
                "backtest_params": self.backtest_params,
                "btc_usd_dtypes": self.btc_usd_dtypes,
                "btc_usd_shared_memory_files": {'combined': '/home/myusuf/Projects/passivbot/btc_usd_tempFile'},
                "exchange_params": self.exchange_params,
                "exchanges": self.exchanges,
                "hlcvs_dtypes": self.hlcvs_dtypes,
                "hlcvs_shapes": self.hlcvs_shapes,
                "shared_memory_files": {'combined': '/home/myusuf/Projects/passivbot/tempFile'},
                "config": self.config
                }
            with open("evaluator.pkl", "wb") as f:
                pickle.dump(allParams, f)

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

        analyses_combined = self.combine_analyses(analyses)
        w_0, w_1 = self.calc_fitness(analyses_combined)
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

    def combine_analyses(self, analyses):
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

    
    
    def calc_fitness(self, analyses_combined,verbose=True):
        modifier = 0.0
        keys = self.config["optimize"]["limits"]
        i = len(keys) + 1
        # i = 5
        prefix = "btc_" if self.config["backtest"]["use_btc_collateral"] else ""

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

            target = self.config["optimize"]["limits"][key]
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
            scoring_keys = self.config["optimize"]["scoring"]
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


    def __del__(self):
        if hasattr(self, "mmap_contexts"):
            for mmap_context in self.mmap_contexts.values():
                mmap_context.__exit__(None, None, None)

    def __getstate__(self):
        # This method is called when pickling. We exclude mmap_contexts and shared_hlcvs_np
        state = self.__dict__.copy()
        del state["mmap_contexts"]
        del state["shared_hlcvs_np"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.mmap_contexts = {}
        self.shared_hlcvs_np = {}
        for exchange in self.exchanges:
            self.mmap_contexts[exchange] = managed_mmap(
                self.shared_memory_files[exchange],
                self.hlcvs_dtypes[exchange],
                self.hlcvs_shapes[exchange],
            )
            self.shared_hlcvs_np[exchange] = self.mmap_contexts[exchange].__enter__()
            if self.shared_hlcvs_np[exchange] is None:
                print(
                    f"Warning: Unable to recreate shared memory mapping during unpickling for {exchange}."
                )


def add_extra_options(parser):
    parser.add_argument(
        "-t",
        "--start",
        type=str,
        required=False,
        dest="starting_configs",
        default=None,
        help="Start with given live configs. Single json file or dir with multiple json files",
    )


def extract_configs(path):
    cfgs = []
    if os.path.exists(path):
        if path.endswith("_all_results.txt"):
            logging.info(f"Skipping {path}")
            return []
        if path.endswith(".json"):
            try:
                cfgs.append(load_config(path, verbose=False))
                return cfgs
            except:
                return []
        if path.endswith("_pareto.txt"):
            with open(path) as f:
                for line in f.readlines():
                    try:
                        cfg = json.loads(line)
                        cfgs.append(format_config(cfg, verbose=False))
                    except Exception as e:
                        logging.error(f"Failed to load starting config {line} {e}")
    return cfgs


def get_starting_configs(starting_configs: str):
    if starting_configs is None:
        return []
    if os.path.isdir(starting_configs):
        return flatten(
            [
                get_starting_configs(os.path.join(starting_configs, f))
                for f in os.listdir(starting_configs)
            ]
        )
    return extract_configs(starting_configs)


def configs_to_individuals(cfgs, param_bounds):
    inds = {}
    for cfg in cfgs:
        try:
            fcfg = format_config(cfg, verbose=False)
            individual = config_to_individual(fcfg, param_bounds)
            inds[calc_hash(individual)] = individual
            # add duplicate of config, but with lowered total wallet exposure limit
            fcfg2 = deepcopy(fcfg)
            for pside in ["long", "short"]:
                fcfg2["bot"][pside]["total_wallet_exposure_limit"] *= 0.75
            individual2 = config_to_individual(fcfg2, param_bounds)
            inds[calc_hash(individual2)] = individual2
        except Exception as e:
            logging.error(f"error loading starting config: {e}")
    return list(inds.values())

from types import SimpleNamespace

async def initEvaluator(config_path: str = None, **kwargs):
    manage_rust_compilation()

    # Prepare template config
    template_config = get_template_live_config("v7")
    del template_config["bot"]
    keep_live_keys = {
        "approved_coins",
        "minimum_coin_age_days",
        "ohlcv_rolling_window",
        "relative_volume_filter_clip_pct",
    }
    for key in sorted(template_config["live"]):
        if key not in keep_live_keys:
            del template_config["live"][key]

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Load config
    if config_path is None:
        logging.info(f"loading default template config configs/template.json")
        config = load_config("configs/template.json", verbose=False)
    else:
        logging.info(f"loading config {config_path}")
        config = load_config(config_path, verbose=False)

    old_config = deepcopy(config)

    # Apply overrides via kwargs if provided
    if kwargs:
        # Convert kwargs to a SimpleNamespace to simulate argparse.Namespace
        arg_namespace = SimpleNamespace(**kwargs)
        update_config_with_args(config, arg_namespace)

    config = format_config(config, verbose=False)
    await add_all_eligible_coins_to_config(config)
    # Prepare data for each exchange
    hlcvs_dict = {}
    shared_memory_files = {}
    hlcvs_shapes = {}
    hlcvs_dtypes = {}
    msss = {}

    # NEW: Store per-exchange BTC arrays in a dict,
    # and store their shared-memory file names in another dict.
    btc_usd_data_dict = {}
    btc_usd_shared_memory_files = {}
    btc_usd_dtypes = {}

    config["backtest"]["coins"] = {}
    if config["backtest"]["combine_ohlcvs"]:
        exchange = "combined"
        coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices = await prepare_hlcvs_mss(
            config, exchange
        )
        exchange_preference = defaultdict(list)
        for coin in coins:
            exchange_preference[mss[coin]["exchange"]].append(coin)
        for ex in exchange_preference:
            logging.info(f"chose {ex} for {','.join(exchange_preference[ex])}")
        config["backtest"]["coins"][exchange] = coins
        hlcvs_dict[exchange] = hlcvs
        hlcvs_shapes[exchange] = hlcvs.shape
        hlcvs_dtypes[exchange] = hlcvs.dtype
        msss[exchange] = mss
        required_space = hlcvs.nbytes * 1.1  # Add 10% buffer
        check_disk_space(tempfile.gettempdir(), required_space)
        logging.info(f"Starting to create shared memory file for {exchange}...")
        validate_array(hlcvs, "hlcvs")
        shared_memory_file = create_shared_memory_file(hlcvs)
        shared_memory_files[exchange] = shared_memory_file
        if config["backtest"].get("use_btc_collateral", False):
            # Use the fetched array
            btc_usd_data_dict[exchange] = btc_usd_prices
        else:
            # Fall back to all ones
            btc_usd_data_dict[exchange] = np.ones(hlcvs.shape[0], dtype=np.float64)
        validate_array(btc_usd_data_dict[exchange], f"btc_usd_data for {exchange}")
        btc_usd_shared_memory_files[exchange] = create_shared_memory_file(
            btc_usd_data_dict[exchange]
        )
        btc_usd_dtypes[exchange] = btc_usd_data_dict[exchange].dtype
        logging.info(f"Finished creating shared memory file for {exchange}: {shared_memory_file}")
    else:
        tasks = {}
        for exchange in config["backtest"]["exchanges"]:
            tasks[exchange] = asyncio.create_task(prepare_hlcvs_mss(config, exchange))
        for exchange in config["backtest"]["exchanges"]:
            coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices = await tasks[exchange]
            config["backtest"]["coins"][exchange] = coins
            hlcvs_dict[exchange] = hlcvs
            hlcvs_shapes[exchange] = hlcvs.shape
            hlcvs_dtypes[exchange] = hlcvs.dtype
            msss[exchange] = mss
            required_space = hlcvs.nbytes * 1.1  # Add 10% buffer
            check_disk_space(tempfile.gettempdir(), required_space)
            logging.info(f"Starting to create shared memory file for {exchange}...")
            validate_array(hlcvs, "hlcvs")
            shared_memory_file = create_shared_memory_file(hlcvs)
            shared_memory_files[exchange] = shared_memory_file
            # Create the BTC array for this exchange
            if config["backtest"].get("use_btc_collateral", False):
                btc_usd_data_dict[exchange] = btc_usd_prices
            else:
                btc_usd_data_dict[exchange] = np.ones(hlcvs.shape[0], dtype=np.float64)

            validate_array(btc_usd_data_dict[exchange], f"btc_usd_data for {exchange}")
            btc_usd_shared_memory_files[exchange] = create_shared_memory_file(
                btc_usd_data_dict[exchange]
            )
            btc_usd_dtypes[exchange] = btc_usd_data_dict[exchange].dtype
            logging.info(
                f"Finished creating shared memory file for {exchange}: {shared_memory_file}"
            )
    exchanges = config["backtest"]["exchanges"]
    exchanges_fname = "combined" if config["backtest"]["combine_ohlcvs"] else "_".join(exchanges)
    date_fname = ts_to_date_utc(utc_ms())[:19].replace(":", "_")
    coins = sorted(set([x for y in config["backtest"]["coins"].values() for x in y]))
    coins_fname = "_".join(coins) if len(coins) <= 6 else f"{len(coins)}_coins"
    hash_snippet = uuid4().hex[:8]
    n_days = int(
        round(
            (
                date_to_ts(config["backtest"]["end_date"])
                - date_to_ts(config["backtest"]["start_date"])
            )
            / (1000 * 60 * 60 * 24)
        )
    )
    config["results_filename"] = make_get_filepath(
        f"optimize_results/{date_fname}_{exchanges_fname}_{n_days}days_{coins_fname}_{hash_snippet}_all_results.txt"
    )
    overrides_list = config.get("optimize", {}).get("enable_overrides", [])

    # Create results queue and start manager process
    manager = multiprocessing.Manager()
    results_queue = manager.Queue()
    writer_process = Process(
        target=results_writer_process,
        args=(results_queue, config["results_filename"]),
        kwargs={"compress": config["optimize"]["compress_results_file"]},
    )
    writer_process.start()

    # Prepare BTC/USD data
    # For optimization, use the BTC/USD prices from the first exchange (or combined)
    # Since all exchanges should align in timesteps, this should be consistent
    btc_usd_data = btc_usd_prices  # Use the fetched btc_usd_prices from prepare_hlcvs_mss
    if config["backtest"].get("use_btc_collateral", False):
        logging.info("Using fetched BTC/USD prices for collateral")
    else:
        logging.info("Using default BTC/USD prices (all 1.0s) as use_btc_collateral is False")
        btc_usd_data = np.ones(hlcvs_dict[next(iter(hlcvs_dict))].shape[0], dtype=np.float64)

    validate_array(btc_usd_data, "btc_usd_data")

    # Initialize evaluator with results queue and BTC/USD shared memory
    evaluator = Evaluator(
        shared_memory_files=shared_memory_files,
        hlcvs_shapes=hlcvs_shapes,
        hlcvs_dtypes=hlcvs_dtypes,
        # Instead of a single file/dtype, pass dictionaries
        btc_usd_shared_memory_files=btc_usd_shared_memory_files,
        btc_usd_dtypes=btc_usd_dtypes,
        msss=msss,
        config=config,
    )

    logging.info(f"Finished initializing evaluator...")
    return evaluator


async def main():
    manage_rust_compilation()
    parser = argparse.ArgumentParser(prog="optimize", description="run optimizer")
    parser.add_argument(
        "config_path", type=str, default=None, nargs="?", help="path to json passivbot config"
    )
    template_config = get_template_live_config("v7")
    del template_config["bot"]
    keep_live_keys = {
        "approved_coins",
        "minimum_coin_age_days",
        "ohlcv_rolling_window",
        "relative_volume_filter_clip_pct",
    }
    for key in sorted(template_config["live"]):
        if key not in keep_live_keys:
            del template_config["live"][key]
    add_arguments_recursively(parser, template_config)
    add_extra_options(parser)
    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    if args.config_path is None:
        logging.info(f"loading default template config configs/template.json")
        config = load_config("configs/template.json", verbose=False)
    else:
        logging.info(f"loading config {args.config_path}")
        config = load_config(args.config_path, verbose=False)
    old_config = deepcopy(config)
    update_config_with_args(config, args)
    config = format_config(config, verbose=False)
    await add_all_eligible_coins_to_config(config)
    try:
        # Prepare data for each exchange
        hlcvs_dict = {}
        shared_memory_files = {}
        hlcvs_shapes = {}
        hlcvs_dtypes = {}
        msss = {}

        # NEW: Store per-exchange BTC arrays in a dict,
        # and store their shared-memory file names in another dict.
        btc_usd_data_dict = {}
        btc_usd_shared_memory_files = {}
        btc_usd_dtypes = {}

        config["backtest"]["coins"] = {}
        if config["backtest"]["combine_ohlcvs"]:
            exchange = "combined"
            coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices = await prepare_hlcvs_mss(
                config, exchange
            )
            exchange_preference = defaultdict(list)
            for coin in coins:
                exchange_preference[mss[coin]["exchange"]].append(coin)
            for ex in exchange_preference:
                logging.info(f"chose {ex} for {','.join(exchange_preference[ex])}")
            config["backtest"]["coins"][exchange] = coins
            hlcvs_dict[exchange] = hlcvs
            hlcvs_shapes[exchange] = hlcvs.shape
            hlcvs_dtypes[exchange] = hlcvs.dtype
            msss[exchange] = mss
            required_space = hlcvs.nbytes * 1.1  # Add 10% buffer
            check_disk_space(tempfile.gettempdir(), required_space)
            logging.info(f"Starting to create shared memory file for {exchange}...")
            validate_array(hlcvs, "hlcvs")
            shared_memory_file = create_shared_memory_file(hlcvs)
            shared_memory_files[exchange] = shared_memory_file
            if config["backtest"].get("use_btc_collateral", False):
                # Use the fetched array
                btc_usd_data_dict[exchange] = btc_usd_prices
            else:
                # Fall back to all ones
                btc_usd_data_dict[exchange] = np.ones(hlcvs.shape[0], dtype=np.float64)
            validate_array(btc_usd_data_dict[exchange], f"btc_usd_data for {exchange}")
            btc_usd_shared_memory_files[exchange] = create_shared_memory_file(
                btc_usd_data_dict[exchange]
            )
            btc_usd_dtypes[exchange] = btc_usd_data_dict[exchange].dtype
            logging.info(f"Finished creating shared memory file for {exchange}: {shared_memory_file}")
        else:
            tasks = {}
            for exchange in config["backtest"]["exchanges"]:
                tasks[exchange] = asyncio.create_task(prepare_hlcvs_mss(config, exchange))
            for exchange in config["backtest"]["exchanges"]:
                coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices = await tasks[exchange]
                config["backtest"]["coins"][exchange] = coins
                hlcvs_dict[exchange] = hlcvs
                hlcvs_shapes[exchange] = hlcvs.shape
                hlcvs_dtypes[exchange] = hlcvs.dtype
                msss[exchange] = mss
                required_space = hlcvs.nbytes * 1.1  # Add 10% buffer
                check_disk_space(tempfile.gettempdir(), required_space)
                logging.info(f"Starting to create shared memory file for {exchange}...")
                validate_array(hlcvs, "hlcvs")
                shared_memory_file = create_shared_memory_file(hlcvs)
                shared_memory_files[exchange] = shared_memory_file
                # Create the BTC array for this exchange
                if config["backtest"].get("use_btc_collateral", False):
                    btc_usd_data_dict[exchange] = btc_usd_prices
                else:
                    btc_usd_data_dict[exchange] = np.ones(hlcvs.shape[0], dtype=np.float64)

                validate_array(btc_usd_data_dict[exchange], f"btc_usd_data for {exchange}")
                btc_usd_shared_memory_files[exchange] = create_shared_memory_file(
                    btc_usd_data_dict[exchange]
                )
                btc_usd_dtypes[exchange] = btc_usd_data_dict[exchange].dtype
                logging.info(
                    f"Finished creating shared memory file for {exchange}: {shared_memory_file}"
                )
        exchanges = config["backtest"]["exchanges"]
        exchanges_fname = "combined" if config["backtest"]["combine_ohlcvs"] else "_".join(exchanges)
        date_fname = ts_to_date_utc(utc_ms())[:19].replace(":", "_")
        coins = sorted(set([x for y in config["backtest"]["coins"].values() for x in y]))
        coins_fname = "_".join(coins) if len(coins) <= 6 else f"{len(coins)}_coins"
        hash_snippet = uuid4().hex[:8]
        n_days = int(
            round(
                (
                    date_to_ts(config["backtest"]["end_date"])
                    - date_to_ts(config["backtest"]["start_date"])
                )
                / (1000 * 60 * 60 * 24)
            )
        )
        config["results_filename"] = make_get_filepath(
            f"optimize_results/{date_fname}_{exchanges_fname}_{n_days}days_{coins_fname}_{hash_snippet}_all_results.txt"
        )
        overrides_list = config.get("optimize", {}).get("enable_overrides", [])

        # Create results queue and start manager process
        manager = multiprocessing.Manager()
        results_queue = manager.Queue()
        writer_process = Process(
            target=results_writer_process,
            args=(results_queue, config["results_filename"]),
            kwargs={"compress": config["optimize"]["compress_results_file"]},
        )
        writer_process.start()

        # Prepare BTC/USD data
        # For optimization, use the BTC/USD prices from the first exchange (or combined)
        # Since all exchanges should align in timesteps, this should be consistent
        btc_usd_data = btc_usd_prices  # Use the fetched btc_usd_prices from prepare_hlcvs_mss
        if config["backtest"].get("use_btc_collateral", False):
            logging.info("Using fetched BTC/USD prices for collateral")
        else:
            logging.info("Using default BTC/USD prices (all 1.0s) as use_btc_collateral is False")
            btc_usd_data = np.ones(hlcvs_dict[next(iter(hlcvs_dict))].shape[0], dtype=np.float64)

        validate_array(btc_usd_data, "btc_usd_data")
        btc_usd_shared_memory_file = create_shared_memory_file(btc_usd_data)

        # Initialize evaluator with results queue and BTC/USD shared memory
        evaluator = Evaluator(
            shared_memory_files=shared_memory_files,
            hlcvs_shapes=hlcvs_shapes,
            hlcvs_dtypes=hlcvs_dtypes,
            # Instead of a single file/dtype, pass dictionaries
            btc_usd_shared_memory_files=btc_usd_shared_memory_files,
            btc_usd_dtypes=btc_usd_dtypes,
            msss=msss,
            config=config,
        )

        logging.info(f"Finished initializing evaluator...")
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # Minimize both objectives
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        # Define parameter bounds
        param_bounds = sort_dict_keys(config["optimize"]["bounds"])
        for k, v in param_bounds.items():
            if len(v) == 1:
                param_bounds[k] = [v[0], v[0]]

        # Register attribute generators
        for i, (param_name, (low, high)) in enumerate(param_bounds.items()):
            toolbox.register(f"attr_{i}", np.random.uniform, low, high)

        def create_individual():
            return creator.Individual(
                [getattr(toolbox, f"attr_{i}")() for i in range(len(param_bounds))]
            )

        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register the evaluation function
        toolbox.register("evaluate", evaluator.evaluate, overrides_list=overrides_list)

        # Register genetic operators
        toolbox.register(
            "mate",
            cxSimulatedBinaryBoundedWrapper,
            eta=20.0,
            low=[low for low, high in param_bounds.values()],
            up=[high for low, high in param_bounds.values()],
        )
        toolbox.register(
            "mutate",
            mutPolynomialBoundedWrapper,
            eta=20.0,
            low=[low for low, high in param_bounds.values()],
            up=[high for low, high in param_bounds.values()],
            indpb=1.0 / len(param_bounds),
        )
        toolbox.register("select", tools.selNSGA2)

        # Parallelization setup
        logging.info(f"Initializing multiprocessing pool. N cpus: {config['optimize']['n_cpus']}")
        pool = multiprocessing.Pool(processes=config["optimize"]["n_cpus"])
        toolbox.register("map", pool.map)
        logging.info(f"Finished initializing multiprocessing pool.")

        # Create initial population
        logging.info(f"Creating initial population...")

        bounds = [(low, high) for low, high in param_bounds.values()]
        starting_individuals = configs_to_individuals(
            get_starting_configs(args.starting_configs), param_bounds
        )
        if (nstart := len(starting_individuals)) > (popsize := config["optimize"]["population_size"]):
            logging.info(f"Number of starting configs greater than population size.")
            logging.info(f"Increasing population size: {popsize} -> {nstart}")
            config["optimize"]["population_size"] = nstart

        population = toolbox.population(n=config["optimize"]["population_size"])
        if starting_individuals:
            bounds = [(low, high) for low, high in param_bounds.values()]
            for i in range(len(starting_individuals)):
                adjusted = [
                    max(min(x, bounds[z][1]), bounds[z][0])
                    for z, x in enumerate(starting_individuals[i])
                ]
                population[i] = creator.Individual(adjusted)

            for i in range(len(starting_individuals), len(population) // 2):
                mutant = deepcopy(population[np.random.choice(range(len(starting_individuals)))])
                toolbox.mutate(mutant)
                population[i] = mutant

        logging.info(f"Initial population size: {len(population)}")

        # Set up statistics and hall of fame
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        hof = tools.ParetoFront()

        # Run the optimization
        logging.info(f"Starting optimize...")
        population, logbook = algorithms.eaMuPlusLambda(
            population,
            toolbox,
            mu=config["optimize"]["population_size"],
            lambda_=config["optimize"]["population_size"],
            cxpb=config["optimize"]["crossover_probability"],
            mutpb=config["optimize"]["mutation_probability"],
            ngen=max(1, int(config["optimize"]["iters"] / len(population))),
            stats=stats,
            halloffame=hof,
            verbose=True,
        )

        # Print statistics
        print(logbook)

        logging.info(f"Optimization complete.")

        try:
            logging.info(f"Extracting best config...")
            result = subprocess.run(
                ["python3", "src/tools/extract_best_config.py", config["results_filename"], "-v"],
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
        except Exception as e:
            logging.error(f"failed to extract best config {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Signal the writer process to shut down and wait for it
        if "results_queue" in locals():
            results_queue.put("DONE")
            writer_process.join()
        if "pool" in locals():
            logging.info("Closing and terminating the process pool...")
            pool.close()
            pool.terminate()
            pool.join()

        # Remove shared memory files (including BTC/USD)
        if "shared_memory_files" in locals():
            for shared_memory_file in shared_memory_files.values():
                if shared_memory_file and os.path.exists(shared_memory_file):
                    logging.info(f"Removing shared memory file: {shared_memory_file}")
                    try:
                        os.unlink(shared_memory_file)
                    except Exception as e:
                        logging.error(f"Error removing shared memory file: {e}")
        if "btc_usd_shared_memory_file" in locals():
            if btc_usd_shared_memory_file and os.path.exists(btc_usd_shared_memory_file):
                logging.info(f"Removing BTC/USD shared memory file: {btc_usd_shared_memory_file}")
                try:
                    os.unlink(btc_usd_shared_memory_file)
                except Exception as e:
                    logging.error(f"Error removing BTC/USD shared memory file: {e}")

        logging.info("Cleanup complete. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
