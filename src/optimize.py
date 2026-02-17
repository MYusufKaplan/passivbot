import os
import sys
import argparse

if sys.platform.startswith("win"):
    # ==== BEGIN fcntl stub for Windows ====
    try:
        import fcntl
    except ImportError:
        # create a fake module so later `import fcntl` works without error
        class _FcntlStub:
            LOCK_EX = None
            LOCK_SH = None
            LOCK_UN = None

            def lockf(self, *args, **kwargs):
                pass

            def ioctl(self, *args, **kwargs):
                pass

        sys.modules["fcntl"] = _FcntlStub()
        fcntl = sys.modules["fcntl"]
    # ==== END fcntl stub for Windows ====

# Rust extension check before importing compiled module
from rust_utils import check_and_maybe_compile

_rust_parser = argparse.ArgumentParser(add_help=False)
_rust_parser.add_argument("--skip-rust-compile", action="store_true", help="Skip Rust build check.")
_rust_parser.add_argument(
    "--force-rust-compile", action="store_true", help="Force rebuild of Rust extension."
)
_rust_parser.add_argument(
    "--fail-on-stale-rust",
    action="store_true",
    help="Abort if Rust extension appears stale instead of attempting rebuild.",
)
_rust_known, _rust_remaining = _rust_parser.parse_known_args()
try:
    check_and_maybe_compile(
        skip=_rust_known.skip_rust_compile
        or os.environ.get("SKIP_RUST_COMPILE", "").lower() in ("1", "true", "yes"),
        force=_rust_known.force_rust_compile,
        fail_on_stale=_rust_known.fail_on_stale_rust,
    )
except Exception as exc:
    print(f"Rust extension check failed: {exc}")
    sys.exit(1)
sys.argv = [sys.argv[0]] + _rust_remaining

import passivbot_rust as pbr
from backtest import (
    prepare_hlcvs_mss,
    build_backtest_payload,
    execute_backtest,
)
import asyncio
import argparse
import multiprocessing
<<<<<<< HEAD
import subprocess
import mmap
import math
from multiprocessing import Queue, Process
=======
import signal
import time
>>>>>>> upstream/master
from collections import defaultdict
from downloader import compute_backtest_warmup_minutes, compute_per_coin_warmup_minutes
from config_utils import (
    get_template_config,
    load_hjson_config,
    load_config,
    format_config,
    add_config_arguments,
    update_config_with_args,
    recursive_config_update,
    require_config_value,
    merge_negative_cli_values,
    strip_config_metadata,
    get_optional_config_value,
)
from pure_funcs import (
    denumpyize,
    sort_dict_keys,
    calc_hash,
    flatten,
    str2bool,
)
from utils import date_to_ts, ts_to_date, utc_ms, make_get_filepath, format_approved_ignored_coins
from logging_setup import configure_logging, resolve_log_level
from copy import deepcopy
import gc
import numpy as np
from uuid import uuid4
import logging
import traceback
import json
import pprint
<<<<<<< HEAD
from deap import base, creator, tools
import alternatives
from deap_optimizer.operators import mutPolynomialBoundedWrapper, cxSimulatedBinaryBoundedWrapper
from contextlib import contextmanager
import tempfile
import time
=======

try:
    from deap import base, creator, tools, algorithms
except ImportError:  # pragma: no cover - allow import in minimal test envs

    class _DummyFitness:
        weights = ()

        def __init__(self, values=()):
            self.values = values

        def wvalues(self):
            return self.values

    class _DummyBase:
        Fitness = _DummyFitness

    class _DummyCreator:
        def create(self, *args, **kwargs):
            return None

        def __getattr__(self, name):
            raise AttributeError

    base = _DummyBase()
    creator = _DummyCreator()
    tools = algorithms = None
import math
>>>>>>> upstream/master
import fcntl
from optimizer_overrides import optimizer_overrides
<<<<<<< HEAD
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich.style import Style
=======
from opt_utils import make_json_serializable, generate_incremental_diff, round_floats, quantize_floats
from limit_utils import expand_limit_checks, compute_limit_violation
from pareto_store import ParetoStore
import msgpack
from typing import Sequence, Tuple, List, Dict, Any, Optional
from itertools import permutations
from shared_arrays import SharedArrayManager, attach_shared_array
from ohlcv_utils import align_and_aggregate_hlcvs
from optimize_suite import (
    ScenarioEvalContext,
    prepare_suite_contexts,
)
from suite_runner import (
    SuiteScenario,
    ScenarioResult,
    extract_suite_config,
    filter_scenarios_by_label,
    aggregate_metrics,
    build_suite_metrics_payload,
)
from metrics_schema import build_scenario_metrics, flatten_metric_stats
from optimization.bounds import (
    Bound,
    enforce_bounds,
)
from optimization.config_adapter import extract_bounds_tuple_list_from_config
from optimization.deap_adapters import (
    mutPolynomialBoundedWrapper,
    cxSimulatedBinaryBoundedWrapper,
)

>>>>>>> upstream/master

def _ignore_sigint_in_worker():
    """Ensure worker processes don't receive SIGINT so the parent controls shutdown."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, ValueError):
        pass


class ConstraintAwareFitness(base.Fitness):
    constraint_violation: float = 0.0

<<<<<<< HEAD
    try:
        total_size = hlcvs.nbytes
        chunk_size = 1024 * 1024  # 1 MB chunks
        hlcvs_bytes = hlcvs.tobytes()

        with open(shared_memory_file, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="Writing to shared memory",
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
        f"Disk space - Total: {total / (1024**3):.2f} GB, Used: {used / (1024**3):.2f} GB, Free: {free / (1024**3):.2f} GB"
    )
    if free < required_space:
        raise IOError(
            f"Not enough disk space. Required: {required_space / (1024**3):.2f} GB, Available: {free / (1024**3):.2f} GB"
        )


def individual_to_config(
    individual, optimizer_overrides, overrides_list, template=None
):
    if template is None:
        template = get_template_live_config("v7")
    keys_ignored = ["enforce_exposure_limit"]
    config = deepcopy(template)
    keys = [k for k in sorted(config["bot"]["long"]) if k not in keys_ignored]
=======
    def dominates(self, other, obj=slice(None)):
        self_violation = getattr(self, "constraint_violation", 0.0)
        other_violation = getattr(other, "constraint_violation", 0.0)
        if math.isclose(self_violation, other_violation, rel_tol=0.0, abs_tol=1e-12):
            return super().dominates(other, obj)
        return self_violation < other_violation


def _apply_config_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    if not overrides:
        return
    for dotted_path, value in overrides.items():
        if not isinstance(dotted_path, str):
            continue
        parts = dotted_path.split(".")
        if not parts:
            continue
        target = config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value


_BOOL_LITERALS = {"1", "0", "true", "false", "t", "f", "yes", "no", "y", "n"}


def _looks_like_bool_token(value: str) -> bool:
    return value.lower() in _BOOL_LITERALS


def _normalize_optional_bool_flag(argv: list[str], flag: str) -> list[str]:
    result: list[str] = []
>>>>>>> upstream/master
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == flag:
            next_token = argv[i + 1] if i + 1 < len(argv) else None
            if (
                next_token
                and not next_token.startswith("-")
                and not _looks_like_bool_token(next_token)
            ):
                result.append(f"{flag}=true")
                i += 1
                continue
        result.append(token)
        i += 1
    return result


def _maybe_aggregate_backtest_data(hlcvs, timestamps, btc_usd_prices, mss, config):
    candle_interval = int(config.get("backtest", {}).get("candle_interval_minutes", 1) or 1)
    if candle_interval <= 1:
        return hlcvs, timestamps, btc_usd_prices
    n_before = hlcvs.shape[0]
    hlcvs, timestamps, btc_usd_prices, offset_bars = align_and_aggregate_hlcvs(
        hlcvs, timestamps, btc_usd_prices, candle_interval
    )
    logging.debug(
        "[optimize] aggregated %dm candles: %d bars -> %d bars (trimmed %d for alignment)",
        candle_interval, n_before, hlcvs.shape[0], offset_bars,
    )
    meta = mss.setdefault("__meta__", {})
    meta["data_interval_minutes"] = candle_interval
    meta["candle_interval_offset_bars"] = int(offset_bars)
    if timestamps is not None and len(timestamps) > 0:
        meta["effective_start_ts"] = int(timestamps[0])
        meta["effective_start_date"] = ts_to_date(int(timestamps[0]))
    return hlcvs, timestamps, btc_usd_prices


class ResultRecorder:
    def __init__(
        self,
        *,
        results_dir: str,
        sig_digits: int,
        flush_interval: int,
        scoring_keys: Sequence[str],
        compress: bool,
        write_all_results: bool,
        pareto_max_size: int = 300,
        bounds: Optional[Sequence[Bound]] = None,
    ):
        self.store = ParetoStore(
            directory=results_dir,
            sig_digits=sig_digits,
            bounds=bounds,
            flush_interval=flush_interval,
            log_name="optimizer.pareto",
            max_size=pareto_max_size,
        )
        self.write_all = write_all_results
        self.compress = compress
        self.results_file = None
        self.packer = None
        if self.write_all:
            filename = os.path.join(results_dir, "all_results.bin")
            self.results_file = open(filename, "ab")
            self.packer = msgpack.Packer(use_bin_type=True)
        self.prev_data = None
        self.counter = 0
        self.scoring_keys = list(scoring_keys)

    def record(self, data: dict) -> None:
        if self.write_all and self.results_file:
            if self.compress:
                if self.prev_data is None or self.counter % 100 == 0:
                    output_data = make_json_serializable(data)
                else:
                    diff = generate_incremental_diff(self.prev_data, data)
                    output_data = make_json_serializable(diff)
                self.counter += 1
                self.prev_data = data
            else:
                output_data = data
            try:
                self.results_file.write(self.packer.pack(output_data))
                self.results_file.flush()
            except Exception as exc:
                logging.error(f"Error writing results: {exc}")
        metrics_block = data.get("metrics", {}) or {}
        violation = metrics_block.get("constraint_violation")
        try:
            updated = self.store.add_entry(data)
        except Exception as exc:
            logging.error(f"ParetoStore error: {exc}")
        else:
            if updated:
                objectives_block = metrics_block.get("objectives", {})
                objective_values = [
                    objectives_block[key]
                    for key in sorted(objectives_block)
                    if objectives_block.get(key) is not None
                ]
                violation_str = (
                    f" | constraint={pbr.round_dynamic(violation, 3)}"
                    if isinstance(violation, (int, float))
                    else ""
                )
                logging.info(
                    "Pareto update | eval=%d | front=%d | objectives=%s%s",
                    self.store.n_iters,
                    len(self.store._front),
                    _format_objectives(objective_values),
                    violation_str,
                )

    def flush(self) -> None:
        self.store.flush_now()

    def close(self) -> None:
        if self.results_file:
            self.results_file.close()


logging.basicConfig(
    format="%(asctime)s %(processName)-12s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)


TEMPLATE_CONFIG_MODE = "v7"


def _format_objectives(values: Sequence[float]) -> str:
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not values:
        return "[]"
    return "[" + ", ".join(f"{float(v):.3g}" for v in values) + "]"


def _record_individual_result(individual, evaluator_config, overrides_list, recorder):
    metrics = getattr(individual, "evaluation_metrics", {}) or {}
    suite_metrics = metrics.pop("suite_metrics", None)
    config = individual_to_config(individual, optimizer_overrides, overrides_list, evaluator_config)
    entry = dict(config)
    if suite_metrics is not None:
        entry["suite_metrics"] = suite_metrics
        bt = entry.get("backtest")
        if isinstance(bt, dict):
            bt.pop("coins", None)
    if metrics:
        if "constraint_violation" not in metrics:
            violation = getattr(individual, "constraint_violation", None)
            if violation is not None:
                metrics["constraint_violation"] = violation
        entry["metrics"] = metrics
    entry = strip_config_metadata(entry)
    recorder.record(entry)
    if hasattr(individual, "evaluation_metrics"):
        del individual.evaluation_metrics


def ea_mu_plus_lambda_stream(
    population,
    toolbox,
    mu,
    lambda_,
    cxpb,
    mutpb,
    ngen,
    stats,
    halloffame,
    verbose,
    recorder,
    evaluator_config,
    overrides_list,
    pool,
    duplicate_counter,
    pool_state,
):
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max"

    start_time = time.time()
    total_evals = 0

    def evaluate_and_record(individuals):
        nonlocal total_evals
        if not individuals:
            return 0
        logging.debug("Evaluating %d candidates", len(individuals))
        pending = {}
        for idx, ind in enumerate(individuals):
            pending[pool.apply_async(toolbox.evaluate, (ind,))] = idx

        completed = 0
        try:
            while pending:
                ready = [res for res in pending if res.ready()]
                if not ready:
                    time.sleep(0.1)
                    continue
                for res in ready:
                    idx = pending.pop(res)
                    fit_values, penalty, metrics = res.get()
                    ind = individuals[idx]
                    ind.fitness.values = fit_values
                    ind.fitness.constraint_violation = penalty
                    ind.constraint_violation = penalty
                    if metrics and isinstance(metrics, dict):
                        suite = metrics.get("suite_metrics", {}) or {}
                        metric_map = suite.get("metrics", {}) or {}
                        adg_entry = metric_map.get("adg_pnl", {}) or {}
                        prh_entry = metric_map.get("peak_recovery_hours_pnl", {}) or {}
                        logging.debug(
                            "Eval metrics | idx=%d adg_pnl=%s peak_recovery_hours_pnl=%s",
                            idx,
                            adg_entry.get("aggregated"),
                            prh_entry.get("aggregated"),
                        )
                        scenario_labels = suite.get("scenario_labels") or []
                        if not scenario_labels and isinstance(adg_entry, dict):
                            scenario_labels = list((adg_entry.get("scenarios") or {}).keys())
                        for label in scenario_labels:
                            adg_val = (adg_entry.get("scenarios") or {}).get(label)
                            prh_val = (prh_entry.get("scenarios") or {}).get(label)
                            logging.debug(
                                "Eval metrics scenario | idx=%d label=%s adg_pnl=%s peak_recovery_hours_pnl=%s",
                                idx,
                                label,
                                adg_val,
                                prh_val,
                            )
                    if metrics is not None:
                        ind.evaluation_metrics = metrics
                        _record_individual_result(ind, evaluator_config, overrides_list, recorder)
                    elif hasattr(ind, "evaluation_metrics"):
                        delattr(ind, "evaluation_metrics")
                    completed += 1
        except KeyboardInterrupt:
            logging.info("Evaluation interrupted; terminating pending tasks...")
            for res in pending:
                try:
                    res.cancel()
                except Exception:
                    pass
            if not pool_state["terminated"]:
                logging.info("Terminating worker pool immediately due to interrupt...")
                pool.terminate()
                pool_state["terminated"] = True
            raise

        total_evals += completed
        return completed

    dup_prev_total = 0
    dup_prev_resolved = 0
    dup_prev_reused = 0

    def log_generation(gen, nevals, record):
        nonlocal dup_prev_total, dup_prev_resolved, dup_prev_reused
        best = record.get("min") if record else None
        front_size = len(halloffame) if halloffame is not None else 0
        dup_tot = duplicate_counter["total"]
        dup_res = duplicate_counter["resolved"]
        dup_reuse = duplicate_counter["reused"]
        dup_ratio = (dup_tot / total_evals) if total_evals else 0.0
        dup_delta = dup_tot - dup_prev_total
        dup_res_delta = dup_res - dup_prev_resolved
        dup_reuse_delta = dup_reuse - dup_prev_reused
        dup_gen_ratio = (dup_delta / nevals) if nevals else 0.0
        logging.info(
            (
                "Gen %d complete | evals=%d | total=%d | front=%d | best=%s | "
                "dups=%d (resolved=%d reused=%d) | dup_delta=%d (res=%d reuse=%d) | "
                "dup_ratio=%.2f%% | dup_gen=%.2f%% | elapsed=%.1fs"
            ),
            gen,
            nevals,
            total_evals,
            front_size,
            _format_objectives(best),
            dup_tot,
            dup_res,
            dup_reuse,
            dup_delta,
            dup_res_delta,
            dup_reuse_delta,
            dup_ratio * 100.0,
            dup_gen_ratio * 100.0,
            time.time() - start_time,
        )
        dup_prev_total = dup_tot
        dup_prev_resolved = dup_res
        dup_prev_reused = dup_reuse
        if verbose and record:
            logging.debug("Logbook: %s", " ".join(f"{k}={v}" for k, v in record.items()))

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    if invalid_ind:
        logging.info("Evaluating initial population (%d candidates)...", len(invalid_ind))
    nevals = evaluate_and_record(invalid_ind)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=nevals, **record)
    log_generation(0, nevals, record)

    if len(population) < 2:
        logging.warning(
            "Population too small for crossover/mutation (size=%d); skipping evolution steps",
            len(population),
        )
        return population, logbook

    for gen in range(1, ngen + 1):
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        nevals = evaluate_and_record(invalid_ind)

        population[:] = toolbox.select(population + offspring, mu)

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=nevals, **record)
        log_generation(gen, nevals, record)

    logging.info(
        "Optimization summary | generations=%d | total_evals=%d | front=%d | duration=%.1fs",
        ngen,
        total_evals,
        len(halloffame) if halloffame is not None else 0,
        time.time() - start_time,
    )
    return population, logbook


def individual_to_config(individual, optimizer_overrides, overrides_list, template):
    """
    assume individual is already bound enforced (or will be after)
    """
    config = deepcopy(template)
    i = 0
    for pside in sorted(config["bot"]):
        for key in sorted(config["bot"][pside]):
            config["bot"][pside][key] = individual[i]
            i += 1
        config = optimizer_overrides(overrides_list, config, pside)

    return config


<<<<<<< HEAD
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
    adjusted = [
        max(min(x, bounds[z][1]), bounds[z][0]) for z, x in enumerate(individual)
    ]
    return adjusted
=======
def config_to_individual(config, bounds, sig_digits=None):
    return enforce_bounds(
        [
            config["bot"][pside][key]
            for pside in sorted(config["bot"])
            for key in sorted(config["bot"][pside])
        ],
        bounds,
        sig_digits,
    )
>>>>>>> upstream/master


def validate_array(arr, name, allow_nan=True):
    if not allow_nan and np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values")
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains inf values")
    if allow_nan and np.isnan(arr).all():
        raise ValueError(f"{name} is entirely NaN")


class Evaluator:
    def __init__(
        self,
        hlcvs_specs,
        btc_usd_specs,
        msss,
        config,
        seen_hashes=None,
        duplicate_counter=None,
        timestamps=None,
        shared_array_manager: SharedArrayManager | None = None,
    ):
        logging.debug("Initializing Evaluator...")
        self.hlcvs_specs = hlcvs_specs
        self.btc_usd_specs = btc_usd_specs
        self.msss = msss
        self.timestamps = timestamps or {}
        self.exchanges = list(hlcvs_specs.keys())
        self.shared_array_manager = shared_array_manager
        self.shared_hlcvs_np = {}
        self.shared_btc_np = {}
        self._attachments = {"hlcvs": {}, "btc": {}}

        for exchange in self.exchanges:
<<<<<<< HEAD
            logging.info(f"Setting up managed_mmap for {exchange}...")
            self.mmap_contexts[exchange] = managed_mmap(
                self.shared_memory_files[exchange],
                self.hlcvs_dtypes[exchange],
                self.hlcvs_shapes[exchange],
            )
            self.shared_hlcvs_np[exchange] = self.mmap_contexts[exchange].__enter__()
            _, self.exchange_params[exchange], self.backtest_params[exchange] = (
                prep_backtest_args(config, self.msss[exchange], exchange)
            )
            logging.info(f"mmap_context entered successfully for {exchange}.")

        self.config = config
        
        logging.info("Evaluator initialization complete.")
        self.results_queue = results_queue

    def evaluate(self, individual, overrides_list=[]):
        """
        Evaluate an individual strategy on the full dataset.
        """
        config = individual_to_config(
            individual, optimizer_overrides, overrides_list, template=self.config
        )
=======
            logging.debug("Preparing cached parameters for %s...", exchange)
            if self.shared_array_manager is not None:
                self.shared_hlcvs_np[exchange] = self.shared_array_manager.view(
                    self.hlcvs_specs[exchange]
                )
                btc_spec = self.btc_usd_specs.get(exchange)
                if btc_spec is not None:
                    self.shared_btc_np[exchange] = self.shared_array_manager.view(btc_spec)

        self.config = config
        logging.debug("Evaluator initialization complete.")
        logging.info("Evaluator ready | exchanges=%d", len(self.exchanges))
        self.seen_hashes = seen_hashes if seen_hashes is not None else {}
        self.duplicate_counter = duplicate_counter if duplicate_counter is not None else {"count": 0}
        self.bounds = extract_bounds_tuple_list_from_config(self.config)
        self.sig_digits = config.get("optimize", {}).get("round_to_n_significant_digits", 6)

        shared_metric_weights = {
            "positions_held_per_day": 1.0,
            "positions_held_per_day_w": 1.0,
            "position_held_hours_mean": 1.0,
            "position_held_hours_max": 1.0,
            "position_held_hours_median": 1.0,
            "position_unchanged_hours_max": 1.0,
            "high_exposure_hours_mean_long": 1.0,
            "high_exposure_hours_max_long": 1.0,
            "high_exposure_hours_mean_short": 1.0,
            "high_exposure_hours_max_short": 1.0,
            "adg_pnl": -1.0,
            "adg_pnl_w": -1.0,
            "mdg_pnl": -1.0,
            "mdg_pnl_w": -1.0,
            "sharpe_ratio_pnl": -1.0,
            "sharpe_ratio_pnl_w": -1.0,
            "sortino_ratio_pnl": -1.0,
            "sortino_ratio_pnl_w": -1.0,
        }

        currency_metric_weights = {
            "adg": -1.0,
            "adg_per_exposure_long": -1.0,
            "adg_per_exposure_short": -1.0,
            "adg_w": -1.0,
            "adg_w_per_exposure_long": -1.0,
            "adg_w_per_exposure_short": -1.0,
            "calmar_ratio": -1.0,
            "calmar_ratio_w": -1.0,
            "drawdown_worst": 1.0,
            "drawdown_worst_mean_1pct": 1.0,
            "equity_balance_diff_neg_max": 1.0,
            "equity_balance_diff_neg_mean": 1.0,
            "equity_balance_diff_pos_max": 1.0,
            "equity_balance_diff_pos_mean": 1.0,
            "equity_choppiness": 1.0,
            "equity_choppiness_w": 1.0,
            "equity_jerkiness": 1.0,
            "equity_jerkiness_w": 1.0,
            "peak_recovery_hours_equity": 1.0,
            "expected_shortfall_1pct": 1.0,
            "exponential_fit_error": 1.0,
            "exponential_fit_error_w": 1.0,
            "gain": -1.0,
            "gain_per_exposure_long": -1.0,
            "gain_per_exposure_short": -1.0,
            "loss_profit_ratio": 1.0,
            "loss_profit_ratio_w": 1.0,
            "mdg": -1.0,
            "mdg_per_exposure_long": -1.0,
            "mdg_per_exposure_short": -1.0,
            "mdg_w": -1.0,
            "mdg_w_per_exposure_long": -1.0,
            "mdg_w_per_exposure_short": -1.0,
            "omega_ratio": -1.0,
            "omega_ratio_w": -1.0,
            "sharpe_ratio": -1.0,
            "sharpe_ratio_w": -1.0,
            "sortino_ratio": -1.0,
            "sortino_ratio_w": -1.0,
            "sterling_ratio": -1.0,
            "sterling_ratio_w": -1.0,
            "total_wallet_exposure_max": 1.0,
            "total_wallet_exposure_mean": 1.0,
            "total_wallet_exposure_median": 1.0,
            "volume_pct_per_day_avg": -1.0,
            "volume_pct_per_day_avg_w": -1.0,
            "entry_initial_balance_pct_long": -1.0,
            "entry_initial_balance_pct_short": -1.0,
        }

        self.scoring_weights = {}
        self.scoring_weights.update(shared_metric_weights)

        for metric, weight in currency_metric_weights.items():
            self.scoring_weights[f"{metric}_usd"] = weight
            self.scoring_weights[f"{metric}_btc"] = weight
            self.scoring_weights.setdefault(metric, weight)
            self.scoring_weights.setdefault(f"usd_{metric}", weight)
            self.scoring_weights.setdefault(f"btc_{metric}", weight)

        self.build_limit_checks()

    def _ensure_attached(self, exchange: str) -> None:
        if exchange not in self.shared_hlcvs_np:
            spec = self.hlcvs_specs[exchange]
            attachment = attach_shared_array(spec)
            self._attachments["hlcvs"][exchange] = attachment
            self.shared_hlcvs_np[exchange] = attachment.array
        if exchange not in self.shared_btc_np:
            btc_spec = self.btc_usd_specs.get(exchange)
            if btc_spec is not None:
                attachment = attach_shared_array(btc_spec)
                self._attachments["btc"][exchange] = attachment
                self.shared_btc_np[exchange] = attachment.array

    def perturb_step_digits(self, individual, change_chance=0.5):
        perturbed = []
        for i, val in enumerate(individual):
            if np.random.random() < change_chance:  # x% chance of leaving unchanged
                perturbed.append(val)
                continue
            bound = self.bounds[i]
            if bound.high == bound.low:
                perturbed.append(val)
                continue

            # For stepped parameters, move by the defined step
            if bound.is_stepped:
                step = bound.step
            elif val != 0.0:
                exponent = math.floor(math.log10(abs(val))) - (self.sig_digits - 1)
                step = 10**exponent
            else:
                step = (bound.high - bound.low) * 10 ** -(self.sig_digits - 1)

            direction = np.random.choice([-1.0, 1.0])
            new_val = val + step * direction
            # For stepped params, don't round_dynamic; quantization will happen in enforce_bounds
            if bound.is_stepped:
                perturbed.append(new_val)
            else:
                perturbed.append(pbr.round_dynamic(new_val, self.sig_digits))

        return perturbed

    def perturb_x_pct(self, individual, magnitude=0.01):
        perturbed = []
        for i, val in enumerate(individual):
            bound = self.bounds[i]
            if bound.high == bound.low:
                perturbed.append(val)
                continue
            new_val = val * (1 + np.random.uniform(-magnitude, magnitude))
            # For stepped params, don't round_dynamic; quantization will happen in enforce_bounds
            if bound.is_stepped:
                perturbed.append(new_val)
            else:
                perturbed.append(pbr.round_dynamic(new_val, self.sig_digits))
        return perturbed

    def perturb_random_subset(self, individual, frac=0.2):
        perturbed = list(individual)
        n = len(individual)
        indices = np.random.choice(n, max(1, int(frac * n)), replace=False)
        for i in indices:
            bound = self.bounds[i]
            if bound.low != bound.high:
                if bound.is_stepped:
                    # For stepped params, move by +/- step
                    direction = np.random.choice([-1.0, 1.0])
                    perturbed[i] = individual[i] + bound.step * direction
                else:
                    delta = (bound.high - bound.low) * 0.01
                    perturbed[i] = individual[i] + delta * np.random.uniform(-1.0, 1.0)
        return perturbed

    def perturb_sample_some(self, individual, frac=0.2):
        perturbed = list(individual)
        n = len(individual)
        indices = np.random.choice(n, max(1, int(frac * n)), replace=False)
        for i in indices:
            bound = self.bounds[i]
            if bound.low != bound.high:
                perturbed[i] = bound.random_on_grid()
        return perturbed

    def perturb_gaussian(self, individual, scale=0.01):
        perturbed = []
        for i, val in enumerate(individual):
            bound = self.bounds[i]
            if bound.high == bound.low:
                perturbed.append(val)
                continue
            if bound.is_stepped:
                # For stepped params, generate gaussian number of steps to move
                max_steps = (bound.high - bound.low) / bound.step
                n_steps = int(np.random.normal(0, scale * max_steps) + 0.5)
                perturbed.append(val + n_steps * bound.step)
            else:
                noise = np.random.normal(0, scale * (bound.high - bound.low))
                perturbed.append(val + noise)
        return perturbed

    def perturb_large_uniform(self, individual):
        perturbed = []
        for i in range(len(individual)):
            bound = self.bounds[i]
            if bound.low == bound.high:
                perturbed.append(bound.low)
            else:
                perturbed.append(bound.random_on_grid())
        return perturbed

    def evaluate(self, individual, overrides_list):
        individual[:] = enforce_bounds(individual, self.bounds, self.sig_digits)
        config = individual_to_config(individual, optimizer_overrides, overrides_list, self.config)
        individual_hash = calc_hash(individual)
        if individual_hash in self.seen_hashes:
            existing_entry = self.seen_hashes[individual_hash]
            existing_score = None
            existing_penalty = 0.0
            if existing_entry is not None:
                existing_score, existing_penalty = existing_entry
            self.duplicate_counter["total"] += 1
            perturbation_funcs = [
                self.perturb_x_pct,
                self.perturb_step_digits,
                self.perturb_gaussian,
                self.perturb_random_subset,
                self.perturb_sample_some,
                self.perturb_large_uniform,
            ]
            for perturb_fn in perturbation_funcs:
                perturbed = perturb_fn(individual)
                perturbed = enforce_bounds(perturbed, self.bounds, self.sig_digits)
                new_hash = calc_hash(perturbed)
                if new_hash not in self.seen_hashes:
                    individual[:] = perturbed
                    self.seen_hashes[new_hash] = None
                    config = individual_to_config(
                        perturbed, optimizer_overrides, overrides_list, self.config
                    )
                    self.duplicate_counter["resolved"] += 1
                    break
            else:
                if existing_score is not None:
                    self.duplicate_counter["reused"] += 1
                    return tuple(existing_score), existing_penalty, None
        else:
            self.seen_hashes[individual_hash] = None
>>>>>>> upstream/master
        analyses = {}
        for exchange in self.exchanges:
            self._ensure_attached(exchange)
            payload = build_backtest_payload(
                self.shared_hlcvs_np[exchange],
                self.msss[exchange],
                config,
                exchange,
                self.shared_btc_np[exchange],
                self.timestamps.get(exchange),
            )
<<<<<<< HEAD
            fills, equities_usd, equities_btc, analysis_usd, analysis_btc = (
                pbr.run_backtest(
                    self.shared_memory_files[exchange],
                    self.hlcvs_shapes[exchange],
                    self.hlcvs_dtypes[exchange].str,
                    self.btc_usd_shared_memory_files[
                        exchange
                    ],  # Pass BTC/USD shared memory file
                    self.btc_usd_dtypes[exchange].str,  # Pass BTC/USD dtype
                    bot_params,
                    self.exchange_params[exchange],
                    self.backtest_params[exchange],
                )
            )
            analyses[exchange] = expand_analysis(
                analysis_usd, analysis_btc, fills, config
            )
            # Store equity length for bankruptcy inference
            analyses[exchange]["equity_length"] = len(equities_usd)

        analyses_combined = self.combine_analyses(analyses)
        w_0, w_1, write_to_file = self.calc_fitness(
            analyses_combined, analyses, individual
        )
        analyses_combined.update({"w_0": w_0, "w_1": w_1})
        
        # Store last analysis for interval mode access
        self.last_analyses_combined = analyses_combined

        if write_to_file:
            data = {
                **config,
                **{
                    "analyses_combined": analyses_combined,
                    "analyses": analyses,
                },
            }
            self.results_queue.put(data)
        return w_0, w_1, not write_to_file

    def combine_analyses(self, analyses):
        analyses_combined = {}
        keys = analyses[next(iter(analyses))].keys()
        for key in keys:
            values = [analysis[key] for analysis in analyses.values()]

            # Special handling for bankruptcy_timestamp - only set if actually bankrupt
            if key == "bankruptcy_timestamp":
                # Find the first non-None bankruptcy timestamp (if any)
                bankruptcy_timestamps = [v for v in values if v is not None]
                if bankruptcy_timestamps:
                    # If any exchange went bankrupt, use the earliest bankruptcy timestamp
                    analyses_combined[f"{key}_mean"] = min(bankruptcy_timestamps)
                    analyses_combined[f"{key}_min"] = min(bankruptcy_timestamps)
                    analyses_combined[f"{key}_max"] = min(bankruptcy_timestamps)
                    analyses_combined[f"{key}_std"] = 0.0
                else:
                    # No bankruptcy occurred - keep as None
                    analyses_combined[f"{key}_mean"] = None
                    analyses_combined[f"{key}_min"] = None
                    analyses_combined[f"{key}_max"] = None
                    analyses_combined[f"{key}_std"] = None

            # Special handling for bankruptcy_reason - use max to get most severe reason
            # 0=none, 1=financial, 2=drawdown, 3=no_positions, 4=stale_position
            elif key == "bankruptcy_reason":
                non_zero_reasons = [v for v in values if v != 0]
                if non_zero_reasons:
                    # Use max to get the most severe bankruptcy reason
                    max_reason = max(non_zero_reasons)
                    analyses_combined[f"{key}_mean"] = max_reason
                    analyses_combined[f"{key}_min"] = min(non_zero_reasons)
                    analyses_combined[f"{key}_max"] = max_reason
                    analyses_combined[f"{key}_std"] = 0.0
                else:
                    # No bankruptcy - all zeros
                    analyses_combined[f"{key}_mean"] = 0
                    analyses_combined[f"{key}_min"] = 0
                    analyses_combined[f"{key}_max"] = 0
                    analyses_combined[f"{key}_std"] = 0.0

            elif (
                not values
                or any([x == np.inf for x in values])
                or any([x is None for x in values])
            ):
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

    def calc_fitness(self, analyses_combined, analyses, individual, verbose=True):
        # Check for bankruptcy first - look for bankruptcy_timestamp in any of the analysis results
        keys = self.config["optimize"]["limits"]

        bankruptcy_timestamp = None
        # Check for bankruptcy in any of the bankruptcy timestamp keys
        for suffix in ["_mean", "_min", "_max"]:
            key = f"bankruptcy_timestamp{suffix}"
            if key in analyses_combined and analyses_combined[key] is not None:
                bankruptcy_timestamp = int(analyses_combined[key])
                break

        # Workaround: If bankruptcy_timestamp is None but we suspect bankruptcy occurred,
        # infer it from the equity series length
        if bankruptcy_timestamp is None:
            # Get actual number of timesteps from hlcvs_shapes
            first_exchange = next(iter(self.hlcvs_shapes))
            n_timesteps = self.hlcvs_shapes[first_exchange][
                0
            ]  # First dimension is timesteps

            # Check if any exchange has significantly fewer equity data points than expected
            # This would indicate the backtest stopped early due to bankruptcy
            for exchange_name, analysis_data in analyses.items():
                if "equity_length" in analysis_data:
                    equity_length = analysis_data["equity_length"]
                    if equity_length < (
                        n_timesteps - 10
                    ):  # 10 timestep buffer as suggested
                        # Infer bankruptcy timestamp from the equity length
                        bankruptcy_timestamp = equity_length
                        break

        if bankruptcy_timestamp is not None:
            # Get actual number of timesteps from hlcvs_shapes
            # Use the first exchange's shape as they should all be the same
            first_exchange = next(iter(self.hlcvs_shapes))
            n_timesteps = self.hlcvs_shapes[first_exchange][
                0
            ]  # First dimension is timesteps

            # Calculate penalty that heavily penalizes early bankruptcies
            # The earlier the bankruptcy, the higher the penalty
            progress_ratio = (
                bankruptcy_timestamp / n_timesteps
            )  # 0.0 = immediate bankruptcy, 1.0 = end

            # Base penalty that scales exponentially with how early the bankruptcy occurred
            # Early bankruptcies (low progress_ratio) get much higher penalties
            base_penalty = 10 ** (len(keys) + 2)  # Large base penalty

            # Exponential scaling: earlier bankruptcies get exponentially higher penalties
            # progress_ratio of 0.1 (10% through) gets ~10x higher penalty than 0.9 (90% through)
            early_bankruptcy_multiplier = (
                1.0 - progress_ratio
            ) + 0.1  # Ensures minimum multiplier of 0.1

            penalty = base_penalty * early_bankruptcy_multiplier

            # Skip logs and table for bankrupt strategies
            return penalty, penalty, False

        # Check for high drawdown early to skip table generation
        prefix = "btc_" if self.config["backtest"]["use_btc_collateral"] else ""
        drawdown = analyses_combined.get(f"{prefix}drawdown_worst_max", 0)
        equity_diff = analyses_combined.get(
            f"{prefix}equity_balance_diff_neg_max_max", 0
        )

        # Skip table and logs for high drawdown strategies (including no-trade strategies)
        if drawdown >= 1.0 or equity_diff >= 1.0:
            penalty = 10 ** (len(keys) + 99)  # Large penalty for high drawdown
            return penalty, penalty, False

        # Debug: Log when we have a normal strategy that should show a table
        if verbose:
            print(
                f"✅ NORMAL STRATEGY: drawdown={drawdown:.3f}, equity_diff={equity_diff:.3f} - showing table"
            )

        # Check if we're in a cron environment or non-interactive shell
        import sys

        is_interactive = sys.stdout.isatty() and sys.stderr.isatty()

        # Force colors even in non-interactive environments like cron
        console = Console(
            force_terminal=True,
            no_color=False,
            log_path=False,
            width=159,
            color_system="truecolor",  # Force truecolor support
            legacy_windows=False,
        )
        modifier = 0.0
        # i = 5

        # Step 1: Initialize min/max values
        min_contribution = float("inf")
        max_contribution = float("-inf")
        min_modifier = float("inf")
        max_modifier = float("-inf")

        # Define color codes
        RESET = "\033[0m"
        CYAN = (0, 255, 255)  # Bright cyan (you can tweak as needed)

        # Define the custom green and red colors
        GREEN_RGB = (195, 232, 141)
        RED_RGB = (255, 83, 112)

        # Store the results for each key to later apply colors
        results = []

        header_aliases = {
            "adg": "ADG",
            "adg_w": "ADG(w)",
            "gadg": "GADG",
            "gadg_w": "GADG(w)",
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
            "time_in_market_percent": "TiM %",
            "days_without_position": "Days w/o Pos",
            "days_with_stale_position": "Days Stale",
            "expected_shortfall_1pct": "Expected Fall 1%",
        }
        i = len(keys) + 1

        # Step 2: Single pass to process and gather data
        for key in keys:
            keym = key.replace("lower_bound_", "").split("-")[0] + "_max"
            myKey = key.replace("lower_bound_", "").split("-")[0]
            if keym not in analyses_combined:
                keym = prefix + keym
                assert keym in analyses_combined, f"❌ malformed key: {keym}"

            target = self.config["optimize"]["limits"][key]
            current = analyses_combined[keym]
            delta = current - target

            # Determine if we expect higher or lower values for the current key
            expect_higher_keys = (
                "gain",
                "rsquared",
                "positions_held_per_day",
                "mdg",
                "adg",
                "sharpe",
                "calmar",
                "omega",
                "sortino",
                "sterling",
                "time_in_market_percent",
            )

            expect_higher = any(key in keym for key in expect_higher_keys)

            def normalize_delta(
                delta, current, target, expect_higher, reward_mode=False
            ):
                eps = 1e-9
                delta = current - target

                if current < 0:
                    return 10 ** (3)  # Treat invalid values as high penalty

                if reward_mode:
                    relative_delta = abs(current - target) / max(target, eps)
                    return -1 * (
                        relative_delta
                    )  # More reward as current exceeds target

                # Non-reward mode: bounded score between 0 and 1
                if expect_higher:
                    if delta >= 0:
                        return 0.0
                    else:
                        norm = abs(delta) / max(target, eps)
                        return 0.9 * min(norm, 1.0) + 0.1
                else:
                    if delta <= 0:
                        return 0.0
                    else:
                        norm = delta / max(current, eps)
                        return 0.9 * min(norm, 1.0) + 0.1

            # def normalize_delta(delta, current, target, expect_higher,reward_mode=False):
            #     eps = 1e-9

            #     if current < 0:
            #         return 1.0
            #     if reward_mode:
            #         return -1 * (abs(delta) / max(target, eps))

            #     if expect_higher:
            #         return max(0, (target - current)/max(target, eps)) # Normalized [0,1]
            #     else:
            #         return max(0, (current - target)/max(current, eps))  # Normalized [0,1]

            # Calculate normalized error based on delta and target
            # contribution = (10 ** i) * normalize_delta(delta,current,target,expect_higher)
            # modifier += contribution
            # i-=1

            # contribution = (10 ** 1) * normalize_delta(delta,current,target,expect_higher)
            # if modifier == 0:
            #     modifier = contribution
            # elif contribution != 0:
            #     modifier = modifier * contribution

            contribution = (10**1) * normalize_delta(
                delta, current, target, expect_higher
            )
            modifier = modifier + contribution

            # Update min/max for contribution and modifier
            min_contribution = min(min_contribution, contribution)
            max_contribution = max(max_contribution, contribution)
            min_modifier = min(min_modifier, modifier)
            max_modifier = max(max_modifier, modifier)

            # Store the result (we'll use this data for printing later)
            results.append(
                {
                    "key": header_aliases[myKey],
                    "target": target,
                    "current": current,
                    "delta": delta,
                    "contribution": contribution,
                    "modifier": modifier,
                    "expect_higher": expect_higher,
                }
            )

        def log_modulus(x):
            x = np.asarray(x, dtype=np.float64)
            return np.sign(x) * np.log10(1 + np.abs(x))

        def rgb_to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(*rgb)

        def value_to_color(value, min_value, max_value):
            # Apply log-modulus transform to all values
            log_val = log_modulus(value)
            log_min = log_modulus(min_value)
            log_max = log_modulus(max_value)

            # Avoid zero division if min == max after transformation
            if log_max == log_min:
                norm_value = 0.5
            else:
                norm_value = (log_val - log_min) / (log_max - log_min)
                norm_value = np.clip(norm_value, 0, 1)

            r = int(GREEN_RGB[0] + norm_value * (RED_RGB[0] - GREEN_RGB[0]))
            g = int(GREEN_RGB[1] + norm_value * (RED_RGB[1] - GREEN_RGB[1]))
            b = int(GREEN_RGB[2] + norm_value * (RED_RGB[2] - GREEN_RGB[2]))

            # Use rgb() format instead of hex for better cron compatibility
            return f"rgb({r},{g},{b})"

        all_zero_contributions = all(r["contribution"] == 0.0 for r in results)

        i = len(keys) + 1

        # Step 4: Print the results with colorized values
        table = Table(
            show_header=True,
            header_style="bold magenta",
            title="📊 Optimization Status",
        )

        table.add_column("Status", justify="center")
        table.add_column("Parameter", justify="center")
        table.add_column("Target", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Δ", justify="right")
        table.add_column("Contribution", justify="right")
        table.add_column("Modifier", justify="right")
        table.add_column("NPos", justify="right")
        table.add_column("%", justify="right")

        # scaling_values = {
        #     "ADG":              10,
        #     "Calmar":           4,
        #     "Days Stale":       4,
        #     "Days w/o Pos":     4,
        #     "DD Worst":         5,
        #     "DD 1%":            5,
        #     "E/B Diff - max":   1,
        #     "E/B Diff - mean":  1,
        #     "E/B Diff + max":   -1,
        #     "E/B Diff + mean":  -1,
        #     "GADG":              5,
        #     "Gain":             10,
        #     "LPR":              2,
        #     "MDG":              5,
        #     "Omega":            4,
        #     "R²":               20,
        #     "Sharpe":           4,
        #     "Sortino":          4,
        #     "Sterling":         4,
        #     "TiM %":            2,
        #     # "Hrs/Pos":          1,
        #     # "Pos/Day":          3,
        #     # "Unchg Max":        3,
        # }
        scaling_values = {
            "DD Worst": -99,
            "DD 1%": -99,
            "Gain": 0,
            "R²": 1,
            "Days w/o Pos": 0,
            "Days Stale": 0,
        }
        for result in results:
            key = result["key"]
            target = result["target"]
            current = result["current"]
            delta = result["delta"]
            expect_higher = result["expect_higher"]

            if all_zero_contributions:
                # target = ideal_targets[key]
                # delta = current - target

                try:
                    contribution = 10 ** (scaling_values[key]) * normalize_delta(
                        delta,
                        current,
                        target,
                        expect_higher,
                        reward_mode=all_zero_contributions,
                    )
                except:
                    contribution = 0
                # contribution = normalize_delta(delta, current, target, expect_higher, reward_mode=all_zero_contributions)
                modifier += contribution
            else:
                contribution = result["contribution"]
                modifier = result["modifier"]

            # Determine status and color
            if (delta >= 0 and expect_higher) or (
                all_zero_contributions and expect_higher
            ):
                status = "✅ above target"
            elif (delta <= 0 and not expect_higher) or (
                all_zero_contributions and not expect_higher
            ):
                status = "✅ below target"
            elif delta < 0 and expect_higher:
                status = "❌ below target"
            elif delta > 0 and not expect_higher:
                status = "❌ above target"

            status_color = (
                rgb_to_hex(GREEN_RGB)
                if "✅" in status
                else rgb_to_hex(RED_RGB)
                if "❌" in status
                else rgb_to_hex(CYAN)
            )

            # keym_display = (key[:12] + '...') if len(key) > 15 else f"{key:<15}"
            keym_display = f"{key}"

            current_color = f"rgb({CYAN[0]},{CYAN[1]},{CYAN[2]})"
            contribution_color = value_to_color(
                contribution, min_contribution, max_contribution
            )
            modifier_color = value_to_color(modifier, min_modifier, max_modifier)

            # Create rich.Text objects
            status_text = Text(status, style=Style(color=status_color))
            current_text = Text(f"{current:>10.5f}", style=Style(color=current_color))
            contribution_text = Text(
                f"{contribution:>12.5e}", style=Style(color=contribution_color)
            )
            modifier_text = Text(
                f"{modifier:>12.5e}", style=Style(color=modifier_color)
            )
            # Add row
            table.add_row(
                status_text,
                f"{keym_display}",
                f"{target:>10.5f}",
                current_text,
                f"{delta:+10.5f}",
                contribution_text,
                modifier_text,
                f"{math.floor(individual[19])}",
                f"{individual[20]:>10.5f}",
            )

        # Display the table
        console.print(table)
        # console.print(f"Final Score: {(modifier) ** (1/(len(results)))}")

        npos = individual[19]

        scoring_keys = self.config["optimize"]["scoring"]
        assert len(scoring_keys) == 2, (
            f"~❌ Expected 2 fitness scoring keys, got {len(scoring_keys)}"
        )

        scores = []
        for sk in scoring_keys:
            skm = f"{sk}_mean"
            if skm not in analyses_combined:
                skm = prefix + skm
                if skm not in analyses_combined:
                    raise Exception(f"~❌ Invalid scoring key: {sk}")

            # score_value = modifier - analyses_combined[skm]
            score_value = modifier
            scores.append(score_value)

            # if verbose:
            # print(f"~🎯 [{skm}] Modifier: {modifier:.5e}, Value: {analyses_combined[skm]:.5f}, Score: {score_value:.5e}")

        # Return scores after processing all scoring keys
        return scores[0], scores[1], True
=======
            fills, equities_array, analysis = execute_backtest(payload, config)
            analyses[exchange] = analysis

            # Explicitly drop large intermediate arrays to keep worker RSS low.
            del fills
            del equities_array
        scenario_metrics = build_scenario_metrics(analyses)
        aggregate_stats = scenario_metrics.get("stats", {})
        flat_stats = flatten_metric_stats(aggregate_stats)
        objectives, total_penalty = self.calc_fitness(flat_stats)
        objectives_map = {f"w_{i}": val for i, val in enumerate(objectives)}
        metrics_payload = {
            "stats": aggregate_stats,
            "objectives": objectives_map,
            "constraint_violation": total_penalty,
        }
        individual.evaluation_metrics = metrics_payload
        actual_hash = calc_hash(individual)
        self.seen_hashes[actual_hash] = (tuple(objectives), total_penalty)
        return tuple(objectives), total_penalty, metrics_payload

    def build_limit_checks(self):
        limits = self.config["optimize"].get("limits", [])
        objective_index_map: Dict[str, List[int]] = {}
        for idx, metric in enumerate(self.config["optimize"].get("scoring", [])):
            objective_index_map.setdefault(metric, []).append(idx)
        self.limit_checks = expand_limit_checks(
            limits,
            self.scoring_weights,
            penalty_weight=1e6,
            objective_index_map=objective_index_map,
        )

    def calc_fitness(self, analyses_combined):
        scoring_keys = self.config["optimize"]["scoring"]
        per_objective_modifier = [0.0] * len(scoring_keys)
        global_modifier = 0.0
        for check in self.limit_checks:
            val = analyses_combined.get(check["metric_key"])
            penalty = compute_limit_violation(check, val)
            if not penalty:
                continue
            targets = check.get("objective_indexes") or []
            if targets:
                for idx in targets:
                    if 0 <= idx < len(per_objective_modifier):
                        per_objective_modifier[idx] += penalty
            else:
                global_modifier += penalty

        total_penalty = global_modifier + sum(per_objective_modifier)
        scores = []
        for idx, sk in enumerate(scoring_keys):
            penalty_total = global_modifier + per_objective_modifier[idx]
            if penalty_total:
                scores.append(penalty_total)
                continue

            parts = sk.split("_")
            candidates = []
            if len(parts) <= 1:
                candidates = [sk]
            else:
                base, rest = parts[0], parts[1:]
                base_candidate = "_".join([base, *rest])
                candidates.append(base_candidate)
                for perm in permutations(rest):
                    candidate = "_".join([base, *perm])
                    candidates.append(candidate)

            extended_candidates = []
            seen = set()
            for candidate in candidates:
                if candidate not in seen:
                    extended_candidates.append(candidate)
                    seen.add(candidate)
                for suffix in ("usd", "btc"):
                    with_suffix = f"{candidate}_{suffix}"
                    if with_suffix not in seen:
                        extended_candidates.append(with_suffix)
                        seen.add(with_suffix)
                    parts_candidate = candidate.split("_")
                    if len(parts_candidate) >= 2:
                        inserted = "_".join(parts_candidate[:-1] + [suffix, parts_candidate[-1]])
                        if inserted not in seen:
                            extended_candidates.append(inserted)
                            seen.add(inserted)

            val = None
            weight = None
            selected_metric = None
            for candidate in extended_candidates:
                metric_key = f"{candidate}_mean"
                if val is None and metric_key in analyses_combined:
                    val = analyses_combined[metric_key]
                    selected_metric = candidate
                if weight is None and candidate in self.scoring_weights:
                    weight = self.scoring_weights[candidate]
                if val is not None and weight is not None:
                    break

            if val is None:
                val = 0
            if weight is None:
                weight = 1.0
            scores.append(val * weight)
        return tuple(scores), total_penalty
>>>>>>> upstream/master

    def __del__(self):
        for attachment_map in self._attachments.values():
            for attachment in attachment_map.values():
                attachment.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("shared_hlcvs_np", None)
        state.pop("shared_btc_np", None)
        state.pop("_attachments", None)
        state.pop("shared_array_manager", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.shared_array_manager = None
        self.shared_hlcvs_np = {}
        self.shared_btc_np = {}
        self._attachments = {"hlcvs": {}, "btc": {}}
        for exchange in self.exchanges:
            self._ensure_attached(exchange)


class SuiteEvaluator:
    def __init__(
        self,
        base_evaluator: Evaluator,
        scenario_contexts: List[ScenarioEvalContext],
        aggregate_cfg: Dict[str, Any],
    ) -> None:
        self.base = base_evaluator
        self.contexts = scenario_contexts
        self.aggregate_cfg = aggregate_cfg
        # Cache for master dataset attachments (shared across scenarios)
        self._master_attachments: Dict[str, Dict[str, Any]] = {"hlcvs": {}, "btc": {}}
        self._master_arrays: Dict[str, Dict[str, np.ndarray]] = {"hlcvs": {}, "btc": {}}

    def _ensure_master_attachment(self, spec, cache_key: str, array_type: str) -> np.ndarray:
        """Attach to master SharedMemory if not already attached."""
        if cache_key not in self._master_arrays[array_type]:
            attachment = attach_shared_array(spec)
            self._master_attachments[array_type][cache_key] = attachment
            self._master_arrays[array_type][cache_key] = attachment.array
        return self._master_arrays[array_type][cache_key]

    def _get_lazy_slice_data(
        self, ctx: ScenarioEvalContext, exchange: str
    ) -> tuple[np.ndarray, np.ndarray | None, list[int] | None]:
        """
        Get data for lazy slicing mode.
        Returns (hlcvs_view, btc_view, coin_indices).

        Only applies TIME slicing here (creates views, O(1) memory).
        Coin subsetting is deferred to build_backtest_payload which does it efficiently.
        """
        master_spec = ctx.master_hlcvs_specs[exchange]
        master_array = self._ensure_master_attachment(master_spec, master_spec.name, "hlcvs")

        time_slice = ctx.time_slice.get(exchange) if ctx.time_slice else None
        coin_indices = ctx.coin_slice_indices.get(exchange) if ctx.coin_slice_indices else None

        # Time slicing creates a VIEW (no copy, O(1) memory)
        if time_slice is not None:
            start_idx, end_idx = time_slice
            hlcvs_view = master_array[start_idx:end_idx]
        else:
            hlcvs_view = master_array

        # BTC slice (time-only slicing creates a view)
        btc_view = None
        master_btc_spec = ctx.master_btc_specs.get(exchange) if ctx.master_btc_specs else None
        if master_btc_spec is not None:
            master_btc = self._ensure_master_attachment(master_btc_spec, master_btc_spec.name, "btc")
            if time_slice is not None:
                start_idx, end_idx = time_slice
                btc_view = master_btc[start_idx:end_idx]
            else:
                btc_view = master_btc

        # Return coin_indices to let build_backtest_payload handle subsetting in one step
        return hlcvs_view, btc_view, coin_indices

    def _uses_lazy_slicing(self, ctx: ScenarioEvalContext, exchange: str) -> bool:
        """Check if context uses lazy slicing for the given exchange."""
        return (
            ctx.master_hlcvs_specs is not None
            and exchange in ctx.master_hlcvs_specs
            and ctx.master_hlcvs_specs[exchange] is not None
        )

    def _ensure_context_attachment(self, ctx: ScenarioEvalContext, exchange: str) -> None:
        """Attach to SharedMemory for non-lazy-slicing contexts only."""
        # Skip if using lazy slicing - slices are computed on-demand in evaluate()
        if self._uses_lazy_slicing(ctx, exchange):
            return

        # Original flow: per-scenario SharedMemory
        if exchange not in ctx.shared_hlcvs_np:
            if exchange in ctx.hlcvs_specs and ctx.hlcvs_specs[exchange] is not None:
                attachment = attach_shared_array(ctx.hlcvs_specs[exchange])
                ctx.attachments["hlcvs"][exchange] = attachment
                ctx.shared_hlcvs_np[exchange] = attachment.array
        if exchange not in ctx.shared_btc_np and exchange in ctx.btc_usd_specs:
            if ctx.btc_usd_specs[exchange] is not None:
                attachment = attach_shared_array(ctx.btc_usd_specs[exchange])
                ctx.attachments["btc"][exchange] = attachment
                ctx.shared_btc_np[exchange] = attachment.array

    def evaluate(self, individual, overrides_list):
        individual[:] = enforce_bounds(individual, self.base.bounds, self.base.sig_digits)
        config = individual_to_config(
            individual, optimizer_overrides, overrides_list, self.base.config
        )
        individual_hash = calc_hash(individual)
        seen_hashes = self.base.seen_hashes
        duplicate_counter = self.base.duplicate_counter

        if individual_hash in seen_hashes:
            existing_entry = seen_hashes[individual_hash]
            existing_score = None
            existing_penalty = 0.0
            if existing_entry is not None:
                existing_score, existing_penalty = existing_entry
            duplicate_counter["total"] += 1
            perturbation_funcs = [
                self.base.perturb_x_pct,
                self.base.perturb_step_digits,
                self.base.perturb_gaussian,
                self.base.perturb_random_subset,
                self.base.perturb_sample_some,
                self.base.perturb_large_uniform,
            ]
            for perturb_fn in perturbation_funcs:
                perturbed = perturb_fn(individual)
                perturbed = enforce_bounds(perturbed, self.base.bounds, self.base.sig_digits)
                new_hash = calc_hash(perturbed)
                if new_hash not in seen_hashes:
                    individual[:] = perturbed
                    seen_hashes[new_hash] = None
                    config = individual_to_config(
                        perturbed, optimizer_overrides, overrides_list, self.base.config
                    )
                    duplicate_counter["resolved"] += 1
                    break
            else:
                if existing_score is not None:
                    duplicate_counter["reused"] += 1
                    return tuple(existing_score), existing_penalty, None
        else:
            seen_hashes[individual_hash] = None

        scenario_results: List[ScenarioResult] = []

        from tools.iterative_backtester import combine_analyses as combine

        for ctx in self.contexts:
            scenario_config = deepcopy(config)
            scenario_config["backtest"]["start_date"] = ctx.config["backtest"]["start_date"]
            scenario_config["backtest"]["end_date"] = ctx.config["backtest"]["end_date"]
            scenario_config["backtest"]["coins"] = deepcopy(ctx.config["backtest"]["coins"])
            scenario_config["backtest"]["cache_dir"] = deepcopy(
                ctx.config["backtest"].get("cache_dir", {})
            )
            scenario_config.setdefault("live", {})
            scenario_config["live"]["approved_coins"] = deepcopy(
                ctx.config["live"].get("approved_coins", {})
            )
            scenario_config["live"]["ignored_coins"] = deepcopy(
                ctx.config["live"].get("ignored_coins", {})
            )
            logging.debug(
                "Optimizer scenario %s | start=%s end=%s coins=%s",
                ctx.label,
                scenario_config["backtest"].get("start_date"),
                scenario_config["backtest"].get("end_date"),
                list(scenario_config["backtest"]["coins"].keys()),
            )
            if ctx.overrides:
                _apply_config_overrides(scenario_config, ctx.overrides)
            scenario_config["disable_plotting"] = True

            analyses = {}
            for exchange in ctx.exchanges:
                # Get data arrays - either from lazy slicing or cached SharedMemory
                if self._uses_lazy_slicing(ctx, exchange):
                    # Get time-sliced VIEW (O(1) memory) + coin indices
                    # Coin subsetting happens inside build_backtest_payload (single copy)
                    hlcvs_data, btc_data, coin_indices = self._get_lazy_slice_data(ctx, exchange)
                else:
                    self._ensure_context_attachment(ctx, exchange)
                    hlcvs_data = ctx.shared_hlcvs_np[exchange]
                    btc_data = ctx.shared_btc_np.get(exchange)
                    coin_indices = ctx.coin_indices.get(exchange)

                payload = build_backtest_payload(
                    hlcvs_data,
                    ctx.msss[exchange],
                    scenario_config,
                    exchange,
                    btc_data,
                    ctx.timestamps.get(exchange),
                    coin_indices=coin_indices,
                )
                fills, equities_array, analysis = execute_backtest(payload, scenario_config)
                analyses[exchange] = analysis

                # Free backtest results to allow memory reuse
                del fills
                del equities_array
                del payload

            combined_metrics = combine(analyses)
            stats = combined_metrics.get("stats", {})
            logging.debug(
                "Scenario metrics | label=%s adg_pnl=%s peak_recovery_hours_pnl=%s",
                ctx.label,
                (
                    stats.get("adg_pnl", {}).get("mean")
                    if isinstance(stats.get("adg_pnl"), dict)
                    else stats.get("adg_pnl")
                ),
                (
                    stats.get("peak_recovery_hours_pnl", {}).get("mean")
                    if isinstance(stats.get("peak_recovery_hours_pnl"), dict)
                    else stats.get("peak_recovery_hours_pnl")
                ),
            )
            scenario_results.append(
                ScenarioResult(
                    scenario=SuiteScenario(
                        label=ctx.label,
                        start_date=None,
                        end_date=None,
                        coins=None,
                        ignored_coins=None,
                    ),
                    per_exchange={},
                    metrics={"stats": combined_metrics.get("stats", {})},
                    elapsed_seconds=0.0,
                    output_path=None,
                )
            )

        aggregate_summary = aggregate_metrics(scenario_results, self.aggregate_cfg)
        suite_payload = build_suite_metrics_payload(scenario_results, aggregate_summary)
        aggregate_stats = aggregate_summary.get("stats", {})

        flat_stats = flatten_metric_stats(aggregate_stats)
        objectives, total_penalty = self.base.calc_fitness(flat_stats)
        objectives_map = {f"w_{i}": val for i, val in enumerate(objectives)}

        metrics_payload = {
            "objectives": objectives_map,
            "suite_metrics": suite_payload,
            "constraint_violation": total_penalty,
        }

        individual.evaluation_metrics = metrics_payload
        actual_hash = calc_hash(individual)
        self.base.seen_hashes[actual_hash] = (tuple(objectives), total_penalty)
        return tuple(objectives), total_penalty, metrics_payload

    def __del__(self):
        for ctx in self.contexts:
            for attachment in ctx.attachments.get("hlcvs", {}).values():
                try:
                    attachment.close()
                except Exception:
                    pass
            for attachment in ctx.attachments.get("btc", {}).values():
                try:
                    attachment.close()
                except Exception:
                    pass


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
    parser.add_argument(
        "-ft",
        "--fine_tune_params",
        "--fine-tune-params",
        type=str,
        default="",
        dest="fine_tune_params",
        help=(
            "Comma-separated optimize bounds keys to tune; other parameters are fixed to their current config values"
        ),
    )


def apply_fine_tune_bounds(
    config: dict,
    fine_tune_params: list[str],
    cli_overridden_bounds: set[str],
) -> None:
    bounds = config.get("optimize", {}).get("bounds", {})
    bot_cfg = config.get("bot", {})
    # First, normalize any CLI overrides such that single values mean fixed bounds
    for key in cli_overridden_bounds:
        if key not in bounds:
            continue
        raw_val = bounds[key]
        if isinstance(raw_val, (list, tuple)):
            if len(raw_val) == 1:
                bounds[key] = [float(raw_val[0]), float(raw_val[0])]
        else:
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                continue
            bounds[key] = [val, val]

    if not fine_tune_params:
        return

    fine_tune_set = set(fine_tune_params)

    for key in list(bounds.keys()):
        if key in fine_tune_set:
            continue
        try:
            pside, param = key.split("_", 1)
        except ValueError:
            logging.warning(f"fine-tune bounds: unable to parse key '{key}', skipping")
            continue
        side_cfg = bot_cfg.get(pside)
        if not isinstance(side_cfg, dict) or param not in side_cfg:
            logging.warning(
                f"fine-tune bounds: missing bot value for '{key}', leaving bounds unchanged"
            )
            continue
        value = side_cfg[param]
        try:
            value_float = float(value)
            bounds[key] = [value_float, value_float]
        except (TypeError, ValueError):
            bounds[key] = [value, value]

    missing = [key for key in fine_tune_set if key not in bounds]
    if missing:
        logging.warning(
            "fine-tune bounds: requested keys not found in optimize bounds: %s",
            ",".join(sorted(missing)),
        )


def extract_configs(path):
    cfgs = []
    if os.path.exists(path):
        if path.endswith("_all_results.bin"):
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


def configs_to_individuals(cfgs, bounds, sig_digits=0):
    inds = set()
    for cfg in cfgs:
        try:
            fcfg = format_config(cfg, verbose=False)
            individual = config_to_individual(fcfg, bounds, sig_digits)
            inds.add(tuple(individual))
            # add duplicate of config, but with lowered total wallet exposure limit
            fcfg2 = deepcopy(fcfg)
            for pside in ["long", "short"]:
                value = fcfg2["bot"][pside]["total_wallet_exposure_limit"] * 0.75
                fcfg2["bot"][pside]["total_wallet_exposure_limit"] = value
            individual2 = config_to_individual(fcfg2, bounds, sig_digits)
            inds.add(tuple(individual2))
        except Exception as e:
            logging.error(f"error loading starting config: {e}")
    return list(inds)


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

    # Store timestamps for interval creation
    timestamps_dict = {}

    config["backtest"]["coins"] = {}
    if config["backtest"]["combine_ohlcvs"]:
        exchange = "combined"
        (
            coins,
            hlcvs,
            mss,
            results_path,
            cache_dir,
            btc_usd_prices,
            timestamps,  # NEW: Get timestamps
        ) = await prepare_hlcvs_mss(config, exchange)
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
        timestamps_dict[exchange] = timestamps  # Store timestamps for interval creation
        required_space = hlcvs.nbytes * 1.1  # Add 10% buffer
        check_disk_space(tempfile.gettempdir(), required_space)
        logging.info(f"Starting to create shared memory file for {exchange}...")
        validate_array(hlcvs, "hlcvs")
        shared_memory_file = create_shared_memory_file(hlcvs)
        shared_memory_files[exchange] = shared_memory_file
        if config["backtest"].get("use_btc_collateral", False):
            # Use the fetched array
            btc_usd_data_dict[exchange] = btc_usd_prices
            # Validate length matches hlcvs
            if len(btc_usd_prices) != hlcvs.shape[0]:
                logging.warning(
                    f"{exchange} BTC/USD prices length ({len(btc_usd_prices)}) doesn't match hlcvs ({hlcvs.shape[0]}). Creating default array."
                )
                btc_usd_data_dict[exchange] = np.ones(hlcvs.shape[0], dtype=np.float64)
        else:
            # Fall back to all ones
            btc_usd_data_dict[exchange] = np.ones(hlcvs.shape[0], dtype=np.float64)
        validate_array(btc_usd_data_dict[exchange], f"btc_usd_data for {exchange}")
        btc_usd_shared_memory_files[exchange] = create_shared_memory_file(
            btc_usd_data_dict[exchange]
        )
        btc_usd_dtypes[exchange] = btc_usd_data_dict[exchange].dtype
        logging.info(
            f"Finished creating shared memory file for {exchange}: {shared_memory_file}"
        )
    else:
        tasks = {}
        for exchange in config["backtest"]["exchanges"]:
            tasks[exchange] = asyncio.create_task(prepare_hlcvs_mss(config, exchange))
        for exchange in config["backtest"]["exchanges"]:
            coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, timestamps = await tasks[
                exchange
            ]
            config["backtest"]["coins"][exchange] = coins
            hlcvs_dict[exchange] = hlcvs
            hlcvs_shapes[exchange] = hlcvs.shape
            hlcvs_dtypes[exchange] = hlcvs.dtype
            msss[exchange] = mss
            timestamps_dict[exchange] = timestamps  # Store timestamps for interval creation
            required_space = hlcvs.nbytes * 1.1  # Add 10% buffer
            check_disk_space(tempfile.gettempdir(), required_space)
            logging.info(f"Starting to create shared memory file for {exchange}...")
            validate_array(hlcvs, "hlcvs")
            shared_memory_file = create_shared_memory_file(hlcvs)
            shared_memory_files[exchange] = shared_memory_file
            # Create the BTC array for this exchange
            if config["backtest"].get("use_btc_collateral", False):
                btc_usd_data_dict[exchange] = btc_usd_prices
                # Validate length matches hlcvs
                if len(btc_usd_prices) != hlcvs.shape[0]:
                    logging.warning(
                        f"{exchange} BTC/USD prices length ({len(btc_usd_prices)}) doesn't match hlcvs ({hlcvs.shape[0]}). Creating default array."
                    )
                    btc_usd_data_dict[exchange] = np.ones(hlcvs.shape[0], dtype=np.float64)
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
    exchanges_fname = (
        "combined" if config["backtest"]["combine_ohlcvs"] else "_".join(exchanges)
    )
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
    btc_usd_data = (
        btc_usd_prices  # Use the fetched btc_usd_prices from prepare_hlcvs_mss
    )
    if config["backtest"].get("use_btc_collateral", False):
        logging.info("Using fetched BTC/USD prices for collateral")
    else:
        logging.info(
            "Using default BTC/USD prices (all 1.0s) as use_btc_collateral is False"
        )
        btc_usd_data = np.ones(
            hlcvs_dict[next(iter(hlcvs_dict))].shape[0], dtype=np.float64
        )

    validate_array(btc_usd_data, "btc_usd_data")
    btc_usd_shared_memory_file = create_shared_memory_file(btc_usd_data)

    # Initialize evaluator with results queue and BTC/USD shared memory
    evaluator = Evaluator(
        shared_memory_files=shared_memory_files,
        hlcvs_shapes=hlcvs_shapes,
        hlcvs_dtypes=hlcvs_dtypes,
        btc_usd_shared_memory_files=btc_usd_shared_memory_files,
        btc_usd_dtypes=btc_usd_dtypes,
        msss=msss,
        config=config,
        results_queue=results_queue,
    )

    logging.info(f"Finished initializing evaluator...")

    # Ensure logs directory exists for best optimization status logging
    os.makedirs("logs", exist_ok=True)

    # Return evaluator, config, and additional data for interval creation
    interval_data = {
        "timestamps": timestamps_dict,
        "hlcvs": hlcvs_dict,
        "btc_usd_data": btc_usd_data_dict,
    }

    return evaluator, config, interval_data


async def myMain(args):
    evaluator, config, interval_data = await initEvaluator(args.config_path)

    creator.create(
        "FitnessMulti", base.Fitness, weights=(-1.0, -1.0)
    )  # Minimize both objectives
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    # Define parameter bounds
    param_bounds = sort_dict_keys(config["optimize"]["bounds"])
    for k, v in param_bounds.items():
        if len(v) == 1:
            param_bounds[k] = [v[0], v[0]]

    # Register attribute generators
    for i, (param_name, (low, high)) in enumerate(param_bounds.items()):
        if param_name == "long_n_positions":
            # Use discrete choice for n_positions to give equal probability to each integer
            choices = list(range(int(low), int(high) + 1))
            toolbox.register(
                f"attr_{i}", lambda choices=choices: float(np.random.choice(choices))
            )
        else:
            toolbox.register(f"attr_{i}", np.random.uniform, low, high)

    def create_individual():
        return creator.Individual(
            [getattr(toolbox, f"attr_{i}")() for i in range(len(param_bounds))]
        )

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    overrides_list = config.get("optimize", {}).get("enable_overrides", [])

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
        param_bounds=param_bounds,
    )
    toolbox.register("select", tools.selNSGA2)

    # Parallelization setup
    logging.info(
        f"Initializing multiprocessing pool. N cpus: {config['optimize']['n_cpus']}"
    )
    # pool = multiprocessing.Pool(processes=config["optimize"]["n_cpus"])
    # toolbox.register("map", pool.map)
    logging.info(f"Finished initializing multiprocessing pool.")

    # Create initial population using Latin Hypercube Sampling for better coverage
    logging.info(f"Creating initial population with LHS...")

    bounds = [(low, high) for low, high in param_bounds.items()]
    starting_individuals = configs_to_individuals(
        get_starting_configs(args.starting_configs), param_bounds
    )
    
    # Also try to load initial individual from optimize.json (like PSO does)
    try:
        from deap_optimizer.utils import create_seeded_individual_from_config
        seeded_individual = create_seeded_individual_from_config(toolbox, args.config_path)
        if seeded_individual is not None:
            # Add seeded individual from optimize.json to starting individuals
            starting_individuals.insert(0, list(seeded_individual))
            logging.info(f"📋 Added seeded individual from {args.config_path}")
    except Exception as e:
        logging.warning(f"Could not create seeded individual from {args.config_path}: {e}")
    
    if (nstart := len(starting_individuals)) > (
        popsize := config["optimize"]["population_size"]
    ):
        logging.info(f"Number of starting configs greater than population size.")
        logging.info(f"Increasing population size: {popsize} -> {nstart}")
        config["optimize"]["population_size"] = nstart

    # Use LHS for initial population creation
    pop_size = config["optimize"]["population_size"]
    
    # Separate variable and fixed parameters (LHS requires lower < upper)
    param_names = list(param_bounds.keys())
    variable_indices = []
    fixed_indices = []
    fixed_values = []
    
    for i, (name, (low, high)) in enumerate(param_bounds.items()):
        if low < high:
            variable_indices.append(i)
        else:
            fixed_indices.append(i)
            fixed_values.append(low)  # Use the fixed value
    
    try:
        from scipy.stats import qmc
        
        if variable_indices:
            # LHS only for variable parameters
            variable_lower = np.array([list(param_bounds.values())[i][0] for i in variable_indices])
            variable_upper = np.array([list(param_bounds.values())[i][1] for i in variable_indices])
            
            sampler = qmc.LatinHypercube(d=len(variable_indices), seed=np.random.randint(0, 2**31))
            unit_samples = sampler.random(n=pop_size)
            variable_samples = qmc.scale(unit_samples, variable_lower, variable_upper)
            
            # Reconstruct full samples with fixed values
            full_samples = np.zeros((pop_size, len(param_bounds)))
            for j, var_idx in enumerate(variable_indices):
                full_samples[:, var_idx] = variable_samples[:, j]
            for j, fix_idx in enumerate(fixed_indices):
                full_samples[:, fix_idx] = fixed_values[j]
            
            population = [creator.Individual(list(sample)) for sample in full_samples]
            logging.info(f"Created {pop_size} individuals using LHS ({len(variable_indices)} variable, {len(fixed_indices)} fixed params)")
        else:
            # All parameters are fixed - just create identical individuals
            fixed_individual = [list(param_bounds.values())[i][0] for i in range(len(param_bounds))]
            population = [creator.Individual(list(fixed_individual)) for _ in range(pop_size)]
            logging.info(f"All parameters fixed, created {pop_size} identical individuals")
            
    except ImportError:
        logging.warning("scipy not available, falling back to random uniform sampling")
        population = toolbox.population(n=pop_size)
    
    # Override with starting individuals if provided
    if starting_individuals:
        bounds_list = [(low, high) for low, high in param_bounds.values()]
        for i in range(len(starting_individuals)):
            adjusted = [
                max(min(x, bounds_list[z][1]), bounds_list[z][0])
                for z, x in enumerate(starting_individuals[i])
            ]
            population[i] = creator.Individual(adjusted)

        for i in range(len(starting_individuals), len(population) // 2):
            mutant = deepcopy(
                population[np.random.choice(range(len(starting_individuals)))]
            )
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

    # population = alternatives.pso(
    #     population,
    #     toolbox,
    #     evaluator=evaluator,
    #     ngen=max(1, int(config["optimize"]["iters"] / len(population))),
    #     verbose=True,
    #     parameter_bounds=param_bounds
    # )
    
    population, logbook = alternatives.deap_ea(
            population=population,
            toolbox=toolbox,
            evaluator=evaluator,
            ngen=max(1, int(config["optimize"]["iters"] / len(population))),
            verbose=True,
            parameter_bounds=param_bounds,
            checkpoint_path="deap_checkpoint.pkl",
            cxpb=config["optimize"]["crossover_probability"],
            mutpb=config["optimize"]["mutation_probability"],
            stagnation_config=config["optimize"].get("stagnation_detection", None),
            config=config,
            interval_data=interval_data
        )

    # population = alternatives.cma_es_restarts(
    #     population,
    #     toolbox,
    #     evaluator=evaluator,
    #     ngen=max(
    #         1, int(config["optimize"]["iters"] / len(population))
    #     ),  # Ignored - runs indefinitely
    #     verbose=True,
    #     parameter_bounds=param_bounds,
    #     checkpoint_path=config["optimize"].get("checkpoint_path", "cma_checkpoint.pkl"),
    #     population_size=config["optimize"].get("population_size", 1000),
    #     sigma0=config["optimize"].get("sigma0", 0.01),
    #     max_iter_per_restart=config["optimize"].get("max_iter_per_restart", 1000),
    #     tol_hist_fun=config["optimize"].get("tol_hist_fun", 1e-12),
    #     equal_fun_vals_k=config["optimize"].get("equal_fun_vals_k", None),
    #     tol_x=config["optimize"].get("tol_x", 1e-11),
    #     tol_up_sigma=config["optimize"].get("tol_up_sigma", 1e20),
    #     stagnation_iter=config["optimize"].get("stagnation_iter", 100),
    #     condition_cov=config["optimize"].get("condition_cov", 1e14),
    #     min_sigma=config["optimize"].get("min_sigma", None),
    # )


async def main():
    parser = argparse.ArgumentParser(prog="optimize", description="run optimizer")
    parser.add_argument(
        "config_path",
        type=str,
        default=None,
        nargs="?",
        help="path to json passivbot config",
    )
    parser.add_argument(
        "--suite",
        nargs="?",
        const="true",
        default=None,
        type=str2bool,
        metavar="y/n",
        help="Enable or disable suite mode for optimizer run (omit to use config's suite_enabled setting).",
    )
    parser.add_argument(
        "--scenarios",
        "-sc",
        type=str,
        default=None,
        metavar="LABELS",
        help="Comma-separated list of scenario labels to run (implies --suite y). "
        "Example: --scenarios base,binance_only",
    )
    parser.add_argument(
        "--suite-config",
        type=str,
        default=None,
        help="Optional config file providing backtest.scenarios overrides.",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        help="Logging verbosity (warning, info, debug, trace or 0-3).",
    )
    template_config = get_template_config()
    del template_config["bot"]
    keep_live_keys = {
        "approved_coins",
        "minimum_coin_age_days",
    }
    for key in sorted(template_config["live"]):
        if key not in keep_live_keys:
            del template_config["live"][key]
    add_config_arguments(parser, template_config)
    add_extra_options(parser)
<<<<<<< HEAD
    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    await myMain(args)
=======
    raw_args = merge_negative_cli_values(sys.argv[1:])
    raw_args = _normalize_optional_bool_flag(raw_args, "--suite")
    args = parser.parse_args(raw_args)
    initial_log_level = resolve_log_level(args.log_level, None, fallback=1)
    configure_logging(debug=initial_log_level)
    if args.config_path is None:
        logging.info(f"loading default template config configs/template.json")
        config = load_config("configs/template.json", verbose=True)
    else:
        logging.info(f"loading config {args.config_path}")
        config = load_config(args.config_path, verbose=True)
    update_config_with_args(config, args, verbose=True)
    config = format_config(config, verbose=False)
    config_logging_value = get_optional_config_value(config, "logging.level", None)
    effective_log_level = resolve_log_level(args.log_level, config_logging_value, fallback=1)
    if effective_log_level != initial_log_level:
        configure_logging(debug=effective_log_level)
    logging.info(
        "Config normalized for optimization | template=%s | scoring=%s",
        TEMPLATE_CONFIG_MODE,
        ",".join(config["optimize"].get("scoring", [])),
    )
    fine_tune_params = (
        [p.strip() for p in (args.fine_tune_params or "").split(",") if p.strip()]
        if getattr(args, "fine_tune_params", "")
        else []
    )
    cli_bounds_overrides = {
        key.split("optimize.bounds.", 1)[1]
        for key, value in vars(args).items()
        if key.startswith("optimize.bounds.") and value is not None
    }
    apply_fine_tune_bounds(config, fine_tune_params, cli_bounds_overrides)
    if fine_tune_params:
        logging.info(
            "Fine-tuning mode active for %s",
            ", ".join(sorted(fine_tune_params)),
        )
    suite_override = None
    if args.suite_config:
        logging.info("loading suite config %s", args.suite_config)
        override_cfg = load_config(args.suite_config, verbose=False)
        override_backtest = override_cfg.get("backtest", {})
        # Support both new (scenarios at top level) and legacy (suite wrapper) formats
        if "scenarios" in override_backtest:
            suite_override = {
                "scenarios": override_backtest.get("scenarios", []),
                "aggregate": override_backtest.get("aggregate", {"default": "mean"}),
            }
        elif "suite" in override_backtest:
            # Legacy format - extract from suite wrapper
            suite_override = override_backtest["suite"]
        else:
            raise ValueError(f"Suite config {args.suite_config} must define backtest.scenarios.")
    suite_cfg = extract_suite_config(config, suite_override)

    # Handle --scenarios filter (implies --suite y)
    scenario_filter = getattr(args, "scenarios", None)
    if scenario_filter:
        labels = [label.strip() for label in scenario_filter.split(",") if label.strip()]
        suite_cfg["scenarios"] = filter_scenarios_by_label(suite_cfg.get("scenarios", []), labels)
        suite_cfg["enabled"] = True  # --scenarios implies suite mode
        logging.info("Filtered to %d scenario(s): %s", len(labels), ", ".join(labels))

    # --suite CLI arg overrides config (applied after --scenarios so explicit --suite n wins)
    if args.suite is not None:
        recursive_config_update(config, "backtest.suite_enabled", bool(args.suite), verbose=True)
        suite_cfg["enabled"] = bool(args.suite)
    backtest_exchanges = require_config_value(config, "backtest.exchanges")
    await format_approved_ignored_coins(config, backtest_exchanges)
    interrupted = False
    pool = None
    pool_terminated = False
    try:
        array_manager = SharedArrayManager()
        hlcvs_specs = {}
        btc_usd_specs = {}
        msss = {}
        timestamps_dict = {}
        config["backtest"]["coins"] = {}
        aggregate_cfg: Dict[str, Any] = {"default": "mean"}
        scenario_contexts: List[ScenarioEvalContext] = []
        suite_enabled = bool(suite_cfg.get("enabled"))

        if suite_enabled:
            scenario_contexts, aggregate_cfg = await prepare_suite_contexts(
                config,
                suite_cfg,
                shared_array_manager=array_manager,
            )
            if not scenario_contexts:
                raise ValueError("Suite configuration produced no scenarios.")
            logging.info("Optimizer suite enabled with %d scenario(s)", len(scenario_contexts))
            first_ctx = scenario_contexts[0]
            hlcvs_specs = first_ctx.hlcvs_specs
            btc_usd_specs = first_ctx.btc_usd_specs
            msss = first_ctx.msss
            timestamps_dict = first_ctx.timestamps
            config["backtest"]["coins"] = deepcopy(first_ctx.config["backtest"]["coins"])
            backtest_exchanges = sorted({ex for ctx in scenario_contexts for ex in ctx.exchanges})

            # Estimate memory usage (per-scenario SharedMemory, shared by all workers)
            total_shm_bytes = 0
            seen_specs = set()
            for ctx in scenario_contexts:
                for spec_map in (
                    ctx.hlcvs_specs,
                    ctx.btc_usd_specs,
                    ctx.master_hlcvs_specs or {},
                    ctx.master_btc_specs or {},
                ):
                    for spec in spec_map.values():
                        if spec is None:
                            continue
                        if spec.name in seen_specs:
                            continue
                        seen_specs.add(spec.name)
                        total_shm_bytes += np.prod(spec.shape) * np.dtype(spec.dtype).itemsize
            if total_shm_bytes > 0:
                total_shm_gb = total_shm_bytes / (1024**3)
                try:
                    import shutil
                    if hasattr(os, "sysconf"):
                        pages = os.sysconf("SC_PHYS_PAGES")
                        page_size = os.sysconf("SC_PAGE_SIZE")
                        available_gb = (pages * page_size) / (1024**3)
                    else:
                        available_gb = None
                    shm_gb = None
                    if os.path.exists("/dev/shm"):
                        usage = shutil.disk_usage("/dev/shm")
                        shm_gb = usage.total / (1024**3)
                except Exception:
                    available_gb = None
                    shm_gb = None
                logging.info(
                    "Memory estimate | scenarios=%d | shared_memory=%.1fGB%s",
                    len(scenario_contexts),
                    total_shm_gb,
                    f" | system={available_gb:.1f}GB" if available_gb else "",
                )
                if shm_gb is not None:
                    logging.info("Shared memory filesystem size | /dev/shm=%.1fGB", shm_gb)
                if available_gb and total_shm_gb > available_gb * 0.7:
                    logging.warning(
                        "Shared memory for scenarios (%.1fGB) is high relative to RAM (%.1fGB). "
                        "Consider using fewer/smaller scenarios.",
                        total_shm_gb,
                        available_gb,
                    )
        else:
            # New behavior: derive data strategy from exchange count
            # - Single exchange = use that exchange's data only
            # - Multiple exchanges = best-per-coin combination (combined)
            use_combined = len(backtest_exchanges) > 1

            if use_combined:
                exchange = "combined"
                coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, _timestamps = (
                    await prepare_hlcvs_mss(config, exchange)
                )
                hlcvs, _timestamps, btc_usd_prices = _maybe_aggregate_backtest_data(
                    hlcvs, _timestamps, btc_usd_prices, mss, config
                )
                timestamps_dict[exchange] = _timestamps
                exchange_preference = defaultdict(list)
                for coin in coins:
                    exchange_preference[mss[coin]["exchange"]].append(coin)
                for ex in exchange_preference:
                    logging.info(f"chose {ex} for {','.join(exchange_preference[ex])}")
                config["backtest"]["coins"][exchange] = coins
                msss[exchange] = mss
                validate_array(hlcvs, "hlcvs")
                hlcvs_array = np.ascontiguousarray(hlcvs, dtype=np.float64)
                hlcvs_spec, _ = array_manager.create_from(hlcvs_array)
                hlcvs_specs[exchange] = hlcvs_spec

                btc_usd_array = np.ascontiguousarray(btc_usd_prices, dtype=np.float64)
                validate_array(btc_usd_array, f"btc_usd_data for {exchange}", allow_nan=False)
                btc_usd_spec, _ = array_manager.create_from(btc_usd_array)
                btc_usd_specs[exchange] = btc_usd_spec
                del hlcvs, hlcvs_array, btc_usd_prices, btc_usd_array
            else:
                tasks = {}
                for exchange in backtest_exchanges:
                    tasks[exchange] = asyncio.create_task(prepare_hlcvs_mss(config, exchange))
                for exchange in backtest_exchanges:
                    coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, _timestamps = (
                        await tasks[exchange]
                    )
                    hlcvs, _timestamps, btc_usd_prices = _maybe_aggregate_backtest_data(
                        hlcvs, _timestamps, btc_usd_prices, mss, config
                    )
                    timestamps_dict[exchange] = _timestamps
                    config["backtest"]["coins"][exchange] = coins
                    msss[exchange] = mss
                    validate_array(hlcvs, "hlcvs")
                    hlcvs_array = np.ascontiguousarray(hlcvs, dtype=np.float64)
                    hlcvs_spec, _ = array_manager.create_from(hlcvs_array)
                    hlcvs_specs[exchange] = hlcvs_spec

                    btc_usd_array = np.ascontiguousarray(btc_usd_prices, dtype=np.float64)
                    validate_array(btc_usd_array, f"btc_usd_data for {exchange}", allow_nan=False)
                    btc_usd_spec, _ = array_manager.create_from(btc_usd_array)
                    btc_usd_specs[exchange] = btc_usd_spec
                    del hlcvs, hlcvs_array, btc_usd_prices, btc_usd_array
        exchanges = backtest_exchanges
        exchanges_fname = "combined" if len(backtest_exchanges) > 1 else "_".join(exchanges)
        date_fname = ts_to_date(utc_ms())[:19].replace(":", "_")
        coins = sorted(set([x for y in config["backtest"]["coins"].values() for x in y]))
        suite_flag = suite_enabled or bool(args.suite)
        if suite_flag:
            coins_fname = f"suite_{len(coins)}_coins"
        else:
            coins_fname = "_".join(coins) if len(coins) <= 6 else f"{len(coins)}_coins"
        hash_snippet = uuid4().hex[:8]
        n_days = int(
            round(
                (
                    date_to_ts(require_config_value(config, "backtest.end_date"))
                    - date_to_ts(require_config_value(config, "backtest.start_date"))
                )
                / (1000 * 60 * 60 * 24)
            )
        )
        results_dir = make_get_filepath(
            f"optimize_results/{date_fname}_{exchanges_fname}_{n_days}days_{coins_fname}_{hash_snippet}/"
        )
        os.makedirs(results_dir, exist_ok=True)
        config["results_dir"] = results_dir
        results_filename = os.path.join(results_dir, "all_results.bin")
        config["results_filename"] = results_filename
        overrides_list = config.get("optimize", {}).get("enable_overrides", [])

        # Shared state used by workers for duplicate detection
        manager = multiprocessing.Manager()
        seen_hashes = manager.dict()
        duplicate_counter = manager.dict()
        duplicate_counter["total"] = 0
        duplicate_counter["resolved"] = 0
        duplicate_counter["reused"] = 0

        # Initialize evaluator with shared memory references
        evaluator = Evaluator(
            hlcvs_specs=hlcvs_specs,
            btc_usd_specs=btc_usd_specs,
            msss=msss,
            config=config,
            seen_hashes=seen_hashes,
            duplicate_counter=duplicate_counter,
            timestamps=timestamps_dict,
            shared_array_manager=array_manager,
        )

        if suite_enabled:
            evaluator_for_pool = SuiteEvaluator(evaluator, scenario_contexts, aggregate_cfg)
        else:
            evaluator_for_pool = evaluator

        logging.info(f"Finished initializing evaluator...")
        flush_interval = 60  # or read from your config
        sig_digits = config["optimize"]["round_to_n_significant_digits"]
        pareto_max = config["optimize"].get("pareto_max_size", 300)
        recorder = ResultRecorder(
            results_dir=results_dir,
            sig_digits=sig_digits,
            flush_interval=flush_interval,
            scoring_keys=config["optimize"]["scoring"],
            compress=config["optimize"]["compress_results_file"],
            write_all_results=config["optimize"].get("write_all_results", True),
            pareto_max_size=pareto_max,
            bounds=evaluator.bounds,
        )

        n_objectives = len(config["optimize"]["scoring"])
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", ConstraintAwareFitness, weights=(-1.0,) * n_objectives)
        else:
            creator.FitnessMulti.weights = (-1.0,) * n_objectives
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        # Define parameter bounds
        bounds = evaluator.bounds
        sig_digits = config["optimize"]["round_to_n_significant_digits"]
        crossover_eta = config["optimize"].get("crossover_eta", 20.0)
        mutation_eta = config["optimize"].get("mutation_eta", 20.0)
        mutation_indpb_raw = config["optimize"].get("mutation_indpb", 0.0)
        if isinstance(mutation_indpb_raw, (int, float)) and mutation_indpb_raw > 0.0:
            mutation_indpb = max(0.0, min(1.0, float(mutation_indpb_raw)))
        else:
            mutation_indpb = 1.0 / len(bounds) if bounds else 1.0
        offspring_multiplier = config["optimize"].get("offspring_multiplier", 1.0)
        if not isinstance(offspring_multiplier, (int, float)) or offspring_multiplier <= 0.0:
            offspring_multiplier = 1.0

        # Register attribute generators (generating on-grid values for stepped params)
        def _make_random_attr(bound):
            """Generate a random value respecting step constraints."""
            return bound.random_on_grid()

        for i, bound in enumerate(bounds):
            toolbox.register(f"attr_{i}", _make_random_attr, bound)

        # Register genetic operators with bounds for step-aware crossover/mutation
        toolbox.register(
            "mate",
            cxSimulatedBinaryBoundedWrapper,
            eta=crossover_eta,
            bounds=bounds,
        )
        toolbox.register(
            "mutate",
            mutPolynomialBoundedWrapper,
            eta=mutation_eta,
            indpb=mutation_indpb,
            bounds=bounds,
        )
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", evaluator_for_pool.evaluate, overrides_list=overrides_list)

        # Parallelization setup
        logging.info(f"Initializing multiprocessing pool. N cpus: {config['optimize']['n_cpus']}")
        pool = multiprocessing.Pool(
            processes=config["optimize"]["n_cpus"],
            initializer=_ignore_sigint_in_worker,
        )
        toolbox.register("map", pool.map)
        logging.info(f"Finished initializing multiprocessing pool.")
        pool_state = {"terminated": False}

        # Create initial population
        logging.info(f"Creating initial population...")

        def _evaluate_initial(individuals):
            if not individuals:
                return 0
            total = len(individuals)
            pending = {}
            for ind in individuals:
                pending[pool.apply_async(toolbox.evaluate, (ind,))] = ind
            completed = 0
            try:
                while pending:
                    ready = [res for res in pending if res.ready()]
                    if not ready:
                        time.sleep(0.05)
                        continue
                    for res in ready:
                        ind = pending.pop(res)
                        fit_values, penalty, metrics = res.get()
                        ind.fitness.values = fit_values
                        ind.fitness.constraint_violation = penalty
                        ind.constraint_violation = penalty
                        if metrics is not None:
                            ind.evaluation_metrics = metrics
                            _record_individual_result(
                                ind,
                                evaluator.config,
                                overrides_list,
                                recorder,
                            )
                        elif hasattr(ind, "evaluation_metrics"):
                            delattr(ind, "evaluation_metrics")
                        completed += 1
                        logging.info("Evaluated %d/%d starting configs", completed, total)
            except KeyboardInterrupt:
                logging.info("Evaluation interrupted; terminating pending starting configs...")
                for res in pending:
                    try:
                        res.cancel()
                    except Exception:
                        pass
                if not pool_state["terminated"]:
                    logging.info("Terminating worker pool immediately due to interrupt...")
                    pool.terminate()
                    pool_state["terminated"] = True
                raise
            return completed

        population_size = config["optimize"]["population_size"]
        starting_configs = get_starting_configs(args.starting_configs)
        if starting_configs:
            logging.info(
                "Loaded %d starting configs before quantization (population size=%d)",
                len(starting_configs),
                population_size,
            )
        else:
            logging.info("No starting configs provided; population will be random-initialized")
        starting_individuals = configs_to_individuals(
            starting_configs,
            bounds,
            sig_digits,
        )

        def _make_random_individual():
            """Generate a random individual respecting step constraints."""
            values = [bound.random_on_grid() for bound in bounds]
            return creator.Individual(values)

        population = [_make_random_individual() for _ in range(population_size)]
        if starting_individuals:
            evaluated_seeds = [creator.Individual(ind) for ind in starting_individuals]
            eval_count = _evaluate_initial(evaluated_seeds)
            logging.info("Evaluated %d starting configs", eval_count)
            if len(evaluated_seeds) > population_size:
                evaluated_seeds = tools.selNSGA2(evaluated_seeds, population_size)
                logging.info(
                    "Trimmed starting configs to population size via NSGA-II crowding (kept %d)",
                    len(evaluated_seeds),
                )
            for i, ind in enumerate(evaluated_seeds):
                population[i] = creator.Individual(ind)

            remaining = population_size - len(evaluated_seeds)
            seed_pool = evaluated_seeds if evaluated_seeds else []
            if seed_pool and remaining > 0:
                for i in range(len(evaluated_seeds), len(evaluated_seeds) + remaining // 2):
                    population[i] = deepcopy(seed_pool[np.random.choice(range(len(seed_pool)))])
        for i in range(len(population)):
            population[i][:] = enforce_bounds(population[i], bounds, sig_digits)

        logging.info(f"Initial population size: {len(population)}")

        # Set up statistics and hall of fame
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        # stats.register("avg", np.mean, axis=0)
        # stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        # logbook.header = "gen", "evals", "std", "min", "avg", "max"
        logbook.header = "gen", "evals", "min", "max"

        hof = tools.ParetoFront()

        # Run the optimization
        logging.info(f"Starting optimize...")
        lambda_size = max(1, int(round(config["optimize"]["population_size"] * offspring_multiplier)))
        population, logbook = ea_mu_plus_lambda_stream(
            population,
            toolbox,
            mu=config["optimize"]["population_size"],
            lambda_=lambda_size,
            cxpb=config["optimize"]["crossover_probability"],
            mutpb=config["optimize"]["mutation_probability"],
            ngen=max(1, int(config["optimize"]["iters"] / len(population))),
            stats=stats,
            halloffame=hof,
            verbose=False,
            recorder=recorder,
            evaluator_config=evaluator.config,
            overrides_list=overrides_list,
            pool=pool,
            duplicate_counter=duplicate_counter,
            pool_state=pool_state,
        )

        logging.info("Optimization complete.")

        pool_terminated = pool_state["terminated"]

    except KeyboardInterrupt:
        interrupted = True
        logging.warning("Keyboard interrupt received; terminating optimization...")
        if "pool" in locals():
            already = pool_state["terminated"] if "pool_state" in locals() else pool_terminated
            if not already:
                logging.info("Terminating worker pool...")
                pool.terminate()
                pool_terminated = True
                if "pool_state" in locals():
                    pool_state["terminated"] = True
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if "recorder" in locals():
            try:
                recorder.flush()
            except Exception:
                logging.exception("Failed to flush recorder")
            recorder.close()
        if "pool" in locals() and pool is not None:
            if pool_terminated or interrupted:
                logging.info("Joining terminated worker pool...")
            else:
                logging.info("Closing worker pool...")
                pool.close()
            pool.join()
        if "array_manager" in locals():
            array_manager.cleanup()

        logging.info("Cleanup complete. Exiting.")
        sys.exit(130 if interrupted else 0)
>>>>>>> upstream/master


if __name__ == "__main__":
    asyncio.run(main())
