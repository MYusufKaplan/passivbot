"""
Monthly interval splitting and shared memory management for DEAP optimizer.

This module handles splitting historical datasets into calendar month intervals,
creating per-month shared memory files, and cleaning them up.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
import os
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta


@dataclass
class MonthlyInterval:
    """Represents a single monthly data interval."""
    month_id: int
    start_date: str          # ISO date string (YYYY-MM-DD)
    end_date: str            # ISO date string (YYYY-MM-DD)
    total_timesteps: int
    shared_memory_files: Dict[str, Tuple[str, str, tuple, np.dtype, np.dtype]] = field(
        default_factory=dict
    )
    # {exchange: (hlcv_file, btc_file, shape, hlcv_dtype, btc_dtype)}


def split_into_monthly_intervals(
    start_date: str,
    end_date: str,
    timestamps: np.ndarray,
    shared_hlcvs: Dict[str, np.ndarray],
    hlcvs_shapes: Dict[str, tuple],
    btc_usd_data: Dict[str, np.ndarray],
    btc_usd_dtypes: Dict[str, np.dtype],
    create_shared_memory_fn,
) -> List[MonthlyInterval]:
    """
    Split the full dataset into monthly intervals and create shared memory files.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    timestamps : np.ndarray
        Array of timestamps (ms since epoch) aligned with HLCV data.
    shared_hlcvs : dict
        {exchange: np.ndarray} — the full HLCV arrays.
    hlcvs_shapes : dict
        {exchange: shape} — shapes of full HLCV arrays.
    btc_usd_data : dict
        {exchange: np.ndarray} — BTC/USD price arrays per exchange.
    btc_usd_dtypes : dict
        {exchange: dtype} — BTC/USD data types per exchange.
    create_shared_memory_fn : callable
        Function to create a shared memory file from an ndarray.

    Returns
    -------
    list of MonthlyInterval

    Raises
    ------
    ValueError
        If the date range produces 0 intervals.
    """
    if len(timestamps) == 0:
        raise ValueError("timestamps array is empty")

    # Group indices by (year, month) of each timestamp
    # timestamps are ms since epoch
    month_groups = {}
    for idx, ts in enumerate(timestamps):
        dt = datetime.fromtimestamp(ts / 1000.0, tz=None)
        key = (dt.year, dt.month)
        if key not in month_groups:
            month_groups[key] = []
        month_groups[key].append(idx)

    # Sort by (year, month) and build intervals
    intervals = []
    for month_id, key in enumerate(sorted(month_groups.keys())):
        indices = month_groups[key]
        start_idx = indices[0]
        end_idx = indices[-1] + 1
        total_timesteps = end_idx - start_idx

        year, month = key
        m_start = datetime(year, month, 1)
        m_end = m_start + relativedelta(months=1)

        # Create shared memory files for each exchange
        smf = {}
        for exchange in shared_hlcvs:
            hlcv_slice = shared_hlcvs[exchange][start_idx:end_idx]
            btc_slice = btc_usd_data[exchange][start_idx:end_idx]

            hlcv_file = create_shared_memory_fn(hlcv_slice)
            btc_file = create_shared_memory_fn(btc_slice)

            smf[exchange] = (
                hlcv_file,
                btc_file,
                hlcv_slice.shape,
                hlcv_slice.dtype,
                btc_slice.dtype,
            )

        interval = MonthlyInterval(
            month_id=month_id,
            start_date=m_start.strftime("%Y-%m-%d"),
            end_date=m_end.strftime("%Y-%m-%d"),
            total_timesteps=total_timesteps,
            shared_memory_files=smf,
        )
        intervals.append(interval)

    return intervals


def cleanup_interval_files(intervals: List[MonthlyInterval]) -> None:
    """Remove all shared memory files created for intervals."""
    for interval in intervals:
        for exchange, (hlcv_file, btc_file, *_rest) in interval.shared_memory_files.items():
            for fpath in (hlcv_file, btc_file):
                if fpath and os.path.exists(fpath):
                    try:
                        os.unlink(fpath)
                        logging.info(f"Removed interval file: {fpath}")
                    except OSError as e:
                        logging.warning(f"Failed to remove interval file {fpath}: {e}")


def compute_adjusted_value(
    fitness_value: float,
    bankruptcy_flag: bool,
    bankruptcy_timestep: int,
    total_timesteps: int,
) -> float:
    """
    Compute the adjusted fitness value for a single month.

    Parameters
    ----------
    fitness_value : float
        The raw fitness value returned by Rust for this month.
    bankruptcy_flag : bool
        Whether the strategy went bankrupt during this month.
    bankruptcy_timestep : int
        The timestep at which bankruptcy occurred (if bankruptcy_flag is True).
        This is the occurrence timestep, not remaining timesteps.
    total_timesteps : int
        The total number of timesteps in this monthly interval.

    Returns
    -------
    float
        The adjusted fitness value:
        - If not bankrupt: fitness_value / total_timesteps
        - If bankrupt: remaining_timesteps × 1e6 (ignores fitness_value)
    """
    if not bankruptcy_flag:
        # Non-bankruptcy: normalize by timesteps
        return fitness_value / total_timesteps
    else:
        # Bankruptcy: compute penalty based on remaining timesteps
        # Clamp remaining_timesteps to 0 if bankruptcy_timestep > total_timesteps
        remaining_timesteps = max(0, total_timesteps - bankruptcy_timestep)
        penalty = remaining_timesteps * 1e6
        return penalty


def compute_monthly_fitness(
    candidate_results: Dict[int, Tuple[float, bool, int, int, int]],
    total_months: int,
) -> Tuple[float, float]:
    """
    Aggregate per-month results into final candidate fitness.

    Parameters
    ----------
    candidate_results : dict
        {month_id: (fitness_value, bankruptcy_flag, bankruptcy_reason, bankruptcy_timestep, total_timesteps)}
        Results for each month for a single candidate.
        bankruptcy_reason: 0=none, 1=financial, 2=drawdown, 3=no_positions, 4=stale_position
    total_months : int
        M — total number of months (used for averaging).

    Returns
    -------
    tuple
        (final_fitness, final_fitness) — fitness tuple for DEAP.
        Both values are identical to match the existing (w_0, w_1) pattern.
    """
    adjusted_sum = 0.0
    for month_id, result in candidate_results.items():
        # Handle both old format (4 elements) and new format (5 elements)
        if len(result) == 5:
            fitness_value, bankruptcy_flag, bankruptcy_reason, bankruptcy_timestep, total_timesteps = result
        else:
            # Old format without bankruptcy_reason
            fitness_value, bankruptcy_flag, bankruptcy_timestep, total_timesteps = result
        adjusted_value = compute_adjusted_value(
            fitness_value, bankruptcy_flag, bankruptcy_timestep, total_timesteps
        )
        adjusted_sum += adjusted_value

    final_fitness = adjusted_sum / total_months
    return (final_fitness, final_fitness)

def build_interval_task_pool(
    individuals: list,
    intervals: List[MonthlyInterval],
    evaluator,
) -> list:
    """
    Build the (candidate_id, month_id) task pool for parallel evaluation.

    Parameters
    ----------
    individuals : list
        List of individuals (candidates) to evaluate.
    intervals : list of MonthlyInterval
        List of monthly intervals to evaluate each candidate on.
    evaluator : Evaluator
        The evaluator instance to use for evaluation.

    Returns
    -------
    list
        List of task tuples: (evaluator, individual, month_id, interval, candidate_id)
        Contains exactly len(individuals) × len(intervals) tasks.
    """
    tasks = []
    for candidate_id, individual in enumerate(individuals):
        for interval in intervals:
            task = (evaluator, individual, interval.month_id, interval, candidate_id)
            tasks.append(task)
    return tasks


