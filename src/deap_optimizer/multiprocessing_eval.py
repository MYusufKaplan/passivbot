"""
Multiprocessing evaluation module for DEAP optimizer.

This module provides parallel fitness evaluation using multiprocessing,
following the PSO implementation pattern.
"""

import logging
import contextlib
from multiprocessing import Pool, cpu_count

# Rich imports for progress bars
try:
    from rich.progress import (
        Progress,
        BarColumn,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def evaluate_population_parallel(population, objective_function, n_processes=None):
    """
    Evaluate population in parallel using multiprocessing.
    
    This function uses pool.imap_unordered for efficient parallel evaluation
    while maintaining proper result ordering by including indices in the
    evaluation results.
    
    Parameters
    ----------
    population : list
        List of individuals to evaluate
    objective_function : callable
        Function to evaluate fitness. Should accept (individual, index) and
        return (fitness, index) to maintain ordering.
    n_processes : int, optional
        Number of processes to use (default: cpu_count - 1)
    
    Returns
    -------
    list
        List of fitness values in same order as population
    
    Requirements: 8.1, 8.2, 8.4
    """
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    
    # Create pool
    pool = Pool(processes=n_processes)
    
    try:
        # Pre-allocate results array to maintain ordering
        fitness_results = [None] * len(population)
        
        # Create evaluation arguments with indices
        eval_args = [(individual, idx) for idx, individual in enumerate(population)]
        
        # Evaluate using pool.imap_unordered for efficiency
        # Results come back in arbitrary order, but include index for proper placement
        for result in pool.imap_unordered(objective_function, eval_args):
            fitness_val, idx = result
            fitness_results[idx] = fitness_val  # Place fitness at correct index
        
        return fitness_results
    
    finally:
        # Clean up pool
        cleanup_multiprocessing_resources(pool)


def batch_evaluate_with_progress(individuals, evaluator, pool_size=None, 
                                 watch_path=None, console=None, verbose=True):
    """
    Batch evaluate individuals with Rich progress bar.
    
    This function uses pool.imap_unordered for efficient parallel evaluation
    while maintaining proper result ordering by including indices in the
    evaluation results.
    
    Parameters
    ----------
    individuals : list
        List of individuals to evaluate
    evaluator : callable
        Evaluation function. Should accept (individual, index) and return
        (fitness, index) to maintain ordering.
    pool_size : int, optional
        Number of processes (default: cpu_count - 1)
    watch_path : str, optional
        Path to log file for progress output
    console : Console, optional
        Rich Console object
    verbose : bool
        Whether to show progress bar
    
    Returns
    -------
    list
        List of fitness values in same order as individuals
    
    Requirements: 8.2, 8.3, 8.4
    """
    if pool_size is None:
        pool_size = max(1, cpu_count() - 1)
    
    pool = Pool(processes=pool_size)
    
    # Pre-allocate results array to maintain ordering
    fitness_results = [None] * len(individuals)
    
    # Create evaluation arguments with indices
    eval_args = [(individual, idx) for idx, individual in enumerate(individuals)]
    
    try:
        if verbose and RICH_AVAILABLE and console and watch_path:
            # Show progress bar with file logging
            with open(watch_path, "a") as f, \
                 contextlib.redirect_stdout(f), \
                 contextlib.redirect_stderr(f):
                
                with Progress(
                    SpinnerColumn(spinner_name="dots12"),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=None),
                    "•",
                    TaskProgressColumn(
                        text_format="[progress.percentage]{task.percentage:>5.1f}%",
                        show_speed=True,
                    ),
                    "•",
                    TimeElapsedColumn(),
                    "•",
                    TimeRemainingColumn(),
                    "•",
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task(
                        "Evaluating population...",
                        total=len(individuals),
                    )
                    
                    # Use imap_unordered for efficiency
                    # Results come back in arbitrary order, but include index for proper placement
                    for result in pool.imap_unordered(evaluator, eval_args):
                        fitness_val, idx = result
                        fitness_results[idx] = fitness_val  # Place fitness at correct index
                        progress.update(task, advance=1)
        else:
            # No progress bar - just evaluate
            for result in pool.imap_unordered(evaluator, eval_args):
                fitness_val, idx = result
                fitness_results[idx] = fitness_val
        
        return fitness_results
    
    finally:
        cleanup_multiprocessing_resources(pool)


def cleanup_multiprocessing_resources(pool):
    """
    Clean up multiprocessing pool resources.
    
    Parameters
    ----------
    pool : multiprocessing.Pool
        Pool to clean up
    
    Requirements: 6.4, 8.5
    """
    if pool is None:
        return
    
    try:
        pool.close()
        pool.join()
        logging.debug("Multiprocessing pool cleaned up successfully")
    except Exception as e:
        logging.error(f"Error cleaning up multiprocessing pool: {e}")
        try:
            pool.terminate()
            pool.join()
        except Exception as e2:
            logging.error(f"Error terminating pool: {e2}")
