"""
DEAP (Distributed Evolutionary Algorithms in Python) reorganization module.

This module provides a clean, modular implementation of DEAP evolutionary algorithms
following the same organizational pattern as PSO and CMA-ES implementations.
"""

from .evolutionary_algorithm import DEAPEvolutionaryAlgorithm
from .interval import (
    MonthlyInterval,
    split_into_monthly_intervals,
    cleanup_interval_files,
    compute_adjusted_value,
    compute_monthly_fitness,
    build_interval_task_pool,
)
from .convergence import (
    ConvergenceState,
    initialize_convergence_state,
    update_convergence_state,
    get_convergence_log_dict,
)

__all__ = [
    'DEAPEvolutionaryAlgorithm',
    'MonthlyInterval',
    'split_into_monthly_intervals',
    'cleanup_interval_files',
    'compute_adjusted_value',
    'compute_monthly_fitness',
    'build_interval_task_pool',
    'ConvergenceState',
    'initialize_convergence_state',
    'update_convergence_state',
    'get_convergence_log_dict',
]
