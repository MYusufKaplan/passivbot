"""
Covariance Eigenvalue-based Convergence Detection for DEAP Optimizer.

This module implements geometric convergence detection using covariance matrix
eigenvalues, combined with fitness-based stagnation detection.

Key concepts:
- Geometric convergence: Population has collapsed in parameter space (lambda_max < epsilon_geom)
- Fitness stagnation: No improvement in best fitness over a window of generations
- Safe termination: Only terminate when geometrically converged OR (stagnated AND in late exploitation)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ConvergenceState:
    """Tracks convergence detection state across generations."""
    # Eigenvalue thresholds
    epsilon_geom: float = 1e-6      # Full geometric convergence threshold
    epsilon_mid: float = 1e-4       # Late exploitation threshold
    epsilon_dim: float = 1e-10      # Threshold for counting active dimensions
    
    # Fitness variance collapse threshold
    # When fitness std dev falls below this, population has converged in fitness space
    # even if still spread in parameter space (plateau detection)
    epsilon_fitness_std: float = 1e-9  # Fitness std dev collapse threshold
    
    # Per-parameter variance threshold (normalized to [0,1] space)
    # Parameters with variance below this are considered "collapsed"
    epsilon_param_var: float = 1e-6  # Per-parameter collapse threshold
    
    # Fitness stagnation parameters
    stagnation_window: int = 30     # Generations to look back for improvement
    
    # History tracking
    best_fitness_history: List[float] = field(default_factory=list)
    lambda_max_history: List[float] = field(default_factory=list)
    lambda_min_history: List[float] = field(default_factory=list)
    active_dims_history: List[int] = field(default_factory=list)
    
    # Current state
    lambda_max: float = float('inf')
    lambda_min: float = 0.0
    active_dims: int = 0
    stagnation_detected: bool = False
    convergence_detected: bool = False
    delta_fitness: float = 0.0
    
    # Per-parameter variance tracking (normalized space)
    param_variances: Optional[np.ndarray] = None  # Variance per parameter
    active_params: Optional[List[str]] = None     # Names of params still being explored
    collapsed_params: Optional[List[str]] = None  # Names of params that collapsed
    
    # Parameter names (for logging)
    param_names: Optional[List[str]] = None
    
    # Parameter bounds for normalization (set during initialization)
    lower_bounds: Optional[np.ndarray] = None
    upper_bounds: Optional[np.ndarray] = None
    variable_mask: Optional[np.ndarray] = None  # True for variable params (low < high)


def initialize_convergence_state(
    parameter_bounds: Dict[str, Tuple[float, float]],
    epsilon_geom: float = 1e-6,
    epsilon_mid: float = 1e-4,
    epsilon_fitness_std: float = 1e-9,
    epsilon_param_var: float = 1e-6,
    stagnation_window: int = 30
) -> ConvergenceState:
    """
    Initialize convergence detection state with parameter bounds.
    
    Parameters
    ----------
    parameter_bounds : dict
        {param_name: (lower, upper)} bounds for each parameter
    epsilon_geom : float
        Threshold for full geometric convergence (lambda_max < epsilon_geom)
    epsilon_mid : float
        Threshold for late exploitation phase
    epsilon_fitness_std : float
        Threshold for fitness variance collapse (std dev < epsilon_fitness_std)
    epsilon_param_var : float
        Threshold for per-parameter collapse (variance < epsilon_param_var)
    stagnation_window : int
        Number of generations to look back for fitness improvement
    
    Returns
    -------
    ConvergenceState
        Initialized convergence state
    """
    param_names = list(parameter_bounds.keys())
    bounds_list = list(parameter_bounds.values())
    lower = np.array([b[0] for b in bounds_list], dtype=np.float64)
    upper = np.array([b[1] for b in bounds_list], dtype=np.float64)
    
    # Identify variable parameters (where lower < upper)
    variable_mask = lower < upper
    
    return ConvergenceState(
        epsilon_geom=epsilon_geom,
        epsilon_mid=epsilon_mid,
        epsilon_fitness_std=epsilon_fitness_std,
        epsilon_param_var=epsilon_param_var,
        stagnation_window=stagnation_window,
        param_names=param_names,
        lower_bounds=lower,
        upper_bounds=upper,
        variable_mask=variable_mask
    )


def normalize_population(
    population: List,
    state: ConvergenceState
) -> np.ndarray:
    """
    Normalize population parameters to [0, 1] using bounds.
    
    Only normalizes variable parameters (where lower < upper).
    Fixed parameters are set to 0.5 (midpoint).
    
    Parameters
    ----------
    population : list
        List of DEAP individuals
    state : ConvergenceState
        Convergence state with bounds
    
    Returns
    -------
    np.ndarray
        Normalized population matrix (N x D)
    """
    # Extract population matrix
    X = np.array([list(ind) for ind in population], dtype=np.float64)
    N, D = X.shape
    
    # Normalize to [0, 1]
    X_norm = np.zeros_like(X)
    
    for i in range(D):
        if state.variable_mask[i]:
            # Variable parameter: normalize
            range_i = state.upper_bounds[i] - state.lower_bounds[i]
            X_norm[:, i] = (X[:, i] - state.lower_bounds[i]) / range_i
        else:
            # Fixed parameter: set to midpoint
            X_norm[:, i] = 0.5
    
    return X_norm


def compute_covariance_eigenvalues(
    X_norm: np.ndarray,
    state: ConvergenceState
) -> Tuple[float, float, int, np.ndarray]:
    """
    Compute covariance matrix eigenvalues from normalized population.
    
    Parameters
    ----------
    X_norm : np.ndarray
        Normalized population matrix (N x D)
    state : ConvergenceState
        Convergence state with thresholds
    
    Returns
    -------
    tuple
        (lambda_max, lambda_min, active_dims, all_eigenvalues)
    """
    N, D = X_norm.shape
    
    # Only use variable dimensions for covariance
    variable_indices = np.where(state.variable_mask)[0]
    
    if len(variable_indices) == 0:
        # All parameters fixed - fully converged by definition
        return 0.0, 0.0, 0, np.array([0.0])
    
    X_var = X_norm[:, variable_indices]
    D_var = len(variable_indices)
    
    # Compute mean and center
    mu = np.mean(X_var, axis=0)
    X_centered = X_var - mu
    
    # Compute covariance matrix
    # Using (1/N) instead of (1/(N-1)) for consistency with spec
    cov = (1.0 / N) * (X_centered.T @ X_centered)
    
    # Force symmetry for numerical stability
    cov = 0.5 * (cov + cov.T)
    
    # Eigenvalue decomposition (symmetric)
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        # Fallback if eigendecomposition fails
        return float('inf'), 0.0, D_var, np.array([float('inf')])
    
    # Sort descending
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Clamp numerical noise (small negative values to zero)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    
    lambda_max = eigenvalues[0]
    lambda_min = eigenvalues[-1]
    active_dims = np.sum(eigenvalues > state.epsilon_dim)
    
    return lambda_max, lambda_min, int(active_dims), eigenvalues


def compute_per_parameter_variance(
    X_norm: np.ndarray,
    state: ConvergenceState
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute variance per parameter and classify as active or collapsed.
    
    Parameters
    ----------
    X_norm : np.ndarray
        Normalized population matrix (N x D)
    state : ConvergenceState
        Convergence state with thresholds and param names
    
    Returns
    -------
    tuple
        (variances, active_params, collapsed_params)
        - variances: Per-parameter variance in normalized [0,1] space
        - active_params: Names of parameters still being explored
        - collapsed_params: Names of parameters that have collapsed
    """
    N, D = X_norm.shape
    
    # Compute variance per dimension
    variances = np.var(X_norm, axis=0)
    
    active_params = []
    collapsed_params = []
    
    for i in range(D):
        param_name = state.param_names[i] if state.param_names else f"param_{i}"
        
        if not state.variable_mask[i]:
            # Fixed parameter (low == high), skip
            continue
        
        if variances[i] > state.epsilon_param_var:
            active_params.append(param_name)
        else:
            collapsed_params.append(param_name)
    
    return variances, active_params, collapsed_params


def check_fitness_stagnation(
    current_best: float,
    state: ConvergenceState
) -> Tuple[bool, float]:
    """
    Check if fitness has stagnated over the stagnation window.
    
    Parameters
    ----------
    current_best : float
        Best fitness of current generation
    state : ConvergenceState
        Convergence state with history
    
    Returns
    -------
    tuple
        (stagnation_detected, delta_fitness)
    """
    history = state.best_fitness_history
    window = state.stagnation_window
    
    if len(history) < window:
        # Not enough history yet
        return False, 0.0
    
    # Get best from window generations ago
    best_past = min(history[-window:])
    
    # Delta: improvement is positive (minimization)
    delta = best_past - current_best
    
    # Relative tolerance
    epsilon_fit = max(1e-8, 1e-6 * abs(current_best))
    
    # Stagnation if no improvement beyond tolerance
    stagnation = delta <= epsilon_fit
    
    return stagnation, delta


def update_convergence_state(
    population: List,
    current_best_fitness: float,
    state: ConvergenceState,
    fitness_values: Optional[List[float]] = None
) -> Tuple[bool, bool, str]:
    """
    Update convergence state and check termination conditions.
    
    Parameters
    ----------
    population : list
        Current population of DEAP individuals
    current_best_fitness : float
        Best fitness of current generation
    state : ConvergenceState
        Convergence state to update
    fitness_values : list, optional
        List of fitness values for all individuals. If None, extracted from population.
    
    Returns
    -------
    tuple
        (should_terminate, should_restart, reason)
        - should_terminate: True if optimization should stop completely
        - should_restart: True if population should be restarted with LHS
        - reason: Human-readable explanation
    """
    # Normalize population
    X_norm = normalize_population(population, state)
    
    # Compute eigenvalues
    lambda_max, lambda_min, active_dims, _ = compute_covariance_eigenvalues(X_norm, state)
    
    # Update state
    state.lambda_max = lambda_max
    state.lambda_min = lambda_min
    state.active_dims = active_dims
    state.lambda_max_history.append(lambda_max)
    state.lambda_min_history.append(lambda_min)
    state.active_dims_history.append(active_dims)
    
    # Compute per-parameter variances
    variances, active_params, collapsed_params = compute_per_parameter_variance(X_norm, state)
    state.param_variances = variances
    state.active_params = active_params
    state.collapsed_params = collapsed_params
    
    # Check fitness stagnation
    stagnation, delta = check_fitness_stagnation(current_best_fitness, state)
    state.stagnation_detected = stagnation
    state.delta_fitness = delta
    
    # Update fitness history
    state.best_fitness_history.append(current_best_fitness)
    
    # Calculate fitness std dev for variance collapse detection
    if fitness_values is None:
        fitness_values = [ind.fitness.values[0] for ind in population if hasattr(ind, 'fitness') and ind.fitness.valid]
    
    fitness_std = np.std(fitness_values) if len(fitness_values) > 1 else 0.0
    
    # Determine phase
    # lambda_max > epsilon_mid → Exploration phase
    # epsilon_geom < lambda_max <= epsilon_mid → Late exploitation
    # lambda_max <= epsilon_geom → Full collapse
    
    should_terminate = False
    should_restart = False
    reason = ""
    
    # Rule 1: Full geometric convergence → restart
    if lambda_max < state.epsilon_geom:
        state.convergence_detected = True
        should_restart = True
        reason = f"Geometric convergence (λ_max={lambda_max:.2e} < ε_geom={state.epsilon_geom:.0e})"
    
    # Rule 2: Fitness variance collapse → restart
    # This catches plateaus where population is spread in parameter space but
    # all individuals have essentially identical fitness
    elif fitness_std < state.epsilon_fitness_std:
        should_restart = True
        reason = f"Fitness variance collapse (std={fitness_std:.2e} < ε_fit={state.epsilon_fitness_std:.0e}, λ_max={lambda_max:.2e})"
    
    # Rule 3: Stagnation AND late exploitation → restart
    elif stagnation and lambda_max < state.epsilon_mid:
        should_restart = True
        reason = f"Stagnation in late exploitation (λ_max={lambda_max:.2e}, Δfit={delta:.2e})"
    
    # Rule 4: Continue
    else:
        if stagnation:
            reason = f"Stagnation but still exploring (λ_max={lambda_max:.2e} > ε_mid={state.epsilon_mid:.0e})"
        else:
            reason = f"Progressing (λ_max={lambda_max:.2e}, Δfit={delta:.2e})"
    
    return should_terminate, should_restart, reason


def get_convergence_log_dict(state: ConvergenceState) -> Dict:
    """
    Get convergence metrics for logging.
    
    Returns
    -------
    dict
        Dictionary of convergence metrics
    """
    return {
        "lambda_max": state.lambda_max,
        "lambda_min": state.lambda_min,
        "active_dims": state.active_dims,
        "stagnation": state.stagnation_detected,
        "convergence": state.convergence_detected,
        "delta_fitness": state.delta_fitness,
        "active_params": state.active_params or [],
        "collapsed_params": state.collapsed_params or [],
    }


def save_neutrality_snapshot(
    state: ConvergenceState,
    population: List,
    fitness_std: float,
    restart_reason: str,
    output_path: str,
    generation: int,
    global_best_fitness: float
) -> None:
    """
    Save a snapshot of parameter neutrality when fitness variance collapses.
    
    This data can be analyzed later to understand conditional parameter relevance -
    i.e., which parameters become irrelevant given certain values of other parameters.
    
    Parameters
    ----------
    state : ConvergenceState
        Current convergence state with param info
    population : list
        Current population of individuals
    fitness_std : float
        Current fitness standard deviation
    restart_reason : str
        Why the restart was triggered
    output_path : str
        Path to save the JSON file
    generation : int
        Current generation number
    global_best_fitness : float
        Best fitness found so far
    """
    import json
    import os
    from datetime import datetime
    
    # Get the best individual's parameter values as the "context"
    best_ind = min(population, key=lambda x: x.fitness.values[0] if x.fitness.valid else float('inf'))
    
    # Build context dict with converged param values
    context_params = {}
    if state.param_names and state.param_variances is not None:
        for i, name in enumerate(state.param_names):
            if state.variable_mask is not None and state.variable_mask[i]:
                # Get the mean value for this param across population
                param_values = [ind[i] for ind in population]
                context_params[name] = {
                    "best_value": float(best_ind[i]),
                    "mean_value": float(np.mean(param_values)),
                    "std_value": float(np.std(param_values)),
                    "variance_normalized": float(state.param_variances[i]),
                    "is_neutral": name in (state.active_params or []),
                    "is_converged": name in (state.collapsed_params or [])
                }
    
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "generation": generation,
        "restart_reason": restart_reason,
        "fitness_std": float(fitness_std),
        "global_best_fitness": float(global_best_fitness),
        "lambda_max": float(state.lambda_max),
        "active_dims": state.active_dims,
        "neutral_params": state.active_params or [],  # Params still spread but don't affect fitness
        "converged_params": state.collapsed_params or [],  # Params that have converged
        "param_context": context_params  # Full context for analysis
    }
    
    # Load existing file or create new structure
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            # Ensure snapshots array exists
            if "snapshots" not in data:
                data["snapshots"] = []
        except (json.JSONDecodeError, IOError):
            data = None
    else:
        data = None
    
    if data is None:
        data = {
            "description": "Parameter Neutrality Analysis Dataset",
            "purpose": """This dataset captures snapshots of parameter states when the optimizer 
detects fitness variance collapse - meaning all individuals have essentially identical fitness 
despite having different parameter values. 

When this happens, parameters that are still 'spread out' (high variance) but produce identical 
fitness are considered 'neutral' - they don't affect the fitness function in the current region 
of the search space.

This is useful for understanding CONDITIONAL parameter relevance. For example, 'short_ema_span' 
might be neutral when 'short_qty_pct' is near zero (because shorts are effectively disabled), 
but highly relevant when 'short_qty_pct' is large.

By analyzing patterns across many snapshots, an AI agent can:
1. Identify which parameters gate/control other parameters
2. Suggest reducing search space by fixing neutral parameters
3. Build a dependency graph of parameter interactions
4. Detect when the optimizer is stuck on a plateau vs genuinely converged""",
            "schema": {
                "timestamp": "ISO timestamp of when snapshot was taken",
                "generation": "Generation number when restart occurred",
                "restart_reason": "Why the optimizer triggered a restart",
                "fitness_std": "Standard deviation of fitness across population (near 0 = collapsed)",
                "global_best_fitness": "Best fitness found so far",
                "lambda_max": "Largest eigenvalue of covariance matrix (geometric spread)",
                "active_dims": "Number of dimensions with significant variance",
                "neutral_params": "Parameters with high variance but no fitness impact (don't matter here)",
                "converged_params": "Parameters that have converged to similar values",
                "param_context": "Detailed per-parameter statistics for correlation analysis"
            },
            "snapshots": []
        }
    
    data["snapshots"].append(snapshot)
    
    # Save back to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
