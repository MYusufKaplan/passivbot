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
    
    # Parameter bounds for normalization (set during initialization)
    lower_bounds: Optional[np.ndarray] = None
    upper_bounds: Optional[np.ndarray] = None
    variable_mask: Optional[np.ndarray] = None  # True for variable params (low < high)


def initialize_convergence_state(
    parameter_bounds: Dict[str, Tuple[float, float]],
    epsilon_geom: float = 1e-6,
    epsilon_mid: float = 1e-4,
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
    stagnation_window : int
        Number of generations to look back for fitness improvement
    
    Returns
    -------
    ConvergenceState
        Initialized convergence state
    """
    bounds_list = list(parameter_bounds.values())
    lower = np.array([b[0] for b in bounds_list], dtype=np.float64)
    upper = np.array([b[1] for b in bounds_list], dtype=np.float64)
    
    # Identify variable parameters (where lower < upper)
    variable_mask = lower < upper
    
    return ConvergenceState(
        epsilon_geom=epsilon_geom,
        epsilon_mid=epsilon_mid,
        stagnation_window=stagnation_window,
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
    state: ConvergenceState
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
    
    # Check fitness stagnation
    stagnation, delta = check_fitness_stagnation(current_best_fitness, state)
    state.stagnation_detected = stagnation
    state.delta_fitness = delta
    
    # Update fitness history
    state.best_fitness_history.append(current_best_fitness)
    
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
    
    # Rule 2: Stagnation AND late exploitation → restart
    elif stagnation and lambda_max < state.epsilon_mid:
        should_restart = True
        reason = f"Stagnation in late exploitation (λ_max={lambda_max:.2e}, Δfit={delta:.2e})"
    
    # Rule 3: Continue
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
    }
