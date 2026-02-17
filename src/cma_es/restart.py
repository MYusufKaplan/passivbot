"""
Restart management and termination criteria for CMA-ES.
"""

import numpy as np
from rich.console import Console
from .core import initialize_cma_state
from .parameters import normalize_solution
from .data_structures import CMAESState

console = Console(
    force_terminal=True,
    no_color=False,
    log_path=False,
    width=191,
    color_system="truecolor",
    legacy_windows=False,
)


def initialize_restart(
    n_dimensions: int,
    sigma0: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    optimizable_param_names: list,
    initial_values: dict = None,
    is_first_restart: bool = True
) -> tuple[np.ndarray, CMAESState]:
    """
    Initialize a fresh restart with new centroid and fresh CMA-ES state.
    
    For the first restart, this function attempts to load initial values from
    configs/optimize.json if available. For subsequent restarts, it generates
    a random centroid in normalized [0, 1] space.
    
    Each restart gets a completely fresh CMA-ES state with identity covariance
    matrix and configured sigma0, ensuring independence between restarts.
    
    Args:
        n_dimensions: Number of dimensions in the search space
        sigma0: Initial step size for CMA-ES
        lower_bounds: Lower bounds for each parameter (original space)
        upper_bounds: Upper bounds for each parameter (original space)
        optimizable_param_names: List of optimizable parameter names
        initial_values: Optional dict of initial values from config (for first restart)
        is_first_restart: Whether this is the first restart (default: True)
        
    Returns:
        Tuple of:
            - initial_centroid: Starting centroid in [0, 1] normalized space
            - fresh_state: Fresh CMA-ES state with identity covariance and sigma0
            
    Requirements: 2.3, 2.4, 9.6
    """
    if is_first_restart and initial_values:
        # First restart: use initial values from config if available
        centroid_original = []
        
        for i, param_name in enumerate(optimizable_param_names):
            if param_name in initial_values:
                # Use value from config
                value = initial_values[param_name]
                # Clamp to bounds
                value = np.clip(value, lower_bounds[i], upper_bounds[i])
                centroid_original.append(value)
            else:
                # Fallback to middle of bounds
                centroid_original.append((lower_bounds[i] + upper_bounds[i]) / 2.0)
        
        centroid_original = np.array(centroid_original)
        
        # Normalize to [0, 1] space
        initial_centroid = normalize_solution(centroid_original, lower_bounds, upper_bounds)
        
        console.print(
            f"[cyan]📍 First restart: Using initial values from config[/cyan]"
        )
    else:
        # Subsequent restarts: random centroid in [0, 1] space
        initial_centroid = np.random.rand(n_dimensions)
        
        console.print(
            f"[cyan]📍 Restart: Random centroid in [0, 1] space[/cyan]"
        )
    
    # Create fresh CMA-ES state with identity covariance and sigma0
    fresh_state = initialize_cma_state(n_dimensions, initial_centroid, sigma0)
    
    return initial_centroid, fresh_state


def check_termination_criteria(
    state: CMAESState,
    fitness_history: list,
    population_fitness: np.ndarray,
    max_iter_per_restart: int,
    tol_hist_fun: float = 1e-12,
    equal_fun_vals_k: int = None,
    tol_x: float = 1e-11,
    tol_up_sigma: float = 1e20,
    stagnation_iter: int = 100,
    condition_cov: float = 1e14,
    sigma0: float = 0.1,
    gens_since_restart_improvement: int = 0,
    min_sigma: float = None,
) -> tuple[bool, list[str]]:
    """
    Check all nine termination criteria for CMA-ES restart.
    
    This function implements the standard CMA-ES termination criteria to detect
    when a restart has converged or stagnated and should be terminated.
    
    Args:
        state: Current CMA-ES state
        fitness_history: List of best fitness values from previous generations
        population_fitness: Fitness values for current population
        max_iter_per_restart: Maximum iterations per restart (MaxIter criterion)
        tol_hist_fun: Tolerance for fitness history (TolHistFun criterion)
        equal_fun_vals_k: Number of top solutions to check for equality (EqualFunVals)
        tol_x: Tolerance for step sizes (TolX criterion)
        tol_up_sigma: Upper threshold for sigma ratio (TolUpSigma criterion)
        stagnation_iter: Iterations for stagnation check (Stagnation criterion)
        condition_cov: Condition number threshold (ConditionCov criterion)
        sigma0: Initial sigma value for ratio calculation
        gens_since_restart_improvement: Generations since last restart improvement
        min_sigma: Minimum sigma threshold (MinSigma criterion)
        
    Returns:
        Tuple of (should_terminate, list_of_triggered_conditions)
        
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9
    """
    n = state.n_dimensions
    triggered_conditions = []
    
    # Default equal_fun_vals_k if not provided
    if equal_fun_vals_k is None:
        equal_fun_vals_k = len(population_fitness) // 3
    
    # 1. MaxIter: Maximum iterations reached
    # Requirement 3.1
    if state.generation >= max_iter_per_restart:
        triggered_conditions.append("MaxIter")
    
    # 2. TolHistFun: Best fitness unchanged over window
    # Requirement 3.2
    window_size = 10 + int(np.ceil(30.0 * n / len(population_fitness)))
    if len(fitness_history) >= window_size:
        recent_history = fitness_history[-window_size:]
        fitness_range = max(recent_history) - min(recent_history)
        if fitness_range < tol_hist_fun:
            triggered_conditions.append("TolHistFun")
    
    # 3. EqualFunVals: Top K solutions have identical fitness
    # Requirement 3.3
    sorted_fitness = np.sort(population_fitness)
    k = min(equal_fun_vals_k, len(sorted_fitness))
    if k > 0:
        top_k_fitness = sorted_fitness[:k]
        # Check if all top K values are identical (within floating point precision)
        if len(set(np.round(top_k_fitness, decimals=12))) == 1:
            triggered_conditions.append("EqualFunVals")
    
    # 4. TolX: All step sizes below threshold
    # Requirement 3.4
    # Step size for each dimension: sigma * sqrt(C[i,i])
    step_sizes = state.sigma * np.sqrt(np.diag(state.C))
    if np.all(step_sizes < tol_x):
        triggered_conditions.append("TolX")
    
    # 5. TolUpSigma: Sigma ratio exceeds threshold
    # Requirement 3.5
    sigma_ratio = state.sigma / sigma0
    if sigma_ratio > tol_up_sigma:
        triggered_conditions.append("TolUpSigma")
    
    # 6. Stagnation: No restart improvement for stagnation_iter generations
    # Requirement 3.6
    # Use the restart improvement counter directly instead of median fitness comparison
    if gens_since_restart_improvement >= stagnation_iter:
        triggered_conditions.append("Stagnation")
    
    # 6b. MinSigma: Sigma below minimum threshold (converged)
    # Custom criterion for restart-based CMA-ES
    if min_sigma is not None and state.sigma < min_sigma:
        triggered_conditions.append("MinSigma")
    
    # 7. ConditionCov: Covariance matrix condition number exceeds threshold
    # Requirement 3.7
    # Condition number = max(eigenvalue) / min(eigenvalue)
    # We have D = sqrt(eigenvalues), so condition = max(D)^2 / min(D)^2 = (max(D) / min(D))^2
    if len(state.D) > 0:
        max_d = np.max(state.D)
        min_d = np.min(state.D)
        if min_d > 0:  # Avoid division by zero
            condition_number = (max_d / min_d) ** 2
            if condition_number > condition_cov:
                triggered_conditions.append("ConditionCov")
    
    # 8. NoEffectAxis: Mutations along principal axis have no effect
    # Requirement 3.8
    # Test if adding a mutation along the principal axis changes the centroid
    epsilon = 1e-20  # Machine epsilon for comparison
    for i in range(n):
        # Mutation along i-th principal axis
        test_point = state.centroid + 0.1 * state.sigma * state.D[i] * state.B[:, i]
        # Check if test_point is effectively equal to centroid
        if np.all(np.abs(test_point - state.centroid) < epsilon):
            triggered_conditions.append("NoEffectAxis")
            break  # Only need to find one axis
    
    # 9. NoEffectCoor: Mutations along coordinate axes have no effect
    # Requirement 3.9
    # Test if adding a mutation along coordinate axes changes the centroid
    for i in range(n):
        # Mutation along i-th coordinate axis
        e_i = np.zeros(n)
        e_i[i] = 1.0
        test_point = state.centroid + 0.2 * state.sigma * np.sqrt(state.C[i, i]) * e_i
        # Check if test_point is effectively equal to centroid
        if np.all(np.abs(test_point - state.centroid) < epsilon):
            triggered_conditions.append("NoEffectCoor")
            break  # Only need to find one coordinate
    
    # Determine if should terminate
    should_terminate = len(triggered_conditions) > 0
    
    return should_terminate, triggered_conditions
