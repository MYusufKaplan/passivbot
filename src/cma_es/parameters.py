"""
Parameter processing and normalization for CMA-ES optimization.
"""

import numpy as np


def _process_parameters_cma_es(parameter_bounds):
    """
    Process parameter bounds for CMA-ES optimization.
    
    Extracts optimizable parameters (those with min != max and not containing 'short'),
    fixed parameters, and identifies integer parameters.
    
    Args:
        parameter_bounds: Dictionary of {param_name: (min_val, max_val)}
        
    Returns:
        Tuple of:
            - all_param_names: List of all parameter names (ordered)
            - optimizable_bounds: Dict of {name: (min, max)} for optimizable params
            - fixed_params: Dict of {name: value} for fixed params
            - integer_params: Set of parameter names that should be integers
            - lower_bounds: np.ndarray of lower bounds for optimizable params
            - upper_bounds: np.ndarray of upper bounds for optimizable params
            
    Raises:
        ValueError: If any parameter has min > max (invalid bounds)
    """
    if not parameter_bounds:
        raise ValueError("parameter_bounds cannot be empty")
    
    optimizable_bounds = {}
    fixed_params = {}
    integer_params = set()
    
    # Validate bounds and separate optimizable vs fixed parameters
    for param_name, bounds in parameter_bounds.items():
        min_val, max_val = bounds
        
        # Validate bounds
        if min_val > max_val:
            raise ValueError(
                f"Invalid bounds for parameter '{param_name}': "
                f"min ({min_val}) > max ({max_val})"
            )
        
        # Check if parameter is optimizable
        if "short" not in param_name and min_val != max_val:
            # Parameter has range to optimize
            optimizable_bounds[param_name] = bounds
            
            # Check if parameter should be integer
            if any(keyword in param_name.lower() for keyword in ["n_positions"]):
                integer_params.add(param_name)
        else:
            # Fixed parameter (no range or contains 'short')
            fixed_params[param_name] = min_val
    
    # Create ordered lists
    all_param_names = list(parameter_bounds.keys())
    optimizable_param_names = list(optimizable_bounds.keys())
    
    # Create bounds arrays for optimization
    lower_bounds = np.array([optimizable_bounds[name][0] for name in optimizable_param_names])
    upper_bounds = np.array([optimizable_bounds[name][1] for name in optimizable_param_names])
    
    return (
        all_param_names,
        optimizable_bounds,
        fixed_params,
        integer_params,
        lower_bounds,
        upper_bounds,
    )


def normalize_solution(
    solution: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray
) -> np.ndarray:
    """
    Transform solution from original parameter bounds to normalized [0, 1] space.
    
    This normalization ensures all parameters have equal scale in the optimization
    space, preventing CMA-ES from favoring large-scale parameters.
    
    Args:
        solution: Solution vector in original parameter space
        lower_bounds: Lower bounds for each parameter
        upper_bounds: Upper bounds for each parameter
        
    Returns:
        Normalized solution in [0, 1] space
        
    Formula:
        normalized = (x - lower) / (upper - lower)
        
    Requirements: 4.2
    """
    # Validate inputs
    if len(solution) != len(lower_bounds) or len(solution) != len(upper_bounds):
        raise ValueError(
            f"Dimension mismatch: solution has {len(solution)} dimensions, "
            f"but bounds have {len(lower_bounds)} dimensions"
        )
    
    # Check for zero-range parameters
    ranges = upper_bounds - lower_bounds
    if np.any(ranges == 0):
        zero_indices = np.where(ranges == 0)[0]
        raise ValueError(
            f"Cannot normalize parameters with zero range at indices: {zero_indices.tolist()}"
        )
    
    # Normalize: (x - lower) / (upper - lower)
    normalized = (solution - lower_bounds) / ranges
    
    return normalized


def denormalize_solution(
    normalized_solution: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    integer_params_indices: list
) -> np.ndarray:
    """
    Transform solution from normalized [0, 1] space to original parameter bounds.
    
    This function reverses the normalization, converting from the uniform [0, 1]
    space used by CMA-ES back to the original parameter ranges. It also handles
    integer parameters by rounding and ensures all values are clamped to bounds.
    
    Args:
        normalized_solution: Solution vector in [0, 1] space
        lower_bounds: Lower bounds for each parameter
        upper_bounds: Upper bounds for each parameter
        integer_params_indices: List of indices for parameters that should be integers
        
    Returns:
        Denormalized solution in original parameter space
        
    Formula:
        denormalized = lower + x * (upper - lower)
        
    Then:
        - Round integer parameters to nearest integer
        - Clamp all values to [lower, upper] bounds
        
    Requirements: 4.3, 4.7, 4.8
    """
    # Validate inputs
    if len(normalized_solution) != len(lower_bounds) or len(normalized_solution) != len(upper_bounds):
        raise ValueError(
            f"Dimension mismatch: solution has {len(normalized_solution)} dimensions, "
            f"but bounds have {len(lower_bounds)} dimensions"
        )
    
    # Denormalize: lower + x * (upper - lower)
    denormalized = lower_bounds + normalized_solution * (upper_bounds - lower_bounds)
    
    # Round integer parameters
    for idx in integer_params_indices:
        if 0 <= idx < len(denormalized):
            denormalized[idx] = np.round(denormalized[idx])
    
    # Clamp all values to bounds
    denormalized = np.clip(denormalized, lower_bounds, upper_bounds)
    
    return denormalized
