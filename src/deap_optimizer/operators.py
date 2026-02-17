"""
Custom DEAP genetic operators module.

This module contains custom genetic operators for DEAP including
mutation and crossover functions with boundary handling.

Requirements: 1.5, 5.1, 5.2
"""

import numpy as np
from deap import base, creator, tools


def mutPolynomialBoundedWrapper(individual, eta, low, up, indpb, param_bounds=None):
    """
    A wrapper around DEAP's mutPolynomialBounded function to pre-process
    bounds and handle the case where lower and upper bounds may be equal.
    Also handles discrete parameters like n_positions.

    Args:
        individual: Sequence individual to be mutated.
        eta: Crowding degree of the mutation.
        low: A value or sequence of values that is the lower bound of the search space.
        up: A value or sequence of values that is the upper bound of the search space.
        indpb: Independent probability for each attribute to be mutated.
        param_bounds: Dictionary of parameter names to bounds for discrete handling.

    Returns:
        A tuple of one individual, mutated with consideration for equal lower and upper bounds.
    
    Requirements: 1.5, 5.1, 5.2
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

    # Handle discrete parameters
    if param_bounds:
        param_names = list(param_bounds.keys())
        for i, param_name in enumerate(param_names):
            if param_name == "long_n_positions" and np.random.random() < indpb:
                # For n_positions, use discrete choice instead of continuous mutation
                low_val, high_val = param_bounds[param_name]
                choices = list(range(int(low_val), int(high_val) + 1))
                individual[i] = float(np.random.choice(choices))

    return (individual,)


def cxSimulatedBinaryBoundedWrapper(ind1, ind2, eta, low, up):
    """
    A wrapper around DEAP's cxSimulatedBinaryBounded function to pre-process
    bounds and handle the case where lower and upper bounds may be equal.
    
    This wrapper also works around a bug in DEAP's cxSimulatedBinaryBounded
    where it tries to access index 19 regardless of the actual parameter count.

    Args:
        ind1: The first individual participating in the crossover.
        ind2: The second individual participating in the crossover.
        eta: Crowding degree of the crossover.
        low: A value or sequence of values that is the lower bound of the search space.
        up: A value or sequence of values that is the upper bound of the search space.

    Returns:
        A tuple of two individuals after crossover operation.
    
    Requirements: 1.5, 5.1, 5.2
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
    
    # Work around DEAP bug: cxSimulatedBinaryBounded tries to access index 19
    # regardless of actual parameter count. Pad arrays if needed.
    original_size = len(low_array)
    if original_size < 20:
        # Pad to at least 20 elements
        padding_size = 20 - original_size
        low_array = np.concatenate([low_array, np.zeros(padding_size)])
        up_array = np.concatenate([up_array, np.ones(padding_size)])
        
        # Extend individuals temporarily
        ind1.extend([0.5] * padding_size)
        ind2.extend([0.5] * padding_size)

    # Call the original cxSimulatedBinaryBounded function with adjusted bounds
    tools.cxSimulatedBinaryBounded(ind1, ind2, eta, list(low_array), list(up_array))
    
    # If we padded, restore original size
    if original_size < 20:
        del ind1[original_size:]
        del ind2[original_size:]

    # Ensure that values in dimensions with originally equal bounds are reset
    # to the bound value (since they should not be modified)
    for i, equal in enumerate(equal_bounds_mask):
        if equal:
            ind1[i] = low[i]
            ind2[i] = low[i]

    return (ind1, ind2)


def create_toolbox(param_bounds, crossover_prob=0.5, mutation_prob=0.2, eta=20.0):
    """
    Create and configure DEAP toolbox with genetic operators.
    
    This function creates a DEAP toolbox configured with:
    - Individual and population initialization
    - Genetic operators (crossover and mutation) with boundary constraints
    - Selection operator (NSGA-II)
    
    Parameters
    ----------
    param_bounds : dict
        Dictionary of parameter bounds {param_name: (min, max)}
    crossover_prob : float, optional
        Crossover probability (default: 0.5)
    mutation_prob : float, optional
        Mutation probability (default: 0.2)
    eta : float, optional
        Crowding degree for mutation and crossover (default: 20.0)
    
    Returns
    -------
    deap.base.Toolbox
        Configured DEAP toolbox with registered operators
    
    Requirements: 5.1, 5.2
    """
    toolbox = base.Toolbox()
    
    # Ensure FitnessMulti and Individual are created
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    # Register attribute generators for each parameter
    for i, (param_name, (low, high)) in enumerate(param_bounds.items()):
        if param_name == "long_n_positions":
            # Use discrete choice for n_positions to give equal probability to each integer
            choices = list(range(int(low), int(high) + 1))
            toolbox.register(
                f"attr_{i}", lambda choices=choices: float(np.random.choice(choices))
            )
        else:
            toolbox.register(f"attr_{i}", np.random.uniform, low, high)
    
    # Register individual and population creation
    def create_individual():
        return creator.Individual(
            [getattr(toolbox, f"attr_{i}")() for i in range(len(param_bounds))]
        )
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register genetic operators with boundary constraints
    toolbox.register(
        "mate",
        cxSimulatedBinaryBoundedWrapper,
        eta=eta,
        low=[low for low, high in param_bounds.values()],
        up=[high for low, high in param_bounds.values()],
    )
    toolbox.register(
        "mutate",
        mutPolynomialBoundedWrapper,
        eta=eta,
        low=[low for low, high in param_bounds.values()],
        up=[high for low, high in param_bounds.values()],
        indpb=1.0 / len(param_bounds),
        param_bounds=param_bounds,
    )
    
    # Register selection operator
    toolbox.register("select", tools.selNSGA2)
    
    return toolbox
