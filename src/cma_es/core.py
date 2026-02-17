"""
Core CMA-ES algorithm implementation.
"""

import numpy as np
from .data_structures import CMAESState


def initialize_cma_state(
    n_dimensions: int,
    initial_centroid: np.ndarray,
    sigma0: float
) -> CMAESState:
    """
    Initialize fresh CMA-ES state with identity covariance.
    
    This function creates a new CMA-ES state for a restart, setting up all
    the algorithm parameters according to the standard CMA-ES formulation
    from Hansen & Ostermeier (2001).
    
    Args:
        n_dimensions: Number of dimensions in the search space
        initial_centroid: Starting centroid in [0, 1] normalized space
        sigma0: Initial step size (e.g., 0.1 for 10% of normalized range)
        
    Returns:
        CMAESState: Fresh CMA-ES state with identity covariance and configured sigma
        
    Requirements: 1.1, 1.2
    """
    n = n_dimensions
    
    # Validate inputs
    if n < 1:
        raise ValueError(f"n_dimensions must be >= 1, got {n}")
    if sigma0 <= 0:
        raise ValueError(f"sigma0 must be positive, got {sigma0}")
    if len(initial_centroid) != n:
        raise ValueError(
            f"initial_centroid has {len(initial_centroid)} dimensions, "
            f"expected {n}"
        )
    
    # Ensure centroid is in [0, 1] bounds
    if not (np.all(initial_centroid >= 0.0) and np.all(initial_centroid <= 1.0)):
        raise ValueError(
            f"initial_centroid must be in [0, 1] space, "
            f"got values in range [{initial_centroid.min()}, {initial_centroid.max()}]"
        )
    
    # Calculate CMA-ES constants (standard formulation)
    # Number of parents (mu) - use top half of population
    mu = n  # Will be set properly when we know population_size, for now use n
    
    # Recombination weights (log-linear)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    
    # Variance effective selection mass
    mueff = 1.0 / (weights ** 2).sum()
    
    # Time constant for cumulation for C (covariance matrix)
    cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    
    # Time constant for cumulation for sigma (step size)
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    
    # Learning rate for rank-one update of C
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    
    # Learning rate for rank-mu update of C
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    
    # Damping for sigma adaptation
    damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    
    # Expected value of ||N(0,I)||
    chiN = n ** 0.5 * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
    
    # Create fresh state
    state = CMAESState(
        # Core state
        centroid=initial_centroid.copy(),
        sigma=sigma0,
        C=np.eye(n),  # Identity covariance matrix
        
        # Evolution paths (initialized to zero)
        pc=np.zeros(n),
        ps=np.zeros(n),
        
        # Eigendecomposition (identity matrix)
        B=np.eye(n),  # Eigenvectors
        D=np.ones(n),  # Square root of eigenvalues (all 1 for identity)
        
        # Tracking
        generation=0,
        eigenvalue_update_gen=0,
        
        # Constants
        n_dimensions=n,
        mu=mu,
        weights=weights,
        mueff=mueff,
        cc=cc,
        cs=cs,
        c1=c1,
        cmu=cmu,
        damps=damps,
        chiN=chiN,
    )
    
    return state


def generate_population(
    state: CMAESState,
    population_size: int
) -> np.ndarray:
    """
    Generate population by sampling from multivariate normal distribution.
    
    Samples from N(centroid, sigma² * C) and clips all solutions to [0, 1] bounds
    in normalized space.
    
    Args:
        state: Current CMA-ES state
        population_size: Number of solutions to generate
        
    Returns:
        Population array of shape (population_size, n_dimensions) in [0, 1] space
        
    Requirements: 1.3, 1.4
    """
    n = state.n_dimensions
    
    # Sample from N(0, C) using eigendecomposition: N(0, C) = B * D * N(0, I)
    # where C = B * D² * B^T
    population = []
    
    for _ in range(population_size):
        # Sample from standard normal N(0, I)
        z = np.random.randn(n)
        
        # Transform to N(0, C): y = B * D * z
        y = state.B @ (state.D * z)
        
        # Scale by sigma and add centroid: x = centroid + sigma * y
        x = state.centroid + state.sigma * y
        
        # Clip to [0, 1] bounds
        x = np.clip(x, 0.0, 1.0)
        
        population.append(x)
    
    return np.array(population)


def update_cma_state(
    state: CMAESState,
    population: np.ndarray,
    fitness_values: np.ndarray,
    population_size: int
) -> CMAESState:
    """
    Update CMA-ES state based on ranked solutions.
    
    Implements the standard CMA-ES update equations:
    1. Rank solutions by fitness (lower is better)
    2. Update centroid as weighted mean of best mu solutions
    3. Update evolution path pc for covariance matrix adaptation
    4. Update evolution path ps for step-size adaptation
    5. Update covariance matrix C
    6. Update sigma based on ps
    7. Periodically update eigendecomposition (B, D)
    
    Args:
        state: Current CMA-ES state
        population: Population array (population_size, n_dimensions) in [0, 1] space
        fitness_values: Fitness values for each solution (lower is better)
        population_size: Size of the population
        
    Returns:
        Updated CMA-ES state
        
    Requirements: 1.5, 1.6, 1.7, 1.8, 1.9
    """
    n = state.n_dimensions
    
    # Update mu based on population_size if needed
    mu = population_size // 2
    if mu != state.mu:
        # Recalculate weights for new mu
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / (weights ** 2).sum()
    else:
        weights = state.weights
        mueff = state.mueff
    
    # 1. Rank solutions by fitness (ascending order - lower is better)
    sorted_indices = np.argsort(fitness_values)
    sorted_population = population[sorted_indices]
    
    # 2. Update centroid as weighted mean of top mu solutions
    old_centroid = state.centroid.copy()
    new_centroid = np.sum(weights[:, np.newaxis] * sorted_population[:mu], axis=0)
    
    # 3. Update evolution path for covariance matrix (pc)
    # Cumulation: c_c * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m_old) / sigma
    c_diff = (new_centroid - old_centroid) / state.sigma
    
    # Compute C^(-1/2) * c_diff using eigendecomposition
    # C^(-1/2) = B * D^(-1) * B^T
    c_diff_normalized = state.B @ (c_diff / state.D)
    
    hsig = (
        np.linalg.norm(state.ps) / 
        np.sqrt(1.0 - (1.0 - state.cs) ** (2.0 * (state.generation + 1))) / 
        state.chiN
    ) < (1.4 + 2.0 / (n + 1.0))
    
    pc = (1.0 - state.cc) * state.pc + \
         hsig * np.sqrt(state.cc * (2.0 - state.cc) * mueff) * c_diff
    
    # 4. Update evolution path for step-size (ps)
    ps = (1.0 - state.cs) * state.ps + \
         np.sqrt(state.cs * (2.0 - state.cs) * mueff) * c_diff_normalized
    
    # 5. Update covariance matrix C
    # Rank-one update
    rank_one = state.c1 * (pc[:, np.newaxis] @ pc[np.newaxis, :])
    
    # Rank-mu update
    y_mu = (sorted_population[:mu] - old_centroid) / state.sigma
    rank_mu = state.cmu * np.sum(
        weights[:, np.newaxis, np.newaxis] * 
        (y_mu[:, :, np.newaxis] @ y_mu[:, np.newaxis, :]),
        axis=0
    )
    
    # Combine updates
    C = (1.0 - state.c1 - state.cmu) * state.C + rank_one + rank_mu
    
    # 6. Update sigma based on ps
    sigma = state.sigma * np.exp(
        (state.cs / state.damps) * 
        (np.linalg.norm(ps) / state.chiN - 1.0)
    )
    
    # 7. Periodically update eigendecomposition
    # Update every 1 / (c1 + cmu) / n / 10 generations
    eigenvalue_update_interval = max(1, int(1.0 / (state.c1 + state.cmu) / n / 10.0))
    
    if (state.generation + 1) % eigenvalue_update_interval == 0:
        # Ensure C is symmetric
        C = (C + C.T) / 2.0
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        
        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Update B and D
        B = eigenvectors
        D = np.sqrt(np.maximum(eigenvalues, 1e-20))  # Avoid negative eigenvalues
        eigenvalue_update_gen = state.generation + 1
    else:
        B = state.B
        D = state.D
        eigenvalue_update_gen = state.eigenvalue_update_gen
    
    # Create updated state
    updated_state = CMAESState(
        centroid=new_centroid,
        sigma=sigma,
        C=C,
        pc=pc,
        ps=ps,
        B=B,
        D=D,
        generation=state.generation + 1,
        eigenvalue_update_gen=eigenvalue_update_gen,
        n_dimensions=n,
        mu=mu,
        weights=weights,
        mueff=mueff,
        cc=state.cc,
        cs=state.cs,
        c1=state.c1,
        cmu=state.cmu,
        damps=state.damps,
        chiN=state.chiN,
    )
    
    return updated_state
