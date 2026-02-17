"""
Data structures for CMA-ES with automatic restarts.
"""

import numpy as np
from dataclasses import dataclass


class RollingBuffer:
    """
    Fixed-size rolling buffer for memory-efficient history tracking.
    
    This buffer maintains a fixed maximum size and automatically discards
    the oldest entries when full, ensuring bounded memory usage for
    indefinite runtime optimization.
    
    Requirements: 11.3, 11.4
    """
    
    def __init__(self, max_size: int):
        """
        Initialize rolling buffer with maximum size.
        
        Args:
            max_size: Maximum number of entries to store
        """
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        
        self.buffer = []
        self.max_size = max_size
    
    def append(self, value):
        """
        Append a new entry to the buffer.
        
        If the buffer is at maximum size, the oldest entry is discarded
        before adding the new entry.
        
        Args:
            value: Value to append to the buffer
        """
        self.buffer.append(value)
        
        # Discard oldest entry if buffer exceeds max size
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)
    
    def __getitem__(self, index):
        """Get item at index."""
        return self.buffer[index]
    
    def __iter__(self):
        """Iterate over buffer entries."""
        return iter(self.buffer)
    
    def to_list(self):
        """Return buffer contents as a list."""
        return list(self.buffer)


@dataclass
class CMAESState:
    """Complete state of CMA-ES algorithm in normalized [0, 1] space."""
    
    # Core state
    centroid: np.ndarray          # (n,) - mean in [0, 1] space
    sigma: float                  # Step size
    C: np.ndarray                 # (n, n) - covariance matrix
    
    # Evolution paths
    pc: np.ndarray                # (n,) - evolution path for C
    ps: np.ndarray                # (n,) - evolution path for sigma
    
    # Eigendecomposition (updated periodically)
    B: np.ndarray                 # (n, n) - eigenvectors
    D: np.ndarray                 # (n,) - sqrt of eigenvalues
    
    # Tracking
    generation: int               # Current generation
    eigenvalue_update_gen: int    # Last eigenvalue update
    
    # Constants (set at initialization)
    n_dimensions: int
    mu: int                       # Number of parents
    weights: np.ndarray           # Recombination weights
    mueff: float                  # Variance effective selection mass
    cc: float                     # Time constant for C evolution path
    cs: float                     # Time constant for sigma evolution path
    c1: float                     # Learning rate for rank-one update
    cmu: float                    # Learning rate for rank-mu update
    damps: float                  # Damping for sigma adaptation
    chiN: float                   # Expected value of ||N(0,I)||


@dataclass
class Checkpoint:
    """Checkpoint for resuming CMA-ES optimization with restarts."""
    
    restart_number: int
    global_best_solution: np.ndarray  # Denormalized
    global_best_fitness: float
    logbook: list                     # All generation statistics
    
    # Note: CMA-ES state is NOT saved (each restart is fresh)
