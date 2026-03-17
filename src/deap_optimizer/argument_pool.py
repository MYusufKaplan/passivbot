"""
Argument Pool for multiprocessing evaluation.

This module provides the ArgumentPool class that pre-allocates and reuses
argument containers across generations to eliminate repeated memory allocations
and reduce per-generation overhead.

Requirements: 1.1, 1.2, 1.3
"""

from typing import Any, List, Set


class ArgumentPool:
    """Pre-allocated argument pool for multiprocessing evaluation.
    
    This class pre-allocates argument containers (lists) that are reused across
    generations. Instead of creating new tuples for each evaluation, the pool
    updates existing containers with new individual references.
    
    The pool uses lists instead of tuples because lists are mutable, allowing
    in-place updates without allocation. The worker functions unpack these
    the same way they would unpack tuples.
    
    Attributes:
        evaluator: The evaluator instance used for fitness evaluation
        max_size: Current maximum capacity of the pool
        _args: List of pre-allocated argument containers
    
    Requirements: 1.1, 1.2, 1.3
    """
    
    def __init__(self, max_size: int, evaluator: Any):
        """Initialize the argument pool with pre-allocated containers.
        
        Pre-allocates argument containers for the maximum population size.
        Each container is a list of [evaluator, individual, show_me, index].
        
        Args:
            max_size: Maximum number of individuals to support initially
            evaluator: The evaluator instance for fitness evaluation
        
        Requirements: 1.1
        """
        self.evaluator = evaluator
        self.max_size = max_size
        # Pre-allocate list of mutable argument containers
        # Format: [evaluator, individual, show_me_flag, index]
        self._args: List[List[Any]] = [
            [evaluator, None, False, i] 
            for i in range(max_size)
        ]
    
    def get_args(self, individuals: List[Any], show_me_indices: Set[int]) -> List[List[Any]]:
        """Return argument list, reusing pre-allocated containers.
        
        Updates the pre-allocated containers with new individual references
        and show_me flags, then returns a slice of the appropriate size.
        
        If the number of individuals exceeds the current capacity, the pool
        is expanded to accommodate the new size.
        
        Args:
            individuals: List of individuals to evaluate
            show_me_indices: Set of indices that should have show_me=True
        
        Returns:
            List of argument containers ready for multiprocessing evaluation.
            Each container is [evaluator, individual, show_me, index].
        
        Requirements: 1.2, 1.3
        """
        n = len(individuals)
        if n > self.max_size:
            # Expand pool if needed
            self._expand(n)
        
        for i, ind in enumerate(individuals):
            self._args[i][1] = ind  # Update individual reference
            self._args[i][2] = i in show_me_indices
            self._args[i][3] = i
        
        return self._args[:n]
    
    def _expand(self, new_size: int) -> None:
        """Expand pool capacity to accommodate larger populations.
        
        Adds new argument containers to the pool to reach the new size.
        Existing containers are preserved and reused.
        
        Args:
            new_size: The new capacity required
        
        Requirements: 1.3
        """
        for i in range(self.max_size, new_size):
            self._args.append([self.evaluator, None, False, i])
        self.max_size = new_size
    
    @property
    def capacity(self) -> int:
        """Return the current capacity of the pool.
        
        Returns:
            The maximum number of individuals the pool can currently handle
            without expansion.
        """
        return self.max_size
    
    def clear_references(self) -> None:
        """Clear individual references to allow garbage collection.
        
        Sets all individual references to None. This can be called after
        evaluation is complete to allow the individuals to be garbage
        collected if they are no longer needed elsewhere.
        """
        for arg in self._args:
            arg[1] = None
