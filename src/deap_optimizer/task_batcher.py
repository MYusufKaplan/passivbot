"""
Interval Task Batcher for DEAP optimizer.

This module provides the IntervalTaskBatcher class that efficiently batches
interval evaluation tasks to reduce multiprocessing overhead when evaluating
many month-individual combinations.

Requirements:
- 3.1: Group tasks by individual rather than creating flat task pools
- 3.2: Use chunked submission with configurable chunk sizes
- 3.3: Use pre-allocated result arrays indexed by candidate ID
"""

from typing import Any, Dict, List, Tuple, Optional
from multiprocessing import Pool


def _evaluate_interval_worker(args: Tuple) -> Tuple:
    """Worker function for interval evaluation.
    
    This function is called by the multiprocessing pool to evaluate a single
    individual on a single interval.
    
    Args:
        args: Tuple of (evaluator, individual, month_id, interval, candidate_id)
    
    Returns:
        Tuple of (candidate_id, month_id, fitness, bankrupt, reason, ts, total_ts, gain)
        where:
        - candidate_id: Index of the individual in the population
        - month_id: Index of the monthly interval
        - fitness: Fitness value from evaluation
        - bankrupt: Boolean indicating if bankruptcy occurred
        - reason: Bankruptcy reason code (0=none, 1=financial, etc.)
        - ts: Bankruptcy timestep (if applicable)
        - total_ts: Total timesteps in the interval
        - gain: Gain value from evaluation
    """
    evaluator, individual, month_id, interval, candidate_id = args
    
    # Call the evaluator's interval evaluation method
    # The evaluator should have an evaluate_interval method that returns
    # (fitness, bankrupt, reason, bankruptcy_timestep, total_timesteps, gain)
    result = evaluator.evaluate_interval(individual, interval)
    
    # Unpack result - handle different result formats
    if len(result) == 6:
        fitness, bankrupt, reason, ts, total_ts, gain = result
    elif len(result) == 5:
        fitness, bankrupt, reason, ts, total_ts = result
        gain = 0.0
    else:
        # Fallback for older format
        fitness, bankrupt, ts, total_ts = result
        reason = 0
        gain = 0.0
    
    return (candidate_id, month_id, fitness, bankrupt, reason, ts, total_ts, gain)


class IntervalTaskBatcher:
    """Batched task submission for interval mode evaluation.
    
    This class optimizes interval mode evaluation by:
    1. Grouping tasks by individual for better cache locality
    2. Submitting tasks in configurable chunks to reduce pool overhead
    3. Using pre-allocated result dictionaries indexed by candidate ID
    
    Attributes:
        batch_size: Number of tasks to submit per chunk (default: 50)
    
    Requirements: 3.1, 3.2, 3.3
    """
    
    def __init__(self, batch_size: int = 50):
        """Initialize the IntervalTaskBatcher.
        
        Args:
            batch_size: Number of tasks to submit per chunk. Must be a positive
                        integer. Default is 50.
        
        Raises:
            ValueError: If batch_size is not a positive integer.
        
        Requirements: 3.2
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        
        self.batch_size = batch_size
        # Pre-allocate result storage
        self._results: Dict[int, Dict[int, Tuple]] = {}
    
    @property
    def results(self) -> Dict[int, Dict[int, Tuple]]:
        """Return the current results dictionary.
        
        Returns:
            Dictionary mapping candidate_id -> {month_id -> result_tuple}
        """
        return self._results
    
    def evaluate_batched(
        self,
        individuals: List[Any],
        intervals: List[Any],
        evaluator: Any,
        pool: Pool,
        worker_func: Optional[callable] = None
    ) -> Dict[int, Dict[int, Tuple]]:
        """Evaluate individuals across intervals with batched submission.
        
        This method builds tasks grouped by individual for better cache locality,
        then submits them in chunks to reduce multiprocessing overhead.
        
        Args:
            individuals: List of individuals (candidates) to evaluate.
            intervals: List of MonthlyInterval objects to evaluate each
                       candidate on.
            evaluator: The evaluator instance to use for evaluation.
            pool: Multiprocessing Pool to use for parallel evaluation.
            worker_func: Optional custom worker function. If None, uses
                         the default _evaluate_interval_worker.
        
        Returns:
            Dictionary mapping candidate_id -> {month_id -> result_tuple}
            where result_tuple is (fitness, bankrupt, reason, ts, total_ts).
        
        Requirements: 3.1, 3.2, 3.3
        """
        if worker_func is None:
            worker_func = _evaluate_interval_worker
        
        n_individuals = len(individuals)
        
        # Pre-allocate results dictionary (Requirement 3.3)
        self._results.clear()
        for i in range(n_individuals):
            self._results[i] = {}
        
        # Build tasks grouped by individual for better cache locality (Requirement 3.1)
        all_tasks = []
        for candidate_id, individual in enumerate(individuals):
            for interval in intervals:
                all_tasks.append((
                    evaluator, individual, interval.month_id,
                    interval, candidate_id
                ))
        
        # Submit in chunks to reduce pool overhead (Requirement 3.2)
        total_tasks = len(all_tasks)
        
        for chunk_start in range(0, total_tasks, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, total_tasks)
            chunk = all_tasks[chunk_start:chunk_end]
            
            # Process chunk using imap_unordered for efficiency
            for result in pool.imap_unordered(worker_func, chunk):
                candidate_id, month_id, fitness, bankrupt, reason, ts, total_ts, gain = result
                # Store result indexed by candidate_id and month_id (Requirement 3.3)
                self._results[candidate_id][month_id] = (
                    fitness, bankrupt, reason, ts, total_ts
                )
        
        return self._results
    
    def build_task_list(
        self,
        individuals: List[Any],
        intervals: List[Any],
        evaluator: Any
    ) -> List[Tuple]:
        """Build the task list grouped by individual.
        
        This method creates the task list without executing evaluation,
        useful for testing or custom evaluation workflows.
        
        Args:
            individuals: List of individuals to evaluate.
            intervals: List of MonthlyInterval objects.
            evaluator: The evaluator instance.
        
        Returns:
            List of task tuples: (evaluator, individual, month_id, interval, candidate_id)
            Tasks are grouped by individual for better cache locality.
        
        Requirements: 3.1
        """
        all_tasks = []
        for candidate_id, individual in enumerate(individuals):
            for interval in intervals:
                all_tasks.append((
                    evaluator, individual, interval.month_id,
                    interval, candidate_id
                ))
        return all_tasks
    
    def get_chunk_boundaries(self, total_tasks: int) -> List[Tuple[int, int]]:
        """Get the chunk boundaries for a given number of tasks.
        
        This method returns the start and end indices for each chunk,
        useful for testing or custom chunking logic.
        
        Args:
            total_tasks: Total number of tasks to chunk.
        
        Returns:
            List of (start, end) tuples representing chunk boundaries.
            The last chunk may be smaller than batch_size.
        
        Requirements: 3.2
        """
        boundaries = []
        for chunk_start in range(0, total_tasks, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, total_tasks)
            boundaries.append((chunk_start, chunk_end))
        return boundaries
    
    def reset(self) -> None:
        """Reset the batcher state.
        
        Clears the results dictionary to free memory.
        """
        self._results.clear()

