"""
Lazy PCA Visualization module for DEAP optimizer.

This module provides a LazyPCAVisualizer class that computes PCA visualization
only at configurable generation intervals to reduce computational overhead.

Requirements:
- 2.1: PCA computed only at configurable generation intervals (default: every 10 generations)
- 2.2: Reuse fitted PCA model for subsequent visualizations until next computation
- 2.3: Optionally trigger immediate PCA update on new global best
- 2.4: Configuration option to disable PCA visualization entirely
"""

from typing import Optional


class LazyPCAVisualizer:
    """PCA visualization with configurable computation intervals.
    
    This class wraps PCA visualization to reduce computational overhead by:
    1. Computing PCA only at configurable intervals (default: every 10 generations)
    2. Caching and reusing the PCA state between computations
    3. Optionally triggering immediate updates on new global best
    4. Providing an option to disable visualization entirely
    
    Attributes:
        interval: Number of generations between PCA computations (default: 10)
        enabled: Whether PCA visualization is enabled (default: True)
        trigger_on_new_best: Whether to compute PCA on new global best (default: True)
    """
    
    def __init__(
        self, 
        interval: int = 10, 
        enabled: bool = True,
        trigger_on_new_best: bool = True
    ):
        """Initialize the LazyPCAVisualizer.
        
        Args:
            interval: Number of generations between PCA computations.
                      Must be a positive integer. Default is 10.
            enabled: Whether PCA visualization is enabled. Default is True.
            trigger_on_new_best: Whether to trigger PCA computation when a new
                                 global best is found. Default is True.
        
        Raises:
            ValueError: If interval is not a positive integer.
        """
        if interval < 1:
            raise ValueError(f"interval must be a positive integer, got {interval}")
        
        self.interval = interval
        self.enabled = enabled
        self.trigger_on_new_best = trigger_on_new_best
        self._pca_state: Optional[dict] = None
        self._last_computed_gen: int = -1
    
    @property
    def pca_state(self) -> Optional[dict]:
        """Return the current cached PCA state."""
        return self._pca_state
    
    @property
    def last_computed_generation(self) -> int:
        """Return the generation number when PCA was last computed."""
        return self._last_computed_gen
    
    def should_compute(self, generation: int, new_global_best: bool = False) -> bool:
        """Determine if PCA should be computed this generation.
        
        Args:
            generation: Current generation number (0-indexed).
            new_global_best: Whether a new global best was found this generation.
        
        Returns:
            True if PCA should be computed, False otherwise.
        
        The method returns True if:
        - PCA is enabled AND one of:
          - This is the first computation (last_computed_gen == -1)
          - A new global best was found (and trigger_on_new_best is True)
          - The interval threshold has been reached since last computation
        """
        if not self.enabled:
            return False
        
        # Always compute on first call (never computed before)
        if self._last_computed_gen == -1:
            return True
        
        # Always compute on new global best if configured to do so
        if new_global_best and self.trigger_on_new_best:
            return True
        
        # Compute if interval threshold reached
        return (generation - self._last_computed_gen) >= self.interval
    
    def visualize(
        self,
        population,
        halloffame,
        generation: int,
        console,
        watch_path: str,
        new_global_best: bool = False,
        pca_visualization_func=None,
        grid_width: int = 172,
        grid_height: int = 31
    ) -> Optional[dict]:
        """Compute and display PCA visualization if needed.
        
        This method checks if PCA should be computed based on the interval
        and new_global_best settings. If computation is needed, it calls
        the provided PCA visualization function and caches the result.
        
        Args:
            population: Current population of DEAP individuals.
            halloffame: Hall of fame containing best individuals.
            generation: Current generation number.
            console: Rich console for output.
            watch_path: Path to write visualization.
            new_global_best: Whether a new global best was found.
            pca_visualization_func: Function to call for PCA visualization.
                                    If None, uses create_pca_visualization_deap
                                    from evolutionary_algorithm module.
            grid_width: Width of ASCII grid (default: 172).
            grid_height: Height of ASCII grid (default: 31).
        
        Returns:
            The current PCA state dictionary, or None if visualization is disabled.
        """
        if not self.enabled:
            return None
        
        if not self.should_compute(generation, new_global_best):
            return self._pca_state
        
        # Import the default visualization function if not provided
        if pca_visualization_func is None:
            from src.deap_optimizer.evolutionary_algorithm import create_pca_visualization_deap
            pca_visualization_func = create_pca_visualization_deap
        
        # Compute PCA visualization
        self._pca_state = pca_visualization_func(
            population=population,
            halloffame=halloffame,
            generation=generation,
            console=console,
            watch_path=watch_path,
            pca_state=self._pca_state,
            grid_width=grid_width,
            grid_height=grid_height
        )
        self._last_computed_gen = generation
        
        return self._pca_state
    
    def reset(self) -> None:
        """Reset the visualizer state.
        
        This clears the cached PCA state and resets the last computed
        generation to -1, causing the next call to should_compute()
        to return True (if enabled).
        """
        self._pca_state = None
        self._last_computed_gen = -1
