"""
Configuration for performance optimizations in the DEAP optimizer.

This module provides the OptimizationConfig dataclass that controls which
performance optimizations are enabled. Each optimization can be toggled
independently, and a safe_mode flag disables all optimizations for debugging.
"""

from dataclasses import dataclass


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations.
    
    All optimizations are enabled by default. Set safe_mode=True to disable
    all optimizations for debugging purposes.
    
    Attributes:
        enable_argument_pool: Reuse multiprocessing argument tuples across generations
        enable_lazy_pca: Compute PCA visualization only at configurable intervals
        pca_interval: Number of generations between PCA computations
        pca_on_new_best: Trigger PCA update when a new global best is found
        enable_task_batching: Use chunked task submission for interval mode
        task_batch_size: Number of tasks per batch in interval mode
        enable_optimized_combine: Use optimized dictionary lookups in combine_analyses
        enable_preallocated_buffers: Pre-allocate buffers in Rust backtest engine
        enable_incremental_volume: Use incremental rolling volume calculation in Rust
        enable_optimized_noisiness: Use optimized noisiness calculation in Rust
        enable_optimized_analysis: Use optimized analyze_backtest_basic in Rust
        safe_mode: When True, disables all optimizations for debugging
    """
    
    # Python-side optimizations
    enable_argument_pool: bool = True
    enable_lazy_pca: bool = True
    pca_interval: int = 10
    pca_on_new_best: bool = True
    enable_task_batching: bool = True
    task_batch_size: int = 50
    enable_optimized_combine: bool = True
    
    # Rust-side optimizations (passed via backtest_params)
    enable_preallocated_buffers: bool = True
    enable_incremental_volume: bool = True
    enable_optimized_noisiness: bool = True
    enable_optimized_analysis: bool = True
    
    # Debugging
    safe_mode: bool = False
    
    def __post_init__(self):
        """Apply safe_mode by disabling all optimizations if set."""
        if self.safe_mode:
            self.enable_argument_pool = False
            self.enable_lazy_pca = False
            self.enable_task_batching = False
            self.enable_optimized_combine = False
            self.enable_preallocated_buffers = False
            self.enable_incremental_volume = False
            self.enable_optimized_noisiness = False
            self.enable_optimized_analysis = False
    
    def get_rust_optimization_flags(self) -> dict:
        """Return a dictionary of Rust optimization flags for backtest_params.
        
        Returns:
            Dictionary with Rust optimization flags that can be passed to the
            Rust backtest engine via backtest_params.
        """
        return {
            "enable_preallocated_buffers": self.enable_preallocated_buffers,
            "enable_incremental_volume": self.enable_incremental_volume,
            "enable_optimized_noisiness": self.enable_optimized_noisiness,
            "enable_optimized_analysis": self.enable_optimized_analysis,
        }
