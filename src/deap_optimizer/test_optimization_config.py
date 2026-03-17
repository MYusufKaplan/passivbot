"""
Unit tests for OptimizationConfig.

This module contains unit tests for validating the OptimizationConfig dataclass,
including default values, safe_mode behavior, and Rust flag generation.

Testing Framework: pytest
Requirements: 11.3
"""

import pytest

# Direct import to avoid loading heavy dependencies from __init__.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deap_optimizer.optimization_config import OptimizationConfig


class TestOptimizationConfigDefaults:
    """Tests for default values of OptimizationConfig."""

    def test_default_python_optimization_flags_are_true(self):
        """Test that all Python-side optimization flags default to True."""
        config = OptimizationConfig()
        
        assert config.enable_argument_pool is True
        assert config.enable_lazy_pca is True
        assert config.enable_task_batching is True
        assert config.enable_optimized_combine is True

    def test_default_rust_optimization_flags_are_true(self):
        """Test that all Rust-side optimization flags default to True."""
        config = OptimizationConfig()
        
        assert config.enable_preallocated_buffers is True
        assert config.enable_incremental_volume is True
        assert config.enable_optimized_noisiness is True
        assert config.enable_optimized_analysis is True

    def test_default_pca_settings(self):
        """Test default PCA configuration values."""
        config = OptimizationConfig()
        
        assert config.pca_interval == 10
        assert config.pca_on_new_best is True

    def test_default_task_batch_size(self):
        """Test default task batch size."""
        config = OptimizationConfig()
        
        assert config.task_batch_size == 50

    def test_default_safe_mode_is_false(self):
        """Test that safe_mode defaults to False."""
        config = OptimizationConfig()
        
        assert config.safe_mode is False


class TestOptimizationConfigSafeMode:
    """Tests for safe_mode behavior of OptimizationConfig."""

    def test_safe_mode_disables_all_python_optimizations(self):
        """Test that safe_mode=True disables all Python-side optimization flags.
        
        **Validates: Requirements 11.3**
        """
        config = OptimizationConfig(safe_mode=True)
        
        assert config.enable_argument_pool is False
        assert config.enable_lazy_pca is False
        assert config.enable_task_batching is False
        assert config.enable_optimized_combine is False

    def test_safe_mode_disables_all_rust_optimizations(self):
        """Test that safe_mode=True disables all Rust-side optimization flags.
        
        **Validates: Requirements 11.3**
        """
        config = OptimizationConfig(safe_mode=True)
        
        assert config.enable_preallocated_buffers is False
        assert config.enable_incremental_volume is False
        assert config.enable_optimized_noisiness is False
        assert config.enable_optimized_analysis is False

    def test_safe_mode_preserves_non_optimization_settings(self):
        """Test that safe_mode preserves non-optimization settings like pca_interval."""
        config = OptimizationConfig(safe_mode=True, pca_interval=20, task_batch_size=100)
        
        # These are configuration values, not optimization flags
        assert config.pca_interval == 20
        assert config.task_batch_size == 100
        assert config.pca_on_new_best is True  # Not disabled by safe_mode

    def test_safe_mode_overrides_explicit_true_flags(self):
        """Test that safe_mode=True overrides explicitly set True flags."""
        config = OptimizationConfig(
            safe_mode=True,
            enable_argument_pool=True,
            enable_lazy_pca=True,
            enable_preallocated_buffers=True,
        )
        
        # safe_mode should override these to False
        assert config.enable_argument_pool is False
        assert config.enable_lazy_pca is False
        assert config.enable_preallocated_buffers is False


class TestOptimizationConfigRustFlags:
    """Tests for get_rust_optimization_flags() method."""

    def test_get_rust_optimization_flags_returns_dict(self):
        """Test that get_rust_optimization_flags returns a dictionary."""
        config = OptimizationConfig()
        
        flags = config.get_rust_optimization_flags()
        
        assert isinstance(flags, dict)

    def test_get_rust_optimization_flags_contains_all_rust_flags(self):
        """Test that get_rust_optimization_flags contains all Rust optimization flags."""
        config = OptimizationConfig()
        
        flags = config.get_rust_optimization_flags()
        
        expected_keys = {
            "enable_preallocated_buffers",
            "enable_incremental_volume",
            "enable_optimized_noisiness",
            "enable_optimized_analysis",
        }
        assert set(flags.keys()) == expected_keys

    def test_get_rust_optimization_flags_default_values(self):
        """Test that get_rust_optimization_flags returns correct default values."""
        config = OptimizationConfig()
        
        flags = config.get_rust_optimization_flags()
        
        assert flags["enable_preallocated_buffers"] is True
        assert flags["enable_incremental_volume"] is True
        assert flags["enable_optimized_noisiness"] is True
        assert flags["enable_optimized_analysis"] is True

    def test_get_rust_optimization_flags_with_safe_mode(self):
        """Test that get_rust_optimization_flags returns False values in safe_mode."""
        config = OptimizationConfig(safe_mode=True)
        
        flags = config.get_rust_optimization_flags()
        
        assert flags["enable_preallocated_buffers"] is False
        assert flags["enable_incremental_volume"] is False
        assert flags["enable_optimized_noisiness"] is False
        assert flags["enable_optimized_analysis"] is False

    def test_get_rust_optimization_flags_with_custom_values(self):
        """Test that get_rust_optimization_flags reflects custom flag values."""
        config = OptimizationConfig(
            enable_preallocated_buffers=False,
            enable_incremental_volume=True,
            enable_optimized_noisiness=False,
            enable_optimized_analysis=True,
        )
        
        flags = config.get_rust_optimization_flags()
        
        assert flags["enable_preallocated_buffers"] is False
        assert flags["enable_incremental_volume"] is True
        assert flags["enable_optimized_noisiness"] is False
        assert flags["enable_optimized_analysis"] is True


class TestOptimizationConfigIndependentFlags:
    """Tests for independent flag toggling."""

    def test_individual_python_flags_can_be_disabled(self):
        """Test that individual Python optimization flags can be disabled independently.
        
        **Validates: Requirements 11.1, 11.2**
        """
        config = OptimizationConfig(enable_argument_pool=False)
        
        assert config.enable_argument_pool is False
        # Other flags should remain True
        assert config.enable_lazy_pca is True
        assert config.enable_task_batching is True
        assert config.enable_optimized_combine is True

    def test_individual_rust_flags_can_be_disabled(self):
        """Test that individual Rust optimization flags can be disabled independently.
        
        **Validates: Requirements 11.1, 11.2**
        """
        config = OptimizationConfig(enable_preallocated_buffers=False)
        
        assert config.enable_preallocated_buffers is False
        # Other Rust flags should remain True
        assert config.enable_incremental_volume is True
        assert config.enable_optimized_noisiness is True
        assert config.enable_optimized_analysis is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
