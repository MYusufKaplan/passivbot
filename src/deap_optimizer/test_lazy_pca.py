"""
Unit tests for LazyPCAVisualizer class.

Tests the core functionality of the lazy PCA visualization component:
- Interval-based computation logic
- State caching between computations
- New global best triggering
- Disabled state handling
"""

import pytest

# Direct import to avoid loading heavy dependencies from __init__.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deap_optimizer.lazy_pca import LazyPCAVisualizer


class TestLazyPCAVisualizerInit:
    """Tests for LazyPCAVisualizer initialization."""
    
    def test_default_values(self):
        """Test default initialization values."""
        visualizer = LazyPCAVisualizer()
        
        assert visualizer.interval == 10
        assert visualizer.enabled is True
        assert visualizer.trigger_on_new_best is True
        assert visualizer.pca_state is None
        assert visualizer.last_computed_generation == -1
    
    def test_custom_interval(self):
        """Test initialization with custom interval."""
        visualizer = LazyPCAVisualizer(interval=5)
        assert visualizer.interval == 5
    
    def test_disabled_initialization(self):
        """Test initialization with disabled flag."""
        visualizer = LazyPCAVisualizer(enabled=False)
        assert visualizer.enabled is False
    
    def test_trigger_on_new_best_disabled(self):
        """Test initialization with trigger_on_new_best disabled."""
        visualizer = LazyPCAVisualizer(trigger_on_new_best=False)
        assert visualizer.trigger_on_new_best is False
    
    def test_invalid_interval_zero(self):
        """Test that interval=0 raises ValueError."""
        with pytest.raises(ValueError, match="interval must be a positive integer"):
            LazyPCAVisualizer(interval=0)
    
    def test_invalid_interval_negative(self):
        """Test that negative interval raises ValueError."""
        with pytest.raises(ValueError, match="interval must be a positive integer"):
            LazyPCAVisualizer(interval=-5)


class TestShouldCompute:
    """Tests for should_compute() method."""
    
    def test_disabled_always_returns_false(self):
        """When disabled, should_compute always returns False."""
        visualizer = LazyPCAVisualizer(enabled=False)
        
        assert visualizer.should_compute(0) is False
        assert visualizer.should_compute(10) is False
        assert visualizer.should_compute(0, new_global_best=True) is False
    
    def test_first_generation_computes(self):
        """First generation (gen 0) should compute when enabled."""
        visualizer = LazyPCAVisualizer(interval=10)
        
        # Generation 0 is 10 generations after -1 (last_computed_gen)
        assert visualizer.should_compute(0) is True
    
    def test_interval_logic(self):
        """Test interval-based computation logic."""
        visualizer = LazyPCAVisualizer(interval=10)
        
        # Simulate that generation 0 was computed
        visualizer._last_computed_gen = 0
        
        # Generations 1-9 should not compute
        for gen in range(1, 10):
            assert visualizer.should_compute(gen) is False
        
        # Generation 10 should compute (10 - 0 = 10 >= 10)
        assert visualizer.should_compute(10) is True
    
    def test_new_global_best_triggers_compute(self):
        """New global best should trigger computation regardless of interval."""
        visualizer = LazyPCAVisualizer(interval=10, trigger_on_new_best=True)
        visualizer._last_computed_gen = 0
        
        # Generation 5 normally wouldn't compute
        assert visualizer.should_compute(5) is False
        
        # But with new_global_best=True, it should
        assert visualizer.should_compute(5, new_global_best=True) is True
    
    def test_new_global_best_disabled(self):
        """When trigger_on_new_best is False, new best doesn't trigger compute."""
        visualizer = LazyPCAVisualizer(interval=10, trigger_on_new_best=False)
        visualizer._last_computed_gen = 0
        
        # Even with new_global_best=True, should not compute before interval
        assert visualizer.should_compute(5, new_global_best=True) is False
        
        # But should still compute at interval
        assert visualizer.should_compute(10) is True
    
    def test_interval_of_one(self):
        """Test with interval=1 (compute every generation)."""
        visualizer = LazyPCAVisualizer(interval=1)
        
        # Should compute every generation
        for gen in range(20):
            visualizer._last_computed_gen = gen - 1
            assert visualizer.should_compute(gen) is True


class TestVisualize:
    """Tests for visualize() method."""
    
    def test_disabled_returns_none(self):
        """When disabled, visualize returns None without calling function."""
        visualizer = LazyPCAVisualizer(enabled=False)
        call_count = [0]
        
        def mock_pca_func(*args, **kwargs):
            call_count[0] += 1
            return {'pca': 'mock'}
        
        result = visualizer.visualize(
            population=[], halloffame=None, generation=0,
            console=None, watch_path='/tmp/test',
            pca_visualization_func=mock_pca_func
        )
        
        assert result is None
        assert call_count[0] == 0
    
    def test_caches_state_between_intervals(self):
        """PCA state should be cached and reused between intervals."""
        visualizer = LazyPCAVisualizer(interval=10)
        call_count = [0]
        
        def mock_pca_func(*args, **kwargs):
            call_count[0] += 1
            return {'pca': f'state_{call_count[0]}', 'generation': kwargs.get('generation')}
        
        # First call at generation 0 should compute
        result1 = visualizer.visualize(
            population=[], halloffame=None, generation=0,
            console=None, watch_path='/tmp/test',
            pca_visualization_func=mock_pca_func
        )
        assert call_count[0] == 1
        assert result1['pca'] == 'state_1'
        
        # Calls at generations 1-9 should return cached state
        for gen in range(1, 10):
            result = visualizer.visualize(
                population=[], halloffame=None, generation=gen,
                console=None, watch_path='/tmp/test',
                pca_visualization_func=mock_pca_func
            )
            assert call_count[0] == 1  # No new calls
            assert result['pca'] == 'state_1'  # Same cached state
        
        # Generation 10 should compute again
        result10 = visualizer.visualize(
            population=[], halloffame=None, generation=10,
            console=None, watch_path='/tmp/test',
            pca_visualization_func=mock_pca_func
        )
        assert call_count[0] == 2
        assert result10['pca'] == 'state_2'
    
    def test_new_global_best_triggers_recompute(self):
        """New global best should trigger recomputation."""
        visualizer = LazyPCAVisualizer(interval=10, trigger_on_new_best=True)
        call_count = [0]
        
        def mock_pca_func(*args, **kwargs):
            call_count[0] += 1
            return {'pca': f'state_{call_count[0]}'}
        
        # First call at generation 0
        visualizer.visualize(
            population=[], halloffame=None, generation=0,
            console=None, watch_path='/tmp/test',
            pca_visualization_func=mock_pca_func
        )
        assert call_count[0] == 1
        
        # Generation 5 with new_global_best should recompute
        result = visualizer.visualize(
            population=[], halloffame=None, generation=5,
            console=None, watch_path='/tmp/test',
            new_global_best=True,
            pca_visualization_func=mock_pca_func
        )
        assert call_count[0] == 2
        assert result['pca'] == 'state_2'
    
    def test_updates_last_computed_generation(self):
        """visualize should update last_computed_generation after computing."""
        visualizer = LazyPCAVisualizer(interval=10)
        
        def mock_pca_func(*args, **kwargs):
            return {'pca': 'mock'}
        
        assert visualizer.last_computed_generation == -1
        
        visualizer.visualize(
            population=[], halloffame=None, generation=0,
            console=None, watch_path='/tmp/test',
            pca_visualization_func=mock_pca_func
        )
        assert visualizer.last_computed_generation == 0
        
        visualizer.visualize(
            population=[], halloffame=None, generation=10,
            console=None, watch_path='/tmp/test',
            pca_visualization_func=mock_pca_func
        )
        assert visualizer.last_computed_generation == 10


class TestReset:
    """Tests for reset() method."""
    
    def test_reset_clears_state(self):
        """reset() should clear PCA state and last computed generation."""
        visualizer = LazyPCAVisualizer(interval=10)
        
        def mock_pca_func(*args, **kwargs):
            return {'pca': 'mock'}
        
        # Compute at generation 0
        visualizer.visualize(
            population=[], halloffame=None, generation=0,
            console=None, watch_path='/tmp/test',
            pca_visualization_func=mock_pca_func
        )
        
        assert visualizer.pca_state is not None
        assert visualizer.last_computed_generation == 0
        
        # Reset
        visualizer.reset()
        
        assert visualizer.pca_state is None
        assert visualizer.last_computed_generation == -1
    
    def test_reset_allows_immediate_recompute(self):
        """After reset, should_compute should return True."""
        visualizer = LazyPCAVisualizer(interval=10)
        visualizer._last_computed_gen = 5
        
        # Before reset, generation 6 shouldn't compute
        assert visualizer.should_compute(6) is False
        
        # After reset, generation 6 should compute
        visualizer.reset()
        assert visualizer.should_compute(6) is True
