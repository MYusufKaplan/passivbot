"""
Property-based tests for LazyPCAVisualizer class.

These tests use Hypothesis to verify universal properties of the lazy PCA
visualization component across many randomly generated inputs.

Properties tested:
- Property 3: PCA Interval Computation
- Property 4: PCA Model Reuse Between Intervals
- Property 5: PCA Triggers on New Global Best
"""

from hypothesis import given, strategies as st, settings, assume

# Direct import to avoid loading heavy dependencies from __init__.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deap_optimizer.lazy_pca import LazyPCAVisualizer


class TestProperty3PCAIntervalComputation:
    """
    Property 3: PCA Interval Computation
    
    *For any* sequence of generations with interval N, the PCA_Visualizer SHALL
    compute PCA only at generations 0, N, 2N, 3N, etc., unless a new global best
    triggers early computation.
    
    **Validates: Requirements 2.1**
    """
    
    @settings(max_examples=100)
    @given(
        interval=st.integers(min_value=1, max_value=50),
        num_generations=st.integers(min_value=1, max_value=200)
    )
    def test_pca_computes_only_at_interval_boundaries(self, interval: int, num_generations: int):
        """
        **Validates: Requirements 2.1**
        
        For any interval N and sequence of generations, PCA should compute
        only at generations 0, N, 2N, 3N, etc.
        """
        visualizer = LazyPCAVisualizer(interval=interval, trigger_on_new_best=False)
        computation_generations = []
        
        # Track which generations trigger computation
        for gen in range(num_generations):
            if visualizer.should_compute(gen, new_global_best=False):
                computation_generations.append(gen)
                # Simulate that computation happened
                visualizer._last_computed_gen = gen
        
        # Verify computations happen at expected intervals
        # First computation should be at generation 0
        if num_generations > 0:
            assert 0 in computation_generations, "First computation should be at generation 0"
        
        # All subsequent computations should be at multiples of interval from the first
        for i, comp_gen in enumerate(computation_generations):
            if i == 0:
                assert comp_gen == 0, f"First computation should be at gen 0, got {comp_gen}"
            else:
                # Each computation should be exactly 'interval' generations after the previous
                prev_gen = computation_generations[i - 1]
                expected_gen = prev_gen + interval
                assert comp_gen == expected_gen, (
                    f"Computation at index {i} should be at gen {expected_gen}, "
                    f"got {comp_gen} (interval={interval})"
                )
    
    @settings(max_examples=100)
    @given(
        interval=st.integers(min_value=1, max_value=50),
        num_generations=st.integers(min_value=1, max_value=200)
    )
    def test_no_computation_between_intervals(self, interval: int, num_generations: int):
        """
        **Validates: Requirements 2.1**
        
        For any generation that is not at an interval boundary (0, N, 2N, ...),
        should_compute should return False (when no new global best).
        """
        visualizer = LazyPCAVisualizer(interval=interval, trigger_on_new_best=False)
        
        # Simulate computation at generation 0
        visualizer._last_computed_gen = 0
        
        # Check all generations between 1 and interval-1
        for gen in range(1, min(interval, num_generations)):
            result = visualizer.should_compute(gen, new_global_best=False)
            assert result is False, (
                f"Generation {gen} should not compute (interval={interval}, "
                f"last_computed=0)"
            )
    
    @settings(max_examples=100)
    @given(
        interval=st.integers(min_value=1, max_value=50),
        multiplier=st.integers(min_value=0, max_value=20)
    )
    def test_computation_at_exact_interval_multiples(self, interval: int, multiplier: int):
        """
        **Validates: Requirements 2.1**
        
        For any interval N and multiplier k, generation k*N should trigger
        computation (after computing at (k-1)*N).
        """
        visualizer = LazyPCAVisualizer(interval=interval, trigger_on_new_best=False)
        
        if multiplier == 0:
            # Generation 0 should always compute (first computation)
            assert visualizer.should_compute(0, new_global_best=False) is True
        else:
            # Simulate that previous interval was computed
            prev_gen = (multiplier - 1) * interval
            visualizer._last_computed_gen = prev_gen
            
            target_gen = multiplier * interval
            result = visualizer.should_compute(target_gen, new_global_best=False)
            assert result is True, (
                f"Generation {target_gen} should compute (interval={interval}, "
                f"last_computed={prev_gen})"
            )


class TestProperty4PCAModelReuseBetweenIntervals:
    """
    Property 4: PCA Model Reuse Between Intervals
    
    *For any* two consecutive calls to visualize() within the same interval,
    the PCA_Visualizer SHALL return the same PCA state object without
    recomputing the transformation.
    
    **Validates: Requirements 2.2**
    """
    
    @settings(max_examples=100)
    @given(
        interval=st.integers(min_value=2, max_value=50),
        start_gen=st.integers(min_value=0, max_value=100),
        num_calls_within_interval=st.integers(min_value=2, max_value=20)
    )
    def test_same_state_returned_within_interval(
        self, interval: int, start_gen: int, num_calls_within_interval: int
    ):
        """
        **Validates: Requirements 2.2**
        
        For any sequence of visualize() calls within the same interval,
        the same PCA state object should be returned without recomputation.
        """
        visualizer = LazyPCAVisualizer(interval=interval, trigger_on_new_best=False)
        call_count = [0]
        state_id = [0]
        
        def mock_pca_func(*args, **kwargs):
            call_count[0] += 1
            state_id[0] += 1
            return {'state_id': state_id[0], 'computed_at': kwargs.get('generation')}
        
        # First call at start_gen should compute
        first_result = visualizer.visualize(
            population=[], halloffame=None, generation=start_gen,
            console=None, watch_path='/tmp/test',
            pca_visualization_func=mock_pca_func
        )
        
        initial_call_count = call_count[0]
        first_state_id = first_result['state_id']
        
        # Subsequent calls within the same interval should return same state
        # Limit calls to stay within the interval
        max_gen_in_interval = start_gen + interval - 1
        
        for i in range(1, min(num_calls_within_interval, interval)):
            gen = start_gen + i
            if gen > max_gen_in_interval:
                break
            
            result = visualizer.visualize(
                population=[], halloffame=None, generation=gen,
                console=None, watch_path='/tmp/test',
                pca_visualization_func=mock_pca_func
            )
            
            # Should not have called the PCA function again
            assert call_count[0] == initial_call_count, (
                f"PCA function should not be called at generation {gen} "
                f"(within interval starting at {start_gen})"
            )
            
            # Should return the same state object
            assert result['state_id'] == first_state_id, (
                f"State ID should be {first_state_id} at generation {gen}, "
                f"got {result['state_id']}"
            )
    
    @settings(max_examples=100)
    @given(
        interval=st.integers(min_value=1, max_value=50),
        num_intervals=st.integers(min_value=1, max_value=10)
    )
    def test_state_identity_preserved_within_each_interval(
        self, interval: int, num_intervals: int
    ):
        """
        **Validates: Requirements 2.2**
        
        For any number of intervals, the state object identity should be
        preserved within each interval (same object returned).
        """
        visualizer = LazyPCAVisualizer(interval=interval, trigger_on_new_best=False)
        
        def mock_pca_func(*args, **kwargs):
            # Return a new dict each time (unique object)
            return {'generation': kwargs.get('generation')}
        
        for interval_idx in range(num_intervals):
            interval_start = interval_idx * interval
            
            # First call in interval should compute
            first_result = visualizer.visualize(
                population=[], halloffame=None, generation=interval_start,
                console=None, watch_path='/tmp/test',
                pca_visualization_func=mock_pca_func
            )
            
            # All subsequent calls in this interval should return same object
            for offset in range(1, interval):
                gen = interval_start + offset
                result = visualizer.visualize(
                    population=[], halloffame=None, generation=gen,
                    console=None, watch_path='/tmp/test',
                    pca_visualization_func=mock_pca_func
                )
                
                # Should be the exact same object (identity check)
                assert result is first_result, (
                    f"Result at gen {gen} should be same object as gen {interval_start}"
                )


class TestProperty5PCATriggerOnNewGlobalBest:
    """
    Property 5: PCA Triggers on New Global Best
    
    *For any* generation where new_global_best=True, the PCA_Visualizer SHALL
    compute PCA regardless of whether the interval threshold has been reached.
    
    **Validates: Requirements 2.3**
    """
    
    @settings(max_examples=100)
    @given(
        interval=st.integers(min_value=2, max_value=50),
        last_computed_gen=st.integers(min_value=0, max_value=100),
        current_gen_offset=st.integers(min_value=1, max_value=49)
    )
    def test_new_global_best_triggers_computation_before_interval(
        self, interval: int, last_computed_gen: int, current_gen_offset: int
    ):
        """
        **Validates: Requirements 2.3**
        
        For any generation before the interval threshold, if new_global_best=True,
        PCA should be computed.
        """
        # Ensure we're testing a generation before the interval threshold
        assume(current_gen_offset < interval)
        
        visualizer = LazyPCAVisualizer(interval=interval, trigger_on_new_best=True)
        visualizer._last_computed_gen = last_computed_gen
        
        current_gen = last_computed_gen + current_gen_offset
        
        # Without new_global_best, should not compute
        assert visualizer.should_compute(current_gen, new_global_best=False) is False, (
            f"Generation {current_gen} should not compute without new_global_best "
            f"(interval={interval}, last_computed={last_computed_gen})"
        )
        
        # With new_global_best, should compute
        assert visualizer.should_compute(current_gen, new_global_best=True) is True, (
            f"Generation {current_gen} should compute with new_global_best=True "
            f"(interval={interval}, last_computed={last_computed_gen})"
        )
    
    @settings(max_examples=100)
    @given(
        interval=st.integers(min_value=1, max_value=50),
        generation=st.integers(min_value=0, max_value=200)
    )
    def test_new_global_best_always_triggers_when_enabled(
        self, interval: int, generation: int
    ):
        """
        **Validates: Requirements 2.3**
        
        For any generation, if trigger_on_new_best is enabled and
        new_global_best=True, PCA should be computed.
        """
        visualizer = LazyPCAVisualizer(interval=interval, trigger_on_new_best=True)
        
        # Set last_computed_gen to something that would normally prevent computation
        if generation > 0:
            visualizer._last_computed_gen = generation - 1
        
        result = visualizer.should_compute(generation, new_global_best=True)
        assert result is True, (
            f"Generation {generation} should compute with new_global_best=True "
            f"(trigger_on_new_best=True)"
        )
    
    @settings(max_examples=100)
    @given(
        interval=st.integers(min_value=2, max_value=50),
        last_computed_gen=st.integers(min_value=0, max_value=100),
        current_gen_offset=st.integers(min_value=1, max_value=49)
    )
    def test_new_global_best_ignored_when_trigger_disabled(
        self, interval: int, last_computed_gen: int, current_gen_offset: int
    ):
        """
        **Validates: Requirements 2.3**
        
        When trigger_on_new_best is False, new_global_best should not
        trigger early computation.
        """
        # Ensure we're testing a generation before the interval threshold
        assume(current_gen_offset < interval)
        
        visualizer = LazyPCAVisualizer(interval=interval, trigger_on_new_best=False)
        visualizer._last_computed_gen = last_computed_gen
        
        current_gen = last_computed_gen + current_gen_offset
        
        # Even with new_global_best=True, should not compute before interval
        result = visualizer.should_compute(current_gen, new_global_best=True)
        assert result is False, (
            f"Generation {current_gen} should not compute even with new_global_best=True "
            f"when trigger_on_new_best=False (interval={interval}, last_computed={last_computed_gen})"
        )
    
    @settings(max_examples=100)
    @given(
        interval=st.integers(min_value=2, max_value=50),
        num_new_bests=st.integers(min_value=1, max_value=10)
    )
    def test_visualize_recomputes_on_each_new_global_best(
        self, interval: int, num_new_bests: int
    ):
        """
        **Validates: Requirements 2.3**
        
        For any sequence of new global bests, each should trigger a fresh
        PCA computation.
        """
        visualizer = LazyPCAVisualizer(interval=interval, trigger_on_new_best=True)
        call_count = [0]
        
        def mock_pca_func(*args, **kwargs):
            call_count[0] += 1
            return {'call_number': call_count[0]}
        
        # Initial computation at generation 0
        visualizer.visualize(
            population=[], halloffame=None, generation=0,
            console=None, watch_path='/tmp/test',
            pca_visualization_func=mock_pca_func
        )
        
        initial_calls = call_count[0]
        
        # Each new global best should trigger a new computation
        for i in range(num_new_bests):
            gen = i + 1  # Generations 1, 2, 3, ... (all before interval)
            if gen >= interval:
                break  # Stay within first interval for this test
            
            result = visualizer.visualize(
                population=[], halloffame=None, generation=gen,
                console=None, watch_path='/tmp/test',
                new_global_best=True,
                pca_visualization_func=mock_pca_func
            )
            
            expected_calls = initial_calls + i + 1
            assert call_count[0] == expected_calls, (
                f"Expected {expected_calls} calls after {i + 1} new global bests, "
                f"got {call_count[0]}"
            )
            assert result['call_number'] == expected_calls, (
                f"Result should reflect call number {expected_calls}"
            )
