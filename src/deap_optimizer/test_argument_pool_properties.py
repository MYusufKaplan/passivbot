"""
Property-based tests for ArgumentPool class.

This module contains Hypothesis property tests for validating the correctness
of the ArgumentPool class, specifically:
- Property 1: Argument Pool Reuse (Requirements 1.2)
- Property 2: Argument Pool Resize Only When Needed (Requirements 1.3)

Testing Framework: pytest + Hypothesis
Minimum iterations: 100 per property test
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import List, Set

# Direct import to avoid loading heavy dependencies from __init__.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deap_optimizer.argument_pool import ArgumentPool


class MockEvaluator:
    """Mock evaluator for property testing."""
    
    def evaluate(self, individual):
        return (sum(individual), sum(individual))


# =============================================================================
# Property 1: Argument Pool Reuse
# =============================================================================

@settings(max_examples=100)
@given(
    initial_size=st.integers(min_value=1, max_value=100),
    num_evaluations=st.integers(min_value=2, max_value=20),
    pop_sizes=st.lists(
        st.integers(min_value=1, max_value=50),
        min_size=2,
        max_size=20
    ),
)
def test_property_1_argument_pool_reuse(
    initial_size: int,
    num_evaluations: int,
    pop_sizes: List[int],
):
    """
    Property 1: Argument Pool Reuse.
    
    For any sequence of population evaluations, the ArgumentPool SHALL reuse
    the same argument container objects across evaluations, verifiable by
    checking object identity of the argument lists returned by consecutive
    calls to get_args().
    
    **Validates: Requirements 1.2**
    
    Tag: Feature: performance-optimization, Property 1: Argument Pool Reuse
    """
    # Ensure we have enough population sizes for the evaluations
    assume(len(pop_sizes) >= num_evaluations)
    
    evaluator = MockEvaluator()
    pool = ArgumentPool(max_size=initial_size, evaluator=evaluator)
    
    # Track container IDs from first evaluation
    first_container_ids = None
    min_common_size = float('inf')
    
    for eval_idx in range(num_evaluations):
        pop_size = pop_sizes[eval_idx]
        individuals = [[eval_idx, i] for i in range(pop_size)]
        show_me_indices: Set[int] = set()
        
        args = pool.get_args(individuals, show_me_indices)
        
        if first_container_ids is None:
            # Store IDs from first evaluation
            first_container_ids = [id(arg) for arg in args]
            min_common_size = len(args)
        else:
            # Update minimum common size
            min_common_size = min(min_common_size, len(args), len(first_container_ids))
            
            # Verify that containers up to min_common_size are reused
            for i in range(min_common_size):
                assert id(args[i]) == first_container_ids[i], (
                    f"Container at index {i} should be reused across evaluations. "
                    f"Expected id {first_container_ids[i]}, got {id(args[i])}"
                )


@settings(max_examples=100)
@given(
    initial_size=st.integers(min_value=10, max_value=100),
    pop_size=st.integers(min_value=1, max_value=50),
    num_generations=st.integers(min_value=5, max_value=50),
)
def test_property_1_reuse_across_generations(
    initial_size: int,
    pop_size: int,
    num_generations: int,
):
    """
    Property 1 (extended): Containers are reused across many generations.
    
    For any number of generations with the same population size, the
    ArgumentPool SHALL reuse the exact same container objects for every
    generation.
    
    **Validates: Requirements 1.2**
    """
    assume(pop_size <= initial_size)  # No expansion needed
    
    evaluator = MockEvaluator()
    pool = ArgumentPool(max_size=initial_size, evaluator=evaluator)
    
    # Get container IDs from first generation
    individuals = [[0, i] for i in range(pop_size)]
    first_args = pool.get_args(individuals, set())
    first_ids = [id(arg) for arg in first_args]
    
    # Verify same containers are reused across all generations
    for gen in range(1, num_generations):
        individuals = [[gen, i] for i in range(pop_size)]
        show_me = {gen % pop_size} if pop_size > 0 else set()
        
        args = pool.get_args(individuals, show_me)
        
        assert len(args) == pop_size
        for i in range(pop_size):
            assert id(args[i]) == first_ids[i], (
                f"Generation {gen}: Container at index {i} should be reused. "
                f"Expected id {first_ids[i]}, got {id(args[i])}"
            )


@settings(max_examples=100)
@given(
    initial_size=st.integers(min_value=5, max_value=50),
    pop_size=st.integers(min_value=1, max_value=30),
    show_me_indices=st.lists(st.integers(min_value=0, max_value=29), max_size=10),
)
def test_property_1_reuse_with_varying_show_me(
    initial_size: int,
    pop_size: int,
    show_me_indices: List[int],
):
    """
    Property 1 (extended): Containers are reused regardless of show_me flags.
    
    For any sequence of evaluations with different show_me_indices, the
    ArgumentPool SHALL reuse the same container objects.
    
    **Validates: Requirements 1.2**
    """
    assume(pop_size <= initial_size)
    
    evaluator = MockEvaluator()
    pool = ArgumentPool(max_size=initial_size, evaluator=evaluator)
    
    individuals = [[i] for i in range(pop_size)]
    
    # First call with no show_me
    first_args = pool.get_args(individuals, set())
    first_ids = [id(arg) for arg in first_args]
    
    # Second call with show_me indices (filtered to valid range)
    valid_show_me = {idx for idx in show_me_indices if idx < pop_size}
    second_args = pool.get_args(individuals, valid_show_me)
    
    # Verify same containers are reused
    for i in range(pop_size):
        assert id(second_args[i]) == first_ids[i], (
            f"Container at index {i} should be reused regardless of show_me flags"
        )


# =============================================================================
# Property 2: Argument Pool Resize Only When Needed
# =============================================================================

@settings(max_examples=100)
@given(
    initial_size=st.integers(min_value=1, max_value=50),
    pop_sizes=st.lists(
        st.integers(min_value=1, max_value=100),
        min_size=1,
        max_size=30
    ),
)
def test_property_2_resize_only_when_needed(
    initial_size: int,
    pop_sizes: List[int],
):
    """
    Property 2: Argument Pool Resize Only When Needed.
    
    For any sequence of population sizes, the ArgumentPool capacity SHALL
    only increase when a population size exceeds the current capacity,
    and SHALL never decrease.
    
    **Validates: Requirements 1.3**
    
    Tag: Feature: performance-optimization, Property 2: Argument Pool Resize Only When Needed
    """
    evaluator = MockEvaluator()
    pool = ArgumentPool(max_size=initial_size, evaluator=evaluator)
    
    max_capacity_seen = initial_size
    
    for pop_size in pop_sizes:
        individuals = [[i] for i in range(pop_size)]
        pool.get_args(individuals, set())
        
        current_capacity = pool.capacity
        
        # Property: capacity should be at least as large as the largest population seen
        expected_min_capacity = max(max_capacity_seen, pop_size)
        assert current_capacity >= expected_min_capacity, (
            f"Capacity {current_capacity} should be at least {expected_min_capacity}"
        )
        
        # Property: capacity should never decrease
        assert current_capacity >= max_capacity_seen, (
            f"Capacity decreased from {max_capacity_seen} to {current_capacity}"
        )
        
        max_capacity_seen = max(max_capacity_seen, current_capacity)


@settings(max_examples=100)
@given(
    initial_size=st.integers(min_value=10, max_value=50),
    small_sizes=st.lists(
        st.integers(min_value=1, max_value=10),
        min_size=5,
        max_size=20
    ),
)
def test_property_2_no_resize_for_smaller_populations(
    initial_size: int,
    small_sizes: List[int],
):
    """
    Property 2 (extended): No resize when population is smaller than capacity.
    
    For any sequence of population sizes all smaller than the initial capacity,
    the ArgumentPool capacity SHALL remain unchanged.
    
    **Validates: Requirements 1.3**
    """
    # Ensure all sizes are smaller than initial
    assume(all(size < initial_size for size in small_sizes))
    
    evaluator = MockEvaluator()
    pool = ArgumentPool(max_size=initial_size, evaluator=evaluator)
    
    for pop_size in small_sizes:
        individuals = [[i] for i in range(pop_size)]
        pool.get_args(individuals, set())
        
        # Capacity should remain at initial_size
        assert pool.capacity == initial_size, (
            f"Capacity changed from {initial_size} to {pool.capacity} "
            f"even though population size {pop_size} was smaller"
        )


@settings(max_examples=100)
@given(
    initial_size=st.integers(min_value=5, max_value=20),
    expansion_sizes=st.lists(
        st.integers(min_value=1, max_value=100),
        min_size=3,
        max_size=15
    ),
)
def test_property_2_capacity_monotonically_increases(
    initial_size: int,
    expansion_sizes: List[int],
):
    """
    Property 2 (extended): Capacity monotonically increases.
    
    For any sequence of population sizes, the ArgumentPool capacity SHALL
    form a monotonically non-decreasing sequence.
    
    **Validates: Requirements 1.3**
    """
    evaluator = MockEvaluator()
    pool = ArgumentPool(max_size=initial_size, evaluator=evaluator)
    
    previous_capacity = initial_size
    
    for pop_size in expansion_sizes:
        individuals = [[i] for i in range(pop_size)]
        pool.get_args(individuals, set())
        
        current_capacity = pool.capacity
        
        # Capacity should never decrease
        assert current_capacity >= previous_capacity, (
            f"Capacity decreased from {previous_capacity} to {current_capacity}"
        )
        
        previous_capacity = current_capacity


@settings(max_examples=100)
@given(
    initial_size=st.integers(min_value=5, max_value=30),
    larger_size=st.integers(min_value=31, max_value=100),
)
def test_property_2_expansion_preserves_existing_containers(
    initial_size: int,
    larger_size: int,
):
    """
    Property 2 (extended): Expansion preserves existing container objects.
    
    When the pool expands to accommodate a larger population, existing
    container objects SHALL be preserved (same object identity).
    
    **Validates: Requirements 1.3**
    """
    evaluator = MockEvaluator()
    pool = ArgumentPool(max_size=initial_size, evaluator=evaluator)
    
    # Get initial containers
    individuals_small = [[i] for i in range(initial_size)]
    args_before = pool.get_args(individuals_small, set())
    ids_before = [id(arg) for arg in args_before]
    
    # Trigger expansion
    individuals_large = [[i] for i in range(larger_size)]
    args_after = pool.get_args(individuals_large, set())
    
    # Verify original containers are preserved
    for i in range(initial_size):
        assert id(args_after[i]) == ids_before[i], (
            f"Container at index {i} was not preserved during expansion. "
            f"Expected id {ids_before[i]}, got {id(args_after[i])}"
        )


@settings(max_examples=100)
@given(
    initial_size=st.integers(min_value=1, max_value=20),
    expansion_sequence=st.lists(
        st.integers(min_value=1, max_value=150),
        min_size=5,
        max_size=20
    ),
)
def test_property_2_capacity_equals_max_population_seen(
    initial_size: int,
    expansion_sequence: List[int],
):
    """
    Property 2 (extended): Capacity equals maximum population size seen.
    
    After any sequence of evaluations, the ArgumentPool capacity SHALL
    equal the maximum of the initial size and all population sizes seen.
    
    **Validates: Requirements 1.3**
    """
    evaluator = MockEvaluator()
    pool = ArgumentPool(max_size=initial_size, evaluator=evaluator)
    
    max_pop_seen = initial_size
    
    for pop_size in expansion_sequence:
        individuals = [[i] for i in range(pop_size)]
        pool.get_args(individuals, set())
        
        max_pop_seen = max(max_pop_seen, pop_size)
        
        # Capacity should equal the maximum population seen
        assert pool.capacity == max_pop_seen, (
            f"Capacity {pool.capacity} should equal max population seen {max_pop_seen}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
