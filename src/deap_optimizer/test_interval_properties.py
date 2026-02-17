"""
Property-based tests for the interval module.

This module contains Hypothesis property tests for validating the correctness
of the monthly interval evaluation system.

Testing Framework: pytest + Hypothesis
Minimum iterations: 100 per property test
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import Dict, Tuple

from deap_optimizer.interval import (
    compute_monthly_fitness,
)


# =============================================================================
# Property 7: Returned fitness tuples are valid DEAP fitness format
# =============================================================================

@settings(max_examples=100)
@given(
    # Generate random candidate results: {month_id: (fitness_value, bankruptcy_flag, bankruptcy_timestep, total_timesteps)}
    num_months=st.integers(min_value=1, max_value=36),
)
def test_property_7_fitness_tuples_valid_deap_format(
    num_months: int,
):
    """
    Property 7: Returned fitness tuples are valid DEAP fitness format.
    
    For any set of candidate results from monthly interval evaluation,
    each returned fitness SHALL be a tuple of exactly 2 floats
    (both equal, matching the existing (w_0, w_1) pattern).
    
    **Validates: Requirements 8.2**
    
    Tag: Feature: monthly-interval-evaluation, Property 7: Returned fitness tuples are valid DEAP fitness format
    """
    import random
    
    # Build candidate_results dictionary
    candidate_results: Dict[int, Tuple[float, bool, int, int]] = {}
    for month_id in range(num_months):
        fitness_value = random.uniform(0.0, 1e10)
        bankruptcy_flag = random.choice([True, False])
        total_timesteps = random.randint(1, 100000)
        
        # Generate bankruptcy_timestep based on bankruptcy_flag
        if bankruptcy_flag:
            # Bankruptcy occurred at some point before total_timesteps
            bankruptcy_timestep = random.randint(0, total_timesteps - 1) if total_timesteps > 1 else 0
        else:
            bankruptcy_timestep = total_timesteps  # No bankruptcy
        
        candidate_results[month_id] = (
            fitness_value,
            bankruptcy_flag,
            bankruptcy_timestep,
            total_timesteps
        )
    
    # Call compute_monthly_fitness
    result = compute_monthly_fitness(candidate_results, num_months)
    
    # Property assertions:
    # 1. Result is a tuple
    assert isinstance(result, tuple), f"Result should be a tuple, got {type(result)}"
    
    # 2. Tuple has exactly 2 elements
    assert len(result) == 2, f"Fitness tuple should have exactly 2 elements, got {len(result)}"
    
    # 3. Both elements are floats (or can be converted to float)
    assert isinstance(result[0], (int, float)), f"First element should be numeric, got {type(result[0])}"
    assert isinstance(result[1], (int, float)), f"Second element should be numeric, got {type(result[1])}"
    
    # 4. Both elements are equal (matching the (w_0, w_1) pattern)
    assert result[0] == result[1], f"Both fitness values should be equal, got {result[0]} and {result[1]}"
    
    # 5. Values are finite (not NaN or infinity)
    import math
    assert not math.isnan(result[0]), "Fitness value should not be NaN"
    assert not math.isinf(result[0]), "Fitness value should not be infinite"


@settings(max_examples=100)
@given(
    # Generate multiple candidates with varying numbers of months
    num_candidates=st.integers(min_value=1, max_value=20),
    num_months=st.integers(min_value=1, max_value=12),
)
def test_property_7_multiple_candidates_valid_format(
    num_candidates: int,
    num_months: int,
):
    """
    Property 7 (extended): Multiple candidates all produce valid DEAP fitness format.
    
    For any number of candidates, each returned fitness tuple SHALL be
    a tuple of exactly 2 equal floats.
    
    **Validates: Requirements 8.2**
    """
    import random
    
    for candidate_id in range(num_candidates):
        # Build random candidate_results for this candidate
        candidate_results: Dict[int, Tuple[float, bool, int, int]] = {}
        for month_id in range(num_months):
            fitness_value = random.uniform(0.0, 1e6)
            bankruptcy_flag = random.choice([True, False])
            total_timesteps = random.randint(100, 10000)
            bankruptcy_timestep = random.randint(0, total_timesteps) if bankruptcy_flag else total_timesteps
            
            candidate_results[month_id] = (
                fitness_value,
                bankruptcy_flag,
                bankruptcy_timestep,
                total_timesteps
            )
        
        # Call compute_monthly_fitness
        result = compute_monthly_fitness(candidate_results, num_months)
        
        # Property assertions
        assert isinstance(result, tuple), f"Candidate {candidate_id}: Result should be a tuple"
        assert len(result) == 2, f"Candidate {candidate_id}: Tuple should have 2 elements"
        assert result[0] == result[1], f"Candidate {candidate_id}: Both values should be equal"
        
        import math
        assert not math.isnan(result[0]), f"Candidate {candidate_id}: Value should not be NaN"
        assert not math.isinf(result[0]), f"Candidate {candidate_id}: Value should not be infinite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
