"""
Unit tests for ArgumentPool class.

Tests the pre-allocation, reuse, and expansion functionality of the
ArgumentPool class.

Requirements: 1.1, 1.2, 1.3
"""

import pytest

# Direct import to avoid loading heavy dependencies from __init__.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deap_optimizer.argument_pool import ArgumentPool


class MockEvaluator:
    """Mock evaluator for testing."""
    
    def evaluate(self, individual):
        return (sum(individual), sum(individual))


class TestArgumentPoolInitialization:
    """Tests for ArgumentPool initialization (Requirement 1.1)."""
    
    def test_init_creates_correct_number_of_containers(self):
        """Pool should pre-allocate the specified number of containers."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=10, evaluator=evaluator)
        
        assert pool.capacity == 10
        assert len(pool._args) == 10
    
    def test_init_containers_have_correct_structure(self):
        """Each container should have [evaluator, None, False, index]."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=5, evaluator=evaluator)
        
        for i, arg in enumerate(pool._args):
            assert len(arg) == 4
            assert arg[0] is evaluator
            assert arg[1] is None
            assert arg[2] is False
            assert arg[3] == i
    
    def test_init_with_zero_size(self):
        """Pool should handle zero size initialization."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=0, evaluator=evaluator)
        
        assert pool.capacity == 0
        assert len(pool._args) == 0


class TestArgumentPoolGetArgs:
    """Tests for ArgumentPool.get_args() (Requirement 1.2)."""
    
    def test_get_args_returns_correct_number_of_containers(self):
        """get_args should return exactly as many containers as individuals."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=10, evaluator=evaluator)
        
        individuals = [[1, 2], [3, 4], [5, 6]]
        args = pool.get_args(individuals, set())
        
        assert len(args) == 3
    
    def test_get_args_updates_individual_references(self):
        """get_args should update individual references in containers."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=10, evaluator=evaluator)
        
        individuals = [[1, 2], [3, 4], [5, 6]]
        args = pool.get_args(individuals, set())
        
        for i, arg in enumerate(args):
            assert arg[1] is individuals[i]
    
    def test_get_args_updates_show_me_flags(self):
        """get_args should set show_me flags based on show_me_indices."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=10, evaluator=evaluator)
        
        individuals = [[1, 2], [3, 4], [5, 6], [7, 8]]
        show_me_indices = {1, 3}
        args = pool.get_args(individuals, show_me_indices)
        
        assert args[0][2] is False
        assert args[1][2] is True
        assert args[2][2] is False
        assert args[3][2] is True
    
    def test_get_args_updates_indices(self):
        """get_args should set correct indices in containers."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=10, evaluator=evaluator)
        
        individuals = [[1, 2], [3, 4], [5, 6]]
        args = pool.get_args(individuals, set())
        
        for i, arg in enumerate(args):
            assert arg[3] == i
    
    def test_get_args_reuses_same_containers(self):
        """get_args should reuse the same container objects across calls."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=10, evaluator=evaluator)
        
        individuals1 = [[1, 2], [3, 4]]
        args1 = pool.get_args(individuals1, set())
        container_ids1 = [id(arg) for arg in args1]
        
        individuals2 = [[5, 6], [7, 8]]
        args2 = pool.get_args(individuals2, set())
        container_ids2 = [id(arg) for arg in args2]
        
        # Same container objects should be reused
        assert container_ids1 == container_ids2
    
    def test_get_args_clears_show_me_flags_between_calls(self):
        """get_args should reset show_me flags from previous calls."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=10, evaluator=evaluator)
        
        individuals = [[1, 2], [3, 4], [5, 6]]
        
        # First call with show_me on index 1
        pool.get_args(individuals, {1})
        
        # Second call with no show_me
        args = pool.get_args(individuals, set())
        
        # All show_me flags should be False
        for arg in args:
            assert arg[2] is False


class TestArgumentPoolExpand:
    """Tests for ArgumentPool._expand() (Requirement 1.3)."""
    
    def test_expand_increases_capacity(self):
        """_expand should increase the pool capacity."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=5, evaluator=evaluator)
        
        pool._expand(10)
        
        assert pool.capacity == 10
        assert len(pool._args) == 10
    
    def test_expand_preserves_existing_containers(self):
        """_expand should preserve existing container objects."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=5, evaluator=evaluator)
        
        original_ids = [id(arg) for arg in pool._args]
        pool._expand(10)
        
        # Original containers should still be the same objects
        for i, original_id in enumerate(original_ids):
            assert id(pool._args[i]) == original_id
    
    def test_expand_new_containers_have_correct_structure(self):
        """New containers from _expand should have correct structure."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=5, evaluator=evaluator)
        
        pool._expand(10)
        
        # Check new containers (indices 5-9)
        for i in range(5, 10):
            arg = pool._args[i]
            assert len(arg) == 4
            assert arg[0] is evaluator
            assert arg[1] is None
            assert arg[2] is False
            assert arg[3] == i
    
    def test_get_args_triggers_expand_when_needed(self):
        """get_args should automatically expand when population exceeds capacity."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=3, evaluator=evaluator)
        
        # Request more individuals than capacity
        individuals = [[i] for i in range(7)]
        args = pool.get_args(individuals, set())
        
        assert pool.capacity == 7
        assert len(args) == 7
    
    def test_capacity_never_decreases(self):
        """Pool capacity should never decrease after expansion."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=5, evaluator=evaluator)
        
        # Expand to 10
        pool.get_args([[i] for i in range(10)], set())
        assert pool.capacity == 10
        
        # Request fewer individuals - capacity should stay at 10
        pool.get_args([[i] for i in range(3)], set())
        assert pool.capacity == 10


class TestArgumentPoolClearReferences:
    """Tests for ArgumentPool.clear_references()."""
    
    def test_clear_references_sets_individuals_to_none(self):
        """clear_references should set all individual references to None."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=5, evaluator=evaluator)
        
        individuals = [[1, 2], [3, 4], [5, 6]]
        pool.get_args(individuals, set())
        
        # Verify individuals are set
        for i in range(3):
            assert pool._args[i][1] is not None
        
        pool.clear_references()
        
        # All individual references should be None
        for arg in pool._args:
            assert arg[1] is None


class TestArgumentPoolIntegration:
    """Integration tests for ArgumentPool with realistic usage patterns."""
    
    def test_multiple_generations_reuse_containers(self):
        """Simulates multiple generations reusing the same containers."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=50, evaluator=evaluator)
        
        # Track container IDs across generations
        all_container_ids = []
        
        for gen in range(5):
            # Simulate varying population sizes
            pop_size = 40 + (gen % 3)  # 40, 41, 42, 40, 41
            individuals = [[gen, i] for i in range(pop_size)]
            show_me = {0} if gen % 2 == 0 else set()
            
            args = pool.get_args(individuals, show_me)
            
            # Store IDs of first 40 containers (common across all generations)
            if gen == 0:
                all_container_ids = [id(arg) for arg in args[:40]]
            else:
                # Verify same containers are reused
                for i in range(40):
                    assert id(args[i]) == all_container_ids[i]
    
    def test_worker_function_compatibility(self):
        """Verify containers can be unpacked like tuples in worker functions."""
        evaluator = MockEvaluator()
        pool = ArgumentPool(max_size=10, evaluator=evaluator)
        
        individuals = [[1, 2, 3], [4, 5, 6]]
        args = pool.get_args(individuals, {1})
        
        # Simulate worker function unpacking
        for arg in args:
            eval_ref, ind, show_me, idx = arg
            assert eval_ref is evaluator
            assert ind in individuals
            assert isinstance(show_me, bool)
            assert isinstance(idx, int)
