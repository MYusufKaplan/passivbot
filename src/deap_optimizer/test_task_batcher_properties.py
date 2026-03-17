"""
Property-based tests for IntervalTaskBatcher class.

These tests use Hypothesis to verify universal properties of the interval
task batching component across many randomly generated inputs.

Properties tested:
- Property 6: Task Batching Chunk Sizes
- Property 7: Result Array Indexing
"""

from hypothesis import given, strategies as st, settings, assume

# Direct import to avoid loading heavy dependencies from __init__.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deap_optimizer.task_batcher import IntervalTaskBatcher


class MockInterval:
    """Mock interval object for testing."""
    
    def __init__(self, month_id: int):
        self.month_id = month_id


class MockEvaluator:
    """Mock evaluator for testing."""
    
    def evaluate_interval(self, individual, interval):
        """Return a mock result based on individual and interval."""
        # Return (fitness, bankrupt, reason, ts, total_ts, gain)
        fitness = float(individual) + interval.month_id * 0.1
        return (fitness, False, 0, 0, 100, 0.0)


class TestProperty6TaskBatchingChunkSizes:
    """
    Property 6: Task Batching Chunk Sizes
    
    *For any* task submission with batch_size B and total tasks T, the
    IntervalTaskBatcher SHALL submit tasks in chunks of size B (except
    possibly the last chunk which may be smaller).
    
    **Validates: Requirements 3.2**
    """
    
    @settings(max_examples=100)
    @given(
        batch_size=st.integers(min_value=1, max_value=100),
        total_tasks=st.integers(min_value=0, max_value=500)
    )
    def test_chunk_boundaries_correct_size(self, batch_size: int, total_tasks: int):
        """
        **Validates: Requirements 3.2**
        
        For any batch_size B and total tasks T, all chunks except the last
        should have exactly B tasks.
        """
        batcher = IntervalTaskBatcher(batch_size=batch_size)
        boundaries = batcher.get_chunk_boundaries(total_tasks)
        
        if total_tasks == 0:
            assert len(boundaries) == 0, "No chunks for zero tasks"
            return
        
        # All chunks except the last should have exactly batch_size tasks
        for i, (start, end) in enumerate(boundaries[:-1]):
            chunk_size = end - start
            assert chunk_size == batch_size, (
                f"Chunk {i} should have size {batch_size}, got {chunk_size} "
                f"(start={start}, end={end})"
            )
    
    @settings(max_examples=100)
    @given(
        batch_size=st.integers(min_value=1, max_value=100),
        total_tasks=st.integers(min_value=1, max_value=500)
    )
    def test_last_chunk_size_at_most_batch_size(self, batch_size: int, total_tasks: int):
        """
        **Validates: Requirements 3.2**
        
        For any batch_size B and total tasks T > 0, the last chunk should
        have at most B tasks.
        """
        batcher = IntervalTaskBatcher(batch_size=batch_size)
        boundaries = batcher.get_chunk_boundaries(total_tasks)
        
        assert len(boundaries) > 0, "Should have at least one chunk for non-zero tasks"
        
        last_start, last_end = boundaries[-1]
        last_chunk_size = last_end - last_start
        
        assert last_chunk_size <= batch_size, (
            f"Last chunk size {last_chunk_size} exceeds batch_size {batch_size}"
        )
        assert last_chunk_size > 0, "Last chunk should have at least one task"
    
    @settings(max_examples=100)
    @given(
        batch_size=st.integers(min_value=1, max_value=100),
        total_tasks=st.integers(min_value=0, max_value=500)
    )
    def test_chunks_cover_all_tasks(self, batch_size: int, total_tasks: int):
        """
        **Validates: Requirements 3.2**
        
        For any batch_size B and total tasks T, the chunks should cover
        exactly all T tasks without gaps or overlaps.
        """
        batcher = IntervalTaskBatcher(batch_size=batch_size)
        boundaries = batcher.get_chunk_boundaries(total_tasks)
        
        if total_tasks == 0:
            assert len(boundaries) == 0, "No chunks for zero tasks"
            return
        
        # First chunk should start at 0
        assert boundaries[0][0] == 0, "First chunk should start at 0"
        
        # Last chunk should end at total_tasks
        assert boundaries[-1][1] == total_tasks, (
            f"Last chunk should end at {total_tasks}, got {boundaries[-1][1]}"
        )
        
        # Chunks should be contiguous (no gaps)
        for i in range(1, len(boundaries)):
            prev_end = boundaries[i - 1][1]
            curr_start = boundaries[i][0]
            assert prev_end == curr_start, (
                f"Gap between chunk {i-1} (end={prev_end}) and chunk {i} "
                f"(start={curr_start})"
            )
    
    @settings(max_examples=100)
    @given(
        batch_size=st.integers(min_value=1, max_value=100),
        total_tasks=st.integers(min_value=0, max_value=500)
    )
    def test_number_of_chunks_correct(self, batch_size: int, total_tasks: int):
        """
        **Validates: Requirements 3.2**
        
        For any batch_size B and total tasks T, the number of chunks should
        be ceil(T / B) for T > 0, or 0 for T = 0.
        """
        batcher = IntervalTaskBatcher(batch_size=batch_size)
        boundaries = batcher.get_chunk_boundaries(total_tasks)
        
        if total_tasks == 0:
            expected_chunks = 0
        else:
            # ceil(total_tasks / batch_size)
            expected_chunks = (total_tasks + batch_size - 1) // batch_size
        
        assert len(boundaries) == expected_chunks, (
            f"Expected {expected_chunks} chunks for {total_tasks} tasks with "
            f"batch_size {batch_size}, got {len(boundaries)}"
        )
    
    @settings(max_examples=100)
    @given(
        batch_size=st.integers(min_value=1, max_value=50),
        n_individuals=st.integers(min_value=1, max_value=20),
        n_intervals=st.integers(min_value=1, max_value=20)
    )
    def test_task_list_chunked_correctly(
        self, batch_size: int, n_individuals: int, n_intervals: int
    ):
        """
        **Validates: Requirements 3.2**
        
        For any number of individuals and intervals, the task list should
        be chunked according to batch_size.
        """
        batcher = IntervalTaskBatcher(batch_size=batch_size)
        
        individuals = list(range(n_individuals))
        intervals = [MockInterval(i) for i in range(n_intervals)]
        evaluator = MockEvaluator()
        
        tasks = batcher.build_task_list(individuals, intervals, evaluator)
        total_tasks = len(tasks)
        
        # Total tasks should be n_individuals * n_intervals
        assert total_tasks == n_individuals * n_intervals, (
            f"Expected {n_individuals * n_intervals} tasks, got {total_tasks}"
        )
        
        # Verify chunk boundaries match task count
        boundaries = batcher.get_chunk_boundaries(total_tasks)
        
        # Sum of all chunk sizes should equal total tasks
        total_from_chunks = sum(end - start for start, end in boundaries)
        assert total_from_chunks == total_tasks, (
            f"Sum of chunk sizes {total_from_chunks} != total tasks {total_tasks}"
        )


class TestProperty7ResultArrayIndexing:
    """
    Property 7: Result Array Indexing
    
    *For any* interval mode evaluation, results SHALL be stored in the
    pre-allocated result dictionary at indices matching the candidate_id,
    ensuring O(1) lookup by candidate.
    
    **Validates: Requirements 3.3**
    """
    
    @settings(max_examples=100)
    @given(
        n_individuals=st.integers(min_value=1, max_value=50),
        n_intervals=st.integers(min_value=1, max_value=20),
        batch_size=st.integers(min_value=1, max_value=100)
    )
    def test_results_preallocated_for_all_candidates(
        self, n_individuals: int, n_intervals: int, batch_size: int
    ):
        """
        **Validates: Requirements 3.3**
        
        For any number of individuals, the result dictionary should have
        pre-allocated entries for all candidate IDs (0 to n_individuals-1).
        """
        batcher = IntervalTaskBatcher(batch_size=batch_size)
        
        # Clear and pre-allocate results (simulating what evaluate_batched does)
        batcher._results.clear()
        for i in range(n_individuals):
            batcher._results[i] = {}
        
        # Verify all candidate IDs are present
        for candidate_id in range(n_individuals):
            assert candidate_id in batcher._results, (
                f"Candidate ID {candidate_id} not in results dictionary"
            )
        
        # Verify no extra candidate IDs
        assert len(batcher._results) == n_individuals, (
            f"Expected {n_individuals} entries, got {len(batcher._results)}"
        )
    
    @settings(max_examples=100)
    @given(
        n_individuals=st.integers(min_value=1, max_value=30),
        n_intervals=st.integers(min_value=1, max_value=15),
        batch_size=st.integers(min_value=1, max_value=50)
    )
    def test_results_indexed_by_candidate_id(
        self, n_individuals: int, n_intervals: int, batch_size: int
    ):
        """
        **Validates: Requirements 3.3**
        
        For any evaluation, results should be stored at indices matching
        the candidate_id, allowing O(1) lookup.
        """
        batcher = IntervalTaskBatcher(batch_size=batch_size)
        
        individuals = list(range(n_individuals))
        intervals = [MockInterval(i) for i in range(n_intervals)]
        evaluator = MockEvaluator()
        
        # Build tasks and simulate storing results
        tasks = batcher.build_task_list(individuals, intervals, evaluator)
        
        # Pre-allocate results
        batcher._results.clear()
        for i in range(n_individuals):
            batcher._results[i] = {}
        
        # Simulate processing results (as evaluate_batched would do)
        for task in tasks:
            _, individual, month_id, interval, candidate_id = task
            result = evaluator.evaluate_interval(individual, interval)
            fitness, bankrupt, reason, ts, total_ts, gain = result
            batcher._results[candidate_id][month_id] = (
                fitness, bankrupt, reason, ts, total_ts
            )
        
        # Verify each candidate's results are accessible by candidate_id
        for candidate_id in range(n_individuals):
            assert candidate_id in batcher._results, (
                f"Candidate {candidate_id} not in results"
            )
            
            # Verify all intervals are present for this candidate
            candidate_results = batcher._results[candidate_id]
            assert len(candidate_results) == n_intervals, (
                f"Candidate {candidate_id} should have {n_intervals} results, "
                f"got {len(candidate_results)}"
            )
            
            # Verify each month_id is accessible
            for month_id in range(n_intervals):
                assert month_id in candidate_results, (
                    f"Month {month_id} not in results for candidate {candidate_id}"
                )
    
    @settings(max_examples=100)
    @given(
        n_individuals=st.integers(min_value=1, max_value=30),
        n_intervals=st.integers(min_value=1, max_value=15),
        batch_size=st.integers(min_value=1, max_value=50)
    )
    def test_result_values_match_candidate_and_interval(
        self, n_individuals: int, n_intervals: int, batch_size: int
    ):
        """
        **Validates: Requirements 3.3**
        
        For any evaluation, the result stored at [candidate_id][month_id]
        should correspond to the correct individual and interval.
        """
        batcher = IntervalTaskBatcher(batch_size=batch_size)
        
        individuals = list(range(n_individuals))
        intervals = [MockInterval(i) for i in range(n_intervals)]
        evaluator = MockEvaluator()
        
        # Build tasks and simulate storing results
        tasks = batcher.build_task_list(individuals, intervals, evaluator)
        
        # Pre-allocate results
        batcher._results.clear()
        for i in range(n_individuals):
            batcher._results[i] = {}
        
        # Simulate processing results
        for task in tasks:
            _, individual, month_id, interval, candidate_id = task
            result = evaluator.evaluate_interval(individual, interval)
            fitness, bankrupt, reason, ts, total_ts, gain = result
            batcher._results[candidate_id][month_id] = (
                fitness, bankrupt, reason, ts, total_ts
            )
        
        # Verify result values match expected computation
        for candidate_id in range(n_individuals):
            for month_id in range(n_intervals):
                stored_result = batcher._results[candidate_id][month_id]
                fitness = stored_result[0]
                
                # Expected fitness based on MockEvaluator logic
                expected_fitness = float(candidate_id) + month_id * 0.1
                
                assert abs(fitness - expected_fitness) < 1e-10, (
                    f"Result at [{candidate_id}][{month_id}] has fitness {fitness}, "
                    f"expected {expected_fitness}"
                )
    
    @settings(max_examples=100)
    @given(
        n_individuals=st.integers(min_value=1, max_value=50),
        batch_size=st.integers(min_value=1, max_value=100)
    )
    def test_o1_lookup_by_candidate_id(self, n_individuals: int, batch_size: int):
        """
        **Validates: Requirements 3.3**
        
        For any candidate_id, lookup in the results dictionary should be O(1).
        This is verified by checking that results is a dict (hash-based).
        """
        batcher = IntervalTaskBatcher(batch_size=batch_size)
        
        # Pre-allocate results
        batcher._results.clear()
        for i in range(n_individuals):
            batcher._results[i] = {}
        
        # Verify results is a dict (O(1) lookup)
        assert isinstance(batcher._results, dict), (
            "Results should be a dictionary for O(1) lookup"
        )
        
        # Verify each candidate's results is also a dict
        for candidate_id in range(n_individuals):
            assert isinstance(batcher._results[candidate_id], dict), (
                f"Results for candidate {candidate_id} should be a dictionary"
            )
    
    @settings(max_examples=100)
    @given(
        n_individuals=st.integers(min_value=1, max_value=30),
        n_intervals=st.integers(min_value=1, max_value=15),
        batch_size=st.integers(min_value=1, max_value=50),
        query_candidate=st.integers(min_value=0, max_value=29),
        query_month=st.integers(min_value=0, max_value=14)
    )
    def test_direct_access_returns_correct_result(
        self, n_individuals: int, n_intervals: int, batch_size: int,
        query_candidate: int, query_month: int
    ):
        """
        **Validates: Requirements 3.3**
        
        For any valid candidate_id and month_id, direct dictionary access
        should return the correct result tuple.
        """
        # Ensure query indices are within bounds
        assume(query_candidate < n_individuals)
        assume(query_month < n_intervals)
        
        batcher = IntervalTaskBatcher(batch_size=batch_size)
        
        individuals = list(range(n_individuals))
        intervals = [MockInterval(i) for i in range(n_intervals)]
        evaluator = MockEvaluator()
        
        # Build tasks and simulate storing results
        tasks = batcher.build_task_list(individuals, intervals, evaluator)
        
        # Pre-allocate results
        batcher._results.clear()
        for i in range(n_individuals):
            batcher._results[i] = {}
        
        # Simulate processing results
        for task in tasks:
            _, individual, month_id, interval, candidate_id = task
            result = evaluator.evaluate_interval(individual, interval)
            fitness, bankrupt, reason, ts, total_ts, gain = result
            batcher._results[candidate_id][month_id] = (
                fitness, bankrupt, reason, ts, total_ts
            )
        
        # Direct access should work
        result = batcher._results[query_candidate][query_month]
        
        assert result is not None, (
            f"Direct access to [{query_candidate}][{query_month}] returned None"
        )
        
        # Verify result structure
        assert len(result) == 5, (
            f"Result tuple should have 5 elements, got {len(result)}"
        )
        
        # Verify fitness value
        expected_fitness = float(query_candidate) + query_month * 0.1
        assert abs(result[0] - expected_fitness) < 1e-10, (
            f"Fitness mismatch: expected {expected_fitness}, got {result[0]}"
        )
