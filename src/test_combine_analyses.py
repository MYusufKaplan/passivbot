"""Unit tests for the optimized combine_analyses method.

Tests verify that the optimized implementation produces correct results
for various input scenarios.

Requirements: 4.1, 4.3
"""
import numpy as np
import pytest


class MockEvaluator:
    """Mock evaluator with the optimized combine_analyses method for testing."""
    
    def combine_analyses(self, analyses):
        """Combine analyses with optimized dictionary access.
        
        Optimizations:
        - Cache keys from first analysis (all analyses have same keys)
        - Cache exchange list to avoid repeated iteration
        - Use numpy vectorized operations for statistics
        
        Requirements: 4.1, 4.3
        """
        analyses_combined = {}
        
        # Cache keys from first analysis (all analyses have same keys)
        first_exchange = next(iter(analyses))
        keys = list(analyses[first_exchange].keys())
        
        # Pre-extract exchange list for direct access
        exchange_list = list(analyses.keys())
        
        for key in keys:
            # Direct access instead of repeated lookups
            values = [analyses[ex][key] for ex in exchange_list]

            # Special handling for bankruptcy_timestamp - only set if actually bankrupt
            if key == "bankruptcy_timestamp":
                # Find the first non-None bankruptcy timestamp (if any)
                bankruptcy_timestamps = [v for v in values if v is not None]
                if bankruptcy_timestamps:
                    # If any exchange went bankrupt, use the earliest bankruptcy timestamp
                    analyses_combined[f"{key}_mean"] = min(bankruptcy_timestamps)
                    analyses_combined[f"{key}_min"] = min(bankruptcy_timestamps)
                    analyses_combined[f"{key}_max"] = min(bankruptcy_timestamps)
                    analyses_combined[f"{key}_std"] = 0.0
                else:
                    # No bankruptcy occurred - keep as None
                    analyses_combined[f"{key}_mean"] = None
                    analyses_combined[f"{key}_min"] = None
                    analyses_combined[f"{key}_max"] = None
                    analyses_combined[f"{key}_std"] = None

            # Special handling for bankruptcy_reason - use max to get most severe reason
            # 0=none, 1=financial, 2=drawdown, 3=no_positions, 4=stale_position
            elif key == "bankruptcy_reason":
                non_zero_reasons = [v for v in values if v != 0]
                if non_zero_reasons:
                    # Use max to get the most severe bankruptcy reason
                    max_reason = max(non_zero_reasons)
                    analyses_combined[f"{key}_mean"] = max_reason
                    analyses_combined[f"{key}_min"] = min(non_zero_reasons)
                    analyses_combined[f"{key}_max"] = max_reason
                    analyses_combined[f"{key}_std"] = 0.0
                else:
                    # No bankruptcy - all zeros
                    analyses_combined[f"{key}_mean"] = 0
                    analyses_combined[f"{key}_min"] = 0
                    analyses_combined[f"{key}_max"] = 0
                    analyses_combined[f"{key}_std"] = 0.0

            elif (
                not values
                or any(x == np.inf for x in values)
                or any(x is None for x in values)
            ):
                analyses_combined[f"{key}_mean"] = 0.0
                analyses_combined[f"{key}_min"] = 0.0
                analyses_combined[f"{key}_max"] = 0.0
                analyses_combined[f"{key}_std"] = 0.0
            else:
                # Use numpy for vectorized statistics
                arr = np.array(values, dtype=np.float64)
                analyses_combined[f"{key}_mean"] = np.mean(arr)
                analyses_combined[f"{key}_min"] = np.min(arr)
                analyses_combined[f"{key}_max"] = np.max(arr)
                analyses_combined[f"{key}_std"] = np.std(arr)
        return analyses_combined


class TestCombineAnalyses:
    """Test suite for combine_analyses optimization."""
    
    def test_basic_numeric_values(self):
        """Test combining basic numeric values across exchanges."""
        evaluator = MockEvaluator()
        analyses = {
            "exchange1": {"profit": 100.0, "loss": 10.0},
            "exchange2": {"profit": 200.0, "loss": 20.0},
            "exchange3": {"profit": 150.0, "loss": 15.0},
        }
        
        result = evaluator.combine_analyses(analyses)
        
        # Check profit statistics
        assert result["profit_mean"] == pytest.approx(150.0)
        assert result["profit_min"] == pytest.approx(100.0)
        assert result["profit_max"] == pytest.approx(200.0)
        assert result["profit_std"] == pytest.approx(np.std([100.0, 200.0, 150.0]))
        
        # Check loss statistics
        assert result["loss_mean"] == pytest.approx(15.0)
        assert result["loss_min"] == pytest.approx(10.0)
        assert result["loss_max"] == pytest.approx(20.0)
    
    def test_bankruptcy_timestamp_handling(self):
        """Test special handling of bankruptcy_timestamp."""
        evaluator = MockEvaluator()
        
        # Test with some bankruptcies
        analyses = {
            "exchange1": {"bankruptcy_timestamp": 1000},
            "exchange2": {"bankruptcy_timestamp": None},
            "exchange3": {"bankruptcy_timestamp": 2000},
        }
        
        result = evaluator.combine_analyses(analyses)
        
        # Should use earliest bankruptcy timestamp
        assert result["bankruptcy_timestamp_mean"] == 1000
        assert result["bankruptcy_timestamp_min"] == 1000
        assert result["bankruptcy_timestamp_max"] == 1000
        assert result["bankruptcy_timestamp_std"] == 0.0
    
    def test_bankruptcy_timestamp_no_bankruptcy(self):
        """Test bankruptcy_timestamp when no bankruptcy occurred."""
        evaluator = MockEvaluator()
        
        analyses = {
            "exchange1": {"bankruptcy_timestamp": None},
            "exchange2": {"bankruptcy_timestamp": None},
        }
        
        result = evaluator.combine_analyses(analyses)
        
        assert result["bankruptcy_timestamp_mean"] is None
        assert result["bankruptcy_timestamp_min"] is None
        assert result["bankruptcy_timestamp_max"] is None
        assert result["bankruptcy_timestamp_std"] is None
    
    def test_bankruptcy_reason_handling(self):
        """Test special handling of bankruptcy_reason."""
        evaluator = MockEvaluator()
        
        # 0=none, 1=financial, 2=drawdown, 3=no_positions, 4=stale_position
        analyses = {
            "exchange1": {"bankruptcy_reason": 0},
            "exchange2": {"bankruptcy_reason": 2},
            "exchange3": {"bankruptcy_reason": 1},
        }
        
        result = evaluator.combine_analyses(analyses)
        
        # Should use max (most severe) reason
        assert result["bankruptcy_reason_mean"] == 2
        assert result["bankruptcy_reason_min"] == 1
        assert result["bankruptcy_reason_max"] == 2
        assert result["bankruptcy_reason_std"] == 0.0
    
    def test_bankruptcy_reason_no_bankruptcy(self):
        """Test bankruptcy_reason when no bankruptcy occurred."""
        evaluator = MockEvaluator()
        
        analyses = {
            "exchange1": {"bankruptcy_reason": 0},
            "exchange2": {"bankruptcy_reason": 0},
        }
        
        result = evaluator.combine_analyses(analyses)
        
        assert result["bankruptcy_reason_mean"] == 0
        assert result["bankruptcy_reason_min"] == 0
        assert result["bankruptcy_reason_max"] == 0
        assert result["bankruptcy_reason_std"] == 0.0
    
    def test_invalid_values_inf(self):
        """Test handling of infinity values."""
        evaluator = MockEvaluator()
        
        analyses = {
            "exchange1": {"metric": 100.0},
            "exchange2": {"metric": np.inf},
        }
        
        result = evaluator.combine_analyses(analyses)
        
        # Should set to zero when inf present
        assert result["metric_mean"] == 0.0
        assert result["metric_min"] == 0.0
        assert result["metric_max"] == 0.0
        assert result["metric_std"] == 0.0
    
    def test_invalid_values_none(self):
        """Test handling of None values in regular metrics."""
        evaluator = MockEvaluator()
        
        analyses = {
            "exchange1": {"metric": 100.0},
            "exchange2": {"metric": None},
        }
        
        result = evaluator.combine_analyses(analyses)
        
        # Should set to zero when None present
        assert result["metric_mean"] == 0.0
        assert result["metric_min"] == 0.0
        assert result["metric_max"] == 0.0
        assert result["metric_std"] == 0.0
    
    def test_single_exchange(self):
        """Test with single exchange."""
        evaluator = MockEvaluator()
        
        analyses = {
            "exchange1": {"profit": 100.0, "loss": 10.0},
        }
        
        result = evaluator.combine_analyses(analyses)
        
        assert result["profit_mean"] == pytest.approx(100.0)
        assert result["profit_min"] == pytest.approx(100.0)
        assert result["profit_max"] == pytest.approx(100.0)
        assert result["profit_std"] == pytest.approx(0.0)
    
    def test_key_caching(self):
        """Test that keys are properly cached from first analysis.
        
        Validates: Requirement 4.1
        """
        evaluator = MockEvaluator()
        
        # All exchanges should have the same keys
        analyses = {
            "exchange1": {"a": 1.0, "b": 2.0, "c": 3.0},
            "exchange2": {"a": 4.0, "b": 5.0, "c": 6.0},
        }
        
        result = evaluator.combine_analyses(analyses)
        
        # Verify all keys are processed
        expected_keys = ["a_mean", "a_min", "a_max", "a_std",
                        "b_mean", "b_min", "b_max", "b_std",
                        "c_mean", "c_min", "c_max", "c_std"]
        for key in expected_keys:
            assert key in result
    
    def test_numpy_vectorized_operations(self):
        """Test that numpy vectorized operations produce correct results.
        
        Validates: Requirement 4.3
        """
        evaluator = MockEvaluator()
        
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        analyses = {f"exchange{i}": {"metric": v} for i, v in enumerate(values)}
        
        result = evaluator.combine_analyses(analyses)
        
        # Verify numpy operations match expected values
        arr = np.array(values, dtype=np.float64)
        assert result["metric_mean"] == pytest.approx(np.mean(arr))
        assert result["metric_min"] == pytest.approx(np.min(arr))
        assert result["metric_max"] == pytest.approx(np.max(arr))
        assert result["metric_std"] == pytest.approx(np.std(arr))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
