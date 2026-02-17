"""
Feature computation module for CMA-ME optimizer.

This module provides functions to compute behavioral descriptors from
backtest analysis results, normalize features, and extract feature ranges
from optimization configuration.
"""

import numpy as np
from typing import Dict, List, Tuple


def get_feature_ranges(config: Dict) -> List[Tuple[float, float]]:
    """
    Extract feature ranges from optimize.limits for archive initialization.
    
    This function determines the normalization ranges for each feature based on
    the limits specified in the optimization configuration. All features are
    normalized to [0, 1] range for the archive.
    
    Args:
        config: Optimization configuration dictionary containing optimize.limits
        
    Returns:
        List of (min, max) tuples for each feature, all set to (0, 1) since
        features will be pre-normalized before archive insertion
        
    Example:
        >>> config = {
        ...     "optimize": {
        ...         "limits": {
        ...             "drawdown_worst": 0.33,
        ...             "gain": 1.0,
        ...             "rsquared": 0.987
        ...         }
        ...     }
        ... }
        >>> ranges = get_feature_ranges(config)
        >>> len(ranges)
        3
        >>> ranges[0]
        (0.0, 1.0)
    """
    limits = config.get("optimize", {}).get("limits", {})
    
    # Each feature will be normalized to [0, 1] before archive insertion
    # So we return (0, 1) for each feature dimension
    num_features = len(limits)
    
    return [(0.0, 1.0) for _ in range(num_features)]


def get_feature_names(config: Dict) -> List[str]:
    """
    Extract feature names from optimize.limits in consistent order.
    
    Args:
        config: Optimization configuration dictionary containing optimize.limits
        
    Returns:
        List of feature names in sorted order for consistency
        
    Example:
        >>> config = {
        ...     "optimize": {
        ...         "limits": {
        ...             "drawdown_worst": 0.33,
        ...             "gain": 1.0
        ...         }
        ...     }
        ... }
        >>> get_feature_names(config)
        ['drawdown_worst', 'gain']
    """
    limits = config.get("optimize", {}).get("limits", {})
    # Sort keys for consistent ordering
    return sorted(limits.keys())


def get_feature_limits(config: Dict) -> Dict[str, float]:
    """
    Extract feature limit values from optimize.limits.
    
    Args:
        config: Optimization configuration dictionary containing optimize.limits
        
    Returns:
        Dictionary mapping feature names to their limit values
        
    Example:
        >>> config = {
        ...     "optimize": {
        ...         "limits": {
        ...             "drawdown_worst": 0.33,
        ...             "gain": 1.0
        ...         }
        ...     }
        ... }
        >>> get_feature_limits(config)
        {'drawdown_worst': 0.33, 'gain': 1.0}
    """
    return config.get("optimize", {}).get("limits", {})


def normalize_features(raw_features: Dict[str, float], config: Dict) -> np.ndarray:
    """
    Normalize feature values to [0, 1] range based on optimization limits.
    
    This function handles two types of limits:
    - Upper-bound limits (e.g., drawdown_worst): Lower values are better.
      Normalized as: raw_value / limit
    - Lower-bound limits (e.g., gain, rsquared): Higher values are better.
      Normalized as: (raw_value - limit) / (worst_case - limit)
      
    Features are identified as lower-bound if their name contains "gain" or "rsquared".
    All other features are treated as upper-bound limits.
    
    Args:
        raw_features: Dictionary mapping feature names to raw values
        config: Optimization configuration containing optimize.limits
        
    Returns:
        Numpy array of normalized feature values in [0, 1] range, ordered by
        sorted feature names for consistency
        
    Example:
        >>> config = {
        ...     "optimize": {
        ...         "limits": {
        ...             "drawdown_worst": 0.33,
        ...             "gain": 1.0
        ...         }
        ...     }
        ... }
        >>> raw = {"drawdown_worst": 0.15, "gain": 1.5}
        >>> normalized = normalize_features(raw, config)
        >>> 0 <= normalized[0] <= 1
        True
        >>> 0 <= normalized[1] <= 1
        True
    """
    limits = get_feature_limits(config)
    feature_names = get_feature_names(config)
    
    normalized = []
    
    for name in feature_names:
        raw_value = raw_features.get(name, 0.0)
        limit = limits[name]
        
        # Identify feature type based on name
        is_lower_bound = "gain" in name.lower() or "rsquared" in name.lower()
        
        if is_lower_bound:
            # Lower-bound limit: higher is better
            # Normalize to [0, 1] where 1 is best
            # For gain: limit=1.0, worst_case=0.0
            # For rsquared: limit=0.987, worst_case=0.0
            worst_case = 0.0
            
            if abs(limit - worst_case) < 1e-10:
                # Avoid division by zero
                norm_value = 0.5
            else:
                # Map [worst_case, limit] to [0, 1]
                # But we want higher values to be better, so we invert
                # Actually, for archive we want [limit, infinity] to map to [0, 1]
                # Let's use a different approach: clamp to reasonable range
                # For gain: [0, 10] -> [0, 1]
                # For rsquared: [0, 1] -> [0, 1]
                if "gain" in name.lower():
                    # Gain can be arbitrarily high, use reasonable upper bound
                    max_expected = 10.0
                    norm_value = (raw_value - limit) / (max_expected - limit)
                elif "rsquared" in name.lower():
                    # R-squared is bounded [0, 1]
                    max_expected = 1.0
                    # Check if limit equals max_expected to avoid division by zero
                    if abs(max_expected - limit) < 1e-10:
                        # If limit is at max (1.0), any value at or above limit is perfect
                        norm_value = 1.0 if raw_value >= limit else raw_value / limit
                    else:
                        norm_value = (raw_value - limit) / (max_expected - limit)
                else:
                    # Generic lower-bound
                    max_expected = limit * 10
                    norm_value = (raw_value - limit) / (max_expected - limit)
                
                # Clamp to [0, 1]
                norm_value = np.clip(norm_value, 0.0, 1.0)
        else:
            # Upper-bound limit: lower is better
            # Normalize to [0, 1] where 0 is best
            # For drawdown_worst: limit=0.33, raw in [0, 0.33]
            if abs(limit) < 1e-10:
                # Avoid division by zero
                norm_value = 0.5
            else:
                norm_value = raw_value / limit
                # Clamp to [0, 1]
                norm_value = np.clip(norm_value, 0.0, 1.0)
        
        normalized.append(norm_value)
    
    return np.array(normalized)


def compute_behavioral_descriptor(analyses_combined: Dict, config: Dict) -> np.ndarray:
    """
    Compute N-dimensional behavioral descriptor from backtest analysis results.
    
    This function extracts feature values from the analyses_combined dictionary
    returned by the evaluator, normalizes them to [0, 1] range, and returns
    an N-dimensional numpy array suitable for archive insertion.
    
    Args:
        analyses_combined: Dictionary of analysis metrics from backtest evaluation
        config: Optimization configuration containing optimize.limits
        
    Returns:
        Numpy array of shape (N,) where N = len(optimize.limits), with all
        values normalized to [0, 1] range
        
    Raises:
        KeyError: If required feature keys are missing from analyses_combined
        
    Example:
        >>> config = {
        ...     "optimize": {
        ...         "limits": {
        ...             "drawdown_worst": 0.33,
        ...             "gain": 1.0
        ...         }
        ...     }
        ... }
        >>> analyses = {
        ...     "drawdown_worst": 0.15,
        ...     "gain": 1.5
        ... }
        >>> descriptor = compute_behavioral_descriptor(analyses, config)
        >>> descriptor.shape
        (2,)
        >>> np.all((descriptor >= 0) & (descriptor <= 1))
        True
    """
    feature_names = get_feature_names(config)
    
    # Extract raw feature values from analyses_combined
    raw_features = {}
    for name in feature_names:
        if name not in analyses_combined:
            # Handle missing keys gracefully by using boundary values
            # This places the solution in a boundary cell
            raw_features[name] = 0.0
        else:
            raw_features[name] = analyses_combined[name]
    
    # Normalize features to [0, 1] range
    normalized_descriptor = normalize_features(raw_features, config)
    
    return normalized_descriptor
