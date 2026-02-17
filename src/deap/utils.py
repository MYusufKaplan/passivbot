"""
Utility functions for DEAP evolutionary algorithm.

This module provides utility functions for individual creation, conversion,
seeding, and Rich logging setup following the PSO implementation pattern.
"""

import json
import os
from copy import deepcopy
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Rich imports for logging
try:
    from rich.console import Console
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# DEAP imports
try:
    from deap import base, creator
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False


def individual_to_config(
    individual: List[float], 
    optimizer_overrides: Dict[str, Any], 
    overrides_list: List[str], 
    template: Optional[Dict] = None
) -> Dict:
    """
    Convert DEAP individual to configuration dictionary.
    
    Parameters
    ----------
    individual : list
        DEAP individual (list of parameter values)
    optimizer_overrides : dict
        Mapping of parameter names to values
    overrides_list : list
        List of parameter names in order
    template : dict, optional
        Configuration template to use as base
    
    Returns
    -------
    dict
        Configuration dictionary
    """
    # This function would need to be implemented based on the specific
    # configuration format used in the system. For now, return a placeholder.
    # The actual implementation should be extracted from optimize.py
    
    if template is None:
        # Use a basic template structure
        template = {
            "bot": {
                "long": {},
                "short": {}
            }
        }
    
    config = deepcopy(template)
    
    # Map individual values to configuration parameters
    for i, param_name in enumerate(overrides_list):
        if i < len(individual):
            # This is a simplified mapping - the actual implementation
            # would need to handle the complex nested structure
            if param_name.startswith("long_"):
                clean_name = param_name.replace("long_", "")
                config["bot"]["long"][clean_name] = individual[i]
            elif param_name.startswith("short_"):
                clean_name = param_name.replace("short_", "")
                config["bot"]["short"][clean_name] = individual[i]
    
    return config


def config_to_individual(
    config: Dict, 
    param_bounds: Dict[str, Tuple[float, float]]
) -> List[float]:
    """
    Convert configuration dictionary to DEAP individual.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    param_bounds : dict
        Parameter bounds mapping
    
    Returns
    -------
    list
        Individual values in order
    """
    individual = []
    
    for param_name in param_bounds.keys():
        if param_name.startswith("long_"):
            clean_name = param_name.replace("long_", "")
            value = config.get("bot", {}).get("long", {}).get(clean_name, 0.0)
        elif param_name.startswith("short_"):
            clean_name = param_name.replace("short_", "")
            value = config.get("bot", {}).get("short", {}).get(clean_name, 0.0)
        else:
            value = 0.0  # Default value
        
        individual.append(float(value))
    
    return individual


def create_seeded_individual_from_config(
    toolbox: base.Toolbox, 
    config_path: str = "configs/optimize.json"
) -> Optional[List[float]]:
    """
    Create a seeded individual from configuration file.
    
    Parameters
    ----------
    toolbox : base.Toolbox
        DEAP toolbox for individual creation
    config_path : str
        Path to configuration file
    
    Returns
    -------
    list or None
        Seeded individual or None if config cannot be loaded
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create individual from toolbox
        individual = toolbox.individual()
        
        # This is a simplified implementation - the actual version would need
        # to properly map configuration values to individual parameters
        long_config = config.get("bot", {}).get("long", {})
        
        # For now, just use the first few values from the config
        # The actual implementation would need proper parameter mapping
        config_values = list(long_config.values())
        for i in range(min(len(individual), len(config_values))):
            if isinstance(config_values[i], (int, float)) and not isinstance(config_values[i], bool):
                individual[i] = float(config_values[i])
        
        return individual
        
    except Exception:
        return None


def create_full_individual(
    solution: List[float],
    optimizable_param_names: List[str],
    all_param_names: List[str],
    fixed_params: Dict[str, float],
    integer_params: set,
    toolbox: base.Toolbox
) -> List[float]:
    """
    Create a full individual from optimization solution vector.
    
    Parameters
    ----------
    solution : list
        Optimization solution vector
    optimizable_param_names : list
        Names of optimizable parameters
    all_param_names : list
        Names of all parameters
    fixed_params : dict
        Fixed parameter values
    integer_params : set
        Set of parameter names that should be integers
    toolbox : base.Toolbox
        DEAP toolbox for individual creation
    
    Returns
    -------
    list
        Full individual with all parameters
    """
    individual = toolbox.individual()
    individual.clear()  # Clear any existing values
    
    # Create full parameter vector
    opt_idx = 0
    
    for param_name in all_param_names:
        if param_name in fixed_params:
            # Fixed parameter
            individual.append(fixed_params[param_name])
        elif param_name in optimizable_param_names:
            # Optimizable parameter
            value = solution[opt_idx]
            if param_name in integer_params:
                value = round(value)
            individual.append(value)
            opt_idx += 1
        else:
            # This shouldn't happen if parameter setup is correct
            raise ValueError(f"Parameter '{param_name}' is neither fixed nor optimizable")
    
    return individual


def setup_rich_logging(
    console: bool = True,
    file_logging: bool = True,
    log_path: Optional[str] = None
) -> Optional[Console]:
    """
    Setup Rich logging for beautiful console output.
    
    Parameters
    ----------
    console : bool
        Enable console logging
    file_logging : bool
        Enable file logging
    log_path : str, optional
        Path for log files
    
    Returns
    -------
    Console or None
        Rich console instance if available, None otherwise
    """
    if not RICH_AVAILABLE:
        return None
    
    # Create console with configuration similar to PSO
    rich_console = Console(
        force_terminal=True,
        no_color=False,
        log_path=False,
        width=191,
        color_system="truecolor",
        legacy_windows=False
    )
    
    if file_logging and log_path:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    return rich_console


def create_progress_display(generation: int, population_size: int) -> str:
    """
    Create a progress display string for the current generation.
    
    Parameters
    ----------
    generation : int
        Current generation number
    population_size : int
        Size of the population
    
    Returns
    -------
    str
        Formatted progress display string
    """
    return f"🧬 Generation {generation} | Population: {population_size}"