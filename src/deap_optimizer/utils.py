"""
Utility functions for DEAP optimizer.

This module contains utility functions for individual-config conversion,
seeding, and Rich logging setup.
"""

import os
import json
import logging
import datetime
import contextlib

# Rich imports
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        TaskProgressColumn,
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def individual_to_config(individual, optimizer_overrides, overrides_list, template=None):
    """
    Convert DEAP individual to configuration dictionary.
    
    This function converts a DEAP individual (list of parameter values) into
    a configuration dictionary following the passivbot configuration format.
    It handles both long and short position sides and applies optimizer overrides.
    
    Parameters
    ----------
    individual : list
        DEAP individual (list of parameter values)
    optimizer_overrides : callable
        Function to apply optimizer-specific overrides to config
    overrides_list : list
        List of parameter names in order
    template : dict, optional
        Template configuration dictionary. If None, uses default v7 template.
    
    Returns
    -------
    dict
        Configuration dictionary with bot parameters set from individual
    
    Requirements: 6.2, 7.4
    """
    from copy import deepcopy
    from pure_funcs import get_template_live_config
    
    if template is None:
        template = get_template_live_config("v7")
    
    keys_ignored = ["enforce_exposure_limit"]
    config = deepcopy(template)
    keys = [k for k in sorted(config["bot"]["long"]) if k not in keys_ignored]
    
    # Map individual values to config parameters
    i = 0
    for pside in ["long", "short"]:
        for key in keys:
            config["bot"][pside][key] = individual[i]
            i += 1
        
        # Check if position side is enabled
        is_enabled = (
            config["bot"][pside]["total_wallet_exposure_limit"] > 0.0
            and config["bot"][pside]["n_positions"] > 0.0
        )
        
        # If disabled, set parameters to minimum valid values
        if not is_enabled:
            for key in config["bot"][pside]:
                if key in keys_ignored:
                    continue
                bounds = config["optimize"]["bounds"][f"{pside}_{key}"]
                if len(bounds) == 1:
                    bounds = [bounds[0], bounds[0]]
                config["bot"][pside][key] = min(max(bounds[0], 0.0), bounds[1])
        
        # Apply optimizer overrides
        config = optimizer_overrides(overrides_list, config, pside)
    
    return config


def config_to_individual(config, param_bounds):
    """
    Convert configuration dictionary to DEAP individual.
    
    This function extracts parameter values from a configuration dictionary
    and creates a DEAP individual (list of values). It handles both long and
    short position sides and adjusts values to stay within bounds.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with bot parameters
    param_bounds : dict
        Dictionary of parameter bounds {param_name: (low, high)}
    
    Returns
    -------
    list
        DEAP individual (list of parameter values)
    
    Requirements: 6.2, 7.4
    """
    individual = []
    keys_ignored = ["enforce_exposure_limit"]
    
    # Extract values for each position side
    for pside in ["long", "short"]:
        # Check if position side is enabled
        is_enabled = (
            param_bounds[f"{pside}_n_positions"][1] > 0.0
            and param_bounds[f"{pside}_total_wallet_exposure_limit"][1] > 0.0
        )
        
        # Add parameter values (0.0 if disabled)
        individual += [
            (v if is_enabled else 0.0)
            for k, v in sorted(config["bot"][pside].items())
            if k not in keys_ignored
        ]
    
    # Adjust values to stay within bounds
    bounds = [(low, high) for low, high in param_bounds.values()]
    adjusted = [
        max(min(x, bounds[z][1]), bounds[z][0]) 
        for z, x in enumerate(individual)
    ]
    
    return adjusted


def create_seeded_individual_from_config(toolbox, config_path="configs/optimize.json"):
    """
    Create a seeded individual from configuration file.
    
    This function loads parameter values from a configuration file and creates
    a DEAP individual with those values. This is useful for seeding the initial
    population with known good configurations.
    
    Parameters
    ----------
    toolbox : deap.base.Toolbox
        DEAP toolbox with individual creator registered
    config_path : str
        Path to configuration file (default: "configs/optimize.json")
    
    Returns
    -------
    Individual
        Seeded individual with values from config file
    
    Requirements: 6.2, 7.4
    """
    import json
    from procedures import format_config
    
    try:
        # Load configuration file
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Format config to ensure it's in the correct format
        formatted_config = format_config(config, verbose=False)
        
        # Get param_bounds from toolbox if available
        # Otherwise, extract from config
        if hasattr(toolbox, 'param_bounds'):
            param_bounds = toolbox.param_bounds
        else:
            # Extract bounds from config
            param_bounds = {}
            if "optimize" in config and "bounds" in config["optimize"]:
                for param_name, bounds in config["optimize"]["bounds"].items():
                    if isinstance(bounds, list):
                        if len(bounds) == 1:
                            param_bounds[param_name] = (bounds[0], bounds[0])
                        else:
                            param_bounds[param_name] = tuple(bounds)
                    else:
                        param_bounds[param_name] = (bounds, bounds)
        
        # Convert config to individual
        individual_values = config_to_individual(formatted_config, param_bounds)
        
        # Create individual using toolbox
        individual = toolbox.individual()
        individual[:] = individual_values
        
        return individual
        
    except Exception as e:
        # If loading fails, return a random individual
        logging.warning(f"Could not create seeded individual from {config_path}: {e}")
        return toolbox.individual()


def create_full_individual(solution, optimizable_param_names, all_param_names, 
                          fixed_params, integer_params, toolbox):
    """
    Create a full individual with parameter mapping.
    
    This function creates a complete DEAP individual by combining optimizable
    parameters from a solution vector with fixed parameters. It handles the
    mapping between the reduced solution space (only optimizable params) and
    the full parameter space (all params).
    
    Parameters
    ----------
    solution : array-like
        Solution vector containing only optimizable parameter values
    optimizable_param_names : list
        Names of optimizable parameters (in order)
    all_param_names : list
        Names of all parameters (in order)
    fixed_params : dict
        Dictionary of fixed parameter values {param_name: value}
    integer_params : list
        Names of parameters that should be rounded to integers
    toolbox : deap.base.Toolbox
        DEAP toolbox with individual creator registered
    
    Returns
    -------
    Individual
        Full individual with all parameters (optimizable + fixed)
    
    Requirements: 6.2, 7.4
    """
    # Create empty individual
    individual = toolbox.individual()
    
    # Build full parameter vector
    full_solution = []
    opt_idx = 0
    
    for param_name in all_param_names:
        if param_name in fixed_params:
            # Use fixed parameter value
            full_solution.append(fixed_params[param_name])
        elif param_name in optimizable_param_names:
            # Use optimizable parameter value from solution
            value = solution[opt_idx]
            
            # Round to integer if needed
            if param_name in integer_params:
                value = round(value)
            
            full_solution.append(value)
            opt_idx += 1
        else:
            # This shouldn't happen if parameter setup is correct
            raise ValueError(
                f"Parameter '{param_name}' is neither fixed nor optimizable"
            )
    
    # Set individual values
    individual[:] = full_solution
    
    return individual


def setup_rich_logging(console=True, file_logging=True, log_path=None):
    """
    Set up Rich logging with console and file output.
    
    This function initializes a Rich Console object with proper configuration
    for colored output, emojis, and panels. It follows the PSO pattern for
    consistent logging across optimizers.
    
    Parameters
    ----------
    console : bool
        Enable console output
    file_logging : bool
        Enable file logging
    log_path : str, optional
        Path to log file (default: "logs/deap_evaluation.log")
    
    Returns
    -------
    Console or None
        Rich Console object if available, None otherwise
    
    Requirements: 2.1, 2.2, 2.3, 7.1
    """
    if not RICH_AVAILABLE:
        return None
    
    # Use default log path if not provided
    if log_path is None:
        log_path = "logs/deap_evaluation.log"
    
    # Create console with PSO-style configuration
    console_obj = Console(
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
    
    return console_obj


def log_message(console, message, emoji=None, panel=False, timestamp=True, 
                log_path=None, title="DEAP Stats", border_style="cyan"):
    """
    Log a message with Rich formatting including emojis, panels, and colors.
    
    This function provides consistent logging output following the PSO pattern.
    It supports both console and file output with Rich formatting.
    
    Parameters
    ----------
    console : Console
        Rich Console object
    message : str
        Message to log
    emoji : str, optional
        Emoji to prepend to message
    panel : bool
        Whether to wrap message in a panel
    timestamp : bool
        Whether to include timestamp
    log_path : str, optional
        Path to log file for file output
    title : str
        Title for panel (if panel=True)
    border_style : str
        Border style for panel (if panel=True)
    
    Requirements: 2.1, 2.2, 2.4, 7.1
    """
    if not RICH_AVAILABLE or console is None:
        # Fallback to standard print
        print(f"{emoji} {message}" if emoji else message)
        return
    
    # Build log text with timestamp and emoji
    if timestamp:
        timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        timestamp_str = ""
    
    emoji_str = f" {emoji}" if emoji else ""
    log_text = f"{timestamp_str}{emoji_str} {message}"
    
    # Create panel or plain text
    if panel:
        panel_message = Panel(log_text, title=title, border_style=border_style)
        output = panel_message
    else:
        output = log_text
    
    # Output to console and file
    if log_path:
        with open(log_path, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            console.print(output)
    else:
        console.print(output)


def log_error(console, error_message, exception=None, log_path=None):
    """
    Log an error message with detailed information and timestamps.
    
    This function provides consistent error logging with Rich formatting,
    including exception details when available.
    
    Parameters
    ----------
    console : Console
        Rich Console object
    error_message : str
        Error message to log
    exception : Exception, optional
        Exception object for detailed error information
    log_path : str, optional
        Path to log file for file output
    
    Requirements: 2.3, 2.5, 7.1
    """
    if not RICH_AVAILABLE or console is None:
        # Fallback to standard print
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] ❌ ERROR: {error_message}")
        if exception:
            print(f"  Exception: {type(exception).__name__}: {str(exception)}")
        return
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Build error text with details
    error_text = Text()
    error_text.append(f"[{timestamp}] ", style="dim")
    error_text.append("❌ ERROR: ", style="bold red")
    error_text.append(error_message, style="red")
    
    if exception:
        error_text.append("\n  Exception: ", style="dim")
        error_text.append(f"{type(exception).__name__}: {str(exception)}", style="yellow")
    
    # Create error panel
    error_panel = Panel(
        error_text,
        title="Error",
        border_style="red",
        expand=False
    )
    
    # Output to console and file
    if log_path:
        with open(log_path, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            console.print(error_panel)
    else:
        console.print(error_panel)


def create_progress_display(generation, population_size, best_fitness=None, 
                            avg_fitness=None, elapsed_time=None):
    """
    Create Rich progress display for generation with emojis and colors.
    
    This function creates a formatted progress display showing generation
    information, fitness statistics, and timing information following the
    PSO pattern.
    
    Parameters
    ----------
    generation : int
        Current generation number
    population_size : int
        Size of population
    best_fitness : float, optional
        Best fitness value in current generation
    avg_fitness : float, optional
        Average fitness value in current generation
    elapsed_time : float, optional
        Elapsed time in seconds
    
    Returns
    -------
    str
        Formatted progress display string with emojis and colors
    
    Requirements: 2.2, 2.4, 7.1
    """
    if not RICH_AVAILABLE:
        # Fallback to plain text
        parts = [f"Generation {generation}"]
        if best_fitness is not None:
            parts.append(f"Best: {best_fitness:.6f}")
        if avg_fitness is not None:
            parts.append(f"Avg: {avg_fitness:.6f}")
        if elapsed_time is not None:
            parts.append(f"Time: {elapsed_time:.2f}s")
        return " | ".join(parts)
    
    # Build Rich text with colors and emojis
    text = Text()
    text.append("🧬 Generation ", style="bold cyan")
    text.append(f"{generation}", style="bold yellow")
    text.append(f" | 👥 Population: ", style="cyan")
    text.append(f"{population_size}", style="yellow")
    
    if best_fitness is not None:
        text.append(" | 🏆 Best: ", style="green")
        text.append(f"{best_fitness:.6f}", style="bold green")
    
    if avg_fitness is not None:
        text.append(" | 📊 Avg: ", style="blue")
        text.append(f"{avg_fitness:.6f}", style="bold blue")
    
    if elapsed_time is not None:
        text.append(" | ⏱️  Time: ", style="magenta")
        text.append(f"{elapsed_time:.2f}s", style="bold magenta")
    
    return text


def create_generation_rule(generation, style="bold blue"):
    """
    Create a Rich rule separator for generation display.
    
    This function creates a horizontal rule separator following the PSO pattern
    for visual separation between generations.
    
    Parameters
    ----------
    generation : int
        Current generation number
    style : str
        Style for the rule (default: "bold blue")
    
    Returns
    -------
    Rule
        Rich Rule object
    
    Requirements: 2.2, 2.4, 7.1
    """
    if not RICH_AVAILABLE:
        return f"{'='*50} Generation {generation} {'='*50}"
    
    return Rule(f"Generation {generation}", style=style)


def create_optimization_summary_panel(total_generations, best_fitness, 
                                     best_individual, total_time,
                                     title="Optimization Complete"):
    """
    Create a Rich panel summarizing optimization results.
    
    This function creates a formatted summary panel with optimization results
    including best fitness, total time, and other statistics following the
    PSO pattern.
    
    Parameters
    ----------
    total_generations : int
        Total number of generations completed
    best_fitness : float
        Best fitness value achieved
    best_individual : list
        Best individual found
    total_time : float
        Total optimization time in seconds
    title : str
        Title for the panel
    
    Returns
    -------
    Panel
        Rich Panel object with summary
    
    Requirements: 2.2, 2.4, 7.1
    """
    if not RICH_AVAILABLE:
        # Fallback to plain text
        summary = f"\n{title}\n"
        summary += f"Generations: {total_generations}\n"
        summary += f"Best Fitness: {best_fitness:.6f}\n"
        summary += f"Total Time: {total_time:.2f}s\n"
        return summary
    
    # Build Rich text summary
    summary = Text()
    summary.append("🎯 ", style="bold green")
    summary.append("Optimization Results\n\n", style="bold green")
    
    summary.append("📈 Generations: ", style="cyan")
    summary.append(f"{total_generations}\n", style="bold yellow")
    
    summary.append("🏆 Best Fitness: ", style="green")
    summary.append(f"{best_fitness:.6f}\n", style="bold green")
    
    summary.append("⏱️  Total Time: ", style="magenta")
    summary.append(f"{total_time:.2f}s\n", style="bold magenta")
    
    if best_individual is not None:
        summary.append("🧬 Best Individual: ", style="blue")
        summary.append(f"{len(best_individual)} parameters\n", style="bold blue")
    
    # Create panel
    panel = Panel(
        summary,
        title=title,
        border_style="green",
        expand=False
    )
    
    return panel


def create_progress_bar(console, total, description="Evaluating"):
    """
    Create a Rich progress bar for tracking operations.
    
    This function creates a progress bar following the PSO pattern for
    tracking long-running operations like fitness evaluation.
    
    Parameters
    ----------
    console : Console
        Rich Console object
    total : int
        Total number of items to process
    description : str
        Description for the progress bar
    
    Returns
    -------
    Progress
        Rich Progress object
    
    Requirements: 2.2, 2.4, 7.1
    """
    if not RICH_AVAILABLE or console is None:
        return None
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    )
    
    return progress
