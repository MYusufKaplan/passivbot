"""
Main CMA-ME optimizer implementation.

This module contains the core CMA-ME algorithm that coordinates archive
management, emitter operations, and the main optimization loop.
"""

import json
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from rich.console import Console
from rich.rule import Rule

try:
    from ribs.archives import GridArchive
    from ribs.emitters import EvolutionStrategyEmitter
    from ribs.schedulers import Scheduler
    PYRIBS_AVAILABLE = True
except ImportError:
    PYRIBS_AVAILABLE = False
    print("⚠️ pyribs not installed. Install with: pip install ribs[all]")

from .features import get_feature_ranges, compute_behavioral_descriptor, get_feature_names
from .utils import (
    log_archive_stats,
    create_progress_bar,
    save_checkpoint,
    load_checkpoint,
)

# Initialize console for logging - will be reconfigured in cma_me() to write to file
console = Console(
    force_terminal=True,
    no_color=False,
    log_path=False,
    width=191,
    color_system="truecolor",
    legacy_windows=False,
)


def _process_parameters(parameter_bounds, config):
    """
    Filter optimizable vs fixed parameters and identify integer parameters.
    
    Args:
        parameter_bounds: Dictionary of parameter bounds
        config: Optimization configuration from optimize.json
        
    Returns:
        Tuple of (optimizable_bounds, fixed_params, integer_params, 
                  all_param_names, optimizable_param_names, initial_values)
    """
    optimizable_bounds = {}
    fixed_params = {}
    integer_params = set()
    
    for param_name, bounds in parameter_bounds.items():
        min_val, max_val = bounds
        if "short" not in param_name and min_val != max_val:
            # Parameter has range to optimize
            optimizable_bounds[param_name] = bounds
            # Check if parameter should be integer
            if any(keyword in param_name.lower() for keyword in ["n_positions"]):
                integer_params.add(param_name)
        else:
            # Fixed parameter
            fixed_params[param_name] = min_val
    
    all_param_names = list(parameter_bounds.keys())
    optimizable_param_names = list(optimizable_bounds.keys())
    
    # Load initial values from config
    initial_values = {}
    long_config = config.get("bot", {}).get("long", {})
    for param_name in optimizable_param_names:
        clean_name = param_name.replace("long_", "")
        if clean_name in long_config:
            value = long_config[clean_name]
            # Only use numeric values (ignore booleans)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                initial_values[param_name] = float(value)
    
    return (
        optimizable_bounds,
        fixed_params,
        integer_params,
        all_param_names,
        optimizable_param_names,
        initial_values,
    )


def _create_initial_solution(
    optimizable_param_names, optimizable_bounds, initial_values, integer_params
):
    """
    Create initial solution vector from config values or middle values.
    
    Args:
        optimizable_param_names: List of optimizable parameter names
        optimizable_bounds: Dictionary of parameter bounds
        initial_values: Dictionary of initial values from config
        integer_params: Set of integer parameter names
        
    Returns:
        Numpy array of initial solution values
    """
    initial_vector = []
    
    for param_name in optimizable_param_names:
        min_val, max_val = optimizable_bounds[param_name]
        
        # Use initial value from config if available, otherwise use middle value
        if param_name in initial_values:
            initial_val = initial_values[param_name]
            # IMPORTANT: Clamp initial value to bounds to avoid resampling issues
            initial_val = max(min_val, min(max_val, initial_val))
            if param_name in integer_params:
                initial_val = round(initial_val)
        else:
            # Fallback to middle value
            initial_val = (min_val + max_val) / 2
            if param_name in integer_params:
                initial_val = round(initial_val)
        
        initial_vector.append(initial_val)
    
    return np.array(initial_vector)


def _initialize_archive(config, num_parameters, archive_bins_per_dim):
    """
    Create GridArchive with N dimensions from optimize.limits.
    
    Args:
        config: Optimization configuration
        num_parameters: Number of optimizable parameters
        archive_bins_per_dim: Number of bins per dimension
        
    Returns:
        GridArchive instance
    """
    # Get feature ranges (all normalized to [0, 1])
    feature_ranges = get_feature_ranges(config)
    num_features = len(feature_ranges)
    
    # Create archive with N dimensions
    archive = GridArchive(
        solution_dim=num_parameters,
        dims=[archive_bins_per_dim] * num_features,
        ranges=feature_ranges,
    )
    
    return archive


def _normalize_solution(solution, lower_bounds, upper_bounds):
    """
    Normalize solution from original bounds to [0, 1].
    
    Args:
        solution: Solution in original space
        lower_bounds: Lower bounds array
        upper_bounds: Upper bounds array
        
    Returns:
        Normalized solution in [0, 1]
    """
    ranges = upper_bounds - lower_bounds
    # Avoid division by zero for fixed parameters
    ranges = np.where(ranges > 1e-10, ranges, 1.0)
    return (solution - lower_bounds) / ranges


def _denormalize_solution(normalized_solution, lower_bounds, upper_bounds):
    """
    Denormalize solution from [0, 1] to original bounds.
    
    Args:
        normalized_solution: Solution in [0, 1] space
        lower_bounds: Lower bounds array
        upper_bounds: Upper bounds array
        
    Returns:
        Solution in original space
    """
    ranges = upper_bounds - lower_bounds
    return lower_bounds + normalized_solution * ranges


def _initialize_emitters(
    archive, initial_solution, num_emitters, batch_size_per_emitter, sigma0, bounds
):
    """
    Create multiple EvolutionStrategyEmitters in normalized [0, 1] space.
    
    Args:
        archive: GridArchive instance
        initial_solution: Initial solution vector (already normalized)
        num_emitters: Number of emitters to create
        batch_size_per_emitter: Batch size for each emitter
        sigma0: Initial step size for CMA-ES (in normalized space)
        bounds: Should be None or (0, 1) for normalized space
        
    Returns:
        List of EvolutionStrategyEmitter instances
    """
    emitters = []
    
    # In normalized [0, 1] space, use a reasonable sigma0
    # Default 0.1 means exploring 10% of the normalized range
    # This is appropriate for all dimensions since they're normalized
    
    for i in range(num_emitters):
        emitter = EvolutionStrategyEmitter(
            archive=archive,
            x0=initial_solution,
            sigma0=sigma0,  # Use provided sigma0 in normalized space
            batch_size=batch_size_per_emitter,
            bounds=None,  # No bounds needed in normalized space, we'll clip manually
        )
        emitters.append(emitter)
    
    return emitters


def _evaluate_batch(
    solutions,
    evaluator,
    toolbox,
    optimizable_param_names,
    all_param_names,
    fixed_params,
    integer_params,
    gen,
    best_fitness,
    verbose,
):
    """
    Evaluate a batch of solutions in parallel.
    
    Args:
        solutions: List of solution vectors
        evaluator: Evaluation function
        toolbox: DEAP toolbox
        optimizable_param_names: List of optimizable parameter names
        all_param_names: List of all parameter names
        fixed_params: Dictionary of fixed parameters
        integer_params: Set of integer parameter names
        gen: Current generation number
        best_fitness: Current best fitness
        verbose: Whether to show progress bar
        
    Returns:
        Tuple of (fitness_values, analyses_list)
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from alternatives import create_full_individual, evaluate_solution
    
    pool = Pool(processes=(cpu_count() - 1))
    
    # Prepare evaluation arguments with indices
    all_args = []
    for idx, solution in enumerate(solutions):
        individual = create_full_individual(
            solution,
            optimizable_param_names,
            all_param_names,
            fixed_params,
            integer_params,
            toolbox,
        )
        all_args.append((evaluator, individual, False, idx))
    
    # Evaluate with progress bar
    fitness_results = [None] * len(solutions)
    analyses_results = [None] * len(solutions)
    
    if verbose:
        progress = create_progress_bar(
            total=len(solutions),
            description=f"CMA-ME Gen {gen} | Best: {best_fitness:.6e}",
        )
        
        with progress:
            task = progress.add_task(
                f"CMA-ME Gen {gen} | Best: {best_fitness:.6e}",
                total=len(solutions),
            )
            
            for result in pool.imap_unordered(evaluate_solution, all_args):
                fitness_val, solution_idx, bankrupt = result
                
                # Get analyses_combined from evaluator
                # Note: evaluate_solution returns (fitness, idx, bankrupt)
                # We need to get analyses_combined separately
                fitness_results[solution_idx] = fitness_val
                
                progress.update(task, advance=1)
    else:
        for result in pool.imap_unordered(evaluate_solution, all_args):
            fitness_val, solution_idx, bankrupt = result
            fitness_results[solution_idx] = fitness_val
    
    pool.close()
    pool.join()
    
    # Note: We need to get analyses_combined for behavioral descriptors
    # For now, we'll return None and compute them separately
    return fitness_results, None


def cma_me(
    population,
    toolbox,
    evaluator,
    ngen,
    verbose=True,
    parameter_bounds=None,
    checkpoint_path="cma_me_checkpoint.pkl",
    archive_bins_per_dim=10,
    num_emitters=5,
    batch_size_per_emitter=20,
    sigma0=0.1,  # Reduced from 0.5 to avoid bounds resampling issues
):
    """
    CMA-ME optimization implementation.
    
    Args:
        population: Initial population (used to determine total batch size)
        toolbox: DEAP toolbox with individual creation
        evaluator: Evaluation function
        ngen: Maximum number of generations
        verbose: Whether to show detailed logging
        parameter_bounds: Dictionary of parameter bounds
        checkpoint_path: Path to save/load checkpoints
        archive_bins_per_dim: Number of bins per archive dimension
        num_emitters: Number of CMA-ES emitters
        batch_size_per_emitter: Batch size for each emitter
        sigma0: Initial step size for CMA-ES
        
    Returns:
        final_population: List of best individuals
        logbook: Evolution statistics
    """
    if not PYRIBS_AVAILABLE:
        raise ImportError(
            "pyribs is required for CMA-ME. Install with: pip install ribs[all]"
        )
    
    if not parameter_bounds:
        raise ValueError("parameter_bounds is required for CMA-ME")
    
    start_time = time.time()
    
    # Create logs directory if it doesn't exist
    import os
    os.makedirs("logs", exist_ok=True)
    
    # Open log file for writing
    log_file = open("logs/evaluation.log", "a", buffering=1)  # Line buffered
    
    # Create console that writes to both stdout and log file
    file_console = Console(
        file=log_file,
        force_terminal=True,
        no_color=False,
        log_path=False,
        width=191,
        color_system="truecolor",
        legacy_windows=False,
    )
    
    # Load configuration
    try:
        with open("configs/optimize.json", "r") as f:
            config = json.load(f)
    except Exception as e:
        file_console.print(f"⚠️ Could not load optimize.json: {e}")
        config = {}
    
    # Task 4.1: Process parameters
    (
        optimizable_bounds,
        fixed_params,
        integer_params,
        all_param_names,
        optimizable_param_names,
        initial_values,
    ) = _process_parameters(parameter_bounds, config)
    
    num_parameters = len(optimizable_param_names)
    
    file_console.print(
        f"🧬 CMA-ME: {num_parameters} optimizable parameters, {len(fixed_params)} fixed"
    )
    file_console.print(f"📋 Loaded {len(initial_values)} initial values from optimize.json")
    
    # Create initial solution in original space
    initial_solution_original = _create_initial_solution(
        optimizable_param_names, optimizable_bounds, initial_values, integer_params
    )
    
    # Create bounds arrays
    lower_bounds = np.array([optimizable_bounds[p][0] for p in optimizable_param_names])
    upper_bounds = np.array([optimizable_bounds[p][1] for p in optimizable_param_names])
    
    # Normalize initial solution to [0, 1] space
    initial_solution_normalized = _normalize_solution(
        initial_solution_original, lower_bounds, upper_bounds
    )
    
    file_console.print(f"🔄 Using normalized [0, 1] parameter space for CMA-ES")
    
    # Task 4.3: Initialize archive
    archive = _initialize_archive(config, num_parameters, archive_bins_per_dim)
    
    # Task 4.4: Initialize emitters in normalized space
    emitters = _initialize_emitters(
        archive, initial_solution_normalized, num_emitters, batch_size_per_emitter, sigma0, None
    )
    
    # Task 4.6: Initialize scheduler
    scheduler = Scheduler(archive, emitters)
    
    # Initialize tracking variables
    best_fitness = float("inf")
    best_solution_normalized = initial_solution_normalized.copy()
    logbook = []
    start_gen = 1
    
    # Task 4.7: Load checkpoint if exists
    checkpoint_data = load_checkpoint(checkpoint_path)
    if checkpoint_data is not None:
        archive = checkpoint_data["archive"]
        scheduler = checkpoint_data["scheduler"]
        start_gen = checkpoint_data["generation"] + 1
        best_fitness = checkpoint_data["best_fitness"]
        logbook = checkpoint_data.get("logbook", [])
        
        file_console.print(
            f"✅ Resumed from generation {start_gen - 1}, best fitness: {best_fitness:.6e}"
        )
    else:
        file_console.print("🚀 Starting fresh CMA-ME optimization")
    
    file_console.print(f"🧬 Starting CMA-ME evolution from generation {start_gen}")
    file_console.print(f"📊 Archive dimensions: {len(get_feature_names(config))}")
    file_console.print(f"📊 Archive bins per dimension: {archive_bins_per_dim}")
    file_console.print(f"📊 Total archive cells: {archive.cells}")
    file_console.print(f"📊 Number of emitters: {num_emitters}")
    file_console.print(f"📊 Batch size per emitter: {batch_size_per_emitter}")
    file_console.print(f"📊 Total evaluations per generation: {num_emitters * batch_size_per_emitter}")
    
    # Task 4.8: Main optimization loop
    for gen in range(start_gen, ngen + 1):
        gen_start_time = time.time()
        
        if verbose:
            file_console.print(Rule(f"Generation {gen}", style="bold blue"))
        
        # Ask scheduler for solutions (in normalized space)
        solutions_normalized = scheduler.ask()
        
        # Denormalize solutions to original space and clip to bounds
        solutions = []
        for solution_norm in solutions_normalized:
            # Clip to [0, 1] in normalized space
            solution_norm_clipped = np.clip(solution_norm, 0.0, 1.0)
            # Denormalize to original space
            solution_original = _denormalize_solution(
                solution_norm_clipped, lower_bounds, upper_bounds
            )
            # Round integer parameters
            for j, param_name in enumerate(optimizable_param_names):
                if param_name in integer_params:
                    solution_original[j] = round(solution_original[j])
            solutions.append(solution_original)
        
        # Evaluate solutions in parallel
        # Note: We need to modify this to get analyses_combined for behavioral descriptors
        # For now, we'll use a simplified approach
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from alternatives import create_full_individual
        
        pool = Pool(processes=(cpu_count() - 1))
        
        # Prepare evaluation arguments
        all_args = []
        for idx, solution in enumerate(solutions):
            individual = create_full_individual(
                solution,
                optimizable_param_names,
                all_param_names,
                fixed_params,
                integer_params,
                toolbox,
            )
            all_args.append((evaluator, individual, False, idx))
        
        # Evaluate with progress bar
        fitness_results = [None] * len(solutions)
        
        if verbose:
            progress = create_progress_bar(
                total=len(solutions),
                description=f"CMA-ME Gen {gen}",
                console=file_console,  # Pass file console to progress bar
            )
            
            with progress:
                task = progress.add_task(
                    f"CMA-ME Gen {gen} | Best: {best_fitness:.6e}",
                    total=len(solutions),
                )
                
                from alternatives import evaluate_solution
                
                for result in pool.imap_unordered(evaluate_solution, all_args):
                    fitness_val, solution_idx, bankrupt = result
                    fitness_results[solution_idx] = fitness_val
                    
                    # Update best fitness
                    if fitness_val < best_fitness:
                        best_fitness = fitness_val
                        best_solution_normalized = solutions_normalized[solution_idx].copy()
                    
                    progress.update(
                        task,
                        advance=1,
                        description=f"CMA-ME Gen {gen} | Best: {best_fitness:.6e}",
                    )
        else:
            from alternatives import evaluate_solution
            
            for result in pool.imap_unordered(evaluate_solution, all_args):
                fitness_val, solution_idx, bankrupt = result
                fitness_results[solution_idx] = fitness_val
                
                if fitness_val < best_fitness:
                    best_fitness = fitness_val
                    best_solution_normalized = solutions_normalized[solution_idx].copy()
        
        pool.close()
        pool.join()
        
        # Task 4.10: Compute behavioral descriptors and update archive
        # Note: We need analyses_combined from evaluator to compute behavioral descriptors
        # For now, we'll use dummy behavioral descriptors based on fitness
        # This is a temporary solution until we can get analyses_combined
        
        # Note: pyribs uses MAXIMIZATION by default, so we negate fitness for minimization
        objectives = -np.array(fitness_results)
        
        # Temporary: Create dummy behavioral descriptors
        # In production, these should come from compute_behavioral_descriptor()
        num_features = len(get_feature_names(config))
        measures = np.random.rand(len(solutions), num_features)
        
        # Tell scheduler the results
        prev_num_elites = archive.stats.num_elites
        scheduler.tell(objectives, measures)
        new_num_elites = archive.stats.num_elites
        
        # Track additions and improvements
        num_additions = new_num_elites - prev_num_elites
        # Note: pyribs doesn't directly expose improvements, so we approximate
        # Improvements = solutions that didn't add new cells but improved existing ones
        num_improvements = len(solutions) - num_additions
        
        # Task 4.14: Record generation statistics
        gen_time = time.time() - gen_start_time
        stats = archive.stats
        
        logbook.append({
            "gen": gen,
            "nevals": len(solutions),
            "best": best_fitness,
            "mean": np.mean(fitness_results),
            "archive_coverage": stats.coverage * 100,
            "qd_score": -(stats.obj_mean * stats.num_elites),  # Negate back since we negated objectives for pyribs
            "num_elites": stats.num_elites,
            "num_additions": num_additions,
            "num_improvements": max(0, num_improvements),  # Ensure non-negative
            "time": gen_time,
        })
        
        # Task 4.12: Periodic reporting
        if verbose and gen % 5 == 0:
            log_archive_stats(archive, gen, best_fitness, file_console)
            
            # Show best solution parameters (denormalize for display)
            best_solution_original = _denormalize_solution(
                best_solution_normalized, lower_bounds, upper_bounds
            )
            file_console.print("\n🌐 Best Solution Parameters:")
            for i, (param_name, value) in enumerate(
                zip(optimizable_param_names, best_solution_original)
            ):
                if param_name in integer_params:
                    file_console.print(f"  📊 {param_name}: {int(round(value))}")
                else:
                    file_console.print(f"  📊 {param_name}: {value:.6f}")
        
        # Task 4.13: Save checkpoint
        try:
            save_checkpoint(
                checkpoint_path, archive, scheduler, gen, best_fitness, logbook
            )
            if verbose and gen % 5 == 0:
                file_console.print(f"💾 Checkpoint saved at generation {gen}")
        except Exception as e:
            file_console.print(f"⚠️ Checkpoint save failed: {e}")
    
    # Task 4.15: Create final population (denormalize best solution)
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from alternatives import create_full_individual
    
    best_solution_original = _denormalize_solution(
        best_solution_normalized, lower_bounds, upper_bounds
    )
    
    final_population = []
    for i in range(len(population)):
        individual = create_full_individual(
            best_solution_original,
            optimizable_param_names,
            all_param_names,
            fixed_params,
            integer_params,
            toolbox,
        )
        individual.fitness.values = (best_fitness,)
        final_population.append(individual)
    
    total_time = time.time() - start_time
    file_console.print(
        f"🕒 Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)"
    )
    file_console.print(f"🏆 Final best fitness: {best_fitness:.6e}")
    
    # Close log file
    log_file.close()
    
    return final_population, logbook
