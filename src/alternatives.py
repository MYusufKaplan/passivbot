"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation
for optimizing trading bot parameters.
"""

import os
import pickle
import datetime
import time
import json
import numpy as np
import random
from multiprocessing import Pool, cpu_count
import contextlib

# Rich imports for beautiful logging
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.rule import Rule
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, TaskProgressColumn

# CMA-ES import
import cmaes


# PSO import
try:
    from my_pyswarms import global_best as ps
    PYSWARMS_AVAILABLE = True
except ImportError:
    PYSWARMS_AVAILABLE = False
    print("⚠️ Custom PySwarms not found. Make sure src/my_pyswarms exists.")

# Nevergrad import
try:
    import nevergrad as ng
    NEVERGRAD_AVAILABLE = True
except ImportError:
    NEVERGRAD_AVAILABLE = False
    print("⚠️ Nevergrad not installed. Install with: pip install nevergrad")

# Color constants
RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Initialize the console for rich output
console = console = Console(
            force_terminal=True, 
            no_color=False, 
            log_path=False, 
            width=191,
            color_system="truecolor",  # Force truecolor support
            legacy_windows=False
        )

# Log paths
LOG_PATH = "logs/evaluation_output.log"
WATCH_PATH = "logs/evaluation.log"
BEST_LOG_PATH = "logs/evaluation_output_best.log"

def log_message(message, emoji=None, panel=False, timestamp=True):
    """Utility function to print logs with Rich panels and rules"""
    if timestamp:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        timestamp = ""
    
    # Check if message already starts with an emoji (up until a regular char is encountered)
    has_emoji_at_start = False
    for char in message:
        if char.isalnum() or char.isspace():
            break
        # If we encounter a non-alphanumeric, non-space character, assume it's an emoji
        has_emoji_at_start = True
        break
    
    emoji_str = f" {emoji}" if emoji and not has_emoji_at_start else ""
    log_text = f"{timestamp}{emoji_str} {message}"

    # Using Rule for major transitions
    if panel:
        panel_message = Panel(log_text, title="CMA-ES Stats", border_style="cyan")
        console_wrapper(panel_message)
    else:
        console_wrapper(log_text)

def console_wrapper(msg):
    with open(WATCH_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        console.print(msg)

def evaluate_solution(args):
    """Evaluate a single solution"""
    if len(args) == 4:
        # Nevergrad case with index tracking
        evaluator, ind, showMe, idx = args
        if showMe:
            with open(BEST_LOG_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                fitness = evaluator.evaluate(ind)
                return fitness[0], idx, fitness[2]
        with open(LOG_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            fitness = evaluator.evaluate(ind)
            return fitness[0], idx, fitness[2]
    else:
        # Original case (CMA-ES, PSO)
        evaluator, ind, showMe = args
        if showMe:
            with open(BEST_LOG_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                return evaluator.evaluate(ind)
        with open(LOG_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            return evaluator.evaluate(ind)

def pso(population, toolbox, evaluator, ngen, verbose=True, parameter_bounds=None, checkpoint_path="pso_checkpoint.pkl"):
    """
    Particle Swarm Optimization (PSO) implementation using enhanced PySwarms
    
    Args:
        population: Initial population (used to determine swarm size)
        toolbox: DEAP toolbox with individual creation
        evaluator: Evaluation function
        ngen: Maximum number of generations
        verbose: Whether to show detailed logging
        parameter_bounds: Dictionary of parameter bounds
        checkpoint_path: Path to save/load checkpoints
    
    Returns:
        final_population: List of best individuals
        logbook: Evolution statistics
    """
    if not PYSWARMS_AVAILABLE:
        raise ImportError("PySwarms is required for PSO. Install with: pip install pyswarms")
    
    start_time = time.time()
    
    if not parameter_bounds:
        raise ValueError("parameter_bounds is required for PSO")
    
    # Filter optimizable vs fixed parameters and identify integer parameters
    optimizable_bounds = {}
    fixed_params = {}
    integer_params = set()
    
    for param_name, bounds in parameter_bounds.items():
        min_val, max_val = bounds
        if "short" not in param_name and min_val != max_val:  # Parameter has range to optimize
            optimizable_bounds[param_name] = bounds
            # Check if parameter should be integer
            if any(keyword in param_name.lower() for keyword in ['n_positions']):
                integer_params.add(param_name)
        else:  # Fixed parameter
            fixed_params[param_name] = min_val
    
    all_param_names = list(parameter_bounds.keys())
    optimizable_param_names = list(optimizable_bounds.keys())
    num_parameters = len(optimizable_param_names)
    
    log_message(f"🐝 PSO: {num_parameters} optimizable parameters, {len(fixed_params)} fixed", emoji="🔬")
    
    # Load initial values from optimize.json
    initial_values = {}
    try:
        with open("configs/optimize.json", "r") as f:
            config = json.load(f)
            long_config = config.get("bot", {}).get("long", {})
            for param_name in optimizable_param_names:
                clean_name = param_name.replace("long_", "")
                if clean_name in long_config:
                    value = long_config[clean_name]
                    # Only use numeric values (ignore booleans like enforce_exposure_limit)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        initial_values[param_name] = float(value)
        log_message(f"📋 Loaded {len(initial_values)} initial values from optimize.json", emoji="📋")
    except Exception as e:
        log_message(f"⚠️ Could not load optimize.json, using middle values: {e}", emoji="⚠️")
    
    # Create bounds arrays for PySwarms
    lower_bounds = []
    upper_bounds = []
    initial_vector = []
    
    for param_name in optimizable_param_names:
        min_val, max_val = optimizable_bounds[param_name]
        
        # Use initial value from config if available, otherwise use middle value
        if param_name in initial_values:
            initial_val = initial_values[param_name]
            if param_name in integer_params:
                initial_val = round(initial_val)
        else:
            # Fallback to middle value
            initial_val = (min_val + max_val) / 2
            if param_name in integer_params:
                initial_val = round(initial_val)
        
        initial_vector.append(initial_val)
        lower_bounds.append(min_val)
        upper_bounds.append(max_val)
    
    # PSO hyperparameters - optimized for trading parameter optimization
    options = {
        'c1': 0.8,    # cognitive parameter (personal best attraction)
        'c2': 0.3,    # social parameter (global best attraction) - increased for better global search
        'w': 0.9      # inertia weight (will be adaptive in the library)
    }
    
    # Swarm size
    n_particles = len(population)
    bounds = (np.array(lower_bounds), np.array(upper_bounds))
    
    # Validate bounds - ensure initial vector is within bounds
    for i, (init_val, lb, ub) in enumerate(zip(initial_vector, lower_bounds, upper_bounds)):
        if not (lb <= init_val <= ub):
            log_message(f"⚠️ Parameter {optimizable_param_names[i]}: initial {init_val} not in bounds [{lb}, {ub}]", emoji="⚠️")
            # Clamp to bounds
            initial_vector[i] = max(lb, min(ub, init_val))
    
    log_message(f"🔍 Initial vector: {initial_vector[:3]}... (showing first 3)", emoji="🔍")
    log_message(f"🔍 Lower bounds: {lower_bounds[:3]}... (showing first 3)", emoji="🔍")
    log_message(f"🔍 Upper bounds: {upper_bounds[:3]}... (showing first 3)", emoji="🔍")
    log_message(f"🔍 PSO options: {options}", emoji="🔍")
    
    # Create PSO optimizer with enhanced features
    optimizer = ps.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=num_parameters,
        options=options,
        bounds=bounds,
        init_pos=np.array(initial_vector)  # Single particle position from config values
    )
    
    # Configure checkpoint and logging
    optimizer.set_checkpoint_config(
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        watch_path=WATCH_PATH
    )

    optimizer.set_competition_config(
        enable=False,
        max_particles_per_cell=10,
        stagnation_window=10,
        eviction_percentage=40,
        check_interval=1
    )
    
    optimizer.set_blacklist_config()

    optimizer.set_initialization_strategy('lhs')
    optimizer.set_velocity_boost_config(interval=10, fraction=0.50, enable=True, optimize_limits_len=5)
    # optimizer.set_scout_config(scout_percentage=0.2, lifecycle=50, performance_threshold=3.0, enable=True)
    optimizer.set_pca_visualization(
        enable=True,
        width=181,
        height=31,
        graphs_path="logs/graphs.log"
    )

    # Configure 10n selection
    optimizer.set_10n_selection_config(
        enable=False,
        multiplier=1000,      # 1000n candidates
        interval=10,        # Every 10 iterations
        quality_threshold_pct=100,
        on_fresh_start=True,
        on_checkpoint_resume=True
    )

    # Add small random perturbations to initial positions
    perturbation_scale = 0.1
    for i in range(n_particles):
        for j in range(num_parameters):
            param_range = upper_bounds[j] - lower_bounds[j]
            perturbation = np.random.normal(0, perturbation_scale * param_range)
            optimizer.swarm.position[i, j] = np.clip(
                optimizer.swarm.position[i, j] + perturbation,
                lower_bounds[j],
                upper_bounds[j]
            )
    
    # Global variables for objective function
    global pso_evaluator, pso_toolbox, pso_optimizable_param_names, pso_all_param_names, pso_fixed_params, pso_integer_params
    
    pso_evaluator = evaluator
    pso_toolbox = toolbox
    pso_optimizable_param_names = optimizable_param_names
    pso_all_param_names = all_param_names
    pso_fixed_params = fixed_params
    pso_integer_params = integer_params

    pool = Pool(processes=(cpu_count() - 1))

    
    def objective_function(swarm_positions):
        """
        Objective function for PySwarms
        Args:
            swarm_positions: numpy array of shape (n_particles, n_dimensions)
        Returns:
            fitness_values: numpy array of shape (n_particles,)
        """
        fitnesses = []
        
        # Prepare multiprocessing evaluation with position tracking
        all_args = []
        
        for idx, position in enumerate(swarm_positions):
            individual = create_full_individual(position, pso_optimizable_param_names, 
                                              pso_all_param_names, pso_fixed_params, 
                                              pso_integer_params, pso_toolbox)
            # Include the position index to maintain pairing
            all_args.append((pso_evaluator, individual, False, idx))
        with open(WATCH_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            
            # Evaluate with progress bar (similar to CMA-ES)
            with Progress(
                SpinnerColumn(spinner_name="dots12"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                "•",
                TaskProgressColumn(text_format="[progress.percentage]{task.percentage:>5.1f}%", show_speed=True),
                "•",
                TimeElapsedColumn(),
                "•",
                TimeRemainingColumn(),
                "•",
                console=console,
                transient=True
            ) as progress:
                
                # Get current global best fitness for progress tracking
                global_best = getattr(optimizer.swarm, 'best_cost', float('inf')) if hasattr(optimizer, 'swarm') else float('inf')
                
                # Initialize generational best tracking and bankruptcy counters
                generational_best = float('inf')
                bankrupt_count = 0
                non_bankrupt_count = 0
                
                task = progress.add_task(f"PSO Evaluating particles | Gen Best: {generational_best:.6e} | Global Best: {global_best:.6e}", total=len(swarm_positions))
                
                # Collect results with their original indices to maintain position-fitness pairing
                fitness_results = [None] * len(swarm_positions)  # Pre-allocate array
                
                for result in pool.imap_unordered(evaluate_solution, all_args):
                    fitness_val, position_idx, bankrupt = result[0], result[1], result[2]
                    fitness_results[position_idx] = fitness_val  # Place fitness at correct index
                    
                    # Track bankruptcy counts
                    if bankrupt:
                        bankrupt_count += 1
                    else:
                        non_bankrupt_count += 1
                    
                    # Track best in current generation (batch)
                    if fitness_val < generational_best:
                        generational_best = fitness_val

                    # Simple count display
                    push_bar = f"✅{non_bankrupt_count} 💀{bankrupt_count}"

                    # Update progress with counts and fitness info
                    emoji = "🐝" if generational_best >= global_best else "🔥"
                    progress.update(
                        task, 
                        advance=1,
                        description=f"{emoji} {push_bar} | Gen Best: {generational_best:.6e} | Global Best: {global_best:.6e}"
                    )
                
                # Convert to list for return (fitness_results is now in correct order)
                fitnesses = fitness_results
            
            # pool.close()
            # pool.join()
        
        return np.array(fitnesses)
    
    # Run PSO optimization - the enhanced library handles all logging and checkpointing
    final_best_cost, final_best_pos = optimizer.optimize(
        objective_function, 
        iters=ngen, 
        n_processes=None,  # We handle multiprocessing in objective_function
        verbose=verbose
    )
    
    # Create logbook from optimizer history
    logbook = []
    for i, (cost_hist, pos_hist) in enumerate(zip(optimizer.cost_history, optimizer.pos_history)):
        if len(cost_hist) > 0:
            # Calculate swarm statistics
            diversity = np.mean(np.std(pos_hist, axis=0))
            if hasattr(optimizer, 'generation_times') and i < len(optimizer.generation_times):
                gen_time = optimizer.generation_times[i]
            else:
                gen_time = 0.0
                
            logbook.append({
                "gen": i,
                "nevals": n_particles,
                "best": np.min(cost_hist),
                "mean": np.mean(cost_hist),
                "std": np.std(cost_hist),
                "diversity": diversity,
                "time": gen_time
            })
    
    # Create final population for return
    final_population = []
    for i in range(len(population)):  # Match original population size
        individual = create_full_individual(final_best_pos, optimizable_param_names, all_param_names, fixed_params, integer_params, toolbox)
        individual.fitness.values = (final_best_cost,)
        final_population.append(individual)
    
    total_time = time.time() - start_time
    log_message(f"🕒 Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)", emoji="🕒")
    log_message(f"🏆 Final best fitness: {final_best_cost:.6e}", emoji="🏆")
    
    return final_population, logbook

def create_full_individual(solution, optimizable_param_names, all_param_names, fixed_params, integer_params, toolbox):
    """Create a full individual from optimization solution vector"""
    individual = toolbox.individual()
    
    # Create full parameter vector
    full_solution = []
    opt_idx = 0
    
    for param_name in all_param_names:
        if param_name in fixed_params:
            # Fixed parameter
            full_solution.append(fixed_params[param_name])
        elif param_name in optimizable_param_names:
            # Optimizable parameter
            value = solution[opt_idx]
            if param_name in integer_params:
                value = round(value)
            full_solution.append(value)
            opt_idx += 1
        else:
            # This shouldn't happen if parameter setup is correct
            raise ValueError(f"Parameter '{param_name}' is neither fixed nor optimizable")
    
    individual[:] = full_solution
    return individual


# CMA-ES with Restarts - lazy import to avoid circular dependency
CMA_ES_AVAILABLE = False
_cma_es_restarts = None

def cma_es_restarts(*args, **kwargs):
    """
    Wrapper function for CMA-ES with automatic restarts.
    Lazy-loads the actual implementation to avoid circular imports.
    """
    global _cma_es_restarts, CMA_ES_AVAILABLE
    
    if _cma_es_restarts is None:
        try:
            from cma_es import cma_es_restarts as _cma_impl
            _cma_es_restarts = _cma_impl
            CMA_ES_AVAILABLE = True
        except ImportError as e:
            raise ImportError(
                f"CMA-ES with Restarts not available: {e}\n"
                "Make sure src/cma_es exists and all dependencies are installed."
            )
    
    return _cma_es_restarts(*args, **kwargs)


def deap_ea(population, toolbox, evaluator, ngen, verbose=True, parameter_bounds=None, checkpoint_path="deap_checkpoint.pkl", cxpb=0.5, mutpb=0.2, stagnation_config=None, config=None, interval_data=None):
    """
    DEAP Evolutionary Algorithm implementation using the reorganized DEAP module.
    
    This function provides the interface between optimize.py and the DEAP
    evolutionary algorithm implementation, following the same pattern as PSO
    and CMA-ES.
    
    Args:
        population: Initial population (DEAP individuals)
        toolbox: DEAP toolbox with genetic operators
        evaluator: Evaluator instance for fitness evaluation
        ngen: Maximum number of generations
        verbose: Whether to show detailed logging
        parameter_bounds: Dictionary of parameter bounds
        checkpoint_path: Path to save/load checkpoints
        cxpb: Crossover probability (default: 0.5)
        mutpb: Mutation probability (default: 0.2)
        stagnation_config: Dictionary with stagnation detection configuration (optional)
        config: Configuration dictionary (optional, needed for interval evaluation)
        interval_data: Dictionary with timestamps, hlcvs, btc_usd_data for interval creation (optional)
    
    Returns:
        final_population: List of final individuals
        logbook: Evolution statistics
    
    Requirements: 1.4, 6.1, 6.3, 6.5, 2.2, 2.3, 2.4, 8.1
    """
    from deap import tools
    from deap_optimizer.evolutionary_algorithm import eaMuPlusLambda
    from deap_optimizer.interval import split_into_monthly_intervals, cleanup_interval_files
    from optimize import create_shared_memory_file
    import numpy as np
    
    start_time = time.time()
    
    if not parameter_bounds:
        raise ValueError("parameter_bounds is required for DEAP EA")
    
    n_individuals = len(population)
    dimensions = len(parameter_bounds)
    
    log_message(f"🧬 DEAP EA: {dimensions} parameters, {n_individuals} individuals", emoji="🔬")
    
    # Set up statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    # Set up hall of fame
    halloffame = tools.HallOfFame(1)
    
    # Initialize Rich console for logging (following PSO pattern)
    deap_console = Console(
        force_terminal=True,
        no_color=False,
        log_path=False,
        width=191,
        color_system="truecolor",
        legacy_windows=False
    )
    
    # Check if interval evaluation is configured
    intervals = None
    use_intervals = config and config.get("backtest", {}).get("use_monthly_intervals", False)
    
    if use_intervals and interval_data:
        log_message("📅 Monthly interval evaluation enabled, creating intervals...", emoji="📅")
        
        # Get the first exchange's data (all exchanges should have aligned timestamps)
        first_exchange = next(iter(interval_data["timestamps"]))
        timestamps = interval_data["timestamps"][first_exchange]
        
        # Create intervals using the interval module
        intervals = split_into_monthly_intervals(
            start_date=config["backtest"]["start_date"],
            end_date=config["backtest"]["end_date"],
            timestamps=timestamps,
            shared_hlcvs=interval_data["hlcvs"],
            hlcvs_shapes={ex: arr.shape for ex, arr in interval_data["hlcvs"].items()},
            btc_usd_data=interval_data["btc_usd_data"],
            btc_usd_dtypes={ex: arr.dtype for ex, arr in interval_data["btc_usd_data"].items()},
            create_shared_memory_fn=create_shared_memory_file,
        )
        
        log_message(f"📅 Created {len(intervals)} monthly intervals", emoji="📅")
    
    # Run the simplified eaMuPlusLambda algorithm directly
    # This follows the pattern where the algorithm is called directly
    # rather than through a wrapper class
    try:
        final_population, logbook = eaMuPlusLambda(
            population=population,
            toolbox=toolbox,
            mu=n_individuals,
            lambda_=n_individuals,
            cxpb=cxpb,  # Use passed crossover probability
            mutpb=mutpb,  # Use passed mutation probability
            ngen=ngen,
            stats=stats,
            halloffame=halloffame,
            verbose=verbose,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=1,
            console=deap_console,
            console_logging=True,
            file_logging=True,
            watch_path=WATCH_PATH,
            evaluator=evaluator,
            parameter_bounds=parameter_bounds,
            stagnation_config=stagnation_config,
            intervals=intervals
        )
        
        # Get best individual from hall of fame
        if halloffame:
            best_individual = halloffame[0]
            best_fitness = best_individual.fitness.values[0]
            log_message(f"✨ DEAP EA complete! Best fitness: {best_fitness:.6e}", emoji="🎉")
        
        total_time = time.time() - start_time
        log_message(f"🕒 Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)", emoji="🕒")
        
        return final_population, logbook
        
    except Exception as e:
        log_message(f"❌ DEAP EA failed: {e}", emoji="❌")
        raise
    finally:
        # Clean up interval files if they were created
        if intervals:
            log_message("🧹 Cleaning up interval shared memory files...", emoji="🧹")
            cleanup_interval_files(intervals)
            log_message("🧹 Interval cleanup complete", emoji="🧹")
