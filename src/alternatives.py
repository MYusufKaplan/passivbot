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
    import pyswarms as ps
    PYSWARMS_AVAILABLE = True
except ImportError:
    PYSWARMS_AVAILABLE = False
    print("⚠️ PySwarms not installed. Install with: pip install pyswarms")

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
            width=159,
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
                return fitness, idx
        with open(LOG_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            fitness = evaluator.evaluate(ind)
            return fitness[0], idx
    else:
        # Original case (CMA-ES, PSO)
        evaluator, ind, showMe = args
        if showMe:
            with open(BEST_LOG_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                return evaluator.evaluate(ind)
        with open(LOG_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            return evaluator.evaluate(ind)

def cma(population, toolbox, evaluator, ngen, verbose=True, parameter_bounds=None, checkpoint_path="cma_checkpoint.pkl"):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation
    
    Args:
        population: Initial population (ignored, CMA-ES determines its own size)
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
    start_time = time.time()
    
    if not parameter_bounds:
        raise ValueError("parameter_bounds is required for CMA-ES")
    
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
    
    log_message(f"🧬 CMA-ES: {num_parameters} optimizable parameters, {len(fixed_params)} fixed", emoji="🔬")
    
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
    
    # Create initial vector and bounds for CMA-ES
    initial_vector = []
    lower_bounds = []
    upper_bounds = []
    
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
    
    # Try loading from checkpoint
    es = None
    start_gen = 1
    best_fitness = float('inf')
    best_solution = None
    logbook = []
    
    if os.path.exists(checkpoint_path):
        log_message("📦 Loading CMA-ES checkpoint...", emoji="📦")
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
                
                # Create bounds in the format cmaes expects: numpy array with shape (n_parameters, 2)
                bounds_array = np.array([[lb, ub] for lb, ub in zip(lower_bounds, upper_bounds)])
                
                # Reconstruct CMA-ES from checkpoint
                es = cmaes.CMA(
                    mean=checkpoint_data["mean"],
                    sigma=checkpoint_data["sigma"],
                    bounds=bounds_array,
                    population_size=checkpoint_data.get("population_size")
                )
                
                start_gen = checkpoint_data["generation"] + 1
                best_fitness = checkpoint_data["best_fitness"]
                best_solution = checkpoint_data["best_solution"]
                logbook = checkpoint_data.get("logbook", [])
                
            log_message(f"✅ Resumed from generation {start_gen-1}, best fitness: {best_fitness:.6e}", emoji="✅")
        except Exception as e:
            log_message(f"⚠️ Failed to load checkpoint: {e}, starting fresh", emoji="⚠️")
            es = None
    
    # Initialize CMA-ES if not loaded from checkpoint
    if es is None:
        log_message("🚀 Starting fresh CMA-ES optimization", emoji="🚀")
        
        # Calculate initial sigma (step size) as 1/6 of the parameter ranges
        sigma = np.mean([(ub - lb) / 6.0 for lb, ub in zip(lower_bounds, upper_bounds)])
        
        # Debug: Print bounds info
        log_message(f"🔍 Initial vector: {initial_vector[:3]}... (showing first 3)", emoji="🔍")
        log_message(f"🔍 Lower bounds: {lower_bounds[:3]}... (showing first 3)", emoji="🔍")
        log_message(f"🔍 Upper bounds: {upper_bounds[:3]}... (showing first 3)", emoji="🔍")
        log_message(f"🔍 Sigma: {sigma}", emoji="🔍")
        
        # Validate bounds - ensure initial vector is within bounds
        for i, (init_val, lb, ub) in enumerate(zip(initial_vector, lower_bounds, upper_bounds)):
            if not (lb <= init_val <= ub):
                log_message(f"⚠️ Parameter {optimizable_param_names[i]}: initial {init_val} not in bounds [{lb}, {ub}]", emoji="⚠️")
                # Clamp to bounds
                initial_vector[i] = max(lb, min(ub, init_val))
        
        # Create bounds in the format cmaes expects: numpy array with shape (n_parameters, 2)
        bounds_array = np.array([[lb, ub] for lb, ub in zip(lower_bounds, upper_bounds)])
        
        es = cmaes.CMA(
            mean=np.array(initial_vector),
            sigma=sigma,
            bounds=bounds_array,
            population_size=len(population)
        )
        
        # Evaluate initial solution
        log_message("🔍 Initial evaluation...", emoji="🔍")
        initial_individual = create_full_individual(initial_vector, optimizable_param_names, all_param_names, fixed_params, integer_params, toolbox)
        best_fitness = evaluator.evaluate(initial_individual)[0]
        best_solution = initial_vector.copy()
        
        logbook.append({
            "gen": 0,
            "nevals": 1,
            "best": best_fitness,
            "mean": best_fitness,
            "std": 0.0,
            "time": 0.0
        })
    
    log_message(f"🧬 Starting CMA-ES evolution from generation {start_gen}", emoji="🧬")
    log_message(f"📊 Population size: {es.population_size}", emoji="📊")
    
    # Initialize tracking variables
    generation_times = []
    stagnation = 0
    last_improvement_gen = 0
    previous_best_fitness = best_fitness
    
    # Main CMA-ES Evolution Loop
    for gen in range(start_gen, ngen + 1):
        # Note: Removed es.should_stop() to allow aggressive optimization for lowest fitness
        # CMA-ES will run for full ngen generations to find the absolute minimum
            
        gen_start_time = time.time()
        console_wrapper(Rule(f"Generation {gen}", style="bold blue"))
        
        # Ask CMA-ES for new solutions
        solutions = []
        for _ in range(es.population_size):
            solutions.append(es.ask())
        
        # Prepare evaluation arguments with solution index tracking
        pool = Pool(processes=(cpu_count() - 1))
        all_args = []
        
        for idx, solution in enumerate(solutions):
            individual = create_full_individual(solution, optimizable_param_names, all_param_names, fixed_params, integer_params, toolbox)
            # Include the solution index to maintain pairing
            all_args.append((evaluator, individual, False, idx))
        
        # Evaluate with progress bar
        fitnesses = []
        with open(WATCH_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):

            with Progress(
                SpinnerColumn(spinner_name="dots12"),
                TextColumn("🧬 [progress.description]{task.description}"),
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
                
                task = progress.add_task(f"🧬 CMA-ES Gen {gen} | Best: {best_fitness:.6e}", total=es.population_size)
                
                # Collect results with their original indices to maintain solution-fitness pairing
                fitness_results = [None] * es.population_size  # Pre-allocate array
                current_best_gen = float('inf')
                
                for result in pool.imap_unordered(evaluate_solution, all_args):
                    fitness_val, solution_idx = result[0], result[1]
                    fitness_results[solution_idx] = fitness_val  # Place fitness at correct index
                    
                    # Track best in current generation
                    if fitness_val < current_best_gen:
                        current_best_gen = fitness_val
                    
                    # Update global best
                    if fitness_val < best_fitness:
                        improvement = best_fitness - fitness_val
                        best_fitness = fitness_val
                        # Use the correct solution corresponding to this fitness
                        best_solution = solutions[solution_idx].copy()
                        
                        # Log new global best with celebration
                        log_message(f"🎉 NEW GLOBAL BEST! 🎉 Fitness: {best_fitness:.6e} (improved by {improvement:.6e})", emoji="🌟")
                    
                    progress.update(
                        task, 
                        advance=1,
                        description=f"🧬 CMA-ES Gen {gen} | Best: {best_fitness:.6e}"
                    )
                
                # Convert to list for CMA-ES (fitness_results is now in correct order)
                fitnesses = fitness_results
            
            pool.close()
            pool.join()
        
        # Tell CMA-ES the results
        es.tell([(sol, fit) for sol, fit in zip(solutions, fitnesses)])
        
        # Update stagnation tracking
        if best_fitness < previous_best_fitness:
            # Improvement found
            stagnation = 0
            last_improvement_gen = gen
            previous_best_fitness = best_fitness
        else:
            # No improvement
            stagnation += 1
        
        # Calculate statistics
        gen_time = time.time() - gen_start_time
        generation_times.append(gen_time)
        avg_gen_time = sum(generation_times) / len(generation_times)
        
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)
        
        logbook.append({
            "gen": gen,
            "nevals": es.population_size,
            "best": best_fitness,
            "mean": mean_fitness,
            "std": std_fitness,
            "time": gen_time
        })
        
        if verbose:
            # Show parameter values for best solution in 2 columns
            param_info = "\n🌐 Best Solution Parameters:"
            param_lines = []
            
            for i, (param_name, value) in enumerate(zip(optimizable_param_names, best_solution)):
                if param_name in integer_params:
                    param_lines.append(f"📊 {param_name}: {int(round(value))}")
                else:
                    param_lines.append(f"📊 {param_name}: {value:.6f}")
            
            # Arrange in 2 columns
            mid_point = (len(param_lines) + 1) // 2
            left_column = param_lines[:mid_point]
            right_column = param_lines[mid_point:]
            
            # Pad right column if needed
            while len(right_column) < len(left_column):
                right_column.append("")
            
            # Create 2-column layout
            for left, right in zip(left_column, right_column):
                if right:  # Both columns have content
                    param_info += f"\n                  {left:<50} {right}"
                else:  # Only left column has content
                    param_info += f"\n                  {left}"
            
            # Stagnation status with emoji
            stagnation_emoji = "🔥" if stagnation == 0 else "😴" if stagnation < 10 else "💤" if stagnation < 50 else "⚰️"
            stagnation_info = f"🔄 Stagnation: {stagnation} gens {stagnation_emoji} (last improvement: gen {last_improvement_gen})"
            
            log_message(
                f"""{CYAN}🌟 Gen {gen}{RESET}
                🧬 Population size: {es.population_size}
                🌍 Best fitness: {best_fitness:.6e}
                📊 Mean fitness: {mean_fitness:.6e}
                📈 Std fitness: {std_fitness:.6e}
                {stagnation_info}
                ⏱️ Generation time: {gen_time:.2f} sec / {(gen_time/60):.2f} min
                📆 Avg gen time: {avg_gen_time:.2f} sec / {(avg_gen_time/60):.2f} min
                🎯 Sigma: {es._sigma:.6f}{param_info}""",
                panel=True, timestamp=False
            )
        
        # Checkpoint saving every 5 generations
        if gen % 1 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "mean": es.mean,
                    "sigma": es._sigma,
                    "population_size": es.population_size,
                    "generation": gen,
                    "best_fitness": best_fitness,
                    "best_solution": best_solution,
                    "logbook": logbook
                }, f)
            log_message(f"💾 Checkpoint saved at generation {gen}", emoji="💾")
    
    # Create final population for return
    final_population = []
    for i in range(len(population)):  # Match original population size
        individual = create_full_individual(best_solution, optimizable_param_names, all_param_names, fixed_params, integer_params, toolbox)
        individual.fitness.values = (best_fitness,)
        final_population.append(individual)
    
    total_time = time.time() - start_time
    log_message(f"🕒 Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)", emoji="🕒")
    log_message(f"🏆 Final best fitness: {best_fitness:.6e}", emoji="🏆")
    
    return final_population, logbook

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
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=num_parameters,
        options=options,
        bounds=bounds,
        # init_pos=np.tile(initial_vector, (n_particles, 1))  # Initialize around config values
    )
    
    # Configure checkpoint and logging
    optimizer.set_checkpoint_config(
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        watch_path=WATCH_PATH
    )
    
    optimizer.set_initialization_strategy('lhs')
    optimizer.set_velocity_boost_config(interval=10, fraction=0.20, enable=True)
    # optimizer.set_scout_config(scout_percentage=0.2, lifecycle=50, performance_threshold=3.0, enable=True)

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
                TextColumn("🐝 [progress.description]{task.description}"),
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
                
                # Initialize generational best tracking
                generational_best = float('inf')
                
                task = progress.add_task(f"🐝 PSO Evaluating particles | Gen Best: {generational_best:.6e} | Global Best: {global_best:.6e}", total=len(swarm_positions))
                
                # Collect results with their original indices to maintain position-fitness pairing
                fitness_results = [None] * len(swarm_positions)  # Pre-allocate array
                
                for result in pool.imap_unordered(evaluate_solution, all_args):
                    fitness_val, position_idx = result[0], result[1]
                    fitness_results[position_idx] = fitness_val  # Place fitness at correct index
                    
                    # Track best in current generation (batch)
                    if fitness_val < generational_best:
                        generational_best = fitness_val

                    # Update progress with generational best and global best
                    emoji = "🐝" if generational_best >= global_best else "🔥"
                    progress.update(
                        task, 
                        advance=1,
                        description=f"{emoji} PSO Evaluating particles | Gen Best: {generational_best:.6e} | Global Best: {global_best:.6e}"
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

def nevergrad_opt(population, toolbox, evaluator, ngen, verbose=True, parameter_bounds=None, checkpoint_path="nevergrad_checkpoint.pkl", optimizer_name="NGOpt"):
    """
    Nevergrad optimization implementation using ngopt
    
    Args:
        population: Initial population (used to determine budget)
        toolbox: DEAP toolbox with individual creation
        evaluator: Evaluation function
        ngen: Maximum number of generations (converted to budget)
        verbose: Whether to show detailed logging
        parameter_bounds: Dictionary of parameter bounds
        checkpoint_path: Path to save/load checkpoints
        optimizer_name: Nevergrad optimizer to use (NGOpt, DE, PSO, CMA, etc.)
    
    Returns:
        final_population: List of best individuals
        logbook: Evolution statistics
    """
    if not NEVERGRAD_AVAILABLE:
        raise ImportError("Nevergrad is required. Install with: pip install nevergrad")
    
    start_time = time.time()
    
    if not parameter_bounds:
        raise ValueError("parameter_bounds is required for Nevergrad")
    
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
    
    log_message(f"🎯 Nevergrad ({optimizer_name}): {num_parameters} optimizable parameters, {len(fixed_params)} fixed", emoji="🔬")
    
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
    
    # Create parameter space for Nevergrad
    parametrization = {}
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
        
        # Create appropriate parameter type
        if param_name in integer_params:
            parametrization[param_name] = ng.p.Scalar(lower=min_val, upper=max_val).set_integer_casting()
        else:
            parametrization[param_name] = ng.p.Scalar(lower=min_val, upper=max_val)
    
    # Create the instrumentation
    instrum = ng.p.Instrumentation(**parametrization)
    
    # Calculate budget (total evaluations)
    population_size = len(population)
    budget = ngen * population_size
    
    log_message(f"🔍 Initial vector: {initial_vector[:3]}... (showing first 3)", emoji="🔍")
    log_message(f"🔍 Budget: {budget} evaluations ({ngen} generations × {population_size} population)", emoji="🔍")
    log_message(f"🔍 Optimizer: {optimizer_name}", emoji="🔍")
    
    # Create optimizer - fix numpy int issue by using numpy int conversion
    try:
        optimizer = ng.optimizers.registry[optimizer_name](parametrization=instrum, budget=np.int64(budget))
        optimizer.enable_pickling()  # Enable picklability for checkpointing
    except Exception as e:
        log_message(f"⚠️ Error creating optimizer {optimizer_name}: {e}", emoji="⚠️")
        log_message("🔄 Falling back to DE optimizer", emoji="🔄")
        optimizer = ng.optimizers.registry["DE"](parametrization=instrum, budget=np.int64(budget))
        optimizer.enable_pickling()  # Enable picklability for checkpointing
        optimizer_name = "DE"
    
    # Try loading from checkpoint
    start_eval = 0
    best_fitness = float('inf')
    best_solution = None
    logbook = []
    evaluation_history = []
    
    if os.path.exists(checkpoint_path):
        log_message("📦 Loading Nevergrad checkpoint...", emoji="📦")
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
                
                # Restore optimizer state
                optimizer = checkpoint_data["optimizer"]
                start_eval = checkpoint_data["num_evaluations"]
                best_fitness = checkpoint_data["best_fitness"]
                best_solution = checkpoint_data["best_solution"]
                logbook = checkpoint_data.get("logbook", [])
                evaluation_history = checkpoint_data.get("evaluation_history", [])
                
            log_message(f"✅ Resumed from evaluation {start_eval}, best fitness: {best_fitness:.6e}", emoji="✅")
        except Exception as e:
            log_message(f"⚠️ Failed to load checkpoint: {e}, starting fresh", emoji="⚠️")
            start_eval = 0
    
    if start_eval == 0:
        log_message("🚀 Starting fresh Nevergrad optimization", emoji="🚀")
        
        # Evaluate initial solution
        log_message("🔍 Initial evaluation...", emoji="🔍")
        initial_individual = create_full_individual(initial_vector, optimizable_param_names, all_param_names, fixed_params, integer_params, toolbox)
        best_fitness = evaluator.evaluate(initial_individual)[0]
        best_solution = initial_vector.copy()
        
        evaluation_history.append({
            "eval": 0,
            "fitness": best_fitness,
            "solution": best_solution.copy()
        })
    
    log_message(f"🎯 Starting Nevergrad optimization from evaluation {start_eval}", emoji="🎯")
    
    # Initialize tracking variables
    generation_times = []
    stagnation = 0
    last_improvement_gen = 0
    previous_best_fitness = best_fitness
    evaluations_per_gen = population_size
    
    # Global variables for objective function
    global ng_evaluator, ng_toolbox, ng_optimizable_param_names, ng_all_param_names, ng_fixed_params, ng_integer_params
    
    ng_evaluator = evaluator
    ng_toolbox = toolbox
    ng_optimizable_param_names = optimizable_param_names
    ng_all_param_names = all_param_names
    ng_fixed_params = fixed_params
    ng_integer_params = integer_params
    
    def objective_function(gen):
        """
        Objective function for Nevergrad - evaluates entire generation
        Args:
            gen: Generation number
        Returns:
            gen_evaluations: List of fitness values for the generation
        """
        nonlocal best_fitness, best_solution, last_improvement_gen, evaluation_history
        
        # Ask for batch of candidates for this generation (mass ask for efficiency)
        remaining_budget = budget - (gen - 1) * evaluations_per_gen
        batch_size = min(evaluations_per_gen, remaining_budget)
        
            
        # Use batch asking for better performance
        try:
            candidates = [optimizer.ask() for _ in range(batch_size)]
        except Exception as e:
            # Fallback to individual asks if batch asking fails
            log_message(f"⚠️ Batch asking failed, falling back to individual asks: {e}", emoji="⚠️")
            candidates = []
            for i in range(batch_size):
                candidates.append(optimizer.ask())
        
        # Prepare multiprocessing evaluation with candidate tracking
        pool = Pool(processes=(cpu_count() - 1))
        all_args = []
        
        for idx, candidate in enumerate(candidates):
            # Extract parameter values from Nevergrad candidate
            # For Instrumentation with named parameters, use candidate.value directly
            try:
                # Try accessing as dictionary (for Instrumentation)
                solution_vector = [candidate.value[param_name] for param_name in ng_optimizable_param_names]
            except (TypeError, KeyError):
                # Fallback: try accessing as args/kwargs
                try:
                    candidate_args, candidate_kwargs = candidate.args, candidate.kwargs
                    solution_vector = [candidate_kwargs[param_name] for param_name in ng_optimizable_param_names]
                except:
                    # Last resort: assume it's an array/tuple in parameter order
                    solution_vector = list(candidate.value)
            
            individual = create_full_individual(solution_vector, ng_optimizable_param_names, 
                                              ng_all_param_names, ng_fixed_params, 
                                              ng_integer_params, ng_toolbox)
            # Include the candidate index to maintain pairing
            all_args.append((ng_evaluator, individual, False, idx))
        
        # Evaluate with progress bar
        fitnesses = []
        gen_evaluations = []
        
        with open(WATCH_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            
            with Progress(
                SpinnerColumn(spinner_name="dots12"),
                TextColumn("🎯 [progress.description]{task.description}"),
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
                
                task = progress.add_task(f"🎯 Nevergrad Gen {gen} | Best: {best_fitness:.6e}", total=len(candidates))
                
                current_best_gen = float('inf')
                for result_idx, result in enumerate(pool.imap_unordered(evaluate_solution, all_args)):
                    fitness_val, candidate_idx = result[0], result[1]
                    fitnesses.append(fitness_val)
                    gen_evaluations.append(fitness_val)
                    
                    # Tell optimizer about the result - NOW WITH CORRECT CANDIDATE!
                    optimizer.tell(candidates[candidate_idx], fitness_val)
                    
                    # Track evaluation
                    # Extract parameter values from Nevergrad candidate - NOW WITH CORRECT CANDIDATE!
                    try:
                        # Try accessing as dictionary (for Instrumentation)
                        solution_vector = [candidates[candidate_idx].value[param_name] for param_name in ng_optimizable_param_names]
                    except (TypeError, KeyError):
                        # Fallback: try accessing as args/kwargs
                        try:
                            candidate_args, candidate_kwargs = candidates[candidate_idx].args, candidates[candidate_idx].kwargs
                            solution_vector = [candidate_kwargs[param_name] for param_name in ng_optimizable_param_names]
                        except:
                            # Last resort: assume it's an array/tuple in parameter order
                            solution_vector = list(candidates[candidate_idx].value)
                    eval_num = (gen - 1) * evaluations_per_gen + candidate_idx
                    evaluation_history.append({
                        "eval": eval_num,
                        "fitness": fitness_val,
                        "solution": solution_vector.copy()
                    })
                    
                    # Track best in current generation
                    if fitness_val < current_best_gen:
                        current_best_gen = fitness_val
                    
                    # Update global best
                    if fitness_val < best_fitness:
                        improvement = best_fitness - fitness_val
                        best_fitness = fitness_val
                        best_solution = solution_vector.copy()
                        last_improvement_gen = gen
                        
                        # Log new global best with celebration
                        log_message(f"🎉 NEW GLOBAL BEST! 🎉 Fitness: {best_fitness:.6e} (improved by {improvement:.6e})", emoji="🌟")
                    
                    progress.update(
                        task, 
                        advance=1,
                        description=f"🎯 Nevergrad Gen {gen} | Best: {best_fitness:.6e}"
                    )
            
            pool.close()
            pool.join()
        
        return gen_evaluations
    
    # Main Nevergrad optimization loop
    for gen in range((start_eval // evaluations_per_gen) + 1, ngen + 1):
        gen_start_time = time.time()
        console_wrapper(Rule(f"Generation {gen}", style="bold green"))
        
        # Evaluate entire generation
        gen_evaluations = objective_function(gen)
        
        # Update stagnation tracking
        if best_fitness < previous_best_fitness:
            stagnation = 0
            previous_best_fitness = best_fitness
        else:
            stagnation += 1
        
        # Calculate statistics for this generation
        gen_time = time.time() - gen_start_time
        generation_times.append(gen_time)
        avg_gen_time = sum(generation_times) / len(generation_times)
        
        mean_fitness = np.mean(gen_evaluations) if gen_evaluations else best_fitness
        std_fitness = np.std(gen_evaluations) if len(gen_evaluations) > 1 else 0.0
        
        logbook.append({
            "gen": gen,
            "nevals": len(gen_evaluations),
            "best": best_fitness,
            "mean": mean_fitness,
            "std": std_fitness,
            "time": gen_time
        })
        
        if verbose:
            # Show parameter values for best solution in 2 columns
            param_info = "\n🌐 Best Solution Parameters:"
            param_lines = []
            
            for i, (param_name, value) in enumerate(zip(optimizable_param_names, best_solution)):
                if param_name in integer_params:
                    param_lines.append(f"📊 {param_name}: {int(round(value))}")
                else:
                    param_lines.append(f"📊 {param_name}: {value:.6f}")
            
            # Arrange in 2 columns
            mid_point = (len(param_lines) + 1) // 2
            left_column = param_lines[:mid_point]
            right_column = param_lines[mid_point:]
            
            # Pad right column if needed
            while len(right_column) < len(left_column):
                right_column.append("")
            
            # Create 2-column layout
            for left, right in zip(left_column, right_column):
                if right:  # Both columns have content
                    param_info += f"\n                  {left:<50} {right}"
                else:  # Only left column has content
                    param_info += f"\n                  {left}"
            
            # Stagnation status with emoji
            stagnation_emoji = "🔥" if stagnation == 0 else "😴" if stagnation < 10 else "💤" if stagnation < 50 else "⚰️"
            stagnation_info = f"🔄 Stagnation: {stagnation} gens {stagnation_emoji} (last improvement: gen {last_improvement_gen})"
            
            # Get the actual optimizer that NGOpt selected (if applicable)
            actual_optimizer_name = optimizer_name
            if optimizer_name == "NGOpt" and hasattr(optimizer, '_optim') and optimizer._optim is not None:
                try:
                    # NGOpt stores the selected optimizer in _optim
                    actual_optimizer_name = optimizer._optim.name
                except:
                    # Fallback to original name if we can't determine the selected optimizer
                    actual_optimizer_name = optimizer_name
            
            log_message(
                f"""{CYAN}🌟 Gen {gen}{RESET}
                🎯 Optimizer: {actual_optimizer_name}
                🌍 Best fitness: {best_fitness:.6e}
                📊 Mean fitness: {mean_fitness:.6e}
                📈 Std fitness: {std_fitness:.6e}
                {stagnation_info}
                ⏱️ Generation time: {gen_time:.2f} sec / {(gen_time/60):.2f} min
                📆 Avg gen time: {avg_gen_time:.2f} sec / {(avg_gen_time/60):.2f} min
                🔢 Evaluations: {gen * evaluations_per_gen}/{budget}{param_info}""",
                panel=True, timestamp=False
            )
        
        # Checkpoint saving every generation
        if gen % 1 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "optimizer": optimizer,
                    "num_evaluations": gen * evaluations_per_gen,
                    "best_fitness": best_fitness,
                    "best_solution": best_solution,
                    "logbook": logbook,
                    "evaluation_history": evaluation_history
                }, f)
            log_message(f"💾 Checkpoint saved at generation {gen}", emoji="💾")
    
    # Create final population for return
    final_population = []
    for i in range(len(population)):  # Match original population size
        individual = create_full_individual(best_solution, optimizable_param_names, all_param_names, fixed_params, integer_params, toolbox)
        individual.fitness.values = (best_fitness,)
        final_population.append(individual)
    
    total_time = time.time() - start_time
    log_message(f"🕒 Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)", emoji="🕒")
    log_message(f"🏆 Final best fitness: {best_fitness:.6e}", emoji="🏆")
    
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