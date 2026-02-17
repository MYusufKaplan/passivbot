"""
Main CMA-ES optimizer with automatic restarts.
"""

import os
import time
import json
import logging
import contextlib
import datetime
import numpy as np
from multiprocessing import Pool, cpu_count
from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel
from rich.text import Text
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    TaskProgressColumn,
)

# Import from parent alternatives module
import sys
import os as _os
_parent_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from alternatives import evaluate_solution, create_full_individual, WATCH_PATH

# Import from cma_es modules
from .parameters import _process_parameters_cma_es, denormalize_solution
from .core import generate_population, update_cma_state
from .restart import initialize_restart, check_termination_criteria
from .checkpoint import save_checkpoint, load_checkpoint
from .data_structures import RollingBuffer

# PCA imports for visualization
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

console = Console(
    force_terminal=True,
    no_color=False,
    log_path=False,
    width=191,
    color_system="truecolor",
    legacy_windows=False,
)


def console_wrapper(msg, watch_path):
    """
    Wrapper for Rich console output with file logging.
    Exactly like PSO's console_wrapper method.
    """
    with open(watch_path, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        console.print(msg)


def log_message(message, watch_path, emoji=None, timestamp=True):
    """
    Utility function to print logs with Rich formatting.
    Exactly like PSO's log_message method.
    """
    if timestamp:
        timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        timestamp_str = ""
    emoji_str = f" {emoji}" if emoji else ""
    log_text = f"{timestamp_str}{emoji_str} {message}"
    
    console_wrapper(log_text, watch_path)


def create_pca_visualization(population_positions, best_position, generation, restart_number, watch_path, pca_state=None, sigma=None, grid_width=172, grid_height=31):
    """
    Create PCA-based ASCII visualization of population distribution.
    Similar to PSO's visualization but adapted for CMA-ES.
    
    Parameters
    ----------
    population_positions : np.ndarray
        Current population positions (n_particles, dimensions)
    best_position : np.ndarray
        Current best position (dimensions,)
    generation : int
        Current generation number
    restart_number : int
        Current restart number
    watch_path : str
        Path to write visualization
    pca_state : dict or None
        Dictionary containing PCA state (pca object, previous metrics)
    sigma : float or None
        Current CMA-ES sigma (step size)
    grid_width : int
        Width of ASCII grid (default: 172)
    grid_height : int
        Height of ASCII grid (default: 31)
        
    Returns
    -------
    dict or None
        Updated PCA state dictionary
    """
    if not SKLEARN_AVAILABLE:
        return pca_state
    
    try:
        if population_positions.shape[0] < 2 or population_positions.shape[1] < 2:
            return pca_state
        
        # Initialize or update PCA - fit every generation for accurate visualization
        if pca_state is None:
            pca_state = {'pca': None, 'prev_sigma': None, 'prev_spread': None}
        
        # Always fit PCA with current population for accurate visualization
        pca_state['pca'] = PCA(n_components=2)
        pca_state['pca'].fit(population_positions)
        
        pca = pca_state['pca']
        
        # Transform positions to 2D
        positions_2d = pca.transform(population_positions)
        best_2d = pca.transform(best_position.reshape(1, -1))[0]
        
        # Create density grid
        density_grid = np.zeros((grid_height, grid_width))
        
        # Find bounds
        min_x, max_x = np.min(positions_2d[:, 0]), np.max(positions_2d[:, 0])
        min_y, max_y = np.min(positions_2d[:, 1]), np.max(positions_2d[:, 1])
        
        # Add padding
        padding_x = (max_x - min_x) * 0.1
        padding_y = (max_y - min_y) * 0.1
        min_x -= padding_x
        max_x += padding_x
        min_y -= padding_y
        max_y += padding_y
        
        # Map positions to grid
        if max_x > min_x and max_y > min_y:
            for pos in positions_2d:
                grid_x = int((pos[0] - min_x) / (max_x - min_x) * (grid_width - 1))
                grid_y = int((pos[1] - min_y) / (max_y - min_y) * (grid_height - 1))
                grid_x = max(0, min(grid_x, grid_width - 1))
                grid_y = max(0, min(grid_y, grid_height - 1))
                density_grid[grid_y, grid_x] += 1
            
            # Find best position in grid
            best_grid_x = int((best_2d[0] - min_x) / (max_x - min_x) * (grid_width - 1))
            best_grid_y = int((best_2d[1] - min_y) / (max_y - min_y) * (grid_height - 1))
            best_grid_x = max(0, min(best_grid_x, grid_width - 1))
            best_grid_y = max(0, min(best_grid_y, grid_height - 1))
        else:
            best_grid_x = grid_width // 2
            best_grid_y = grid_height // 2
        
        # Create colored visualization
        max_density = np.max(density_grid) if np.max(density_grid) > 0 else 1
        
        def get_gradient_color(normalized_density):
            """Smooth gradient from purple to red"""
            if normalized_density <= 0.0:
                return (128, 0, 128)
            
            hue = 280 - (normalized_density * 280)
            saturation = 1.0
            value = 1.0
            
            h = hue / 60.0
            c = value * saturation
            x = c * (1 - abs(h % 2 - 1))
            m = value - c
            
            if h < 1:
                r_prime, g_prime, b_prime = c, x, 0
            elif h < 2:
                r_prime, g_prime, b_prime = x, c, 0
            elif h < 3:
                r_prime, g_prime, b_prime = 0, c, x
            elif h < 4:
                r_prime, g_prime, b_prime = 0, x, c
            elif h < 5:
                r_prime, g_prime, b_prime = x, 0, c
            else:
                r_prime, g_prime, b_prime = c, 0, x
            
            r = int((r_prime + m) * 255)
            g = int((g_prime + m) * 255)
            b = int((b_prime + m) * 255)
            
            return (r, g, b)
        
        def get_colored_char(density):
            if density == 0:
                return Text("·", style="dim black")
            
            normalized = density / max_density
            r, g, b = get_gradient_color(normalized)
            color_str = f"rgb({r},{g},{b})"
            
            return Text("▪", style=color_str)
        
        def get_style_for_density(density):
            if density == 0:
                return "dim black"
            
            normalized = density / max_density
            r, g, b = get_gradient_color(normalized)
            
            return f"rgb({r},{g},{b})"
        
        # Build visualization
        full_viz = Text()
        
        # Add header
        header = "    " + "".join([f"{i:2d}" if i % 20 == 0 else "  " for i in range(0, grid_width, 2)])
        full_viz.append(header[:grid_width + 4] + "\n")
        
        # Add grid
        for y in range(grid_height):
            line_text = Text(f"{y:2d} ")
            for x in range(grid_width):
                density = density_grid[y, x]
                
                if x == best_grid_x and y == best_grid_y:
                    style_for_best = get_style_for_density(density)
                    line_text.append("@", style=style_for_best)
                else:
                    colored_char = get_colored_char(density)
                    line_text.append(colored_char)
            
            full_viz.append(line_text)
            full_viz.append("\n")
        
        # Add legend with convergence metrics
        variance_explained = np.sum(pca.explained_variance_ratio_) * 100
        
        # Calculate population spread (bounding box area in PCA space)
        spread_x = max_x - min_x
        spread_y = max_y - min_y
        spread = spread_x * spread_y
        
        # Build legend
        full_viz.append("\n")
        full_viz.append("Legend: ·=empty ▪=density (gradient: purple→blue→cyan→green→yellow→red) @=best (colored by density)\n")
        
        # First line: basic info
        full_viz.append(f"PC1 vs PC2 | Variance: {variance_explained:.1f}% | Population: {len(population_positions)} | Max density: {int(max_density)}\n")
        
        # Second line: convergence metrics with color-coded changes
        convergence_line = Text()
        
        # Sigma metric
        if sigma is not None:
            convergence_line.append(f"Sigma: {sigma:.6f}", style="cyan")
            
            if pca_state.get('prev_sigma') is not None:
                prev_sigma = pca_state['prev_sigma']
                if prev_sigma > 0:
                    sigma_change = ((sigma - prev_sigma) / prev_sigma) * 100
                    if sigma_change < 0:  # Decreasing = converging = good
                        convergence_line.append(f" ({sigma_change:+.1f}%)", style="green")
                    else:  # Increasing = diverging
                        convergence_line.append(f" ({sigma_change:+.1f}%)", style="red")
            
            convergence_line.append(" | ")
        
        # Spread metric
        convergence_line.append(f"Spread: {spread:.2f}", style="cyan")
        
        if pca_state.get('prev_spread') is not None:
            prev_spread = pca_state['prev_spread']
            if prev_spread > 0:
                spread_change = ((spread - prev_spread) / prev_spread) * 100
                if spread_change < 0:  # Decreasing = converging = good
                    convergence_line.append(f" ({spread_change:+.1f}%)", style="green")
                else:  # Increasing = diverging
                    convergence_line.append(f" ({spread_change:+.1f}%)", style="red")
        
        full_viz.append(convergence_line)
        
        # Update state for next generation
        pca_state['prev_sigma'] = sigma
        pca_state['prev_spread'] = spread
        
        # Create panel
        panel = Panel(
            full_viz,
            title=f"🗺️ CMA-ES Population - Restart #{restart_number} Gen {generation}",
            border_style="cyan",
            padding=(1, 2)
        )
        
        # Write to file
        graphs_path = watch_path.replace("evaluation.log", "graphs.log")
        with open(graphs_path, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            console.print(panel)
        
        return pca_state
        
    except Exception as e:
        log_message(f"⚠️ Error creating PCA visualization: {e}", watch_path)
        return pca_state


def cma_es_restarts(
    population,
    toolbox,
    evaluator,
    ngen,  # Ignored - runs indefinitely
    verbose=True,
    parameter_bounds=None,
    checkpoint_path="cma_checkpoint.pkl",
    population_size=200,
    sigma0=0.1,
    max_iter_per_restart=1000,
    tol_hist_fun=1e-12,
    equal_fun_vals_k=None,
    tol_x=1e-11,
    tol_up_sigma=1e20,
    stagnation_iter=100,
    condition_cov=1e14,
    min_sigma=None,
):
    """
    CMA-ES with automatic restarts for indefinite optimization.
    
    This optimizer runs indefinitely, automatically restarting from random
    locations when convergence is detected. It tracks the global best solution
    across all restarts and supports checkpointing for resume capability.
    
    Key Features:
    - Indefinite runtime until user interruption (KeyboardInterrupt)
    - Automatic restarts from random locations when convergence detected
    - Nine termination criteria for robust convergence detection
    - Global best tracking across all restarts
    - Checkpoint save/load for resume capability
    - Multiprocessing for parallel fitness evaluation
    - All optimization in normalized [0, 1] space for uniform parameter treatment
    
    Args:
        population: Initial population (used for interface compatibility, size ignored)
        toolbox: DEAP toolbox with individual creation
        evaluator: Evaluation function
        ngen: Maximum generations (IGNORED - runs indefinitely)
        verbose: Whether to show detailed logging (default: True)
        parameter_bounds: Dictionary of {param_name: (min, max)}
        checkpoint_path: Path to save/load checkpoints (default: "cma_checkpoint.pkl")
        population_size: CMA-ES population size (default: 200)
        sigma0: Initial step size in normalized space (default: 0.1)
        max_iter_per_restart: Maximum iterations per restart (default: 1000)
        tol_hist_fun: Tolerance for fitness history (default: 1e-12)
        equal_fun_vals_k: Number of top solutions to check for equality (default: population_size // 3)
        tol_x: Tolerance for step sizes (default: 1e-11)
        tol_up_sigma: Upper threshold for sigma ratio (default: 1e20)
        stagnation_iter: Iterations for stagnation check (default: 100)
        condition_cov: Condition number threshold (default: 1e14)
        min_sigma: Minimum sigma threshold for convergence (default: None)
        
    Returns:
        Tuple of:
            - final_population: List of individuals with global best solution
            - logbook: Empty list (no logbook accumulation for memory efficiency)
            
    Requirements: 5.1, 6.1, 7.1, 9.1, 9.2
    """
    start_time = time.time()
    
    # ========================================================================
    # Subtask 12.1: Logging Setup
    # ========================================================================
    # Configure logging to logs/evaluation.log in append mode with line buffering
    # Requirements: 8.1, 8.2, 8.7
    
    import logging
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging to file with append mode
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(WATCH_PATH, mode='a'),
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not parameter_bounds:
        raise ValueError("parameter_bounds is required for CMA-ES with restarts")
    
    # Log configuration parameters at start
    # Requirements: 8.1, 8.2, 8.7
    console_wrapper(Rule("[bold cyan]CMA-ES with Automatic Restarts[/bold cyan]"), WATCH_PATH)
    logger.info("=" * 80)
    logger.info("CMA-ES with Automatic Restarts - Configuration")
    logger.info("=" * 80)
    logger.info(f"Population size: {population_size}")
    logger.info(f"Initial sigma (σ₀): {sigma0}")
    logger.info(f"Max iterations per restart: {max_iter_per_restart}")
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"Termination criteria:")
    logger.info(f"  - TolHistFun: {tol_hist_fun}")
    logger.info(f"  - EqualFunValsK: {equal_fun_vals_k if equal_fun_vals_k else 'population_size // 3'}")
    logger.info(f"  - TolX: {tol_x}")
    logger.info(f"  - TolUpSigma: {tol_up_sigma}")
    logger.info(f"  - Stagnation iterations: {stagnation_iter}")
    logger.info(f"  - ConditionCov: {condition_cov}")
    logger.info(f"Verbose mode: {verbose}")
    logger.info("=" * 80)
    
    console_wrapper(f"[cyan]Population size: {population_size}[/cyan]", WATCH_PATH)
    console_wrapper(f"[cyan]Initial sigma (σ₀): {sigma0}[/cyan]", WATCH_PATH)
    console_wrapper(f"[cyan]Max iterations per restart: {max_iter_per_restart}[/cyan]", WATCH_PATH)
    console_wrapper(f"[cyan]Checkpoint path: {checkpoint_path}[/cyan]", WATCH_PATH)
    
    # Process parameters
    try:
        (
            all_param_names,
            optimizable_bounds,
            fixed_params,
            integer_params,
            lower_bounds,
            upper_bounds,
        ) = _process_parameters_cma_es(parameter_bounds)
    except ValueError as e:
        console_wrapper(f"[red]❌ Parameter validation failed: {e}[/red]", WATCH_PATH)
        logger.error(f"Parameter validation failed: {e}")
        raise
    
    optimizable_param_names = list(optimizable_bounds.keys())
    n_dimensions = len(optimizable_param_names)
    
    # Get integer parameter indices
    integer_params_indices = [
        i for i, name in enumerate(optimizable_param_names)
        if name in integer_params
    ]
    
    logger.info(f"Parameter processing complete:")
    logger.info(f"  - {n_dimensions} optimizable parameters")
    logger.info(f"  - {len(fixed_params)} fixed parameters")
    logger.info(f"  - {len(integer_params)} integer parameters")
    
    console_wrapper(
        f"[cyan]📊 {n_dimensions} optimizable parameters, "
        f"{len(fixed_params)} fixed, {len(integer_params)} integer[/cyan]",
        WATCH_PATH
    )
    
    # Load initial values from optimize.json for first restart
    initial_values = {}
    try:
        with open("configs/optimize.json", "r") as f:
            config = json.load(f)
            long_config = config.get("bot", {}).get("long", {})
            for param_name in optimizable_param_names:
                clean_name = param_name.replace("long_", "")
                if clean_name in long_config:
                    value = long_config[clean_name]
                    # Only use numeric values
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        initial_values[param_name] = float(value)
        logger.info(f"Loaded {len(initial_values)} initial values from optimize.json")
        console_wrapper(
            f"[cyan]📋 Loaded {len(initial_values)} initial values from optimize.json[/cyan]",
            WATCH_PATH
        )
    except Exception as e:
        logger.warning(f"Could not load optimize.json: {e}")
        console_wrapper(
            f"[yellow]⚠️ Could not load optimize.json: {e}[/yellow]",
            WATCH_PATH
        )
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Initialize global best tracking
    if checkpoint:
        restart_number = checkpoint.restart_number + 1
        global_best_solution = checkpoint.global_best_solution
        global_best_fitness = checkpoint.global_best_fitness
        logger.info(f"Resuming from restart #{restart_number}, global best: {global_best_fitness:.6e}")
        console_wrapper(
            f"[green]🔄 Resuming from restart #{restart_number}[/green]",
            WATCH_PATH
        )
    else:
        restart_number = 1
        global_best_solution = None
        global_best_fitness = float('inf')
        logger.info("Starting fresh optimization")
        console_wrapper(
            f"[cyan]🆕 Starting fresh optimization[/cyan]",
            WATCH_PATH
        )
    
    # Track global improvement across all restarts
    gens_since_global_improvement = 0
    
    # Create multiprocessing pool
    num_workers = max(1, cpu_count() - 1)
    logger.info(f"Creating multiprocessing pool with {num_workers} workers")
    console_wrapper(f"[cyan]🔧 Creating multiprocessing pool with {num_workers} workers[/cyan]", WATCH_PATH)
    pool = Pool(processes=num_workers)
    
    # Default equal_fun_vals_k if not provided
    if equal_fun_vals_k is None:
        equal_fun_vals_k = population_size // 3
    
    try:
        # Run indefinite restart loop until KeyboardInterrupt
        while True:
            # ====================================================================
            # Subtask 12.2: Restart Logging
            # ====================================================================
            # Log restart number and starting centroid
            # Requirements: 2.6, 3.10, 8.3, 8.5
            
            # Initialize restart
            logger.info("=" * 80)
            logger.info(f"Starting Restart #{restart_number}")
            logger.info("=" * 80)
            console_wrapper(Rule(f"[bold yellow]Restart #{restart_number}[/bold yellow]"), WATCH_PATH)
            
            is_first_restart = (restart_number == 1 and checkpoint is None)
            
            initial_centroid, state = initialize_restart(
                n_dimensions=n_dimensions,
                sigma0=sigma0,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                optimizable_param_names=optimizable_param_names,
                initial_values=initial_values if is_first_restart else None,
                is_first_restart=is_first_restart,
            )
            
            # Log restart start with centroid location
            centroid_denorm = denormalize_solution(
                initial_centroid, lower_bounds, upper_bounds, integer_params_indices
            )
            
            # Log first 5 parameters of centroid
            centroid_preview = centroid_denorm[:min(5, len(centroid_denorm))].tolist()
            param_names_preview = optimizable_param_names[:min(5, len(optimizable_param_names))]
            
            logger.info(f"Restart #{restart_number} starting centroid (first {len(centroid_preview)} params):")
            for param_name, value in zip(param_names_preview, centroid_preview):
                logger.info(f"  {param_name}: {value:.6f}")
            
            console_wrapper(
                f"[cyan]📍 Starting centroid (first 3 params): "
                f"{centroid_denorm[:min(3, len(centroid_denorm))].tolist()}...[/cyan]",
                WATCH_PATH
            )
            
            # Initialize fitness history for this restart
            fitness_history = RollingBuffer(
                max_size=max(
                    10 + int(np.ceil(30.0 * n_dimensions / population_size)),
                    stagnation_iter
                )
            )
            
            restart_best_fitness = float('inf')
            restart_best_solution = None
            
            # Track improvement for detailed reporting
            gens_since_restart_improvement = 0
            total_evaluations = 0
            
            # Initialize PCA state for visualization
            pca_state = None
            
            # Run generation loop until termination criterion
            generation = 0
            while generation < max_iter_per_restart:
                gen_start_time = time.time()
                
                # Generate population in normalized space
                population_normalized = generate_population(state, population_size)
                
                # Denormalize solutions for evaluation
                population_denormalized = np.array([
                    denormalize_solution(
                        ind, lower_bounds, upper_bounds, integer_params_indices
                    )
                    for ind in population_normalized
                ])
                
                # Create full individuals
                individuals = [
                    create_full_individual(
                        sol, optimizable_param_names, all_param_names,
                        fixed_params, integer_params, toolbox
                    )
                    for sol in population_denormalized
                ]
                
                # Evaluate in parallel with error handling
                all_args = [(evaluator, ind, False) for ind in individuals]
                
                fitness_values = []
                bankrupt_count = 0
                non_bankrupt_count = 0
                gen_best_fitness = float('inf')
                
                if verbose:
                    # Redirect progress bar output to log file
                    with open(WATCH_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        with Progress(
                            SpinnerColumn(spinner_name="dots12"),
                            TextColumn("[progress.description]{task.description}"),
                            BarColumn(bar_width=None),
                            "•",
                            TaskProgressColumn(
                                text_format="[progress.percentage]{task.percentage:>5.1f}%",
                                show_speed=True,
                            ),
                            "•",
                            TimeElapsedColumn(),
                            "•",
                            TimeRemainingColumn(),
                            "•",
                            console=console,
                            transient=True,
                        ) as progress:
                            task = progress.add_task(
                                f"Gen {generation} | Best: {gen_best_fitness:.6e} | Global: {global_best_fitness:.6e}",
                                total=population_size,
                            )
                            
                            for result in pool.imap_unordered(evaluate_solution, all_args):
                                try:
                                    # Handle both 2-tuple and 3-tuple returns
                                    if len(result) == 3:
                                        fitness_val, _, bankrupt = result
                                    else:
                                        fitness_val = result[0]
                                        bankrupt = False
                                    
                                    # Validate fitness value
                                    if not isinstance(fitness_val, (int, float)) or np.isnan(fitness_val) or np.isinf(fitness_val):
                                        raise ValueError(f"Invalid fitness value: {fitness_val}")
                                    
                                    fitness_values.append(fitness_val)
                                    
                                    if bankrupt:
                                        bankrupt_count += 1
                                    else:
                                        non_bankrupt_count += 1
                                    
                                    if fitness_val < gen_best_fitness:
                                        gen_best_fitness = fitness_val
                                    
                                except Exception as e:
                                    # Evaluation failed - assign worst fitness and continue
                                    worst_fitness = 1e10
                                    fitness_values.append(worst_fitness)
                                    non_bankrupt_count += 1  # Count as non-bankrupt to avoid confusion
                                    
                                    # Log error inside the file redirect context
                                    console.print(f"[yellow]⚠️ Evaluation failed for solution: {e}[/yellow]")
                                    console.print(f"[yellow]   Assigning worst fitness ({worst_fitness:.6e}) and continuing[/yellow]")
                                
                                push_bar = f"✅{non_bankrupt_count} 💀{bankrupt_count}"
                                emoji = "🔥" if gen_best_fitness < global_best_fitness else "🔄"
                                
                                progress.update(
                                    task,
                                    advance=1,
                                    description=f"{emoji} {push_bar} | Gen {generation} Best: {gen_best_fitness:.6e} | Global: {global_best_fitness:.6e}",
                                )
                else:
                    # No progress bar - just log to file
                    with open(WATCH_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        for result in pool.imap_unordered(evaluate_solution, all_args):
                            try:
                                if len(result) == 3:
                                    fitness_val, _, _ = result
                                else:
                                    fitness_val = result[0]
                                
                                # Validate fitness value
                                if not isinstance(fitness_val, (int, float)) or np.isnan(fitness_val) or np.isinf(fitness_val):
                                    raise ValueError(f"Invalid fitness value: {fitness_val}")
                                
                                fitness_values.append(fitness_val)
                                
                            except Exception as e:
                                # Evaluation failed - assign worst fitness and continue
                                worst_fitness = 1e10
                                fitness_values.append(worst_fitness)
                                
                                console.print(f"[yellow]⚠️ Evaluation failed for solution: {e}[/yellow]")
                                console.print(f"[yellow]   Assigning worst fitness ({worst_fitness:.6e}) and continuing[/yellow]")
                
                fitness_values = np.array(fitness_values)
                total_evaluations += len(fitness_values)
                
                # Update CMA-ES state in normalized space
                state = update_cma_state(
                    state, population_normalized, fitness_values, population_size
                )
                
                # Track best in this generation
                best_idx = np.argmin(fitness_values)
                gen_best_fitness = fitness_values[best_idx]
                gen_best_solution = population_denormalized[best_idx]
                
                # Update restart best
                restart_improved = False
                if gen_best_fitness < restart_best_fitness:
                    restart_best_fitness = gen_best_fitness
                    restart_best_solution = gen_best_solution.copy()
                    gens_since_restart_improvement = 0
                    restart_improved = True
                else:
                    gens_since_restart_improvement += 1
                
                # Update global best
                global_improved = False
                if gen_best_fitness < global_best_fitness:
                    global_best_fitness = gen_best_fitness
                    global_best_solution = gen_best_solution.copy()
                    gens_since_global_improvement = 0
                    global_improved = True
                    
                    # Log global best improvements
                    # Requirements: 8.4, 8.6, 8.8
                    logger.info(f"🔥 NEW GLOBAL BEST! Restart #{restart_number}, Gen {generation}, Fitness: {global_best_fitness:.6e}")
                    
                    console_wrapper(
                        f"[bold green]🔥 New global best! Fitness: {global_best_fitness:.6e}[/bold green]",
                        WATCH_PATH
                    )
                else:
                    gens_since_global_improvement += 1
                
                # Add to fitness history
                fitness_history.append(gen_best_fitness)
                
                # Log generation details
                # Requirements: 8.4, 8.6, 8.8
                gen_time = time.time() - gen_start_time
                
                # Calculate statistics
                worst_fitness = np.max(fitness_values)
                median_fitness = np.median(fitness_values)
                fitness_range = worst_fitness - gen_best_fitness
                
                # Log to file with detailed statistics
                logger.info(
                    f"Restart #{restart_number}, Gen {generation}: "
                    f"Evals={population_size}, "
                    f"Best={gen_best_fitness:.6e}, "
                    f"Mean={np.mean(fitness_values):.6e}, "
                    f"Std={np.std(fitness_values):.6e}, "
                    f"Sigma={state.sigma:.6e}, "
                    f"Time={gen_time:.2f}s"
                )
                
                # Detailed console output with emojis and colors
                if verbose:
                    # Determine status emoji
                    if global_improved:
                        status_emoji = "🔥"
                        status_color = "bold green"
                        status_text = "GLOBAL BEST IMPROVED"
                    elif restart_improved:
                        status_emoji = "⭐"
                        status_color = "green"
                        status_text = "RESTART BEST IMPROVED"
                    else:
                        status_emoji = "🔄"
                        status_color = "white"
                        status_text = "NO IMPROVEMENT"
                    
                    # Build detailed report with fixed-width formatting
                    report_lines = []
                    
                    # Header
                    report_lines.append(f"[{status_color}]{status_emoji} {status_text}[/{status_color}]")
                    report_lines.append("")
                    
                    # Fitness statistics
                    report_lines.append("[yellow]📊 FITNESS STATISTICS[/yellow]")
                    report_lines.append(f"  Best      : {gen_best_fitness:.6e}")
                    report_lines.append(f"  Mean      : {np.mean(fitness_values):.6e}")
                    report_lines.append(f"  Median    : {median_fitness:.6e}")
                    report_lines.append(f"  Worst     : {worst_fitness:.6e}")
                    report_lines.append(f"  Std Dev   : {np.std(fitness_values):.6e}")
                    report_lines.append(f"  Range     : {fitness_range:.6e}")
                    report_lines.append("")
                    
                    # Progress tracking
                    report_lines.append("[cyan]🎯 PROGRESS TRACKING[/cyan]")
                    report_lines.append(f"  Restart Best : {restart_best_fitness:.6e}")
                    report_lines.append(f"  Global Best  : {global_best_fitness:.6e}")
                    report_lines.append("")
                    
                    # Improvement tracking
                    report_lines.append("[blue]🔍 IMPROVEMENT STATUS[/blue]")
                    if gens_since_restart_improvement == 0:
                        report_lines.append(f"  Restart      : [green]✨ Improved this generation![/green]")
                    else:
                        report_lines.append(f"  Restart      : [dim]{gens_since_restart_improvement} generations since improvement[/dim]")
                    
                    if gens_since_global_improvement == 0:
                        report_lines.append(f"  Global       : [bold green]🌟 Improved this generation![/bold green]")
                    else:
                        report_lines.append(f"  Global       : [dim]{gens_since_global_improvement} generations since improvement[/dim]")
                    report_lines.append("")
                    
                    # CMA-ES parameters
                    report_lines.append("[white]⚙️  CMA-ES PARAMETERS[/white]")
                    report_lines.append(f"  Sigma (σ)    : {state.sigma:.6e}")
                    report_lines.append(f"  Evaluations  : {total_evaluations:,}")
                    report_lines.append(f"  Gen Time     : {gen_time:.2f}s")
                    
                    # Add bankruptcy info if any
                    if bankrupt_count > 0:
                        report_lines.append("")
                        report_lines.append("[red]💀 BANKRUPTCIES[/red]")
                        report_lines.append(f"  Count        : {bankrupt_count}/{population_size} ({100*bankrupt_count/population_size:.1f}%)")
                    
                    # Create panel
                    panel = Panel(
                        "\n".join(report_lines),
                        title=f"Restart #{restart_number} | Generation {generation}",
                        border_style=status_color,
                        padding=(1, 2)
                    )
                    
                    console_wrapper(panel, WATCH_PATH)
                
                # Check termination criteria
                should_terminate, triggered_conditions = check_termination_criteria(
                    state=state,
                    fitness_history=fitness_history.to_list(),
                    population_fitness=fitness_values,
                    max_iter_per_restart=max_iter_per_restart,
                    tol_hist_fun=tol_hist_fun,
                    equal_fun_vals_k=equal_fun_vals_k,
                    tol_x=tol_x,
                    tol_up_sigma=tol_up_sigma,
                    stagnation_iter=stagnation_iter,
                    condition_cov=condition_cov,
                    sigma0=sigma0,
                    gens_since_restart_improvement=gens_since_restart_improvement,
                    min_sigma=min_sigma,
                )
                
                if should_terminate:
                    # Log termination conditions
                    # Requirements: 2.6, 3.10, 8.3, 8.5
                    logger.info(f"Restart #{restart_number} terminated after {generation + 1} generations")
                    logger.info(f"Triggered termination conditions: {', '.join(triggered_conditions)}")
                    logger.info(f"Restart best fitness: {restart_best_fitness:.6e}")
                    logger.info(f"Global best fitness: {global_best_fitness:.6e}")
                    logger.info(f"Generations since restart improvement: {gens_since_restart_improvement}")
                    logger.info(f"Generations since global improvement: {gens_since_global_improvement}")
                    
                    # Enhanced termination report
                    term_lines = []
                    term_lines.append(f"[bold yellow]🛑 Restart #{restart_number} Terminated[/bold yellow]")
                    term_lines.append(f"  [white]📊 Restart Generations:[/white] {generation + 1}")
                    term_lines.append(f"  [white]🎯 Restart Best:[/white] {restart_best_fitness:.6e}")
                    term_lines.append(f"  [white]🌟 Global Best:[/white] {global_best_fitness:.6e}")
                    term_lines.append(f"  [white]🔄 Gens Since Restart Improvement:[/white] {gens_since_restart_improvement}")
                    term_lines.append(f"  [white]🌍 Gens Since Global Improvement:[/white] {gens_since_global_improvement}")
                    term_lines.append(f"  [yellow]⚠️  Termination Criteria:[/yellow]")
                    for condition in triggered_conditions:
                        # Add clarification for Stagnation criterion
                        if condition == "Stagnation":
                            term_lines.append(f"     • {condition} (based on restart fitness history)")
                        else:
                            term_lines.append(f"     • {condition}")
                    
                    console_wrapper("\n".join(term_lines), WATCH_PATH)
                    break
                
                # Silently evaluate global best solution every generation
                # This keeps the global best "fresh" without displaying results
                if global_best_solution is not None:
                    try:
                        global_best_individual = create_full_individual(
                            global_best_solution,
                            optimizable_param_names,
                            all_param_names,
                            fixed_params,
                            integer_params,
                            toolbox,
                        )
                        # Evaluate silently (results not used, just for side effects like file updates)
                        _ = evaluate_solution((evaluator, global_best_individual, False))
                    except Exception as e:
                        # Silently ignore evaluation errors for global best
                        pass
                
                # Save checkpoint after each generation
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    restart_number=restart_number,
                    global_best_solution=global_best_solution,
                    global_best_fitness=global_best_fitness,
                    logbook=[],  # No logbook accumulation for memory efficiency
                )
                
                # Create PCA visualization
                if SKLEARN_AVAILABLE:
                    pca_state = create_pca_visualization(
                        population_positions=population_denormalized,
                        best_position=global_best_solution if global_best_solution is not None else restart_best_solution,
                        generation=generation,
                        restart_number=restart_number,
                        watch_path=WATCH_PATH,
                        pca_state=pca_state,
                        sigma=state.sigma,
                    )
                
                generation += 1
            
            # Log restart completion
            logger.info(f"Restart #{restart_number} completed")
            console_wrapper(f"[dim]✅ Restart #{restart_number} completed[/dim]", WATCH_PATH)
            
            # Increment restart counter
            restart_number += 1
            
    except KeyboardInterrupt:
        logger.info("=" * 80)
        logger.info("Optimization interrupted by user (KeyboardInterrupt)")
        logger.info("=" * 80)
        console_wrapper("\n[bold red]🛑 Optimization interrupted by user[/bold red]", WATCH_PATH)
        
        # Save final checkpoint
        if global_best_solution is not None:
            logger.info("Saving final checkpoint before exit")
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                restart_number=restart_number,
                global_best_solution=global_best_solution,
                global_best_fitness=global_best_fitness,
                logbook=[],
            )
            logger.info("Final checkpoint saved successfully")
        
        console_wrapper("[green]💾 Final checkpoint saved[/green]", WATCH_PATH)
    
    finally:
        # Close and join multiprocessing pool
        logger.info("Closing multiprocessing pool...")
        console_wrapper("[cyan]🔧 Closing multiprocessing pool...[/cyan]", WATCH_PATH)
        pool.close()
        pool.join()
        logger.info("Multiprocessing pool closed successfully")
        console_wrapper("[green]✅ Pool closed successfully[/green]", WATCH_PATH)
    
    # Create final population with global best solution
    final_population = []
    if global_best_solution is not None:
        for _ in range(len(population)):
            individual = create_full_individual(
                global_best_solution,
                optimizable_param_names,
                all_param_names,
                fixed_params,
                integer_params,
                toolbox,
            )
            individual.fitness.values = (global_best_fitness,)
            final_population.append(individual)
    
    total_time = time.time() - start_time
    
    # Log final summary
    logger.info("=" * 80)
    logger.info("Optimization Complete - Final Summary")
    logger.info("=" * 80)
    logger.info(f"Total optimization time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")
    logger.info(f"Final global best fitness: {global_best_fitness:.6e}")
    logger.info(f"Total restarts completed: {restart_number - 1}")
    logger.info("=" * 80)
    
    console_wrapper(
        f"[bold green]🕒 Total optimization time: {total_time:.2f}s "
        f"({total_time / 60:.2f} minutes)[/bold green]",
        WATCH_PATH
    )
    console_wrapper(
        f"[bold green]🏆 Final global best fitness: {global_best_fitness:.6e}[/bold green]",
        WATCH_PATH
    )
    
    # Return final population and empty logbook (for interface compatibility)
    return final_population, []
