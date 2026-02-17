"""
Main DEAP evolutionary algorithm optimizer class.

This module provides the DEAPEvolutionaryAlgorithm class that encapsulates
all DEAP functionality following the PSO implementation pattern.
"""

import os
import time
import pickle
import logging
import contextlib
import datetime
import random
import numpy as np
from multiprocessing import Pool, cpu_count
from deap import tools, creator

# Rich imports for beautiful logging
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text
    from rich.progress import (
        Progress,
        BarColumn,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# PCA imports for visualization
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Module-level function for multiprocessing (must be picklable)
def _evaluate_solution_worker(args):
    """
    Evaluate a single solution - module level for pickling.
    
    This function is defined at module level so it can be pickled
    for multiprocessing.
    
    Parameters
    ----------
    args : tuple
        (evaluator, individual, showMe, idx)
    
    Returns
    -------
    tuple
        (fitness_tuple, index, bankrupt_flag)
    """
    evaluator, ind, showMe, idx = args
    
    # For testing, we may not have the log file infrastructure
    # Just evaluate without logging to avoid permission issues in tests
    try:
        log_path = "logs/evaluation_output.log"
        # Check if logs directory exists
        if os.path.exists("logs"):
            with open(log_path, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                fitness = evaluator.evaluate(ind)
        else:
            # No logs directory, evaluate without logging
            fitness = evaluator.evaluate(ind)
    except (PermissionError, OSError):
        # If we can't write to log file, just evaluate without logging
        fitness = evaluator.evaluate(ind)
    
    # Handle both 2-tuple and 3-tuple returns from evaluator
    # Real evaluator returns (fitness_value, fitness_value, bankrupt_flag)
    # Test evaluator may return (fitness_value, fitness_value)
    if len(fitness) >= 3:
        # Real evaluator: (fitness_val, fitness_val, bankrupt)
        # Return the full fitness tuple (both values) for multi-objective
        return (fitness[0], fitness[1]), idx, fitness[2]
    else:
        # Test evaluator: (fitness_val, fitness_val)
        # Return the full fitness tuple
        return fitness, idx, False

def _evaluate_interval_worker(args):
    """
    Evaluate a single (candidate, month) pair for interval-based evaluation.

    This function is defined at module level so it can be pickled
    for multiprocessing.

    Parameters
    ----------
    args : tuple
        (evaluator, individual, month_id, interval, candidate_id)
        interval is a MonthlyInterval containing shared_memory_files, total_timesteps

    Returns
    -------
    tuple
        (candidate_id, month_id, fitness_value, bankruptcy_flag, bankruptcy_reason, bankruptcy_timestep, total_timesteps, gain)
        bankruptcy_reason: 0=none, 1=financial, 2=drawdown, 3=no_positions, 4=stale_position

    Requirements: 4.1, 4.2, 4.3
    """
    evaluator, individual, month_id, interval, candidate_id = args

    # Save original evaluator state
    orig_files = evaluator.shared_memory_files
    orig_shapes = evaluator.hlcvs_shapes
    orig_dtypes = evaluator.hlcvs_dtypes
    orig_btc_files = evaluator.btc_usd_shared_memory_files
    orig_btc_dtypes = evaluator.btc_usd_dtypes
    orig_mmap_contexts = evaluator.mmap_contexts
    orig_shared_hlcvs_np = evaluator.shared_hlcvs_np

    try:
        # Override with interval data
        evaluator.shared_memory_files = {
            ex: info[0] for ex, info in interval.shared_memory_files.items()
        }
        evaluator.hlcvs_shapes = {
            ex: info[2] for ex, info in interval.shared_memory_files.items()
        }
        evaluator.hlcvs_dtypes = {
            ex: info[3] for ex, info in interval.shared_memory_files.items()
        }
        evaluator.btc_usd_shared_memory_files = {
            ex: info[1] for ex, info in interval.shared_memory_files.items()
        }
        evaluator.btc_usd_dtypes = {
            ex: info[4] for ex, info in interval.shared_memory_files.items()
        }

        # Re-enter mmap contexts for the interval files
        # Import managed_mmap from optimize module
        from optimize import managed_mmap

        evaluator.mmap_contexts = {}
        evaluator.shared_hlcvs_np = {}
        for exchange in evaluator.exchanges:
            evaluator.mmap_contexts[exchange] = managed_mmap(
                evaluator.shared_memory_files[exchange],
                evaluator.hlcvs_dtypes[exchange],
                evaluator.hlcvs_shapes[exchange],
            )
            evaluator.shared_hlcvs_np[exchange] = evaluator.mmap_contexts[exchange].__enter__()

        # Evaluate the individual - suppress output for interval mode (per-month)
        # Output will be printed at candidate level after aggregation
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                fitness = evaluator.evaluate(individual)

        # Extract fitness value and bankruptcy info
        # Real evaluator returns (fitness_value, fitness_value, bankrupt_flag)
        # Test evaluator may return (fitness_value, fitness_value)
        if len(fitness) >= 3:
            fitness_value = fitness[0]
            bankruptcy_flag = fitness[2]
        else:
            fitness_value = fitness[0]
            bankruptcy_flag = False

        # Extract gain and bankruptcy_reason from the evaluator's last analysis
        gain = 1.0  # Default if not available
        bankruptcy_reason = 0  # 0=none
        if hasattr(evaluator, 'last_analyses_combined') and evaluator.last_analyses_combined:
            gain = evaluator.last_analyses_combined.get('gain_mean', 1.0)
            # bankruptcy_reason_mean comes from combine_analyses
            br = evaluator.last_analyses_combined.get('bankruptcy_reason_mean', 0)
            bankruptcy_reason = int(br) if br is not None else 0

        # For bankruptcy_timestep, we need to infer it from the evaluator's analysis
        # The evaluator doesn't directly return bankruptcy_timestep, so we use 0 for now
        # In a real scenario, this would need to be extracted from the Rust backtest result
        bankruptcy_timestep = 0 if bankruptcy_flag else interval.total_timesteps

        return (candidate_id, month_id, fitness_value, bankruptcy_flag, bankruptcy_reason, bankruptcy_timestep, interval.total_timesteps, gain)

    finally:
        # Exit mmap contexts for interval files
        for exchange in evaluator.exchanges:
            if exchange in evaluator.mmap_contexts:
                try:
                    evaluator.mmap_contexts[exchange].__exit__(None, None, None)
                except Exception:
                    pass

        # Restore original evaluator state
        evaluator.shared_memory_files = orig_files
        evaluator.hlcvs_shapes = orig_shapes
        evaluator.hlcvs_dtypes = orig_dtypes
        evaluator.btc_usd_shared_memory_files = orig_btc_files
        evaluator.btc_usd_dtypes = orig_btc_dtypes
        evaluator.mmap_contexts = orig_mmap_contexts
        evaluator.shared_hlcvs_np = orig_shared_hlcvs_np




class DEAPEvolutionaryAlgorithm:
    """
    DEAP-based evolutionary algorithm optimizer.
    
    This class provides a clean interface for DEAP evolutionary algorithms,
    following the same pattern as the PSO GlobalBestPSO implementation.
    
    Attributes
    ----------
    n_individuals : int
        Number of individuals in the population
    dimensions : int
        Number of dimensions in the optimization space
    bounds : tuple of numpy.ndarray
        Tuple of (lower_bounds, upper_bounds) arrays
    options : dict
        Dictionary containing algorithm options:
        - 'cxpb': Crossover probability (default: 0.5)
        - 'mutpb': Mutation probability (default: 0.2)
        - 'mu': Number of individuals to select for next generation
        - 'lambda_': Number of offspring to produce
    
    Examples
    --------
    >>> bounds = (np.array([0, 0]), np.array([1, 1]))
    >>> options = {'cxpb': 0.5, 'mutpb': 0.2}
    >>> optimizer = DEAPEvolutionaryAlgorithm(
    ...     n_individuals=50,
    ...     dimensions=2,
    ...     bounds=bounds,
    ...     options=options
    ... )
    >>> optimizer.set_checkpoint_config('deap_checkpoint.pkl', checkpoint_interval=10)
    >>> best_cost, best_pos = optimizer.optimize(objective_func, ngen=100)
    
    Requirements: 1.3, 5.1, 5.3
    """
    
    def __init__(self, n_individuals, dimensions, bounds, options=None):
        """
        Initialize the DEAP evolutionary algorithm optimizer.
        
        Parameters
        ----------
        n_individuals : int
            Number of individuals in the population
        dimensions : int
            Number of dimensions in the optimization space
        bounds : tuple of numpy.ndarray
            Tuple of (lower_bounds, upper_bounds) arrays
        options : dict, optional
            Dictionary containing algorithm options:
            - 'cxpb': Crossover probability (default: 0.5)
            - 'mutpb': Mutation probability (default: 0.2)
            - 'mu': Number of individuals to select (default: n_individuals)
            - 'lambda_': Number of offspring (default: n_individuals)
        
        Requirements: 1.3, 5.1
        """
        self.n_individuals = n_individuals
        self.dimensions = dimensions
        self.bounds = bounds
        
        # Set default options
        if options is None:
            options = {}
        
        self.options = {
            'cxpb': options.get('cxpb', 0.5),
            'mutpb': options.get('mutpb', 0.2),
            'mu': options.get('mu', n_individuals),
            'lambda_': options.get('lambda_', n_individuals),
        }
        
        # Initialize Rich console if available
        if RICH_AVAILABLE:
            self.console = Console(
                force_terminal=True,
                no_color=False,
                log_path=False,
                width=191,
                color_system="truecolor",
                legacy_windows=False
            )
        else:
            self.console = None
        
        # Checkpoint and logging attributes
        self.checkpoint_path = None
        self.checkpoint_interval = 1
        self.watch_path = "logs/evaluation.log"
        self.graphs_path = "logs/graphs.log"
        
        # History data tracking
        self.history_data_path = None
        self.parameter_names = None
        self.detailed_history = {
            'generation': [],
            'population_positions': [],
            'fitness_scores': [],
            'best_individual': [],
            'best_fitness': [],
            'statistics': [],
            'parameter_names': None,
            'generation_times': [],
            'improvement_rates': [],
            'convergence_metrics': [],
            'diversity_metrics': []
        }
        
        # Logging configuration
        self.console_logging = True
        self.file_logging = True
        self.log_path = None
        
        # Generation tracking
        self.generation_times = []
        self.start_time = None
        
        # Best solution tracking
        self.best_individual = None
        self.best_fitness = float('inf')
        
        # Global best tracking (separate from population)
        self.global_best_individual = None
        self.global_best_fitness = float('inf')
        self.stagnation_counter = 0
        
        # Multiprocessing pool
        self.pool = None
        
    def set_checkpoint_config(self, checkpoint_path="deap_checkpoint.pkl", checkpoint_interval=5):
        """
        Configure checkpoint settings.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to save checkpoint files
        checkpoint_interval : int
            Number of generations between checkpoints
        
        Requirements: 3.3
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.log_message(f"💾 Checkpoint configured: {checkpoint_path} (interval: {checkpoint_interval})")
    
    def set_history_config(self, history_data_path="deap_history_data.pkl", parameter_names=None):
        """
        Configure history data collection for analysis.
        
        Parameters
        ----------
        history_data_path : str
            Path to save detailed history data
        parameter_names : list of str, optional
            Names of the parameters being optimized
        
        Requirements: 4.4
        """
        self.history_data_path = history_data_path
        self.parameter_names = parameter_names
        if parameter_names:
            self.detailed_history['parameter_names'] = parameter_names
        self.log_message(f"📈 History data will be saved to: {history_data_path}")
    
    def set_logging_config(self, console_logging=True, file_logging=True, log_path=None):
        """
        Configure logging settings.
        
        Parameters
        ----------
        console_logging : bool
            Enable console output
        file_logging : bool
            Enable file logging
        log_path : str, optional
            Custom log file path (default: logs/evaluation.log)
        
        Requirements: 2.3, 7.1
        """
        self.console_logging = console_logging
        self.file_logging = file_logging
        if log_path:
            self.log_path = log_path
            self.watch_path = log_path
        self.log_message(f"📝 Logging configured (console: {console_logging}, file: {file_logging})")
    
    def optimize(self, objective_function, ngen, verbose=True, toolbox=None, 
                 stats=None, halloffame=None, evaluator=None, parameter_bounds=None):
        """
        Run the evolutionary algorithm optimization.
        
        Parameters
        ----------
        objective_function : callable
            Function to minimize, takes individual as input
        ngen : int
            Number of generations to run
        verbose : bool
            Whether to show detailed progress
        toolbox : deap.base.Toolbox, optional
            DEAP toolbox with registered operators. If None, must be provided
            by caller.
        stats : deap.tools.Statistics, optional
            Statistics object for tracking evolution metrics
        halloffame : deap.tools.HallOfFame, optional
            Hall of fame to track best individuals
        evaluator : Evaluator, optional
            Evaluator instance for fitness evaluation (for integration with
            existing infrastructure)
        parameter_bounds : dict, optional
            Dictionary of parameter bounds for integration with existing
            infrastructure
        
        Returns
        -------
        tuple
            (best_cost, best_position) - Best fitness value and position found
        
        Requirements: 1.3, 5.1, 6.1
        """
        self.start_time = time.time()
        
        # Log optimization start
        self.log_message("🚀 Starting DEAP evolutionary algorithm optimization", panel=True)
        self.log_message(f"Population size: {self.n_individuals}")
        self.log_message(f"Dimensions: {self.dimensions}")
        self.log_message(f"Generations: {ngen}")
        self.log_message(f"Crossover probability: {self.options['cxpb']}")
        self.log_message(f"Mutation probability: {self.options['mutpb']}")
        
        if toolbox is None:
            raise ValueError("Toolbox must be provided to optimize method")
        
        # Register the evaluation function in the toolbox
        toolbox.register("evaluate", objective_function)
        
        # Create initial population
        population = toolbox.population(n=self.n_individuals)
        
        # Run the simplified eaMuPlusLambda algorithm
        final_population, logbook = eaMuPlusLambda(
            population=population,
            toolbox=toolbox,
            mu=self.options['mu'],
            lambda_=self.options['lambda_'],
            cxpb=self.options['cxpb'],
            mutpb=self.options['mutpb'],
            ngen=ngen,
            stats=stats,
            halloffame=halloffame,
            verbose=verbose,
            checkpoint_path=self.checkpoint_path,
            checkpoint_interval=self.checkpoint_interval,
            console=self.console,
            console_logging=self.console_logging,
            file_logging=self.file_logging,
            watch_path=self.watch_path,
            evaluator=evaluator,
            parameter_bounds=parameter_bounds
        )
        
        # Extract best individual
        best_ind = min(final_population, key=lambda ind: ind.fitness.values[0])
        self.best_individual = best_ind
        self.best_fitness = best_ind.fitness.values[0]
        
        self.log_message(f"✨ Optimization complete! Best fitness: {self.best_fitness:.6e}", panel=True)
        
        return self.best_fitness, self.best_individual
    
    def save_checkpoint(self, generation, additional_data=None):
        """
        Save checkpoint to file.
        
        Parameters
        ----------
        generation : int
            Current generation number
        additional_data : dict, optional
            Additional data to save in checkpoint
        
        Requirements: 3.1, 3.4
        """
        if not self.checkpoint_path:
            return
        
        checkpoint_data = {
            'generation': generation,
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'generation_times': self.generation_times,
            'parameter_names': self.parameter_names,
            'options': self.options,
        }
        
        if additional_data:
            checkpoint_data.update(additional_data)
        
        try:
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            self.log_message(f"💾 Checkpoint saved at generation {generation}")
        except Exception as e:
            self.log_message(f"⚠️ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self):
        """
        Load checkpoint from file.
        
        Returns
        -------
        dict or None
            Checkpoint data if file exists, None otherwise
        
        Requirements: 3.2, 3.5
        """
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return None
        
        try:
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            self.log_message(f"🔄 Checkpoint loaded from generation {checkpoint_data.get('generation', 0)}")
            return checkpoint_data
        except Exception as e:
            self.log_message(f"⚠️ Failed to load checkpoint: {e}")
            return None
    
    def get_statistics(self):
        """
        Get optimization statistics.
        
        Returns
        -------
        dict
            Dictionary containing optimization statistics
        
        Requirements: 4.5
        """
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'best_fitness': self.best_fitness,
            'best_individual': self.best_individual,
            'generations_completed': len(self.generation_times),
            'total_time': elapsed_time,
            'average_generation_time': np.mean(self.generation_times) if self.generation_times else 0,
        }
    
    def cleanup_resources(self):
        """
        Clean up multiprocessing resources.
        
        Requirements: 6.4, 8.5
        """
        if self.pool is not None:
            try:
                self.pool.close()
                self.pool.join()
                self.log_message("🧹 Multiprocessing pool cleaned up")
            except Exception as e:
                self.log_message(f"⚠️ Error cleaning up pool: {e}")
            finally:
                self.pool = None
    
    def log_message(self, message, emoji=None, panel=False, timestamp=True):
        """
        Log a message with Rich formatting.
        
        Parameters
        ----------
        message : str
            Message to log
        emoji : str, optional
            Emoji to prepend to message
        panel : bool
            Whether to display message in a panel
        timestamp : bool
            Whether to include timestamp
        
        Requirements: 2.1, 2.2, 7.1
        """
        if not self.console_logging and not self.file_logging:
            return
        
        if not RICH_AVAILABLE:
            print(f"{emoji} {message}" if emoji else message)
            return
        
        if timestamp:
            timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = ""
        
        emoji_str = f" {emoji}" if emoji else ""
        log_text = f"{timestamp_str}{emoji_str} {message}"
        
        if panel:
            panel_message = Panel(log_text, title="DEAP Stats", border_style="cyan")
            self._console_wrapper(panel_message)
        else:
            self._console_wrapper(log_text)
    
    def _console_wrapper(self, msg):
        """
        Wrapper for Rich console output with file logging.
        
        Parameters
        ----------
        msg : str or Rich object
            Message to output
        
        Requirements: 2.3
        """
        if not RICH_AVAILABLE or not self.console:
            return
        
        if self.file_logging:
            # Ensure logs directory exists
            os.makedirs(os.path.dirname(self.watch_path), exist_ok=True)
            
            with open(self.watch_path, "a") as f, \
                 contextlib.redirect_stdout(f), \
                 contextlib.redirect_stderr(f):
                self.console.print(msg)
        elif self.console_logging:
            self.console.print(msg)


def create_pca_visualization_deap(population, halloffame, generation, console, watch_path, pca_state=None, grid_width=172, grid_height=31):
    """
    Create PCA-based ASCII visualization of DEAP population distribution.
    
    Parameters
    ----------
    population : list
        Current population of DEAP individuals
    halloffame : HallOfFame or None
        Hall of fame containing best individuals
    generation : int
        Current generation number
    console : rich.console.Console
        Rich console for output
    watch_path : str
        Path to write visualization
    pca_state : dict or None
        Dictionary containing PCA state (pca object)
    grid_width : int
        Width of ASCII grid (default: 172)
    grid_height : int
        Height of ASCII grid (default: 31)
        
    Returns
    -------
    dict or None
        Updated PCA state dictionary
    """
    if not SKLEARN_AVAILABLE or not RICH_AVAILABLE:
        return pca_state
    
    try:
        # Extract positions from population
        positions = np.array([list(ind) for ind in population])
        
        if positions.shape[0] < 2 or positions.shape[1] < 2:
            return pca_state
        
        # Initialize or update PCA
        if pca_state is None:
            pca_state = {'pca': None}
        
        # Fit PCA with current population
        pca_state['pca'] = PCA(n_components=2)
        pca_state['pca'].fit(positions)
        
        pca = pca_state['pca']
        
        # Transform positions to 2D
        positions_2d = pca.transform(positions)
        
        # Get best individual position
        if halloffame and len(halloffame) > 0:
            best_position = np.array(list(halloffame[0]))
            best_2d = pca.transform(best_position.reshape(1, -1))[0]
        else:
            # Use best from current population (only valid individuals)
            valid_indices = [i for i in range(len(population)) if population[i].fitness.valid]
            if valid_indices:
                best_idx = min(valid_indices, key=lambda i: population[i].fitness.values[0])
                best_2d = positions_2d[best_idx]
            else:
                # No valid individuals, use center of grid
                best_2d = [0, 0]
        
        # Create density grid
        density_grid = np.zeros((grid_height, grid_width))
        
        # Find bounds
        min_x, max_x = np.min(positions_2d[:, 0]), np.max(positions_2d[:, 0])
        min_y, max_y = np.min(positions_2d[:, 1]), np.max(positions_2d[:, 1])
        
        # Add padding
        padding_x = (max_x - min_x) * 0.1 if max_x > min_x else 1.0
        padding_y = (max_y - min_y) * 0.1 if max_y > min_y else 1.0
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
        
        # Add legend
        variance_explained = np.sum(pca.explained_variance_ratio_) * 100
        
        # Calculate population spread
        spread = np.sqrt(np.var(positions_2d[:, 0]) + np.var(positions_2d[:, 1]))
        
        # Calculate Pareto front size
        fronts = tools.sortNondominated(population, len(population))
        pareto_size = len(fronts[0])
        pareto_ratio = pareto_size / len(population) * 100
        
        full_viz.append("\n")
        full_viz.append("Legend: ·=empty ▪=density (gradient: purple→blue→cyan→green→yellow→red) @=best\n")
        full_viz.append(f"PC1 vs PC2 | Variance: {variance_explained:.1f}% | Pop: {len(population)} | ")
        full_viz.append(f"Pareto: {pareto_size} ({pareto_ratio:.1f}%) | Spread: {spread:.2f} | Max density: {int(max_density)}")
        
        # Create Rich panel
        panel = Panel(
            full_viz,
            title=f"🗺️ DEAP Search Space Visualization - Generation {generation}",
            border_style="cyan",
            padding=(1, 2)
        )
        
        # Write to graphs file
        os.makedirs(os.path.dirname(watch_path), exist_ok=True)
        with open(watch_path, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            console.print(panel)
        
        return pca_state
        
    except Exception as e:
        # Silently fail if visualization fails
        return pca_state


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """
    Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.

    Parameters
    ----------
    population : list
        A list of individuals to vary.
    toolbox : deap.base.Toolbox
        A toolbox that contains the evolution operators.
    lambda_ : int
        The number of children to produce
    cxpb : float
        The probability of mating two individuals.
    mutpb : float
        The probability of mutating an individual.

    Returns
    -------
    list
        The offspring population.

    Notes
    -----
    The variation goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population, those individuals are cloned using the
    toolbox.clone method and then mated using the toolbox.mate method. Only
    the first child is appended to the offspring population, the second child
    is discarded. In the case of a mutation, one individual is selected at
    random, it is cloned and then mutated using the toolbox.mutate method.
    The resulting mutant is appended to the offspring. In the case of a
    reproduction, one individual is selected at random, cloned and appended
    to the offspring.

    This variation is named *Or* because an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in [0, 1], the reproduction probability is 1 - *cxpb* - *mutpb*.
    
    Requirements: 1.2, 7.3
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = [toolbox.clone(i) for i in random.sample(population, 2)]
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=True,
                   checkpoint_path="deap_checkpoint.pkl", checkpoint_interval=5,
                   console=None, console_logging=True, file_logging=True,
                   watch_path="logs/evaluation.log", evaluator=None, parameter_bounds=None,
                   stagnation_config=None, intervals=None):
    """
    This is a simplified (μ+λ) evolutionary algorithm implementation with stagnation detection.
    
    This implementation is extracted from DEAP's eaMuPlusLambda and enhanced with
    multi-criteria stagnation detection and diversity injection mechanisms.
    It uses standard DEAP mutation rates and integrates with Rich logging for progress display.
    
    Parameters
    ----------
    population : list
        A list of individuals.
    toolbox : deap.base.Toolbox
        A toolbox that contains the evolution operators.
    mu : int
        The number of individuals to select for the next generation.
    lambda_ : int
        The number of children to produce at each generation.
    cxpb : float
        The probability that an offspring is produced by crossover.
    mutpb : float
        The probability that an offspring is produced by mutation.
    ngen : int
        The number of generation.
    stats : deap.tools.Statistics, optional
        A Statistics object that is updated inplace.
    halloffame : deap.tools.HallOfFame, optional
        A HallOfFame object that will contain the best individuals.
    verbose : bool, optional
        Whether or not to log the statistics.
    checkpoint_path : str, optional
        Path to save/load checkpoint files.
    checkpoint_interval : int, optional
        Number of generations between checkpoints.
    console : rich.console.Console, optional
        Rich console for formatted output.
    console_logging : bool, optional
        Whether to enable console logging.
    file_logging : bool, optional
        Whether to enable file logging.
    watch_path : str, optional
        Path to log file for file logging.
    evaluator : Evaluator, optional
        Evaluator instance for fitness evaluation (for integration with
        existing infrastructure).
    parameter_bounds : dict, optional
        Dictionary of parameter bounds for integration with existing
        infrastructure.
    stagnation_config : dict, optional
        Configuration for stagnation detection and diversity injection.
        If None, uses default values.
    intervals : list of MonthlyInterval, optional
        List of monthly intervals for interval-based evaluation.
        If None, uses legacy single-evaluation mode.
    
    Returns
    -------
    tuple
        The final population and a Logbook with the statistics of the evolution.
    
    Notes
    -----
    The algorithm takes in a population and evolves it in place using the
    varOr function. It returns the optimized population and a logbook with
    the statistics of the evolution. The logbook will contain the generation
    number, the number of evaluations, and the statistics if a Statistics
    object is given.
    
    The algorithm uses elitism by always preserving the best individual from
    the combined parent and offspring populations.
    
    Requirements: 1.2, 6.1, 6.3, 7.3, 8.1, 8.2
    """
    start_time = time.time()
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # Set up stagnation detection configuration with defaults
    if stagnation_config is None:
        stagnation_config = {}
    
    stag_cfg = {
        'enabled': stagnation_config.get('enabled', True),
        'check_interval': stagnation_config.get('check_interval', 1),
        'min_generation': stagnation_config.get('min_generation', 10),
        'diversity_threshold': stagnation_config.get('diversity_threshold', 0.05),
        'non_dom_ratio_threshold': stagnation_config.get('non_dom_ratio_threshold', 0.90),
        'fitness_std_threshold': stagnation_config.get('fitness_std_threshold', 1e-6),
        'hypervolume_improvement_threshold': stagnation_config.get('hypervolume_improvement_threshold', 0.001),
        'hypervolume_window': stagnation_config.get('hypervolume_window', 4),
        'stagnation_score_threshold': stagnation_config.get('stagnation_score_threshold', 5),
        'min_generations_between_injections': stagnation_config.get('min_generations_between_injections', 2),
        'replacement_ratio': stagnation_config.get('replacement_ratio', 0.20),
        'pca_visualization_enabled': stagnation_config.get('pca_visualization_enabled', True),
        'pca_visualization_interval': stagnation_config.get('pca_visualization_interval', 10),
        'pca_grid_width': stagnation_config.get('pca_grid_width', 172),
        'pca_grid_height': stagnation_config.get('pca_grid_height', 31),
        'pca_graphs_path': stagnation_config.get('pca_graphs_path', 'logs/graphs.log')
    }
    
    # Helper function for logging with Rich
    def log_message(message, emoji=None, panel=False, timestamp=True):
        """Log a message with Rich formatting."""
        if not console_logging and not file_logging:
            return
        
        if not RICH_AVAILABLE or console is None:
            print(f"{emoji} {message}" if emoji else message)
            return
        
        if timestamp:
            timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = ""
        
        emoji_str = f" {emoji}" if emoji else ""
        log_text = f"{timestamp_str}{emoji_str} {message}"
        
        if panel:
            panel_message = Panel(log_text, title="DEAP Stats", border_style="cyan")
            _console_wrapper(panel_message, console, file_logging, watch_path)
        else:
            _console_wrapper(log_text, console, file_logging, watch_path)
    
    # Set up multiprocessing pool if evaluator is provided
    pool = None
    if evaluator is not None:
        from multiprocessing import Pool, cpu_count
        pool = Pool(processes=(cpu_count() - 1))
        log_message(f"Multiprocessing pool created with {cpu_count() - 1} processes", emoji="🔧")
    
    # Try loading from checkpoint
    start_gen = 0
    stagnation_detector_loaded = None  # Will be restored from checkpoint if available
    global_best_fitness_loaded = None
    global_best_individual_loaded = None
    gens_since_global_best_loaded = None
    stagnation_counter_loaded = None
    prev_generation_best_loaded = None
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        log_message("Checkpoint found. Loading...", emoji="📦")
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
                population = checkpoint_data["population"]
                logbook = checkpoint_data["logbook"]
                start_gen = checkpoint_data["generation"] + 1
                # Restore stagnation detector if available
                stagnation_detector_loaded = checkpoint_data.get("stagnation_detector", None)
                # Restore global best tracking with backwards compatibility
                global_best_fitness_loaded = checkpoint_data.get("global_best_fitness", None)
                global_best_individual_loaded = checkpoint_data.get("global_best_individual", None)
                gens_since_global_best_loaded = checkpoint_data.get("gens_since_global_best", None)
                stagnation_counter_loaded = checkpoint_data.get("stagnation_counter", 0)  # Default to 0 for old checkpoints
                prev_generation_best_loaded = checkpoint_data.get("prev_generation_best", float('inf'))  # Default to inf for old checkpoints
            log_message(f"Resuming from generation {start_gen}", emoji="✅")
            
            if stagnation_detector_loaded:
                log_message(
                    f"Restored stagnation detector (injections: {stagnation_detector_loaded.get('injection_count', 0)})",
                    emoji="🔄"
                )
            
            if global_best_fitness_loaded is not None:
                log_message(
                    f"Restored global best: {global_best_fitness_loaded:.6e} "
                    f"({gens_since_global_best_loaded} gens since improvement, "
                    f"gen stagnation: {stagnation_counter}/100)",
                    emoji="🏆"
                )
            
            # Adjust population size if needed
            if len(population) < mu:
                log_message(f"Expanding population to: {mu}", emoji="🌱")
                new_inds = [toolbox.individual() for _ in range(mu - len(population))]
                population.extend(new_inds)
            elif len(population) > mu:
                population = population[:mu]
                log_message(f"Reduced population to: {mu}", emoji="🍂")
            
            # Invalidate fitness values for re-evaluation
            for ind in population:
                del ind.fitness.values
            
            # Re-evaluate the loaded population
            log_message("Re-evaluating loaded population...", emoji="🔄")
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            
            if evaluator is not None and pool is not None:
                # Use multiprocessing evaluation with existing infrastructure
                # Use infinity as global best since we're just starting
                fitnesses = _evaluate_population_parallel(
                    invalid_ind, evaluator, pool, console, watch_path, file_logging, float('inf'), intervals
                )
            else:
                # Use standard DEAP evaluation
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            if halloffame is not None:
                halloffame.update(population)
                
        except Exception as e:
            log_message(f"Failed to load checkpoint: {e}", emoji="⚠️")
            start_gen = 0
    
    # Evaluate initial population if starting fresh
    if start_gen == 0:
        log_message("No checkpoint found. Starting fresh.", emoji="🚀")
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        
        if evaluator is not None and pool is not None:
            # Use multiprocessing evaluation with existing infrastructure
            # Use infinity as global best since we're just starting
            fitnesses = _evaluate_population_parallel(
                invalid_ind, evaluator, pool, console, watch_path, file_logging, float('inf'), intervals
            )
        else:
            # Use standard DEAP evaluation
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        if halloffame is not None:
            halloffame.update(population)
        
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        
        if verbose:
            # Extract initial statistics safely
            min_vals = record.get('min')
            avg_vals = record.get('avg') 
            std_vals = record.get('std')
            max_vals = record.get('max')
            
            best_fitness = min_vals[0] if min_vals is not None and len(min_vals) > 0 else float('inf')
            mean_fitness = avg_vals[0] if avg_vals is not None and len(avg_vals) > 0 else float('inf')
            std_fitness = std_vals[0] if std_vals is not None and len(std_vals) > 0 else 0.0
            worst_fitness = max_vals[0] if max_vals is not None and len(max_vals) > 0 else float('inf')
            
            log_message(
                f"""🌟 Initial Status
                🌍 Best fitness: {best_fitness:.6e}
                📊 Mean fitness: {mean_fitness:.6e}
                👎 Worst fitness: {worst_fitness:.6e}
                📉 Fitness std dev: {std_fitness:.6e}""",
                panel=True
            )
    
    # Track generation times and global best (separate from population)
    gen_runtimes = []
    global_best_fitness = global_best_fitness_loaded if global_best_fitness_loaded is not None else float('inf')
    global_best_individual = global_best_individual_loaded if global_best_individual_loaded is not None else None
    gens_since_global_best = gens_since_global_best_loaded if gens_since_global_best_loaded is not None else 0
    stagnation_counter = stagnation_counter_loaded if stagnation_counter_loaded is not None else 0  # Handle None case
    
    # Track previous generation's best for stagnation detection
    prev_generation_best = prev_generation_best_loaded if prev_generation_best_loaded is not None else float('inf')  # Handle None case
    
    # Initialize global best from initial population if not loaded from checkpoint
    if global_best_fitness_loaded is None and population:
        for ind in population:
            if ind.fitness.valid and ind.fitness.values[0] < global_best_fitness:
                global_best_fitness = ind.fitness.values[0]
                global_best_individual = toolbox.clone(ind)  # Keep separate copy
    
    # Initialize stagnation detection
    if stagnation_detector_loaded:
        # Restore from checkpoint
        stagnation_detector = stagnation_detector_loaded
        log_message(
            f"Stagnation tracking resumed: {stagnation_detector['stagnation_counter']} gens stagnant, "
            f"{stagnation_detector['injection_count']} injections",
            emoji="📊"
        )
    else:
        # Initialize fresh
        stagnation_detector = {
            'hypervolume_history': [],
            'diversity_history': [],
            'fitness_std_history': [],
            'non_dom_ratio_history': [],
            'initial_diversity': None,
            'last_injection_gen': 0,
            'injection_count': 0,
            'stagnation_counter': 0,  # Tracks consecutive stagnant generations
            'last_improvement_gen': 0,  # Last generation with improvement
            'prev_global_best': float('inf')  # Previous best for comparison
        }
    
    # Initialize covariance eigenvalue convergence detection
    from deap_optimizer.convergence import (
        initialize_convergence_state,
        update_convergence_state,
        get_convergence_log_dict
    )
    convergence_state = initialize_convergence_state(
        parameter_bounds,
        epsilon_geom=1e-6,
        epsilon_mid=1e-4,
        stagnation_window=30
    )
    restart_count = 0  # Track number of LHS restarts
    
    # Initialize PCA visualization state
    pca_state = None
    
    # Save initial checkpoint if starting fresh
    if start_gen == 0 and checkpoint_path:
        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "population": population,
                    "logbook": logbook,
                    "generation": 0,
                    "stagnation_detector": stagnation_detector,
                    "global_best_fitness": global_best_fitness,
                    "global_best_individual": global_best_individual,
                    "gens_since_global_best": gens_since_global_best,
                    "stagnation_counter": stagnation_counter,
                    "prev_generation_best": prev_generation_best
                }, f)
            log_message(f"Saved checkpoint at generation 0", emoji="💾")
        except Exception as e:
            log_message(f"Failed to save checkpoint: {e}", emoji="⚠️")
    
    log_message(f"Starting Generational Loop", emoji="🧬")
    
    # Begin the generational process
    for gen in range(start_gen + 1, ngen + 1):
        gen_start_time = time.time()
        
        # Display generation separator with Rich Rule
        if RICH_AVAILABLE and console:
            _console_wrapper(
                Rule(f"Generation {gen}", style="bold blue"),
                console, file_logging, watch_path
            )
        
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness (including new population after replacement)
        invalid_ind = [ind for ind in population + offspring if not ind.fitness.valid]
        
        if evaluator is not None and pool is not None:
            # Use multiprocessing evaluation with existing infrastructure
            fitnesses = _evaluate_population_parallel(
                invalid_ind, evaluator, pool, console, watch_path, file_logging, global_best_fitness, intervals
            )
        else:
            # Use standard DEAP evaluation
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
        
        # Select the next generation population (mu individuals)
        # Using elitism: best individual from combined population always survives
        # But first ensure we have valid fitness values
        valid_individuals = [ind for ind in population + offspring if ind.fitness.valid]
        
        if not valid_individuals:
            # This shouldn't happen, but if it does, log an error and continue
            log_message("⚠️ No valid individuals found! This shouldn't happen.", emoji="❌")
            continue
            
        best_ind = min(valid_individuals, key=lambda ind: ind.fitness.values[0])
        
        # Update global best and track improvements (separate from population)
        current_best_fitness = best_ind.fitness.values[0]
        improvement_threshold = 1e-8  # Minimum improvement to count as progress
        
        # Update global best if improved
        if current_best_fitness < global_best_fitness - improvement_threshold:
            improvement = global_best_fitness - current_best_fitness
            global_best_fitness = current_best_fitness
            global_best_individual = toolbox.clone(best_ind)  # Keep separate copy
            gens_since_global_best = 0  # Reset counter
            
            # Log new global best with celebration
            log_message(
                f"🎉 NEW GLOBAL BEST! 🎉\n"
                f"   Fitness: {global_best_fitness:.6e}\n"
                f"   Improvement: {improvement:.6e}\n"
                f"   Generation: {gen}",
                emoji="🏆",
                panel=True
            )
            
            # Log to best individual file (like PSO does)
            if evaluator is not None:
                try:
                    best_log_path = "logs/evaluation_output_best.log"
                    os.makedirs(os.path.dirname(best_log_path), exist_ok=True)
                    
                    with open(best_log_path, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(f"\n🌟 NEW BEST INDIVIDUAL STATUS - Generation {gen} - {timestamp}")
                        print(f"🎯 Best Fitness: {global_best_fitness:.6e}")
                        print(f"📈 Improvement: {improvement:.6e}")
                        print("")
                        
                        # Evaluate with verbose output (this will show the optimization status table)
                        evaluator.evaluate(list(global_best_individual))
                        
                        # Add PCA visualization if available
                        if SKLEARN_AVAILABLE and console is not None:
                            print("\n")  # Add spacing before PCA visualization
                            # Create a temporary PCA state for this visualization
                            temp_pca_state = create_pca_visualization_deap(
                                population=population,
                                halloffame=halloffame,
                                generation=gen,
                                console=console,
                                watch_path=best_log_path,
                                pca_state=None,  # Fresh PCA for best log
                                grid_width=172,
                                grid_height=31
                            )
                    
                    log_message(f"🌟 Best individual status logged to {best_log_path} (generation {gen})")
                except Exception as e:
                    log_message(f"⚠️ Failed to log best individual status: {e}")
        else:
            gens_since_global_best += 1  # Increment counter
            
        # Update stagnation counter based on generation best improvement
        if current_best_fitness < prev_generation_best - improvement_threshold:
            # Generation best improved - reset stagnation counter
            stagnation_counter = 0
            log_message(f"🔥 Generation best improved: {current_best_fitness:.6e} (prev: {prev_generation_best:.6e})", emoji="📈")
        else:
            # Generation best did not improve - increment stagnation counter
            stagnation_counter += 1
            
        # Update previous generation best for next iteration
        prev_generation_best = current_best_fitness
            
        # Evaluate global best every generation (like PSO does)
        if global_best_individual is not None and evaluator is not None:
            try:
                # Evaluate global best without using the result (just trigger evaluator)
                evaluator.evaluate(list(global_best_individual))
            except Exception as e:
                log_message(f"⚠️ Failed to evaluate global best: {e}")
        
        # Check for convergence/stagnation using covariance eigenvalues
        should_terminate, should_restart, convergence_reason = update_convergence_state(
            population, current_best_fitness, convergence_state
        )
        
        conv_metrics = get_convergence_log_dict(convergence_state)
        
        if should_restart:
            restart_count += 1
            log_message(
                f"🔄 RESTART #{restart_count}: {convergence_reason}",
                emoji="⚠️",
                panel=True
            )
            log_message(
                f"   λ_max={conv_metrics['lambda_max']:.2e}, λ_min={conv_metrics['lambda_min']:.2e}, "
                f"active_dims={conv_metrics['active_dims']}, Δfit={conv_metrics['delta_fitness']:.2e}",
                emoji="📊"
            )
            
            # Create entirely new population using LHS for better coverage
            try:
                from scipy.stats import qmc
                
                # Separate variable and fixed parameters (LHS requires lower < upper)
                variable_indices = []
                fixed_indices = []
                fixed_values = []
                
                for i, (name, (low, high)) in enumerate(parameter_bounds.items()):
                    if low < high:
                        variable_indices.append(i)
                    else:
                        fixed_indices.append(i)
                        fixed_values.append(low)
                
                if variable_indices:
                    # LHS only for variable parameters
                    variable_lower = np.array([list(parameter_bounds.values())[i][0] for i in variable_indices])
                    variable_upper = np.array([list(parameter_bounds.values())[i][1] for i in variable_indices])
                    
                    sampler = qmc.LatinHypercube(d=len(variable_indices), seed=np.random.randint(0, 2**31))
                    unit_samples = sampler.random(n=mu)
                    variable_samples = qmc.scale(unit_samples, variable_lower, variable_upper)
                    
                    # Reconstruct full samples with fixed values
                    full_samples = np.zeros((mu, len(parameter_bounds)))
                    for j, var_idx in enumerate(variable_indices):
                        full_samples[:, var_idx] = variable_samples[:, j]
                    for j, fix_idx in enumerate(fixed_indices):
                        full_samples[:, fix_idx] = fixed_values[j]
                    
                    new_population = [creator.Individual(list(sample)) for sample in full_samples]
                    log_message(f"✨ Created {mu} new individuals using LHS ({len(variable_indices)} variable params)", emoji="🌱")
                else:
                    # All parameters fixed
                    fixed_individual = [list(parameter_bounds.values())[i][0] for i in range(len(parameter_bounds))]
                    new_population = [creator.Individual(list(fixed_individual)) for _ in range(mu)]
                    log_message(f"✨ All parameters fixed, created {mu} identical individuals", emoji="🌱")
                    
            except ImportError:
                log_message("⚠️ scipy not available, using random sampling", emoji="⚠️")
                new_population = [toolbox.individual() for _ in range(mu)]
            
            # Invalidate fitness for all new individuals
            for ind in new_population:
                del ind.fitness.values
            
            # Replace the entire population
            population[:] = new_population
            
            # Reset convergence state history for fresh start
            convergence_state.best_fitness_history.clear()
            convergence_state.lambda_max_history.clear()
            convergence_state.lambda_min_history.clear()
            convergence_state.active_dims_history.clear()
            convergence_state.stagnation_detected = False
            convergence_state.convergence_detected = False
            
            log_message(
                f"✨ Global best preserved separately: {global_best_fitness:.6e}",
                emoji="🏆"
            )
        else:
            # Normal selection: best individual from combined population always survives
            selected = toolbox.select(population + offspring, mu - 1)
            population[:] = selected + [best_ind]
            
            log_message(f"Elite preserved with fitness: {best_ind.fitness.values[0]:.6e}", emoji="🏅")
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        
        if verbose:
            # Extract statistics safely
            min_vals = record.get('min')
            avg_vals = record.get('avg') 
            std_vals = record.get('std')
            max_vals = record.get('max')
            
            gen_best_fitness = min_vals[0] if min_vals is not None and len(min_vals) > 0 else float('inf')
            mean_fitness = avg_vals[0] if avg_vals is not None and len(avg_vals) > 0 else float('inf')
            std_fitness = std_vals[0] if std_vals is not None and len(std_vals) > 0 else 0.0
            worst_fitness = max_vals[0] if max_vals is not None and len(max_vals) > 0 else float('inf')
            
            # Time tracking
            gen_time = time.time() - gen_start_time
            gen_runtimes.append(gen_time)
            avg_gen_time = sum(gen_runtimes) / len(gen_runtimes)
            
            # Get convergence metrics
            conv_metrics = get_convergence_log_dict(convergence_state)
            
            # Build status message with convergence metrics
            status_msg = f"""🌟 Gen {gen}
                🧬 Gen Best: {gen_best_fitness:.6e}
                🏆 Global Best: {global_best_fitness:.6e}
                ⏳ Gens since global best: {gens_since_global_best}
                🔄 Restarts: {restart_count}
                📊 Mean fitness: {mean_fitness:.6e}
                👎 Worst fitness: {worst_fitness:.6e}
                📉 Fitness std dev: {std_fitness:.6e}
                🎯 λ_max: {conv_metrics['lambda_max']:.2e} (ε_geom={convergence_state.epsilon_geom:.0e}, ε_mid={convergence_state.epsilon_mid:.0e})
                📐 Active dims: {conv_metrics['active_dims']}
                📈 Δ fitness: {conv_metrics['delta_fitness']:.2e}
                ⏱️ Generation time: {gen_time:.2f} sec / {(gen_time/60):.2f} min
                📆 Avg gen time: {avg_gen_time:.2f} sec / {(avg_gen_time/60):.2f} min"""
            
            log_message(
                status_msg,
                panel=True,
                timestamp=False
            )
        
        # Stagnation detection and diversity injection (DISABLED - using covariance eigenvalue system)
        if False and stag_cfg['enabled'] and gen % stag_cfg['check_interval'] == 0 and gen > stag_cfg['min_generation']:
            # Calculate diversity metrics
            positions = np.array([list(ind) for ind in population])
            current_diversity = np.mean(np.std(positions, axis=0))
            
            # Initialize baseline diversity
            if stagnation_detector['initial_diversity'] is None:
                stagnation_detector['initial_diversity'] = current_diversity
            
            # Track diversity
            stagnation_detector['diversity_history'].append(current_diversity)
            
            # Calculate fitness standard deviation (only for valid individuals)
            fitness_values = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
            fitness_std = np.std(fitness_values)
            stagnation_detector['fitness_std_history'].append(fitness_std)
            
            # Calculate non-dominated ratio
            fronts = tools.sortNondominated(population, len(population))
            non_dom_ratio = len(fronts[0]) / len(population)
            stagnation_detector['non_dom_ratio_history'].append(non_dom_ratio)
            
            # Calculate hypervolume if possible (for 2-objective problems)
            try:
                # Get Pareto front
                pareto_front = fronts[0]
                if len(pareto_front) > 0 and len(pareto_front[0].fitness.values) == 2:
                    # Calculate reference point (slightly worse than worst values)
                    all_obj1 = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
                    all_obj2 = [ind.fitness.values[1] for ind in population if ind.fitness.valid]
                    ref_point = [max(all_obj1) * 1.1, max(all_obj2) * 1.1]
                    
                    # Calculate hypervolume
                    hv = tools.hypervolume(pareto_front, ref_point)
                    stagnation_detector['hypervolume_history'].append(hv)
            except Exception as e:
                # Hypervolume calculation failed, skip it
                pass
            
            # Detect stagnation using multiple criteria
            stagnation_score = 0
            stagnation_reasons = []
            
            # Criterion 1: Diversity loss (weight: 3)
            if stagnation_detector['initial_diversity'] > 0:
                diversity_ratio = current_diversity / stagnation_detector['initial_diversity']
                if diversity_ratio < stag_cfg['diversity_threshold']:
                    stagnation_score += 3
                    stagnation_reasons.append(f"diversity_loss({diversity_ratio:.3f})")
            
            # Criterion 2: Non-dominated ratio (weight: 2)
            if non_dom_ratio > stag_cfg['non_dom_ratio_threshold']:
                stagnation_score += 2
                stagnation_reasons.append(f"non_dom_ratio({non_dom_ratio:.3f})")
            
            # Criterion 3: Fitness std dev (weight: 2)
            if fitness_std < stag_cfg['fitness_std_threshold']:
                stagnation_score += 2
                stagnation_reasons.append(f"fitness_std({fitness_std:.3e})")
            
            # Criterion 4: Hypervolume stagnation (weight: 3)
            if len(stagnation_detector['hypervolume_history']) >= stag_cfg['hypervolume_window']:
                recent_hvs = stagnation_detector['hypervolume_history'][-stag_cfg['hypervolume_window']:]
                if max(recent_hvs) > 0:
                    hv_improvement = (max(recent_hvs) - min(recent_hvs)) / max(recent_hvs)
                    if hv_improvement < stag_cfg['hypervolume_improvement_threshold']:
                        stagnation_score += 3
                        stagnation_reasons.append(f"hv_stagnation({hv_improvement:.4f})")
            
            # Update stagnation counter based on improvement
            # Check if global best improved
            prev_best = stagnation_detector.get('prev_global_best', float('inf'))
            improvement_threshold = 1e-8  # Minimum improvement to count as progress
            
            if global_best_fitness < prev_best - improvement_threshold:
                # Improvement detected - reset counter
                stagnation_detector['stagnation_counter'] = 0
                stagnation_detector['last_improvement_gen'] = gen
            else:
                # No improvement - increment counter
                stagnation_detector['stagnation_counter'] += stag_cfg['check_interval']
            
            # Store current best for next comparison
            stagnation_detector['prev_global_best'] = global_best_fitness
            
            # Advanced stagnation injection disabled - using simple LHS recreation instead
            # The tracking above is kept for PCA visualization purposes
        
        # PCA Visualization (create every generation)
        if stag_cfg['pca_visualization_enabled']:
            if SKLEARN_AVAILABLE and console is not None:
                pca_state = create_pca_visualization_deap(
                    population=population,
                    halloffame=halloffame,
                    generation=gen,
                    console=console,
                    watch_path=stag_cfg['pca_graphs_path'],
                    pca_state=pca_state,
                    grid_width=stag_cfg['pca_grid_width'],
                    grid_height=stag_cfg['pca_grid_height']
                )
            elif not SKLEARN_AVAILABLE and gen == 1:
                log_message("⚠️ sklearn not available, PCA visualization disabled", emoji="⚠️")
            elif console is None and gen == 1:
                log_message("⚠️ console is None, PCA visualization disabled", emoji="⚠️")
        
        # Save checkpoint at specified intervals
        if checkpoint_path and gen % checkpoint_interval == 0:
            try:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump({
                        "population": population,
                        "logbook": logbook,
                        "generation": gen,
                        "stagnation_detector": stagnation_detector,
                        "global_best_fitness": global_best_fitness,
                        "global_best_individual": global_best_individual,
                        "gens_since_global_best": gens_since_global_best,
                        "stagnation_counter": stagnation_counter,
                        "prev_generation_best": prev_generation_best
                    }, f)
                log_message(f"Saved checkpoint at generation {gen}", emoji="💾")
            except Exception as e:
                log_message(f"Failed to save checkpoint: {e}", emoji="⚠️")
    
    # Clean up multiprocessing pool
    if pool is not None:
        try:
            pool.close()
            pool.join()
            log_message("Multiprocessing pool cleaned up", emoji="🧹")
        except Exception as e:
            log_message(f"Error cleaning up pool: {e}", emoji="⚠️")
    
    total_time = time.time() - start_time
    log_message(
        f"Total time for optimization: {total_time:.2f} seconds ({total_time/60:.2f} minutes, {total_time/3600:.2f} hours)",
        emoji="🕒"
    )
    
    return population, logbook


def _console_wrapper(msg, console, file_logging, watch_path):
    """
    Wrapper for Rich console output with file logging.
    
    Parameters
    ----------
    msg : str or Rich object
        Message to output
    console : rich.console.Console
        Rich console instance
    file_logging : bool
        Whether to enable file logging
    watch_path : str
        Path to log file
    
    Requirements: 2.3
    """
    if not RICH_AVAILABLE or console is None:
        return
    
    if file_logging:
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(watch_path), exist_ok=True)
        
        with open(watch_path, "a") as f, \
             contextlib.redirect_stdout(f), \
             contextlib.redirect_stderr(f):
            console.print(msg)
    else:
        console.print(msg)



def _evaluate_population_parallel(individuals, evaluator, pool, console, watch_path, file_logging, global_best=float('inf'), intervals=None):
    """
    Evaluate a population of individuals using multiprocessing.

    This function evaluates individuals in parallel using the existing Evaluator
    infrastructure and multiprocessing pool, following the PSO pattern.

    When intervals is provided, evaluates each individual on each monthly interval
    and aggregates results using compute_monthly_fitness.

    Parameters
    ----------
    individuals : list
        List of DEAP individuals to evaluate
    evaluator : Evaluator
        Evaluator instance for fitness evaluation
    pool : multiprocessing.Pool
        Multiprocessing pool for parallel evaluation
    console : rich.console.Console
        Rich console for progress display
    watch_path : str
        Path to log file
    file_logging : bool
        Whether to enable file logging
    global_best : float, optional
        Current global best fitness for progress tracking
    intervals : list of MonthlyInterval, optional
        List of monthly intervals for interval-based evaluation.
        If None, uses legacy single-evaluation mode.

    Returns
    -------
    list
        List of fitness tuples (one per individual)

    Requirements: 6.1, 6.3, 7.1, 7.2, 7.3, 8.1, 8.2, 8.4
    """
    import contextlib
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    # Interval mode: evaluate each individual on each monthly interval
    if intervals is not None:
        from deap_optimizer.interval import build_interval_task_pool, compute_monthly_fitness

        # Build task pool: (evaluator, individual, month_id, interval, candidate_id)
        all_args = build_interval_task_pool(individuals, intervals, evaluator)
        total_tasks = len(all_args)
        total_months = len(intervals)
        total_candidates = len(individuals)

        # Results structure: {candidate_id: {month_id: (fitness_value, bankruptcy_flag, bankruptcy_reason, bankruptcy_timestep, total_timesteps)}}
        candidate_results = {i: {} for i in range(total_candidates)}
        
        # Track completed candidates and their bankruptcy status
        completed_candidates = {}  # candidate_id -> is_bankrupt (any month bankrupt)
        # Track bankruptcy reasons per completed candidate
        completed_bankruptcy_reasons = {}  # candidate_id -> bankruptcy_reason (first bankruptcy reason encountered)

        # Evaluate with progress bar
        if RICH_AVAILABLE and console:
            with open(watch_path, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
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
                    # Initialize tracking
                    generational_best = float('inf')
                    completed_tasks = 0
                    
                    # Track bankruptcy per candidate (True if any month is bankrupt)
                    candidate_has_bankruptcy = {i: False for i in range(total_candidates)}
                    # Track first bankruptcy reason per candidate
                    candidate_bankruptcy_reason = {i: 0 for i in range(total_candidates)}
                    # Track gains per candidate for aggregation
                    candidate_gains = {i: [] for i in range(total_candidates)}

                    task = progress.add_task(
                        f"🧬 ✅0 😴0 🕰️0 💸0 📉0 | Tasks: 0/{total_tasks} | Gen Best: {generational_best:.6e} | Global Best: {global_best:.6e}",
                        total=total_tasks
                    )

                    for result in pool.imap_unordered(_evaluate_interval_worker, all_args):
                        candidate_id, month_id, fitness_value, bankruptcy_flag, bankruptcy_reason, bankruptcy_timestep, total_timesteps, gain = result

                        # Store result (without gain, for fitness computation)
                        candidate_results[candidate_id][month_id] = (
                            fitness_value, bankruptcy_flag, bankruptcy_reason, bankruptcy_timestep, total_timesteps
                        )
                        # Store gain separately
                        candidate_gains[candidate_id].append(gain)

                        # Track if this candidate has any bankruptcy and its reason
                        if bankruptcy_flag:
                            candidate_has_bankruptcy[candidate_id] = True
                            # Store first bankruptcy reason encountered
                            if candidate_bankruptcy_reason[candidate_id] == 0:
                                candidate_bankruptcy_reason[candidate_id] = bankruptcy_reason

                        completed_tasks += 1

                        # Check if this candidate is now complete
                        if len(candidate_results[candidate_id]) == total_months and candidate_id not in completed_candidates:
                            # Mark as complete with bankruptcy status
                            completed_candidates[candidate_id] = candidate_has_bankruptcy[candidate_id]
                            completed_bankruptcy_reasons[candidate_id] = candidate_bankruptcy_reason[candidate_id]
                            
                            # Compute this candidate's fitness to update generational best
                            results = candidate_results[candidate_id]
                            fitness_tuple = compute_monthly_fitness(results, total_months)
                            if fitness_tuple[0] < generational_best:
                                generational_best = fitness_tuple[0]

                        # Count passed vs bankrupt candidates by reason
                        passed_count = sum(1 for is_bankrupt in completed_candidates.values() if not is_bankrupt)
                        # Count by bankruptcy reason: 1=financial(💸), 2=drawdown(📉), 3=no_positions(😴), 4=stale_position(🕰️)
                        no_pos_count = sum(1 for cid, reason in completed_bankruptcy_reasons.items() if completed_candidates.get(cid, False) and reason == 3)
                        stale_count = sum(1 for cid, reason in completed_bankruptcy_reasons.items() if completed_candidates.get(cid, False) and reason == 4)
                        financial_count = sum(1 for cid, reason in completed_bankruptcy_reasons.items() if completed_candidates.get(cid, False) and reason == 1)
                        drawdown_count = sum(1 for cid, reason in completed_bankruptcy_reasons.items() if completed_candidates.get(cid, False) and reason == 2)

                        # Use flame emoji when gen best beats global best
                        emoji = "🧬" if generational_best >= global_best else "🔥"

                        progress.update(
                            task,
                            advance=1,
                            description=f"{emoji} ✅{passed_count} 😴{no_pos_count} 🕰️{stale_count} 💸{financial_count} 📉{drawdown_count} | Tasks: {completed_tasks}/{total_tasks} | Gen Best: {generational_best:.6e} | Global Best: {global_best:.6e}"
                        )
        else:
            # Fallback without progress bar
            candidate_has_bankruptcy = {i: False for i in range(total_candidates)}
            candidate_bankruptcy_reason = {i: 0 for i in range(total_candidates)}
            candidate_gains = {i: [] for i in range(total_candidates)}
            for result in pool.imap_unordered(_evaluate_interval_worker, all_args):
                candidate_id, month_id, fitness_value, bankruptcy_flag, bankruptcy_reason, bankruptcy_timestep, total_timesteps, gain = result
                candidate_results[candidate_id][month_id] = (
                    fitness_value, bankruptcy_flag, bankruptcy_reason, bankruptcy_timestep, total_timesteps
                )
                candidate_gains[candidate_id].append(gain)
                if bankruptcy_flag:
                    candidate_has_bankruptcy[candidate_id] = True
                    if candidate_bankruptcy_reason[candidate_id] == 0:
                        candidate_bankruptcy_reason[candidate_id] = bankruptcy_reason

        # Aggregate results for each candidate and log non-bankrupt ones
        fitness_results = []
        log_path = "logs/evaluation_output.log"
        
        for candidate_id in range(len(individuals)):
            results = candidate_results[candidate_id]
            fitness_tuple = compute_monthly_fitness(results, total_months)
            fitness_results.append(fitness_tuple)
            
            # Check if candidate had any bankruptcy
            has_bankruptcy = any(r[1] for r in results.values())  # r[1] is bankruptcy_flag
            
            # Compute total gain (product of monthly gains)
            gains = candidate_gains.get(candidate_id, [])
            total_gain = 1.0
            for g in gains:
                total_gain *= g
            
            # Log non-bankrupt candidates with gain
            if not has_bankruptcy and file_logging:
                try:
                    with open(log_path, "a") as f:
                        f.write(f"\n{'='*60}\n")
                        f.write(f"📅 Candidate {candidate_id} - Monthly Interval Results\n")
                        f.write(f"{'='*60}\n")
                        f.write(f"Final Fitness: {fitness_tuple[0]:.6e}\n")
                        f.write(f"Total Gain:    {total_gain:.5f}\n")
                        f.write(f"Months:        {total_months}\n")
                        f.write(f"{'='*60}\n\n")
                except (PermissionError, OSError):
                    pass

        return fitness_results

    # Legacy mode: single evaluation per individual
    # Prepare evaluation arguments
    all_args = []
    for idx, individual in enumerate(individuals):
        # Include the index to maintain pairing
        all_args.append((evaluator, individual, False, idx))

    # Evaluate with progress bar
    if RICH_AVAILABLE and console:
        with open(watch_path, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
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
                # Initialize generational best tracking
                generational_best = float('inf')
                bankrupt_count = 0
                non_bankrupt_count = 0

                task = progress.add_task(
                    f"DEAP Evaluating individuals | Gen Best: {generational_best:.6e} | Global Best: {global_best:.6e}",
                    total=len(individuals)
                )

                # Collect results with their original indices to maintain ordering
                fitness_results = [None] * len(individuals)  # Pre-allocate array

                for result in pool.imap_unordered(_evaluate_solution_worker, all_args):
                    fitness_tuple, position_idx, bankrupt = result
                    fitness_results[position_idx] = fitness_tuple  # Place fitness tuple at correct index

                    # Track bankruptcy counts
                    if bankrupt:
                        bankrupt_count += 1
                    else:
                        non_bankrupt_count += 1

                    # Track best in current generation (batch)
                    if fitness_tuple and len(fitness_tuple) > 0:
                        fitness_val = fitness_tuple[0]  # First element of fitness tuple
                        if fitness_val < generational_best:
                            generational_best = fitness_val

                    # Simple count display
                    push_bar = f"✅{non_bankrupt_count} 💀{bankrupt_count}"

                    # Update progress with counts and fitness info
                    emoji = "🧬" if generational_best >= global_best else "🔥"
                    progress.update(
                        task,
                        advance=1,
                        description=f"{emoji} {push_bar} | Gen Best: {generational_best:.6e} | Global Best: {global_best:.6e}"
                    )

                return fitness_results
    else:
        # Fallback without progress bar
        fitness_results = [None] * len(individuals)
        for result in pool.imap_unordered(_evaluate_solution_worker, all_args):
            fitness_tuple, position_idx, bankrupt = result
            fitness_results[position_idx] = fitness_tuple
        return fitness_results

