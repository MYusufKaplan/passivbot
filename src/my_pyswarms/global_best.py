# -*- coding: utf-8 -*-

r"""
A Global-best Particle Swarm Optimization (gbest PSO) algorithm.

It takes a set of candidate solutions, and tries to find the best
solution using a position-velocity update method. Uses a
star-topology where each particle is attracted to the best
performing particle.

The position update can be defined as:

.. math::

   x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

Where the position at the current timestep :math:`t` is updated using
the computed velocity at :math:`t+1`. Furthermore, the velocity update
is defined as:

.. math::

   v_{ij}(t + 1) = w * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                   + c_{2}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

Here, :math:`c1` and :math:`c2` are the cognitive and social parameters
respectively. They control the particle's behavior given two choices: (1) to
follow its *personal best* or (2) follow the swarm's *global best* position.
Overall, this dictates if the swarm is explorative or exploitative in nature.
In addition, a parameter :math:`w` controls the inertia of the swarm's
movement.

An example usage is as follows:

.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions import single_obj as fx

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Call instance of GlobalBestPSO
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                        options=options)

    # Perform optimization
    stats = optimizer.optimize(fx.sphere, iters=100)

This algorithm was adapted from the earlier works of J. Kennedy and
R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

.. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.
"""

# Import standard library
import contextlib
import datetime
import itertools
import logging
import os
import pickle
import time
from collections import deque

# Import modules
import numpy as np

# Advanced initialization imports
try:
    from scipy.stats import qmc
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Rich imports for beautiful logging
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.rule import Rule
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# PCA imports for visualization
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import subprocess

from .utils.base_single import SwarmOptimizer
from .utils.handlers import BoundaryHandler, OptionsHandler, VelocityHandler
from .utils.operators import compute_objective_function, compute_pbest
from .utils.reporter import Reporter
from .utils.topology_star import Star


def play_sound(path):
    subprocess.run(["aplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class GlobalBestPSO(SwarmOptimizer):
    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        bounds=None,
        oh_strategy=None,
        bh_strategy="periodic",
        velocity_clamp=None,
        vh_strategy="unmodified",
        center=1.00,
        ftol=-np.inf,
        ftol_iter=1,
        init_pos=None,
        scout_config=None,
    ):
        """Initialize the swarm

        This class maintains full backward compatibility with the original PySwarms
        GlobalBestPSO implementation. All existing PySwarms code will work without
        modification. The scout_config parameter is optional and defaults to
        disabled hill scout functionality, ensuring standard PSO behavior when not specified.

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        options : dict with keys :code:`{'c1', 'c2', 'w'}`
            a dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
        bounds : tuple of numpy.ndarray, optional
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        oh_strategy : dict, optional, default=None(constant options)
            a dict of update strategies for each option.
        bh_strategy : str
            a strategy for the handling of out-of-bounds particles.
        velocity_clamp : tuple, optional
            a tuple of size 2 where the first entry is the minimum velocity and
            the second entry is the maximum velocity. It sets the limits for
            velocity clamping.
        vh_strategy : str
            a strategy for the handling of the velocity of out-of-bounds particles.
        center : list (default is :code:`None`)
            an array of size :code:`dimensions`
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence. Default is :code:`-np.inf`
        ftol_iter : int
            number of iterations over which the relative error in
            objective_func(best_pos) is acceptable for convergence.
            Default is :code:`1`
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
        scout_config : dict, optional
            configuration parameters for hill scout functionality. If None or if
            'enable_scouts' is False, the optimizer behaves exactly like
            standard PySwarms GlobalBestPSO. Available parameters:
                * scouts_per_spawn : int
                    number of hill scout particles per spawn (default: 100)
                * scout_lifetime : int
                    lifetime of hill scout groups in iterations (default: 200)
                * ocean_hit_threshold : float
                    percentage threshold for ocean hits to trigger phase transition (default: 0.8)
                * negative_fitness_threshold : float
                    fitness threshold for spawning hill scouts (default: 1e3)
                * ocean_fitness_threshold : float
                    fitness threshold for ocean detection (default: 1e3)
                * initial_radius_percentage : float
                    initial sampling radius as percentage of parameter ranges (default: 0.01)
                * radius_increment_percentage : float
                    radius increment as percentage of parameter ranges (default: 0.01)
                * max_concurrent_groups : int
                    maximum number of concurrent hill scout groups (default: 999)
                * enable_scouts : bool
                    enable/disable hill scout functionality (default: False for backward compatibility)
        
        Notes
        -----
        **Backward Compatibility:**
        This implementation maintains 100% backward compatibility with PySwarms:
        
        - All existing PySwarms parameters work unchanged
        - Return format is identical: (best_cost, best_position)
        - Error handling follows PySwarms conventions
        - When scout_config is None or omitted, behaves exactly like standard PSO
        - Hill scout functionality is disabled by default to ensure existing code works
        
        **Example Usage:**
        
        Standard PySwarms usage (no changes needed)::
        
            optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9})
            best_cost, best_pos = optimizer.optimize(objective_func, iters=100)
        
        With hill scout enhancement::
        
            hill_config = {'enable_scouts': True, 'scouts_per_spawn': 50}
            optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
                                     scout_config=hill_config)
            best_cost, best_pos = optimizer.optimize(objective_func, iters=100)
        """
        # Store init_pos for our custom injection logic
        self.custom_init_pos = init_pos
        
        # Pass None to base class to avoid shape conflicts
        super(GlobalBestPSO, self).__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
            center=center,
            ftol=ftol,
            ftol_iter=ftol_iter,
            init_pos=None,  # Always None to avoid base class shape issues
        )

        if oh_strategy is None:
            oh_strategy = {}
        # Initialize logger
        self.rep = Reporter(logger=logging.getLogger(__name__))
        # Initialize the resettable attributes
        self.reset()
        # Initialize the topology
        self.top = Star()
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.vh = VelocityHandler(strategy=vh_strategy)
        self.oh = OptionsHandler(strategy=oh_strategy)
        self.name = __name__
        
        # Initialize Rich console if available
        if RICH_AVAILABLE:
            self.console = console = Console(
            force_terminal=True, 
            no_color=False, 
            log_path=False, 
            width=191,
            color_system="truecolor",  # Force truecolor support
            legacy_windows=False
        )
        
        # Checkpoint and logging attributes
        self.checkpoint_path = None
        self.checkpoint_interval = 1
        self.generation_times = []
        self.start_time = None
        self.watch_path = "logs/evaluation.log"
        self.graphs_path = "logs/graphs.log"
        self.best_fitness_history = []
        self.stagnation_count = 0
        self.last_improvement_iter = 0
        
        # History data tracking for analysis
        self.history_data_path = "pso_history_data.pkl"
        self.parameter_names = None  # Will be set when bounds are provided
        self.detailed_history = {
            'positions': [],      # All particle positions per iteration
            'fitness_scores': [], # All particle fitness scores per iteration
            'iteration': [],      # Iteration numbers
            'global_best_pos': [],# Global best position per iteration
            'global_best_cost': [],# Global best cost per iteration
            'parameter_names': None # Parameter names for analysis
        }
        
        # Velocity restart mechanism attributes
        self.velocity_threshold = 1e-7  # Threshold for detecting low velocity
        self.restart_fraction = 0.25    # Fraction of particles to restart when velocity is low
        self.low_velocity_count = 0     # Counter for consecutive low velocity iterations
        self.low_velocity_threshold = 5 # Iterations before triggering restart
        
        # Velocity boost attributes
        self.reinit_interval = 10       # Apply velocity boost every X iterations
        self.reinit_fraction = 0.70     # Boost worst Y% of particles
        self.enable_reinit = True       # Enable/disable velocity boost
        
        # 10n selection attributes
        self.enable_10n_selection = True           # Enable/disable 10n selection
        self.selection_multiplier = 10             # Multiplier for candidate pool (10n)
        self.selection_interval = 10               # Run selection every X iterations
        self.selection_on_fresh_start = True       # Run on fresh start (iteration 0)
        self.selection_on_checkpoint_resume = True # Run on checkpoint resume
        self.selection_quality_threshold_pct = 100.0  # Percentage of particles that must be below threshold (100% = all)
        
        # Adaptive inertia weight parameters
        self.initial_w = options.get('w', 0.9)
        self.final_w = 0.4
        self.adaptive_inertia = True
        
        # PCA visualization attributes
        self.enable_pca_visualization = True
        self.pca_grid_width = 172   # Fit in 191 char terminal (with panel borders)
        self.pca_grid_height = 31   # Fit in 34 line terminal (with panel borders + legend)
        self.pca = None  # Will be initialized when needed
        
        # Best particle logging
        self.best_log_path = "logs/evaluation_output_best.log"

        # Farthest-point candidate placement configuration
        self.fp_candidate_pool_size = 50000
        self.fp_discovered_cap = 2000
        self.fp_use_sobol = False
        self.fp_dtype = np.float32
        self._discovered_coords = None  # Will be initialized with grid
        
        # Grid-based competitive evolution configuration
        self.competition_enabled = True
        self.max_particles_per_cell = 2
        self.stagnation_window = 15
        self.eviction_percentage = 40  # Percentage of worst particles to evict from stagnant cells
        self.competition_check_interval = 10
        
        # Competition tracking data
        self.visited_cells = set()  # Sparse set of visited grid coordinates
        self.cell_improvement_history = {}  # {grid_coords: [fitness_improvements]}
        self.cell_last_check = {}  # {grid_coords: iteration_number}
        self.global_improvement_history = []  # Track global improvement rate
        self.cell_occupancy = {}  # {grid_coords: [particle_indices]}
        self.cell_fitness_history = {}  # {grid_coords: deque of best fitness values}
        
        # Cell blacklisting system
        self.blacklisted_cells = set()  # Set of blacklisted grid coordinates
        self.blacklist_fitness_threshold = 1e5  # Fitness threshold for blacklisting (10^5)
        self.blacklist_window = 100  # Number of iterations to track for blacklisting
        self.cell_fitness_tracking = {}  # {grid_coords: deque of (iteration, best_fitness) tuples}
        self.cell_stagnation_tracking = {}  # {grid_coords: deque of (iteration, had_improvement) tuples}

        # Scout configuration and data structures
        # Set up default configuration values
        # NOTE: Scouts are DISABLED by default for backward compatibility with PySwarms
        # Existing PySwarms code will work without any changes
        default_scout_config = {
            'scouts_per_spawn': 10,
            'scout_lifetime': 50,
            'adaptive_negative_fitness_threshold_enabled': False,
            'negative_fitness_threshold': 0,
            'spawn_lhs_percentage': 0.01,  # LHS hypercube size as % of parameter ranges
            'max_concurrent_groups': 100,
            'enable_scouts': True,  # Disabled by default for backward compatibility
            # Local search parameters
            'local_search_neighbors': 10,  # Number of neighbors to generate per scout
            'initial_search_radius_percentage': 0.01,  # 1% of parameter ranges
            'radius_shrink_factor': 0.8,  # Shrink by 20% when no improvement
            'min_search_radius_percentage': 0.00001,  # 0.001% minimum radius
        }
        
        # Merge user-provided config with defaults
        if scout_config is not None:
            # Validate and merge configuration
            self._validate_scout_config(scout_config)
            default_scout_config.update(scout_config)
        
        self.scout_config = default_scout_config
        
        # Initialize scout data structures
        self.active_scout_groups = {}  # group_id -> group_data
        self.scout_particles = {}      # particle_id -> particle_data
        self.spawning_history = []     # List of spawning events
        self.attribution_records = []  # List of attribution records
        self._next_group_id = 0        # Counter for generating unique group IDs
        self._next_scout_id = 0        # Counter for generating unique scout IDs
        
        # Perform backward compatibility check
        # This ensures the optimizer can function as a drop-in replacement for PySwarms
        if not self._check_backward_compatibility():
            # Log warning but don't fail - allow optimizer to continue
            self.rep.log(
                "Warning: Backward compatibility check failed. "
                "Some PySwarms features may not work as expected.",
                lvl=logging.WARNING
            )
        
        # Log scout status for transparency
        if self.scout_config['enable_scouts']:
            self.rep.log(
                "Scout functionality ENABLED. "
                f"Configuration: {self.scout_config}",
                lvl=logging.INFO
            )
        else:
            self.rep.log(
                "Scout functionality DISABLED (standard PySwarms PSO behavior). "
                "Set scout_config={'enable_scouts': True} to enable.",
                lvl=logging.DEBUG
            )

    def _validate_scout_config(self, config):
        """Validate scout configuration parameters
        
        This method ensures that all scout configuration parameters are valid
        and follow PySwarms error handling conventions.
        
        Parameters
        ----------
        config : dict
            Scout configuration dictionary to validate
            
        Raises
        ------
        ValueError
            If any configuration parameter is invalid (PySwarms convention)
        TypeError
            If config is not a dictionary (PySwarms convention)
        """
        # Follow PySwarms error handling pattern: use TypeError for type errors
        if not isinstance(config, dict):
            raise TypeError(
                f"scout_config must be a dictionary, got {type(config).__name__}"
            )
        
        # Define valid parameters and their validation rules
        valid_params = {
            'scouts_per_spawn': (int, lambda x: x > 0, "must be a positive integer"),
            'scout_lifetime': (int, lambda x: x > 0, "must be a positive integer"),
            'negative_fitness_threshold': ((int, float), lambda x: True, "can be any numeric value (positive, negative, or zero)"),
            'spawn_lhs_percentage': ((int, float), lambda x: 0.0 < x <= 1.0, "must be between 0.0 and 1.0"),
            'max_concurrent_groups': (int, lambda x: x > 0, "must be a positive integer"),
            'enable_scouts': (bool, lambda x: True, "must be a boolean"),
            'adaptive_negative_fitness_threshold_enabled': (bool, lambda x: True, "must be a boolean"),
            'local_search_neighbors': (int, lambda x: x > 0, "must be a positive integer"),
            'initial_search_radius_percentage': ((int, float), lambda x: 0.0 < x <= 1.0, "must be between 0.0 and 1.0"),
            'radius_shrink_factor': ((int, float), lambda x: 0.0 < x < 1.0, "must be between 0.0 and 1.0"),
            'min_search_radius_percentage': ((int, float), lambda x: 0.0 < x <= 1.0, "must be between 0.0 and 1.0"),
        }
        
        # Validate each parameter in the config
        for param_name, param_value in config.items():
            if param_name not in valid_params:
                # Follow PySwarms pattern: use ValueError for invalid parameter names
                raise ValueError(
                    f"Unknown scout parameter: '{param_name}'. "
                    f"Valid parameters are: {list(valid_params.keys())}"
                )
            
            param_type, validator, error_msg = valid_params[param_name]
            
            # Check type - use TypeError for type mismatches (PySwarms convention)
            if not isinstance(param_value, param_type):
                expected_type = param_type.__name__ if not isinstance(param_type, tuple) else f"one of {[t.__name__ for t in param_type]}"
                raise TypeError(
                    f"Parameter '{param_name}' must be {expected_type}, "
                    f"got {type(param_value).__name__}"
                )
            
            # Check value constraints - use ValueError for invalid values (PySwarms convention)
            if not validator(param_value):
                raise ValueError(
                    f"Parameter '{param_name}' {error_msg}, got {param_value}"
                )

    def get_optimizer_info(self):
        """Get information about the optimizer version and configuration
        
        This method provides information about the optimizer's configuration,
        including whether hill scout functionality is enabled and the current
        version information.
        
        Returns
        -------
        dict
            Dictionary containing optimizer information:
            - 'version': Optimizer version string
            - 'pyswarms_compatible': Whether optimizer is PySwarms compatible
            - 'scouts_enabled': Whether hill scout functionality is enabled
            - 'scout_config': Current hill scout configuration
            - 'n_particles': Number of PSO particles
            - 'dimensions': Problem dimensions
            - 'options': PSO options (c1, c2, w)
        
        Examples
        --------
        >>> optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9})
        >>> info = optimizer.get_optimizer_info()
        >>> print(f"Hill scouts enabled: {info['scouts_enabled']}")
        Hill scouts enabled: False
        """
        return {
            'version': 'Hybrid PSO-Scout v1.0 (PySwarms Compatible)',
            'pyswarms_compatible': self._check_backward_compatibility(),
            'scouts_enabled': self.scout_config.get('enable_scouts', False),
            'scout_config': self.scout_config.copy(),
            'n_particles': self.n_particles,
            'dimensions': self.dimensions,
            'options': self.options.copy() if hasattr(self, 'options') else None,
            'bounds': self.bounds,
            'velocity_clamp': self.velocity_clamp
        }

    def _check_backward_compatibility(self):
        """Check that the optimizer maintains backward compatibility with PySwarms
        
        This method verifies that all required PySwarms attributes and methods exist
        and that the optimizer can function as a drop-in replacement for the original
        GlobalBestPSO implementation.
        
        Returns
        -------
        bool
            True if backward compatible, False otherwise
            
        Notes
        -----
        This method is called internally to ensure compatibility. It checks:
        - All required PySwarms attributes exist
        - The optimize method signature is unchanged
        - Return format is compatible (tuple of best_cost, best_position)
        - Error handling follows PySwarms conventions
        """
        try:
            # Check required PySwarms attributes exist
            required_attrs = [
                'n_particles', 'dimensions', 'options', 'bounds',
                'velocity_clamp', 'ftol', 'ftol_iter', 'swarm',
                'top', 'bh', 'vh', 'oh', 'rep'
            ]
            
            for attr in required_attrs:
                if not hasattr(self, attr):
                    self.rep.log(
                        f"Backward compatibility check failed: missing attribute '{attr}'",
                        lvl=logging.WARNING
                    )
                    return False
            
            # Check that optimize method exists and has correct signature
            if not hasattr(self, 'optimize'):
                self.rep.log(
                    "Backward compatibility check failed: missing optimize method",
                    lvl=logging.WARNING
                )
                return False
            
            # Verify hill scout config has enable flag
            if not isinstance(self.scout_config, dict):
                self.rep.log(
                    "Backward compatibility check failed: scout_config is not a dict",
                    lvl=logging.WARNING
                )
                return False
            
            if 'enable_scouts' not in self.scout_config:
                self.rep.log(
                    "Backward compatibility check failed: missing enable_scouts flag",
                    lvl=logging.WARNING
                )
                return False
            
            return True
            
        except Exception as e:
            self.rep.log(
                f"Backward compatibility check failed with exception: {e}",
                lvl=logging.WARNING
            )
            return False

    def set_checkpoint_config(self, checkpoint_path="pso_checkpoint.pkl", checkpoint_interval=5, watch_path="logs/evaluation.log"):
        """Configure checkpoint and logging settings"""
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.watch_path = watch_path

    def set_history_config(self, history_data_path="pso_history_data.pkl", parameter_names=None):
        """Configure history data collection for analysis
        
        Parameters
        ----------
        history_data_path : str
            Path to save detailed history data for analysis
        parameter_names : list of str, optional
            Names of the parameters being optimized (for analysis readability)
        """
        self.history_data_path = history_data_path
        self.parameter_names = parameter_names
        if parameter_names:
            self.detailed_history['parameter_names'] = parameter_names
        self.log_message(f"📈 History data will be saved to: {history_data_path}")

    def save_history_data(self, iteration):
        """Save detailed history data for analysis (append mode)"""
        if not self.history_data_path:
            return
            
        try:
            # Load existing history if file exists and we haven't loaded it yet
            if os.path.exists(self.history_data_path) and not hasattr(self, '_history_loaded'):
                try:
                    with open(self.history_data_path, "rb") as f:
                        existing_data = pickle.load(f)
                    
                    # Merge existing data with current structure
                    for key in ['iteration', 'positions', 'fitness_scores', 'global_best_pos', 'global_best_cost']:
                        if key in existing_data and existing_data[key]:
                            self.detailed_history[key].extend(existing_data[key])
                    
                    # Preserve parameter names from existing data if available
                    if existing_data.get('parameter_names') and not self.detailed_history.get('parameter_names'):
                        self.detailed_history['parameter_names'] = existing_data['parameter_names']
                    
                    self._history_loaded = True
                    existing_count = len(existing_data.get('iteration', []))
                    if existing_count > 0:
                        self.log_message(f"📚 Loaded {existing_count} existing history records")
                        
                except Exception as e:
                    self.log_message(f"⚠️ Could not load existing history: {e}, starting fresh")
                    self._history_loaded = True  # Mark as attempted to avoid repeated tries
            
            # Store current iteration data (append to existing)
            self.detailed_history['iteration'].append(iteration)
            self.detailed_history['positions'].append(self.swarm.position.copy())
            self.detailed_history['fitness_scores'].append(self.swarm.current_cost.copy())
            self.detailed_history['global_best_pos'].append(self.swarm.best_pos.copy())
            self.detailed_history['global_best_cost'].append(self.swarm.best_cost)
            
            # Save complete history to file
            with open(self.history_data_path, "wb") as f:
                pickle.dump(self.detailed_history, f)
            
            # Log progress every 10 iterations to avoid spam
            if iteration % 10 == 0:
                total_records = len(self.detailed_history['iteration'])
                self.log_message(f"📊 History data saved (iteration {iteration}, total records: {total_records})")
                
        except Exception as e:
            self.log_message(f"⚠️ Failed to save history data: {e}")

    def console_wrapper(self, msg):
        """Wrapper for Rich console output with file logging"""
        if RICH_AVAILABLE and hasattr(self, 'console'):
            with open(self.watch_path, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                self.console.print(msg)

    def log_message(self, message, emoji=None, panel=False, timestamp=True):
        """Utility function to print logs with Rich panels and rules"""
        if not RICH_AVAILABLE:
            print(f"{emoji} {message}" if emoji else message)
            return
            
        if timestamp:
            timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = ""
        emoji_str = f" {emoji}" if emoji else ""
        log_text = f"{timestamp_str}{emoji_str} {message}"

        # Using Rule for major transitions
        if panel:
            panel_message = Panel(log_text, title="PSO Stats", border_style="cyan")
            self.console_wrapper(panel_message)
        else:
            self.console_wrapper(log_text)

    def save_checkpoint(self, iteration, additional_data=None):
        """Save PSO state to checkpoint file"""
        if not self.checkpoint_path:
            return
            
        checkpoint_data = {
            "iteration": iteration,
            "swarm_position": self.swarm.position,
            "swarm_velocity": self.swarm.velocity,
            "swarm_pbest_pos": self.swarm.pbest_pos,
            "swarm_pbest_cost": self.swarm.pbest_cost,
            "swarm_best_pos": self.swarm.best_pos,
            "swarm_best_cost": self.swarm.best_cost,
            "cost_history": self.cost_history,
            "pos_history": self.pos_history,
            "options": self.options,
            "bounds": self.bounds,
            "generation_times": self.generation_times,
            "best_fitness_history": self.best_fitness_history,
            "stagnation_count": self.stagnation_count,
            "last_improvement_iter": self.last_improvement_iter,
            # Sparse exploration tracking is saved via competitive evolution data
            # Save competitive evolution data (checkpoint-friendly sparse format)
            "visited_cells": getattr(self, 'visited_cells', set()),
            "global_improvement_history": getattr(self, 'global_improvement_history', []),
            "cell_fitness_history": getattr(self, 'cell_fitness_history', {}),
            # Save blacklist data
            "blacklisted_cells": getattr(self, 'blacklisted_cells', set()),
            "cell_fitness_tracking": getattr(self, 'cell_fitness_tracking', {}),
            "cell_stagnation_tracking": getattr(self, 'cell_stagnation_tracking', {}),
            # Save hill scout data with explicit numpy array conversion
            "active_scout_groups": self._serialize_scout_groups(),
            "scout_particles": self._serialize_scout_particles(),
            "spawning_history": getattr(self, 'spawning_history', []),
            "attribution_records": getattr(self, 'attribution_records', []),
            "_next_group_id": getattr(self, '_next_group_id', 0),
            "_next_scout_id": getattr(self, '_next_scout_id', 0),
        }
        
        if additional_data:
            checkpoint_data.update(additional_data)
        
        try:
            with open(self.checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f)
            
            # Log checkpoint with hill scout info if enabled
            if self.scout_config['enable_scouts']:
                n_groups = len(getattr(self, 'active_scout_groups', {}))
                n_scouts = len(getattr(self, 'scout_particles', {}))
                self.log_message(f"💾 PSO checkpoint saved at iteration {iteration} ({n_groups} HC groups, {n_scouts} scouts)")
            else:
                self.log_message(f"💾 PSO checkpoint saved at iteration {iteration}")
        except Exception as e:
            self.log_message(f"⚠️ Failed to save checkpoint: {e}")

    def _serialize_scout_groups(self):
        """Serialize hill scout groups with explicit numpy array handling"""
        if not hasattr(self, 'active_scout_groups'):
            return {}
        
        serialized = {}
        for group_id, group in self.active_scout_groups.items():
            serialized_group = group.copy()
            # Ensure numpy arrays are properly copied
            if 'spawn_point' in serialized_group and isinstance(serialized_group['spawn_point'], np.ndarray):
                serialized_group['spawn_point'] = serialized_group['spawn_point'].copy()
            if 'best_point' in serialized_group and isinstance(serialized_group['best_point'], np.ndarray):
                serialized_group['best_point'] = serialized_group['best_point'].copy()
            if 'best_M_points' in serialized_group and serialized_group['best_M_points']:
                serialized_group['best_M_points'] = [p.copy() if isinstance(p, np.ndarray) else p for p in serialized_group['best_M_points']]
            serialized[group_id] = serialized_group
        
        return serialized

    def _serialize_scout_particles(self):
        """Serialize scout particles with explicit numpy array handling
        
        Saves all scout data including neighbors for exact checkpoint restoration.
        """
        if not hasattr(self, 'scout_particles'):
            return {}
        
        serialized = {}
        for scout_id, scout in self.scout_particles.items():
            serialized_scout = scout.copy()
            
            # Ensure numpy arrays are properly copied
            if 'position' in serialized_scout and isinstance(serialized_scout['position'], np.ndarray):
                serialized_scout['position'] = serialized_scout['position'].copy()
            if 'velocity' in serialized_scout and isinstance(serialized_scout['velocity'], np.ndarray):
                serialized_scout['velocity'] = serialized_scout['velocity'].copy()
            if 'local_best' in serialized_scout and isinstance(serialized_scout['local_best'], np.ndarray):
                serialized_scout['local_best'] = serialized_scout['local_best'].copy()
            if 'assigned_point' in serialized_scout and isinstance(serialized_scout['assigned_point'], np.ndarray):
                serialized_scout['assigned_point'] = serialized_scout['assigned_point'].copy()
            
            # Handle neighbors array (2D numpy array)
            if 'neighbors' in serialized_scout and isinstance(serialized_scout['neighbors'], np.ndarray):
                serialized_scout['neighbors'] = serialized_scout['neighbors'].copy()
            
            # Handle neighbor_fitness array
            if 'neighbor_fitness' in serialized_scout and isinstance(serialized_scout['neighbor_fitness'], np.ndarray):
                serialized_scout['neighbor_fitness'] = serialized_scout['neighbor_fitness'].copy()
            
            serialized[scout_id] = serialized_scout
        
        return serialized

    def load_checkpoint(self):
        """Load PSO state from checkpoint file with dynamic particle count support"""
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return None
            
        try:
            with open(self.checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
            
            # Get checkpoint swarm size
            checkpoint_n_particles = checkpoint_data["swarm_position"].shape[0]
            current_n_particles = self.n_particles
            
            # Restore global best state (always preserved)
            self.swarm.best_pos = checkpoint_data["swarm_best_pos"]
            self.swarm.best_cost = checkpoint_data["swarm_best_cost"]
            
            # Handle particle count changes
            if checkpoint_n_particles == current_n_particles:
                # Same size - direct restore
                self.swarm.position = checkpoint_data["swarm_position"]
                self.swarm.velocity = checkpoint_data["swarm_velocity"]
                self.swarm.pbest_pos = checkpoint_data["swarm_pbest_pos"]
                self.swarm.pbest_cost = checkpoint_data["swarm_pbest_cost"]
                self.log_message(f"✅ Resumed with same swarm size ({current_n_particles} particles)")
                
            elif checkpoint_n_particles > current_n_particles:
                # Shrinking swarm - keep best particles
                checkpoint_pbest_cost = checkpoint_data["swarm_pbest_cost"]
                best_indices = np.argsort(checkpoint_pbest_cost)[:current_n_particles]
                
                self.swarm.position = checkpoint_data["swarm_position"][best_indices]
                self.swarm.velocity = checkpoint_data["swarm_velocity"][best_indices]
                self.swarm.pbest_pos = checkpoint_data["swarm_pbest_pos"][best_indices]
                self.swarm.pbest_cost = checkpoint_data["swarm_pbest_cost"][best_indices]
                
                self.log_message(f"📉 Shrunk swarm from {checkpoint_n_particles} to {current_n_particles} particles (kept best performers)")
                
            else:
                # Growing swarm - preserve existing + add new particles
                self.swarm.position = np.zeros((current_n_particles, self.dimensions))
                self.swarm.velocity = np.zeros((current_n_particles, self.dimensions))
                self.swarm.pbest_pos = np.zeros((current_n_particles, self.dimensions))
                self.swarm.pbest_cost = np.full(current_n_particles, np.inf)
                
                # Copy existing particles
                self.swarm.position[:checkpoint_n_particles] = checkpoint_data["swarm_position"]
                self.swarm.velocity[:checkpoint_n_particles] = checkpoint_data["swarm_velocity"]
                self.swarm.pbest_pos[:checkpoint_n_particles] = checkpoint_data["swarm_pbest_pos"]
                self.swarm.pbest_cost[:checkpoint_n_particles] = checkpoint_data["swarm_pbest_cost"]
                
                # Initialize new particles
                new_particles = current_n_particles - checkpoint_n_particles
                
                # Initialize positions for new particles
                if self.bounds is not None:
                    min_bounds, max_bounds = self.bounds
                    self.swarm.position[checkpoint_n_particles:] = np.random.uniform(
                        min_bounds, max_bounds, (new_particles, self.dimensions)
                    )
                else:
                    # Use center-based initialization if no bounds
                    self.swarm.position[checkpoint_n_particles:] = np.random.uniform(
                        -1, 1, (new_particles, self.dimensions)
                    ) * self.center
                
                # Initialize velocities for new particles
                if self.velocity_clamp is not None:
                    min_vel, max_vel = self.velocity_clamp
                    self.swarm.velocity[checkpoint_n_particles:] = np.random.uniform(
                        min_vel, max_vel, (new_particles, self.dimensions)
                    )
                else:
                    # Default velocity initialization
                    if self.bounds is not None:
                        min_bounds, max_bounds = self.bounds
                        vel_range = 0.1 * (max_bounds - min_bounds)
                        self.swarm.velocity[checkpoint_n_particles:] = np.random.uniform(
                            -vel_range, vel_range, (new_particles, self.dimensions)
                        )
                    else:
                        self.swarm.velocity[checkpoint_n_particles:] = np.random.uniform(
                            -0.1, 0.1, (new_particles, self.dimensions)
                        )
                
                # Initialize pbest for new particles (will be set properly in first iteration)
                self.swarm.pbest_pos[checkpoint_n_particles:] = self.swarm.position[checkpoint_n_particles:]
                
                self.log_message(f"📈 Expanded swarm from {checkpoint_n_particles} to {current_n_particles} particles (added {new_particles} new particles)")
            
            # Restore other state variables
            self.cost_history = checkpoint_data.get("cost_history", [])
            self.pos_history = checkpoint_data.get("pos_history", [])
            self.generation_times = checkpoint_data.get("generation_times", [])
            self.best_fitness_history = checkpoint_data.get("best_fitness_history", [])
            self.stagnation_count = checkpoint_data.get("stagnation_count", 0)
            self.last_improvement_iter = checkpoint_data.get("last_improvement_iter", 0)
            
            # Restore exploration tracking data (backwards compatible)
            if "visited_grid" in checkpoint_data and checkpoint_data["visited_grid"] is not None:
                # Convert old dense grid to sparse format
                old_visited_grid = checkpoint_data["visited_grid"]
                self.grid_resolution = checkpoint_data.get("grid_resolution", None)
                self._total_grid_cells = checkpoint_data.get("total_grid_cells", None)
                
                # Convert dense grid to sparse visited_cells set
                if not hasattr(self, 'visited_cells'):
                    self.visited_cells = set()
                
                for coords in np.ndindex(old_visited_grid.shape):
                    if old_visited_grid[coords]:
                        self.visited_cells.add(coords)
                
                visited_count = len(self.visited_cells)
                self.log_message(f"🗺️ Converted old dense grid to sparse format: {visited_count:,} cells visited")
            else:
                # Backwards compatibility - no exploration data in old checkpoints
                self.log_message("🗺️ No exploration data in checkpoint - will initialize fresh sparse grid")
            
            # Restore competitive evolution data (backwards compatible)
            self.visited_cells = checkpoint_data.get("visited_cells", set())
            self.global_improvement_history = checkpoint_data.get("global_improvement_history", [])
            self.cell_fitness_history = checkpoint_data.get("cell_fitness_history", {})
            
            # Convert cell_fitness_history back to deques with proper maxlen
            for grid_coords in self.cell_fitness_history:
                fitness_list = self.cell_fitness_history[grid_coords]
                self.cell_fitness_history[grid_coords] = deque(fitness_list, maxlen=self.stagnation_window)
            
            # Restore blacklist data (backwards compatible)
            self.blacklisted_cells = checkpoint_data.get("blacklisted_cells", set())
            self.cell_fitness_tracking = checkpoint_data.get("cell_fitness_tracking", {})
            self.cell_stagnation_tracking = checkpoint_data.get("cell_stagnation_tracking", {})
            
            # Convert blacklist tracking back to deques with proper maxlen
            for grid_coords in self.cell_fitness_tracking:
                tracking_list = self.cell_fitness_tracking[grid_coords]
                self.cell_fitness_tracking[grid_coords] = deque(tracking_list, maxlen=self.blacklist_window)
            
            for grid_coords in self.cell_stagnation_tracking:
                tracking_list = self.cell_stagnation_tracking[grid_coords]
                self.cell_stagnation_tracking[grid_coords] = deque(tracking_list, maxlen=self.blacklist_window)
            
            if self.visited_cells:
                self.log_message(f"⚔️ Restored competitive evolution: {len(self.visited_cells):,} visited cells")
            
            if self.blacklisted_cells:
                self.log_message(f"🚫 Restored blacklist: {len(self.blacklisted_cells):,} blacklisted cells")
            
            # Restore hill scout data (backwards compatible)
            if self.scout_config['enable_scouts']:
                self.active_scout_groups = checkpoint_data.get("active_scout_groups", {})
                self.scout_particles = checkpoint_data.get("scout_particles", {})
                self.spawning_history = checkpoint_data.get("spawning_history", [])
                self.attribution_records = checkpoint_data.get("attribution_records", [])
                self._next_group_id = checkpoint_data.get("_next_group_id", 0)
                self._next_scout_id = checkpoint_data.get("_next_scout_id", 0)
                
                # Clean up old phase-related fields from groups (backward compatibility)
                for group_id, group in self.active_scout_groups.items():
                    # Remove old phase system fields if they exist
                    group.pop('phase', None)
                    group.pop('ocean_hit_percentage', None)
                    group.pop('best_M_points', None)
                    group.pop('current_radius', None)
                    group.pop('radius_increment', None)
                    group.pop('samples_taken', None)
                    group.pop('ocean_hits', None)
                    group.pop('radial_sampling_iterations', None)
                
                # Validate and clean up scout particles
                invalid_scouts = []
                for scout_id, scout in self.scout_particles.items():
                    # Check if position exists and is valid
                    if 'position' not in scout or scout['position'] is None:
                        invalid_scouts.append(scout_id)
                    elif not isinstance(scout['position'], np.ndarray):
                        invalid_scouts.append(scout_id)
                    elif scout['position'].shape != (self.dimensions,):
                        invalid_scouts.append(scout_id)
                
                # Remove invalid scouts
                for scout_id in invalid_scouts:
                    del self.scout_particles[scout_id]
                    self.log_message(f"⚠️ Removed invalid scout {scout_id} during checkpoint load")
                
                # Remove groups that have no valid scouts
                invalid_groups = []
                for group_id, group in self.active_scout_groups.items():
                    scout_ids = group.get('scout_ids', [])
                    valid_scout_ids = [cid for cid in scout_ids if cid in self.scout_particles]
                    
                    if len(valid_scout_ids) == 0:
                        invalid_groups.append(group_id)
                    else:
                        # Update group with only valid scouts
                        group['scout_ids'] = valid_scout_ids
                
                # Remove invalid groups
                for group_id in invalid_groups:
                    del self.active_scout_groups[group_id]
                    self.log_message(f"⚠️ Removed invalid scout group {group_id} (no valid scouts)")
                
                # Log scout restoration
                n_groups = len(self.active_scout_groups)
                n_scouts = len(self.scout_particles)
                n_spawning_events = len(self.spawning_history)
                n_attributions = len(self.attribution_records)
                
                if n_groups > 0 or n_scouts > 0:
                    self.log_message(
                        f"🏔️ Restored scouts: {n_groups} active groups, {n_scouts} scouts, "
                        f"{n_spawning_events} spawning events, {n_attributions} attributions"
                    )
                    
                    if invalid_scouts or invalid_groups:
                        self.log_message(
                            f"   🧹 Cleaned up: {len(invalid_scouts)} invalid scouts, {len(invalid_groups)} invalid groups"
                        )
            
            start_iter = checkpoint_data["iteration"] + 1
            self.log_message(f"✅ Resumed from iteration {checkpoint_data['iteration']}, best fitness: {self.swarm.best_cost:.6e}")
            return start_iter
            
        except Exception as e:
            self.log_message(f"⚠️ Failed to load checkpoint: {e}, starting fresh")
            return None

    def _generate_latin_hypercube_positions(self, n_particles, bounds):
        """Generate initial positions using Latin Hypercube Sampling for better space coverage"""
        if not SCIPY_AVAILABLE:
            self.log_message("⚠️ SciPy not available, falling back to uniform random initialization")
            return self._generate_uniform_positions(n_particles, bounds)
        
        try:
            sampler = qmc.LatinHypercube(d=self.dimensions, seed=np.random.randint(0, 2**31))
            sample = sampler.random(n=n_particles)
            
            if bounds is not None:
                lower_bounds, upper_bounds = bounds
                positions = qmc.scale(sample, lower_bounds, upper_bounds)
            else:
                # Scale to [-center, center] if no bounds
                positions = qmc.scale(sample, -self.center, self.center)
            
            # self.log_message(f"🎯 Initialized {n_particles} particles using Latin Hypercube Sampling")
            return positions
            
        except Exception as e:
            self.log_message(f"⚠️ LHS initialization failed: {e}, falling back to uniform")
            return self._generate_uniform_positions(n_particles, bounds)

    def _generate_sobol_positions(self, n_particles, bounds):
        """Generate initial positions using Sobol sequences for low-discrepancy coverage"""
        if not SCIPY_AVAILABLE:
            self.log_message("⚠️ SciPy not available, falling back to uniform random initialization")
            return self._generate_uniform_positions(n_particles, bounds)
        
        try:
            sampler = qmc.Sobol(d=self.dimensions, scramble=True, seed=np.random.randint(0, 2**31))
            # Generate next power of 2 >= n_particles for Sobol efficiency
            n_sobol = 2**int(np.ceil(np.log2(n_particles)))
            sample = sampler.random(n=n_sobol)
            
            # Take only the particles we need
            sample = sample[:n_particles]
            
            if bounds is not None:
                lower_bounds, upper_bounds = bounds
                positions = qmc.scale(sample, lower_bounds, upper_bounds)
            else:
                positions = qmc.scale(sample, -self.center, self.center)
            
            self.log_message(f"🌐 Initialized {n_particles} particles using Sobol sequences")
            return positions
            
        except Exception as e:
            self.log_message(f"⚠️ Sobol initialization failed: {e}, falling back to uniform")
            return self._generate_uniform_positions(n_particles, bounds)

    def _generate_stratified_positions(self, n_particles, bounds):
        """Generate positions using stratified sampling - divide each dimension into equal segments"""
        try:
            if bounds is None:
                lower_bounds = np.full(self.dimensions, -self.center)
                upper_bounds = np.full(self.dimensions, self.center)
            else:
                lower_bounds, upper_bounds = bounds
            
            positions = np.zeros((n_particles, self.dimensions))
            
            # For each dimension, create stratified samples
            for dim in range(self.dimensions):
                # Divide the dimension into n_particles equal segments
                segments = np.linspace(lower_bounds[dim], upper_bounds[dim], n_particles + 1)
                
                # Sample randomly within each segment
                for i in range(n_particles):
                    positions[i, dim] = np.random.uniform(segments[i], segments[i + 1])
            
            # Shuffle particles to avoid correlation between dimensions
            np.random.shuffle(positions)
            
            self.log_message(f"📊 Initialized {n_particles} particles using stratified sampling")
            return positions
            
        except Exception as e:
            self.log_message(f"⚠️ Stratified initialization failed: {e}, falling back to uniform")
            return self._generate_uniform_positions(n_particles, bounds)

    def _generate_opposition_based_positions(self, n_particles, bounds):
        """Generate positions using Opposition-Based Learning - for each random particle, create its opposite"""
        try:
            if bounds is None:
                lower_bounds = np.full(self.dimensions, -self.center)
                upper_bounds = np.full(self.dimensions, self.center)
            else:
                lower_bounds, upper_bounds = bounds
            
            # Generate half the particles randomly
            half_particles = n_particles // 2
            random_positions = np.random.uniform(lower_bounds, upper_bounds, (half_particles, self.dimensions))
            
            # Generate opposite positions: opposite = lower + upper - original
            opposite_positions = lower_bounds + upper_bounds - random_positions
            
            # Combine random and opposite positions
            if n_particles % 2 == 0:
                positions = np.vstack([random_positions, opposite_positions])
            else:
                # Add one more random particle if odd number
                extra_particle = np.random.uniform(lower_bounds, upper_bounds, (1, self.dimensions))
                positions = np.vstack([random_positions, opposite_positions, extra_particle])
            
            # Shuffle to mix random and opposite particles
            np.random.shuffle(positions)
            
            self.log_message(f"🔄 Initialized {n_particles} particles using Opposition-Based Learning")
            return positions
            
        except Exception as e:
            self.log_message(f"⚠️ Opposition-based initialization failed: {e}, falling back to uniform")
            return self._generate_uniform_positions(n_particles, bounds)

    def _generate_hybrid_positions(self, n_particles, bounds):
        """Generate positions using a hybrid approach combining multiple initialization strategies"""
        try:
            # Divide particles among different strategies
            n_lhs = n_particles // 4
            n_sobol = n_particles // 4  
            n_stratified = n_particles // 4
            n_opposition = n_particles - n_lhs - n_sobol - n_stratified  # Remainder
            
            positions_list = []
            
            # Latin Hypercube Sampling
            if n_lhs > 0:
                lhs_pos = self._generate_latin_hypercube_positions(n_lhs, bounds)
                positions_list.append(lhs_pos)
            
            # Sobol sequences
            if n_sobol > 0:
                sobol_pos = self._generate_sobol_positions(n_sobol, bounds)
                positions_list.append(sobol_pos)
            
            # Stratified sampling
            if n_stratified > 0:
                strat_pos = self._generate_stratified_positions(n_stratified, bounds)
                positions_list.append(strat_pos)
            
            # Opposition-based learning
            if n_opposition > 0:
                opp_pos = self._generate_opposition_based_positions(n_opposition, bounds)
                positions_list.append(opp_pos)
            
            # Combine all positions
            positions = np.vstack(positions_list)
            
            # Shuffle to mix different initialization strategies
            np.random.shuffle(positions)
            
            self.log_message(f"🎭 Initialized {n_particles} particles using hybrid approach (LHS:{n_lhs}, Sobol:{n_sobol}, Stratified:{n_stratified}, Opposition:{n_opposition})")
            return positions
            
        except Exception as e:
            self.log_message(f"⚠️ Hybrid initialization failed: {e}, falling back to uniform")
            return self._generate_uniform_positions(n_particles, bounds)

    def _generate_uniform_positions(self, n_particles, bounds):
        """Generate positions using uniform random sampling (fallback method)"""
        if bounds is None:
            positions = np.random.uniform(-self.center, self.center, (n_particles, self.dimensions))
        else:
            lower_bounds, upper_bounds = bounds
            positions = np.random.uniform(lower_bounds, upper_bounds, (n_particles, self.dimensions))
        
        self.log_message(f"🎲 Initialized {n_particles} particles using uniform random sampling")
        return positions

    def _handle_history_on_reevaluation(self, old_global_best, new_global_best):
        """Handle history data when objective function may have changed during checkpoint resume
        
        Parameters
        ----------
        old_global_best : float
            Global best fitness before reevaluation
        new_global_best : float
            Global best fitness after reevaluation
        """
        # Calculate the magnitude of change
        change_magnitude = abs(new_global_best - old_global_best)
        relative_change = change_magnitude / (abs(old_global_best) + 1e-10)  # Avoid division by zero
        
        # Determine if the change is significant enough to warrant history clearing
        significant_change_threshold = getattr(self, 'significant_change_threshold', 0.01)  # 1% relative change
        
        if relative_change > significant_change_threshold:
            # Significant change detected - clear potentially outdated history
            self.log_message(f"⚠️ Significant fitness change detected ({relative_change:.2%}), clearing potentially outdated history")
            
            # Clear internal PSO history (cost_history, pos_history)
            self.cost_history = []
            self.pos_history = []
            self.best_fitness_history = []
            
            # Clear detailed history for analysis
            self.detailed_history = {
                'positions': [],
                'fitness_scores': [],
                'iteration': [],
                'global_best_pos': [],
                'global_best_cost': [],
                'parameter_names': self.detailed_history.get('parameter_names')  # Preserve parameter names
            }
            
            # Clear the history file to start fresh
            if self.history_data_path and os.path.exists(self.history_data_path):
                try:
                    os.remove(self.history_data_path)
                    self.log_message(f"🗑️ Cleared history file: {self.history_data_path}")
                except Exception as e:
                    self.log_message(f"⚠️ Failed to clear history file: {e}")
            
            # Reset history loading flag
            if hasattr(self, '_history_loaded'):
                delattr(self, '_history_loaded')
            
            self.log_message("🔄 History cleared due to objective function changes - starting fresh tracking")
        else:
            # Minor change - keep history but add a note
            self.log_message(f"✅ Minor fitness change ({relative_change:.2%}), keeping existing history")

    def set_history_clear_threshold(self, threshold=0.01):
        """Set the threshold for clearing history when objective function changes
        
        Parameters
        ----------
        threshold : float
            Relative change threshold (default: 0.01 = 1%)
            If the relative change in global best fitness exceeds this threshold
            when resuming from checkpoint, history will be cleared
        """
        self.significant_change_threshold = threshold
        self.log_message(f"📊 Set history clear threshold to {threshold:.2%}")

    def set_grid_resolution(self, resolution=None):
        """Set the grid resolution for search space exploration tracking
        
        Parameters
        ----------
        resolution : int, optional
            Number of bins per dimension for the exploration grid.
            If None, uses adaptive resolution based on problem dimensions.
            Higher values give more precise tracking but use more memory.
        """
        if resolution is not None:
            self.grid_resolution = resolution
            # Clear existing sparse grid to use new resolution
            self.visited_cells.clear()
            self._total_grid_cells = resolution ** self.dimensions
            self.log_message(f"🔢 Set sparse grid resolution to {resolution}^{self.dimensions} = {resolution**self.dimensions:,} cells")
        else:
            self.log_message("🔢 Using adaptive sparse grid resolution based on problem dimensions")

    def reset_exploration_tracking(self):
        """Reset the sparse exploration grid to start fresh tracking"""
        self.visited_cells.clear()
        if hasattr(self, '_discovered_coords'):
            self._discovered_coords.clear()
        self.log_message("🔄 Reset sparse exploration tracking grid")

    def set_pca_visualization(self, enable=True, width=150, height=25, graphs_path="logs/graphs.log"):
        """Configure PCA visualization settings
        
        Parameters
        ----------
        enable : bool
            Enable/disable PCA visualization (default: True)
        width : int
            Width of the ASCII grid in characters (default: 140 for 159-char terminal)
        height : int
            Height of the ASCII grid in lines (default: 25 for 34-line terminal)
        graphs_path : str
            Path to write visualization graphs (default: "logs/graphs.log")
        """
        self.enable_pca_visualization = enable
        self.pca_grid_width = width
        self.pca_grid_height = height
        self.graphs_path = graphs_path
        self.log_message(f"📊 PCA visualization: {'enabled' if enable else 'disabled'} (grid: {width}x{height})")

    def _create_pca_visualization(self, iteration):
        """Create PCA-based ASCII visualization of particle distribution"""
        if not self.enable_pca_visualization or not RICH_AVAILABLE or not SKLEARN_AVAILABLE:
            if not SKLEARN_AVAILABLE and iteration == 0:  # Only warn once
                self.log_message("⚠️ sklearn not available, PCA visualization disabled")
            return
        
        # Get current particle positions
        positions = self.swarm.position
        
        if positions.shape[0] < 2 or positions.shape[1] < 2:
            return  # Need at least 2 particles and 2 dimensions
        
        # Fit PCA if not already done or refit periodically
        if self.pca is None or iteration % 10 == 0:  # Refit every 10 iterations
            self.pca = PCA(n_components=2)
            self.pca.fit(positions)
        
        # Transform positions to 2D
        positions_2d = self.pca.transform(positions)
        
        # Also transform personal best positions for more comprehensive view
        pbest_2d = self.pca.transform(self.swarm.pbest_pos)
        
        # Transform hill scout positions if available
        scout_positions_2d = None
        if hasattr(self, 'scout_particles') and len(self.scout_particles) > 0:
            scout_positions = []
            for scout_id, scout in self.scout_particles.items():
                if 'position' in scout and scout['position'] is not None:
                    scout_positions.append(scout['position'])
            
            if scout_positions:
                scout_positions = np.array(scout_positions)
                scout_positions_2d = self.pca.transform(scout_positions)
        
        # Find global best position in 2D
        global_best_idx = np.argmin(self.swarm.pbest_cost)
        global_best_2d = pbest_2d[global_best_idx]
        
        # Create ASCII grid
        grid_width = self.pca_grid_width
        grid_height = self.pca_grid_height
        density_grid = np.zeros((grid_height, grid_width))
        
        # Combine current and personal best positions for density calculation
        all_positions_2d = np.vstack([positions_2d, pbest_2d])
        if scout_positions_2d is not None:
            all_positions_2d = np.vstack([all_positions_2d, scout_positions_2d])
        
        # Find bounds for the grid
        min_x, max_x = np.min(all_positions_2d[:, 0]), np.max(all_positions_2d[:, 0])
        min_y, max_y = np.min(all_positions_2d[:, 1]), np.max(all_positions_2d[:, 1])
        
        # Add padding to avoid edge effects
        padding_x = (max_x - min_x) * 0.1
        padding_y = (max_y - min_y) * 0.1
        min_x -= padding_x
        max_x += padding_x
        min_y -= padding_y
        max_y += padding_y
        
        # Map positions to grid indices
        if max_x > min_x and max_y > min_y:
            for pos in all_positions_2d:
                grid_x = int((pos[0] - min_x) / (max_x - min_x) * (grid_width - 1))
                grid_y = int((pos[1] - min_y) / (max_y - min_y) * (grid_height - 1))
                grid_x = max(0, min(grid_x, grid_width - 1))
                grid_y = max(0, min(grid_y, grid_height - 1))
                density_grid[grid_y, grid_x] += 1
        
        # Find global best position in grid
        if max_x > min_x and max_y > min_y:
            best_grid_x = int((global_best_2d[0] - min_x) / (max_x - min_x) * (grid_width - 1))
            best_grid_y = int((global_best_2d[1] - min_y) / (max_y - min_y) * (grid_height - 1))
            best_grid_x = max(0, min(best_grid_x, grid_width - 1))
            best_grid_y = max(0, min(best_grid_y, grid_height - 1))
        else:
            best_grid_x = grid_width // 2
            best_grid_y = grid_height // 2
        
        # Convert density to Rich colored characters with smooth gradient
        max_density = np.max(density_grid) if np.max(density_grid) > 0 else 1
        
        # Smooth color gradient from purple to red based on exact density values
        # Using Rich's RGB color support for true gradient coloring
        from rich.text import Text
        
        def get_gradient_color(normalized_density):
            """
            Create a smooth continuous gradient from purple (0.0) to red (1.0)
            Using HSV color space for smooth rainbow-like transition
            
            Hue range: 280° (purple/violet) → 0° (red)
            This creates a reverse rainbow: purple → blue → cyan → green → yellow → orange → red
            """
            if normalized_density <= 0.0:
                return (128, 0, 128)  # Purple for zero density
            
            # Map normalized_density (0-1) to hue (280-0 degrees)
            # 280° is purple/violet, 0° is red
            hue = 280 - (normalized_density * 280)
            
            # Full saturation and value for vibrant colors
            saturation = 1.0
            value = 1.0
            
            # Convert HSV to RGB
            # HSV to RGB conversion algorithm
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
            
            # Convert to 0-255 range
            r = int((r_prime + m) * 255)
            g = int((g_prime + m) * 255)
            b = int((b_prime + m) * 255)
            
            return (r, g, b)
        
        def get_colored_char(density):
            """Get colored character based on exact density value"""
            if density == 0:
                return Text("·", style="dim black")  # Empty
            
            # Normalize density to 0-1 range
            normalized = density / max_density
            
            # Get RGB color for this density
            r, g, b = get_gradient_color(normalized)
            
            # Create Rich color string
            color_str = f"rgb({r},{g},{b})"
            
            return Text("▪", style=color_str)
        
        def get_style_for_density(density):
            """Get style string for a given density value"""
            if density == 0:
                return "dim black"
            
            # Normalize density to 0-1 range
            normalized = density / max_density
            
            # Get RGB color for this density
            r, g, b = get_gradient_color(normalized)
            
            return f"rgb({r},{g},{b})"
        
        # Create the visualization
        lines = []
        
        # Add compact header with axis labels (every 20 chars for readability)
        header = "    " + "".join([f"{i:2d}" if i % 20 == 0 else "  " for i in range(0, grid_width, 2)])
        lines.append(header[:grid_width + 4])  # Truncate if too long
        
        # Build the grid using Rich Text objects for proper coloring
        grid_lines = []
        for y in range(grid_height):
            line_text = Text(f"{y:2d} ")
            for x in range(grid_width):
                density = density_grid[y, x]
                
                if x == best_grid_x and y == best_grid_y:
                    # Global best marker with color based on density
                    style_for_best = get_style_for_density(density)
                    line_text.append("@", style=style_for_best)
                else:
                    # Regular cell with gradient color based on exact density
                    colored_char = get_colored_char(density)
                    line_text.append(colored_char)
            
            grid_lines.append(line_text)
        
        # Convert Rich Text objects to strings for the panel
        lines = []
        # Add compact header
        header = "    " + "".join([f"{i:2d}" if i % 20 == 0 else "  " for i in range(0, grid_width, 2)])
        lines.append(header[:grid_width + 4])
        
        # Add grid lines (we'll handle coloring in the panel)
        for line_text in grid_lines:
            lines.append(str(line_text))
        
        # Add legend and statistics
        variance_explained = np.sum(self.pca.explained_variance_ratio_) * 100
        lines.append("")
        lines.append("Legend: ·=empty ▪=density (smooth gradient: purple→blue→cyan→green→yellow→red) @=global best (colored by density)")
        lines.append(f"PC1 vs PC2 | Variance explained: {variance_explained:.1f}% | Particles: {len(positions)} | Max density: {int(max_density)}")
        
        # Create Rich panel with colored content
        if RICH_AVAILABLE and hasattr(self, 'console'):
            # Create a combined Rich Text object for the entire visualization
            full_viz = Text()
            
            # Add header
            full_viz.append(lines[0] + "\n")
            
            # Add colored grid lines
            for line_text in grid_lines:
                full_viz.append(line_text)
                full_viz.append("\n")
            
            # Add legend and stats
            full_viz.append("\n")
            full_viz.append(lines[-2] + "\n")
            full_viz.append(lines[-1])
            
            panel = Panel(
                full_viz,
                title=f"🗺️ Search Space Visualization - Iteration {iteration}",
                border_style="cyan",
                padding=(1, 2)
            )
            
            # Write to graphs file
            import contextlib
            with open(self.graphs_path, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                self.console.print(panel)
                # No extra spacing - panels will be adjacent

    def set_competition_config(self, enable=True, max_particles_per_cell=2, stagnation_window=15, eviction_percentage=40, check_interval=10):
        """Configure grid-based competitive evolution settings
        
        Parameters
        ----------
        enable : bool
            Enable/disable competitive evolution (default: True)
        max_particles_per_cell : int
            Maximum number of particles allowed per grid cell (default: 2)
        stagnation_window : int
            Number of iterations with unchanged cell-best to consider stagnant (default: 15)
        eviction_percentage : int
            Percentage of worst particles to evict from stagnant cells (default: 40)
        check_interval : int
            Check for competition every X iterations (default: 10)
        """
        self.competition_enabled = enable
        self.max_particles_per_cell = max_particles_per_cell
        self.stagnation_window = stagnation_window
        self.eviction_percentage = eviction_percentage
        self.competition_check_interval = check_interval
        
        # Update existing cell fitness history deques with new maxlen
        if hasattr(self, 'cell_fitness_history'):
            for grid_coords in self.cell_fitness_history:
                old_deque = self.cell_fitness_history[grid_coords]
                self.cell_fitness_history[grid_coords] = deque(old_deque, maxlen=stagnation_window)
        
        self.log_message(f"⚔️ Competitive evolution: {'enabled' if enable else 'disabled'}")
        if enable:
            self.log_message(f"📊 Max {max_particles_per_cell} particles/cell, {eviction_percentage}% eviction, {stagnation_window}-iter stagnation window")
            self.log_message(f"🔄 Competition check every {check_interval} iterations")

    def set_blacklist_config(self, fitness_threshold=1e5, blacklist_window=100):
        """Configure cell blacklisting system settings
        
        Parameters
        ----------
        fitness_threshold : float
            Fitness threshold for blacklisting - cells with no fitness < threshold get blacklisted (default: 1e5)
        blacklist_window : int
            Number of iterations to track for blacklisting decisions (default: 100)
        """
        self.blacklist_fitness_threshold = fitness_threshold
        self.blacklist_window = blacklist_window
        
        # Update existing tracking deques with new maxlen
        if hasattr(self, 'cell_fitness_tracking'):
            for grid_coords in self.cell_fitness_tracking:
                old_deque = self.cell_fitness_tracking[grid_coords]
                self.cell_fitness_tracking[grid_coords] = deque(old_deque, maxlen=blacklist_window)
        
        if hasattr(self, 'cell_stagnation_tracking'):
            for grid_coords in self.cell_stagnation_tracking:
                old_deque = self.cell_stagnation_tracking[grid_coords]
                self.cell_stagnation_tracking[grid_coords] = deque(old_deque, maxlen=blacklist_window)
        
        self.log_message(f"🚫 Cell blacklisting: fitness threshold {fitness_threshold:.0e}, window {blacklist_window} iterations")

    def set_velocity_boost_config(self, interval=10, fraction=0.20, enable=True, optimize_limits_len=5, use_exploration_prediction=True, n_alternative_swarms=200, prediction_steps=10, grid_refinement_threshold=95.0, max_grid_resolution=50):
        """Configure velocity boost settings with sophisticated particle selection
        
        Parameters
        ----------
        interval : int
            Apply velocity boost every X iterations (default: 10)
        fraction : float
            Maximum fraction of particles to boost (default: 0.20 = 20%, max 90%)
        enable : bool
            Enable/disable velocity boost (default: True)
        optimize_limits_len : int
            Length of optimize limits for fitness threshold calculation (default: 5)
        use_exploration_prediction : bool
            Enable multi-step exploration prediction for boost direction (default: True)
        n_alternative_swarms : int
            Number of alternative mini-swarms to test for exploration prediction (default: 200)
        prediction_steps : int
            Number of PSO iterations to simulate for each alternative (default: 10)
        grid_refinement_threshold : float
            Exploration saturation percentage at which to refine grid (default: 95.0%)
        max_grid_resolution : int
            Maximum allowed grid resolution to prevent memory issues (default: 50)
        """
        self.reinit_interval = interval
        self.reinit_fraction = fraction  # This is now the max fraction (up to 90%)
        self.enable_reinit = enable
        self.optimize_limits_len = optimize_limits_len
        self.use_exploration_prediction = use_exploration_prediction
        self.n_alternative_swarms = n_alternative_swarms
        self.prediction_steps = prediction_steps
        self.grid_refinement_threshold = grid_refinement_threshold
        self.max_grid_resolution = max_grid_resolution
        
        fitness_threshold = 10 ** optimize_limits_len
        self.log_message(f"🚀 Smart velocity boost: {'enabled' if enable else 'disabled'} (every {interval} iters)")
        self.log_message(f"📊 Selection: 20%-90% particles, fitness threshold: {fitness_threshold:.0e}")
        if use_exploration_prediction:
            self.log_message(f"🎯 Multi-step exploration prediction: enabled ({n_alternative_swarms} alternatives × {prediction_steps} steps)")
            self.log_message(f"🔍 Grid refinement: enabled at {grid_refinement_threshold}% saturation (max resolution: {max_grid_resolution})")

    def set_scout_config(self, enable_scouts=None, scouts_per_spawn=None, scout_lifetime=None, 
                        adaptive_negative_fitness_threshold_enabled=None, negative_fitness_threshold=None,
                        spawn_lhs_percentage=None, max_concurrent_groups=None, local_search_neighbors=None,
                        initial_search_radius_percentage=None, radius_shrink_factor=None, 
                        min_search_radius_percentage=None):
        """Configure hill scout functionality settings
        
        This method allows updating scout configuration after initialization. Only provided
        parameters will be updated; others will retain their current values.
        
        Parameters
        ----------
        enable_scouts : bool, optional
            Enable/disable hill scout functionality
        scouts_per_spawn : int, optional
            Number of hill scout particles per spawn
        scout_lifetime : int, optional
            Lifetime of hill scout groups in iterations
        adaptive_negative_fitness_threshold_enabled : bool, optional
            Enable adaptive threshold (becomes 0 when global best < 0)
        negative_fitness_threshold : float, optional
            Fitness threshold for spawning hill scouts
        spawn_lhs_percentage : float, optional
            LHS hypercube size as percentage of parameter ranges
        max_concurrent_groups : int, optional
            Maximum number of concurrent hill scout groups
        local_search_neighbors : int, optional
            Number of neighbors to generate per scout
        initial_search_radius_percentage : float, optional
            Initial search radius as percentage of parameter ranges
        radius_shrink_factor : float, optional
            Shrink factor when no improvement (0.0-1.0)
        min_search_radius_percentage : float, optional
            Minimum search radius as percentage of parameter ranges
            
        Examples
        --------
        >>> # Enable scouts with custom settings
        >>> optimizer.set_scout_config(
        ...     enable_scouts=True,
        ...     scouts_per_spawn=100,
        ...     scout_lifetime=200
        ... )
        
        >>> # Disable scouts
        >>> optimizer.set_scout_config(enable_scouts=False)
        
        >>> # Update only specific parameters
        >>> optimizer.set_scout_config(
        ...     negative_fitness_threshold=1e4,
        ...     max_concurrent_groups=50
        ... )
        """
        # Update only provided parameters
        if enable_scouts is not None:
            self.scout_config['enable_scouts'] = enable_scouts
        if scouts_per_spawn is not None:
            self.scout_config['scouts_per_spawn'] = scouts_per_spawn
        if scout_lifetime is not None:
            self.scout_config['scout_lifetime'] = scout_lifetime
        if adaptive_negative_fitness_threshold_enabled is not None:
            self.scout_config['adaptive_negative_fitness_threshold_enabled'] = adaptive_negative_fitness_threshold_enabled
        if negative_fitness_threshold is not None:
            self.scout_config['negative_fitness_threshold'] = negative_fitness_threshold
        if spawn_lhs_percentage is not None:
            self.scout_config['spawn_lhs_percentage'] = spawn_lhs_percentage
        if max_concurrent_groups is not None:
            self.scout_config['max_concurrent_groups'] = max_concurrent_groups
        if local_search_neighbors is not None:
            self.scout_config['local_search_neighbors'] = local_search_neighbors
        if initial_search_radius_percentage is not None:
            self.scout_config['initial_search_radius_percentage'] = initial_search_radius_percentage
        if radius_shrink_factor is not None:
            self.scout_config['radius_shrink_factor'] = radius_shrink_factor
        if min_search_radius_percentage is not None:
            self.scout_config['min_search_radius_percentage'] = min_search_radius_percentage
        
        # Validate updated config
        try:
            self._validate_scout_config(self.scout_config)
        except (ValueError, TypeError) as e:
            self.log_message(f"⚠️ Scout config validation failed: {e}")
            raise
        
        # Log updated configuration
        self.log_message("🏔️ Scout configuration updated:")
        self.log_message(f"   Enable scouts: {self.scout_config['enable_scouts']}")
        if self.scout_config['enable_scouts']:
            self.log_message(f"   Scouts per spawn: {self.scout_config['scouts_per_spawn']}")
            self.log_message(f"   Scout lifetime: {self.scout_config['scout_lifetime']} iterations")
            self.log_message(f"   Adaptive threshold: {self.scout_config['adaptive_negative_fitness_threshold_enabled']}")
            self.log_message(f"   Negative fitness threshold: {self.scout_config['negative_fitness_threshold']:.0e}")
            self.log_message(f"   Spawn LHS percentage: {self.scout_config['spawn_lhs_percentage']:.2%}")
            self.log_message(f"   Max concurrent groups: {self.scout_config['max_concurrent_groups']}")
            self.log_message(f"   Local search neighbors: {self.scout_config['local_search_neighbors']}")
            self.log_message(f"   Initial search radius: {self.scout_config['initial_search_radius_percentage']:.2%}")
            self.log_message(f"   Radius shrink factor: {self.scout_config['radius_shrink_factor']:.2f}")
            self.log_message(f"   Min search radius: {self.scout_config['min_search_radius_percentage']:.2%}")

    def set_10n_selection_config(self, enable=True, multiplier=10, interval=10, on_fresh_start=True, on_checkpoint_resume=True, quality_threshold_pct=100.0, max_batches=100, dynamic_target=False):
        """Configure adaptive quality selection for continuous quality injection
        
        This method configures the adaptive quality selection strategy, which samples
        batches of multiplier*n candidates until it finds enough particles with fitness below
        the negative_fitness_threshold (from scout_config). Each batch uses fresh LHS sampling.
        
        The strategy keeps sampling until:
        - quality_threshold_pct% of n particles are found (fitness < threshold), OR
        - dynamic_target mode: matches the number of current particles above threshold, OR
        - max_batches is reached (fallback to best n from all evaluated)
        
        Parameters
        ----------
        enable : bool
            Enable/disable adaptive quality selection (default: True)
        multiplier : int
            Batch size multiplier (default: 10 for 10n candidates per batch)
        interval : int
            Run selection every X iterations (default: 10)
        on_fresh_start : bool
            Run selection on fresh start at iteration 0 (default: True)
        on_checkpoint_resume : bool
            Run selection when resuming from checkpoint (default: True)
        quality_threshold_pct : float
            Percentage of particles that must be below threshold (default: 100.0 = all particles)
            Examples: 80.0 = stop when 80% are below threshold, 50.0 = stop when 50% are below
            Note: Ignored when dynamic_target=True
        max_batches : int
            Maximum number of batches to sample before fallback (default: 100)
        dynamic_target : bool
            If True, target number adapts to current swarm state (default: False)
            When enabled, targets the number of particles currently above threshold
            This makes selection more adaptive: if 30 particles are above threshold,
            it will try to find 30 quality replacements
            
        Notes
        -----
        The fitness threshold is taken from scout_config['negative_fitness_threshold'].
        Each batch is sampled independently using LHS for good space coverage.
        
        Setting quality_threshold_pct < 100.0 allows the optimizer to stop earlier on hard
        landscapes where finding all n quality particles would take too many batches.
        
        Dynamic target mode is useful for periodic selection where you want to replace
        only the particles that are currently performing poorly (above threshold).
        
        Examples
        --------
        >>> # Require all particles below threshold (strict)
        >>> optimizer.set_10n_selection_config(
        ...     enable=True,
        ...     multiplier=10,
        ...     quality_threshold_pct=100.0
        ... )
        
        >>> # Allow stopping when 80% are below threshold (more flexible)
        >>> optimizer.set_10n_selection_config(
        ...     enable=True,
        ...     multiplier=10,
        ...     quality_threshold_pct=80.0
        ... )
        
        >>> # Dynamic target: adapt to current swarm state
        >>> optimizer.set_10n_selection_config(
        ...     enable=True,
        ...     multiplier=10,
        ...     dynamic_target=True
        ... )
        """
        self.enable_10n_selection = enable
        self.selection_multiplier = multiplier
        self.selection_interval = interval
        self.selection_on_fresh_start = on_fresh_start
        self.selection_on_checkpoint_resume = on_checkpoint_resume
        self.selection_quality_threshold_pct = quality_threshold_pct
        self.selection_max_batches = max_batches
        self.selection_dynamic_target = dynamic_target
        
        self.log_message(f"🎯 Adaptive quality selection: {'enabled' if enable else 'disabled'}")
        if enable:
            threshold = self.scout_config['negative_fitness_threshold']
            if dynamic_target:
                self.log_message(f"📊 {multiplier}n candidates per batch, dynamic target mode (adapts to current swarm state)")
                self.log_message(f"   Target: number of particles currently > {threshold:.0e}")
            else:
                target_n = int(self.n_particles * quality_threshold_pct / 100.0)
                self.log_message(f"📊 {multiplier}n candidates per batch, target: {target_n}/{self.n_particles} particles < {threshold:.0e} ({quality_threshold_pct:.0f}%)")
            self.log_message(f"🔁 Runs every {interval} iterations, max {max_batches} batches")
            self.log_message(f"🚀 Fresh start: {'enabled' if on_fresh_start else 'disabled'}")
            self.log_message(f"🔄 Checkpoint resume: {'enabled' if on_checkpoint_resume else 'disabled'}")

    def set_initialization_strategy(self, strategy="hybrid"):
        """Set the initialization strategy for particle positions
        
        Parameters
        ----------
        strategy : str
            Initialization strategy. Options:
            - 'uniform': Standard uniform random (default PySwarms behavior)
            - 'lhs': Latin Hypercube Sampling (best space-filling)
            - 'sobol': Sobol sequences (low-discrepancy)
            - 'stratified': Stratified sampling (equal segments per dimension)
            - 'opposition': Opposition-Based Learning (explore opposites)
            - 'hybrid': Combination of multiple strategies (recommended)
        """
        valid_strategies = ['uniform', 'lhs', 'sobol', 'stratified', 'opposition', 'hybrid']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")
        
        self.initialization_strategy = strategy
        self.log_message(f"🎯 Set initialization strategy to: {strategy}")

    def _calculate_search_space_exploration(self):
        """Calculate the percentage of search space explored using sparse grid coverage
        
        This method uses the sparse visited_cells set to track exploration without
        maintaining a dense grid array, making it checkpoint-friendly.
        
        Returns
        -------
        float
            Percentage of grid cells visited (0-100%)
        """
        if self.bounds is None:
            # For unbounded case, we can't calculate meaningful grid coverage
            # Return diversity as a proxy (normalized to 0-100%)
            positions = self.swarm.position
            diversity = np.mean(np.std(positions, axis=0))
            return min(diversity * 10, 100.0)  # Scale diversity to percentage
        
        # Initialize grid resolution if not exists (but don't create dense grid)
        if not hasattr(self, 'grid_resolution'):
            # Grid resolution per dimension (adjustable based on problem size)
            if self.dimensions <= 3:
                self.grid_resolution = 20  # 20^3 = 8,000 cells max
            elif self.dimensions <= 6:
                self.grid_resolution = 10  # 10^6 = 1,000,000 cells max
            elif self.dimensions <= 10:
                self.grid_resolution = 5   # 5^10 = 9,765,625 cells max
            else:
                self.grid_resolution = 3   # 3^n cells (manageable for high dimensions)
            
            self._total_grid_cells = self.grid_resolution ** self.dimensions
            self.log_message(f"🔢 Sparse grid-based exploration tracking: {self.grid_resolution}^{self.dimensions} = {self._total_grid_cells:,} cells")
        
        # Update visited cells with current positions
        self._update_visited_cells_sparse()
        
        # Calculate exploration percentage using sparse set
        visited_count = len(self.visited_cells)
        exploration_percentage = (visited_count / self._total_grid_cells) * 100.0
        
        return exploration_percentage

    def _update_visited_cells_sparse(self):
        """Update the sparse visited_cells set with current and historical positions"""
        
        if self.bounds is None:
            return
        
        # Get all historical positions (current + personal bests for comprehensive coverage)
        all_positions = np.vstack([
            self.swarm.position,           # Current positions
            self.swarm.pbest_pos          # Personal best positions
        ])
        
        # Convert positions to grid coordinates and add to sparse set
        for pos in all_positions:
            grid_coords = self._position_to_grid_coords(pos)
            if grid_coords is not None:
                self.visited_cells.add(grid_coords)

    def _boost_worst_particles_velocity(self, iteration):
        """Give particles a large velocity boost to explore new regions using sophisticated selection criteria
        
        Selection criteria:
        1) Worst fitness is prioritized
        2) Fitness which is above 10**len(optimize_limits) are to be boosted (up until 90% is reached)
        3) If (2) cannot reach 20%, we add the worst fitness from particles with fitness < 10**len(optimize_limits)
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        """
        if not self.enable_reinit:
            return
            
        # Check if it's time for velocity boost (every n iterations)
        if iteration % self.reinit_interval != 0 or iteration == 0:
            return
        
        # Use adaptive boost strategy based on exploration saturation
        if getattr(self, 'use_exploration_prediction', True):
            self._adaptive_boost_strategy(iteration)
        else:
            self._boost_particles_traditional(iteration)

    def _boost_particles_traditional(self, iteration):
        """Traditional velocity boost method (original implementation)"""
        # Get fitness threshold - estimate optimize limits length as 5 (typical for trading optimization)
        # This could be made configurable if needed
        optimize_limits_len = getattr(self, 'optimize_limits_len', 5)
        fitness_threshold = 10 ** optimize_limits_len
        global_best_particle_idx = self._get_global_best_particle_idx()
        
        # Get all particle indices and their pbest fitness values (personal best performance)
        all_indices = np.arange(self.n_particles)
        all_pbest_fitness = self.swarm.pbest_cost
        
        # Remove global best particle from consideration
        eligible_indices = all_indices[all_indices != global_best_particle_idx]
        eligible_pbest_fitness = all_pbest_fitness[eligible_indices]
        
        # Separate eligible particles into high fitness (above threshold) and low fitness (below threshold)
        high_fitness_mask = eligible_pbest_fitness >= fitness_threshold
        low_fitness_mask = eligible_pbest_fitness < fitness_threshold
        
        high_fitness_indices = eligible_indices[high_fitness_mask]
        low_fitness_indices = eligible_indices[low_fitness_mask]
        
        # Calculate min and max boost counts (adjusted for protecting global best)
        max_eligible = len(eligible_indices)
        min_boost = max(1, int(0.20 * max_eligible))  # Minimum 20% of eligible particles
        max_boost = max(1, int(0.30 * max_eligible))  # Maximum 30% of eligible particles
        
        selected_indices = []
        
        # Step 1: Add all high pbest fitness particles (above threshold), prioritizing worst first
        if len(high_fitness_indices) > 0:
            # Sort high fitness particles by pbest fitness (worst first)
            high_fitness_sorted = high_fitness_indices[np.argsort(eligible_pbest_fitness[high_fitness_mask])[::-1]]
            # Take up to max_boost particles
            selected_indices.extend(high_fitness_sorted[:max_boost])
        
        # Step 2: If we haven't reached min_boost (20%), add worst particles from low fitness group
        if len(selected_indices) < min_boost and len(low_fitness_indices) > 0:
            # Sort low fitness particles by pbest fitness (worst first)
            low_fitness_sorted = low_fitness_indices[np.argsort(eligible_pbest_fitness[low_fitness_mask])[::-1]]
            # Add particles until we reach min_boost
            needed = min_boost - len(selected_indices)
            selected_indices.extend(low_fitness_sorted[:needed])
        
        # Convert to numpy array and ensure we don't exceed max_boost
        boost_indices = np.array(selected_indices[:max_boost])
        n_boost = len(boost_indices)
        
        if n_boost == 0:
            self.log_message("⚠️ No particles selected for velocity boost")
            return
        
        # Use global best position to bias direction away from current best solution
        global_best_pos = self.swarm.best_pos
        
        # Ensure we have bounds for safe distance calculation
        if self.bounds is None:
            self.log_message("⚠️ No bounds specified, cannot perform safe velocity boost")
            return
            
        lower_bounds, upper_bounds = self.bounds
        
        # Apply velocity boost to selected particles
        boosted_particles = []
        high_fitness_count = 0
        low_fitness_count = 0
        
        for idx in boost_indices:
            current_pos = self.swarm.position[idx]
            pbest_fitness = self.swarm.pbest_cost[idx]
            
            # Track selection categories
            if pbest_fitness >= fitness_threshold:
                high_fitness_count += 1
            else:
                low_fitness_count += 1
            
            # Calculate direction away from global best position
            away_vector = current_pos - global_best_pos
            away_norm = np.linalg.norm(away_vector)
            
            # Handle case where particle is at global best position
            if away_norm < 1e-10:
                away_vector = np.random.uniform(-1, 1, self.dimensions)
                away_norm = np.linalg.norm(away_vector)
            
            away_vector = away_vector / away_norm  # Normalize
            
            # Add randomness to direction (70% away from global best, 30% random)
            random_vector = np.random.uniform(-1, 1, self.dimensions)
            random_vector = random_vector / np.linalg.norm(random_vector)
            
            # Combine directions (70% away from global best,
            direction = 0.7 * away_vector + 0.3 * random_vector
            direction = direction / np.linalg.norm(direction)  # Normalize final direction
            
            # Calculate maximum safe distance in this direction (only for moving dimensions)
            max_distances = []
            for dim in range(self.dimensions):
                if abs(direction[dim]) >= 1e-10:  # Only calculate for moving dimensions
                    if direction[dim] > 0:  # Moving toward upper bound
                        max_dist = (upper_bounds[dim] - current_pos[dim]) / direction[dim]
                        max_distances.append(max_dist)
                    else:  # Moving toward lower bound
                        max_dist = (current_pos[dim] - lower_bounds[dim]) / abs(direction[dim])
                        max_distances.append(max_dist)
            
            safe_distance = np.mean(max_distances) if max_distances else 1.0
            
            # Set magnitude to average safe distance
            magnitude = safe_distance
            
            # Apply velocity boost
            self.swarm.velocity[idx] = magnitude * direction
            
            # Reset pbest for boosted particles - give them a fresh start
            self.swarm.pbest_cost[idx] = np.inf
            self.swarm.pbest_pos[idx] = current_pos.copy()  # Reset to current position
            
            boosted_particles.append({
                'idx': idx,
                'pbest_fitness': pbest_fitness,
                'magnitude': magnitude,
                'safe_distance': safe_distance
            })
        
        # Calculate pbest fitness statistics for the particles BEFORE boosting
        particles_to_boost_pbest_fitness = [p['pbest_fitness'] for p in boosted_particles]
        before_boost_stats = {
            'min': np.min(particles_to_boost_pbest_fitness),
            'max': np.max(particles_to_boost_pbest_fitness),
            'mean': np.mean(particles_to_boost_pbest_fitness)
        }
        
        # Store the boosted particle indices for tracking in next iteration
        self.last_boosted_indices = boost_indices.copy()
        self.last_boost_iteration = iteration
        
        # Log the sophisticated velocity boost with before/after statistics
        avg_magnitude = np.mean([p['magnitude'] for p in boosted_particles])
        boost_percentage = (n_boost / self.n_particles) * 100
        
        global_best_particle_idx = self._get_global_best_particle_idx()
        self.log_message(f"🚀 Smart velocity boost applied to {n_boost}/{self.n_particles} particles ({boost_percentage:.1f}%) (global best particle #{global_best_particle_idx} protected)")
        self.log_message(f"📊 Selection: High pbest (≥{fitness_threshold:.0e}): {high_fitness_count}, Low pbest: {low_fitness_count}")
        self.log_message(f"📈 Boosted particles pbest - Min: {before_boost_stats['min']:.6e}, Avg: {before_boost_stats['mean']:.6e}, Max: {before_boost_stats['max']:.6e}")
        self.log_message("🔄 Reset pbest for all boosted particles - fresh start opportunity")
        
        # Show "after boost" statistics if we have data from previous boost
        if hasattr(self, 'last_boosted_indices') and hasattr(self, 'last_boost_iteration') and self.last_boost_iteration == iteration - self.reinit_interval:
            # Get current fitness of previously boosted particles
            prev_boosted_current_fitness = self.swarm.current_cost[self.last_boosted_indices]
            after_boost_stats = {
                'min': np.min(prev_boosted_current_fitness),
                'max': np.max(prev_boosted_current_fitness),
                'mean': np.mean(prev_boosted_current_fitness)
            }
            self.log_message(f"📉 Previous boosted particles fitness - Min: {after_boost_stats['min']:.6e}, Avg: {after_boost_stats['mean']:.6e}, Max: {after_boost_stats['max']:.6e}")
        
        self.log_message(f"⚡ Avg velocity magnitude: {avg_magnitude:.6f}")

    def _initialize_positions_with_strategy(self, n_particles, bounds):
        """Initialize particle positions using the selected strategy"""
        strategy = getattr(self, 'initialization_strategy', 'hybrid')
        
        if strategy == 'lhs':
            return self._generate_latin_hypercube_positions(n_particles, bounds)
        elif strategy == 'sobol':
            return self._generate_sobol_positions(n_particles, bounds)
        elif strategy == 'stratified':
            return self._generate_stratified_positions(n_particles, bounds)
        elif strategy == 'opposition':
            return self._generate_opposition_based_positions(n_particles, bounds)
        elif strategy == 'hybrid':
            return self._generate_hybrid_positions(n_particles, bounds)
        else:  # uniform
            return self._generate_uniform_positions(n_particles, bounds)

    def _inject_init_pos_fresh_start(self):
        """Inject init_pos particle during fresh start initialization"""
        if self.custom_init_pos is None:
            return
        
        # Validate init_pos dimensions
        init_pos = np.array(self.custom_init_pos)
        if init_pos.shape != (self.dimensions,):
            self.log_message(f"⚠️ init_pos shape {init_pos.shape} doesn't match dimensions {self.dimensions}, skipping injection")
            return
        
        # Validate bounds if they exist
        if self.bounds is not None:
            lower_bounds, upper_bounds = self.bounds
            if not np.all((init_pos >= lower_bounds) & (init_pos <= upper_bounds)):
                self.log_message("⚠️ init_pos is outside bounds, clipping to bounds")
                init_pos = np.clip(init_pos, lower_bounds, upper_bounds)
        
        # Replace the first particle with init_pos
        self.swarm.position[0] = init_pos.copy()
        
        # Give it a random velocity like other particles
        if self.velocity_clamp is not None:
            min_vel, max_vel = self.velocity_clamp
            self.swarm.velocity[0] = np.random.uniform(min_vel, max_vel, self.dimensions)
        else:
            if self.bounds is not None:
                lower_bounds, upper_bounds = self.bounds
                vel_range = 0.1 * (upper_bounds - lower_bounds)
                self.swarm.velocity[0] = np.random.uniform(-vel_range, vel_range, self.dimensions)
            else:
                self.swarm.velocity[0] = np.random.uniform(-0.1, 0.1, self.dimensions)
        
        self.log_message("🎯 Injected init_pos particle at position 0 (fresh start)")

    def _inject_init_pos_checkpoint_resume(self, objective_func, **kwargs):
        """Inject init_pos particle during checkpoint resume by replacing worst particle"""
        if self.custom_init_pos is None:
            return
        
        # Validate init_pos dimensions
        init_pos = np.array(self.custom_init_pos)
        if init_pos.shape != (self.dimensions,):
            self.log_message(f"⚠️ init_pos shape {init_pos.shape} doesn't match dimensions {self.dimensions}, skipping injection")
            return
        
        # Validate bounds if they exist
        if self.bounds is not None:
            lower_bounds, upper_bounds = self.bounds
            if not np.all((init_pos >= lower_bounds) & (init_pos <= upper_bounds)):
                self.log_message("⚠️ init_pos is outside bounds, clipping to bounds")
                init_pos = np.clip(init_pos, lower_bounds, upper_bounds)
        
        # Find the worst performing particle (highest pbest_cost)
        worst_particle_idx = np.argmax(self.swarm.pbest_cost)
        worst_fitness = self.swarm.pbest_cost[worst_particle_idx]
        
        # Replace worst particle with init_pos
        self.swarm.position[worst_particle_idx] = init_pos.copy()
        
        # Give it a random velocity
        if self.velocity_clamp is not None:
            min_vel, max_vel = self.velocity_clamp
            self.swarm.velocity[worst_particle_idx] = np.random.uniform(min_vel, max_vel, self.dimensions)
        else:
            if self.bounds is not None:
                lower_bounds, upper_bounds = self.bounds
                vel_range = 0.1 * (upper_bounds - lower_bounds)
                self.swarm.velocity[worst_particle_idx] = np.random.uniform(-vel_range, vel_range, self.dimensions)
            else:
                self.swarm.velocity[worst_particle_idx] = np.random.uniform(-0.1, 0.1, self.dimensions)
        
        # Evaluate the new particle to get its fitness
        # Create a temporary swarm with just this particle for evaluation
        temp_swarm = type('TempSwarm', (), {})()
        temp_swarm.position = init_pos.reshape(1, -1)
        temp_fitness = compute_objective_function(temp_swarm, objective_func, pool=None, **kwargs)
        
        # Update the particle's costs
        self.swarm.current_cost[worst_particle_idx] = temp_fitness[0]
        self.swarm.pbest_cost[worst_particle_idx] = temp_fitness[0]
        self.swarm.pbest_pos[worst_particle_idx] = init_pos.copy()
        
        # Recompute global best in case the new particle is better
        old_global_best = self.swarm.best_cost
        self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)
        
        # Log the injection
        if self.swarm.best_cost < old_global_best:
            improvement = old_global_best - self.swarm.best_cost
            self.log_message(f"🌟 init_pos particle replaced worst (fitness: {worst_fitness:.6e}) and became NEW GLOBAL BEST! (fitness: {temp_fitness[0]:.6e}, improvement: {improvement:.6e})")
            
            # Handle scout group spawning if scouts are enabled
            if self.scout_config['enable_scouts']:
                # Spawn a new scout group at the init_pos location
                # It will automatically be immortal since its spawn_point matches the global best
                new_group_id = self._spawn_scout_group(
                    spawner_particle_id=worst_particle_idx,
                    spawn_point=init_pos.copy(),
                    trigger_fitness=temp_fitness[0],
                    iteration=0  # This happens during initialization
                )
                
                if new_group_id is not None:
                    self.log_message(
                        f"♾️ Global best group {new_group_id} spawned from init_pos (immortal)",
                        emoji="♾️"
                    )
        else:
            self.log_message(f"🎯 init_pos particle replaced worst (fitness: {worst_fitness:.6e} → {temp_fitness[0]:.6e}) at position {worst_particle_idx}")

    def optimize(
        self, objective_func, iters, n_processes=None, verbose=True, **kwargs
    ):
        """Optimize the swarm for a number of iterations with checkpoint support

        This method maintains full backward compatibility with PySwarms GlobalBestPSO.
        It performs the optimization to evaluate the objective function for a number
        of iterations, with optional hill scout enhancement.

        Parameters
        ----------
        objective_func : callable
            objective function to be evaluated
        iters : int
            number of iterations
        n_processes : int
            number of processes to use for parallel particle evaluation (default: None = no parallelization)
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        tuple
            the global best cost and the global best position (PySwarms compatible format)
            
        Raises
        ------
        TypeError
            If objective_func is not callable (PySwarms convention)
        ValueError
            If iters is not a positive integer (PySwarms convention)
            
        Notes
        -----
        **Return Format Compatibility:**
        This method returns a tuple (best_cost, best_position) which is identical
        to the original PySwarms GlobalBestPSO implementation, ensuring complete
        backward compatibility.
        
        **Scout Behavior:**
        - If scout_config['enable_scouts'] is False (default), this
          method behaves exactly like standard PySwarms PSO
        - If scout_config['enable_scouts'] is True, hill scouts
          are spawned when negative fitness regions are discovered
        
        **Error Handling:**
        All errors follow PySwarms conventions:
        - TypeError for type mismatches
        - ValueError for invalid parameter values
        - Logging uses PySwarms Reporter system
        """
        # Validate inputs following PySwarms conventions
        if not callable(objective_func):
            raise TypeError(
                f"objective_func must be callable, got {type(objective_func).__name__}"
            )
        
        if not isinstance(iters, int) or iters <= 0:
            raise ValueError(
                f"iters must be a positive integer, got {iters}"
            )
        
        if n_processes is not None and (not isinstance(n_processes, int) or n_processes <= 0):
            raise ValueError(
                f"n_processes must be a positive integer or None, got {n_processes}"
            )
        # Color constants
        CYAN = "\033[96m"
        RESET = "\033[0m"
        
        # Initialize timing
        self.start_time = time.time()
        
        # Try to load from checkpoint
        start_iter = self.load_checkpoint()
        if start_iter is None:
            start_iter = 0
            self.log_message("🚀 Starting fresh PSO optimization")
        
        # Apply verbosity
        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=log_level,
        )
        
        self.log_message(f"🐝 Starting PSO evolution from iteration {start_iter}")
        self.log_message(f"📊 Swarm size: {self.n_particles}")
        
        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        # pool = None if n_processes is None else mp.Pool(n_processes)

        # Initialize if starting fresh
        if start_iter == 0:
            # Run 10n selection if enabled for fresh start
            if self.enable_10n_selection and self.selection_on_fresh_start:
                self._run_10n_selection(iteration=0, objective_func=objective_func, is_fresh_start=True, **kwargs)
            else:
                # Standard initialization
                self._standard_initialization(objective_func, **kwargs)
            
            self.best_fitness_history = []
            self.stagnation_count = 0
            self.last_improvement_iter = 0
            
            self.log_message(f"🎯 Initial evaluation complete - Global best: {self.swarm.best_cost:.6e}")
            
            # If init_pos was injected and became the global best, spawn a scout group for it
            if self.custom_init_pos is not None and self.scout_config['enable_scouts']:
                # Check if particle 0 (init_pos) is the global best
                best_particle_idx = np.argmin(self.swarm.pbest_cost)
                if best_particle_idx == 0:
                    # Spawn a scout group at the init_pos location
                    # It will automatically be immortal since its spawn_point matches the global best
                    new_group_id = self._spawn_scout_group(
                        spawner_particle_id=0,
                        spawn_point=self.swarm.position[0].copy(),
                        trigger_fitness=self.swarm.pbest_cost[0],
                        iteration=0
                    )
                    
                    if new_group_id is not None:
                        self.log_message(
                            f"♾️ Global best group {new_group_id} spawned from init_pos (immortal)"
                        )
        else:
            # Checkpoint resume - run 10n selection if enabled
            if self.enable_10n_selection and self.selection_on_checkpoint_resume:
                self._run_10n_selection(iteration=start_iter, objective_func=objective_func, is_checkpoint_resume=True, **kwargs)
        
        # Initialize sparse exploration grid if not already done (for both fresh start and checkpoint resume)
        if not hasattr(self, 'grid_resolution') and self.bounds is not None:
            self._initialize_exploration_grid()
            # Mark initial positions as visited in sparse grid
            self._update_exploration_grid(self.swarm.position)
        
        ftol_history = deque(maxlen=self.ftol_iter)
        previous_best_fitness = self.swarm.best_cost if hasattr(self.swarm, 'best_cost') else float('inf')
        
        # Main PSO iteration loop
        for i in range(start_iter, iters):
            iter_start_time = time.time()
            
            # Store current iteration for hill scout attribution system
            self._current_iteration = i
            
            if RICH_AVAILABLE:
                self.console_wrapper(Rule(f"Iteration {i}", style="bold blue"))
            
            # Velocity boost for worst particles to escape stagnation
            self._boost_worst_particles_velocity(i)
            
            # Run 10n selection every N iterations (if enabled)
            if self.enable_10n_selection and i > 0 and i % self.selection_interval == 0:
                self._run_10n_selection(iteration=i, objective_func=objective_func, is_periodic=True, **kwargs)
            
            # Store previous personal best costs for improvement tracking and blacklist evaluation
            previous_pbest_cost = self.swarm.pbest_cost.copy()
            self._previous_pbest_costs = previous_pbest_cost  # Store for blacklist tracking
            
            # Create evaluation batch combining PSO and scout positions
            if self.scout_config['enable_scouts'] and len(self.scout_particles) > 0:
                try:
                    # Generate LHS neighbors for all scouts before evaluation
                    self._generate_scout_neighbors()
                    
                    # Use batch evaluation system for parallel operation
                    batch_positions, attribution_map = self._create_evaluation_batch()
                    
                    # Create temporary swarm for batch evaluation
                    temp_swarm = type('TempSwarm', (), {})()
                    temp_swarm.position = batch_positions
                    
                    # Evaluate all positions in a single batch
                    batch_fitness = compute_objective_function(temp_swarm, objective_func, pool=None, **kwargs)
                    
                    # Distribute results back to PSO particles, scouts, and neighbors
                    self._distribute_evaluation_results(batch_fitness, attribution_map)
                    
                    # Update PSO personal bests
                    self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
                    
                    # Log batch evaluation statistics
                    n_pso = self.n_particles
                    n_scouts = len(self.scout_particles)
                    n_neighbors = len(batch_positions) - n_pso - n_scouts
                    self.log_message(
                        f"📦 Batch evaluation: {n_pso} PSO + {n_scouts} scouts + {n_neighbors} neighbors = {len(batch_positions)} total",
                        emoji="📦"
                    )
                except Exception as e:
                    # Fallback to standard PSO evaluation if batch evaluation fails
                    self.log_message(f"⚠️ Batch evaluation failed: {e}, falling back to standard PSO evaluation")
                    self.swarm.current_cost = compute_objective_function(self.swarm, objective_func, pool=None, **kwargs)
                    self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            else:
                # Standard PSO evaluation (no scouts active)
                self.swarm.current_cost = compute_objective_function(self.swarm, objective_func, pool=None, **kwargs)
                self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            
            # Singular run using global best position (results discarded)
            if hasattr(self.swarm, 'best_pos') and self.swarm.best_pos is not None:
                temp_swarm = type('TempSwarm', (), {})()
                temp_swarm.position = self.swarm.best_pos.reshape(1, -1)
                compute_objective_function(temp_swarm, objective_func, pool=None, **kwargs)
            
            # Hill scout spawning detection - check for negative fitness
            if self.scout_config['enable_scouts']:
                try:
                    self._detect_and_spawn_scouts(i, objective_func, **kwargs)
                except Exception as e:
                    self.log_message(f"⚠️ Error in hill scout spawning detection: {e}")
                
                try:
                    # Process hill scout groups (perform sampling/climbing)
                    self._process_scout_groups(i, objective_func, **kwargs)
                except Exception as e:
                    self.log_message(f"⚠️ Error processing hill scout groups: {e}")
                
                try:
                    # Update hill scout phases and manage lifecycle
                    self._update_scout_phases(i)
                except Exception as e:
                    self.log_message(f"⚠️ Error updating hill scout phases: {e}")
                
                try:
                    # Update hill scout lifetimes (decrement iteration counters)
                    self._update_scout_lifetimes()
                except Exception as e:
                    self.log_message(f"⚠️ Error updating hill scout lifetimes: {e}")
                
                try:
                    # Terminate expired hill scout groups
                    self._terminate_expired_groups(i)
                except Exception as e:
                    self.log_message(f"⚠️ Error terminating expired groups: {e}")
                
                try:
                    # Advance hill scout positions for next iteration
                    self._advance_scout_positions()
                except Exception as e:
                    self.log_message(f"⚠️ Error advancing hill scout positions: {e}")
            
            # Calculate how many particles improved their personal best
            pbest_improvements = self.swarm.pbest_cost < previous_pbest_cost
            n_pbest_improvements = np.sum(pbest_improvements)
            
            # Set best_cost_yet_found for ftol
            best_cost_yet_found = self.swarm.best_cost if hasattr(self.swarm, 'best_cost') else float('inf')
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)
            
            # Track improvements and stagnation
            if self.swarm.best_cost < previous_best_fitness:
                improvement = previous_best_fitness - self.swarm.best_cost
                self.stagnation_count = 0
                self.last_improvement_iter = i
                previous_best_fitness = self.swarm.best_cost
                if verbose and RICH_AVAILABLE:
                    self.log_message(f"🎉 NEW GLOBAL BEST! 🎉 Fitness: {self.swarm.best_cost:.6e} (improved by {improvement:.6e})")
                
                # Log the best particle's optimization status
                self._log_best_particle_status(i, objective_func, **kwargs)
            else:
                self.stagnation_count += 1
            
            self.best_fitness_history.append(self.swarm.best_cost)
            
            # Update sparse exploration grid with current positions
            if hasattr(self, 'grid_resolution'):
                self._update_exploration_grid(self.swarm.position)
            
            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            
            # # Verify stop criteria based on the relative acceptable cost ftol
            # relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            # delta = (
            #     np.abs(self.swarm.best_cost - best_cost_yet_found)
            #     < relative_measure
            # )
            # if i < self.ftol_iter:
            #     ftol_history.append(delta)
            # else:
            #     ftol_history.append(delta)
            #     if all(ftol_history):
            #         if verbose:
            #             self.log_message(f"🎯 Convergence achieved at iteration {i}")
            #         break
            
            # Perform options update with adaptive inertia weight
            if self.adaptive_inertia:
                # Linear decrease from initial_w to final_w
                current_w = self.initial_w - (self.initial_w - self.final_w) * (i / iters)
                self.swarm.options['w'] = current_w
            
            self.swarm.options = self.oh(
                self.options, iternow=i, itermax=iters
            )
            
            # Perform velocity and position updates
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds
            )
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh
            )
            
            # Velocity restart mechanism
            velocities = self.swarm.velocity
            avg_velocity = np.mean(np.linalg.norm(velocities, axis=1))
            
            if avg_velocity < self.velocity_threshold:
                self.low_velocity_count += 1
                if self.low_velocity_count >= self.low_velocity_threshold:
                    # Restart velocities for a fraction of particles
                    n_restart = int(self.restart_fraction * self.n_particles)
                    restart_indices = np.random.choice(self.n_particles, n_restart, replace=False)
                    
                    # Calculate velocity bounds based on parameter bounds
                    if self.bounds is not None:
                        lower_bounds, upper_bounds = self.bounds
                        v_max = 0.1 * (upper_bounds - lower_bounds)  # 10% of parameter range
                        v_min = -v_max
                    else:
                        # Fallback if no bounds specified
                        v_max = np.ones(self.dimensions) * 0.1
                        v_min = -v_max
                    
                    # Restart velocities for selected particles
                    for idx in restart_indices:
                        self.swarm.velocity[idx] = np.random.uniform(v_min, v_max)
                    
                    self.low_velocity_count = 0  # Reset counter
                    
                    if verbose and RICH_AVAILABLE:
                        self.log_message(f"🔄 Velocity restart triggered! Restarted {n_restart} particles (avg_vel: {avg_velocity:.6e})")
            else:
                self.low_velocity_count = 0  # Reset counter if velocity is healthy
            
            # Track iteration time
            iter_time = time.time() - iter_start_time
            self.generation_times.append(iter_time)
            
            # Show iteration statistics
            if verbose and RICH_AVAILABLE:
                avg_iter_time = sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0
                
                # Calculate swarm statistics
                positions = self.swarm.position
                velocities = self.swarm.velocity
                diversity = np.mean(np.std(positions, axis=0))
                avg_velocity = np.mean(np.linalg.norm(velocities, axis=1))
                current_mean = np.mean(self.swarm.current_cost)
                current_std = np.std(self.swarm.current_cost)
                
                # Calculate convergence metric - average distance to global best
                distances_to_global_best = np.linalg.norm(positions - self.swarm.best_pos, axis=1)
                avg_convergence_distance = np.mean(distances_to_global_best)
                
                # Calculate search space exploration
                exploration_percentage = self._calculate_search_space_exploration()
                
                # Calculate generational best (best fitness in current iteration)
                generational_best = np.min(self.swarm.current_cost)
                generational_best_particle = np.argmin(self.swarm.current_cost)
                
                # Find which particle has the global best (based on personal best costs)
                global_best_particle = np.argmin(self.swarm.pbest_cost)
                
                # Stagnation status with emoji
                stagnation_emoji = "🔥" if self.stagnation_count == 0 else "😴" if self.stagnation_count < 10 else "💤" if self.stagnation_count < 50 else "⚰️"
                stagnation_info = f"🔄 Stagnation: {self.stagnation_count} iters {stagnation_emoji} (last improvement: iter {self.last_improvement_iter})"
                
                # Exploration status with emoji and additional info
                exploration_emoji = "🗺️" if exploration_percentage > 80 else "🔍" if exploration_percentage > 50 else "🎯" if exploration_percentage > 20 else "📍"
                
                # Calculate additional exploration metrics using sparse grid
                if hasattr(self, 'grid_resolution'):
                    visited_cells = len(self.visited_cells)
                    total_cells = self._total_grid_cells
                    exploration_detail = f"{visited_cells:,}/{total_cells:,} cells"
                else:
                    exploration_detail = "initializing..."
                
                # Get competition statistics
                competition_stats = self.get_competition_stats()
                if competition_stats["enabled"]:
                    competition_info = f"⚔️ Competition: {len(self.visited_cells):,} visited cells"
                    if competition_stats["overcrowded_cells"] > 0:
                        competition_info += f", {competition_stats['overcrowded_cells']} overcrowded"
                    
                    # Add blacklist information
                    blacklist_count = competition_stats["blacklisted_cells"]
                    if blacklist_count > 0:
                        blacklist_percentage = competition_stats["blacklist_percentage"]
                        competition_info += f", {blacklist_count:,} blacklisted ({blacklist_percentage:.1f}%)"
                else:
                    competition_info = "⚔️ Competition: disabled"
                
                # Get scout statistics
                if self.scout_config['enable_scouts']:
                    n_hc_groups = len(self.active_scout_groups)
                    n_hc_scouts = len(self.scout_particles)
                    max_groups = self.scout_config['max_concurrent_groups']
                    
                    scout_info = f"🏔️ Scouts: {n_hc_groups}/{max_groups} groups, {n_hc_scouts} total scouts"
                else:
                    scout_info = "🏔️ Scouts: disabled"
                
                self.log_message(
                    f"""{CYAN}🌟 Iter {i}{RESET}
                    🐝 Swarm size: {self.n_particles}
                    🌍 Global best: {self.swarm.best_cost:.6e} (particle #{global_best_particle})
                    🏆 Generational best: {generational_best:.6e} (particle #{generational_best_particle})
                    📊  Mean fitness: {current_mean:.6e}
                    📈 Std fitness: {current_std:.6e}
                    🎯 Personal best improvements: {n_pbest_improvements}/{self.n_particles} particles
                    {stagnation_info}
                    🌀 Swarm diversity: {diversity:.6f}
                    🎯 Convergence radius: {avg_convergence_distance:.6f}
                    {exploration_emoji} Search space explored: {exploration_percentage:.6f}% ({exploration_detail})
                    {competition_info}
                    {scout_info}
                    ⚡ Avg velocity: {avg_velocity:.6f}
                    🔧 Inertia (w): {self.swarm.options['w']:.3f}
                    ⏱️ Iter time: {iter_time:.2f} sec / {(iter_time/60):.2f} min
                    📆 Avg iter time: {avg_iter_time:.2f} sec""",
                    panel=True, timestamp=False
                )
            
            # Update blacklist tracking and evaluate blacklist every iteration
            self._update_blacklist_tracking(i)
            self._evaluate_blacklist(i)
            
            # Run competitive evolution to prevent clustering
            self._run_competitive_evolution(i)
            
            # Create PCA visualization
            self._create_pca_visualization(i)
            
            # Save checkpoint every iteration
            if self.checkpoint_path:
                self.save_checkpoint(i)
            
            # Save detailed history data for analysis
            # self.save_history_data(i)
        
        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[
            self.swarm.pbest_cost.argmin()
        ].copy()
        
        # Final statistics
        total_time = time.time() - self.start_time
        if verbose:
            self.log_message(f"🕒 Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            self.log_message(f"🏆 Final best fitness: {final_best_cost:.6e}")
            
            # Log hill scout summary statistics if enabled
            if self.scout_config['enable_scouts']:
                self._log_scout_summary()
        
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=log_level,
        )
        
            
        return (final_best_cost, final_best_pos)

    def _log_best_particle_status(self, iteration, objective_func, **kwargs):
        """Log the best particle's optimization status to file"""
        try:
            
            play_sound("sounds/fire.wav")
            # Ensure logs directory exists
            os.makedirs("logs", exist_ok=True)
            
            # Find the best particle
            best_particle_idx = np.argmin(self.swarm.pbest_cost)
            best_particle = self.swarm.pbest_pos[best_particle_idx]
            
            # The objective function in PSO context expects a swarm, but we need to evaluate a single particle
            # We need to access the global evaluator that was set up in the PSO function
            try:
                # Import the global variables from alternatives.py
                try:
                    import src.alternatives as alt
                except ImportError:
                    # Try alternative import path
                    import alternatives as alt
                    
                if hasattr(alt, 'pso_evaluator') and hasattr(alt, 'pso_optimizable_param_names'):
                    evaluator = alt.pso_evaluator
                    optimizable_param_names = alt.pso_optimizable_param_names
                    all_param_names = alt.pso_all_param_names
                    fixed_params = alt.pso_fixed_params
                    integer_params = alt.pso_integer_params
                    toolbox = alt.pso_toolbox
                    
                    # Create full individual from the best particle position
                    individual = alt.create_full_individual(
                        best_particle, optimizable_param_names, all_param_names, 
                        fixed_params, integer_params, toolbox
                    )
                    
                    # Log header information
                    with open(self.best_log_path, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(f"\n🌟 NEW BEST PARTICLE STATUS - Iteration {iteration} - {timestamp}")
                        print(f"🎯 Best Fitness: {self.swarm.best_cost:.6e}")
                        print(f"📊 Particle Index: {best_particle_idx}")
                        print("")
                        
                        # Evaluate with verbose output (this will show the optimization status table)
                        evaluator.evaluate(individual)
                    
                    self.log_message(f"🌟 Best particle status logged to {self.best_log_path} (iteration {iteration})")
                else:
                    self.log_message("⚠️ PSO global variables not found, cannot log best particle status")
                    
            except ImportError as e:
                self.log_message(f"⚠️ Could not import alternatives module: {e}")
            except Exception as e:
                self.log_message(f"⚠️ Error accessing PSO globals: {e}")
                
        except Exception as e:
            self.log_message(f"⚠️ Failed to log best particle status: {e}")

    def _adaptive_boost_strategy(self, iteration):
        """Adaptive boost strategy based on exploration saturation"""
        
        # Calculate exploration saturation
        exploration_saturation = self._calculate_exploration_saturation()
        
        # Get refinement threshold (default to 95% if not set)
        refinement_threshold = getattr(self, 'grid_refinement_threshold', 95.0)
        
        # Determine strategy based on saturation level
        if exploration_saturation >= refinement_threshold:
            # High saturation - refine grid and continue exploration
            self.log_message(f"🎯 Exploration saturation: {exploration_saturation:.1f}% (≥{refinement_threshold}%), refining grid resolution")
            self._refine_exploration_grid()
            self._boost_particles_with_exploration_prediction(iteration)
        else:
            # Normal saturation - use exploration prediction
            self.log_message(f"🎯 Exploration saturation: {exploration_saturation:.1f}%, using exploration prediction")
            self._boost_particles_with_exploration_prediction(iteration)

    def _calculate_exploration_saturation(self):
        """Calculate the percentage of search space explored using sparse grid"""
        
        if not hasattr(self, 'grid_resolution'):
            return 0.0
        
        visited_count = len(self.visited_cells)
        total_cells = self._total_grid_cells
        saturation = (visited_count / total_cells) * 100.0
        
        return saturation

    def _refine_exploration_grid(self):
        """Double the grid resolution to create new unexplored regions for finer exploration using sparse storage"""
        
        if not hasattr(self, 'grid_resolution'):
            self.log_message("⚠️ No exploration grid to refine")
            return
        
        old_resolution = self.grid_resolution
        new_resolution = old_resolution * 2
        
        # Prevent excessive memory usage - limit maximum resolution
        max_resolution = getattr(self, 'max_grid_resolution', 50)
        if new_resolution > max_resolution:
            self.log_message(f"⚠️ Grid resolution limit reached ({max_resolution}), cannot refine further")
            return
        
        try:
            self.log_message(f"🔍 Refining sparse exploration grid: {old_resolution}^{self.dimensions} → {new_resolution}^{self.dimensions}")
            
            # Create new visited cells set for higher resolution
            new_visited_cells = set()
            
            # Map old sparse coordinates to new higher-resolution coordinates
            for old_coords in self.visited_cells:
                # Each old cell becomes 2^dimensions new cells
                new_coords_base = tuple(coord * 2 for coord in old_coords)
                
                # Mark the corresponding 2^d cells in new grid as visited
                for offset in np.ndindex(tuple([2] * self.dimensions)):
                    new_coords = tuple(base + off for base, off in zip(new_coords_base, offset))
                    if all(coord < new_resolution for coord in new_coords):
                        new_visited_cells.add(new_coords)
            
            # Update grid attributes
            self.visited_cells = new_visited_cells
            self.grid_resolution = new_resolution
            self._total_grid_cells = new_resolution ** self.dimensions
            
            # Calculate new exploration statistics
            visited_count = len(self.visited_cells)
            new_saturation = (visited_count / self._total_grid_cells) * 100.0
            unexplored_cells = self._total_grid_cells - visited_count
            
            self.log_message(f"📊 New sparse grid: {self._total_grid_cells:,} total cells, {visited_count:,} visited ({new_saturation:.1f}%)")
            self.log_message(f"🗺️ Created {unexplored_cells:,} new unexplored cells for finer exploration")
            
        except Exception as e:
            self.log_message(f"⚠️ Error refining sparse exploration grid: {e}")

    def _boost_particles_with_exploration_prediction(self, iteration):
        """Boost particles using distance-maximization placement"""
        
        # Step 1: Select particles to boost using existing logic
        boost_indices = self._select_particles_for_boost()
        
        if len(boost_indices) == 0:
            self.log_message("⚠️ No particles selected for distance-maximization boost")
            return
        
        # Calculate statistics for particles to be boosted (before boosting)
        particles_to_boost_pbest_fitness = self.swarm.pbest_cost[boost_indices]
        before_boost_stats = {
            'min': np.min(particles_to_boost_pbest_fitness),
            'max': np.max(particles_to_boost_pbest_fitness),
            'mean': np.mean(particles_to_boost_pbest_fitness)
        }
        
        # Count selection categories
        optimize_limits_len = getattr(self, 'optimize_limits_len', 5)
        fitness_threshold = 10 ** optimize_limits_len
        high_fitness_count = np.sum(particles_to_boost_pbest_fitness >= fitness_threshold)
        low_fitness_count = len(boost_indices) - high_fitness_count
        
        # Step 2: Find optimal positions using candidate-based farthest-point sampling
        optimal_positions = self._find_farthest_point_positions(len(boost_indices))
        
        # Step 3: Apply the optimal positions to the selected particles
        if optimal_positions is not None and len(optimal_positions) > 0:
            for i, particle_idx in enumerate(boost_indices):
                self.swarm.position[particle_idx] = optimal_positions[i]
                
                # Initialize random velocity
                self.swarm.velocity[particle_idx] = self._initialize_random_velocity()
                
                # Reset pbest to give fresh start
                self.swarm.pbest_pos[particle_idx] = optimal_positions[i]
                self.swarm.pbest_cost[particle_idx] = np.inf
            
            # Store the boosted particle indices for tracking in next iteration
            self.last_boosted_indices = boost_indices.copy()
            self.last_boost_iteration = iteration
            
            # Calculate boost statistics
            n_boost = len(boost_indices)
            boost_percentage = (n_boost / self.n_particles) * 100
            
            # Log detailed boost information
            global_best_particle_idx = self._get_global_best_particle_idx()
            self.log_message(f"🚀 Farthest-point boost applied to {n_boost}/{self.n_particles} particles ({boost_percentage:.1f}%) (global best particle #{global_best_particle_idx} protected)")
            self.log_message(f"📊 Selection: High pbest (≥{fitness_threshold:.0e}): {high_fitness_count}, Low pbest: {low_fitness_count}")
            self.log_message(f"📈 Boosted particles pbest - Min: {before_boost_stats['min']:.6e}, Avg: {before_boost_stats['mean']:.6e}, Max: {before_boost_stats['max']:.6e}")
            self.log_message("🔄 Reset pbest for all boosted particles - fresh start opportunity")
            self.log_message("🎯 Placed particles at farthest candidates from discovered regions")
            
            # Show "after boost" statistics if we have data from previous boost
            if hasattr(self, 'last_boosted_indices') and hasattr(self, 'last_boost_iteration') and self.last_boost_iteration == iteration - self.reinit_interval:
                # Get current fitness of previously boosted particles
                prev_boosted_current_fitness = self.swarm.current_cost[self.last_boosted_indices]
                after_boost_stats = {
                    'min': np.min(prev_boosted_current_fitness),
                    'max': np.max(prev_boosted_current_fitness),
                    'mean': np.mean(prev_boosted_current_fitness)
                }
                self.log_message(f"� Preveious boosted particles fitness - Min: {after_boost_stats['min']:.6e}, Avg: {after_boost_stats['mean']:.6e}, Max: {after_boost_stats['max']:.6e}")
            
        else:
            # Fallback to traditional boost
            self.log_message("⚠️ Distance maximization failed, using traditional boost")
            self._boost_particles_traditional(iteration)

    def _select_particles_for_boost(self):
        """Select particles for boosting using existing sophisticated criteria, protecting global best"""
        
        # Get fitness threshold
        optimize_limits_len = getattr(self, 'optimize_limits_len', 5)
        fitness_threshold = 10 ** optimize_limits_len
        global_best_particle_idx = self._get_global_best_particle_idx()
        
        # Get all particle indices and their pbest fitness values
        all_indices = np.arange(self.n_particles)
        all_pbest_fitness = self.swarm.pbest_cost
        
        # Remove global best particle from consideration
        eligible_indices = all_indices[all_indices != global_best_particle_idx]
        eligible_pbest_fitness = all_pbest_fitness[eligible_indices]
        
        # Separate eligible particles into high fitness (above threshold) and low fitness (below threshold)
        high_fitness_mask = eligible_pbest_fitness >= fitness_threshold
        low_fitness_mask = eligible_pbest_fitness < fitness_threshold
        
        high_fitness_indices = eligible_indices[high_fitness_mask]
        low_fitness_indices = eligible_indices[low_fitness_mask]
        
        # Calculate min and max boost counts (adjusted for protecting global best)
        max_eligible = len(eligible_indices)
        min_boost = max(1, int(0.20 * max_eligible))  # Minimum 20% of eligible particles
        max_boost = max(1, int(0.30 * max_eligible))  # Maximum 30% of eligible particles
        
        selected_indices = []
        
        # Step 1: Add all high pbest fitness particles (above threshold), prioritizing worst first
        if len(high_fitness_indices) > 0:
            # Sort high fitness particles by pbest fitness (worst first)
            high_fitness_sorted = high_fitness_indices[np.argsort(eligible_pbest_fitness[high_fitness_mask])[::-1]]
            # Take up to max_boost particles
            selected_indices.extend(high_fitness_sorted[:max_boost])
        
        # Step 2: If we haven't reached min_boost (20%), add worst particles from low fitness group
        if len(selected_indices) < min_boost and len(low_fitness_indices) > 0:
            # Sort low fitness particles by pbest fitness (worst first)
            low_fitness_sorted = low_fitness_indices[np.argsort(eligible_pbest_fitness[low_fitness_mask])[::-1]]
            # Add particles until we reach min_boost
            needed = min_boost - len(selected_indices)
            selected_indices.extend(low_fitness_sorted[:needed])
        
        # Convert to numpy array and ensure we don't exceed max_boost
        boost_indices = np.array(selected_indices[:max_boost])
        
        return boost_indices



    def _position_to_grid_coords(self, position):
        """Convert position to grid coordinates"""
        
        if not hasattr(self, 'grid_resolution') or self.bounds is None:
            return None
        
        lower_bounds, upper_bounds = self.bounds
        grid_coords = []
        
        for dim in range(self.dimensions):
            # Normalize position to [0, 1] range
            normalized_pos = (position[dim] - lower_bounds[dim]) / (upper_bounds[dim] - lower_bounds[dim])
            
            # Convert to grid index (clamp to valid range)
            grid_idx = int(normalized_pos * self.grid_resolution)
            grid_idx = max(0, min(grid_idx, self.grid_resolution - 1))
            grid_coords.append(grid_idx)
        
        return tuple(grid_coords)

    def _initialize_exploration_grid(self):
        """Initialize the sparse exploration grid for tracking visited regions"""
        
        if self.bounds is None:
            self.log_message("⚠️ Cannot initialize exploration grid without bounds")
            return
        
        # Set grid resolution based on problem dimensions (if not already set)
        if not hasattr(self, 'grid_resolution'):
            if self.dimensions <= 3:
                self.grid_resolution = 20  # 20^3 = 8,000 cells max
            elif self.dimensions <= 6:
                self.grid_resolution = 10  # 10^6 = 1,000,000 cells max
            elif self.dimensions <= 10:
                self.grid_resolution = 5   # 5^10 = 9,765,625 cells max
            else:
                self.grid_resolution = 3   # 3^n cells (manageable for high dimensions)
        
        # Calculate total cells (but don't create dense grid)
        self._total_grid_cells = self.grid_resolution ** self.dimensions
        
        # Initialize discovered coords buffer for farthest-point sampling
        self._discovered_coords = deque(maxlen=self.fp_discovered_cap)
        
        # visited_cells set is already initialized in __init__
        self.log_message(f"🗺️ Initialized sparse exploration grid: {self.grid_resolution}^{self.dimensions} = {self._total_grid_cells:,} cells")

    def _update_exploration_grid(self, positions):
        """Update sparse exploration grid with visited positions"""
        
        new_cells = 0
        for pos in positions:
            grid_coords = self._position_to_grid_coords(pos)
            if grid_coords is not None:
                if grid_coords not in self.visited_cells:
                    self.visited_cells.add(grid_coords)
                    new_cells += 1
                    # Track newly discovered coord for farthest-point sampling
                    if self._discovered_coords is not None:
                        self._discovered_coords.append(grid_coords)
        
        return new_cells

    def _find_farthest_point_positions(self, n_particles):
        """Find positions via candidate-based farthest-point sampling.
        
        This avoids enumerating all unexplored grid cells. It uses a candidate pool
        sampled in continuous space and maximizes the minimum distance to discovered
        regions (derived from the exploration grid and swarm/pbest positions).
        """
        try:
            # Gather discovered anchor positions
            anchors = []
            # 1) From discovered grid coords tracked incrementally
            if isinstance(self._discovered_coords, deque) and len(self._discovered_coords) > 0:
                # Convert a sample (all currently stored, already capped) to positions
                anchors.extend([self._grid_coords_to_position(coords) for coords in list(self._discovered_coords)])
            
            # 2) Add current swarm and pbest positions to anchors (sample/cap later)
            anchors.extend(list(self.swarm.position))
            anchors.extend(list(self.swarm.pbest_pos))
            
            if len(anchors) == 0:
                # No anchors yet, use initialization strategy
                self.log_message("🎯 No discovered anchors, using LHS initialization")
                return self._generate_latin_hypercube_positions(n_particles, self.bounds)
            
            # Convert anchors to numpy and cap to discovered_cap
            anchors_np = np.asarray(anchors, dtype=self.fp_dtype)
            if anchors_np.shape[0] > self.fp_discovered_cap:
                idx = np.random.choice(anchors_np.shape[0], self.fp_discovered_cap, replace=False)
                anchors_np = anchors_np[idx]
            
            # Generate candidate pool
            C = int(max(n_particles, self.fp_candidate_pool_size))
            candidates = self._generate_candidate_positions(C).astype(self.fp_dtype, copy=False)
            
            # Compute initial nearest distances from candidates to anchors
            # Use chunking to limit memory if needed
            def chunked_min_distance(cands, anchors, chunk=2000):
                nearest = np.full(cands.shape[0], np.inf, dtype=self.fp_dtype)
                for start in range(0, anchors.shape[0], chunk):
                    end = min(start + chunk, anchors.shape[0])
                    diff = cands[:, None, :] - anchors[None, start:end, :]
                    dist2 = np.sum(diff * diff, axis=2)
                    local_min = np.min(dist2, axis=1)
                    nearest = np.minimum(nearest, local_min)
                return np.sqrt(nearest).astype(self.fp_dtype, copy=False)
            
            nearest_dist = chunked_min_distance(candidates, anchors_np)
            
            # Greedy farthest-point selection
            selected = []
            for _ in range(n_particles):
                best_idx = int(np.argmax(nearest_dist))
                best_pos = candidates[best_idx]
                selected.append(best_pos)
                
                # Update nearest distances using the newly selected point
                diff = candidates - best_pos
                dist = np.sqrt(np.sum(diff * diff, axis=1)).astype(self.fp_dtype, copy=False)
                nearest_dist = np.minimum(nearest_dist, dist)
                
                # Optional: mask out selected index to avoid reselection
                nearest_dist[best_idx] = -np.inf
            
            self.log_message(f"📌 Farthest-point selected {len(selected)} positions from {C} candidates and {anchors_np.shape[0]} anchors")
            return np.asarray(selected, dtype=np.float64)
        except Exception as e:
            self.log_message(f"⚠️ Farthest-point sampling failed: {e}")
            return self._generate_latin_hypercube_positions(n_particles, self.bounds)

    def _generate_candidate_positions(self, n_candidates):
        """Generate candidate positions in continuous space within bounds/center, avoiding blacklisted cells."""
        try:
            # Generate more candidates than needed to account for blacklisted cell filtering
            n_raw_candidates = n_candidates * 3  # Generate 3x more to account for blacklisted rejections
            
            if SCIPY_AVAILABLE and self.fp_use_sobol:
                sampler = qmc.Sobol(d=self.dimensions, scramble=True, seed=np.random.randint(0, 2**31))
                n_sobol = 2 ** int(np.ceil(np.log2(n_raw_candidates)))
                sample = sampler.random(n=n_sobol)[:n_raw_candidates]
                if self.bounds is not None:
                    lower, upper = self.bounds
                    raw_candidates = qmc.scale(sample, lower, upper)
                else:
                    raw_candidates = qmc.scale(sample, -self.center, self.center)
            elif SCIPY_AVAILABLE:
                sampler = qmc.LatinHypercube(d=self.dimensions, seed=np.random.randint(0, 2**31))
                sample = sampler.random(n=n_raw_candidates)
                if self.bounds is not None:
                    lower, upper = self.bounds
                    raw_candidates = qmc.scale(sample, lower, upper)
                else:
                    raw_candidates = qmc.scale(sample, -self.center, self.center)
            else:
                # Uniform fallback
                if self.bounds is not None:
                    lower, upper = self.bounds
                    raw_candidates = np.random.uniform(lower, upper, size=(n_raw_candidates, self.dimensions))
                else:
                    raw_candidates = np.random.uniform(-self.center, self.center, size=(n_raw_candidates, self.dimensions))
            
            # Filter out blacklisted cells
            valid_candidates = []
            blacklisted_rejections = 0
            
            for pos in raw_candidates:
                grid_coords = self._position_to_grid_coords(pos)
                if grid_coords is not None and grid_coords in self.blacklisted_cells:
                    blacklisted_rejections += 1
                    continue  # Skip blacklisted cells
                valid_candidates.append(pos)
                if len(valid_candidates) >= n_candidates:
                    break
            
            # If we don't have enough valid candidates, generate more with uniform sampling
            while len(valid_candidates) < n_candidates:
                if self.bounds is not None:
                    lower, upper = self.bounds
                    extra_pos = np.random.uniform(lower, upper, size=(1, self.dimensions))[0]
                else:
                    extra_pos = np.random.uniform(-self.center, self.center, size=(1, self.dimensions))[0]
                
                grid_coords = self._position_to_grid_coords(extra_pos)
                if grid_coords is None or grid_coords not in self.blacklisted_cells:
                    valid_candidates.append(extra_pos)
            
            # Log blacklist impact
            if blacklisted_rejections > 0:
                self.log_message(f"🚫 Rejected {blacklisted_rejections} blacklisted candidates during farthest-point sampling")
            
            return np.asarray(valid_candidates[:n_candidates])
            
        except Exception as e:
            self.log_message(f"⚠️ Candidate generation failed: {e}, using uniform fallback")
            if self.bounds is not None:
                lower, upper = self.bounds
                return np.random.uniform(lower, upper, size=(n_candidates, self.dimensions))
            else:
                return np.random.uniform(-self.center, self.center, size=(n_candidates, self.dimensions))

    def _initialize_random_velocity(self):
        """Initialize random velocity for a boosted particle"""
        
        if self.velocity_clamp is not None:
            min_vel, max_vel = self.velocity_clamp
            return np.random.uniform(min_vel, max_vel, self.dimensions)
        else:
            if self.bounds is not None:
                lower_bounds, upper_bounds = self.bounds
                vel_range = 0.1 * (upper_bounds - lower_bounds)  # 10% of parameter range
                return np.random.uniform(-vel_range, vel_range, self.dimensions)
            else:
                return np.random.uniform(-0.1, 0.1, self.dimensions)

    def _grid_coords_to_position(self, grid_coords):
        # Convert integer grid coordinates to continuous position within bounds/center
        if self.bounds is None:
            position = np.zeros(self.dimensions, dtype=float)
            for dim in range(self.dimensions):
                normalized = grid_coords[dim] / (self.grid_resolution - 1)
                position[dim] = (normalized - 0.5) * 2.0 * self.center
            return position

        lower, upper = self.bounds
        position = np.zeros(self.dimensions, dtype=float)
        for dim in range(self.dimensions):
            normalized = grid_coords[dim] / (self.grid_resolution - 1)
            position[dim] = lower[dim] + normalized * (upper[dim] - lower[dim])
        return position

    def _run_competitive_evolution(self, iteration):
        """Run grid-based competitive evolution to prevent clustering and improve exploration"""
        
        if not self.competition_enabled:
            return
        
        # Only run competition check at specified intervals
        if iteration % self.competition_check_interval != 0 or iteration == 0:
            return
        
        if not hasattr(self, 'grid_resolution') or self.bounds is None:
            self.log_message("⚠️ Competition requires exploration grid and bounds, skipping")
            return
        
        # Step 1: Update cell occupancy mapping
        self._update_cell_occupancy()
        
        # Step 2: Update cell fitness history for stagnation detection
        # (This happens automatically in _is_cell_stagnant method)
        
        # Step 3: Check for overcrowding and stagnation
        particles_to_relocate = []
        
        # Handle overcrowding
        overcrowded_particles = self._resolve_overcrowding()
        particles_to_relocate.extend(overcrowded_particles)
        
        # Handle stagnation (only if we have enough history)
        if len(self.global_improvement_history) >= self.stagnation_window:
            stagnant_particles = self._resolve_stagnation(iteration)
            particles_to_relocate.extend(stagnant_particles)
        
        # Step 4: Relocate particles to unvisited cells
        if particles_to_relocate:
            self._relocate_particles(particles_to_relocate, iteration)
        
        # Log competition results
        if particles_to_relocate:
            global_best_particle_idx = self._get_global_best_particle_idx()
            self.log_message(f"⚔️ Competitive evolution: relocated {len(particles_to_relocate)} particles (global best particle #{global_best_particle_idx} protected)")

    def _update_cell_occupancy(self):
        """Update mapping of grid cells to particle indices"""
        
        self.cell_occupancy = {}
        
        for particle_idx in range(self.n_particles):
            position = self.swarm.position[particle_idx]
            grid_coords = self._position_to_grid_coords(position)
            
            if grid_coords is not None:
                if grid_coords not in self.cell_occupancy:
                    self.cell_occupancy[grid_coords] = []
                self.cell_occupancy[grid_coords].append(particle_idx)
                
                # Track visited cells
                self.visited_cells.add(grid_coords)

    def _update_global_improvement_history(self, iteration):
        """Track global swarm improvement rate over time"""
        
        current_global_best = self.swarm.best_cost
        
        # Calculate improvement from previous iteration
        if len(self.global_improvement_history) > 0:
            previous_best = self.global_improvement_history[-1]['best_cost']
            improvement = max(0, previous_best - current_global_best)  # Positive improvement
        else:
            improvement = 0
        
        self.global_improvement_history.append({
            'iteration': iteration,
            'best_cost': current_global_best,
            'improvement': improvement
        })
        
        # Keep only the stagnation window worth of history
        if len(self.global_improvement_history) > self.stagnation_window:
            self.global_improvement_history.pop(0)

    def _resolve_overcrowding(self):
        """Identify and mark particles for relocation from overcrowded cells"""
        
        particles_to_relocate = []
        overcrowded_cells = 0
        global_best_particle_idx = self._get_global_best_particle_idx()
        
        for grid_coords, particle_indices in self.cell_occupancy.items():
            if len(particle_indices) > self.max_particles_per_cell:
                overcrowded_cells += 1
                
                # Sort particles by personal best fitness (worst first)
                particle_fitness = [(idx, self.swarm.pbest_cost[idx]) for idx in particle_indices]
                particle_fitness.sort(key=lambda x: x[1], reverse=True)  # Worst first
                
                # Mark excess particles for relocation, but protect global best particle
                excess_count = len(particle_indices) - self.max_particles_per_cell
                relocated_count = 0
                
                for i in range(len(particle_fitness)):
                    if relocated_count >= excess_count:
                        break
                    
                    particle_idx = particle_fitness[i][0]
                    
                    # Skip global best particle - it stays in its cell
                    if particle_idx == global_best_particle_idx:
                        continue
                    
                    particles_to_relocate.append({
                        'particle_idx': particle_idx,
                        'reason': 'overcrowding',
                        'cell': grid_coords,
                        'fitness': particle_fitness[i][1]
                    })
                    relocated_count += 1
        
        if overcrowded_cells > 0:
            self.log_message(f"📊 Overcrowding: {overcrowded_cells} cells exceed {self.max_particles_per_cell} particles")
        
        return particles_to_relocate

    def _resolve_stagnation(self, iteration):
        """Identify and mark particles for relocation from stagnant cells based on cell-best unchanged"""
        
        particles_to_relocate = []
        global_best_particle_idx = self._get_global_best_particle_idx()
        stagnant_cells = 0
        
        for grid_coords, particle_indices in self.cell_occupancy.items():
            # Check if cell is stagnant (cell best unchanged for stagnation_window iterations)
            if self._is_cell_stagnant(grid_coords, iteration):
                stagnant_cells += 1
                
                # Sort particles by personal best fitness (worst first)
                particle_fitness = [(idx, self.swarm.pbest_cost[idx]) for idx in particle_indices]
                particle_fitness.sort(key=lambda x: x[1], reverse=True)  # Worst first
                
                # Mark worst percentage for relocation, but protect global best particle
                evict_count = max(1, int(len(particle_indices) * self.eviction_percentage / 100))
                relocated_count = 0
                
                for i in range(len(particle_fitness)):
                    if relocated_count >= evict_count:
                        break
                    
                    particle_idx = particle_fitness[i][0]
                    
                    # Skip global best particle - it stays in its cell
                    if particle_idx == global_best_particle_idx:
                        continue
                    
                    # Get stagnation info for logging
                    stagnation_iterations = self._get_cell_stagnation_duration(grid_coords)
                    
                    particles_to_relocate.append({
                        'particle_idx': particle_idx,
                        'reason': 'stagnation',
                        'cell': grid_coords,
                        'fitness': particle_fitness[i][1],
                        'stagnation_iterations': stagnation_iterations
                    })
                    relocated_count += 1
        
        if stagnant_cells > 0:
            self.log_message(f"🐌 Stagnation: {stagnant_cells} cells with unchanged cell-best for {self.stagnation_window}+ iterations")
        
        return particles_to_relocate

    def _is_cell_stagnant(self, grid_coords, iteration):
        """Check if a cell is stagnant (cell-best unchanged for stagnation_window iterations)"""
        
        # Initialize cell fitness history if not exists
        if grid_coords not in self.cell_fitness_history:
            self.cell_fitness_history[grid_coords] = deque(maxlen=self.stagnation_window)
        
        # Get current best fitness in this cell
        if grid_coords in self.cell_occupancy:
            particle_indices = self.cell_occupancy[grid_coords]
            cell_best_fitness = min(self.swarm.pbest_cost[idx] for idx in particle_indices)
        else:
            cell_best_fitness = np.inf
        
        # Add to history
        self.cell_fitness_history[grid_coords].append(cell_best_fitness)
        
        # Check for stagnation: cell-best unchanged for full window
        fitness_history = list(self.cell_fitness_history[grid_coords])
        
        # Need full window to detect stagnation
        if len(fitness_history) < self.stagnation_window:
            return False
        
        # Check if the best fitness has been unchanged (within small tolerance for numerical precision)
        first_best = fitness_history[0]
        tolerance = abs(first_best) * 1e-12 + 1e-15  # Relative + absolute tolerance
        
        # Cell is stagnant if all values in the window are essentially the same
        for fitness in fitness_history[1:]:
            if abs(fitness - first_best) > tolerance:
                return False  # Found improvement, not stagnant
        
        return True  # No improvement found in the entire window

    def _get_cell_stagnation_duration(self, grid_coords):
        """Get how many iterations the cell has been stagnant"""
        
        if grid_coords not in self.cell_fitness_history:
            return 0
        
        fitness_history = list(self.cell_fitness_history[grid_coords])
        if len(fitness_history) < 2:
            return 0
        
        # Count consecutive iterations with no improvement from the end
        current_best = fitness_history[-1]
        tolerance = abs(current_best) * 1e-12 + 1e-15
        stagnant_count = 1  # Current iteration
        
        # Count backwards while fitness remains unchanged
        for i in range(len(fitness_history) - 2, -1, -1):
            if abs(fitness_history[i] - current_best) <= tolerance:
                stagnant_count += 1
            else:
                break
        
        return stagnant_count

    def _relocate_particles(self, particles_to_relocate, iteration):
        """Relocate particles to unvisited cells using LHS sampling"""
        
        if not particles_to_relocate:
            return
        
        # Get unvisited cells for relocation
        unvisited_positions = self._sample_unvisited_positions(len(particles_to_relocate))
        
        if len(unvisited_positions) == 0:
            self.log_message("⚠️ No unvisited positions available for relocation")
            return
        
        # Relocate particles
        relocated_count = 0
        overcrowding_count = 0
        stagnation_count = 0
        
        for i, particle_info in enumerate(particles_to_relocate):
            if i >= len(unvisited_positions):
                break  # Not enough unvisited positions
            
            particle_idx = particle_info['particle_idx']
            old_fitness = particle_info['fitness']
            reason = particle_info['reason']
            
            # Set new position
            new_position = unvisited_positions[i]
            self.swarm.position[particle_idx] = new_position
            
            # Reset velocity
            self.swarm.velocity[particle_idx] = self._initialize_random_velocity()
            
            # Reset personal best to give fresh start
            self.swarm.pbest_pos[particle_idx] = new_position.copy()
            self.swarm.pbest_cost[particle_idx] = np.inf
            
            relocated_count += 1
            if reason == 'overcrowding':
                overcrowding_count += 1
            elif reason == 'stagnation':
                stagnation_count += 1
        
        # Log relocation details
        self.log_message(f"🚀 Relocated {relocated_count} particles: {overcrowding_count} overcrowding, {stagnation_count} stagnation")
        self.log_message("🎯 Particles placed in unvisited regions with fresh velocities and reset pbest")
        
        # Calculate and log average cell density on populated cells
        if hasattr(self, 'cell_occupancy') and self.cell_occupancy:
            populated_cells = [len(particles) for particles in self.cell_occupancy.values() if len(particles) > 0]
            if populated_cells:
                avg_density = np.mean(populated_cells)
                max_density = max(populated_cells)
                min_density = min(populated_cells)
                total_populated_cells = len(populated_cells)
                self.log_message(f"📊 Cell density: avg {avg_density:.2f}, min {min_density}, max {max_density} particles/cell ({total_populated_cells} populated cells)")

    def _sample_unvisited_positions(self, n_positions):
        """Sample positions in unvisited and non-blacklisted grid cells using LHS"""
        
        if not hasattr(self, 'grid_resolution') or self.bounds is None:
            return []
        
        try:
            # Generate candidate positions using LHS
            n_candidates = max(n_positions * 20, 2000)  # Generate more candidates to account for blacklisted cells
            
            if SCIPY_AVAILABLE:
                sampler = qmc.LatinHypercube(d=self.dimensions, seed=np.random.randint(0, 2**31))
                sample = sampler.random(n=n_candidates)
                lower_bounds, upper_bounds = self.bounds
                candidate_positions = qmc.scale(sample, lower_bounds, upper_bounds)
            else:
                # Fallback to uniform sampling
                lower_bounds, upper_bounds = self.bounds
                candidate_positions = np.random.uniform(lower_bounds, upper_bounds, (n_candidates, self.dimensions))
            
            # Filter to only unvisited and non-blacklisted cells
            valid_positions = []
            blacklisted_rejections = 0
            
            for pos in candidate_positions:
                grid_coords = self._position_to_grid_coords(pos)
                if grid_coords is not None:
                    # Check if cell is unvisited
                    if grid_coords not in self.visited_cells:
                        # Check if cell is blacklisted
                        if grid_coords in self.blacklisted_cells:
                            blacklisted_rejections += 1
                            continue  # Skip blacklisted cells
                        valid_positions.append(pos)
                        if len(valid_positions) >= n_positions:
                            break
            
            # If not enough valid positions found with LHS, try uniform sampling of all valid cells
            if len(valid_positions) < n_positions:
                additional_needed = n_positions - len(valid_positions)
                additional_positions = self._uniform_sample_valid_cells(additional_needed)
                valid_positions.extend(additional_positions)
            
            # Log blacklist impact
            if blacklisted_rejections > 0:
                self.log_message(f"🚫 Rejected {blacklisted_rejections} blacklisted candidates during sampling")
            
            return valid_positions[:n_positions]
            
        except Exception as e:
            self.log_message(f"⚠️ Error sampling unvisited positions: {e}")
            return []

    def _uniform_sample_unvisited_cells(self, n_positions):
        """Uniform sampling fallback for unvisited cells (legacy method)"""
        return self._uniform_sample_valid_cells(n_positions)

    def _uniform_sample_valid_cells(self, n_positions):
        """Uniform sampling fallback for unvisited and non-blacklisted cells"""
        
        if not hasattr(self, 'grid_resolution') or self.bounds is None:
            return []
        
        try:
            # Generate all possible grid coordinates
            grid_ranges = [range(self.grid_resolution) for _ in range(self.dimensions)]
            all_coords = list(itertools.product(*grid_ranges))
            
            # Filter to unvisited and non-blacklisted coordinates
            valid_coords = []
            for coords in all_coords:
                if coords not in self.visited_cells and coords not in self.blacklisted_cells:
                    valid_coords.append(coords)
            
            if len(valid_coords) == 0:
                self.log_message("⚠️ No valid (unvisited + non-blacklisted) cells available")
                return []
            
            # Sample random valid coordinates
            n_sample = min(n_positions, len(valid_coords))
            sampled_indices = np.random.choice(len(valid_coords), n_sample, replace=False)
            
            # Convert to positions
            positions = []
            for idx in sampled_indices:
                grid_coords = valid_coords[idx]
                position = self._grid_coords_to_position(grid_coords)
                positions.append(position)
            
            return positions
            
        except Exception as e:
            self.log_message(f"⚠️ Error in uniform sampling fallback: {e}")
            return []

    def _get_global_best_particle_idx(self):
        """Get the index of the particle with the global best personal best fitness
        
        Returns
        -------
        int
            Index of the particle with the best personal best fitness
        """
        return np.argmin(self.swarm.pbest_cost)

    def _standard_initialization(self, objective_func, **kwargs):
        """Standard PSO initialization without 10n selection
        
        Parameters
        ----------
        objective_func : callable
            Objective function for evaluation
        **kwargs : dict
            Additional arguments for objective function
        """
        # Use advanced initialization strategy for better space coverage
        self.swarm.position = self._initialize_positions_with_strategy(self.n_particles, self.bounds)
        
        # Initialize velocities based on position bounds
        if self.velocity_clamp is not None:
            min_vel, max_vel = self.velocity_clamp
            self.swarm.velocity = np.random.uniform(min_vel, max_vel, (self.n_particles, self.dimensions))
        else:
            # Default velocity initialization based on position range
            if self.bounds is not None:
                lower_bounds, upper_bounds = self.bounds
                vel_range = 0.1 * (upper_bounds - lower_bounds)  # 10% of parameter range
                self.swarm.velocity = np.random.uniform(-vel_range, vel_range, (self.n_particles, self.dimensions))
            else:
                # Fallback velocity initialization
                self.swarm.velocity = np.random.uniform(-0.1, 0.1, (self.n_particles, self.dimensions))
        
        # Handle init_pos injection for fresh start
        if self.custom_init_pos is not None:
            self._inject_init_pos_fresh_start()
        
        # Initialize pbest arrays before evaluation
        self.swarm.pbest_pos = self.swarm.position.copy()
        self.swarm.pbest_cost = np.full(self.n_particles, np.inf)
        
        # Evaluate all particles to get proper fitness values (not np.inf)
        self.swarm.current_cost = compute_objective_function(self.swarm, objective_func, pool=None, **kwargs)
        self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
        self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)

    def _run_10n_selection(self, iteration, objective_func, is_fresh_start=False, is_checkpoint_resume=False, is_periodic=False, **kwargs):
        """Run adaptive quality selection: sample batches until we have n particles below threshold
        
        This method keeps sampling batches of selection_multiplier*n candidates until it finds
        n particles with fitness below negative_fitness_threshold. Each batch uses fresh LHS sampling.
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        objective_func : callable
            Objective function for evaluation
        is_fresh_start : bool
            True if this is a fresh start (iteration 0)
        is_checkpoint_resume : bool
            True if this is a checkpoint resume
        is_periodic : bool
            True if this is a periodic selection (every N iterations)
        **kwargs : dict
            Additional arguments for objective function
        """
        try:
            # Determine selection type for logging
            if is_fresh_start:
                selection_type = "Fresh Start"
                log_emoji = "🚀"
            elif is_checkpoint_resume:
                selection_type = "Checkpoint Resume"
                log_emoji = "🔄"
            elif is_periodic:
                selection_type = f"Periodic (iter {iteration})"
                log_emoji = "🔁"
            else:
                selection_type = "Unknown"
                log_emoji = "❓"
            
            # Get fitness threshold from scout config
            fitness_threshold = self.scout_config['negative_fitness_threshold']
            batch_size = self.n_particles * self.selection_multiplier
            max_batches = self.selection_max_batches
            
            # Calculate target number of quality particles
            if getattr(self, 'selection_dynamic_target', False) and not is_fresh_start:
                # Dynamic target mode: count how many current particles are above threshold
                n_above_threshold = np.sum(self.swarm.pbest_cost >= fitness_threshold)
                target_quality_particles = n_above_threshold
                target_quality_particles = max(1, target_quality_particles)  # At least 1
                
                dynamic_target_info = f"dynamic target: {n_above_threshold} particles currently ≥ {fitness_threshold:.0e}"
            else:
                # Static target mode: use percentage
                target_quality_particles = int(self.n_particles * self.selection_quality_threshold_pct / 100.0)
                target_quality_particles = max(1, target_quality_particles)  # At least 1
                
                dynamic_target_info = f"static target: {self.selection_quality_threshold_pct:.0f}%"
            
            self.log_message(
                f"{log_emoji} {selection_type}: Adaptive quality selection\n"
                f"   Target: {target_quality_particles}/{self.n_particles} particles with fitness < {fitness_threshold:.0e} ({dynamic_target_info})\n"
                f"   Batch size: {batch_size} candidates ({self.selection_multiplier}n)\n"
                f"   Max batches: {max_batches}",
                emoji=log_emoji
            )
            
            # Generate single large LHS sample upfront for better space-filling
            total_lhs_size = batch_size * max_batches
            self.log_message(
                f"   Generating single LHS sample: {total_lhs_size:,} candidates "
                f"({batch_size:,} per batch × {max_batches} max batches)"
            )
            
            lhs_start_time = time.time()
            all_candidate_positions = self._initialize_positions_with_strategy(total_lhs_size, self.bounds)
            lhs_time = time.time() - lhs_start_time
            
            self.log_message(f"   LHS generation complete in {lhs_time:.2f}s")
            
            # Handle init_pos injection for fresh start (replace first candidate)
            inject_init_pos = is_fresh_start and self.custom_init_pos is not None
            if inject_init_pos:
                all_candidate_positions[0] = np.array(self.custom_init_pos)
                self.log_message("🎯 Injected init_pos into first candidate")
            
            # Collect good candidates across batches
            good_positions = []
            good_fitness = []
            total_evaluated = 0
            batch_count = 0
            eval_start_time = time.time()
            
            # Evaluate in batches with early termination
            for batch_idx in range(max_batches):
                if len(good_positions) >= target_quality_particles:
                    break  # Early termination - found enough good candidates
                
                batch_count += 1
                
                # Extract batch from pre-generated LHS sample
                batch_start = batch_idx * batch_size
                batch_end = batch_start + batch_size
                batch_positions = all_candidate_positions[batch_start:batch_end]
                
                # Evaluate batch
                temp_swarm = type('TempSwarm', (), {})()
                temp_swarm.position = batch_positions
                batch_fitness = compute_objective_function(temp_swarm, objective_func, pool=None, **kwargs)
                
                # Keep candidates below threshold
                good_mask = batch_fitness < fitness_threshold
                n_good_in_batch = np.sum(good_mask)
                
                if n_good_in_batch > 0:
                    good_positions.extend(batch_positions[good_mask])
                    good_fitness.extend(batch_fitness[good_mask])
                
                total_evaluated += batch_size
                
                # Log batch progress
                self.log_message(
                    f"   Batch {batch_count}: {n_good_in_batch}/{batch_size} below threshold "
                    f"(total good: {len(good_positions)}/{target_quality_particles})"
                )
            
            eval_time = time.time() - eval_start_time
            
            # Calculate early termination savings
            unevaluated_batches = max_batches - batch_count
            saved_evaluations = unevaluated_batches * batch_size
            
            if saved_evaluations > 0:
                self.log_message(
                    f"⚡ Early termination: saved {saved_evaluations:,} evaluations "
                    f"({unevaluated_batches} batches skipped)"
                )
            
            # Handle results based on how many good candidates we found
            if len(good_positions) >= target_quality_particles:
                # Success - we have enough good candidates
                good_positions = np.array(good_positions)
                good_fitness = np.array(good_fitness)
                
                # For fresh start, select best n from good candidates
                # For resume/periodic, combine with existing particles
                if is_fresh_start:
                    # Fresh start - only use good candidates
                    all_positions = good_positions
                    all_fitness = good_fitness
                    n_existing = 0
                else:
                    # Combine with existing particles
                    existing_positions = self.swarm.position.copy()
                    existing_fitness = self.swarm.pbest_cost.copy()
                    
                    all_positions = np.vstack([existing_positions, good_positions])
                    all_fitness = np.concatenate([existing_fitness, good_fitness])
                    n_existing = self.n_particles
                
                # Select best n
                best_indices = np.argsort(all_fitness)[:self.n_particles]
                
                # Count existing vs new
                if n_existing > 0:
                    n_existing_kept = np.sum(best_indices < n_existing)
                    n_new_selected = self.n_particles - n_existing_kept
                else:
                    n_existing_kept = 0
                    n_new_selected = self.n_particles
                
                # Update swarm
                self.swarm.position = all_positions[best_indices].copy()
                self.swarm.pbest_pos = self.swarm.position.copy()
                self.swarm.pbest_cost = all_fitness[best_indices].copy()
                
                # Count how many of the selected particles are actually below threshold
                n_below_threshold = np.sum(self.swarm.pbest_cost < fitness_threshold)
                pct_below_threshold = (n_below_threshold / self.n_particles) * 100.0
                
                # Log success
                self.log_message(
                    f"✅ {selection_type} selection complete:\n"
                    f"   Found {len(good_positions)} quality particles in {batch_count} batches\n"
                    f"   Evaluated {total_evaluated} candidates in {eval_time:.1f}s\n"
                    f"   Existing kept: {n_existing_kept}, New selected: {n_new_selected}\n"
                    f"   Selected particles below threshold: {n_below_threshold}/{self.n_particles} ({pct_below_threshold:.1f}%)\n"
                    f"   Best fitness: {all_fitness[best_indices[0]]:.6e}\n"
                    f"   Worst selected: {all_fitness[best_indices[-1]]:.6e}\n"
                    f"   Median selected: {np.median(all_fitness[best_indices]):.6e}",
                    emoji="✅"
                )
                
            else:
                # Fallback - didn't find enough good candidates
                self.log_message(
                    f"⚠️ Only found {len(good_positions)} particles below threshold after {max_batches} batches\n"
                    f"   Target was {target_quality_particles} ({self.selection_quality_threshold_pct:.0f}%)\n"
                    f"   Evaluated {total_evaluated} candidates in {eval_time:.1f}s\n"
                    f"   Falling back to best {self.n_particles} from all evaluated",
                    emoji="⚠️"
                )
                
                # Collect all evaluated candidates
                # We need to re-evaluate or keep track - for now, use good ones + sample more
                if len(good_positions) > 0:
                    good_positions = np.array(good_positions)
                    good_fitness = np.array(good_fitness)
                    
                    # Need more candidates - sample additional batch and take best regardless of threshold
                    additional_needed = self.n_particles - len(good_positions)
                    additional_positions = self._initialize_positions_with_strategy(additional_needed, self.bounds)
                    
                    temp_swarm = type('TempSwarm', (), {})()
                    temp_swarm.position = additional_positions
                    additional_fitness = compute_objective_function(temp_swarm, objective_func, pool=None, **kwargs)
                    
                    # Combine good + additional
                    all_positions = np.vstack([good_positions, additional_positions])
                    all_fitness = np.concatenate([good_fitness, additional_fitness])
                else:
                    # No good candidates found at all - sample fresh batch
                    all_positions = self._initialize_positions_with_strategy(self.n_particles, self.bounds)
                    temp_swarm = type('TempSwarm', (), {})()
                    temp_swarm.position = all_positions
                    all_fitness = compute_objective_function(temp_swarm, objective_func, pool=None, **kwargs)
                
                # For resume/periodic, combine with existing
                if not is_fresh_start:
                    existing_positions = self.swarm.position.copy()
                    existing_fitness = self.swarm.pbest_cost.copy()
                    all_positions = np.vstack([existing_positions, all_positions])
                    all_fitness = np.concatenate([existing_fitness, all_fitness])
                
                # Select best n
                best_indices = np.argsort(all_fitness)[:self.n_particles]
                
                # Update swarm
                self.swarm.position = all_positions[best_indices].copy()
                self.swarm.pbest_pos = self.swarm.position.copy()
                self.swarm.pbest_cost = all_fitness[best_indices].copy()
            
            # Initialize velocities for all selected particles
            self.swarm.velocity = self._initialize_velocities_for_particles(self.n_particles)
            
            # Update global best
            old_global_best = self.swarm.best_cost if hasattr(self.swarm, 'best_cost') else np.inf
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)
            
            # Check if global best improved
            if self.swarm.best_cost < old_global_best:
                improvement = old_global_best - self.swarm.best_cost
                self.log_message(
                    f"🎉 Global best improved by adaptive selection! "
                    f"{old_global_best:.6e} → {self.swarm.best_cost:.6e} (Δ {improvement:.6e})",
                    emoji="🎉"
                )
            
            # Update current_cost for consistency
            self.swarm.current_cost = self.swarm.pbest_cost.copy()
            
        except Exception as e:
            self.log_message(f"⚠️ Error in adaptive quality selection: {e}", emoji="⚠️")
            # Fallback to standard initialization if this is fresh start
            if is_fresh_start:
                self.log_message("⚠️ Falling back to standard initialization", emoji="⚠️")
                self._standard_initialization(objective_func, **kwargs)

    def _initialize_velocities_for_particles(self, n_particles):
        """Initialize velocities for n particles
        
        Parameters
        ----------
        n_particles : int
            Number of particles to initialize velocities for
            
        Returns
        -------
        np.ndarray
            Array of shape (n_particles, dimensions) with initialized velocities
        """
        if self.velocity_clamp is not None:
            min_vel, max_vel = self.velocity_clamp
            return np.random.uniform(min_vel, max_vel, (n_particles, self.dimensions))
        else:
            # Default velocity initialization based on position range
            if self.bounds is not None:
                lower_bounds, upper_bounds = self.bounds
                vel_range = 0.1 * (upper_bounds - lower_bounds)  # 10% of parameter range
                return np.random.uniform(-vel_range, vel_range, (n_particles, self.dimensions))
            else:
                # Fallback velocity initialization
                return np.random.uniform(-0.1, 0.1, (n_particles, self.dimensions))

    def _get_global_best_particle_idx(self):
        """Get the index of the particle with the global best personal best fitness
        
        Returns
        -------
        int
            Index of the particle with the best personal best fitness
        """
        return np.argmin(self.swarm.pbest_cost)

    def _update_blacklist_tracking(self, iteration):
        """Update tracking data for blacklist evaluation"""
        
        if not hasattr(self, 'grid_resolution') or self.bounds is None:
            return
        
        # Update cell occupancy first
        self._update_cell_occupancy()
        
        # Track fitness and stagnation for each occupied cell
        for grid_coords, particle_indices in self.cell_occupancy.items():
            # Initialize tracking if not exists
            if grid_coords not in self.cell_fitness_tracking:
                self.cell_fitness_tracking[grid_coords] = deque(maxlen=self.blacklist_window)
            if grid_coords not in self.cell_stagnation_tracking:
                self.cell_stagnation_tracking[grid_coords] = deque(maxlen=self.blacklist_window)
            
            # Get current cell best fitness
            cell_best_fitness = min(self.swarm.pbest_cost[idx] for idx in particle_indices)
            
            # Track fitness
            self.cell_fitness_tracking[grid_coords].append((iteration, cell_best_fitness))
            
            # Track stagnation - check if any particle improved its personal best this iteration
            had_improvement = False
            for particle_idx in particle_indices:
                # Check if this particle's current cost is better than its previous pbest
                # (This would indicate an improvement happened this iteration)
                if hasattr(self, '_previous_pbest_costs'):
                    if self.swarm.current_cost[particle_idx] < self._previous_pbest_costs[particle_idx]:
                        had_improvement = True
                        break
            
            self.cell_stagnation_tracking[grid_coords].append((iteration, had_improvement))

    def _evaluate_blacklist(self, iteration):
        """Evaluate cells for blacklisting and removal from blacklist"""
        
        if not hasattr(self, 'grid_resolution') or self.bounds is None:
            return
        
        newly_blacklisted = set()
        removed_from_blacklist = set()
        
        # Check all cells with tracking data
        all_tracked_cells = set(self.cell_fitness_tracking.keys()) | set(self.cell_stagnation_tracking.keys())
        
        for grid_coords in all_tracked_cells:
            is_currently_blacklisted = grid_coords in self.blacklisted_cells
            
            # Get tracking data
            fitness_history = self.cell_fitness_tracking.get(grid_coords, deque())
            stagnation_history = self.cell_stagnation_tracking.get(grid_coords, deque())
            
            # Only evaluate if we have enough history
            if len(fitness_history) >= self.blacklist_window or len(stagnation_history) >= self.blacklist_window:
                
                should_be_blacklisted = False
                
                # Check fitness criterion: no fitness < threshold for blacklist_window iterations
                if len(fitness_history) >= self.blacklist_window:
                    recent_fitness = [fitness for _, fitness in list(fitness_history)[-self.blacklist_window:]]
                    if all(fitness >= self.blacklist_fitness_threshold for fitness in recent_fitness):
                        should_be_blacklisted = True
                
                # Check stagnation criterion: no improvements for blacklist_window iterations
                if len(stagnation_history) >= self.blacklist_window:
                    recent_improvements = [improved for _, improved in list(stagnation_history)[-self.blacklist_window:]]
                    if not any(recent_improvements):  # No improvements in the window
                        should_be_blacklisted = True
                
                # Update blacklist status
                if should_be_blacklisted and not is_currently_blacklisted:
                    self.blacklisted_cells.add(grid_coords)
                    newly_blacklisted.add(grid_coords)
                elif not should_be_blacklisted and is_currently_blacklisted:
                    self.blacklisted_cells.discard(grid_coords)
                    removed_from_blacklist.add(grid_coords)
        
        # Log blacklist changes
        if newly_blacklisted:
            self.log_message(f"🚫 Blacklisted {len(newly_blacklisted)} cells (poor fitness or stagnation)")
        
        if removed_from_blacklist:
            self.log_message(f"✅ Removed {len(removed_from_blacklist)} cells from blacklist (improvement detected)")
        
        # Log current blacklist status
        if len(self.blacklisted_cells) > 0:
            total_cells = getattr(self, '_total_grid_cells', 1)
            blacklist_percentage = (len(self.blacklisted_cells) / total_cells) * 100
            self.log_message(f"🚫 Current blacklist: {len(self.blacklisted_cells):,} cells ({blacklist_percentage:.2f}% of search space)")

    def get_competition_stats(self):
        """Get current competitive evolution and blacklist statistics
        
        Returns
        -------
        dict
            Dictionary containing competition and blacklist statistics
        """
        if not self.competition_enabled:
            return {"enabled": False}
        
        stats = {
            "enabled": True,
            "visited_cells": len(self.visited_cells),
            "total_possible_cells": getattr(self, '_total_grid_cells', 0),
            "exploration_percentage": 0.0,
            "current_cell_occupancy": {},
            "overcrowded_cells": 0,
            "stagnant_cells": 0,
            "global_improvement_rate": 0.0,
            "blacklisted_cells": len(getattr(self, 'blacklisted_cells', set())),
            "blacklist_percentage": 0.0
        }
        
        # Calculate exploration percentage
        if hasattr(self, '_total_grid_cells') and self._total_grid_cells > 0:
            stats["exploration_percentage"] = (len(self.visited_cells) / self._total_grid_cells) * 100.0
            stats["blacklist_percentage"] = (len(getattr(self, 'blacklisted_cells', set())) / self._total_grid_cells) * 100.0
        
        # Update cell occupancy and count overcrowded cells
        if hasattr(self, 'cell_occupancy'):
            stats["current_cell_occupancy"] = {str(k): len(v) for k, v in self.cell_occupancy.items()}
            stats["overcrowded_cells"] = sum(1 for v in self.cell_occupancy.values() if len(v) > self.max_particles_per_cell)
        
        # Calculate global improvement rate
        if len(self.global_improvement_history) > 0:
            improvements = [entry['improvement'] for entry in self.global_improvement_history]
            stats["global_improvement_rate"] = np.mean(improvements)
        
        return stats

    def _detect_and_spawn_scouts(self, iteration, objective_func, **kwargs):
        """Detect negative fitness and spawn hill scout groups
        
        Performance optimization: Limits spawning based on max_concurrent_groups
        configuration to prevent resource exhaustion with large swarms.
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        objective_func : callable
            Objective function for evaluation
        **kwargs : dict
            Additional arguments for objective function
        """
        if not self.scout_config['enable_scouts']:
            return
        
        try:
            # Check if we've reached the maximum number of concurrent groups
            max_groups = self.scout_config['max_concurrent_groups']
            current_groups = len(self.active_scout_groups)
            
            at_capacity = current_groups >= max_groups
            
            # Adaptive negative fitness threshold: becomes 0 when global best < 0
            base_threshold = self.scout_config['negative_fitness_threshold']
            current_global_best = self.swarm.best_cost
            
            if self.scout_config.get('adaptive_negative_fitness_threshold_enabled', False):
                if current_global_best < 0:
                    negative_threshold = 0  # Only spawn in truly negative regions
                    if not hasattr(self, '_adaptive_threshold_logged'):
                        self.log_message(
                            f"🎯 Adaptive threshold activated: global best {current_global_best:.6e} < 0, "
                            f"negative_fitness_threshold → 0 (was {base_threshold:.0e})"
                        )
                        self._adaptive_threshold_logged = True
                else:
                    negative_threshold = base_threshold
            else:
                negative_threshold = base_threshold
            
            spawned_this_iteration = 0
            
            # Check each PSO particle for negative fitness
            for particle_idx in range(self.n_particles):
                try:
                    current_fitness = self.swarm.current_cost[particle_idx]
                    
                    # Check if fitness is negative (below threshold)
                    if current_fitness < negative_threshold:
                        # Allow spawning if:
                        # 1. Below capacity, OR
                        # 2. Truly negative fitness (< 0) - bypasses capacity limit
                        can_spawn = (not at_capacity) or (current_fitness < 0)
                        
                        if not can_spawn:
                            # At capacity and not truly negative, skip this particle
                            continue
                        # Spawn hill scout group at this location
                        spawn_point = self.swarm.position[particle_idx].copy()
                        group_id = self._spawn_scout_group(
                            spawner_particle_id=particle_idx,
                            spawn_point=spawn_point,
                            trigger_fitness=current_fitness,
                            iteration=iteration
                        )
                        
                        if group_id is not None:
                            # Record the spawning event
                            self._record_spawning_event(
                                iteration=iteration,
                                spawner_particle_id=particle_idx,
                                spawn_point=spawn_point,
                                trigger_fitness=current_fitness,
                                group_id=group_id
                            )
                            
                            # Launch spawner particle away from spawn point
                            # self._launch_spawner_particle(particle_idx, spawn_point)  # Disabled: keep spawner at spawn location
                            
                            spawned_this_iteration += 1
                            
                            # Check if we're over capacity (truly negative spawn bypassed limit)
                            over_capacity = len(self.active_scout_groups) > max_groups
                            capacity_bypass = current_fitness < 0 and at_capacity
                            
                            # Log spawning event
                            if capacity_bypass:
                                self.log_message(
                                    f"🌱 Hill scout group spawned! Particle #{particle_idx} "
                                    f"(fitness: {current_fitness:.6e}) spawned group {group_id} "
                                    f"with {self.scout_config['scouts_per_spawn']} scouts "
                                    f"({len(self.active_scout_groups)}/{max_groups} active groups) "
                                    f"⚡ BYPASSED CAPACITY LIMIT (truly negative)",
                                    emoji="🌱"
                                )
                            else:
                                self.log_message(
                                    f"🌱 Hill scout group spawned! Particle #{particle_idx} "
                                    f"(fitness: {current_fitness:.6e}) spawned group {group_id} "
                                    f"with {self.scout_config['scouts_per_spawn']} scouts "
                                    f"({len(self.active_scout_groups)}/{max_groups} active groups)",
                                    emoji="🌱"
                                )
                            
                            # Update at_capacity flag for next particle
                            at_capacity = len(self.active_scout_groups) >= max_groups
                            
                except Exception as e:
                    self.log_message(
                        f"⚠️ Error spawning hill scout for particle {particle_idx}: {e}",
                        emoji="⚠️"
                    )
            
            # Log summary if multiple groups were spawned
            if spawned_this_iteration > 1:
                self.log_message(
                    f"🌱 Spawned {spawned_this_iteration} hill scout groups this iteration "
                    f"({len(self.active_scout_groups)}/{max_groups} total active)",
                    emoji="🌱"
                )
                
        except Exception as e:
            self.log_message(f"⚠️ Error in _detect_and_spawn_scouts: {e}")

    def _spawn_scout_group(self, spawner_particle_id, spawn_point, trigger_fitness, iteration):
        """Create a new scout group at the specified spawn point
        
        Uses Latin Hypercube Sampling (LHS) to initialize scout positions in a hypercube
        centered around the spawn point. This provides good initial coverage of the local region.
        
        Parameters
        ----------
        spawner_particle_id : int
            Index of the PSO particle that triggered spawning
        spawn_point : np.ndarray
            Position where the scout group should be spawned
        trigger_fitness : float
            Fitness value that triggered the spawning
        iteration : int
            Current iteration number
            
        Returns
        -------
        str or None
            Group ID if successful, None if spawning failed
        """
        try:
            # Generate unique group ID
            group_id = f"hc_group_{self._next_group_id}"
            self._next_group_id += 1
            
            scouts_per_spawn = self.scout_config['scouts_per_spawn']
            
            # Generate LHS positions in hypercube around spawn point
            scout_positions = self._generate_lhs_positions(spawn_point, scouts_per_spawn)
            
            # Create scout particles with LHS positions
            scout_ids = []
            for i in range(scouts_per_spawn):
                scout_id = f"hc_{self._next_scout_id}"
                self._next_scout_id += 1
                
                # Initialize scout particle at LHS position
                scout_particle = {
                    'id': scout_id,
                    'position': scout_positions[i].copy(),
                    'velocity': np.zeros(self.dimensions),  # Start with zero velocity
                    'group_id': group_id,
                    'local_best': scout_positions[i].copy(),
                    'local_best_fitness': np.inf
                }
                
                self.scout_particles[scout_id] = scout_particle
                scout_ids.append(scout_id)
            
            # Create scout group (simplified - no phase system)
            scout_group = {
                'id': group_id,
                'spawner_id': spawner_particle_id,
                'spawn_point': spawn_point.copy(),
                'best_point': spawn_point.copy(),
                'best_fitness': trigger_fitness,
                'remaining_iterations': self.scout_config['scout_lifetime'],
                'scout_ids': scout_ids,
                'spawn_iteration': iteration,  # Track when this group was spawned
            }
            
            self.active_scout_groups[group_id] = scout_group
            
            return group_id
            
        except Exception as e:
            self.log_message(f"⚠️ Failed to spawn scout group: {e}")
            return None

    def _generate_lhs_positions_around_center(self, center_point, hypercube_percentage, n_positions):
        """Generate positions using Latin Hypercube Sampling around a center point
        
        This is a general-purpose LHS function used for:
        1. Initial scout spawning (center=spawn_point, percentage=spawn_lhs_percentage)
        2. Scout neighbor generation (center=scout_position, percentage=initial_search_radius_percentage)
        
        Parameters
        ----------
        center_point : np.ndarray
            Center point for the LHS hypercube
        hypercube_percentage : float
            Size of hypercube as percentage of parameter ranges (0.0 to 1.0)
        n_positions : int
            Number of positions to generate
            
        Returns
        -------
        np.ndarray
            Array of shape (n_positions, dimensions) with LHS positions
        """
        try:
            # Calculate hypercube bounds centered at center_point
            if self.bounds is not None:
                lower_bounds, upper_bounds = self.bounds
                parameter_ranges = upper_bounds - lower_bounds
                half_ranges = hypercube_percentage * parameter_ranges / 2
                hypercube_lower = center_point - half_ranges
                hypercube_upper = center_point + half_ranges
                
                # Clip to global bounds
                hypercube_lower = np.maximum(hypercube_lower, lower_bounds)
                hypercube_upper = np.minimum(hypercube_upper, upper_bounds)
            else:
                # Fallback if no bounds
                half_ranges = hypercube_percentage * np.abs(center_point)
                hypercube_lower = center_point - half_ranges
                hypercube_upper = center_point + half_ranges
            
            # Generate LHS samples
            if SCIPY_AVAILABLE:
                from scipy.stats import qmc
                sampler = qmc.LatinHypercube(d=self.dimensions, seed=np.random.randint(0, 2**31))
                unit_samples = sampler.random(n=n_positions)
                
                # Scale to hypercube bounds
                lhs_positions = qmc.scale(unit_samples, hypercube_lower, hypercube_upper)
            else:
                # Fallback to uniform random if scipy not available
                self.log_message("⚠️ SciPy not available, using uniform random instead of LHS")
                lhs_positions = np.random.uniform(hypercube_lower, hypercube_upper, (n_positions, self.dimensions))
            
            return lhs_positions
            
        except Exception as e:
            self.log_message(f"⚠️ Error generating LHS positions: {e}, using center point")
            # Fallback: return center point for all positions
            return np.tile(center_point, (n_positions, 1))

    def _generate_lhs_positions(self, spawn_point, n_positions):
        """Generate positions using Latin Hypercube Sampling around spawn point
        
        This is a wrapper for initial scout spawning that uses the shared LHS function.
        
        Parameters
        ----------
        spawn_point : np.ndarray
            Center point for the LHS hypercube
        n_positions : int
            Number of positions to generate
            
        Returns
        -------
        np.ndarray
            Array of shape (n_positions, dimensions) with LHS positions
        """
        return self._generate_lhs_positions_around_center(
            spawn_point,
            self.scout_config['spawn_lhs_percentage'],
            n_positions
        )

    def _generate_scout_neighbors(self):
        """Generate LHS neighbors for all active scouts
        
        For each scout, generates local_search_neighbors positions using LHS
        in a hypercube centered at the scout's current position. These neighbors
        will be evaluated in the batch system along with PSO particles and scouts.
        
        The neighbors are stored in each scout's data structure for later evaluation.
        """
        if not self.scout_config['enable_scouts']:
            return
        
        try:
            n_neighbors = self.scout_config['local_search_neighbors']
            radius_pct = self.scout_config['initial_search_radius_percentage']
            
            # Get current search radius from group (adaptive)
            for group_id, group in self.active_scout_groups.items():
                try:
                    scout_ids = group['scout_ids']
                    
                    # Get adaptive search radius for this group
                    if 'search_radius' in group:
                        # Use adaptive radius (as percentage of parameter ranges)
                        if self.bounds is not None:
                            lower_bounds, upper_bounds = self.bounds
                            parameter_ranges = upper_bounds - lower_bounds
                            # Convert absolute radius back to percentage
                            adaptive_radius_pct = np.mean(group['search_radius'] / parameter_ranges)
                        else:
                            adaptive_radius_pct = radius_pct
                    else:
                        # Use initial radius
                        adaptive_radius_pct = radius_pct
                    
                    # Generate neighbors for each scout in the group
                    for scout_id in scout_ids:
                        if scout_id not in self.scout_particles:
                            continue
                        
                        try:
                            scout = self.scout_particles[scout_id]
                            current_position = scout['position']
                            
                            # Generate LHS neighbors around current position
                            neighbor_positions = self._generate_lhs_positions_around_center(
                                current_position,
                                adaptive_radius_pct,
                                n_neighbors
                            )
                            
                            # Store neighbors in scout data
                            scout['neighbors'] = neighbor_positions
                            scout['neighbor_fitness'] = np.full(n_neighbors, np.inf)
                            
                        except Exception as e:
                            self.log_message(
                                f"⚠️ Error generating neighbors for scout {scout_id}: {e}",
                                emoji="⚠️"
                            )
                            # Fallback: no neighbors for this scout (proper shape)
                            scout['neighbors'] = np.empty((0, self.dimensions))
                            scout['neighbor_fitness'] = np.array([])
                    
                except Exception as e:
                    self.log_message(
                        f"⚠️ Error generating neighbors for group {group_id}: {e}",
                        emoji="⚠️"
                    )
                    
        except Exception as e:
            self.log_message(f"⚠️ Error in _generate_scout_neighbors: {e}")

    def _record_spawning_event(self, iteration, spawner_particle_id, spawn_point, trigger_fitness, group_id):
        """Record a hill scout spawning event for tracking and analysis
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        spawner_particle_id : int
            Index of the PSO particle that triggered spawning
        spawn_point : np.ndarray
            Position where the hill scout group was spawned
        trigger_fitness : float
            Fitness value that triggered the spawning
        group_id : str
            ID of the spawned hill scout group
        """
        spawning_event = {
            'iteration': iteration,
            'spawner_particle_id': spawner_particle_id,
            'spawn_point': spawn_point.copy(),
            'trigger_fitness': trigger_fitness,
            'group_id': group_id,
            'timestamp': time.time()
        }
        
        self.spawning_history.append(spawning_event)

    def _launch_spawner_particle(self, particle_idx, spawn_point):
        """Launch spawner particle away from spawn point to avoid redundant exploration
        
        This method gives the spawner particle a velocity boost away from the spawn point,
        similar to the velocity boost mechanism. This ensures the spawner explores new
        regions while hill scouts optimize the spawn region.
        
        Parameters
        ----------
        particle_idx : int
            Index of the spawner particle to launch
        spawn_point : np.ndarray
            Position where hill scouts were spawned (to move away from)
        """
        try:
            if self.bounds is None:
                self.log_message("⚠️ No bounds specified, cannot launch spawner particle")
                return
            
            lower_bounds, upper_bounds = self.bounds
            current_pos = self.swarm.position[particle_idx]
            
            # Calculate direction away from spawn point (which is current position)
            # Use global best position to bias direction
            global_best_pos = self.swarm.best_pos
            
            # Direction: away from spawn point, with some bias toward exploring new regions
            # 50% away from spawn point, 50% random exploration
            away_vector = np.random.uniform(-1, 1, self.dimensions)
            away_vector = away_vector / np.linalg.norm(away_vector)  # Normalize
            
            # Calculate maximum safe distance in this direction
            max_distances = []
            for dim in range(self.dimensions):
                if abs(away_vector[dim]) >= 1e-10:
                    if away_vector[dim] > 0:
                        max_dist = (upper_bounds[dim] - current_pos[dim]) / away_vector[dim]
                        max_distances.append(max_dist)
                    else:
                        max_dist = (current_pos[dim] - lower_bounds[dim]) / abs(away_vector[dim])
                        max_distances.append(max_dist)
            
            safe_distance = np.mean(max_distances) if max_distances else 1.0
            
            # Set velocity magnitude to average safe distance (similar to velocity boost)
            magnitude = safe_distance
            
            # Apply velocity boost
            self.swarm.velocity[particle_idx] = magnitude * away_vector
            
            # Don't reset pbest - let the particle keep its best known position
            # This way it can still benefit from hill scout discoveries via attribution
            
        except Exception as e:
            self.log_message(f"⚠️ Error launching spawner particle {particle_idx}: {e}")



    def _process_scout_groups(self, iteration, objective_func, **kwargs):
        """Process all active scout groups using adaptive local search
        
        This simplified method performs local search around each scout's current position
        by generating neighbors and evaluating them in the batch system.
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        objective_func : callable
            Objective function for evaluation
        **kwargs : dict
            Additional arguments for objective function
        """
        if not self.scout_config['enable_scouts']:
            return
        
        # Get list of group IDs to avoid modification during iteration
        group_ids = list(self.active_scout_groups.keys())
        
        for group_id in group_ids:
            if group_id not in self.active_scout_groups:
                continue  # Group may have been terminated
            
            group = self.active_scout_groups[group_id]
            scout_ids = group['scout_ids']
            
            # Generate local search neighbors for each scout
            # This happens in _advance_scout_positions which is called after evaluation
            # Here we just check if group should be terminated
            if group['remaining_iterations'] <= 0:
                self._terminate_scout_group(group_id, iteration)

    def _terminate_scout_group(self, group_id, iteration):
        """Terminate a hill scout group and clean up resources
        
        Parameters
        ----------
        group_id : str
            ID of the hill scout group to terminate
        iteration : int
            Current iteration number
        """
        if group_id not in self.active_scout_groups:
            return
        
        try:
            group = self.active_scout_groups[group_id]
            scout_ids = group['scout_ids']
            spawner_id = group['spawner_id']
            
            # Ensure all attributed improvements are preserved in spawner's personal best
            # This is already handled by the attribution system during the hill scout's lifetime
            # The spawner's personal best has been updated throughout the process
            
            # Log termination with final attribution status
            final_best_fitness = group['best_fitness']
            spawner_pbest_fitness = self.swarm.pbest_cost[spawner_id] if spawner_id < self.n_particles else np.inf
            
            # Check if spawner benefited from hill scout discoveries
            spawner_benefited = spawner_pbest_fitness < np.inf and spawner_pbest_fitness <= final_best_fitness
            
            self.log_message(
                f"⚰️ Hill scout group terminated - Group {group_id}: "
                f"spawner #{spawner_id}, final best fitness: {final_best_fitness:.6e}, "
                f"spawner pbest: {spawner_pbest_fitness:.6e}, "
                f"spawner benefited: {'Yes' if spawner_benefited else 'No'}, "
                f"lifetime expired at iteration {iteration}",
                emoji="⚰️"
            )
            
            # Clean up hill scout particles
            for scout_id in scout_ids:
                if scout_id in self.scout_particles:
                    del self.scout_particles[scout_id]
            
            # Remove group from active groups
            del self.active_scout_groups[group_id]
            
            # Note: All attributed improvements remain preserved in the spawner's personal best
            # and in the global best if applicable. The attribution records are also preserved
            # in self.attribution_records for analysis.
            
        except Exception as e:
            self.log_message(f"⚠️ Error terminating hill scout group {group_id}: {e}")

    def _attribute_improvement(self, scout_id, improved_position, improved_fitness, iteration):
        """Attribute improvement found by hill scout to its spawning PSO particle
        
        Parameters
        ----------
        scout_id : str
            ID of the hill scout particle that found the improvement
        improved_position : np.ndarray
            Position where improvement was found
        improved_fitness : float
            Improved fitness value
        iteration : int
            Current iteration number
            
        Returns
        -------
        bool
            True if attribution was successful, False otherwise
        """
        if scout_id not in self.scout_particles:
            return False
        
        try:
            # Get hill scout particle and its group
            scout = self.scout_particles[scout_id]
            group_id = scout['group_id']
            
            if group_id not in self.active_scout_groups:
                return False
            
            group = self.active_scout_groups[group_id]
            spawner_id = group['spawner_id']
            
            # Validate spawner particle index
            if spawner_id < 0 or spawner_id >= self.n_particles:
                self.log_message(f"⚠️ Invalid spawner particle ID {spawner_id} for hill scout {scout_id}")
                return False
            
            # Update spawner personal best if improvement is better
            spawner_updated = self._update_spawner_personal_best(
                spawner_id, improved_position, improved_fitness, scout_id, iteration
            )
            
            # Update global best if improvement is better
            global_updated = self._update_global_best_from_scout(
                improved_position, improved_fitness, scout_id, spawner_id, iteration
            )
            
            # Record attribution event
            attribution_record = {
                'scout_id': scout_id,
                'group_id': group_id,
                'spawner_id': spawner_id,
                'improvement_fitness': improved_fitness,
                'improvement_position': improved_position.copy(),
                'iteration': iteration,
                'attribution_type': 'global_best' if global_updated else ('personal_best' if spawner_updated else 'none')
            }
            self.attribution_records.append(attribution_record)
            
            # Log attribution if any update occurred
            if spawner_updated or global_updated:
                attribution_type = "global best" if global_updated else "personal best"
                self.log_message(
                    f"🎯 Attribution success - Hill scout {scout_id} → spawner #{spawner_id}: "
                    f"updated {attribution_type} (fitness: {improved_fitness:.6e})",
                    emoji="🎯"
                )
            
            return spawner_updated or global_updated
            
        except Exception as e:
            self.log_message(f"⚠️ Error attributing improvement from hill scout {scout_id}: {e}")
            return False

    def _update_spawner_personal_best(self, spawner_id, improved_position, improved_fitness, scout_id, iteration):
        """Update spawner particle's personal best if improvement is better
        
        Parameters
        ----------
        spawner_id : int
            Index of the spawning PSO particle
        improved_position : np.ndarray
            Position where improvement was found
        improved_fitness : float
            Improved fitness value
        scout_id : str
            ID of the hill scout that found the improvement
        iteration : int
            Current iteration number
            
        Returns
        -------
        bool
            True if spawner personal best was updated, False otherwise
        """
        try:
            # Validate spawner particle index
            if spawner_id < 0 or spawner_id >= self.n_particles:
                return False
            
            current_spawner_pbest_fitness = self.swarm.pbest_cost[spawner_id]
            
            # Only update if improvement is better than current personal best
            if improved_fitness < current_spawner_pbest_fitness:
                # Update spawner's personal best
                self.swarm.pbest_pos[spawner_id] = improved_position.copy()
                self.swarm.pbest_cost[spawner_id] = improved_fitness
                
                # Log spawner personal best update
                improvement = current_spawner_pbest_fitness - improved_fitness
                self.log_message(
                    f"🌟 Spawner personal best updated - Particle #{spawner_id}: "
                    f"{current_spawner_pbest_fitness:.6e} → {improved_fitness:.6e} "
                    f"(improvement: {improvement:.6e}) via hill scout {scout_id}",
                    emoji="🌟"
                )
                
                return True
            
            return False
            
        except Exception as e:
            self.log_message(f"⚠️ Error updating spawner #{spawner_id} personal best: {e}")
            return False

    def _update_global_best_from_scout(self, improved_position, improved_fitness, scout_id, spawner_id, iteration):
        """Update global best position if hill scout improvement is better
        
        When a hill scout discovers a new global best, this method:
        1. Updates the global best position and fitness
        2. Spawns a new hill scout group at the discovery point
        
        The new group will automatically be immortal (not decrement lifetime) because
        its spawn_point will match the global best position.
        
        Parameters
        ----------
        improved_position : np.ndarray
            Position where improvement was found
        improved_fitness : float
            Improved fitness value
        scout_id : str
            ID of the hill scout that found the improvement
        spawner_id : int
            Index of the spawning PSO particle
        iteration : int
            Current iteration number
            
        Returns
        -------
        bool
            True if global best was updated, False otherwise
        """
        try:
            current_global_best_fitness = self.swarm.best_cost
            
            # Only update if improvement is better than current global best
            if improved_fitness < current_global_best_fitness:
                # Update global best
                self.swarm.best_pos = improved_position.copy()
                self.swarm.best_cost = improved_fitness
                
                # Log global best update
                improvement = current_global_best_fitness - improved_fitness
                self.log_message(
                    f"🏆 GLOBAL BEST UPDATED BY SCOUT! 🏆 "
                    f"{current_global_best_fitness:.6e} → {improved_fitness:.6e} "
                    f"(improvement: {improvement:.6e}) via hill scout {scout_id} "
                    f"(spawned by particle #{spawner_id})",
                    emoji="🏆"
                )
                
                # Spawn a new hill scout group at the discovery point
                # This explores the neighborhood of the new global best
                # The new group will automatically be immortal since its spawn_point
                # will match the global best position
                self._spawn_scout_at_discovery(
                    discovery_position=improved_position,
                    discovery_fitness=improved_fitness,
                    original_spawner_id=spawner_id,
                    discovering_scout_id=scout_id,
                    iteration=iteration
                )
                
                # Ensure all PSO particles will be attracted to the new global best
                # This happens automatically in the next PSO velocity update since
                # the global best position is used in the velocity computation
                
                return True
            
            return False
            
        except Exception as e:
            self.log_message(f"⚠️ Error updating global best from hill scout {scout_id}: {e}")
            return False

    def _spawn_scout_at_discovery(self, discovery_position, discovery_fitness, original_spawner_id, discovering_scout_id, iteration):
        """Spawn a new hill scout group at a global best discovery point
        
        This method spawns a new hill scout group when an existing hill scout
        discovers a new global best. The new group is attributed to the original
        spawner particle (not the discovering scout).
        
        Parameters
        ----------
        discovery_position : np.ndarray
            Position where the global best was discovered
        discovery_fitness : float
            Fitness value at the discovery point
        original_spawner_id : int
            Index of the original PSO particle that spawned the discovering scout's group
        discovering_scout_id : str
            ID of the hill scout that made the discovery
        iteration : int
            Current iteration number
            
        Returns
        -------
        str or None
            Group ID if successful, None if spawning failed
        """
        try:
            # Check if we've reached the maximum number of concurrent groups
            max_groups = self.scout_config['max_concurrent_groups']
            current_groups = len(self.active_scout_groups)
            
            if current_groups >= max_groups:
                self.log_message(
                    f"⚠️ Cannot spawn at discovery: max concurrent groups reached ({current_groups}/{max_groups})",
                    emoji="⚠️"
                )
                return None
            
            # Spawn new hill scout group at discovery point
            new_group_id = self._spawn_scout_group(
                spawner_particle_id=original_spawner_id,
                spawn_point=discovery_position.copy(),
                trigger_fitness=discovery_fitness,
                iteration=iteration
            )
            
            if new_group_id is not None:
                # Record the spawning event with special note about discovery-triggered spawn
                spawning_event = {
                    'iteration': iteration,
                    'spawner_particle_id': original_spawner_id,
                    'spawn_point': discovery_position.copy(),
                    'trigger_fitness': discovery_fitness,
                    'group_id': new_group_id,
                    'timestamp': time.time(),
                    'triggered_by': 'global_best_discovery',
                    'discovering_scout_id': discovering_scout_id
                }
                self.spawning_history.append(spawning_event)
                
                # Log the discovery-triggered spawn
                self.log_message(
                    f"🌟 Discovery-triggered spawn! New group {new_group_id} spawned at global best "
                    f"(fitness: {discovery_fitness:.6e}) discovered by {discovering_scout_id}, "
                    f"attributed to original spawner #{original_spawner_id} "
                    f"({len(self.active_scout_groups)}/{max_groups} active groups)",
                    emoji="🌟"
                )
            
            return new_group_id
            
        except Exception as e:
            self.log_message(f"⚠️ Error spawning at discovery: {e}")
            return None

    def get_attribution_stats(self):
        """Get current attribution system statistics
        
        Returns
        -------
        dict
            Dictionary containing attribution statistics
        """
        stats = {
            "active_scout_groups": len(self.active_scout_groups),
            "active_scout_particles": len(self.scout_particles),
            "total_attribution_records": len(self.attribution_records),
            "spawning_events": len(self.spawning_history),
            "attribution_by_type": {
                "global_best": 0,
                "personal_best": 0,
                "none": 0
            },
            "groups_by_phase": {
                "radial_sampling": 0,
                "hill_climbing": 0
            }
        }
        
        # Count attribution records by type
        for record in self.attribution_records:
            attribution_type = record.get('attribution_type', 'none')
            if attribution_type in stats["attribution_by_type"]:
                stats["attribution_by_type"][attribution_type] += 1
        
        # Count groups by phase
        for group in self.active_scout_groups.values():
            phase = group.get('phase', 'unknown')
            if phase in stats["groups_by_phase"]:
                stats["groups_by_phase"][phase] += 1
        
        return stats

    def _create_evaluation_batch(self):
        """Create evaluation batch combining PSO particles, scouts, and scout neighbors
        
        This method combines all positions that need evaluation:
        1. PSO particles (spawners)
        2. Scout current positions
        3. Scout neighbors (generated via LHS)
        
        The attribution map tracks which positions belong to which entity for
        result distribution.
        
        Returns
        -------
        tuple
            (batch_positions, attribution_map) where:
            - batch_positions: np.ndarray of shape (total_positions, dimensions)
            - attribution_map: list of dicts with keys:
              - 'type': 'pso', 'scout', or 'neighbor'
              - 'index': particle index (for PSO)
              - 'id': scout ID (for scouts)
              - 'scout_id': parent scout ID (for neighbors)
              - 'neighbor_idx': neighbor index within scout (for neighbors)
        """
        try:
            # Calculate total size for pre-allocation
            n_pso = self.n_particles
            
            # Filter scouts with valid positions
            valid_scouts = []
            total_neighbors = 0
            
            for scout_id, scout in self.scout_particles.items():
                if 'position' in scout and scout['position'] is not None:
                    if isinstance(scout['position'], np.ndarray) and scout['position'].shape == (self.dimensions,):
                        valid_scouts.append((scout_id, scout))
                        # Count neighbors for this scout
                        if 'neighbors' in scout and len(scout['neighbors']) > 0:
                            total_neighbors += len(scout['neighbors'])
            
            n_scouts = len(valid_scouts)
            total_positions = n_pso + n_scouts + total_neighbors
            
            # Pre-allocate arrays
            batch_positions = np.empty((total_positions, self.dimensions), dtype=np.float64)
            attribution_map = []
            current_idx = 0
            
            # Add PSO particle positions
            batch_positions[current_idx:current_idx + n_pso] = self.swarm.position
            for particle_idx in range(n_pso):
                attribution_map.append({
                    'type': 'pso',
                    'index': particle_idx
                })
            current_idx += n_pso
            
            # Add scout positions
            for scout_id, scout in valid_scouts:
                batch_positions[current_idx] = scout['position']
                attribution_map.append({
                    'type': 'scout',
                    'id': scout_id
                })
                current_idx += 1
            
            # Add scout neighbor positions
            for scout_id, scout in valid_scouts:
                if 'neighbors' in scout and len(scout['neighbors']) > 0:
                    neighbors = scout['neighbors']
                    n_neighbors = len(neighbors)
                    
                    # Add all neighbors for this scout
                    batch_positions[current_idx:current_idx + n_neighbors] = neighbors
                    
                    # Create attribution entries for each neighbor
                    for neighbor_idx in range(n_neighbors):
                        attribution_map.append({
                            'type': 'neighbor',
                            'scout_id': scout_id,
                            'neighbor_idx': neighbor_idx
                        })
                    
                    current_idx += n_neighbors
            
            return batch_positions, attribution_map
            
        except Exception as e:
            self.log_message(f"⚠️ Error creating evaluation batch: {e}")
            # Fallback: return just PSO particles
            return self.swarm.position.copy(), [{'type': 'pso', 'index': i} for i in range(self.n_particles)]

    def _distribute_evaluation_results(self, fitness_results, attribution_map):
        """Distribute fitness evaluation results back to correct particles
        
        This method takes the batch evaluation results and distributes them back to
        the appropriate PSO particles or hill scout particles based on the attribution map.
        
        Performance optimization: Uses vectorized operations for PSO particles and
        efficient dictionary lookups for hill scouts.
        
        Parameters
        ----------
        fitness_results : np.ndarray
            Array of fitness values from batch evaluation
        attribution_map : list of dict
            Attribution map created by _create_evaluation_batch()
            
        Returns
        -------
        bool
            True if distribution was successful, False otherwise
        """
        try:
            if len(fitness_results) != len(attribution_map):
                self.log_message(
                    f"⚠️ Fitness results length ({len(fitness_results)}) doesn't match "
                    f"attribution map length ({len(attribution_map)})",
                    emoji="⚠️"
                )
                return False
            
            # First pass: Update PSO particles (vectorized)
            pso_indices = []
            pso_fitness = []
            
            for i, attribution in enumerate(attribution_map):
                if attribution['type'] == 'pso':
                    pso_indices.append(attribution['index'])
                    pso_fitness.append(fitness_results[i])
            
            if pso_indices:
                pso_indices = np.array(pso_indices)
                pso_fitness = np.array(pso_fitness)
                
                # Validate indices
                valid_mask = (pso_indices >= 0) & (pso_indices < self.n_particles)
                if not np.all(valid_mask):
                    invalid_indices = pso_indices[~valid_mask]
                    self.log_message(
                        f"⚠️ Invalid PSO particle indices found: {invalid_indices}, skipping",
                        emoji="⚠️"
                    )
                    pso_indices = pso_indices[valid_mask]
                    pso_fitness = pso_fitness[valid_mask]
                
                if len(pso_indices) > 0:
                    self.swarm.current_cost[pso_indices] = pso_fitness
            
            # Second pass: Collect neighbor fitness for each scout
            neighbor_fitness_map = {}  # scout_id -> {neighbor_idx: fitness}
            
            for i, attribution in enumerate(attribution_map):
                if attribution['type'] == 'neighbor':
                    scout_id = attribution['scout_id']
                    neighbor_idx = attribution['neighbor_idx']
                    fitness = fitness_results[i]
                    
                    if scout_id not in neighbor_fitness_map:
                        neighbor_fitness_map[scout_id] = {}
                    neighbor_fitness_map[scout_id][neighbor_idx] = fitness
            
            # Third pass: Update scouts and evaluate neighbors
            iteration = getattr(self, '_current_iteration', 0)
            
            for i, attribution in enumerate(attribution_map):
                if attribution['type'] == 'scout':
                    scout_id = attribution['id']
                    scout_fitness = fitness_results[i]
                    
                    if scout_id not in self.scout_particles:
                        continue
                    
                    scout = self.scout_particles[scout_id]
                    
                    # Get all neighbor fitness for this scout
                    neighbor_fitness = neighbor_fitness_map.get(scout_id, {})
                    
                    # Find best among scout current position and all neighbors
                    best_fitness = scout_fitness
                    best_position = scout['position'].copy()
                    best_is_neighbor = False
                    
                    for neighbor_idx, fitness in neighbor_fitness.items():
                        if fitness < best_fitness:
                            best_fitness = fitness
                            # Safely get neighbor position
                            if 'neighbors' in scout and len(scout['neighbors']) > 0:
                                if neighbor_idx < len(scout['neighbors']):
                                    best_position = scout['neighbors'][neighbor_idx].copy()
                                    best_is_neighbor = True
                    
                    # Update scout if improvement found
                    previous_best = scout.get('local_best_fitness', np.inf)
                    
                    if best_fitness < previous_best:
                        # Improvement found - move scout to best position
                        improvement = previous_best - best_fitness
                        
                        # Log if a neighbor was better than scout's current position
                        if best_is_neighbor:
                            self.log_message(
                                f"🎯 Scout {scout_id} neighbor improvement: {best_fitness:.6e} (Δ {improvement:.6e})",
                                emoji="🎯"
                            )
                        
                        scout['position'] = best_position
                        scout['local_best'] = best_position
                        scout['local_best_fitness'] = best_fitness
                        
                        # Attribute improvement to spawner
                        try:
                            self._attribute_improvement(
                                scout_id=scout_id,
                                improved_position=best_position,
                                improved_fitness=best_fitness,
                                iteration=iteration
                            )
                        except Exception as e:
                            self.log_message(
                                f"⚠️ Error attributing improvement for scout {scout_id}: {e}",
                                emoji="⚠️"
                            )
                        
                        # Mark improvement for radius adaptation
                        scout['improved_this_iteration'] = True
                    else:
                        # No improvement
                        scout['improved_this_iteration'] = False
                        # Update current fitness even if not improved
                        scout['local_best_fitness'] = scout_fitness
            
            return True
            
        except Exception as e:
            self.log_message(f"⚠️ Error distributing evaluation results: {e}")
            return False

    def _update_scout_phases(self, iteration):
        """Update scout group lifecycle (simplified - no phase system)
        
        This method manages scout group lifecycle by decrementing lifetimes
        and terminating expired groups. The phase system has been removed.
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        """
        if not self.scout_config['enable_scouts']:
            return
        
        try:
            # Get list of group IDs to avoid modification during iteration
            group_ids = list(self.active_scout_groups.keys())
            
            for group_id in group_ids:
                if group_id not in self.active_scout_groups:
                    continue  # Group may have been terminated
                
                try:
                    group = self.active_scout_groups[group_id]
                    
                    # Check if group should be terminated (lifetime expired)
                    if group['remaining_iterations'] <= 0:
                        self._terminate_scout_group(group_id, iteration)
                        
                except Exception as e:
                    self.log_message(
                        f"⚠️ Error updating lifecycle for group {group_id}: {e}",
                        emoji="⚠️"
                    )
                    
        except Exception as e:
            self.log_message(f"⚠️ Error in _update_scout_phases: {e}")

    def _advance_scout_positions(self):
        """Adapt search radius based on scout performance
        
        This method is called after evaluation results have been distributed.
        It adapts the search radius for each group based on whether scouts improved:
        - If improved: expand radius (up to initial radius)
        - If not improved: shrink radius (down to minimum)
        
        The adaptive radius strategy balances exploration and exploitation.
        """
        if not self.scout_config['enable_scouts']:
            return
        
        try:
            # Get parameters
            initial_radius_pct = self.scout_config['initial_search_radius_percentage']
            shrink_factor = self.scout_config['radius_shrink_factor']
            min_radius_pct = self.scout_config['min_search_radius_percentage']
            
            # Calculate parameter ranges
            if self.bounds is not None:
                lower_bounds, upper_bounds = self.bounds
                parameter_ranges = upper_bounds - lower_bounds
            else:
                parameter_ranges = np.ones(self.dimensions)
            
            # Calculate absolute radius values
            initial_radius = initial_radius_pct * parameter_ranges
            min_radius = min_radius_pct * parameter_ranges
            
            # Process each scout group
            for group_id, group in self.active_scout_groups.items():
                try:
                    scout_ids = group['scout_ids']
                    
                    # Initialize search radius if not exists
                    if 'search_radius' not in group:
                        group['search_radius'] = initial_radius.copy()
                    
                    # Check if any scout in group improved
                    group_improved = False
                    for scout_id in scout_ids:
                        if scout_id in self.scout_particles:
                            scout = self.scout_particles[scout_id]
                            if scout.get('improved_this_iteration', False):
                                group_improved = True
                                break
                    
                    # Adapt radius based on performance
                    if group_improved:
                        # Improvement found - expand radius (but cap at initial)
                        new_radius = group['search_radius'] / shrink_factor
                        group['search_radius'] = np.minimum(new_radius, initial_radius)
                        
                        # # Reset scout lifecycle when improvement is found
                        # # Disabled for now
                        # old_remaining = group['remaining_iterations']
                        # group['remaining_iterations'] = self.scout_config['scout_lifetime']
                        
                        # # Log lifecycle reset
                        # self.log_message(
                        #     f"🔄 Scout group {group_id} lifecycle reset: {old_remaining} → {group['remaining_iterations']} iterations (improvement found)",
                        #     emoji="🔄"
                        # )
                    else:
                        # No improvement - shrink radius (but floor at minimum)
                        new_radius = group['search_radius'] * shrink_factor
                        group['search_radius'] = np.maximum(new_radius, min_radius)
                    
                except Exception as e:
                    self.log_message(
                        f"⚠️ Error adapting radius for group {group_id}: {e}",
                        emoji="⚠️"
                    )
                    
        except Exception as e:
            self.log_message(f"⚠️ Error in _advance_scout_positions: {e}")

    def _update_scout_lifetimes(self):
        """Update scout group lifetimes by decrementing iteration counters
        
        This method decrements the remaining iterations for all active scout groups,
        EXCEPT for groups whose spawn_point matches the current global best position.
        This method should be called once per iteration after phase updates.
        """
        if not self.scout_config['enable_scouts']:
            return
        
        # Get current global best position
        current_global_best_pos = self.swarm.best_pos if hasattr(self.swarm, 'best_pos') else None
        
        # Get list of group IDs to avoid modification during iteration
        group_ids = list(self.active_scout_groups.keys())
        
        for group_id in group_ids:
            if group_id not in self.active_scout_groups:
                continue  # Group may have been terminated
            
            group = self.active_scout_groups[group_id]
            
            # Skip lifetime decrement if this group's spawn point matches the current global best
            if current_global_best_pos is not None:
                spawn_point = group.get('spawn_point')
                if spawn_point is not None and np.allclose(spawn_point, current_global_best_pos, rtol=1e-9, atol=1e-12):
                    continue  # This group is immortal (spawned at current global best)
            
            # Decrement lifetime for all other groups
            if group['remaining_iterations'] > 0:
                group['remaining_iterations'] -= 1
        
        # Show remaining lifetime of all scout groups after each iteration
        if len(self.active_scout_groups) > 0:
            self._log_scout_group_lifetimes()

    def _log_scout_group_lifetimes(self):
        """Log the remaining lifetime of all active scout groups"""
        if not self.scout_config['enable_scouts'] or len(self.active_scout_groups) == 0:
            return
        
        try:
            # Get current global best position for comparison
            current_global_best_pos = self.swarm.best_pos if hasattr(self.swarm, 'best_pos') else None
            
            self.log_message("⏱️ Scout group lifetimes:")
            for group_id, group in self.active_scout_groups.items():
                remaining = group.get('remaining_iterations', 0)
                spawner_id = group.get('spawner_id', 'unknown')
                
                # Check if this group is immortal (spawn point matches global best)
                is_immortal = False
                if current_global_best_pos is not None:
                    spawn_point = group.get('spawn_point')
                    if spawn_point is not None and np.allclose(spawn_point, current_global_best_pos, rtol=1e-9, atol=1e-12):
                        is_immortal = True
                
                if is_immortal:
                    self.log_message(f"   {group_id} (particle {spawner_id}): {remaining} iterations (♾️ immortal - at global best)")
                else:
                    self.log_message(f"   {group_id} (particle {spawner_id}): {remaining} iterations remaining")
            
        except Exception as e:
            self.log_message(f"⚠️ Error logging scout group lifetimes: {e}")

    def _terminate_expired_groups(self, iteration):
        """Terminate scout groups that have reached zero remaining iterations
        
        This method checks all active scout groups and terminates those that have
        exhausted their lifetime (remaining_iterations <= 0). Termination includes cleanup
        of all associated resources and logging of termination events.
        
        Parameters
        ----------
        iteration : int
            Current iteration number for logging purposes
        """
        if not self.scout_config['enable_scouts']:
            return
        
        # Get list of group IDs to check for expiration
        group_ids = list(self.active_scout_groups.keys())
        
        expired_groups = []
        
        for group_id in group_ids:
            if group_id not in self.active_scout_groups:
                continue  # Group may have already been terminated
            
            group = self.active_scout_groups[group_id]
            
            # Check if group has expired
            if group['remaining_iterations'] <= 0:
                expired_groups.append(group_id)
        
        # Terminate all expired groups
        for group_id in expired_groups:
            self._terminate_scout_group(group_id, iteration)

    def _terminate_scout_group(self, group_id, iteration):
        """Terminate a scout group and cleanup all associated resources
        
        This method handles the complete termination of a scout group, including:
        - Logging termination event with statistics
        - Removing all scout particles associated with the group
        - Removing the group from active groups
        - Removing from immortal set if applicable
        - Preserving all attributed improvements in the spawner particle's personal best
        
        Parameters
        ----------
        group_id : str
            Unique identifier of the scout group to terminate
        iteration : int
            Current iteration number for logging purposes
        """
        if group_id not in self.active_scout_groups:
            return  # Group already terminated
        
        try:
            group = self.active_scout_groups[group_id]
            
            # Collect statistics before cleanup
            spawner_id = group['spawner_id']
            scout_ids = group['scout_ids']
            best_fitness = group.get('best_fitness', np.inf)
            spawn_point = group['spawn_point']
            
            # Calculate how many iterations the group was active
            lifetime = self.scout_config['scout_lifetime']
            iterations_active = lifetime - group['remaining_iterations']
            
            # Cleanup scout particles associated with this group
            self._cleanup_scout_resources(group_id)
            
            # Remove group from active groups
            del self.active_scout_groups[group_id]
            
            # Log termination event with statistics
            self.log_message(
                f"⏹️ Scout group {group_id} terminated at iteration {iteration} "
                f"(spawned by particle #{spawner_id}, "
                f"active for {iterations_active} iterations, "
                f"best fitness: {best_fitness:.6e}, "
                f"{len(scout_ids)} scouts)",
                emoji="⏹️"
            )

            
            # Note: All improvements have already been attributed to the spawner particle's
            # personal best during the optimization process via _attribute_improvement(),
            # so they are preserved automatically
            
        except Exception as e:
            self.log_message(
                f"⚠️ Error terminating scout group {group_id}: {e}",
                emoji="⚠️"
            )

    def _cleanup_scout_resources(self, group_id):
        """Cleanup all resources associated with a scout group
        
        This method removes all scout particles associated with the specified group
        from the scout_particles dictionary, freeing associated memory and data structures.
        
        Parameters
        ----------
        group_id : str
            Unique identifier of the scout group whose resources should be cleaned up
        """
        if group_id not in self.active_scout_groups:
            return  # Group doesn't exist
        
        try:
            group = self.active_scout_groups[group_id]
            scout_ids = group['scout_ids']
            
            # Remove all scout particles associated with this group
            removed_count = 0
            for scout_id in scout_ids:
                if scout_id in self.scout_particles:
                    del self.scout_particles[scout_id]
                    removed_count += 1
            
            # Clear the scout_ids list in the group
            group['scout_ids'] = []
            
            # Log cleanup if any particles were removed
            if removed_count > 0:
                self.log_message(
                    f"🧹 Cleaned up {removed_count} scout particles from group {group_id}",
                    emoji="🧹"
                )
            
        except Exception as e:
            self.log_message(
                f"⚠️ Error cleaning up resources for group {group_id}: {e}",
                emoji="⚠️"
            )

    def _log_scout_summary(self):
        """Log summary statistics for hill scout activity during optimization
        
        This method logs comprehensive statistics about hill scout performance,
        including spawning events, attribution records, and overall contribution
        to the optimization process.
        
        Requirements: 8.1, 8.2, 8.3, 8.4 - Log spawning, improvements, phase transitions, terminations
        """
        try:
            # Calculate summary statistics
            total_spawning_events = len(self.spawning_history)
            total_attribution_records = len(self.attribution_records)
            active_groups = len(self.active_scout_groups)
            active_scouts = len(self.scout_particles)
            
            # Count attribution types
            personal_best_attributions = sum(
                1 for record in self.attribution_records 
                if record['attribution_type'] == 'personal_best'
            )
            global_best_attributions = sum(
                1 for record in self.attribution_records 
                if record['attribution_type'] == 'global_best'
            )
            
            # Calculate best fitness found by hill scouts
            best_scout_fitness = np.inf
            if self.attribution_records:
                best_scout_fitness = min(
                    record['improvement_fitness'] 
                    for record in self.attribution_records
                )
            
            # Log summary with emoji-rich formatting
            self.log_message("", emoji=None)  # Empty line for spacing
            self.log_message("=" * 80, emoji=None)
            self.log_message("🏔️ SCOUT SUMMARY STATISTICS 🏔️")
            self.log_message("=" * 80, emoji=None)
            self.log_message(f"🌱 Total spawning events: {total_spawning_events}")
            self.log_message(f"🎯 Total attribution records: {total_attribution_records}")
            self.log_message(f"   ├─ Personal best updates: {personal_best_attributions}", emoji=None)
            self.log_message(f"   └─ Global best updates: {global_best_attributions}", emoji=None)
            self.log_message(f"📊 Active groups at end: {active_groups}")
            self.log_message(f"🧗 Active scouts at end: {active_scouts}")
            
            if best_scout_fitness < np.inf:
                self.log_message(
                    f"🏆 Best fitness found by hill scouts: {best_scout_fitness:.6e}",
                    emoji="🏆"
                )
            
            # Log spawning details if any spawning occurred
            if total_spawning_events > 0:
                self.log_message("", emoji=None)  # Empty line
                self.log_message("📋 Spawning Event Details:")
                
                # Group spawning events by spawner particle
                spawner_counts = {}
                for event in self.spawning_history:
                    spawner_id = event['spawner_particle_id']
                    spawner_counts[spawner_id] = spawner_counts.get(spawner_id, 0) + 1
                
                # Log top spawners
                top_spawners = sorted(spawner_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                self.log_message("   Top spawning particles:", emoji=None)
                for spawner_id, count in top_spawners:
                    self.log_message(f"   ├─ Particle #{spawner_id}: {count} groups spawned", emoji=None)
            
            # Log attribution details if any attributions occurred
            if total_attribution_records > 0:
                self.log_message("", emoji=None)  # Empty line
                self.log_message("🎯 Attribution Details:")
                
                # Group attributions by spawner particle
                spawner_attributions = {}
                for record in self.attribution_records:
                    spawner_id = record['spawner_id']
                    if spawner_id not in spawner_attributions:
                        spawner_attributions[spawner_id] = {
                            'count': 0,
                            'best_fitness': np.inf,
                            'global_best_count': 0
                        }
                    spawner_attributions[spawner_id]['count'] += 1
                    spawner_attributions[spawner_id]['best_fitness'] = min(
                        spawner_attributions[spawner_id]['best_fitness'],
                        record['improvement_fitness']
                    )
                    if record['attribution_type'] == 'global_best':
                        spawner_attributions[spawner_id]['global_best_count'] += 1
                
                # Log top contributors
                top_contributors = sorted(
                    spawner_attributions.items(),
                    key=lambda x: x[1]['count'],
                    reverse=True
                )[:5]
                
                self.log_message("   Top contributing spawners:", emoji=None)
                for spawner_id, stats in top_contributors:
                    global_best_info = f" ({stats['global_best_count']} global best)" if stats['global_best_count'] > 0 else ""
                    self.log_message(
                        f"   ├─ Particle #{spawner_id}: {stats['count']} improvements{global_best_info}, "
                        f"best: {stats['best_fitness']:.6e}",
                        emoji=None
                    )
            
            self.log_message("=" * 80, emoji=None)
            
        except Exception as e:
            self.log_message(f"⚠️ Error logging hill scout summary: {e}")
