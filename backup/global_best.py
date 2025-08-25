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
import logging
import pickle
import os
import time
import datetime
import contextlib

# Import modules
import numpy as np
import multiprocessing as mp

from collections import deque

# Advanced initialization imports
try:
    from scipy.stats import qmc
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Rich imports for beautiful logging
try:
    from rich.console import Console
    from rich.text import Text
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, TaskProgressColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# PCA imports for visualization
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..backend.operators import compute_pbest, compute_objective_function
from ..backend.topology import Star
from ..backend.handlers import BoundaryHandler, VelocityHandler, OptionsHandler
from ..base import SwarmOptimizer
from ..utils.reporter import Reporter
import subprocess

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
    ):
        """Initialize the swarm

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
        self.log_message(f"📈 History data will be saved to: {history_data_path}", emoji="📈")

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
                        self.log_message(f"📚 Loaded {existing_count} existing history records", emoji="📚")
                        
                except Exception as e:
                    self.log_message(f"⚠️ Could not load existing history: {e}, starting fresh", emoji="⚠️")
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
                self.log_message(f"📊 History data saved (iteration {iteration}, total records: {total_records})", emoji="📊")
                
        except Exception as e:
            self.log_message(f"⚠️ Failed to save history data: {e}", emoji="⚠️")

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
            # Save exploration tracking data
            # "visited_grid": getattr(self, '_visited_grid', None),
            # "grid_resolution": getattr(self, 'grid_resolution', None),
            # "total_grid_cells": getattr(self, '_total_grid_cells', None)
        }
        
        if additional_data:
            checkpoint_data.update(additional_data)
        
        try:
            with open(self.checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f)
            self.log_message(f"💾 PSO checkpoint saved at iteration {iteration}", emoji="💾")
        except Exception as e:
            self.log_message(f"⚠️ Failed to save checkpoint: {e}", emoji="⚠️")

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
                self.log_message(f"✅ Resumed with same swarm size ({current_n_particles} particles)", emoji="✅")
                
            elif checkpoint_n_particles > current_n_particles:
                # Shrinking swarm - keep best particles
                checkpoint_pbest_cost = checkpoint_data["swarm_pbest_cost"]
                best_indices = np.argsort(checkpoint_pbest_cost)[:current_n_particles]
                
                self.swarm.position = checkpoint_data["swarm_position"][best_indices]
                self.swarm.velocity = checkpoint_data["swarm_velocity"][best_indices]
                self.swarm.pbest_pos = checkpoint_data["swarm_pbest_pos"][best_indices]
                self.swarm.pbest_cost = checkpoint_data["swarm_pbest_cost"][best_indices]
                
                self.log_message(f"📉 Shrunk swarm from {checkpoint_n_particles} to {current_n_particles} particles (kept best performers)", emoji="📉")
                
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
                
                self.log_message(f"📈 Expanded swarm from {checkpoint_n_particles} to {current_n_particles} particles (added {new_particles} new particles)", emoji="📈")
            
            # Restore other state variables
            self.cost_history = checkpoint_data.get("cost_history", [])
            self.pos_history = checkpoint_data.get("pos_history", [])
            self.generation_times = checkpoint_data.get("generation_times", [])
            self.best_fitness_history = checkpoint_data.get("best_fitness_history", [])
            self.stagnation_count = checkpoint_data.get("stagnation_count", 0)
            self.last_improvement_iter = checkpoint_data.get("last_improvement_iter", 0)
            
            # Restore exploration tracking data (backwards compatible)
            if "visited_grid" in checkpoint_data and checkpoint_data["visited_grid"] is not None:
                self._visited_grid = checkpoint_data["visited_grid"]
                self.grid_resolution = checkpoint_data.get("grid_resolution", None)
                self._total_grid_cells = checkpoint_data.get("total_grid_cells", None)
                visited_cells = np.sum(self._visited_grid)
                self.log_message(f"🗺️ Restored exploration grid: {visited_cells:,} cells visited", emoji="🗺️")
            else:
                # Backwards compatibility - no exploration data in old checkpoints
                self.log_message("🗺️ No exploration data in checkpoint - will initialize fresh grid", emoji="🗺️")
            
            start_iter = checkpoint_data["iteration"] + 1
            self.log_message(f"✅ Resumed from iteration {checkpoint_data['iteration']}, best fitness: {self.swarm.best_cost:.6e}", emoji="✅")
            return start_iter
            
        except Exception as e:
            self.log_message(f"⚠️ Failed to load checkpoint: {e}, starting fresh", emoji="⚠️")
            return None

    def _generate_latin_hypercube_positions(self, n_particles, bounds):
        """Generate initial positions using Latin Hypercube Sampling for better space coverage"""
        if not SCIPY_AVAILABLE:
            self.log_message("⚠️ SciPy not available, falling back to uniform random initialization", emoji="⚠️")
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
            
            # self.log_message(f"🎯 Initialized {n_particles} particles using Latin Hypercube Sampling", emoji="🎯")
            return positions
            
        except Exception as e:
            self.log_message(f"⚠️ LHS initialization failed: {e}, falling back to uniform", emoji="⚠️")
            return self._generate_uniform_positions(n_particles, bounds)

    def _generate_sobol_positions(self, n_particles, bounds):
        """Generate initial positions using Sobol sequences for low-discrepancy coverage"""
        if not SCIPY_AVAILABLE:
            self.log_message("⚠️ SciPy not available, falling back to uniform random initialization", emoji="⚠️")
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
            
            self.log_message(f"🌐 Initialized {n_particles} particles using Sobol sequences", emoji="🌐")
            return positions
            
        except Exception as e:
            self.log_message(f"⚠️ Sobol initialization failed: {e}, falling back to uniform", emoji="⚠️")
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
            
            self.log_message(f"📊 Initialized {n_particles} particles using stratified sampling", emoji="📊")
            return positions
            
        except Exception as e:
            self.log_message(f"⚠️ Stratified initialization failed: {e}, falling back to uniform", emoji="⚠️")
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
            
            self.log_message(f"🔄 Initialized {n_particles} particles using Opposition-Based Learning", emoji="🔄")
            return positions
            
        except Exception as e:
            self.log_message(f"⚠️ Opposition-based initialization failed: {e}, falling back to uniform", emoji="⚠️")
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
            
            self.log_message(f"🎭 Initialized {n_particles} particles using hybrid approach (LHS:{n_lhs}, Sobol:{n_sobol}, Stratified:{n_stratified}, Opposition:{n_opposition})", emoji="🎭")
            return positions
            
        except Exception as e:
            self.log_message(f"⚠️ Hybrid initialization failed: {e}, falling back to uniform", emoji="⚠️")
            return self._generate_uniform_positions(n_particles, bounds)

    def _generate_uniform_positions(self, n_particles, bounds):
        """Generate positions using uniform random sampling (fallback method)"""
        if bounds is None:
            positions = np.random.uniform(-self.center, self.center, (n_particles, self.dimensions))
        else:
            lower_bounds, upper_bounds = bounds
            positions = np.random.uniform(lower_bounds, upper_bounds, (n_particles, self.dimensions))
        
        self.log_message(f"🎲 Initialized {n_particles} particles using uniform random sampling", emoji="🎲")
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
            self.log_message(f"⚠️ Significant fitness change detected ({relative_change:.2%}), clearing potentially outdated history", emoji="⚠️")
            
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
                    self.log_message(f"🗑️ Cleared history file: {self.history_data_path}", emoji="🗑️")
                except Exception as e:
                    self.log_message(f"⚠️ Failed to clear history file: {e}", emoji="⚠️")
            
            # Reset history loading flag
            if hasattr(self, '_history_loaded'):
                delattr(self, '_history_loaded')
            
            self.log_message("🔄 History cleared due to objective function changes - starting fresh tracking", emoji="🔄")
        else:
            # Minor change - keep history but add a note
            self.log_message(f"✅ Minor fitness change ({relative_change:.2%}), keeping existing history", emoji="✅")

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
        self.log_message(f"📊 Set history clear threshold to {threshold:.2%}", emoji="📊")

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
            # Clear existing grid to use new resolution
            if hasattr(self, '_visited_grid'):
                delattr(self, '_visited_grid')
            self.log_message(f"🔢 Set grid resolution to {resolution}^{self.dimensions} = {resolution**self.dimensions:,} cells", emoji="🔢")
        else:
            self.log_message("🔢 Using adaptive grid resolution based on problem dimensions", emoji="🔢")

    def reset_exploration_tracking(self):
        """Reset the exploration grid to start fresh tracking"""
        if hasattr(self, '_visited_grid'):
            delattr(self, '_visited_grid')
        self.log_message("🔄 Reset exploration tracking grid", emoji="🔄")

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
        self.log_message(f"📊 PCA visualization: {'enabled' if enable else 'disabled'} (grid: {width}x{height})", emoji="📊")

    def _create_pca_visualization(self, iteration):
        """Create PCA-based ASCII visualization of particle distribution"""
        if not self.enable_pca_visualization or not RICH_AVAILABLE or not SKLEARN_AVAILABLE:
            if not SKLEARN_AVAILABLE and iteration == 0:  # Only warn once
                self.log_message("⚠️ sklearn not available, PCA visualization disabled", emoji="⚠️")
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
        
        # Find global best position in 2D
        global_best_idx = np.argmin(self.swarm.pbest_cost)
        global_best_2d = pbest_2d[global_best_idx]
        
        # Create ASCII grid
        grid_width = self.pca_grid_width
        grid_height = self.pca_grid_height
        density_grid = np.zeros((grid_height, grid_width))
        
        # Combine current and personal best positions for density calculation
        all_positions_2d = np.vstack([positions_2d, pbest_2d])
        
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
        
        # Convert density to Rich colored characters (10 levels + empty + global best)
        max_density = np.max(density_grid) if np.max(density_grid) > 0 else 1
        
        # 10-level color gradient from cold (low density) to hot (high density)
        # Using Rich color styling with single characters for perfect alignment
        from rich.text import Text
        
        def get_colored_char(level):
            if level == 0:
                return Text("·", style="dim black")  # Empty
            elif level == 1:
                return Text("▪", style="blue")       # Coldest
            elif level == 2:
                return Text("▪", style="bright_blue")
            elif level == 3:
                return Text("▪", style="cyan")
            elif level == 4:
                return Text("▪", style="bright_cyan")
            elif level == 5:
                return Text("▪", style="green")
            elif level == 6:
                return Text("▪", style="bright_green")
            elif level == 7:
                return Text("▪", style="yellow")
            elif level == 8:
                return Text("▪", style="bright_yellow")
            elif level == 9:
                return Text("▪", style="red")
            else:  # level 10
                return Text("▪", style="bright_red")  # Hottest
        
        # Return the style string for a given level so we can color '@' similarly
        def get_style_for_level(level):
            if level == 0:
                return "dim black"
            elif level == 1:
                return "blue"
            elif level == 2:
                return "bright_blue"
            elif level == 3:
                return "cyan"
            elif level == 4:
                return "bright_cyan"
            elif level == 5:
                return "green"
            elif level == 6:
                return "bright_green"
            elif level == 7:
                return "yellow"
            elif level == 8:
                return "bright_yellow"
            elif level == 9:
                return "red"
            else:
                return "bright_red"
        
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
                if x == best_grid_x and y == best_grid_y:
                    density = density_grid[y, x]
                    if density == 0:
                        char_level = 0
                    else:
                        if max_density == 1:
                            char_level = 1
                        else:
                            normalized_density = density / max_density
                            char_level = min(int(normalized_density * 9) + 1, 10)
                    style_for_best = get_style_for_level(char_level)
                    line_text.append("@", style=style_for_best)
                else:
                    density = density_grid[y, x]
                    if density == 0:
                        char_level = 0
                    else:
                        # Map density to color level (1-10 for non-zero densities)
                        if max_density == 1:
                            char_level = 1
                        else:
                            normalized_density = density / max_density
                            char_level = min(int(normalized_density * 9) + 1, 10)
                    
                    colored_char = get_colored_char(char_level)
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
        lines.append("Legend: ·=empty ▪=density (blue→cyan→green→yellow→bright_yellow→red→bright_red) @=global best (colored by density)")
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
        self.log_message(f"🚀 Smart velocity boost: {'enabled' if enable else 'disabled'} (every {interval} iters)", emoji="🚀")
        self.log_message(f"   📊 Selection: 20%-90% particles, fitness threshold: {fitness_threshold:.0e}", emoji="📊")
        if use_exploration_prediction:
            self.log_message(f"   🎯 Multi-step exploration prediction: enabled ({n_alternative_swarms} alternatives × {prediction_steps} steps)", emoji="🎯")
            self.log_message(f"   🔍 Grid refinement: enabled at {grid_refinement_threshold}% saturation (max resolution: {max_grid_resolution})", emoji="🔍")

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
        self.log_message(f"🎯 Set initialization strategy to: {strategy}", emoji="🎯")

    def _calculate_search_space_exploration(self):
        """Calculate the percentage of search space explored using grid-based coverage
        
        This method divides the search space into a grid and tracks which grid cells
        have been visited by particles throughout the optimization process.
        
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
        
        # Initialize visited grid if not exists
        if not hasattr(self, '_visited_grid'):
            # Grid resolution per dimension (adjustable based on problem size)
            # Use fewer bins for high-dimensional problems to avoid memory explosion
            if self.dimensions <= 3:
                self.grid_resolution = 20  # 20^3 = 8,000 cells max
            elif self.dimensions <= 6:
                self.grid_resolution = 10  # 10^6 = 1,000,000 cells max
            elif self.dimensions <= 10:
                self.grid_resolution = 5   # 5^10 = 9,765,625 cells max
            else:
                self.grid_resolution = 3   # 3^n cells (manageable for high dimensions)
            
            # Create grid shape
            grid_shape = tuple([self.grid_resolution] * self.dimensions)
            self._visited_grid = np.zeros(grid_shape, dtype=bool)
            self._total_grid_cells = np.prod(grid_shape)
            
            self.log_message(f"🔢 Grid-based exploration tracking: {self.grid_resolution}^{self.dimensions} = {self._total_grid_cells:,} cells", emoji="🔢")
        
        lower_bounds, upper_bounds = self.bounds
        
        # Get all historical positions (current + personal bests for more comprehensive coverage)
        all_positions = np.vstack([
            self.swarm.position,           # Current positions
            self.swarm.pbest_pos          # Personal best positions
        ])
        
        # Convert positions to grid indices
        for pos in all_positions:
            grid_indices = []
            valid_position = True
            
            for dim in range(self.dimensions):
                # Normalize position to [0, 1] range
                normalized_pos = (pos[dim] - lower_bounds[dim]) / (upper_bounds[dim] - lower_bounds[dim])
                
                # Convert to grid index (clamp to valid range)
                grid_idx = int(normalized_pos * self.grid_resolution)
                grid_idx = max(0, min(grid_idx, self.grid_resolution - 1))
                grid_indices.append(grid_idx)
            
            if valid_position:
                # Mark this grid cell as visited
                self._visited_grid[tuple(grid_indices)] = True
        
        # Calculate exploration percentage
        visited_cells = np.sum(self._visited_grid)
        exploration_percentage = (visited_cells / self._total_grid_cells) * 100.0
        
        return exploration_percentage

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
        
        # Get all particle indices and their pbest fitness values (personal best performance)
        all_indices = np.arange(self.n_particles)
        all_pbest_fitness = self.swarm.pbest_cost
        
        # Separate particles into high fitness (above threshold) and low fitness (below threshold)
        high_fitness_mask = all_pbest_fitness >= fitness_threshold
        low_fitness_mask = all_pbest_fitness < fitness_threshold
        
        high_fitness_indices = all_indices[high_fitness_mask]
        low_fitness_indices = all_indices[low_fitness_mask]
        
        # Calculate min and max boost counts
        min_boost = max(1, int(0.20 * self.n_particles))  # Minimum 20%
        max_boost = max(1, int(0.30 * self.n_particles))  # Maximum 30%
        
        selected_indices = []
        
        # Step 1: Add all high pbest fitness particles (above threshold), prioritizing worst first
        if len(high_fitness_indices) > 0:
            # Sort high fitness particles by pbest fitness (worst first)
            high_fitness_sorted = high_fitness_indices[np.argsort(all_pbest_fitness[high_fitness_indices])[::-1]]
            # Take up to max_boost particles
            selected_indices.extend(high_fitness_sorted[:max_boost])
        
        # Step 2: If we haven't reached min_boost (20%), add worst particles from low fitness group
        if len(selected_indices) < min_boost and len(low_fitness_indices) > 0:
            # Sort low fitness particles by pbest fitness (worst first)
            low_fitness_sorted = low_fitness_indices[np.argsort(all_pbest_fitness[low_fitness_indices])[::-1]]
            # Add particles until we reach min_boost
            needed = min_boost - len(selected_indices)
            selected_indices.extend(low_fitness_sorted[:needed])
        
        # Convert to numpy array and ensure we don't exceed max_boost
        boost_indices = np.array(selected_indices[:max_boost])
        n_boost = len(boost_indices)
        
        if n_boost == 0:
            self.log_message("⚠️ No particles selected for velocity boost", emoji="⚠️")
            return
        
        # Use global best position to bias direction away from current best solution
        global_best_pos = self.swarm.best_pos
        
        # Ensure we have bounds for safe distance calculation
        if self.bounds is None:
            self.log_message("⚠️ No bounds specified, cannot perform safe velocity boost", emoji="⚠️")
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
        
        self.log_message(f"🚀 Smart velocity boost applied to {n_boost}/{self.n_particles} particles ({boost_percentage:.1f}%)", emoji="🚀")
        self.log_message(f"   📊 Selection: High pbest (≥{fitness_threshold:.0e}): {high_fitness_count}, Low pbest: {low_fitness_count}", emoji="📊")
        self.log_message(f"   📈 Boosted particles pbest - Min: {before_boost_stats['min']:.6e}, Avg: {before_boost_stats['mean']:.6e}, Max: {before_boost_stats['max']:.6e}", emoji="📈")
        self.log_message(f"   🔄 Reset pbest for all boosted particles - fresh start opportunity", emoji="🔄")
        
        # Show "after boost" statistics if we have data from previous boost
        if hasattr(self, 'last_boosted_indices') and hasattr(self, 'last_boost_iteration') and self.last_boost_iteration == iteration - self.reinit_interval:
            # Get current fitness of previously boosted particles
            prev_boosted_current_fitness = self.swarm.current_cost[self.last_boosted_indices]
            after_boost_stats = {
                'min': np.min(prev_boosted_current_fitness),
                'max': np.max(prev_boosted_current_fitness),
                'mean': np.mean(prev_boosted_current_fitness)
            }
            self.log_message(f"   📉 Previous boosted particles fitness - Min: {after_boost_stats['min']:.6e}, Avg: {after_boost_stats['mean']:.6e}, Max: {after_boost_stats['max']:.6e}", emoji="📉")
        
        self.log_message(f"   ⚡ Avg velocity magnitude: {avg_magnitude:.6f}", emoji="⚡")

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
            self.log_message(f"⚠️ init_pos shape {init_pos.shape} doesn't match dimensions {self.dimensions}, skipping injection", emoji="⚠️")
            return
        
        # Validate bounds if they exist
        if self.bounds is not None:
            lower_bounds, upper_bounds = self.bounds
            if not np.all((init_pos >= lower_bounds) & (init_pos <= upper_bounds)):
                self.log_message("⚠️ init_pos is outside bounds, clipping to bounds", emoji="⚠️")
                init_pos = np.clip(init_pos, lower_bounds, upper_bounds)
        
        # Replace the first particle with init_pos
        self.swarm.position[0] = init_pos.copy()
        
        # Give it a random velocity like other particles
        if self.velocity_clamp is not None:
            min_vel, max_vel = self.velocity_clamp
            self.swarm.velocity[0] = 0
            # self.swarm.velocity[0] = np.random.uniform(min_vel, max_vel, self.dimensions)
        else:
            if self.bounds is not None:
                lower_bounds, upper_bounds = self.bounds
                vel_range = 0.1 * (upper_bounds - lower_bounds)
                # self.swarm.velocity[0] = np.random.uniform(-vel_range, vel_range, self.dimensions)
                self.swarm.velocity[0] = 0
            else:
                self.swarm.velocity[0] = 0
                # self.swarm.velocity[0] = np.random.uniform(-0.1, 0.1, self.dimensions)
        
        self.log_message(f"🎯 Injected init_pos particle at position 0 (fresh start)", emoji="🎯")

    def _inject_init_pos_checkpoint_resume(self, objective_func, **kwargs):
        """Inject init_pos particle during checkpoint resume by replacing worst particle"""
        if self.custom_init_pos is None:
            return
        
        # Validate init_pos dimensions
        init_pos = np.array(self.custom_init_pos)
        if init_pos.shape != (self.dimensions,):
            self.log_message(f"⚠️ init_pos shape {init_pos.shape} doesn't match dimensions {self.dimensions}, skipping injection", emoji="⚠️")
            return
        
        # Validate bounds if they exist
        if self.bounds is not None:
            lower_bounds, upper_bounds = self.bounds
            if not np.all((init_pos >= lower_bounds) & (init_pos <= upper_bounds)):
                self.log_message("⚠️ init_pos is outside bounds, clipping to bounds", emoji="⚠️")
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
            self.log_message(f"🌟 init_pos particle replaced worst (fitness: {worst_fitness:.6e}) and became NEW GLOBAL BEST! (fitness: {temp_fitness[0]:.6e}, improvement: {improvement:.6e})", emoji="🌟")
        else:
            self.log_message(f"🎯 init_pos particle replaced worst (fitness: {worst_fitness:.6e} → {temp_fitness[0]:.6e}) at position {worst_particle_idx}", emoji="🎯")

    def optimize(
        self, objective_func, iters, n_processes=None, verbose=True, **kwargs
    ):
        """Optimize the swarm for a number of iterations with checkpoint support

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

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
            the global best cost and the global best position.
        """
        # Color constants
        CYAN = "\033[96m"
        RESET = "\033[0m"
        
        # Initialize timing
        self.start_time = time.time()
        
        # Try to load from checkpoint
        start_iter = self.load_checkpoint()
        if start_iter is None:
            start_iter = 0
            self.log_message("🚀 Starting fresh PSO optimization", emoji="🚀")
        
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
        
        self.log_message(f"🐝 Starting PSO evolution from iteration {start_iter}", emoji="🐝")
        self.log_message(f"📊 Swarm size: {self.n_particles}", emoji="📊")
        
        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        # pool = None if n_processes is None else mp.Pool(n_processes)

        # Initialize if starting fresh
        if start_iter == 0:
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
            
            self.best_fitness_history = []
            self.stagnation_count = 0
            self.last_improvement_iter = 0
            
            self.log_message(f"🎯 Initial evaluation complete - Global best: {self.swarm.best_cost:.6e}", emoji="🎯")
        
        # Initialize exploration grid if not already done (for both fresh start and checkpoint resume)
        if not hasattr(self, '_visited_grid') and self.bounds is not None:
            self._initialize_exploration_grid()
            # Mark initial positions as visited
            self._update_exploration_grid(self.swarm.position)
        
        ftol_history = deque(maxlen=self.ftol_iter)
        previous_best_fitness = self.swarm.best_cost if hasattr(self.swarm, 'best_cost') else float('inf')
        
        # Main PSO iteration loop
        for i in range(start_iter, iters):
            iter_start_time = time.time()
            
            if RICH_AVAILABLE:
                self.console_wrapper(Rule(f"Iteration {i}", style="bold blue"))
            
            # Velocity boost for worst particles to escape stagnation
            self._boost_worst_particles_velocity(i)
            
            # Store previous personal best costs for improvement tracking
            previous_pbest_cost = self.swarm.pbest_cost.copy()
            
            # Compute cost for current position and personal best
            self.swarm.current_cost = compute_objective_function(self.swarm, objective_func, pool=None, **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            
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
                    self.log_message(f"🎉 NEW GLOBAL BEST! 🎉 Fitness: {self.swarm.best_cost:.6e} (improved by {improvement:.6e})", emoji="🌟")
                
                # Log the best particle's optimization status
                self._log_best_particle_status(i, objective_func, **kwargs)
            else:
                self.stagnation_count += 1
            
            self.best_fitness_history.append(self.swarm.best_cost)
            
            # Update exploration grid with current positions
            if hasattr(self, '_visited_grid'):
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
            #             self.log_message(f"🎯 Convergence achieved at iteration {i}", emoji="🎯")
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
                        self.log_message(f"🔄 Velocity restart triggered! Restarted {n_restart} particles (avg_vel: {avg_velocity:.6e})", emoji="🚀")
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
                
                # Calculate additional exploration metrics
                if hasattr(self, '_visited_grid'):
                    visited_cells = np.sum(self._visited_grid)
                    total_cells = self._total_grid_cells
                    exploration_detail = f"{visited_cells:,}/{total_cells:,} cells"
                else:
                    exploration_detail = "initializing..."
                
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
                    ⚡ Avg velocity: {avg_velocity:.6f}
                    🔧 Inertia (w): {self.swarm.options['w']:.3f}
                    ⏱️ Iter time: {iter_time:.2f} sec / {(iter_time/60):.2f} min
                    📆 Avg iter time: {avg_iter_time:.2f} sec""",
                    panel=True, timestamp=False
                )
            
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
            self.log_message(f"🕒 Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)", emoji="🕒")
            self.log_message(f"🏆 Final best fitness: {final_best_cost:.6e}", emoji="🏆")
        
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=log_level,
        )
        
        # Close Pool of Processes
        if n_processes is not None:
            pool.close()
            
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
                    
                    self.log_message(f"🌟 Best particle status logged to {self.best_log_path} (iteration {iteration})", emoji="🌟")
                else:
                    self.log_message("⚠️ PSO global variables not found, cannot log best particle status", emoji="⚠️")
                    
            except ImportError as e:
                self.log_message(f"⚠️ Could not import alternatives module: {e}", emoji="⚠️")
            except Exception as e:
                self.log_message(f"⚠️ Error accessing PSO globals: {e}", emoji="⚠️")
                
        except Exception as e:
            self.log_message(f"⚠️ Failed to log best particle status: {e}", emoji="⚠️")

    def _adaptive_boost_strategy(self, iteration):
        """Adaptive boost strategy based on exploration saturation"""
        
        # Calculate exploration saturation
        exploration_saturation = self._calculate_exploration_saturation()
        
        # Get refinement threshold (default to 95% if not set)
        refinement_threshold = getattr(self, 'grid_refinement_threshold', 95.0)
        
        # Determine strategy based on saturation level
        if exploration_saturation >= refinement_threshold:
            # High saturation - refine grid and continue exploration
            self.log_message(f"🎯 Exploration saturation: {exploration_saturation:.1f}% (≥{refinement_threshold}%), refining grid resolution", emoji="🎯")
            self._refine_exploration_grid()
            self._boost_particles_with_exploration_prediction(iteration)
        else:
            # Normal saturation - use exploration prediction
            self.log_message(f"🎯 Exploration saturation: {exploration_saturation:.1f}%, using exploration prediction", emoji="🎯")
            self._boost_particles_with_exploration_prediction(iteration)

    def _calculate_exploration_saturation(self):
        """Calculate the percentage of search space explored"""
        
        if not hasattr(self, '_visited_grid') or self._visited_grid is None:
            return 0.0
        
        visited_cells = np.sum(self._visited_grid)
        total_cells = self._total_grid_cells
        saturation = (visited_cells / total_cells) * 100.0
        
        return saturation

    def _refine_exploration_grid(self):
        """Double the grid resolution to create new unexplored regions for finer exploration"""
        
        if not hasattr(self, '_visited_grid') or self._visited_grid is None:
            self.log_message("⚠️ No exploration grid to refine", emoji="⚠️")
            return
        
        old_resolution = self.grid_resolution
        new_resolution = old_resolution * 2
        
        # Prevent excessive memory usage - limit maximum resolution
        max_resolution = getattr(self, 'max_grid_resolution', 50)
        if new_resolution > max_resolution:
            self.log_message(f"⚠️ Grid resolution limit reached ({max_resolution}), cannot refine further", emoji="⚠️")
            return
        
        try:
            # Create new higher-resolution grid
            new_grid_shape = tuple([new_resolution] * self.dimensions)
            new_visited_grid = np.zeros(new_grid_shape, dtype=bool)
            
            self.log_message(f"🔍 Refining exploration grid: {old_resolution}^{self.dimensions} → {new_resolution}^{self.dimensions}", emoji="🔍")
            
            # Map old grid to new grid (each old cell becomes 2^dimensions new cells)
            for old_coords in np.ndindex(self._visited_grid.shape):
                if self._visited_grid[old_coords]:
                    # Map to corresponding region in new grid
                    new_coords_base = tuple(coord * 2 for coord in old_coords)
                    
                    # Mark the corresponding 2^d cells in new grid as visited
                    for offset in np.ndindex(tuple([2] * self.dimensions)):
                        new_coords = tuple(base + off for base, off in zip(new_coords_base, offset))
                        if all(coord < new_resolution for coord in new_coords):
                            new_visited_grid[new_coords] = True
            
            # Update grid attributes
            self._visited_grid = new_visited_grid
            self.grid_resolution = new_resolution
            self._total_grid_cells = np.prod(new_grid_shape)
            
            # Calculate new exploration statistics
            visited_cells = np.sum(self._visited_grid)
            new_saturation = (visited_cells / self._total_grid_cells) * 100.0
            unexplored_cells = self._total_grid_cells - visited_cells
            
            self.log_message(f"   📊 New grid: {self._total_grid_cells:,} total cells, {visited_cells:,} visited ({new_saturation:.1f}%)", emoji="📊")
            self.log_message(f"   🗺️ Created {unexplored_cells:,} new unexplored cells for finer exploration", emoji="🗺️")
            
        except MemoryError:
            self.log_message(f"⚠️ Not enough memory to refine grid to {new_resolution}^{self.dimensions}, keeping current resolution", emoji="⚠️")
        except Exception as e:
            self.log_message(f"⚠️ Error refining exploration grid: {e}", emoji="⚠️")

    def _boost_particles_with_exploration_prediction(self, iteration):
        """Boost particles using distance-maximization placement"""
        
        # Step 1: Select particles to boost using existing logic
        boost_indices = self._select_particles_for_boost()
        
        if len(boost_indices) == 0:
            self.log_message("⚠️ No particles selected for distance-maximization boost", emoji="⚠️")
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
            self.log_message(f"   🚀 Farthest-point boost applied to {n_boost}/{self.n_particles} particles ({boost_percentage:.1f}%)", emoji="🚀")
            self.log_message(f"   📊 Selection: High pbest (≥{fitness_threshold:.0e}): {high_fitness_count}, Low pbest: {low_fitness_count}", emoji="📊")
            self.log_message(f"   📈 Boosted particles pbest - Min: {before_boost_stats['min']:.6e}, Avg: {before_boost_stats['mean']:.6e}, Max: {before_boost_stats['max']:.6e}", emoji="📈")
            self.log_message(f"   🔄 Reset pbest for all boosted particles - fresh start opportunity", emoji="🔄")
            self.log_message(f"   🎯 Placed particles at farthest candidates from discovered regions", emoji="🎯")
            
            # Show "after boost" statistics if we have data from previous boost
            if hasattr(self, 'last_boosted_indices') and hasattr(self, 'last_boost_iteration') and self.last_boost_iteration == iteration - self.reinit_interval:
                # Get current fitness of previously boosted particles
                prev_boosted_current_fitness = self.swarm.current_cost[self.last_boosted_indices]
                after_boost_stats = {
                    'min': np.min(prev_boosted_current_fitness),
                    'max': np.max(prev_boosted_current_fitness),
                    'mean': np.mean(prev_boosted_current_fitness)
                }
                self.log_message(f"   � Preveious boosted particles fitness - Min: {after_boost_stats['min']:.6e}, Avg: {after_boost_stats['mean']:.6e}, Max: {after_boost_stats['max']:.6e}", emoji="📉")
            
        else:
            # Fallback to traditional boost
            self.log_message("⚠️ Distance maximization failed, using traditional boost", emoji="⚠️")
            self._boost_particles_traditional(iteration)

    def _select_particles_for_boost(self):
        """Select particles for boosting using existing sophisticated criteria"""
        
        # Get fitness threshold
        optimize_limits_len = getattr(self, 'optimize_limits_len', 5)
        fitness_threshold = 10 ** optimize_limits_len
        
        # Get all particle indices and their pbest fitness values
        all_indices = np.arange(self.n_particles)
        all_pbest_fitness = self.swarm.pbest_cost
        
        # Separate particles into high fitness (above threshold) and low fitness (below threshold)
        high_fitness_mask = all_pbest_fitness >= fitness_threshold
        low_fitness_mask = all_pbest_fitness < fitness_threshold
        
        high_fitness_indices = all_indices[high_fitness_mask]
        low_fitness_indices = all_indices[low_fitness_mask]
        
        # Calculate min and max boost counts
        min_boost = max(1, int(0.60 * self.n_particles))  # Minimum 20%
        max_boost = max(1, int(0.90 * self.n_particles))  # Maximum 30%
        
        selected_indices = []
        
        # Step 1: Add all high pbest fitness particles (above threshold), prioritizing worst first
        if len(high_fitness_indices) > 0:
            # Sort high fitness particles by pbest fitness (worst first)
            high_fitness_sorted = high_fitness_indices[np.argsort(all_pbest_fitness[high_fitness_indices])[::-1]]
            # Take up to max_boost particles
            selected_indices.extend(high_fitness_sorted[:max_boost])
        
        # Step 2: If we haven't reached min_boost (20%), add worst particles from low fitness group
        if len(selected_indices) < min_boost and len(low_fitness_indices) > 0:
            # Sort low fitness particles by pbest fitness (worst first)
            low_fitness_sorted = low_fitness_indices[np.argsort(all_pbest_fitness[low_fitness_indices])[::-1]]
            # Add particles until we reach min_boost
            needed = min_boost - len(selected_indices)
            selected_indices.extend(low_fitness_sorted[:needed])
        
        # Convert to numpy array and ensure we don't exceed max_boost
        boost_indices = np.array(selected_indices[:max_boost])
        
        return boost_indices



    def _position_to_grid_coords(self, position):
        """Convert position to grid coordinates"""
        
        if not hasattr(self, '_visited_grid') or self.bounds is None:
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
        """Initialize the exploration grid for tracking visited regions"""
        
        if self.bounds is None:
            self.log_message("⚠️ Cannot initialize exploration grid without bounds", emoji="⚠️")
            return
        
        # Set grid resolution based on problem dimensions
        if self.dimensions <= 3:
            self.grid_resolution = 20  # 20^3 = 8,000 cells max
        elif self.dimensions <= 6:
            self.grid_resolution = 10  # 10^6 = 1,000,000 cells max
        elif self.dimensions <= 10:
            self.grid_resolution = 5   # 5^10 = 9,765,625 cells max
        else:
            self.grid_resolution = 3   # 3^n cells (manageable for high dimensions)
        
        # Create grid shape
        grid_shape = tuple([self.grid_resolution] * self.dimensions)
        self._visited_grid = np.zeros(grid_shape, dtype=bool)
        self._total_grid_cells = np.prod(grid_shape)
        
        # Initialize discovered coords buffer
        self._discovered_coords = deque(maxlen=self.fp_discovered_cap)
        
        self.log_message(f"🗺️ Initialized exploration grid: {self.grid_resolution}^{self.dimensions} = {self._total_grid_cells:,} cells", emoji="🗺️")

    def _update_exploration_grid(self, positions):
        """Update exploration grid with visited positions"""
        
        if not hasattr(self, '_visited_grid'):
            return
        
        new_cells = 0
        for pos in positions:
            grid_coords = self._position_to_grid_coords(pos)
            if grid_coords is not None:
                if not self._visited_grid[grid_coords]:
                    self._visited_grid[grid_coords] = True
                    new_cells += 1
                    # Track newly discovered coord without scanning whole grid
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
                self.log_message("🎯 No discovered anchors, using LHS initialization", emoji="🎯")
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
            
            self.log_message(f"📌 Farthest-point selected {len(selected)} positions from {C} candidates and {anchors_np.shape[0]} anchors", emoji="📌")
            return np.asarray(selected, dtype=np.float64)
        except Exception as e:
            self.log_message(f"⚠️ Farthest-point sampling failed: {e}", emoji="⚠️")
            return self._generate_latin_hypercube_positions(n_particles, self.bounds)

    def _generate_candidate_positions(self, n_candidates):
        """Generate candidate positions in continuous space within bounds/center."""
        try:
            if SCIPY_AVAILABLE and self.fp_use_sobol:
                sampler = qmc.Sobol(d=self.dimensions, scramble=True, seed=np.random.randint(0, 2**31))
                n_sobol = 2 ** int(np.ceil(np.log2(n_candidates)))
                sample = sampler.random(n=n_sobol)[:n_candidates]
                if self.bounds is not None:
                    lower, upper = self.bounds
                    return qmc.scale(sample, lower, upper)
                else:
                    return qmc.scale(sample, -self.center, self.center)
            elif SCIPY_AVAILABLE:
                sampler = qmc.LatinHypercube(d=self.dimensions, seed=np.random.randint(0, 2**31))
                sample = sampler.random(n=n_candidates)
                if self.bounds is not None:
                    lower, upper = self.bounds
                    return qmc.scale(sample, lower, upper)
                else:
                    return qmc.scale(sample, -self.center, self.center)
            else:
                # Uniform fallback
                if self.bounds is not None:
                    lower, upper = self.bounds
                    return np.random.uniform(lower, upper, size=(n_candidates, self.dimensions))
                else:
                    return np.random.uniform(-self.center, self.center, size=(n_candidates, self.dimensions))
        except Exception as e:
            self.log_message(f"⚠️ Candidate generation failed: {e}, using uniform fallback", emoji="⚠️")
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
