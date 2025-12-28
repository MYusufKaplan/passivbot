#!/usr/bin/env python3
"""
Fitness Landscape Mapper for Trading Strategy Optimization

This tool maps the fitness landscape characteristics to help design better
optimization strategies. It analyzes:
- Ruggedness (local optima density)
- Modality (number of basins)
- Gradient information
- Correlation structure
- Neutrality (flat regions)
"""

import numpy as np
import asyncio
import argparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import logging
from multiprocessing import Pool, cpu_count
import contextlib
import time

# Rich imports for progress bar
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, TaskProgressColumn

# Import your optimization setup
from optimize import initEvaluator

# Initialize console
console = Console(
    force_terminal=True,
    no_color=False,
    log_path=False,
    width=191,
    color_system="truecolor",
    legacy_windows=False
)


class FitnessLandscapeMapper:
    """Map and analyze fitness landscape characteristics"""
    
    def __init__(self, evaluator, config, param_bounds):
        self.evaluator = evaluator
        self.config = config
        
        # Filter out fixed parameters (where min == max)
        self.param_bounds = {}
        self.fixed_params = {}
        
        for param_name, bounds in param_bounds.items():
            min_val, max_val = bounds
            if min_val != max_val:
                self.param_bounds[param_name] = bounds
            else:
                self.fixed_params[param_name] = min_val
        
        self.all_param_names = list(param_bounds.keys())
        self.optimizable_param_names = list(self.param_bounds.keys())
        self.dimensions = len(self.param_bounds)
        
        # Storage for samples
        self.samples = []
        self.fitness_values = []
        
        logging.info(f"Initialized landscape mapper for {self.dimensions}D problem ({len(self.fixed_params)} fixed parameters)")
    
    def sample_landscape(self, n_samples=1000, method='lhs', adaptive=True, focus_regions=None):
        """Sample the fitness landscape with adaptive refinement and multiprocessing
        
        ENHANCED: Adaptive sampling focuses on interesting regions with parallel evaluation
        
        Parameters
        ----------
        n_samples : int
            Number of samples to take
        method : str
            Sampling method: 'lhs', 'random', 'grid', 'sobol'
        adaptive : bool
            Use adaptive sampling to focus on high-gradient regions
        focus_regions : list of tuples, optional
            Specific regions to focus sampling (list of (center, radius) tuples)
        """
        logging.info(f"Sampling landscape with {n_samples} points using {method}")
        
        if adaptive and n_samples >= 500:
            # Phase 1: Initial broad sampling (30% of budget)
            n_initial = int(n_samples * 0.3)
            logging.info(f"Phase 1: Initial broad sampling ({n_initial} samples)")
            positions_initial = self._generate_samples(n_initial, method)
            
            # Evaluate with multiprocessing and progress bar
            self._evaluate_positions_parallel(positions_initial, "Phase 1: Broad Sampling")
            
            # Phase 2: Identify interesting regions
            logging.info("Phase 2: Identifying interesting regions...")
            self.samples = np.array(self.samples)
            self.fitness_values = np.array(self.fitness_values)
            
            interesting_regions = self._identify_interesting_regions()
            
            # Phase 3: Focused sampling (70% of budget)
            n_focused = n_samples - n_initial
            logging.info(f"Phase 3: Focused sampling in {len(interesting_regions)} regions ({n_focused} samples)")
            
            positions_focused = self._generate_focused_samples(n_focused, interesting_regions, method)
            
            # Convert back to list for appending
            self.samples = list(self.samples)
            self.fitness_values = list(self.fitness_values)
            
            # Evaluate with multiprocessing and progress bar
            self._evaluate_positions_parallel(positions_focused, "Phase 3: Focused Sampling")
            
            # Final conversion to numpy arrays
            self.samples = np.array(self.samples)
            self.fitness_values = np.array(self.fitness_values)
        else:
            # Standard sampling
            positions = self._generate_samples(n_samples, method)
            
            # Evaluate with multiprocessing and progress bar
            self._evaluate_positions_parallel(positions, "Evaluating Landscape")
            
            self.samples = np.array(self.samples)
            self.fitness_values = np.array(self.fitness_values)
        
        logging.info(f"Sampling complete. Fitness range: [{np.min(self.fitness_values):.2e}, {np.max(self.fitness_values):.2e}]")
        
        # Calculate sampling efficiency
        self._calculate_sampling_efficiency()
    
    def _evaluate_positions_parallel(self, positions, description="Evaluating"):
        """Evaluate positions using multiprocessing with rich progress bar
        
        Parameters
        ----------
        positions : np.ndarray
            Array of positions to evaluate
        description : str
            Description for progress bar
        """
        # Prepare evaluation arguments with fixed parameters
        eval_args = [
            (self.evaluator, pos, i, self.fixed_params, self.all_param_names, self.optimizable_param_names)
            for i, pos in enumerate(positions)
        ]
        
        # Create multiprocessing pool
        n_processes = max(1, cpu_count() - 1)
        pool = Pool(processes=n_processes)
        
        # Evaluate with progress bar
        with Progress(
            SpinnerColumn(spinner_name="dots12"),
            TextColumn(f"🗺️ [progress.description]{{task.description}}"),
            BarColumn(bar_width=None),
            "•",
            TaskProgressColumn(text_format="[progress.percentage]{task.percentage:>5.1f}%", show_speed=True),
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            "•",
            console=console,
            transient=False
        ) as progress:
            
            task = progress.add_task(description, total=len(positions))
            
            # Track best fitness found
            best_fitness = float('inf')
            
            # Collect results
            for result in pool.imap_unordered(self._evaluate_single, eval_args):
                fitness, idx = result
                self.samples.append(positions[idx])
                self.fitness_values.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                
                progress.update(
                    task,
                    advance=1,
                    description=f"{description} | Best: {best_fitness:.2e}"
                )
        
        pool.close()
        pool.join()
    
    @staticmethod
    def _evaluate_single(args):
        """Static method for multiprocessing evaluation
        
        Parameters
        ----------
        args : tuple
            (evaluator, position, index, fixed_params, all_param_names, optimizable_param_names)
        
        Returns
        -------
        tuple
            (fitness, index)
        """
        evaluator, pos, idx, fixed_params, all_param_names, optimizable_param_names = args
        
        # Reconstruct full parameter vector with fixed parameters
        full_pos = []
        opt_idx = 0
        
        for param_name in all_param_names:
            if param_name in fixed_params:
                full_pos.append(fixed_params[param_name])
            else:
                full_pos.append(pos[opt_idx])
                opt_idx += 1
        
        fitness, _, _ = evaluator.evaluate(full_pos)
        return fitness, idx
    
    def _generate_samples(self, n_samples, method):
        """Generate samples using specified method"""
        if method == 'lhs':
            return self._generate_lhs_samples(n_samples)
        elif method == 'sobol':
            return self._generate_sobol_samples(n_samples)
        elif method == 'grid':
            return self._generate_grid_samples(n_samples)
        else:  # random
            return self._generate_random_samples(n_samples)
    
    def _identify_interesting_regions(self):
        """Identify regions with high gradients or extreme values
        
        Returns
        -------
        list of dict
            List of interesting regions with center and radius
        """
        interesting_regions = []
        
        # Find regions with extreme fitness (top 10% and bottom 10%)
        top_10_indices = np.argsort(self.fitness_values)[:int(len(self.fitness_values) * 0.1)]
        bottom_10_indices = np.argsort(self.fitness_values)[-int(len(self.fitness_values) * 0.1):]
        
        extreme_indices = np.concatenate([top_10_indices, bottom_10_indices])
        
        # Find regions with high gradients
        distances = squareform(pdist(self.samples))
        fitness_diff = np.abs(self.fitness_values[:, None] - self.fitness_values[None, :])
        
        # Calculate local gradient for each point
        gradients = []
        for i in range(len(self.samples)):
            neighbors = np.argsort(distances[i])[1:11]  # 10 nearest neighbors
            local_gradient = np.mean(fitness_diff[i, neighbors] / (distances[i, neighbors] + 1e-10))
            gradients.append(local_gradient)
        
        gradients = np.array(gradients)
        high_gradient_indices = np.where(gradients > np.percentile(gradients, 75))[0]
        
        # Combine extreme and high-gradient regions
        focus_indices = np.unique(np.concatenate([extreme_indices, high_gradient_indices]))
        
        # Create regions around focus points
        bounds = np.array(list(self.param_bounds.values()))
        param_ranges = bounds[:, 1] - bounds[:, 0]
        
        for idx in focus_indices:
            center = self.samples[idx]
            # Radius is 5% of parameter ranges
            radius = 0.05 * param_ranges
            
            interesting_regions.append({
                'center': center,
                'radius': radius,
                'fitness': self.fitness_values[idx],
                'gradient': gradients[idx]
            })
        
        logging.info(f"Identified {len(interesting_regions)} interesting regions:")
        logging.info(f"  - {len(extreme_indices)} extreme fitness regions")
        logging.info(f"  - {len(high_gradient_indices)} high gradient regions")
        
        return interesting_regions
    
    def _generate_focused_samples(self, n_samples, regions, method):
        """Generate samples focused on interesting regions
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        regions : list of dict
            Interesting regions to focus on
        method : str
            Sampling method
        
        Returns
        -------
        np.ndarray
            Focused sample positions
        """
        if not regions:
            return self._generate_samples(n_samples, method)
        
        samples_per_region = n_samples // len(regions)
        focused_samples = []
        
        for region in regions:
            center = region['center']
            radius = region['radius']
            
            # Generate samples in this region
            if method == 'lhs':
                from scipy.stats import qmc
                sampler = qmc.LatinHypercube(d=self.dimensions)
                unit_samples = sampler.random(n=samples_per_region)
                
                # Scale to region bounds
                lower = np.maximum(center - radius, np.array([b[0] for b in self.param_bounds.values()]))
                upper = np.minimum(center + radius, np.array([b[1] for b in self.param_bounds.values()]))
                
                region_samples = qmc.scale(unit_samples, lower, upper)
            else:
                # Random sampling in region
                lower = np.maximum(center - radius, np.array([b[0] for b in self.param_bounds.values()]))
                upper = np.minimum(center + radius, np.array([b[1] for b in self.param_bounds.values()]))
                region_samples = np.random.uniform(lower, upper, (samples_per_region, self.dimensions))
            
            focused_samples.append(region_samples)
        
        return np.vstack(focused_samples)
    
    def _calculate_sampling_efficiency(self):
        """Calculate how efficiently we sampled the space"""
        if len(self.samples) < 2:
            return
        
        # Calculate coverage using minimum spanning tree length
        from scipy.sparse.csgraph import minimum_spanning_tree
        distances = squareform(pdist(self.samples))
        mst = minimum_spanning_tree(distances)
        coverage_score = mst.sum() / len(self.samples)
        
        # Calculate diversity (average distance between samples)
        diversity = np.mean(distances[np.triu_indices_from(distances, k=1)])
        
        logging.info(f"Sampling efficiency:")
        logging.info(f"  - Coverage score: {coverage_score:.4f}")
        logging.info(f"  - Average diversity: {diversity:.4f}")
        
        self.sampling_efficiency = {
            'coverage_score': float(coverage_score),
            'diversity': float(diversity)
        }
    
    def _generate_lhs_samples(self, n_samples):
        """Generate Latin Hypercube Samples"""
        from scipy.stats import qmc
        
        sampler = qmc.LatinHypercube(d=self.dimensions)
        unit_samples = sampler.random(n=n_samples)
        
        # Scale to bounds
        bounds = np.array(list(self.param_bounds.values()))
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        
        return qmc.scale(unit_samples, lower, upper)
    
    def _generate_sobol_samples(self, n_samples):
        """Generate Sobol sequence samples"""
        from scipy.stats import qmc
        
        sampler = qmc.Sobol(d=self.dimensions, scramble=True)
        n_sobol = 2**int(np.ceil(np.log2(n_samples)))
        unit_samples = sampler.random(n=n_sobol)[:n_samples]
        
        bounds = np.array(list(self.param_bounds.values()))
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        
        return qmc.scale(unit_samples, lower, upper)
    
    def _generate_random_samples(self, n_samples):
        """Generate uniform random samples"""
        bounds = np.array(list(self.param_bounds.values()))
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        
        return np.random.uniform(lower, upper, (n_samples, self.dimensions))
    
    def _generate_grid_samples(self, n_samples):
        """Generate grid samples (limited dimensions)"""
        if self.dimensions > 4:
            logging.warning("Grid sampling not recommended for >4D, using LHS instead")
            return self._generate_lhs_samples(n_samples)
        
        # Calculate points per dimension
        points_per_dim = int(np.ceil(n_samples ** (1/self.dimensions)))
        
        bounds = np.array(list(self.param_bounds.values()))
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        
        # Create grid
        grids = [np.linspace(lower[i], upper[i], points_per_dim) for i in range(self.dimensions)]
        mesh = np.meshgrid(*grids)
        positions = np.column_stack([m.ravel() for m in mesh])
        
        # Subsample if too many points
        if len(positions) > n_samples:
            indices = np.random.choice(len(positions), n_samples, replace=False)
            positions = positions[indices]
        
        return positions
    
    def calculate_ruggedness(self):
        """Calculate landscape ruggedness using autocorrelation
        
        ENHANCED: Detects extreme sensitivity (small changes → massive fitness shifts)
        Memory optimized with chunked processing
        
        Returns
        -------
        dict
            Ruggedness metrics including sensitivity analysis
        """
        logging.info("Calculating ruggedness and sensitivity...")
        
        # Use chunked distance calculation to save memory
        n_samples = len(self.samples)
        chunk_size = min(1000, n_samples)
        
        # Sample subset for sensitivity analysis (don't need all pairs)
        sample_indices = np.random.choice(n_samples, min(500, n_samples), replace=False)
        
        # Calculate distances and fitness differences for sample
        sample_positions = self.samples[sample_indices]
        sample_fitness = self.fitness_values[sample_indices]
        
        distances = squareform(pdist(sample_positions))
        fitness_diff = np.abs(sample_fitness[:, None] - sample_fitness[None, :])
        
        # CRITICAL: Detect extreme sensitivity (small distance → huge fitness change)
        very_small_distances = distances < np.percentile(distances[distances > 0], 5)
        if np.any(very_small_distances):
            small_dist_fitness_changes = fitness_diff[very_small_distances]
            extreme_sensitivity_ratio = np.sum(small_dist_fitness_changes > np.percentile(fitness_diff, 75)) / len(small_dist_fitness_changes)
            max_sensitivity = np.max(small_dist_fitness_changes / (distances[very_small_distances] + 1e-10))
        else:
            extreme_sensitivity_ratio = 0
            max_sensitivity = 0
        
        # Autocorrelation at different distance scales
        distance_bins = np.percentile(distances[distances > 0], [1, 5, 10, 25, 50, 75, 90])
        autocorr = []
        fitness_volatility = []
        
        for d in distance_bins:
            mask = (distances > 0) & (distances < d)
            if np.any(mask):
                corr = spearmanr(distances[mask], fitness_diff[mask])[0]
                autocorr.append(corr)
                volatility = np.std(fitness_diff[mask]) / (np.mean(fitness_diff[mask]) + 1e-10)
                fitness_volatility.append(volatility)
        
        # Ruggedness score
        ruggedness = 1 - np.mean(autocorr) if autocorr else 0
        
        # Detect "cliff" behavior
        cliff_score = np.sum(fitness_diff > 10 * np.median(fitness_diff)) / len(fitness_diff.flatten())
        
        return {
            'ruggedness_score': ruggedness,
            'autocorrelation': autocorr,
            'distance_scales': distance_bins.tolist(),
            'fitness_volatility': fitness_volatility,
            'extreme_sensitivity_ratio': float(extreme_sensitivity_ratio),
            'max_sensitivity': float(max_sensitivity),
            'cliff_score': float(cliff_score),
            'interpretation': 'extreme' if extreme_sensitivity_ratio > 0.3 else 'high' if ruggedness > 0.7 else 'medium' if ruggedness > 0.4 else 'low',
            'warning': 'EXTREME SENSITIVITY DETECTED!' if extreme_sensitivity_ratio > 0.3 else None
        }
    
    def detect_modality(self, eps_percentile=5):
        """Detect number of basins/modes using clustering
        
        Parameters
        ----------
        eps_percentile : float
            Percentile of distances to use as DBSCAN epsilon
        
        Returns
        -------
        dict
            Modality information
        """
        logging.info("Detecting modality...")
        
        # Use PCA for high-dimensional data
        if self.dimensions > 10:
            pca = PCA(n_components=min(10, self.dimensions))
            samples_reduced = pca.fit_transform(self.samples)
            logging.info(f"Reduced to {samples_reduced.shape[1]}D (variance explained: {pca.explained_variance_ratio_.sum():.2%})")
        else:
            samples_reduced = self.samples
        
        # Find local optima (samples better than neighbors)
        distances = squareform(pdist(samples_reduced))
        eps = np.percentile(distances[distances > 0], eps_percentile)
        
        local_optima_mask = np.zeros(len(self.fitness_values), dtype=bool)
        for i in range(len(self.fitness_values)):
            neighbors = distances[i] < eps
            neighbors[i] = False
            if np.any(neighbors):
                if self.fitness_values[i] < np.min(self.fitness_values[neighbors]):
                    local_optima_mask[i] = True
        
        n_local_optima = np.sum(local_optima_mask)
        
        # Cluster local optima to find basins
        if n_local_optima > 1:
            local_optima_samples = samples_reduced[local_optima_mask]
            clustering = DBSCAN(eps=eps*3, min_samples=1).fit(local_optima_samples)
            n_basins = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        else:
            n_basins = n_local_optima
        
        return {
            'n_local_optima': int(n_local_optima),
            'n_basins': int(n_basins),
            'local_optima_density': n_local_optima / len(self.fitness_values),
            'interpretation': 'multimodal' if n_basins > 3 else 'bimodal' if n_basins == 2 else 'unimodal'
        }
    
    def analyze_gradients(self, n_neighbors=10):
        """Analyze gradient information (memory optimized)
        
        Parameters
        ----------
        n_neighbors : int
            Number of neighbors to use for gradient estimation
        
        Returns
        -------
        dict
            Gradient statistics
        """
        logging.info("Analyzing gradients...")
        
        # Sample subset to save memory
        n_samples = len(self.samples)
        sample_size = min(500, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        
        sample_positions = self.samples[sample_indices]
        sample_fitness = self.fitness_values[sample_indices]
        
        distances = squareform(pdist(sample_positions))
        gradients = []
        
        for i in range(len(sample_positions)):
            # Find k nearest neighbors
            neighbor_indices = np.argsort(distances[i])[1:n_neighbors+1]
            
            # Estimate gradient magnitude
            for j in neighbor_indices:
                dist = distances[i, j]
                if dist > 0:
                    fitness_diff = abs(sample_fitness[j] - sample_fitness[i])
                    gradient = fitness_diff / dist
                    gradients.append(gradient)
        
        gradients = np.array(gradients)
        
        return {
            'mean_gradient': float(np.mean(gradients)),
            'median_gradient': float(np.median(gradients)),
            'std_gradient': float(np.std(gradients)),
            'max_gradient': float(np.max(gradients)),
            'gradient_variability': float(np.std(gradients) / (np.mean(gradients) + 1e-10)),
            'interpretation': 'steep' if np.median(gradients) > np.percentile(gradients, 75) else 'moderate'
        }
    
    def detect_neutrality(self, tolerance=1e-3):
        """Detect flat/neutral regions (memory optimized)
        
        Parameters
        ----------
        tolerance : float
            Fitness difference threshold for neutrality
        
        Returns
        -------
        dict
            Neutrality information
        """
        logging.info("Detecting neutrality...")
        
        # Sample subset to save memory
        n_samples = len(self.samples)
        sample_size = min(500, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        
        sample_positions = self.samples[sample_indices]
        sample_fitness = self.fitness_values[sample_indices]
        
        distances = squareform(pdist(sample_positions))
        fitness_diff = np.abs(sample_fitness[:, None] - sample_fitness[None, :])
        
        # Find pairs with small fitness difference despite distance
        neutral_mask = (distances > 0) & (fitness_diff < tolerance)
        neutrality_ratio = np.sum(neutral_mask) / np.sum(distances > 0)
        
        # Find neutral networks (connected components of neutral points)
        threshold = np.percentile(self.fitness_values, 50)
        near_threshold = np.abs(self.fitness_values - threshold) < tolerance
        n_neutral_points = np.sum(near_threshold)
        
        return {
            'neutrality_ratio': float(neutrality_ratio),
            'n_neutral_points': int(n_neutral_points),
            'neutral_percentage': float(n_neutral_points / len(self.fitness_values) * 100),
            'interpretation': 'high' if neutrality_ratio > 0.3 else 'medium' if neutrality_ratio > 0.1 else 'low'
        }
    
    def assess_sampling_adequacy(self):
        """Assess if we have enough samples to characterize the landscape
        
        CRITICAL: Determines if analysis is reliable given the huge search space
        
        Returns
        -------
        dict
            Adequacy metrics and warnings
        """
        logging.info("Assessing sampling adequacy...")
        
        # Calculate theoretical minimum samples needed
        # Rule of thumb: need ~10^d samples for d dimensions (curse of dimensionality)
        theoretical_min = 10 ** self.dimensions
        
        # Calculate actual coverage
        bounds = np.array(list(self.param_bounds.values()))
        param_ranges = bounds[:, 1] - bounds[:, 0]
        total_volume = np.prod(param_ranges)
        
        # Estimate sampled volume using convex hull (for low dimensions)
        if self.dimensions <= 10:
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(self.samples)
                sampled_volume = hull.volume
                coverage_ratio = sampled_volume / total_volume
            except:
                coverage_ratio = 0
        else:
            # For high dimensions, estimate using sample spread
            sample_ranges = np.max(self.samples, axis=0) - np.min(self.samples, axis=0)
            coverage_ratio = np.prod(sample_ranges / param_ranges)
        
        # Calculate sample density
        sample_density = len(self.samples) / total_volume
        
        # Assess local coverage (are samples well-distributed?)
        distances = squareform(pdist(self.samples))
        nearest_neighbor_dists = np.min(distances + np.eye(len(distances)) * 1e10, axis=1)
        coverage_uniformity = np.std(nearest_neighbor_dists) / (np.mean(nearest_neighbor_dists) + 1e-10)
        
        # Determine adequacy level
        if len(self.samples) < 100:
            adequacy = 'very_low'
            confidence = 0.1
            warning = 'CRITICAL: Far too few samples for reliable analysis'
        elif len(self.samples) < 500:
            adequacy = 'low'
            confidence = 0.3
            warning = 'WARNING: Limited samples, results may be unreliable'
        elif len(self.samples) < 2000:
            adequacy = 'moderate'
            confidence = 0.6
            warning = 'CAUTION: Moderate sampling, consider increasing for high-dimensional spaces'
        elif len(self.samples) < 5000:
            adequacy = 'good'
            confidence = 0.8
            warning = None
        else:
            adequacy = 'excellent'
            confidence = 0.95
            warning = None
        
        # Adjust confidence based on dimensionality
        dimensionality_penalty = min(1.0, 10 / self.dimensions)
        adjusted_confidence = confidence * dimensionality_penalty
        
        # Specific warnings for your case
        warnings = []
        if warning:
            warnings.append(warning)
        
        if self.dimensions > 20:
            warnings.append(f'HIGH DIMENSIONALITY ({self.dimensions}D): Curse of dimensionality severely limits coverage')
            warnings.append(f'Theoretical minimum samples: {theoretical_min:.2e} (you have {len(self.samples)})')
        
        if coverage_ratio < 0.01:
            warnings.append(f'LOW COVERAGE: Only {coverage_ratio*100:.2f}% of search space sampled')
        
        if coverage_uniformity > 2.0:
            warnings.append('UNEVEN SAMPLING: Samples not uniformly distributed')
        
        return {
            'adequacy_level': adequacy,
            'confidence_score': float(adjusted_confidence),
            'n_samples': len(self.samples),
            'dimensions': self.dimensions,
            'theoretical_min_samples': float(theoretical_min),
            'coverage_ratio': float(coverage_ratio),
            'sample_density': float(sample_density),
            'coverage_uniformity': float(coverage_uniformity),
            'warnings': warnings,
            'recommendation': self._get_sampling_recommendation(adequacy, self.dimensions, len(self.samples))
        }
    
    def _get_sampling_recommendation(self, adequacy, dimensions, current_samples):
        """Get recommendation for improving sampling"""
        if adequacy in ['very_low', 'low']:
            recommended = max(1000, current_samples * 3)
            return f'Increase to at least {recommended} samples for more reliable results'
        elif adequacy == 'moderate' and dimensions > 10:
            recommended = max(5000, current_samples * 2)
            return f'For {dimensions}D space, recommend {recommended}+ samples'
        elif adequacy == 'good' and dimensions > 20:
            return f'Consider adaptive sampling to focus on interesting regions'
        else:
            return 'Sampling appears adequate for current analysis'
        """Detect flat/neutral regions
        
        Parameters
        ----------
        tolerance : float
            Fitness difference threshold for neutrality
        
        Returns
        -------
        dict
            Neutrality information
        """
        logging.info("Detecting neutrality...")
        
        distances = squareform(pdist(self.samples))
        fitness_diff = np.abs(self.fitness_values[:, None] - self.fitness_values[None, :])
        
        # Find pairs with small fitness difference despite distance
        neutral_mask = (distances > 0) & (fitness_diff < tolerance)
        neutrality_ratio = np.sum(neutral_mask) / np.sum(distances > 0)
        
        # Find neutral networks (connected components of neutral points)
        threshold = np.percentile(self.fitness_values, 50)
        near_threshold = np.abs(self.fitness_values - threshold) < tolerance
        n_neutral_points = np.sum(near_threshold)
        
        return {
            'neutrality_ratio': float(neutrality_ratio),
            'n_neutral_points': int(n_neutral_points),
            'neutral_percentage': float(n_neutral_points / len(self.fitness_values) * 100),
            'interpretation': 'high' if neutrality_ratio > 0.3 else 'medium' if neutrality_ratio > 0.1 else 'low'
        }
    
    def analyze_parameter_sensitivity(self):
        """Analyze which parameters have most impact on fitness
        
        ENHANCED: Detects local vs global sensitivity and interaction effects
        
        Returns
        -------
        dict
            Parameter sensitivity rankings with local/global analysis
        """
        logging.info("Analyzing parameter sensitivity...")
        
        param_names = list(self.param_bounds.keys())
        sensitivities = []
        local_sensitivities = []
        
        for dim in range(self.dimensions):
            # Global sensitivity: overall correlation
            global_corr = np.abs(spearmanr(self.samples[:, dim], self.fitness_values)[0])
            sensitivities.append(global_corr)
            
            # Local sensitivity: fitness change per unit parameter change
            # Look at nearest neighbors
            distances = squareform(pdist(self.samples))
            local_sens_values = []
            
            for i in range(min(100, len(self.samples))):  # Sample subset for speed
                # Find 10 nearest neighbors
                neighbor_indices = np.argsort(distances[i])[1:11]
                
                for j in neighbor_indices:
                    param_change = abs(self.samples[j, dim] - self.samples[i, dim])
                    if param_change > 1e-10:
                        fitness_change = abs(self.fitness_values[j] - self.fitness_values[i])
                        local_sens = fitness_change / param_change
                        local_sens_values.append(local_sens)
            
            local_sensitivity = np.median(local_sens_values) if local_sens_values else 0
            local_sensitivities.append(local_sensitivity)
        
        # Rank parameters by sensitivity
        sensitivity_ranking = sorted(
            zip(param_names, sensitivities, local_sensitivities),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Detect parameters with high local but low global sensitivity
        # (these cause the "small change → massive shift" problem!)
        dangerous_params = []
        for name, global_sens, local_sens in sensitivity_ranking:
            if local_sens > np.percentile(local_sensitivities, 75) and global_sens < np.percentile(sensitivities, 50):
                dangerous_params.append(name)
        
        return {
            'sensitivities': {name: float(sens) for name, sens, _ in sensitivity_ranking},
            'local_sensitivities': {name: float(local_sens) for name, _, local_sens in sensitivity_ranking},
            'most_sensitive': sensitivity_ranking[0][0],
            'least_sensitive': sensitivity_ranking[-1][0],
            'sensitivity_range': float(sensitivity_ranking[0][1] - sensitivity_ranking[-1][1]),
            'dangerous_parameters': dangerous_params,
            'warning': f'Parameters with extreme local sensitivity: {dangerous_params}' if dangerous_params else None
        }
    
    def generate_report(self):
        """Generate comprehensive landscape analysis report
        
        ENHANCED: Includes adequacy assessment and extreme sensitivity warnings
        
        Returns
        -------
        dict
            Complete landscape analysis with confidence metrics
        """
        logging.info("Generating landscape report...")
        
        # First assess sampling adequacy
        adequacy = self.assess_sampling_adequacy()
        
        report = {
            'dimensions': self.dimensions,
            'n_samples': len(self.samples),
            'sampling_adequacy': adequacy,
            'fitness_statistics': {
                'min': float(np.min(self.fitness_values)),
                'max': float(np.max(self.fitness_values)),
                'mean': float(np.mean(self.fitness_values)),
                'median': float(np.median(self.fitness_values)),
                'std': float(np.std(self.fitness_values)),
                'q25': float(np.percentile(self.fitness_values, 25)),
                'q75': float(np.percentile(self.fitness_values, 75)),
                'range': float(np.max(self.fitness_values) - np.min(self.fitness_values)),
                'coefficient_of_variation': float(np.std(self.fitness_values) / (np.abs(np.mean(self.fitness_values)) + 1e-10))
            },
            'ruggedness': self.calculate_ruggedness(),
            'modality': self.detect_modality(),
            'gradients': self.analyze_gradients(),
            'neutrality': self.detect_neutrality(),
            'parameter_sensitivity': self.analyze_parameter_sensitivity()
        }
        
        # Add critical warnings
        report['critical_warnings'] = []
        
        if adequacy['confidence_score'] < 0.5:
            report['critical_warnings'].append({
                'severity': 'HIGH',
                'message': f"Low confidence ({adequacy['confidence_score']:.1%}) - results may be unreliable",
                'action': adequacy['recommendation']
            })
        
        if report['ruggedness'].get('warning'):
            report['critical_warnings'].append({
                'severity': 'HIGH',
                'message': report['ruggedness']['warning'],
                'action': 'Use very small step sizes and dense sampling near good solutions'
            })
        
        if report['parameter_sensitivity'].get('warning'):
            report['critical_warnings'].append({
                'severity': 'MEDIUM',
                'message': report['parameter_sensitivity']['warning'],
                'action': 'Use parameter-specific step sizes and careful bounds'
            })
        
        # Add recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report):
        """Generate optimization strategy recommendations based on landscape
        
        Parameters
        ----------
        report : dict
            Landscape analysis report
        
        Returns
        -------
        list
            Strategy recommendations
        """
        recommendations = []
        
        # Ruggedness recommendations
        if report['ruggedness']['interpretation'] == 'high':
            recommendations.append({
                'aspect': 'ruggedness',
                'finding': 'Highly rugged landscape with many local optima',
                'strategies': [
                    'Use multi-start optimization with diverse initial positions',
                    'Increase exploration (higher PSO inertia weight)',
                    'Consider simulated annealing or genetic algorithms',
                    'Use hill climbers to escape local optima'
                ]
            })
        
        # Modality recommendations
        if report['modality']['interpretation'] == 'multimodal':
            recommendations.append({
                'aspect': 'modality',
                'finding': f"Multiple basins detected ({report['modality']['n_basins']} basins)",
                'strategies': [
                    'Use island models or multiple swarms',
                    'Implement niching to maintain diversity',
                    'Consider basin-hopping strategies',
                    'Use farthest-point sampling for exploration'
                ]
            })
        
        # Gradient recommendations
        if report['gradients']['gradient_variability'] > 2.0:
            recommendations.append({
                'aspect': 'gradients',
                'finding': 'High gradient variability (mix of steep and flat regions)',
                'strategies': [
                    'Use adaptive step sizes',
                    'Implement momentum-based methods',
                    'Consider trust-region approaches',
                    'Use gradient-free methods in flat regions'
                ]
            })
        
        # Neutrality recommendations
        if report['neutrality']['interpretation'] == 'high':
            recommendations.append({
                'aspect': 'neutrality',
                'finding': 'Significant neutral/flat regions detected',
                'strategies': [
                    'Use larger step sizes to traverse plateaus',
                    'Implement plateau detection and escape mechanisms',
                    'Consider random restarts when stuck',
                    'Use mutation operators to escape neutrality'
                ]
            })
        
        # Parameter sensitivity recommendations
        sens = report['parameter_sensitivity']
        if sens['sensitivity_range'] > 0.5:
            recommendations.append({
                'aspect': 'parameter_sensitivity',
                'finding': f"High sensitivity variation (most: {sens['most_sensitive']}, least: {sens['least_sensitive']})",
                'strategies': [
                    'Use different step sizes per parameter',
                    'Focus search on sensitive parameters',
                    'Consider parameter-specific mutation rates',
                    'Use coordinate descent for sensitive parameters'
                ]
            })
        
        return recommendations
    
    def visualize_landscape_2d(self, param1_idx=0, param2_idx=1, output_path='landscape_2d.png'):
        """Visualize 2D slice of landscape
        
        Parameters
        ----------
        param1_idx : int
            First parameter index
        param2_idx : int
            Second parameter index
        output_path : str
            Output file path
        """
        logging.info(f"Creating 2D visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        scatter = axes[0].scatter(
            self.samples[:, param1_idx],
            self.samples[:, param2_idx],
            c=self.fitness_values,
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        axes[0].set_xlabel(list(self.param_bounds.keys())[param1_idx])
        axes[0].set_ylabel(list(self.param_bounds.keys())[param2_idx])
        axes[0].set_title('Fitness Landscape (2D Slice)')
        plt.colorbar(scatter, ax=axes[0], label='Fitness')
        
        # Contour plot (if enough samples)
        if len(self.samples) > 100:
            from scipy.interpolate import griddata
            
            param_names = list(self.param_bounds.keys())
            bounds = list(self.param_bounds.values())
            
            grid_x = np.linspace(bounds[param1_idx][0], bounds[param1_idx][1], 50)
            grid_y = np.linspace(bounds[param2_idx][0], bounds[param2_idx][1], 50)
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            
            grid_z = griddata(
                self.samples[:, [param1_idx, param2_idx]],
                self.fitness_values,
                (grid_x, grid_y),
                method='cubic',
                fill_value=np.nan
            )
            
            contour = axes[1].contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis')
            axes[1].set_xlabel(param_names[param1_idx])
            axes[1].set_ylabel(param_names[param2_idx])
            axes[1].set_title('Fitness Landscape (Interpolated)')
            plt.colorbar(contour, ax=axes[1], label='Fitness')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved 2D visualization to {output_path}")
        plt.close()
    
    def visualize_pca_projection(self, output_path='landscape_pca.png'):
        """Visualize landscape using PCA projection
        
        Parameters
        ----------
        output_path : str
            Output file path
        """
        logging.info("Creating PCA visualization...")
        
        # Apply PCA
        pca = PCA(n_components=min(3, self.dimensions))
        samples_pca = pca.fit_transform(self.samples)
        
        if samples_pca.shape[1] >= 3:
            # 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(
                samples_pca[:, 0],
                samples_pca[:, 1],
                samples_pca[:, 2],
                c=self.fitness_values,
                cmap='viridis',
                alpha=0.6,
                s=20
            )
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
            ax.set_title('Fitness Landscape (PCA Projection)')
            plt.colorbar(scatter, ax=ax, label='Fitness', shrink=0.5)
        else:
            # 2D plot
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                samples_pca[:, 0],
                samples_pca[:, 1],
                c=self.fitness_values,
                cmap='viridis',
                alpha=0.6,
                s=20
            )
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_title('Fitness Landscape (PCA Projection)')
            plt.colorbar(scatter, ax=ax, label='Fitness')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved PCA visualization to {output_path}")
        plt.close()
    
    def generate_ai_agent_report(self, report):
        """Generate comprehensive report optimized for AI agent consumption (memory optimized)
        
        Parameters
        ----------
        report : dict
            Base analysis report
            
        Returns
        -------
        dict
            Enhanced report with detailed context for AI agents
        """
        # Use sampling to reduce memory usage
        n_samples = len(self.samples)
        sample_size = min(300, n_samples)  # Limit topology analysis to 300 samples
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        
        sample_positions = self.samples[sample_indices]
        sample_fitness = self.fitness_values[sample_indices]
        
        # Calculate distances only for sample
        distances = squareform(pdist(sample_positions))
        fitness_diff = np.abs(sample_fitness[:, None] - sample_fitness[None, :])
        
        # Detailed parameter analysis (use all samples for this)
        param_names = list(self.param_bounds.keys())
        param_analysis = {}
        
        for i, param_name in enumerate(param_names):
            bounds = self.param_bounds[param_name]
            values = self.samples[:, i]
            
            # Calculate parameter-specific metrics
            param_fitness_corr = np.corrcoef(values, self.fitness_values)[0, 1]
            
            # Local sensitivity per parameter (use sample)
            sample_values = values[sample_indices]
            local_sens_values = []
            for j in range(min(50, len(sample_positions))):  # Further reduce
                neighbor_indices = np.argsort(distances[j])[1:6]  # Fewer neighbors
                for k in neighbor_indices:
                    param_change = abs(sample_values[k] - sample_values[j])
                    if param_change > 1e-10:
                        fitness_change = abs(sample_fitness[k] - sample_fitness[j])
                        local_sens_values.append(fitness_change / param_change)
            
            param_analysis[param_name] = {
                'bounds': bounds,
                'sampled_range': [float(np.min(values)), float(np.max(values))],
                'mean_value': float(np.mean(values)),
                'std_value': float(np.std(values)),
                'global_correlation': float(param_fitness_corr),
                'local_sensitivity_median': float(np.median(local_sens_values)) if local_sens_values else 0,
                'local_sensitivity_max': float(np.max(local_sens_values)) if local_sens_values else 0,
                'local_sensitivity_std': float(np.std(local_sens_values)) if local_sens_values else 0,
                'is_dangerous': param_name in report['parameter_sensitivity'].get('dangerous_parameters', []),
                'sensitivity_rank': list(report['parameter_sensitivity']['sensitivities'].keys()).index(param_name) + 1
            }
        
        # Detailed landscape topology (use sample)
        topology = {
            'local_optima_positions': [],
            'high_gradient_regions': [],
            'flat_regions': [],
            'extreme_fitness_regions': []
        }
        
        # Find local optima (limit to top 10)
        local_optima_found = 0
        for i in range(len(sample_positions)):
            if local_optima_found >= 10:
                break
            neighbors = np.argsort(distances[i])[1:6]
            if np.all(sample_fitness[i] < sample_fitness[neighbors]):
                topology['local_optima_positions'].append({
                    'position': sample_positions[i].tolist(),
                    'fitness': float(sample_fitness[i]),
                    'basin_size_estimate': float(np.min(distances[i, neighbors]))
                })
                local_optima_found += 1
        
        # Find high gradient regions (limit to top 10)
        gradients = []
        for i in range(len(sample_positions)):
            neighbors = np.argsort(distances[i])[1:6]
            local_gradient = np.mean(fitness_diff[i, neighbors] / (distances[i, neighbors] + 1e-10))
            gradients.append(local_gradient)
        
        gradients = np.array(gradients)
        high_gradient_threshold = np.percentile(gradients, 90)
        
        high_gradient_found = 0
        for i in range(len(sample_positions)):
            if high_gradient_found >= 10:
                break
            if gradients[i] > high_gradient_threshold:
                topology['high_gradient_regions'].append({
                    'position': sample_positions[i].tolist(),
                    'fitness': float(sample_fitness[i]),
                    'gradient_magnitude': float(gradients[i])
                })
                high_gradient_found += 1
        
        # Find flat regions (limit to top 10)
        flat_threshold = np.percentile(gradients, 10)
        flat_found = 0
        for i in range(len(sample_positions)):
            if flat_found >= 10:
                break
            if gradients[i] < flat_threshold:
                topology['flat_regions'].append({
                    'position': sample_positions[i].tolist(),
                    'fitness': float(sample_fitness[i]),
                    'gradient_magnitude': float(gradients[i])
                })
                flat_found += 1
        
        # Find extreme fitness regions (use full dataset, limit to 10 each)
        top_10_indices = np.argsort(self.fitness_values)[:10]
        bottom_10_indices = np.argsort(self.fitness_values)[-10:]
        
        for idx in top_10_indices:
            topology['extreme_fitness_regions'].append({
                'type': 'best',
                'position': self.samples[idx].tolist(),
                'fitness': float(self.fitness_values[idx]),
                'local_gradient': 0.0  # Not available for full dataset
            })
        
        for idx in bottom_10_indices:
            topology['extreme_fitness_regions'].append({
                'type': 'worst',
                'position': self.samples[idx].tolist(),
                'fitness': float(self.fitness_values[idx]),
                'local_gradient': 0.0  # Not available for full dataset
            })
        
        # Optimization strategy recommendations with detailed reasoning
        strategy_recommendations = self._generate_detailed_recommendations(report, param_analysis, topology)
        
        # Construct comprehensive AI agent report
        ai_report = {
            'metadata': {
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'dimensions': self.dimensions,
                'n_samples': len(self.samples),
                'sampling_method': getattr(self, 'sampling_method', 'unknown'),
                'parameter_names': param_names
            },
            
            'sampling_quality': {
                'adequacy': report['sampling_adequacy'],
                'confidence_score': report['sampling_adequacy']['confidence_score'],
                'coverage_ratio': report['sampling_adequacy']['coverage_ratio'],
                'reliability_assessment': self._assess_reliability(report['sampling_adequacy'])
            },
            
            'fitness_landscape': {
                'statistics': report['fitness_statistics'],
                'ruggedness': report['ruggedness'],
                'modality': report['modality'],
                'gradients': report['gradients'],
                'neutrality': report['neutrality']
            },
            
            'parameter_analysis': param_analysis,
            
            'topology': topology,
            
            'critical_findings': {
                'has_extreme_sensitivity': report['ruggedness'].get('extreme_sensitivity_ratio', 0) > 0.2,
                'extreme_sensitivity_details': {
                    'ratio': report['ruggedness'].get('extreme_sensitivity_ratio', 0),
                    'max_sensitivity': report['ruggedness'].get('max_sensitivity', 0),
                    'cliff_score': report['ruggedness'].get('cliff_score', 0)
                },
                'dangerous_parameters': report['parameter_sensitivity'].get('dangerous_parameters', []),
                'is_multimodal': report['modality']['n_basins'] > 2,
                'has_high_neutrality': report['neutrality']['interpretation'] == 'high',
                'warnings': report.get('critical_warnings', [])
            },
            
            'optimization_strategies': strategy_recommendations,
            
            'traversal_guidance': self._generate_traversal_guidance(report, param_analysis, topology),
            
            'raw_data_references': {
                'samples_file': 'samples.npy',
                'fitness_values_file': 'fitness_values.npy',
                'full_report_file': 'landscape_report.json'
            }
        }
        
        return ai_report
    
    def _assess_reliability(self, adequacy):
        """Assess reliability of different analysis aspects"""
        confidence = adequacy['confidence_score']
        
        return {
            'overall_confidence': confidence,
            'reliable_aspects': [
                'parameter_sensitivity',
                'extreme_sensitivity_detection',
                'local_gradient_estimation'
            ] if confidence > 0.3 else [],
            'moderate_confidence_aspects': [
                'ruggedness_estimation',
                'local_optima_count',
                'gradient_variability'
            ] if confidence > 0.5 else [],
            'low_confidence_aspects': [
                'global_basin_count',
                'complete_modality',
                'full_space_coverage'
            ],
            'trust_level': 'high' if confidence > 0.8 else 'medium' if confidence > 0.5 else 'low'
        }
    
    def _generate_detailed_recommendations(self, report, param_analysis, topology):
        """Generate detailed strategy recommendations with reasoning"""
        recommendations = []
        
        # Extreme sensitivity handling
        if report['ruggedness'].get('extreme_sensitivity_ratio', 0) > 0.2:
            dangerous_params = report['parameter_sensitivity'].get('dangerous_parameters', [])
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'extreme_sensitivity',
                'finding': f"Extreme sensitivity detected: {report['ruggedness']['extreme_sensitivity_ratio']:.1%} of nearby points show massive fitness changes",
                'reasoning': "Small parameter changes cause disproportionate fitness shifts, making optimization highly unstable",
                'strategies': [
                    {
                        'name': 'micro_step_optimization',
                        'description': 'Use very small step sizes (0.1-1% of parameter range)',
                        'implementation': 'Set PSO velocity_clamp to 0.01 * (upper_bound - lower_bound)',
                        'parameters': {
                            'velocity_scale': 0.01,
                            'apply_to': dangerous_params if dangerous_params else 'all'
                        }
                    },
                    {
                        'name': 'parameter_specific_bounds',
                        'description': 'Tighten bounds on dangerous parameters',
                        'implementation': 'Reduce search range for high-sensitivity parameters by 50-75%',
                        'parameters': {
                            'dangerous_parameters': dangerous_params,
                            'bound_reduction': 0.5
                        }
                    },
                    {
                        'name': 'gradient_free_methods',
                        'description': 'Use gradient-free optimization (PSO, genetic algorithms)',
                        'implementation': 'Avoid gradient-based methods that assume smoothness',
                        'parameters': {
                            'recommended_methods': ['PSO', 'genetic_algorithm', 'simulated_annealing']
                        }
                    },
                    {
                        'name': 'ensemble_evaluation',
                        'description': 'Evaluate multiple nearby points and average',
                        'implementation': 'Sample 5-10 points in small neighborhood and use median fitness',
                        'parameters': {
                            'n_samples': 5,
                            'neighborhood_radius': 0.001
                        }
                    }
                ],
                'expected_impact': 'Reduces catastrophic fitness jumps, improves optimization stability'
            })
        
        # Multimodal handling
        if report['modality']['n_basins'] > 2:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'multimodality',
                'finding': f"Multimodal landscape with {report['modality']['n_basins']} basins and {report['modality']['n_local_optima']} local optima",
                'reasoning': "Multiple distinct regions require diverse exploration to avoid premature convergence",
                'strategies': [
                    {
                        'name': 'multi_start_optimization',
                        'description': 'Run multiple independent optimizations from diverse starting points',
                        'implementation': 'Initialize 5-10 swarms with LHS or Sobol sampling',
                        'parameters': {
                            'n_starts': max(5, report['modality']['n_basins']),
                            'initialization': 'lhs'
                        }
                    },
                    {
                        'name': 'farthest_point_sampling',
                        'description': 'Use distance-maximization for exploration',
                        'implementation': 'Enable farthest-point candidate placement in PSO',
                        'parameters': {
                            'fp_candidate_pool_size': 100000,
                            'fp_discovered_cap': 5000
                        }
                    },
                    {
                        'name': 'basin_hopping',
                        'description': 'Periodically jump to unexplored regions',
                        'implementation': 'Boost worst particles to distant locations every N iterations',
                        'parameters': {
                            'boost_interval': 10,
                            'boost_fraction': 0.3
                        }
                    },
                    {
                        'name': 'niching',
                        'description': 'Maintain diversity through competition',
                        'implementation': 'Enable grid-based competitive evolution',
                        'parameters': {
                            'max_particles_per_cell': 2,
                            'competition_enabled': True
                        }
                    }
                ],
                'expected_impact': 'Explores multiple basins, reduces risk of missing global optimum'
            })
        
        # High neutrality handling
        if report['neutrality']['interpretation'] == 'high':
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'neutrality',
                'finding': f"High neutrality: {report['neutrality']['neutral_percentage']:.1f}% of space is flat",
                'reasoning': "Large plateau regions require special traversal strategies to avoid stagnation",
                'strategies': [
                    {
                        'name': 'large_step_traversal',
                        'description': 'Use larger steps to cross plateaus quickly',
                        'implementation': 'Increase velocity boost magnitude in flat regions',
                        'parameters': {
                            'velocity_scale': 2.0,
                            'detect_plateau_threshold': 1e-6
                        }
                    },
                    {
                        'name': 'momentum_methods',
                        'description': 'Use momentum to maintain direction across plateaus',
                        'implementation': 'Increase PSO inertia weight (w) to 0.9-0.95',
                        'parameters': {
                            'inertia_weight': 0.9,
                            'adaptive': False
                        }
                    },
                    {
                        'name': 'plateau_escape',
                        'description': 'Detect stagnation and trigger random jumps',
                        'implementation': 'Monitor fitness changes and restart when stuck',
                        'parameters': {
                            'stagnation_threshold': 10,
                            'restart_fraction': 0.25
                        }
                    }
                ],
                'expected_impact': 'Reduces time spent on plateaus, improves convergence speed'
            })
        
        # Parameter-specific recommendations
        sensitive_params = sorted(
            param_analysis.items(),
            key=lambda x: x[1]['local_sensitivity_median'],
            reverse=True
        )[:5]
        
        if sensitive_params:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'parameter_sensitivity',
                'finding': f"High sensitivity variation: top parameter is {sensitive_params[0][1]['local_sensitivity_median']/sensitive_params[-1][1]['local_sensitivity_median']:.1f}x more sensitive",
                'reasoning': "Different parameters require different search strategies",
                'strategies': [
                    {
                        'name': 'parameter_specific_step_sizes',
                        'description': 'Use smaller steps for sensitive parameters',
                        'implementation': 'Scale velocity by inverse sensitivity',
                        'parameters': {
                            'sensitive_parameters': {
                                name: {
                                    'step_scale': 0.1,
                                    'sensitivity': data['local_sensitivity_median']
                                }
                                for name, data in sensitive_params
                            }
                        }
                    },
                    {
                        'name': 'coordinate_descent',
                        'description': 'Optimize sensitive parameters separately',
                        'implementation': 'Alternate between sensitive and insensitive parameter updates',
                        'parameters': {
                            'sensitive_params': [name for name, _ in sensitive_params],
                            'update_frequency': 'alternating'
                        }
                    },
                    {
                        'name': 'focused_search',
                        'description': 'Concentrate search on sensitive parameter subspace',
                        'implementation': 'Fix insensitive parameters, optimize sensitive ones',
                        'parameters': {
                            'focus_on': [name for name, _ in sensitive_params[:3]]
                        }
                    }
                ],
                'expected_impact': 'More efficient search, faster convergence on critical parameters'
            })
        
        return recommendations
    
    def _generate_traversal_guidance(self, report, param_analysis, topology):
        """Generate specific traversal guidance for optimization algorithms"""
        return {
            'initialization': {
                'recommended_method': 'lhs' if report['modality']['n_basins'] > 2 else 'sobol',
                'n_initial_points': max(100, report['modality']['n_basins'] * 20),
                'focus_regions': [
                    {
                        'center': region['position'],
                        'radius': 0.05,
                        'reason': 'local_optimum'
                    }
                    for region in topology['local_optima_positions'][:5]
                ]
            },
            
            'step_size_guidance': {
                'global_step_size': 0.01 if report['ruggedness'].get('extreme_sensitivity_ratio', 0) > 0.2 else 0.05,
                'parameter_specific': {
                    name: {
                        'recommended_step': 0.001 if data['is_dangerous'] else 0.01,
                        'reason': 'extreme_local_sensitivity' if data['is_dangerous'] else 'normal'
                    }
                    for name, data in param_analysis.items()
                }
            },
            
            'exploration_vs_exploitation': {
                'recommended_balance': 'exploration' if report['modality']['n_basins'] > 3 else 'balanced',
                'pso_parameters': {
                    'w': 0.9 if report['modality']['n_basins'] > 3 else 0.7,
                    'c1': 0.5,
                    'c2': 0.3
                },
                'boost_frequency': 5 if report['neutrality']['interpretation'] == 'high' else 10
            },
            
            'convergence_criteria': {
                'fitness_tolerance': 1e-6 if report['ruggedness'].get('extreme_sensitivity_ratio', 0) > 0.2 else 1e-4,
                'stagnation_threshold': 20 if report['neutrality']['interpretation'] == 'high' else 10,
                'max_iterations': 10000
            },
            
            'regions_to_avoid': [
                {
                    'center': region['position'],
                    'radius': 0.02,
                    'reason': 'high_gradient_instability'
                }
                for region in topology['high_gradient_regions']
                if region['gradient_magnitude'] > np.percentile([r['gradient_magnitude'] for r in topology['high_gradient_regions']], 95)
            ][:10],
            
            'promising_regions': [
                {
                    'center': region['position'],
                    'fitness': region['fitness'],
                    'search_radius': 0.05,
                    'reason': 'local_optimum'
                }
                for region in topology['local_optima_positions'][:10]
            ]
        }
    
    def save_results(self, report, output_dir='landscape_analysis'):
        """Save analysis results optimized for AI agent consumption (JSON only)
        
        Parameters
        ----------
        report : dict
            Analysis report
        output_dir : str
            Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate AI-optimized report
        ai_report = self.generate_ai_agent_report(report)
        
        # Save AI agent report (primary output - JSON only)
        with open(output_path / 'ai_agent_report.json', 'w') as f:
            json.dump(ai_report, f, indent=2)
        
        logging.info(f"Saved results to {output_path}")
        logging.info(f"AI agent report: {output_path / 'ai_agent_report.json'}")


async def main():
    parser = argparse.ArgumentParser(description='Map fitness landscape characteristics')
    parser.add_argument('config_path', type=str, help='Path to optimization config')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--method', type=str, default='lhs', choices=['lhs', 'sobol', 'random', 'grid'])
    parser.add_argument('--output', type=str, default='landscape_analysis', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--adaptive', action='store_true', default=True, help='Use adaptive sampling')
    parser.add_argument('--no-adaptive', dest='adaptive', action='store_false', help='Disable adaptive sampling')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize evaluator
    logging.info(f"Initializing evaluator from {args.config_path}")
    evaluator, config = await initEvaluator(args.config_path)
    
    # Get parameter bounds
    param_bounds = config['optimize']['bounds']
    for k, v in param_bounds.items():
        if len(v) == 1:
            param_bounds[k] = [v[0], v[0]]
    
    # Create mapper
    mapper = FitnessLandscapeMapper(evaluator, config, param_bounds)
    
    # Sample landscape
    mapper.sample_landscape(n_samples=args.samples, method=args.method, adaptive=args.adaptive)
    
    # Generate report
    report = mapper.generate_report()
    
    # Print summary
    print("\n" + "="*80)
    print("FITNESS LANDSCAPE ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nDimensions: {report['dimensions']}")
    print(f"Samples: {report['n_samples']}")
    
    # Print adequacy assessment
    adequacy = report['sampling_adequacy']
    print(f"\n{'='*80}")
    print(f"SAMPLING ADEQUACY: {adequacy['adequacy_level'].upper()}")
    print(f"Confidence: {adequacy['confidence_score']:.1%}")
    print(f"{'='*80}")
    if adequacy['warnings']:
        for warning in adequacy['warnings']:
            print(f"⚠️  {warning}")
    print(f"\n💡 {adequacy['recommendation']}")
    
    print(f"\n{'='*80}")
    print(f"FITNESS STATISTICS")
    print(f"{'='*80}")
    print(f"Range: [{report['fitness_statistics']['min']:.2e}, {report['fitness_statistics']['max']:.2e}]")
    print(f"Spread: {report['fitness_statistics']['range']:.2e}")
    print(f"Coefficient of Variation: {report['fitness_statistics']['coefficient_of_variation']:.2f}")
    
    print(f"\n{'='*80}")
    print(f"LANDSCAPE CHARACTERISTICS")
    print(f"{'='*80}")
    print(f"Ruggedness: {report['ruggedness']['interpretation']} ({report['ruggedness']['ruggedness_score']:.3f})")
    if report['ruggedness'].get('extreme_sensitivity_ratio', 0) > 0.2:
        print(f"  ⚠️  EXTREME SENSITIVITY: {report['ruggedness']['extreme_sensitivity_ratio']:.1%} of nearby points have massive fitness changes")
        print(f"  ⚠️  Max sensitivity: {report['ruggedness']['max_sensitivity']:.2e}")
    print(f"Modality: {report['modality']['interpretation']} ({report['modality']['n_basins']} basins, {report['modality']['n_local_optima']} local optima)")
    print(f"Neutrality: {report['neutrality']['interpretation']} ({report['neutrality']['neutral_percentage']:.1f}% neutral)")
    print(f"Gradient: {report['gradients']['interpretation']} (median: {report['gradients']['median_gradient']:.2e})")
    
    # Print critical warnings
    if report['critical_warnings']:
        print(f"\n{'='*80}")
        print(f"⚠️  CRITICAL WARNINGS")
        print(f"{'='*80}")
        for warning in report['critical_warnings']:
            print(f"\n[{warning['severity']}] {warning['message']}")
            print(f"  → {warning['action']}")
    
    # Print parameter sensitivity
    print(f"\n{'='*80}")
    print(f"PARAMETER SENSITIVITY")
    print(f"{'='*80}")
    sens = report['parameter_sensitivity']
    print(f"Most sensitive: {sens['most_sensitive']}")
    print(f"Least sensitive: {sens['least_sensitive']}")
    if sens.get('dangerous_parameters'):
        print(f"\n⚠️  DANGEROUS PARAMETERS (high local, low global sensitivity):")
        for param in sens['dangerous_parameters']:
            print(f"  - {param}")
    
    print("\n" + "-"*80)
    print("OPTIMIZATION STRATEGY RECOMMENDATIONS:")
    print("-"*80)
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"\n{i}. {rec['aspect'].upper()}: {rec['finding']}")
        print("   Strategies:")
        for strategy in rec['strategies']:
            print(f"   - {strategy}")
    
    # Save results
    mapper.save_results(report, args.output)
    
    print(f"\n✅ Analysis complete! Results saved to {args.output}/")
    print(f"\n📊 Confidence in results: {adequacy['confidence_score']:.1%}")
    print(f"\n🤖 AI AGENT REPORT: {args.output}/ai_agent_report.json")
    print(f"   This file contains comprehensive landscape analysis optimized for AI consumption")
    if adequacy['confidence_score'] < 0.7:
        print(f"⚠️  Consider running with more samples for higher confidence")


if __name__ == '__main__':
    asyncio.run(main())
