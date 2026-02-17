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
import sys

# Rich imports for progress bar and panels
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.logging import RichHandler

# Import your optimization setup
from optimize import initEvaluator

# Initialize console with file logging
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
    
    def __init__(self, evaluator, config, param_bounds, batch_size=5000, cache_dir='landscape_cache', max_fitness=None):
        self.evaluator = evaluator
        self.config = config
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_fitness = max_fitness
        
        # Setup logging to file
        self.log_file = Path('logs/mapper.log')
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Create file console for logging
        self.file_console = Console(
            file=open(self.log_file, 'w', encoding='utf-8'),
            force_terminal=False,
            width=120
        )
        
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
        
        # Storage for samples - now batched
        self.samples = []
        self.fitness_values = []
        self.total_samples = 0
        self.batch_files = []
        
        # Log initialization
        init_msg = f"Initialized landscape mapper for {self.dimensions}D problem ({len(self.fixed_params)} fixed parameters)"
        if self.max_fitness is not None:
            init_msg += f" - FILTERED MODE (fitness <= {self.max_fitness})"
        logging.info(init_msg)
        self._log_to_file(init_msg)
        
        batch_msg = f"Using batch size: {batch_size}, cache directory: {cache_dir}"
        logging.info(batch_msg)
        self._log_to_file(batch_msg)
    
    def _log_to_file(self, message, level="INFO"):
        """Log message to file with timestamp"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        self.file_console.print(f"[{timestamp}] [{level}] {message}")
    
    def _create_batch_panel(self, batch_idx, total_batches, batch_size, best_fitness):
        """Create a rich panel for batch progress"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="bold cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Batch", f"{batch_idx + 1}/{total_batches}")
        table.add_row("Samples", f"{batch_size}")
        table.add_row("Best Fitness", f"{best_fitness:.2e}")
        table.add_row("Total Processed", f"{self.total_samples + batch_size}")
        
        return Panel(
            table,
            title=f"🗺️ Landscape Mapping Progress",
            border_style="blue",
            width=50
        )
    
    def sample_landscape(self, n_samples=1000, method='lhs', adaptive=True, focus_regions=None):
        """Sample the fitness landscape with adaptive refinement and multiprocessing
        
        ENHANCED: Adaptive sampling focuses on interesting regions with parallel evaluation
        Now supports batched processing to handle large sample counts without memory issues
        
        Parameters
        ----------
        n_samples : int
            Number of samples to take (if max_fitness is set, this is the target number of filtered samples)
        method : str
            Sampling method: 'lhs', 'random', 'grid', 'sobol'
        adaptive : bool
            Use adaptive sampling to focus on high-gradient regions (disabled in filtered mode)
        focus_regions : list of tuples, optional
            Specific regions to focus sampling (list of (center, radius) tuples)
        """
        if self.max_fitness is not None:
            start_msg = f"Sampling landscape for {n_samples} FILTERED samples (fitness <= {self.max_fitness}) using {method} (filtered mode)"
            logging.info(start_msg)
            self._log_to_file(start_msg)
            
            # Clear any existing cache
            self._clear_cache()
            
            # Single phase: keep generating until we have enough filtered samples
            self._sample_filtered_only(n_samples, method)
        else:
            start_msg = f"Sampling landscape with {n_samples} points using {method} (batched processing)"
            logging.info(start_msg)
            self._log_to_file(start_msg)
            
            # Clear any existing cache
            self._clear_cache()
            
            if adaptive and n_samples >= 500:
                # Phase 1: Initial broad sampling (30% of budget)
                n_initial = int(n_samples * 0.3)
                phase1_msg = f"Phase 1: Initial broad sampling ({n_initial} samples)"
                logging.info(phase1_msg)
                self._log_to_file(phase1_msg)
                self._sample_and_save_batch(n_initial, method, "Phase 1: Broad Sampling")
                
                # Phase 2: Identify interesting regions from saved batches
                phase2_msg = "Phase 2: Identifying interesting regions..."
                logging.info(phase2_msg)
                self._log_to_file(phase2_msg)
                interesting_regions = self._identify_interesting_regions_from_cache()
                
                # Phase 3: Focused sampling (70% of budget)
                n_focused = n_samples - n_initial
                phase3_msg = f"Phase 3: Focused sampling in {len(interesting_regions)} regions ({n_focused} samples)"
                logging.info(phase3_msg)
                self._log_to_file(phase3_msg)
                self._sample_focused_batched(n_focused, interesting_regions, method)
                
            else:
                # Standard batched sampling
                self._sample_and_save_batch(n_samples, method, "Evaluating Landscape")
        
        # Load summary statistics without loading all data
        self._calculate_summary_stats()
        
        complete_msg = f"Sampling complete. {self.total_samples} samples processed in {len(self.batch_files)} batches"
        logging.info(complete_msg)
        self._log_to_file(complete_msg)
        
        # Calculate sampling efficiency from sample of data
        self._calculate_sampling_efficiency_batched()
        
        # Create final summary panel
        self._create_final_summary_panel()
    
    def _sample_filtered_only(self, target_samples, method):
        """Sample until we have target_samples with fitness <= max_fitness
        
        Parameters
        ----------
        target_samples : int
            Target number of filtered fitness samples
        method : str
            Sampling method to use
        """
        filtered_samples_found = 0
        total_evaluations = 0
        batch_counter = 0
        
        while filtered_samples_found < target_samples:
            # Generate a batch of candidate positions
            positions = self._generate_samples(self.batch_size, method)
            
            # Evaluate batch and filter for acceptable fitness
            batch_samples = []
            batch_fitness = []
            
            # Track progress
            description = f"Searching for filtered samples ({filtered_samples_found}/{target_samples} found, {total_evaluations} total evaluations)"
            
            # Evaluate positions
            best_fitness = self._evaluate_positions_parallel(
                positions, 
                description,
                batch_samples,
                batch_fitness
            )
            
            total_evaluations += len(positions)
            
            # Filter for acceptable fitness samples
            filtered_mask = np.array(batch_fitness) <= self.max_fitness
            filtered_samples = np.array(batch_samples)[filtered_mask]
            filtered_fitness = np.array(batch_fitness)[filtered_mask]
            
            if len(filtered_samples) > 0:
                # Save batch with only filtered samples
                batch_file = self.cache_dir / f"batch_{batch_counter:04d}.npz"
                np.savez_compressed(
                    batch_file,
                    samples=filtered_samples,
                    fitness_values=filtered_fitness
                )
                
                self.batch_files.append(batch_file)
                filtered_samples_found += len(filtered_samples)
                self.total_samples += len(filtered_samples)
                
                # Create progress panel
                panel = self._create_filtered_search_panel(
                    filtered_samples_found, target_samples, len(filtered_samples), 
                    total_evaluations, best_fitness, np.min(filtered_fitness)
                )
                console.print(panel)
                
                # Log batch completion
                batch_msg = f"Batch {batch_counter}: Found {len(filtered_samples)} filtered samples (total: {filtered_samples_found}/{target_samples})"
                logging.info(batch_msg)
                self._log_to_file(batch_msg)
                
                batch_counter += 1
            else:
                # No filtered samples found in this batch
                no_filtered_msg = f"Batch evaluation: 0 filtered samples found out of {len(positions)} evaluations"
                logging.info(no_filtered_msg)
                self._log_to_file(no_filtered_msg)
        
        final_msg = f"Filtered sampling complete: {filtered_samples_found} filtered samples found after {total_evaluations} total evaluations"
        logging.info(final_msg)
        self._log_to_file(final_msg)
        
        # Calculate efficiency
        efficiency = (filtered_samples_found / total_evaluations) * 100
        efficiency_msg = f"Filtered sample efficiency: {efficiency:.2f}% ({filtered_samples_found}/{total_evaluations})"
        logging.info(efficiency_msg)
        self._log_to_file(efficiency_msg)
    
    def _create_filtered_search_panel(self, found, target, batch_found, total_evals, best_fitness, best_filtered):
        """Create a rich panel for filtered sample search progress"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="bold cyan")
        table.add_column("Value", style="white")
        
        progress_pct = (found / target) * 100
        efficiency = (found / total_evals) * 100 if total_evals > 0 else 0
        
        table.add_row("Progress", f"{found}/{target} ({progress_pct:.1f}%)")
        table.add_row("This Batch", f"{batch_found} filtered samples")
        table.add_row("Total Evaluations", f"{total_evals}")
        table.add_row("Efficiency", f"{efficiency:.2f}%")
        table.add_row("Best Overall", f"{best_fitness:.2e}")
        table.add_row("Best Filtered", f"{best_filtered:.2e}")
        
        return Panel(
            table,
            title=f"🎯 Filtered Fitness Search (≤ {self.max_fitness})",
            border_style="green" if progress_pct > 50 else "yellow",
            width=50
        )
        """Clear existing cache files"""
        for file in self.cache_dir.glob("batch_*.npz"):
            file.unlink()
        self.batch_files = []
        self.total_samples = 0
    
    def _sample_and_save_batch(self, n_samples, method, description):
        """Sample and save in batches to avoid memory issues"""
        n_batches = max(1, (n_samples + self.batch_size - 1) // self.batch_size)
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_size = end_idx - start_idx
            
            # Generate samples for this batch
            positions = self._generate_samples(batch_size, method)
            
            # Evaluate batch
            batch_samples = []
            batch_fitness = []
            
            best_fitness = self._evaluate_positions_parallel(
                positions, 
                f"{description} (Batch {batch_idx+1}/{n_batches})",
                batch_samples,
                batch_fitness
            )
            
            # Save batch to disk
            batch_file = self.cache_dir / f"batch_{len(self.batch_files):04d}.npz"
            np.savez_compressed(
                batch_file,
                samples=np.array(batch_samples),
                fitness_values=np.array(batch_fitness)
            )
            
            self.batch_files.append(batch_file)
            self.total_samples += len(batch_samples)
            
            # Create and display batch panel
            panel = self._create_batch_panel(batch_idx, n_batches, len(batch_samples), best_fitness)
            console.print(panel)
            
            # Log batch completion
            batch_msg = f"Saved batch {batch_idx+1}/{n_batches} ({len(batch_samples)} samples) to {batch_file}"
            logging.info(batch_msg)
            self._log_to_file(batch_msg)
    
    def _create_final_summary_panel(self):
        """Create final summary panel after all batches"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="bold green")
        table.add_column("Value", style="white")
        
        table.add_row("Total Samples", f"{self.total_samples:,}")
        if self.max_fitness is not None:
            table.add_row("Sample Type", f"Filtered fitness only (≤ {self.max_fitness})")
        table.add_row("Batches Created", f"{len(self.batch_files)}")
        table.add_row("Fitness Range", f"[{self.fitness_stats['min']:.2e}, {self.fitness_stats['max']:.2e}]")
        table.add_row("Cache Directory", str(self.cache_dir))
        table.add_row("Coverage Score", f"{self.sampling_efficiency.get('coverage_score', 0):.4f}")
        
        title = "🎯 Filtered Fitness Mapping Complete" if self.max_fitness is not None else "🎯 Landscape Mapping Complete"
        
        panel = Panel(
            table,
            title=title,
            border_style="green",
            width=60
        )
        
        console.print(panel)
        completion_msg = "Filtered fitness landscape mapping completed successfully" if self.max_fitness is not None else "Landscape mapping completed successfully"
        self._log_to_file(completion_msg)
    
    def _clear_cache(self):
        """Clear existing cache files"""
        for file in self.cache_dir.glob("batch_*.npz"):
            file.unlink()
        self.batch_files = []
        self.total_samples = 0
    
    def _sample_focused_batched(self, n_samples, regions, method):
        """Generate focused samples in batches"""
        if not regions:
            self._sample_and_save_batch(n_samples, method, "Focused Sampling")
            return
        
        samples_per_region = n_samples // len(regions)
        remaining_samples = n_samples % len(regions)
        
        for region_idx, region in enumerate(regions):
            region_samples = samples_per_region
            if region_idx < remaining_samples:
                region_samples += 1
            
            if region_samples == 0:
                continue
            
            # Generate samples for this region in batches
            n_region_batches = max(1, (region_samples + self.batch_size - 1) // self.batch_size)
            
            for batch_idx in range(n_region_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, region_samples)
                batch_size = end_idx - start_idx
                
                # Generate focused samples for this batch
                positions = self._generate_focused_samples_single_region(batch_size, region, method)
                
                # Evaluate batch
                batch_samples = []
                batch_fitness = []
                
                self._evaluate_positions_parallel(
                    positions,
                    f"Focused Sampling Region {region_idx+1}/{len(regions)} Batch {batch_idx+1}/{n_region_batches}",
                    batch_samples,
                    batch_fitness
                )
                
                # Save batch to disk
                batch_file = self.cache_dir / f"batch_{len(self.batch_files):04d}.npz"
                np.savez_compressed(
                    batch_file,
                    samples=np.array(batch_samples),
                    fitness_values=np.array(batch_fitness)
                )
                
                self.batch_files.append(batch_file)
                self.total_samples += len(batch_samples)
    
    def _generate_focused_samples_single_region(self, n_samples, region, method):
        """Generate samples focused on a single region"""
        center = region['center']
        radius = region['radius']
        
        if method == 'lhs':
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=self.dimensions)
            unit_samples = sampler.random(n=n_samples)
            
            # Scale to region bounds
            lower = np.maximum(center - radius, np.array([b[0] for b in self.param_bounds.values()]))
            upper = np.minimum(center + radius, np.array([b[1] for b in self.param_bounds.values()]))
            
            region_samples = qmc.scale(unit_samples, lower, upper)
        else:
            # Random sampling in region
            lower = np.maximum(center - radius, np.array([b[0] for b in self.param_bounds.values()]))
            upper = np.minimum(center + radius, np.array([b[1] for b in self.param_bounds.values()]))
            region_samples = np.random.uniform(lower, upper, (n_samples, self.dimensions))
        
        return region_samples
    
    def _evaluate_positions_parallel(self, positions, description="Evaluating", batch_samples=None, batch_fitness=None):
        """Evaluate positions using multiprocessing with rich progress bar
        
        Parameters
        ----------
        positions : np.ndarray
            Array of positions to evaluate
        description : str
            Description for progress bar
        batch_samples : list, optional
            List to append samples to (for batched processing)
        batch_fitness : list, optional
            List to append fitness values to (for batched processing)
            
        Returns
        -------
        float
            Best fitness found in this batch
        """
        # Use provided lists or default to instance variables
        if batch_samples is None:
            batch_samples = self.samples
        if batch_fitness is None:
            batch_fitness = self.fitness_values
        
        # Prepare evaluation arguments with fixed parameters
        eval_args = [
            (self.evaluator, pos, i, self.fixed_params, self.all_param_names, self.optimizable_param_names)
            for i, pos in enumerate(positions)
        ]
        
        # Create multiprocessing pool
        n_processes = max(1, cpu_count() - 1)
        pool = Pool(processes=n_processes)
        
        # Track best fitness found
        best_fitness = float('inf')
        
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
            
            # Collect results
            for result in pool.imap_unordered(self._evaluate_single, eval_args):
                fitness, idx = result
                batch_samples.append(positions[idx])
                batch_fitness.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                
                progress.update(
                    task,
                    advance=1,
                    description=f"{description} | Best: {best_fitness:.2e}"
                )
        
        pool.close()
        pool.join()
        
        # Log progress to file
        progress_msg = f"Completed {description}: {len(positions)} samples, best fitness: {best_fitness:.2e}"
        self._log_to_file(progress_msg)
        
        return best_fitness
    
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
    
    def _identify_interesting_regions_from_cache(self):
        """Identify regions with high gradients or extreme values from cached batches
        
        Returns
        -------
        list of dict
            List of interesting regions with center and radius
        """
        interesting_regions = []
        
        # Load a sample of data from batches to identify regions
        sample_size = min(2000, self.total_samples)  # Use up to 2000 samples for analysis
        samples_per_batch = sample_size // len(self.batch_files)
        
        all_samples = []
        all_fitness = []
        
        for batch_file in self.batch_files:
            data = np.load(batch_file)
            batch_samples = data['samples']
            batch_fitness = data['fitness_values']
            
            # Sample from this batch
            n_from_batch = min(samples_per_batch, len(batch_samples))
            if n_from_batch > 0:
                indices = np.random.choice(len(batch_samples), n_from_batch, replace=False)
                all_samples.append(batch_samples[indices])
                all_fitness.append(batch_fitness[indices])
        
        if not all_samples:
            return []
        
        samples = np.vstack(all_samples)
        fitness_values = np.concatenate(all_fitness)
        
        # Find regions with extreme fitness (top 10% and bottom 10%)
        top_10_indices = np.argsort(fitness_values)[:int(len(fitness_values) * 0.1)]
        bottom_10_indices = np.argsort(fitness_values)[-int(len(fitness_values) * 0.1):]
        
        extreme_indices = np.concatenate([top_10_indices, bottom_10_indices])
        
        # Find regions with high gradients
        distances = squareform(pdist(samples))
        fitness_diff = np.abs(fitness_values[:, None] - fitness_values[None, :])
        
        # Calculate local gradient for each point
        gradients = []
        for i in range(len(samples)):
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
            center = samples[idx]
            # Radius is 5% of parameter ranges
            radius = 0.05 * param_ranges
            
            interesting_regions.append({
                'center': center,
                'radius': radius,
                'fitness': fitness_values[idx],
                'gradient': gradients[idx]
            })
        
        logging.info(f"Identified {len(interesting_regions)} interesting regions:")
        logging.info(f"  - {len(extreme_indices)} extreme fitness regions")
        logging.info(f"  - {len(high_gradient_indices)} high gradient regions")
        
        return interesting_regions
    
    def _calculate_summary_stats(self):
        """Calculate summary statistics from all batches without loading everything into memory"""
        min_fitness = float('inf')
        max_fitness = float('-inf')
        sum_fitness = 0.0
        sum_squared_fitness = 0.0
        all_fitness_values = []
        
        for batch_file in self.batch_files:
            data = np.load(batch_file)
            batch_fitness = data['fitness_values']
            
            min_fitness = min(min_fitness, np.min(batch_fitness))
            max_fitness = max(max_fitness, np.max(batch_fitness))
            sum_fitness += np.sum(batch_fitness)
            sum_squared_fitness += np.sum(batch_fitness ** 2)
            all_fitness_values.extend(batch_fitness.tolist())
        
        # Calculate statistics
        mean_fitness = sum_fitness / self.total_samples
        variance = (sum_squared_fitness / self.total_samples) - (mean_fitness ** 2)
        std_fitness = np.sqrt(max(0, variance))
        
        # Calculate percentiles
        all_fitness_values = np.array(all_fitness_values)
        q25 = np.percentile(all_fitness_values, 25)
        q75 = np.percentile(all_fitness_values, 75)
        median = np.percentile(all_fitness_values, 50)
        
        # Store summary stats
        self.fitness_stats = {
            'min': float(min_fitness),
            'max': float(max_fitness),
            'mean': float(mean_fitness),
            'median': float(median),
            'std': float(std_fitness),
            'q25': float(q25),
            'q75': float(q75),
            'range': float(max_fitness - min_fitness),
            'coefficient_of_variation': float(std_fitness / (abs(mean_fitness) + 1e-10))
        }
        
        logging.info(f"Summary stats calculated. Fitness range: [{min_fitness:.2e}, {max_fitness:.2e}]")
    
    def _calculate_sampling_efficiency_batched(self):
        """Calculate sampling efficiency using a sample of the data"""
        # Load a sample for efficiency calculation
        sample_size = min(1000, self.total_samples)
        samples_per_batch = sample_size // len(self.batch_files)
        
        all_samples = []
        
        for batch_file in self.batch_files:
            data = np.load(batch_file)
            batch_samples = data['samples']
            
            n_from_batch = min(samples_per_batch, len(batch_samples))
            if n_from_batch > 0:
                indices = np.random.choice(len(batch_samples), n_from_batch, replace=False)
                all_samples.append(batch_samples[indices])
        
        if not all_samples:
            return
        
        samples = np.vstack(all_samples)
        
        # Calculate coverage using minimum spanning tree length
        from scipy.sparse.csgraph import minimum_spanning_tree
        distances = squareform(pdist(samples))
        mst = minimum_spanning_tree(distances)
        coverage_score = mst.sum() / len(samples)
        
        # Calculate diversity (average distance between samples)
        diversity = np.mean(distances[np.triu_indices_from(distances, k=1)])
        
        logging.info(f"Sampling efficiency (from {len(samples)} sample points):")
        logging.info(f"  - Coverage score: {coverage_score:.4f}")
        logging.info(f"  - Average diversity: {diversity:.4f}")
        
        self.sampling_efficiency = {
            'coverage_score': float(coverage_score),
            'diversity': float(diversity)
        }
    
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
        Memory optimized with chunked processing and batched data loading
        
        Returns
        -------
        dict
            Ruggedness metrics including sensitivity analysis
        """
        logging.info("Calculating ruggedness and sensitivity...")
        
        # Load a sample of data for analysis
        sample_size = min(1000, self.total_samples)
        samples, fitness_values = self._load_sample_data(sample_size)
        
        if len(samples) < 10:
            logging.warning("Too few samples for ruggedness analysis")
            return {'ruggedness_score': 0, 'interpretation': 'insufficient_data'}
        
        # Calculate distances and fitness differences for sample
        distances = squareform(pdist(samples))
        fitness_diff = np.abs(fitness_values[:, None] - fitness_values[None, :])
        
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
    
    def _load_all_samples(self):
        """Load all samples from batches for final analysis
        
        Returns
        -------
        tuple
            (samples, fitness_values) arrays containing all data
        """
        all_samples = []
        all_fitness = []
        
        load_msg = f"Loading all {self.total_samples} samples from {len(self.batch_files)} batches for final analysis"
        logging.info(load_msg)
        self._log_to_file(load_msg)
        
        for i, batch_file in enumerate(self.batch_files):
            data = np.load(batch_file)
            all_samples.append(data['samples'])
            all_fitness.append(data['fitness_values'])
            
            if (i + 1) % 10 == 0 or i == len(self.batch_files) - 1:
                progress_msg = f"Loaded {i + 1}/{len(self.batch_files)} batch files"
                logging.info(progress_msg)
                self._log_to_file(progress_msg)
        
        samples = np.vstack(all_samples)
        fitness_values = np.concatenate(all_fitness)
        
        complete_msg = f"Successfully loaded all {len(samples)} samples for final analysis"
        logging.info(complete_msg)
        self._log_to_file(complete_msg)
        
        return samples, fitness_values
    
    def _load_sample_data(self, sample_size):
        """Load a sample of data from batches for analysis
        
        Parameters
        ----------
        sample_size : int
            Number of samples to load
        
        Returns
        -------
        tuple
            (samples, fitness_values) arrays
        """
        if sample_size >= self.total_samples:
            # Load all data
            return self._load_all_samples()
        
        # Load a sample from each batch
        samples_per_batch = sample_size // len(self.batch_files)
        remaining = sample_size % len(self.batch_files)
        
        all_samples = []
        all_fitness = []
        
        for i, batch_file in enumerate(self.batch_files):
            data = np.load(batch_file)
            batch_samples = data['samples']
            batch_fitness = data['fitness_values']
            
            n_from_batch = samples_per_batch
            if i < remaining:
                n_from_batch += 1
            
            if n_from_batch > 0 and len(batch_samples) > 0:
                n_from_batch = min(n_from_batch, len(batch_samples))
                indices = np.random.choice(len(batch_samples), n_from_batch, replace=False)
                all_samples.append(batch_samples[indices])
                all_fitness.append(batch_fitness[indices])
        
        if not all_samples:
            return np.array([]), np.array([])
        
        return np.vstack(all_samples), np.concatenate(all_fitness)
    
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
        
        # Load sample data for analysis
        sample_size = min(2000, self.total_samples)
        samples, fitness_values = self._load_sample_data(sample_size)
        
        if len(samples) < 10:
            logging.warning("Too few samples for modality analysis")
            return {'n_local_optima': 0, 'n_basins': 0, 'interpretation': 'insufficient_data'}
        
        # Use PCA for high-dimensional data
        if self.dimensions > 10:
            pca = PCA(n_components=min(10, self.dimensions))
            samples_reduced = pca.fit_transform(samples)
            logging.info(f"Reduced to {samples_reduced.shape[1]}D (variance explained: {pca.explained_variance_ratio_.sum():.2%})")
        else:
            samples_reduced = samples
        
        # Find local optima (samples better than neighbors)
        distances = squareform(pdist(samples_reduced))
        eps = np.percentile(distances[distances > 0], eps_percentile)
        
        local_optima_mask = np.zeros(len(fitness_values), dtype=bool)
        for i in range(len(fitness_values)):
            neighbors = distances[i] < eps
            neighbors[i] = False
            if np.any(neighbors):
                if fitness_values[i] < np.min(fitness_values[neighbors]):
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
            'local_optima_density': n_local_optima / len(fitness_values),
            'interpretation': 'multimodal' if n_basins > 3 else 'bimodal' if n_basins == 2 else 'unimodal'
        }
    
    def analyze_gradients(self, n_neighbors=10):
        """Analyze gradient information (memory optimized with batched data)
        
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
        
        # Load sample data for analysis
        sample_size = min(1000, self.total_samples)
        samples, fitness_values = self._load_sample_data(sample_size)
        
        if len(samples) < n_neighbors + 1:
            logging.warning("Too few samples for gradient analysis")
            return {'mean_gradient': 0, 'interpretation': 'insufficient_data'}
        
        distances = squareform(pdist(samples))
        gradients = []
        
        for i in range(len(samples)):
            # Find k nearest neighbors
            neighbor_indices = np.argsort(distances[i])[1:n_neighbors+1]
            
            # Estimate gradient magnitude
            for j in neighbor_indices:
                dist = distances[i, j]
                if dist > 0:
                    fitness_diff = abs(fitness_values[j] - fitness_values[i])
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
        """Detect flat/neutral regions (memory optimized with batched data)
        
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
        
        # Load sample data for analysis
        sample_size = min(1000, self.total_samples)
        samples, fitness_values = self._load_sample_data(sample_size)
        
        if len(samples) < 10:
            logging.warning("Too few samples for neutrality analysis")
            return {'neutrality_ratio': 0, 'interpretation': 'insufficient_data'}
        
        distances = squareform(pdist(samples))
        fitness_diff = np.abs(fitness_values[:, None] - fitness_values[None, :])
        
        # Find pairs with small fitness difference despite distance
        neutral_mask = (distances > 0) & (fitness_diff < tolerance)
        neutrality_ratio = np.sum(neutral_mask) / np.sum(distances > 0)
        
        # Find neutral networks (connected components of neutral points)
        threshold = np.percentile(fitness_values, 50)
        near_threshold = np.abs(fitness_values - threshold) < tolerance
        n_neutral_points = np.sum(near_threshold)
        
        return {
            'neutrality_ratio': float(neutrality_ratio),
            'n_neutral_points': int(n_neutral_points),
            'neutral_percentage': float(n_neutral_points / len(fitness_values) * 100),
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
        
        # Calculate actual coverage using sample data
        sample_size = min(1000, self.total_samples)
        samples, _ = self._load_sample_data(sample_size)
        
        bounds = np.array(list(self.param_bounds.values()))
        param_ranges = bounds[:, 1] - bounds[:, 0]
        total_volume = np.prod(param_ranges)
        
        # Estimate sampled volume using convex hull (for low dimensions)
        if self.dimensions <= 10 and len(samples) > self.dimensions:
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(samples)
                sampled_volume = hull.volume
                coverage_ratio = sampled_volume / total_volume
            except:
                coverage_ratio = 0
        else:
            # For high dimensions, estimate using sample spread
            if len(samples) > 0:
                sample_ranges = np.max(samples, axis=0) - np.min(samples, axis=0)
                coverage_ratio = np.prod(sample_ranges / param_ranges)
            else:
                coverage_ratio = 0
        
        # Calculate sample density
        sample_density = self.total_samples / total_volume
        
        # Assess local coverage (are samples well-distributed?)
        if len(samples) > 1:
            distances = squareform(pdist(samples))
            nearest_neighbor_dists = np.min(distances + np.eye(len(distances)) * 1e10, axis=1)
            coverage_uniformity = np.std(nearest_neighbor_dists) / (np.mean(nearest_neighbor_dists) + 1e-10)
        else:
            coverage_uniformity = 0
        
        # Determine adequacy level
        if self.total_samples < 100:
            adequacy = 'very_low'
            confidence = 0.1
            warning = 'CRITICAL: Far too few samples for reliable analysis'
        elif self.total_samples < 500:
            adequacy = 'low'
            confidence = 0.3
            warning = 'WARNING: Limited samples, results may be unreliable'
        elif self.total_samples < 2000:
            adequacy = 'moderate'
            confidence = 0.6
            warning = 'CAUTION: Moderate sampling, consider increasing for high-dimensional spaces'
        elif self.total_samples < 5000:
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
            warnings.append(f'Theoretical minimum samples: {theoretical_min:.2e} (you have {self.total_samples})')
        
        if coverage_ratio < 0.01:
            warnings.append(f'LOW COVERAGE: Only {coverage_ratio*100:.2f}% of search space sampled')
        
        if coverage_uniformity > 2.0:
            warnings.append('UNEVEN SAMPLING: Samples not uniformly distributed')
        
        return {
            'adequacy_level': adequacy,
            'confidence_score': float(adjusted_confidence),
            'n_samples': self.total_samples,
            'dimensions': self.dimensions,
            'theoretical_min_samples': float(theoretical_min),
            'coverage_ratio': float(coverage_ratio),
            'sample_density': float(sample_density),
            'coverage_uniformity': float(coverage_uniformity),
            'warnings': warnings,
            'recommendation': self._get_sampling_recommendation(adequacy, self.dimensions, self.total_samples)
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
        Uses batched data loading for memory efficiency
        
        Returns
        -------
        dict
            Parameter sensitivity rankings with local/global analysis
        """
        logging.info("Analyzing parameter sensitivity...")
        
        # Load sample data for analysis
        sample_size = min(2000, self.total_samples)
        samples, fitness_values = self._load_sample_data(sample_size)
        
        if len(samples) < 10:
            logging.warning("Too few samples for parameter sensitivity analysis")
            return {'sensitivities': {}, 'most_sensitive': 'unknown', 'interpretation': 'insufficient_data'}
        
        param_names = list(self.param_bounds.keys())
        sensitivities = []
        local_sensitivities = []
        
        for dim in range(self.dimensions):
            # Global sensitivity: overall correlation
            global_corr = np.abs(spearmanr(samples[:, dim], fitness_values)[0])
            sensitivities.append(global_corr)
            
            # Local sensitivity: fitness change per unit parameter change
            # Look at nearest neighbors
            distances = squareform(pdist(samples))
            local_sens_values = []
            
            for i in range(min(100, len(samples))):  # Sample subset for speed
                # Find 10 nearest neighbors
                neighbor_indices = np.argsort(distances[i])[1:11]
                
                for j in neighbor_indices:
                    param_change = abs(samples[j, dim] - samples[i, dim])
                    if param_change > 1e-10:
                        fitness_change = abs(fitness_values[j] - fitness_values[i])
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
            'most_sensitive': sensitivity_ranking[0][0] if sensitivity_ranking else 'unknown',
            'least_sensitive': sensitivity_ranking[-1][0] if sensitivity_ranking else 'unknown',
            'sensitivity_range': float(sensitivity_ranking[0][1] - sensitivity_ranking[-1][1]) if len(sensitivity_ranking) > 1 else 0,
            'dangerous_parameters': dangerous_params,
            'warning': f'Parameters with extreme local sensitivity: {dangerous_params}' if dangerous_params else None
        }
    
    def generate_report(self, use_all_samples=True):
        """Generate comprehensive landscape analysis report
        
        ENHANCED: Includes adequacy assessment and extreme sensitivity warnings
        Processes all samples without loading them into RAM simultaneously
        
        Parameters
        ----------
        use_all_samples : bool
            If True, processes all samples from batches (streaming analysis)
            If False, uses sample-based analysis (faster but less accurate)
        
        Returns
        -------
        dict
            Complete landscape analysis with confidence metrics
        """
        report_msg = f"Generating landscape report (use_all_samples={use_all_samples})..."
        logging.info(report_msg)
        self._log_to_file(report_msg)
        
        # First assess sampling adequacy
        adequacy = self.assess_sampling_adequacy()
        
        # Use streaming analysis for all samples without loading into RAM
        if use_all_samples:
            streaming_msg = f"Using streaming analysis of all {self.total_samples} samples (memory efficient)"
            logging.info(streaming_msg)
            self._log_to_file(streaming_msg)
            
            report = {
                'dimensions': self.dimensions,
                'n_samples': self.total_samples,
                'sampling_adequacy': adequacy,
                'fitness_statistics': self.fitness_stats,
                'ruggedness': self.calculate_ruggedness_streaming(),
                'modality': self.detect_modality_streaming(),
                'gradients': self.analyze_gradients_streaming(),
                'neutrality': self.detect_neutrality_streaming(),
                'parameter_sensitivity': self.analyze_parameter_sensitivity_streaming()
            }
            
            final_msg = "Final report generated using streaming analysis of all samples"
            logging.info(final_msg)
            self._log_to_file(final_msg)
        else:
            # Use sample-based analysis
            sample_msg = f"Using sample-based analysis (faster mode)"
            logging.info(sample_msg)
            self._log_to_file(sample_msg)
            
            report = {
                'dimensions': self.dimensions,
                'n_samples': self.total_samples,
                'sampling_adequacy': adequacy,
                'fitness_statistics': self.fitness_stats,
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
    
        return report
    
    def calculate_ruggedness_streaming(self):
        """Calculate ruggedness by streaming through batches (memory efficient)"""
        logging.info("Calculating ruggedness using streaming analysis...")
        self._log_to_file("Calculating ruggedness using streaming analysis...")
        
        # Collect samples from all batches for analysis
        all_gradients = []
        all_fitness_diffs = []
        all_distances = []
        
        # Process batches in pairs to calculate distances and fitness differences
        for i, batch_file_1 in enumerate(self.batch_files):
            data_1 = np.load(batch_file_1)
            samples_1 = data_1['samples']
            fitness_1 = data_1['fitness_values']
            
            # Sample from this batch to manage memory
            n_sample = min(200, len(samples_1))
            if len(samples_1) > n_sample:
                indices = np.random.choice(len(samples_1), n_sample, replace=False)
                samples_1 = samples_1[indices]
                fitness_1 = fitness_1[indices]
            
            # Compare with samples from same and other batches
            for j, batch_file_2 in enumerate(self.batch_files[i:], i):
                data_2 = np.load(batch_file_2)
                samples_2 = data_2['samples']
                fitness_2 = data_2['fitness_values']
                
                # Sample from second batch
                n_sample_2 = min(200, len(samples_2))
                if len(samples_2) > n_sample_2:
                    indices_2 = np.random.choice(len(samples_2), n_sample_2, replace=False)
                    samples_2 = samples_2[indices_2]
                    fitness_2 = fitness_2[indices_2]
                
                # Calculate distances and fitness differences
                for idx1, (s1, f1) in enumerate(zip(samples_1, fitness_1)):
                    start_idx = idx1 + 1 if i == j else 0
                    for s2, f2 in zip(samples_2[start_idx:], fitness_2[start_idx:]):
                        dist = np.linalg.norm(s1 - s2)
                        if dist > 0:
                            fitness_diff = abs(f1 - f2)
                            all_distances.append(dist)
                            all_fitness_diffs.append(fitness_diff)
                            all_gradients.append(fitness_diff / dist)
        
        if not all_distances:
            return {'ruggedness_score': 0, 'interpretation': 'insufficient_data'}
        
        distances = np.array(all_distances)
        fitness_diffs = np.array(all_fitness_diffs)
        gradients = np.array(all_gradients)
        
        # Detect extreme sensitivity
        very_small_distances = distances < np.percentile(distances, 5)
        if np.any(very_small_distances):
            small_dist_fitness_changes = fitness_diffs[very_small_distances]
            extreme_sensitivity_ratio = np.sum(small_dist_fitness_changes > np.percentile(fitness_diffs, 75)) / len(small_dist_fitness_changes)
            max_sensitivity = np.max(gradients[very_small_distances])
        else:
            extreme_sensitivity_ratio = 0
            max_sensitivity = 0
        
        # Autocorrelation analysis
        distance_bins = np.percentile(distances, [1, 5, 10, 25, 50, 75, 90])
        autocorr = []
        fitness_volatility = []
        
        for d in distance_bins:
            mask = distances < d
            if np.any(mask):
                corr = spearmanr(distances[mask], fitness_diffs[mask])[0]
                autocorr.append(corr)
                volatility = np.std(fitness_diffs[mask]) / (np.mean(fitness_diffs[mask]) + 1e-10)
                fitness_volatility.append(volatility)
        
        ruggedness = 1 - np.mean(autocorr) if autocorr else 0
        cliff_score = np.sum(fitness_diffs > 10 * np.median(fitness_diffs)) / len(fitness_diffs)
        
        return {
            'ruggedness_score': ruggedness,
            'autocorrelation': autocorr,
            'distance_scales': distance_bins.tolist(),
            'fitness_volatility': fitness_volatility,
            'extreme_sensitivity_ratio': float(extreme_sensitivity_ratio),
            'max_sensitivity': float(max_sensitivity),
            'cliff_score': float(cliff_score),
            'interpretation': 'extreme' if extreme_sensitivity_ratio > 0.3 else 'high' if ruggedness > 0.7 else 'medium' if ruggedness > 0.4 else 'low',
            'warning': 'EXTREME SENSITIVITY DETECTED!' if extreme_sensitivity_ratio > 0.3 else None,
            'analysis_method': 'streaming'
        }
    
    def detect_modality_streaming(self):
        """Detect modality by streaming through batches"""
        logging.info("Detecting modality using streaming analysis...")
        self._log_to_file("Detecting modality using streaming analysis...")
        
        # Collect representative samples from all batches
        all_samples = []
        all_fitness = []
        
        for batch_file in self.batch_files:
            data = np.load(batch_file)
            samples = data['samples']
            fitness = data['fitness_values']
            
            # Sample from each batch
            n_sample = min(300, len(samples))
            if len(samples) > n_sample:
                indices = np.random.choice(len(samples), n_sample, replace=False)
                samples = samples[indices]
                fitness = fitness[indices]
            
            all_samples.append(samples)
            all_fitness.append(fitness)
        
        samples = np.vstack(all_samples)
        fitness_values = np.concatenate(all_fitness)
        
        if len(samples) < 10:
            return {'n_local_optima': 0, 'n_basins': 0, 'interpretation': 'insufficient_data'}
        
        # Use PCA for high dimensions
        if self.dimensions > 10:
            pca = PCA(n_components=min(10, self.dimensions))
            samples_reduced = pca.fit_transform(samples)
        else:
            samples_reduced = samples
        
        # Find local optima
        distances = squareform(pdist(samples_reduced))
        eps = np.percentile(distances[distances > 0], 5)
        
        local_optima_mask = np.zeros(len(fitness_values), dtype=bool)
        for i in range(len(fitness_values)):
            neighbors = distances[i] < eps
            neighbors[i] = False
            if np.any(neighbors):
                if fitness_values[i] < np.min(fitness_values[neighbors]):
                    local_optima_mask[i] = True
        
        n_local_optima = np.sum(local_optima_mask)
        
        if n_local_optima > 1:
            local_optima_samples = samples_reduced[local_optima_mask]
            clustering = DBSCAN(eps=eps*3, min_samples=1).fit(local_optima_samples)
            n_basins = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        else:
            n_basins = n_local_optima
        
        return {
            'n_local_optima': int(n_local_optima),
            'n_basins': int(n_basins),
            'local_optima_density': n_local_optima / len(fitness_values),
            'interpretation': 'multimodal' if n_basins > 3 else 'bimodal' if n_basins == 2 else 'unimodal',
            'analysis_method': 'streaming'
        }
    
    def analyze_gradients_streaming(self):
        """Analyze gradients by streaming through batches"""
        logging.info("Analyzing gradients using streaming analysis...")
        self._log_to_file("Analyzing gradients using streaming analysis...")
        
        all_gradients = []
        
        # Process each batch
        for batch_file in self.batch_files:
            data = np.load(batch_file)
            samples = data['samples']
            fitness = data['fitness_values']
            
            # Sample from batch
            n_sample = min(200, len(samples))
            if len(samples) > n_sample:
                indices = np.random.choice(len(samples), n_sample, replace=False)
                samples = samples[indices]
                fitness = fitness[indices]
            
            if len(samples) < 11:
                continue
            
            # Calculate gradients within batch
            distances = squareform(pdist(samples))
            for i in range(len(samples)):
                neighbor_indices = np.argsort(distances[i])[1:11]
                for j in neighbor_indices:
                    dist = distances[i, j]
                    if dist > 0:
                        fitness_diff = abs(fitness[j] - fitness[i])
                        gradient = fitness_diff / dist
                        all_gradients.append(gradient)
        
        if not all_gradients:
            return {'mean_gradient': 0, 'interpretation': 'insufficient_data'}
        
        gradients = np.array(all_gradients)
        
        return {
            'mean_gradient': float(np.mean(gradients)),
            'median_gradient': float(np.median(gradients)),
            'std_gradient': float(np.std(gradients)),
            'max_gradient': float(np.max(gradients)),
            'gradient_variability': float(np.std(gradients) / (np.mean(gradients) + 1e-10)),
            'interpretation': 'steep' if np.median(gradients) > np.percentile(gradients, 75) else 'moderate',
            'analysis_method': 'streaming'
        }
    
    def detect_neutrality_streaming(self):
        """Detect neutrality by streaming through batches"""
        logging.info("Detecting neutrality using streaming analysis...")
        self._log_to_file("Detecting neutrality using streaming analysis...")
        
        all_neutral_ratios = []
        all_fitness_values = []
        
        # Collect all fitness values for global threshold
        for batch_file in self.batch_files:
            data = np.load(batch_file)
            all_fitness_values.extend(data['fitness_values'].tolist())
        
        threshold = np.percentile(all_fitness_values, 50)
        tolerance = 1e-3
        
        # Process each batch for neutrality analysis
        total_pairs = 0
        neutral_pairs = 0
        total_neutral_points = 0
        
        for batch_file in self.batch_files:
            data = np.load(batch_file)
            samples = data['samples']
            fitness = data['fitness_values']
            
            # Count neutral points in this batch
            near_threshold = np.abs(fitness - threshold) < tolerance
            total_neutral_points += np.sum(near_threshold)
            
            # Sample for pairwise analysis
            n_sample = min(150, len(samples))
            if len(samples) > n_sample:
                indices = np.random.choice(len(samples), n_sample, replace=False)
                samples = samples[indices]
                fitness = fitness[indices]
            
            if len(samples) < 2:
                continue
            
            # Calculate neutrality within batch
            distances = squareform(pdist(samples))
            fitness_diff = np.abs(fitness[:, None] - fitness[None, :])
            
            neutral_mask = (distances > 0) & (fitness_diff < tolerance)
            batch_neutral_pairs = np.sum(neutral_mask)
            batch_total_pairs = np.sum(distances > 0)
            
            neutral_pairs += batch_neutral_pairs
            total_pairs += batch_total_pairs
        
        neutrality_ratio = neutral_pairs / total_pairs if total_pairs > 0 else 0
        
        return {
            'neutrality_ratio': float(neutrality_ratio),
            'n_neutral_points': int(total_neutral_points),
            'neutral_percentage': float(total_neutral_points / len(all_fitness_values) * 100),
            'interpretation': 'high' if neutrality_ratio > 0.3 else 'medium' if neutrality_ratio > 0.1 else 'low',
            'analysis_method': 'streaming'
        }
    
    def analyze_parameter_sensitivity_streaming(self):
        """Analyze parameter sensitivity by streaming through batches"""
        logging.info("Analyzing parameter sensitivity using streaming analysis...")
        self._log_to_file("Analyzing parameter sensitivity using streaming analysis...")
        
        param_names = list(self.param_bounds.keys())
        
        # Collect all samples and fitness for global correlation
        all_samples = []
        all_fitness = []
        
        for batch_file in self.batch_files:
            data = np.load(batch_file)
            all_samples.append(data['samples'])
            all_fitness.append(data['fitness_values'])
        
        all_samples = np.vstack(all_samples)
        all_fitness = np.concatenate(all_fitness)
        
        # Calculate global sensitivities
        sensitivities = []
        for dim in range(self.dimensions):
            global_corr = np.abs(spearmanr(all_samples[:, dim], all_fitness)[0])
            sensitivities.append(global_corr)
        
        # Calculate local sensitivities using samples from batches
        local_sensitivities = []
        
        for dim in range(self.dimensions):
            all_local_sens = []
            
            # Process each batch for local sensitivity
            for batch_file in self.batch_files:
                data = np.load(batch_file)
                samples = data['samples']
                fitness = data['fitness_values']
                
                # Sample from batch
                n_sample = min(100, len(samples))
                if len(samples) > n_sample:
                    indices = np.random.choice(len(samples), n_sample, replace=False)
                    samples = samples[indices]
                    fitness = fitness[indices]
                
                if len(samples) < 11:
                    continue
                
                # Calculate local sensitivity within batch
                distances = squareform(pdist(samples))
                for i in range(min(50, len(samples))):
                    neighbor_indices = np.argsort(distances[i])[1:6]
                    for j in neighbor_indices:
                        param_change = abs(samples[j, dim] - samples[i, dim])
                        if param_change > 1e-10:
                            fitness_change = abs(fitness[j] - fitness[i])
                            local_sens = fitness_change / param_change
                            all_local_sens.append(local_sens)
            
            local_sensitivity = np.median(all_local_sens) if all_local_sens else 0
            local_sensitivities.append(local_sensitivity)
        
        # Rank parameters
        sensitivity_ranking = sorted(
            zip(param_names, sensitivities, local_sensitivities),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Detect dangerous parameters
        dangerous_params = []
        for name, global_sens, local_sens in sensitivity_ranking:
            if local_sens > np.percentile(local_sensitivities, 75) and global_sens < np.percentile(sensitivities, 50):
                dangerous_params.append(name)
        
        return {
            'sensitivities': {name: float(sens) for name, sens, _ in sensitivity_ranking},
            'local_sensitivities': {name: float(local_sens) for name, _, local_sens in sensitivity_ranking},
            'most_sensitive': sensitivity_ranking[0][0] if sensitivity_ranking else 'unknown',
            'least_sensitive': sensitivity_ranking[-1][0] if sensitivity_ranking else 'unknown',
            'sensitivity_range': float(sensitivity_ranking[0][1] - sensitivity_ranking[-1][1]) if len(sensitivity_ranking) > 1 else 0,
            'dangerous_parameters': dangerous_params,
            'warning': f'Parameters with extreme local sensitivity: {dangerous_params}' if dangerous_params else None,
            'analysis_method': 'streaming'
        }
    
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
    
    def visualize_parameter_fitness_analysis(self, output_path='parameter_fitness_analysis.png'):
        """Create parameter-by-parameter fitness analysis plots
        
        Shows which parameter values give the best fitness for each parameter individually.
        Much more actionable than 2D projections for optimization guidance.
        
        Parameters
        ----------
        output_path : str
            Output file path for the analysis plots
        """
        viz_msg = "Creating parameter-by-parameter fitness analysis..."
        logging.info(viz_msg)
        self._log_to_file(viz_msg)
        
        # Stream samples from batches for visualization
        all_samples = []
        all_fitness = []
        
        # Load samples from batches (limit to reasonable number for visualization)
        max_samples_per_batch = 1000
        total_viz_samples = 0
        
        for batch_file in self.batch_files:
            data = np.load(batch_file)
            samples = data['samples']
            fitness = data['fitness_values']
            
            # Sample from this batch
            n_from_batch = min(max_samples_per_batch, len(samples))
            if len(samples) > n_from_batch:
                indices = np.random.choice(len(samples), n_from_batch, replace=False)
                samples = samples[indices]
                fitness = fitness[indices]
            
            all_samples.append(samples)
            all_fitness.append(fitness)
            total_viz_samples += len(samples)
            
            # Limit total samples for visualization
            if total_viz_samples >= 10000:
                break
        
        if not all_samples:
            logging.warning("No samples available for visualization")
            return
        
        samples = np.vstack(all_samples)
        fitness_values = np.concatenate(all_fitness)
        
        param_names = list(self.param_bounds.keys())
        n_params = len(param_names)
        
        # Calculate optimal grid layout
        n_cols = min(4, n_params)  # Max 4 columns
        n_rows = (n_params + n_cols - 1) // n_cols
        
        # Create the figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten() if n_params > 1 else axes
        
        # Get parameter sensitivity for sorting
        sensitivity_report = self.analyze_parameter_sensitivity()
        param_sensitivities = sensitivity_report['sensitivities']
        
        # Sort parameters by sensitivity (most sensitive first)
        sorted_param_indices = sorted(range(n_params), 
                                    key=lambda i: param_sensitivities.get(param_names[i], 0), 
                                    reverse=True)
        
        # Find best fitness values for highlighting
        best_10_percent = np.percentile(fitness_values, 10)  # Lower is better
        best_25_percent = np.percentile(fitness_values, 25)
        
        # Handle log scaling for wide fitness ranges
        fitness_min = np.min(fitness_values)
        fitness_max = np.max(fitness_values)
        
        # Use log scaling if fitness range spans multiple orders of magnitude
        fitness_range_ratio = fitness_max / (abs(fitness_min) + 1e-10)
        use_log_scale = fitness_range_ratio > 100 or abs(fitness_min) > 100
        
        if use_log_scale:
            # Shift fitness values to be positive for log scaling
            fitness_offset = abs(fitness_min) + 1 if fitness_min <= 0 else 0
            fitness_for_color = fitness_values + fitness_offset
            fitness_min_color = np.min(fitness_for_color)
            fitness_max_color = np.max(fitness_for_color)
            
            # Use log scale for color mapping
            fitness_log = np.log10(fitness_for_color)
            fitness_min_log = np.log10(fitness_min_color)
            fitness_max_log = np.log10(fitness_max_color)
            
            logging.info(f"Using log-scaled colors: fitness range [{fitness_min:.2e}, {fitness_max:.2e}], log range [{fitness_min_log:.2f}, {fitness_max_log:.2f}]")
        else:
            fitness_for_color = fitness_values
            fitness_log = fitness_values
            fitness_min_log = fitness_min
            fitness_max_log = fitness_max
        
        param_analysis_results = {}
        
        for plot_idx, param_idx in enumerate(sorted_param_indices):
            ax = axes_flat[plot_idx]
            param_name = param_names[param_idx]
            param_values = samples[:, param_idx]
            bounds = self.param_bounds[param_name]
            
            # Create scatter plot with log-scaled color coding
            scatter = ax.scatter(param_values, fitness_values, 
                               c=fitness_log, cmap='plasma_r',  # Red (best) to purple (worst)
                               alpha=0.6, s=20, 
                               vmin=fitness_min_log, vmax=fitness_max_log)
            
            # Highlight best performing regions
            best_mask = fitness_values <= best_10_percent
            if np.any(best_mask):
                ax.scatter(param_values[best_mask], fitness_values[best_mask], 
                          c='red', s=30, alpha=0.8, marker='o', 
                          label=f'Top 10% (≤{best_10_percent:.2e})', zorder=5)
                
                # Calculate and show optimal parameter range
                best_param_values = param_values[best_mask]
                optimal_min = np.min(best_param_values)
                optimal_max = np.max(best_param_values)
                optimal_mean = np.mean(best_param_values)
                optimal_std = np.std(best_param_values)
                
                # Get y-axis limits for proper shading
                y_min, y_max = ax.get_ylim()
                if len(fitness_values) > 0:
                    y_min = min(y_min, np.min(fitness_values))
                    y_max = max(y_max, np.max(fitness_values))
                
                # Highlight optimal range with proper y-limits
                ax.axvspan(optimal_min, optimal_max, ymin=0, ymax=1, 
                          alpha=0.2, color='red', zorder=1,
                          label=f'Optimal range: [{optimal_min:.3f}, {optimal_max:.3f}]')
                ax.axvline(optimal_mean, color='red', linestyle='--', alpha=0.7, zorder=2,
                          label=f'Optimal mean: {optimal_mean:.3f}')
                
                param_analysis_results[param_name] = {
                    'optimal_range': [float(optimal_min), float(optimal_max)],
                    'optimal_mean': float(optimal_mean),
                    'optimal_std': float(optimal_std),
                    'sensitivity': param_sensitivities.get(param_name, 0),
                    'n_best_samples': int(np.sum(best_mask))
                }
            else:
                param_analysis_results[param_name] = {
                    'optimal_range': [float(np.min(param_values)), float(np.max(param_values))],
                    'optimal_mean': float(np.mean(param_values)),
                    'optimal_std': float(np.std(param_values)),
                    'sensitivity': param_sensitivities.get(param_name, 0),
                    'n_best_samples': 0
                }
            
            # Set labels and title
            ax.set_xlabel(f'{param_name}', fontsize=10)
            ax.set_ylabel('Fitness', fontsize=10)
            
            # Add sensitivity info to title
            sensitivity = param_sensitivities.get(param_name, 0)
            ax.set_title(f'{param_name}\nSensitivity: {sensitivity:.3f}', fontsize=11)
            
            # Set axis limits to parameter bounds
            ax.set_xlim(bounds[0], bounds[1])
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend for plots with best samples (limit to avoid clutter)
            if np.any(best_mask) and plot_idx < 8:  # Show legend on first 8 plots
                ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
        
        # Hide unused subplots
        for plot_idx in range(n_params, len(axes_flat)):
            axes_flat[plot_idx].set_visible(False)
        
        # Add overall title
        fig.suptitle(f'Parameter-by-Parameter Fitness Analysis\n{len(samples):,} samples across {n_params} parameters', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.15)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log completion
        viz_complete_msg = f"Saved parameter fitness analysis to {output_path}"
        logging.info(viz_complete_msg)
        self._log_to_file(viz_complete_msg)
        
        # Create visualization panel
        viz_table = Table(show_header=False, box=None, padding=(0, 1))
        viz_table.add_column("Metric", style="bold magenta")
        viz_table.add_column("Value", style="white")
        
        viz_table.add_row("Visualization", "Parameter-by-Parameter Analysis")
        viz_table.add_row("Parameters Analyzed", f"{n_params}")
        viz_table.add_row("Samples Plotted", f"{len(samples):,}")
        viz_table.add_row("Best Samples (Top 10%)", f"{np.sum(fitness_values <= best_10_percent)}")
        viz_table.add_row("Output File", output_path)
        
        viz_panel = Panel(
            viz_table,
            title="🎨 Parameter Analysis Complete",
            border_style="magenta",
            width=60
        )
        
        console.print(viz_panel)
        
        # Print parameter optimization recommendations
        self._print_parameter_recommendations(param_analysis_results)
        
        return {
            'output_path': output_path,
            'n_parameters': n_params,
            'samples_plotted': len(samples),
            'parameter_analysis': param_analysis_results,
            'fitness_range': [float(np.min(fitness_values)), float(np.max(fitness_values))],
            'best_fitness_threshold': float(best_10_percent)
        }
    
    def _print_parameter_recommendations(self, param_analysis):
        """Print parameter optimization recommendations"""
        print(f"\n{'='*80}")
        print(f"PARAMETER OPTIMIZATION RECOMMENDATIONS")
        print(f"{'='*80}")
        
        # Sort by sensitivity
        sorted_params = sorted(param_analysis.items(), 
                             key=lambda x: x[1]['sensitivity'], reverse=True)
        
        for i, (param_name, analysis) in enumerate(sorted_params[:10], 1):  # Top 10 most sensitive
            optimal_range = analysis['optimal_range']
            optimal_mean = analysis['optimal_mean']
            sensitivity = analysis['sensitivity']
            n_best = analysis['n_best_samples']
            
            print(f"\n{i}. {param_name}")
            print(f"   Sensitivity: {sensitivity:.4f} (rank {i})")
            print(f"   Optimal range: [{optimal_range[0]:.4f}, {optimal_range[1]:.4f}]")
            print(f"   Optimal center: {optimal_mean:.4f}")
            print(f"   Best samples in range: {n_best}")
            
            if sensitivity > 0.5:
                print(f"   🔥 HIGH IMPACT: Focus optimization on this parameter")
            elif sensitivity > 0.2:
                print(f"   ⚡ MEDIUM IMPACT: Important for fine-tuning")
            else:
                print(f"   💤 LOW IMPACT: Less critical for optimization")
        
        print(f"\n💡 OPTIMIZATION STRATEGY:")
        print(f"   1. Focus on top 5 most sensitive parameters first")
        print(f"   2. Use optimal ranges as initial bounds for search")
        print(f"   3. Parameters with low sensitivity can use wider exploration")
        print(f"   4. Red dots show best performing parameter values")
    
    def generate_ai_agent_report(self, report):
        """Generate comprehensive report optimized for AI agent consumption (streaming)
        
        Parameters
        ----------
        report : dict
            Base analysis report
            
        Returns
        -------
        dict
            Enhanced report with detailed context for AI agents
        """
        # Load sample data for topology analysis using streaming approach
        sample_size = min(300, self.total_samples)
        samples, fitness_values = self._load_sample_data(sample_size)
        
        if len(samples) == 0:
            # Return minimal report if no samples available
            return {
                'metadata': {
                    'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'dimensions': self.dimensions,
                    'n_samples': self.total_samples,
                    'parameter_names': list(self.param_bounds.keys())
                },
                'error': 'No samples available for AI agent report generation'
            }
        
        # Calculate distances for topology analysis
        distances = squareform(pdist(samples))
        fitness_diff = np.abs(fitness_values[:, None] - fitness_values[None, :])
        
        # Load all samples for parameter analysis (streaming)
        all_samples, all_fitness = self._load_all_samples()
        
        # Detailed parameter analysis using all samples
        param_names = list(self.param_bounds.keys())
        param_analysis = {}
        
        for i, param_name in enumerate(param_names):
            bounds = self.param_bounds[param_name]
            values = all_samples[:, i]
            
            # Calculate parameter-specific metrics
            param_fitness_corr = np.corrcoef(values, all_fitness)[0, 1]
            
            # Local sensitivity per parameter (use sample for efficiency)
            sample_values = samples[:, i]
            local_sens_values = []
            for j in range(min(50, len(samples))):
                neighbor_indices = np.argsort(distances[j])[1:6]
                for k in neighbor_indices:
                    param_change = abs(sample_values[k] - sample_values[j])
                    if param_change > 1e-10:
                        fitness_change = abs(fitness_values[k] - fitness_values[j])
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
                'sensitivity_rank': list(report['parameter_sensitivity']['sensitivities'].keys()).index(param_name) + 1 if param_name in report['parameter_sensitivity']['sensitivities'] else len(param_names)
            }
        
        # Detailed landscape topology using sample data
        topology = {
            'local_optima_positions': [],
            'high_gradient_regions': [],
            'flat_regions': [],
            'extreme_fitness_regions': []
        }
        
        # Find local optima (limit to top 10)
        local_optima_found = 0
        for i in range(len(samples)):
            if local_optima_found >= 10:
                break
            neighbors = np.argsort(distances[i])[1:6]
            if len(neighbors) > 0 and np.all(fitness_values[i] < fitness_values[neighbors]):
                topology['local_optima_positions'].append({
                    'position': samples[i].tolist(),
                    'fitness': float(fitness_values[i]),
                    'basin_size_estimate': float(np.min(distances[i, neighbors]))
                })
                local_optima_found += 1
        
        # Find high gradient regions (limit to top 10)
        gradients = []
        for i in range(len(samples)):
            neighbors = np.argsort(distances[i])[1:6]
            if len(neighbors) > 0:
                local_gradient = np.mean(fitness_diff[i, neighbors] / (distances[i, neighbors] + 1e-10))
                gradients.append(local_gradient)
            else:
                gradients.append(0)
        
        gradients = np.array(gradients)
        if len(gradients) > 0:
            high_gradient_threshold = np.percentile(gradients, 90)
            
            high_gradient_found = 0
            for i in range(len(samples)):
                if high_gradient_found >= 10:
                    break
                if gradients[i] > high_gradient_threshold:
                    topology['high_gradient_regions'].append({
                        'position': samples[i].tolist(),
                        'fitness': float(fitness_values[i]),
                        'gradient_magnitude': float(gradients[i])
                    })
                    high_gradient_found += 1
            
            # Find flat regions (limit to top 10)
            flat_threshold = np.percentile(gradients, 10)
            flat_found = 0
            for i in range(len(samples)):
                if flat_found >= 10:
                    break
                if gradients[i] < flat_threshold:
                    topology['flat_regions'].append({
                        'position': samples[i].tolist(),
                        'fitness': float(fitness_values[i]),
                        'gradient_magnitude': float(gradients[i])
                    })
                    flat_found += 1
        
        # Find extreme fitness regions using all samples (limit to 10 each)
        top_10_indices = np.argsort(all_fitness)[:10]
        bottom_10_indices = np.argsort(all_fitness)[-10:]
        
        for idx in top_10_indices:
            topology['extreme_fitness_regions'].append({
                'type': 'best',
                'position': all_samples[idx].tolist(),
                'fitness': float(all_fitness[idx]),
                'local_gradient': 0.0  # Not calculated for efficiency
            })
        
        for idx in bottom_10_indices:
            topology['extreme_fitness_regions'].append({
                'type': 'worst',
                'position': all_samples[idx].tolist(),
                'fitness': float(all_fitness[idx]),
                'local_gradient': 0.0  # Not calculated for efficiency
            })
        
        # Optimization strategy recommendations with detailed reasoning
        strategy_recommendations = self._generate_detailed_recommendations(report, param_analysis, topology)
        
        # Construct comprehensive AI agent report
        ai_report = {
            'metadata': {
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'dimensions': self.dimensions,
                'n_samples': self.total_samples,
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
                'batch_files': [str(f) for f in self.batch_files],
                'cache_directory': str(self.cache_dir),
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
    parser.add_argument('--batch-size', type=int, default=5000, help='Batch size for memory management')
    parser.add_argument('--method', type=str, default='lhs', choices=['lhs', 'sobol', 'random', 'grid'])
    parser.add_argument('--output', type=str, default='landscape_analysis', help='Output directory')
    parser.add_argument('--cache-dir', type=str, default='landscape_cache', help='Cache directory for batches')
    parser.add_argument('--visualize', action='store_true', help='Create parameter-by-parameter fitness analysis')
    parser.add_argument('--adaptive', action='store_true', default=True, help='Use adaptive sampling')
    parser.add_argument('--no-adaptive', dest='adaptive', action='store_false', help='Disable adaptive sampling')
    parser.add_argument('--max-fitness', type=float, help='Maximum fitness threshold - only accept samples with fitness <= this value (e.g., 1e2, -1e2, 0)')
    parser.add_argument('--use-all-samples', action='store_true', default=True, help='Use all samples for final report')
    parser.add_argument('--sample-based', dest='use_all_samples', action='store_false', help='Use sample-based analysis for final report')
    
    args = parser.parse_args()
    
    # Setup logging to both file and console
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'mapper.log', mode='w'),
            RichHandler(console=console, show_path=False)
        ]
    )
    
    # Parse arguments
    # (no additional parsing needed for new visualization)
    
    # Initialize evaluator
    init_msg = f"Initializing evaluator from {args.config_path}"
    logging.info(init_msg)
    evaluator, config, interval_data = await initEvaluator(args.config_path)
    
    # Get parameter bounds
    param_bounds = config['optimize']['bounds']
    for k, v in param_bounds.items():
        if len(v) == 1:
            param_bounds[k] = [v[0], v[0]]
    
    # Create mapper with batching support
    mapper = FitnessLandscapeMapper(
        evaluator, 
        config, 
        param_bounds, 
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        max_fitness=args.max_fitness
    )
    
    # Sample landscape (disable adaptive sampling in filtered mode)
    adaptive_sampling = args.adaptive and args.max_fitness is None
    mapper.sample_landscape(n_samples=args.samples, method=args.method, adaptive=adaptive_sampling)
    
    # Generate report
    report = mapper.generate_report(use_all_samples=args.use_all_samples)
    
    # Create visualization if requested
    viz_info = None
    if args.visualize:
        viz_info = mapper.visualize_parameter_fitness_analysis(
            output_path=f"{args.output}/parameter_fitness_analysis.png"
        )
    
    # Print summary
    print("\n" + "="*80)
    if args.max_fitness is not None:
        print("FILTERED FITNESS LANDSCAPE ANALYSIS SUMMARY")
    else:
        print("FITNESS LANDSCAPE ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nDimensions: {report['dimensions']}")
    if args.max_fitness is not None:
        print(f"Filtered Samples: {report['n_samples']} (processed in {len(mapper.batch_files)} batches)")
        print(f"Note: Only samples with fitness ≤ {args.max_fitness} were retained")
    else:
        print(f"Samples: {report['n_samples']} (processed in {len(mapper.batch_files)} batches)")
    
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
    
    # Print analysis method used
    analysis_method = report['ruggedness'].get('analysis_method', 'sample_based')
    print(f"\nAnalysis Method: {analysis_method}")
    
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
    
    # Print visualization info
    if viz_info:
        print(f"\n{'='*80}")
        print(f"VISUALIZATION")
        print(f"{'='*80}")
        print(f"Parameter analysis saved: {viz_info['output_path']}")
        print(f"Parameters analyzed: {viz_info['n_parameters']}")
        print(f"Samples plotted: {viz_info['samples_plotted']:,}")
        print(f"Best fitness threshold: {viz_info['best_fitness_threshold']:.2e}")
    
    # Save results
    mapper.save_results(report, args.output)
    
    print(f"\n✅ Analysis complete! Results saved to {args.output}/")
    print(f"📁 Cache directory: {args.cache_dir}/ ({len(mapper.batch_files)} batch files)")
    print(f"📋 Log file: logs/mapper.log")
    print(f"\n📊 Confidence in results: {adequacy['confidence_score']:.1%}")
    print(f"\n🤖 AI AGENT REPORT: {args.output}/ai_agent_report.json")
    print(f"   This file contains comprehensive landscape analysis optimized for AI consumption")
    if args.max_fitness is not None:
        print(f"   Analysis focused on filtered fitness regions only (≤ {args.max_fitness})")
    if adequacy['confidence_score'] < 0.7:
        print(f"⚠️  Consider running with more samples for higher confidence")
    
    print(f"\n💾 Memory usage optimized with batch processing (batch size: {args.batch_size})")
    if args.max_fitness is not None:
        print(f"   Filtered mode: efficiently searches for parameter regions with fitness ≤ {args.max_fitness}")
    else:
        print(f"   You can safely run with 25,000+ samples without memory issues")
    
    # Close file console
    if hasattr(mapper, 'file_console') and hasattr(mapper.file_console, 'file'):
        mapper.file_console.file.close()


if __name__ == '__main__':
    asyncio.run(main())
