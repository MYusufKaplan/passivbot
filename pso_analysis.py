#!/usr/bin/env python3
"""
PSO Analysis Script for Parameter Dimensionality Reduction

This script analyzes PSO history data to identify:
1. Parameter correlations
2. Parameter sensitivity/importance
3. Parameter variance during optimization
4. Recommendations for dimensionality reduction

Usage:
    python pso_analysis.py [--history_file pso_history_data.pkl] [--output_dir analysis_results]
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PSOAnalyzer:
    def __init__(self, history_file="pso_history_data.pkl"):
        """Initialize PSO analyzer with history data"""
        self.history_file = history_file
        self.history_data = None
        self.parameter_names = None
        self.n_dimensions = None
        self.n_iterations = None
        self.n_particles = None
        
        # Analysis results
        self.correlation_matrix = None
        self.sensitivity_scores = None
        self.variance_scores = None
        self.pca_results = None
        self.importance_ranking = None
        
    def load_data(self):
        """Load PSO history data from pickle file"""
        try:
            with open(self.history_file, 'rb') as f:
                self.history_data = pickle.load(f)
            
            # Extract basic info
            self.parameter_names = self.history_data.get('parameter_names', None)
            self.n_iterations = len(self.history_data['iteration'])
            
            if self.n_iterations > 0:
                self.n_particles = self.history_data['positions'][0].shape[0]
                self.n_dimensions = self.history_data['positions'][0].shape[1]
            
            # Generate parameter names if not provided
            if self.parameter_names is None:
                self.parameter_names = self._try_infer_parameter_names()
                if self.parameter_names is None:
                    self.parameter_names = [f"param_{i}" for i in range(self.n_dimensions)]
            
            print(f"✅ Loaded PSO history data:")
            print(f"   📊 Iterations: {self.n_iterations}")
            print(f"   🐝 Particles: {self.n_particles}")
            print(f"   📐 Dimensions: {self.n_dimensions}")
            print(f"   🏷️ Parameter names: {len(self.parameter_names)} provided")
            
            # Diagnostic: Check data structure
            print(f"\n🔍 Data structure diagnostic:")
            print(f"   Keys in history_data: {list(self.history_data.keys())}")
            if self.parameter_names and len(self.parameter_names) > 0:
                print(f"   Sample parameter names: {self.parameter_names[:5]}...")
            
            # Check fitness data quality
            self._diagnose_fitness_data()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading history data: {e}")
            return False
    
    def _try_infer_parameter_names(self):
        """Try to infer parameter names from optimization config files"""
        
        # First, try to load parameter names from optimize config files
        config_files = [
            'configs/optimize.json',
            'optimize.json', 
            'config.json',
            'configs/config.json'
        ]
        
        for config_file in config_files:
            try:
                if os.path.exists(config_file):
                    print(f"   🔍 Found config file: {config_file}")
                    import json
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    # Extract parameter names from bounds section
                    if 'optimize' in config and 'bounds' in config['optimize']:
                        bounds = config['optimize']['bounds']
                        param_names = list(bounds.keys())
                        
                        # Filter out parameters that have identical bounds (fixed parameters)
                        variable_params = []
                        for param in param_names:
                            bounds_range = bounds[param]
                            if len(bounds_range) == 2 and bounds_range[0] != bounds_range[1]:
                                variable_params.append(param)
                        
                        if len(variable_params) == self.n_dimensions:
                            print(f"   ✅ Extracted {len(variable_params)} parameter names from {config_file}")
                            print(f"   📝 Sample parameters: {variable_params[:3]}...")
                            return variable_params
                        elif len(param_names) == self.n_dimensions:
                            print(f"   ✅ Extracted {len(param_names)} parameter names from {config_file} (including fixed)")
                            print(f"   📝 Sample parameters: {param_names[:3]}...")
                            return param_names
                        else:
                            print(f"   ⚠️ Parameter count mismatch: config has {len(variable_params)} variable params, PSO has {self.n_dimensions} dimensions")
                    
            except Exception as e:
                print(f"   ⚠️ Could not read {config_file}: {e}")
                continue
        
        # Check if we have any hints in the PSO history data structure
        if hasattr(self, 'history_data') and self.history_data:
            # Look for any parameter-related keys
            for key in self.history_data.keys():
                if 'param' in key.lower() and 'name' in key.lower():
                    param_names = self.history_data.get(key)
                    if param_names and len(param_names) == self.n_dimensions:
                        print(f"   🔍 Found parameter names in PSO history key '{key}': {param_names[:3]}...")
                        return param_names
        
        # Fallback to common parameter name patterns
        common_patterns = [
            # Trading bot parameters (passivbot style)
            ['long_close_grid_markup_range', 'long_close_grid_min_markup', 'long_close_grid_qty_pct', 
             'long_close_trailing_grid_ratio', 'long_close_trailing_qty_pct', 'long_close_trailing_retracement_pct',
             'long_close_trailing_threshold_pct', 'long_ema_span_0', 'long_ema_span_1', 'long_entry_grid_double_down_factor',
             'long_entry_grid_spacing_pct', 'long_entry_grid_spacing_weight', 'long_entry_initial_ema_dist',
             'long_entry_initial_qty_pct', 'long_entry_trailing_grid_ratio', 'long_entry_trailing_retracement_pct',
             'long_entry_trailing_threshold_pct', 'long_filter_relative_volume_clip_pct', 'long_filter_rolling_window',
             'long_total_wallet_exposure_limit', 'long_unstuck_close_pct', 'long_unstuck_ema_dist',
             'long_unstuck_loss_allowance_pct', 'long_unstuck_threshold'],
            
            # Generic optimization parameters
            ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'lambda', 'mu', 'sigma', 'theta', 'phi',
             'omega', 'kappa', 'tau', 'rho', 'xi', 'zeta', 'eta', 'nu', 'chi', 'psi'],
            
            # PSO parameters
            ['c1', 'c2', 'w', 'inertia', 'cognitive', 'social']
        ]
        
        # If we have exactly the right number of dimensions for a common pattern
        for pattern in common_patterns:
            if len(pattern) == self.n_dimensions:
                print(f"   🔍 Inferred parameter names based on dimension count ({self.n_dimensions}): {pattern[:3]}...")
                return pattern
        
        # If we have a subset that matches
        for pattern in common_patterns:
            if self.n_dimensions <= len(pattern):
                inferred_names = pattern[:self.n_dimensions]
                print(f"   🔍 Inferred parameter names (subset of {len(pattern)} pattern): {inferred_names[:3]}...")
                return inferred_names
        
        print(f"   ⚠️ Could not infer parameter names for {self.n_dimensions} dimensions")
        return None

    def _diagnose_fitness_data(self):
        """Diagnose fitness data quality issues"""
        print(f"\n🩺 Fitness data diagnosis:")
        
        # Sample some fitness scores
        sample_fitness = []
        for i in range(min(5, len(self.history_data['fitness_scores']))):
            fitness_iter = self.history_data['fitness_scores'][i]
            sample_fitness.extend(fitness_iter[:5])  # First 5 particles
        
        sample_fitness = np.array(sample_fitness, dtype=np.float64)
        
        # Check for problematic values
        n_finite = np.sum(np.isfinite(sample_fitness))
        n_inf = np.sum(np.isinf(sample_fitness))
        n_nan = np.sum(np.isnan(sample_fitness))
        
        print(f"   Sample fitness values (first 25): {sample_fitness[:10]}")
        print(f"   Finite values: {n_finite}/{len(sample_fitness)}")
        print(f"   Infinite values: {n_inf}/{len(sample_fitness)}")
        print(f"   NaN values: {n_nan}/{len(sample_fitness)}")
        
        if n_finite > 0:
            finite_values = sample_fitness[np.isfinite(sample_fitness)]
            print(f"   Finite range: [{np.min(finite_values):.6f}, {np.max(finite_values):.6f}]")
            print(f"   Finite std: {np.std(finite_values):.6f}")
        
        # Warning about analysis reliability
        if n_inf > n_finite:
            print(f"   ⚠️  WARNING: More infinite than finite fitness values!")
            print(f"   ⚠️  Sensitivity analysis may be unreliable.")
            print(f"   ⚠️  Consider focusing on convergence/variance metrics instead.")
    
    def analyze_parameter_correlations(self):
        """Analyze correlations between parameters across all particles and iterations"""
        print("\n🔍 Analyzing parameter correlations...")
        
        # Combine all particle positions across all iterations
        all_positions = []
        for positions in self.history_data['positions']:
            all_positions.extend(positions)
        
        positions_array = np.array(all_positions)
        
        # Calculate correlation matrix
        self.correlation_matrix = np.corrcoef(positions_array.T)
        
        # Find highly correlated parameter pairs
        high_correlations = []
        for i in range(self.n_dimensions):
            for j in range(i+1, self.n_dimensions):
                corr = abs(self.correlation_matrix[i, j])
                if corr > 0.7:  # High correlation threshold
                    high_correlations.append({
                        'param1': self.parameter_names[i],
                        'param2': self.parameter_names[j],
                        'correlation': self.correlation_matrix[i, j],
                        'abs_correlation': corr
                    })
        
        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        print(f"   📈 Found {len(high_correlations)} highly correlated parameter pairs (|r| > 0.7)")
        for corr in high_correlations[:10]:  # Show top 10
            print(f"      {corr['param1']} ↔ {corr['param2']}: r = {corr['correlation']:.3f}")
        
        return high_correlations
    
    def _preprocess_fitness_values(self, fitness_values):
        """Preprocess extreme fitness values for meaningful analysis"""
        fitness_array = np.array(fitness_values, dtype=np.float64)
        
        # Remove infinite and NaN values
        finite_mask = np.isfinite(fitness_array)
        finite_fitness = fitness_array[finite_mask]
        
        if len(finite_fitness) == 0:
            return np.zeros_like(fitness_array), finite_mask
        
        # For extremely large values, use log transformation
        # But handle negative values and zeros carefully
        min_positive = np.min(finite_fitness[finite_fitness > 0]) if np.any(finite_fitness > 0) else 1e-10
        
        # Shift all values to be positive if needed
        if np.min(finite_fitness) <= 0:
            shifted_fitness = finite_fitness - np.min(finite_fitness) + min_positive
        else:
            shifted_fitness = finite_fitness
        
        # Apply log transformation to compress the range
        log_fitness = np.log10(shifted_fitness)
        
        # Create output array
        processed_fitness = np.full_like(fitness_array, np.nan)
        processed_fitness[finite_mask] = log_fitness
        
        return processed_fitness, finite_mask

    def analyze_parameter_sensitivity(self):
        """Analyze parameter sensitivity to fitness changes (with extreme value handling)"""
        print("\n🎯 Analyzing parameter sensitivity...")
        
        sensitivity_scores = []
        
        for dim in range(self.n_dimensions):
            param_changes = []
            fitness_changes = []
            
            # Analyze changes across iterations
            for i in range(1, self.n_iterations):
                prev_positions = self.history_data['positions'][i-1]
                curr_positions = self.history_data['positions'][i]
                prev_fitness = self.history_data['fitness_scores'][i-1]
                curr_fitness = self.history_data['fitness_scores'][i]
                
                # Handle varying particle counts across iterations
                min_particles = min(len(prev_positions), len(curr_positions), 
                                  len(prev_fitness), len(curr_fitness))
                
                if min_particles == 0:
                    continue
                
                # Truncate to minimum size to ensure consistent shapes
                prev_positions = prev_positions[:min_particles]
                curr_positions = curr_positions[:min_particles]
                prev_fitness = prev_fitness[:min_particles]
                curr_fitness = curr_fitness[:min_particles]
                
                # Preprocess fitness values to handle extreme ranges
                prev_fitness_processed, prev_mask = self._preprocess_fitness_values(prev_fitness)
                curr_fitness_processed, curr_mask = self._preprocess_fitness_values(curr_fitness)
                
                # Only use particles where both fitness values are finite
                valid_mask = prev_mask & curr_mask
                
                if np.sum(valid_mask) > 0:
                    # Calculate parameter changes and fitness changes for valid particles
                    param_change = np.abs(curr_positions[valid_mask, dim] - prev_positions[valid_mask, dim])
                    fitness_change = np.abs(curr_fitness_processed[valid_mask] - prev_fitness_processed[valid_mask])
                    
                    param_changes.extend(param_change)
                    fitness_changes.extend(fitness_change)
            
            # Calculate correlation between parameter changes and fitness changes
            if len(param_changes) > 10:  # Need sufficient data
                try:
                    # Convert to numpy arrays and ensure numeric type
                    param_changes_array = np.array(param_changes, dtype=np.float64)
                    fitness_changes_array = np.array(fitness_changes, dtype=np.float64)
                    
                    # Remove any NaN or infinite values
                    valid_mask = np.isfinite(param_changes_array) & np.isfinite(fitness_changes_array)
                    param_changes_clean = param_changes_array[valid_mask]
                    fitness_changes_clean = fitness_changes_array[valid_mask]
                    
                    # Additional filtering for extreme fitness values that might cause issues
                    if len(fitness_changes_clean) > 0:
                        fitness_q99 = np.percentile(fitness_changes_clean, 99)
                        fitness_q01 = np.percentile(fitness_changes_clean, 1)
                        reasonable_mask = (fitness_changes_clean <= fitness_q99) & (fitness_changes_clean >= fitness_q01)
                        param_changes_clean = param_changes_clean[reasonable_mask]
                        fitness_changes_clean = fitness_changes_clean[reasonable_mask]
                    
                    if len(param_changes_clean) > 10:
                        param_std = np.std(param_changes_clean)
                        fitness_std = np.std(fitness_changes_clean)
                        
                        # Debug info for first few parameters
                        if dim < 3:
                            print(f"   Debug {self.parameter_names[dim]}: n={len(param_changes_clean)}, param_std={param_std:.6f}, fitness_std={fitness_std:.6f} (log-transformed)")
                        
                        if param_std > 1e-10 and fitness_std > 1e-10:
                            # Try Spearman correlation as backup if Pearson fails
                            try:
                                correlation, p_value = pearsonr(param_changes_clean, fitness_changes_clean)
                                if np.isnan(correlation):
                                    correlation, p_value = spearmanr(param_changes_clean, fitness_changes_clean)
                                sensitivity = abs(correlation) if not np.isnan(correlation) else 0
                            except Exception as inner_e:
                                # Fallback to simple variance-based sensitivity
                                sensitivity = param_std / (param_std + fitness_std) if (param_std + fitness_std) > 0 else 0
                        else:
                            sensitivity = 0
                    else:
                        sensitivity = 0
                except Exception as e:
                    print(f"   ⚠️ Warning: Could not calculate sensitivity for {self.parameter_names[dim]}: {e}")
                    sensitivity = 0
            else:
                sensitivity = 0
            
            sensitivity_scores.append({
                'parameter': self.parameter_names[dim],
                'sensitivity': sensitivity,
                'dimension': dim
            })
        
        # Sort by sensitivity
        sensitivity_scores.sort(key=lambda x: x['sensitivity'], reverse=True)
        self.sensitivity_scores = sensitivity_scores
        
        print(f"   🎯 Parameter sensitivity ranking (top 10):")
        for i, score in enumerate(sensitivity_scores[:10]):
            print(f"      {i+1:2d}. {score['parameter']}: {score['sensitivity']:.4f}")
        
        return sensitivity_scores
    
    def analyze_parameter_variance(self):
        """Analyze parameter variance during optimization (normalized by range)"""
        print("\n📊 Analyzing parameter variance...")
        
        variance_scores = []
        
        for dim in range(self.n_dimensions):
            all_values = []
            
            # Collect all values for this parameter across iterations and particles
            for positions in self.history_data['positions']:
                all_values.extend(positions[:, dim])
            
            all_values = np.array(all_values)
            
            # Calculate raw variance and range
            variance = np.var(all_values)
            std_dev = np.std(all_values)
            param_range = np.max(all_values) - np.min(all_values)
            
            # Normalize variance by range squared (coefficient of variation squared)
            # This gives us a scale-independent measure of variability
            if param_range > 1e-10:
                normalized_variance = variance / (param_range ** 2)
                # Alternative: coefficient of variation
                mean_val = np.mean(all_values)
                if abs(mean_val) > 1e-10:
                    coeff_variation = std_dev / abs(mean_val)
                else:
                    coeff_variation = 0
            else:
                normalized_variance = 0
                coeff_variation = 0
            
            variance_scores.append({
                'parameter': self.parameter_names[dim],
                'variance': variance,
                'normalized_variance': normalized_variance,
                'coeff_variation': coeff_variation,
                'std_dev': std_dev,
                'range': param_range,
                'dimension': dim
            })
        
        # Sort by normalized variance (scale-independent)
        variance_scores.sort(key=lambda x: x['normalized_variance'], reverse=True)
        self.variance_scores = variance_scores
        
        print(f"   📊 Parameter variance ranking (normalized, top 10):")
        for i, score in enumerate(variance_scores[:10]):
            print(f"      {i+1:2d}. {score['parameter']}: norm_σ² = {score['normalized_variance']:.6f} "
                  f"(CV = {score['coeff_variation']:.4f}, range = {score['range']:.4f})")
        
        return variance_scores
    
    def perform_pca_analysis(self):
        """Perform Principal Component Analysis"""
        print("\n🔬 Performing PCA analysis...")
        
        # Combine all particle positions
        all_positions = []
        for positions in self.history_data['positions']:
            all_positions.extend(positions)
        
        positions_array = np.array(all_positions)
        
        # Standardize the data
        scaler = StandardScaler()
        positions_scaled = scaler.fit_transform(positions_array)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(positions_scaled)
        
        # Calculate cumulative explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Find number of components for different variance thresholds
        components_80 = np.argmax(cumulative_variance >= 0.80) + 1
        components_90 = np.argmax(cumulative_variance >= 0.90) + 1
        components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        self.pca_results = {
            'pca': pca,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': cumulative_variance,
            'components_80': components_80,
            'components_90': components_90,
            'components_95': components_95,
            'scaler': scaler
        }
        
        print(f"   🔬 PCA Results:")
        print(f"      📐 Original dimensions: {self.n_dimensions}")
        print(f"      📊 Components for 80% variance: {components_80}")
        print(f"      📊 Components for 90% variance: {components_90}")
        print(f"      📊 Components for 95% variance: {components_95}")
        print(f"      📈 Top 5 component variance ratios: {pca.explained_variance_ratio_[:5]}")
        
        return self.pca_results
    
    def analyze_convergence_behavior(self):
        """Analyze how parameters converge over time"""
        print("\n� Arnalyzing parameter convergence behavior...")
        
        convergence_scores = []
        
        for dim in range(self.n_dimensions):
            # Calculate variance in early vs late iterations
            early_positions = []
            late_positions = []
            
            early_cutoff = max(1, self.n_iterations // 4)  # First 25%
            late_start = max(1, 3 * self.n_iterations // 4)  # Last 25%
            
            # Collect early and late positions
            for i in range(min(early_cutoff, len(self.history_data['positions']))):
                early_positions.extend(self.history_data['positions'][i][:, dim])
            
            for i in range(late_start, len(self.history_data['positions'])):
                late_positions.extend(self.history_data['positions'][i][:, dim])
            
            if len(early_positions) > 0 and len(late_positions) > 0:
                early_var = np.var(early_positions)
                late_var = np.var(late_positions)
                
                # Convergence ratio: how much variance decreased
                if early_var > 1e-10:
                    convergence_ratio = (early_var - late_var) / early_var
                    convergence_ratio = max(0, convergence_ratio)  # Clamp to 0-1
                else:
                    convergence_ratio = 0
                
                # Stability: low variance in late iterations suggests convergence
                param_range = np.max(early_positions + late_positions) - np.min(early_positions + late_positions)
                if param_range > 1e-10:
                    stability_score = 1 - (np.sqrt(late_var) / param_range)
                    stability_score = max(0, min(1, stability_score))
                else:
                    stability_score = 1
            else:
                convergence_ratio = 0
                stability_score = 0
            
            convergence_scores.append({
                'parameter': self.parameter_names[dim],
                'dimension': dim,
                'convergence_ratio': convergence_ratio,
                'stability_score': stability_score,
                'early_var': early_var if 'early_var' in locals() else 0,
                'late_var': late_var if 'late_var' in locals() else 0
            })
        
        return convergence_scores

    def create_importance_ranking(self):
        """Create overall parameter importance ranking with improved interpretation"""
        print("\n🏆 Creating parameter importance ranking...")
        
        # Get convergence analysis
        convergence_scores = self.analyze_convergence_behavior()
        
        # Normalize scores to 0-1 range
        def normalize_scores(scores, key):
            values = [s[key] for s in scores]
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return [0.5 for _ in values]
            return [(v - min_val) / (max_val - min_val) for v in values]
        
        # Create improved importance score
        importance_ranking = []
        for dim in range(self.n_dimensions):
            # Find scores for this dimension
            sens_data = next(s for s in self.sensitivity_scores if s['dimension'] == dim)
            var_data = next(s for s in self.variance_scores if s['dimension'] == dim)
            conv_data = next(s for s in convergence_scores if s['dimension'] == dim)
            
            sensitivity = sens_data['sensitivity']
            normalized_variance = var_data['normalized_variance']
            convergence_ratio = conv_data['convergence_ratio']
            stability_score = conv_data['stability_score']
            
            # Improved importance calculation:
            # 1. High sensitivity = important (affects fitness)
            # 2. Good convergence = important (optimization is working)
            # 3. High stability = important (found good region)
            # 4. Moderate variance = good (being explored but not chaotic)
            
            # Penalize extremely high variance (likely noise)
            variance_penalty = 1.0
            if normalized_variance > 0.1:  # Very high variance might be noise
                variance_penalty = 0.5
            
            # Reward parameters that show good optimization behavior
            optimization_quality = (convergence_ratio + stability_score) / 2
            
            # Combined importance score
            combined_score = (
                0.4 * sensitivity +                    # Direct impact on fitness
                0.3 * optimization_quality +           # Shows good optimization
                0.2 * min(normalized_variance, 0.1) * 10 +  # Moderate exploration (capped)
                0.1 * variance_penalty                 # Penalty for excessive noise
            )
            
            importance_ranking.append({
                'parameter': self.parameter_names[dim],
                'dimension': dim,
                'sensitivity': sensitivity,
                'normalized_variance': normalized_variance,
                'convergence_ratio': convergence_ratio,
                'stability_score': stability_score,
                'optimization_quality': optimization_quality,
                'combined_score': combined_score
            })
        
        # Sort by combined score
        importance_ranking.sort(key=lambda x: x['combined_score'], reverse=True)
        self.importance_ranking = importance_ranking
        
        print(f"   🏆 Parameter importance ranking (improved methodology, top 15):")
        for i, param in enumerate(importance_ranking[:15]):
            print(f"      {i+1:2d}. {param['parameter']}: {param['combined_score']:.4f}")
            print(f"          Sensitivity: {param['sensitivity']:.4f} | "
                  f"Convergence: {param['convergence_ratio']:.3f} | "
                  f"Stability: {param['stability_score']:.3f}")
        
        return importance_ranking
    
    def generate_dimensionality_reduction_recommendations(self):
        """Generate recommendations for dimensionality reduction"""
        print("\n💡 Generating dimensionality reduction recommendations...")
        
        recommendations = {
            'correlation_based': [],
            'importance_based': [],
            'pca_based': [],
            'combined_strategy': []
        }
        
        # 1. Correlation-based reduction
        high_corr = self.analyze_parameter_correlations()
        if high_corr:
            recommendations['correlation_based'] = [
                f"Remove one parameter from highly correlated pairs:",
                *[f"  - Consider removing either '{pair['param1']}' or '{pair['param2']}' (r={pair['correlation']:.3f})" 
                  for pair in high_corr[:5]]
            ]
        
        # 2. Importance-based reduction
        low_importance = [p for p in self.importance_ranking if p['combined_score'] < 0.1]
        if low_importance:
            recommendations['importance_based'] = [
                f"Consider fixing low-importance parameters at reasonable defaults:",
                *[f"  - {param['parameter']}: importance = {param['combined_score']:.4f}" 
                  for param in low_importance[:8]]
            ]
        
        # 3. PCA-based reduction
        recommendations['pca_based'] = [
            f"PCA suggests {self.pca_results['components_80']} components capture 80% of variance",
            f"PCA suggests {self.pca_results['components_90']} components capture 90% of variance",
            f"Original {self.n_dimensions}D → {self.pca_results['components_90']}D reduces dimensions by {self.n_dimensions - self.pca_results['components_90']}"
        ]
        
        # 4. Combined strategy
        keep_top_n = min(15, self.n_dimensions - len(low_importance))
        recommendations['combined_strategy'] = [
            f"Recommended combined approach:",
            f"  1. Keep top {keep_top_n} most important parameters",
            f"  2. Fix {len(low_importance)} low-importance parameters at defaults",
            f"  3. Remove one parameter from each highly correlated pair",
            f"  4. This could reduce {self.n_dimensions}D → ~{keep_top_n - len(high_corr[:3])}D"
        ]
        
        # Print recommendations
        for strategy, recs in recommendations.items():
            print(f"\n📋 {strategy.replace('_', ' ').title()} Reduction:")
            for rec in recs:
                print(f"   {rec}")
        
        return recommendations
    
    def calculate_parameter_ranges(self):
        """Calculate actual parameter ranges used during optimization"""
        print("\n📏 Analyzing parameter ranges...")
        
        parameter_ranges = []
        
        for dim in range(self.n_dimensions):
            all_values = []
            
            # Collect all values for this parameter
            for positions in self.history_data['positions']:
                all_values.extend(positions[:, dim])
            
            min_val = np.min(all_values)
            max_val = np.max(all_values)
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            
            # Calculate how much of the range was actually used
            range_used = max_val - min_val
            
            parameter_ranges.append({
                'parameter': self.parameter_names[dim],
                'dimension': dim,
                'min_used': min_val,
                'max_used': max_val,
                'mean': mean_val,
                'std': std_val,
                'range_used': range_used
            })
        
        return parameter_ranges

    def get_best_fitness_solution(self):
        """Get the parameter values from the best fitness solution"""
        # Combine all positions and fitness scores
        all_positions = []
        all_fitness = []
        
        for i in range(len(self.history_data['positions'])):
            positions = self.history_data['positions'][i]
            fitness_scores = self.history_data['fitness_scores'][i]
            
            all_positions.extend(positions)
            all_fitness.extend(fitness_scores)
        
        all_positions = np.array(all_positions)
        all_fitness = np.array(all_fitness, dtype=np.float64)
        
        # Find best finite fitness solution
        finite_mask = np.isfinite(all_fitness)
        if np.sum(finite_mask) == 0:
            print("   ⚠️ No finite fitness values found, using first solution")
            return all_positions[0]
        
        finite_fitness = all_fitness[finite_mask]
        finite_positions = all_positions[finite_mask]
        
        # Get best solution (minimum fitness)
        best_idx = np.argmin(finite_fitness)
        best_fitness = finite_fitness[best_idx]
        best_params = finite_positions[best_idx]
        
        print(f"   🏆 Best fitness found: {best_fitness:.6e}")
        return best_params

    def get_top_performers(self, percentile=85):
        """Get parameter values from top performing solutions"""
        # Combine all positions and fitness scores
        all_positions = []
        all_fitness = []
        
        for i in range(len(self.history_data['positions'])):
            positions = self.history_data['positions'][i]
            fitness_scores = self.history_data['fitness_scores'][i]
            
            all_positions.extend(positions)
            all_fitness.extend(fitness_scores)
        
        all_positions = np.array(all_positions)
        all_fitness = np.array(all_fitness, dtype=np.float64)
        
        # Filter finite values
        finite_mask = np.isfinite(all_fitness)
        if np.sum(finite_mask) == 0:
            return all_positions[:10]  # Fallback to first 10
        
        finite_fitness = all_fitness[finite_mask]
        finite_positions = all_positions[finite_mask]
        
        # Get top performers (lower fitness is better)
        threshold = np.percentile(finite_fitness, percentile)
        top_mask = finite_fitness <= threshold
        top_performers = finite_positions[top_mask]
        
        print(f"   📈 Found {len(top_performers)} top performers (≤{percentile}th percentile)")
        return top_performers

    def load_original_bounds(self):
        """Load original parameter bounds from config file"""
        config_files = [
            'configs/optimize.json',
            'optimize.json', 
            'config.json',
            'configs/config.json'
        ]
        
        for config_file in config_files:
            try:
                if os.path.exists(config_file):
                    import json
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    if 'optimize' in config and 'bounds' in config['optimize']:
                        bounds = config['optimize']['bounds']
                        param_names = list(bounds.keys())
                        
                        # Filter variable parameters only
                        original_bounds = {}
                        for param in param_names:
                            bounds_range = bounds[param]
                            if len(bounds_range) == 2 and bounds_range[0] != bounds_range[1]:
                                original_bounds[param] = bounds_range
                        
                        if len(original_bounds) == self.n_dimensions:
                            print(f"   📋 Loaded original bounds for {len(original_bounds)} parameters")
                            return original_bounds
                        
            except Exception as e:
                continue
        
        print("   ⚠️ Could not load original bounds from config")
        return None

    def generate_actionable_recommendations(self):
        """Generate exploration-focused, proportional range reduction recommendations"""
        print("\n" + "="*80)
        print("🎯 EXPLORATION-FOCUSED PARAMETER OPTIMIZATION STRATEGY")
        print("="*80)
        
        # Get best solution and top performers
        best_params = self.get_best_fitness_solution()
        top_performers = self.get_top_performers(percentile=85)  # Top 15%
        original_bounds = self.load_original_bounds()
        
        # Get parameter ranges
        param_ranges = self.calculate_parameter_ranges()
        
        # Combine all data for comprehensive analysis
        comprehensive_analysis = []
        
        for dim in range(self.n_dimensions):
            param_name = self.parameter_names[dim]
            
            # Find data for this parameter
            importance_data = next(p for p in self.importance_ranking if p['dimension'] == dim)
            sensitivity_data = next(p for p in self.sensitivity_scores if p['dimension'] == dim)
            variance_data = next(p for p in self.variance_scores if p['dimension'] == dim)
            range_data = next(p for p in param_ranges if p['dimension'] == dim)
            
            # Get original bounds if available
            original_min = original_bounds[param_name][0] if original_bounds and param_name in original_bounds else range_data['min_used']
            original_max = original_bounds[param_name][1] if original_bounds and param_name in original_bounds else range_data['max_used']
            original_range = original_max - original_min
            
            comprehensive_analysis.append({
                'parameter': param_name,
                'dimension': dim,
                'importance_score': importance_data['combined_score'],
                'sensitivity': sensitivity_data['sensitivity'],
                'variance': variance_data['variance'],
                'min_used': range_data['min_used'],
                'max_used': range_data['max_used'],
                'mean': range_data['mean'],
                'std': range_data['std'],
                'range_used': range_data['range_used'],
                'original_min': original_min,
                'original_max': original_max,
                'original_range': original_range,
                'best_value': best_params[dim]
            })
        
        # Sort by importance score
        comprehensive_analysis.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Dynamic thresholds based on score distribution
        scores = [p['importance_score'] for p in comprehensive_analysis]
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        
        # Use percentile-based thresholds for better differentiation
        high_threshold = np.percentile(scores, 75)  # Top 25%
        low_threshold = np.percentile(scores, 25)   # Bottom 25%
        
        print(f"\n📊 Score distribution: mean={score_mean:.3f}, std={score_std:.3f}")
        print(f"   Thresholds: high>{high_threshold:.3f}, low<{low_threshold:.3f}")
        
        # Categorize parameters using dynamic thresholds
        high_impact = [p for p in comprehensive_analysis if p['importance_score'] > high_threshold]
        medium_impact = [p for p in comprehensive_analysis if low_threshold <= p['importance_score'] <= high_threshold]
        low_impact = [p for p in comprehensive_analysis if p['importance_score'] < low_threshold]
        
        # Proportional reduction factors (exploration-focused)
        reduction_factors = {
            'high': 0.90,      # Keep 90% - maximum exploration where it matters
            'medium': 0.60,    # Keep 60% - balanced approach  
            'low': 0.25        # Keep 25% - focused but not fixed
        }
        
        print(f"\n🌍 EXPLORATION-FOCUSED RANGE RECOMMENDATIONS:")
        print(f"   Reduction factors: High={reduction_factors['high']:.0%}, Medium={reduction_factors['medium']:.0%}, Low={reduction_factors['low']:.0%}")
        
        # Calculate and display recommendations
        all_recommendations = []
        
        for category, params, factor in [
            ('HIGH IMPACT', high_impact, reduction_factors['high']),
            ('MEDIUM IMPACT', medium_impact, reduction_factors['medium']),
            ('LOW IMPACT', low_impact, reduction_factors['low'])
        ]:
            
            if not params:
                continue
                
            icon = "🔥" if category == "HIGH IMPACT" else "⚡" if category == "MEDIUM IMPACT" else "🎯"
            action = "Maximum exploration" if category == "HIGH IMPACT" else "Balanced exploration" if category == "MEDIUM IMPACT" else "Focused exploration"
            
            print(f"\n{icon} {category} PARAMETERS ({len(params)}) - {action}:")
            print("-" * 80)
            
            for i, param in enumerate(params):
                dim = param['dimension']
                param_name = param['parameter']
                best_value = param['best_value']
                original_range = param['original_range']
                
                # Get top performer values for this parameter
                top_values = top_performers[:, dim]
                
                # Calculate proportional exploration buffer
                exploration_buffer = factor * original_range / 2
                
                # Center around best solution + top performers
                center_candidates = list(top_values) + [best_value]
                range_center = np.mean(center_candidates)
                
                # Create proportional range
                suggested_min = range_center - exploration_buffer
                suggested_max = range_center + exploration_buffer
                
                # Ensure best solution is always included with safety margin
                safety_margin = 0.1 * exploration_buffer
                suggested_min = min(suggested_min, best_value - safety_margin)
                suggested_max = max(suggested_max, best_value + safety_margin)
                
                # Clamp to original bounds
                suggested_min = max(suggested_min, param['original_min'])
                suggested_max = min(suggested_max, param['original_max'])
                
                # Calculate efficiency metrics
                new_range = suggested_max - suggested_min
                range_reduction = (original_range - new_range) / original_range * 100
                
                print(f"{i+1:2d}. {param_name:<40} | Impact: {param['importance_score']:.3f}")
                print(f"    Original range:  [{param['original_min']:.6f}, {param['original_max']:.6f}] (size: {original_range:.6f})")
                print(f"    Best value:      {best_value:.6f}")
                print(f"    Suggested range: [{suggested_min:.6f}, {suggested_max:.6f}] (size: {new_range:.6f})")
                print(f"    Reduction:       {range_reduction:.1f}% | Exploration factor: {factor:.0%}")
                
                all_recommendations.append({
                    'parameter': param_name,
                    'category': category.lower().replace(' ', '_'),
                    'original_min': param['original_min'],
                    'original_max': param['original_max'],
                    'suggested_min': suggested_min,
                    'suggested_max': suggested_max,
                    'best_value': best_value,
                    'reduction_factor': factor,
                    'range_reduction_pct': range_reduction
                })
        
        # Calculate overall efficiency gains
        total_original_space = 1.0
        total_reduced_space = 1.0
        
        for rec in all_recommendations:
            original_size = rec['original_max'] - rec['original_min']
            reduced_size = rec['suggested_max'] - rec['suggested_min']
            if original_size > 0:
                total_original_space *= original_size
                total_reduced_space *= reduced_size
        
        if total_original_space > 0:
            space_reduction = (1 - total_reduced_space / total_original_space) * 100
        else:
            space_reduction = 0
        
        # Summary recommendations
        print(f"\n📋 EXPLORATION-FOCUSED OPTIMIZATION SUMMARY:")
        print("-" * 60)
        print(f"🔥 High-impact parameters:   {len(high_impact)} (90% range retention)")
        print(f"⚡ Medium-impact parameters: {len(medium_impact)} (60% range retention)")
        print(f"🎯 Low-impact parameters:    {len(low_impact)} (25% range retention)")
        print(f"📉 Total dimensions:         {self.n_dimensions}D (no dimension removal)")
        print(f"🚀 Search space reduction:   {space_reduction:.1f}%")
        print(f"🌍 Strategy:                 Exploration-focused with proportional reduction")
        
        print(f"\n💡 KEY BENEFITS:")
        print(f"   ✅ All parameters retain exploration capability")
        print(f"   ✅ Best solution always within new bounds")
        print(f"   ✅ Proportional reduction ensures fairness")
        print(f"   ✅ Massive efficiency gains while preserving discovery potential")
        print(f"   ✅ Reduced risk of local minima compared to fixed parameters")
        
        # Generate bounds comparison and output file
        self.generate_bounds_comparison(all_recommendations)
        
        return {
            'high_impact': high_impact,
            'medium_impact': medium_impact,
            'low_impact': low_impact,
            'recommendations': all_recommendations,
            'space_reduction_pct': space_reduction,
            'best_params': best_params
        }

    def generate_bounds_comparison(self, recommendations):
        """Compare suggested bounds with normalize bounds and generate encapsulating bounds file"""
        print("\n" + "="*80)
        print("📊 BOUNDS COMPARISON AND GENERATION")
        print("="*80)
        
        # Load normalize bounds
        normalize_bounds_file = 'configs/optimize_normalBounds.json'
        normalize_bounds = {}
        
        try:
            import json
            with open(normalize_bounds_file, 'r') as f:
                normalize_config = json.load(f)
                normalize_bounds = normalize_config.get('optimize', {}).get('bounds', {})
            print(f"✅ Loaded normalize bounds from {normalize_bounds_file}")
        except Exception as e:
            print(f"⚠️ Could not load normalize bounds: {e}")
            print("   Proceeding with suggested bounds only...")
        
        # Load current bounds from optimize.json
        current_bounds_file = 'configs/optimize.json'
        current_bounds = {}
        
        try:
            with open(current_bounds_file, 'r') as f:
                current_config = json.load(f)
                current_bounds = current_config.get('optimize', {}).get('bounds', {})
            print(f"✅ Loaded current bounds from {current_bounds_file}")
        except Exception as e:
            print(f"⚠️ Could not load current bounds: {e}")
            print("   Proceeding without current bounds comparison...")
        
        # Create ordered encapsulating bounds following current_bounds order
        encapsulating_bounds = {}
        comparison_data = []
        
        # Create lookup dictionaries for recommendations
        recommendations_dict = {rec['parameter']: rec for rec in recommendations}
        
        # Process parameters in the order they appear in current_bounds
        for param_name in current_bounds.keys():
            current_min, current_max = current_bounds[param_name]
            
            if param_name.startswith('short_'):
                # For short parameters, use normalize bounds (fixed values)
                if param_name in normalize_bounds:
                    encapsulating_bounds[param_name] = normalize_bounds[param_name]
                else:
                    encapsulating_bounds[param_name] = [current_min, current_max]
                    
            elif param_name.startswith('long_'):
                # For long parameters, calculate encapsulating bounds
                if param_name in recommendations_dict:
                    rec = recommendations_dict[param_name]
                    suggested_min = rec['suggested_min']
                    suggested_max = rec['suggested_max']
                    
                    # Get normalize bounds if available
                    if param_name in normalize_bounds:
                        normalize_min, normalize_max = normalize_bounds[param_name]
                        
                        # Calculate encapsulating bounds (smallest range that contains both)
                        encap_min = min(suggested_min, normalize_min)
                        encap_max = max(suggested_max, normalize_max)
                        
                        comparison_data.append({
                            'parameter': param_name,
                            'suggested_min': suggested_min,
                            'suggested_max': suggested_max,
                            'current_min': current_min,
                            'current_max': current_max,
                            'normalize_min': normalize_min,
                            'normalize_max': normalize_max,
                            'encap_min': encap_min,
                            'encap_max': encap_max,
                            'suggested_range': suggested_max - suggested_min,
                            'current_range': current_max - current_min,
                            'normalize_range': normalize_max - normalize_min,
                            'encap_range': encap_max - encap_min
                        })
                        
                        encapsulating_bounds[param_name] = [encap_min, encap_max]
                        
                    else:
                        # No normalize bounds available, use suggested bounds
                        encapsulating_bounds[param_name] = [suggested_min, suggested_max]
                        comparison_data.append({
                            'parameter': param_name,
                            'suggested_min': suggested_min,
                            'suggested_max': suggested_max,
                            'current_min': current_min,
                            'current_max': current_max,
                            'normalize_min': None,
                            'normalize_max': None,
                            'encap_min': suggested_min,
                            'encap_max': suggested_max,
                            'suggested_range': suggested_max - suggested_min,
                            'current_range': current_max - current_min,
                            'normalize_range': None,
                            'encap_range': suggested_max - suggested_min
                        })
                else:
                    # Long parameter not in recommendations, use normalize bounds or current bounds
                    if param_name in normalize_bounds:
                        encapsulating_bounds[param_name] = normalize_bounds[param_name]
                        print(f"   ℹ️ Added missing long_* parameter from normalize bounds: {param_name}")
                    else:
                        encapsulating_bounds[param_name] = [current_min, current_max]
                        print(f"   ℹ️ Added missing long_* parameter from current bounds: {param_name}")
        
        # Add any parameters that exist in normalize_bounds but not in current_bounds
        for param_name, bounds in normalize_bounds.items():
            if param_name not in encapsulating_bounds:
                encapsulating_bounds[param_name] = bounds
                print(f"   ℹ️ Added parameter from normalize bounds not in current: {param_name}")
        
        # Display comparison table
        print(f"\n📋 BOUNDS COMPARISON TABLE (long_* parameters only):")
        print("-" * 150)
        print(f"{'Parameter':<35} {'Suggested Range':<25} {'Current Range':<25} {'Normalize Range':<25} {'Encapsulating Range':<25}")
        print("-" * 150)
        
        for data in comparison_data:
            param = data['parameter']
            
            # Format suggested range
            sugg_range = f"[{data['suggested_min']:.6f}, {data['suggested_max']:.6f}]"
            
            # Format current range
            if data['current_min'] is not None and data['current_max'] is not None:
                curr_range = f"[{data['current_min']:.6f}, {data['current_max']:.6f}]"
            else:
                curr_range = "N/A"
            
            # Format normalize range
            if data['normalize_min'] is not None:
                norm_range = f"[{data['normalize_min']:.6f}, {data['normalize_max']:.6f}]"
            else:
                norm_range = "N/A"
            
            # Format encapsulating range
            encap_range = f"[{data['encap_min']:.6f}, {data['encap_max']:.6f}]"
            
            print(f"{param:<35} {sugg_range:<25} {curr_range:<25} {norm_range:<25} {encap_range:<25}")
        
        # Count parameters by type
        long_params = [k for k in encapsulating_bounds.keys() if k.startswith('long_')]
        short_params = [k for k in encapsulating_bounds.keys() if k.startswith('short_')]
        
        # Create ordered bounds dictionary following the order from current bounds
        ordered_bounds = {}
        
        # First, add parameters in the order they appear in current_bounds
        for param_name in current_bounds.keys():
            if param_name in encapsulating_bounds:
                ordered_bounds[param_name] = encapsulating_bounds[param_name]
        
        # Then add any remaining parameters that weren't in current_bounds
        for param_name, bounds in encapsulating_bounds.items():
            if param_name not in ordered_bounds:
                ordered_bounds[param_name] = bounds
        
        # Generate bounds.json file with proper structure and ordering
        bounds_output = encapsulating_bounds
        
        output_file = 'bounds.json'
        try:
            with open(output_file, 'w') as f:
                json.dump(bounds_output, f, indent=2)
            
            print(f"\n✅ Generated encapsulating bounds file: {output_file}")
            print(f"   📊 Contains {len(long_params)} long_* parameters (optimized)")
            print(f"   📊 Contains {len(short_params)} short_* parameters (fixed)")
            print(f"   📊 Total parameters: {len(encapsulating_bounds)}")
            print(f"   🔄 Parameter order: Preserved from configs/optimize.json")
            
            # Calculate space efficiency
            total_suggested_space = 1.0
            total_normalize_space = 1.0
            total_encap_space = 1.0
            
            params_with_both = [d for d in comparison_data if d['normalize_min'] is not None]
            
            if params_with_both:
                for data in params_with_both:
                    total_suggested_space *= data['suggested_range']
                    total_normalize_space *= data['normalize_range']
                    total_encap_space *= data['encap_range']
                
                if total_normalize_space > 0:
                    efficiency_vs_normalize = (1 - total_encap_space / total_normalize_space) * 100
                    expansion_vs_suggested = (total_encap_space / total_suggested_space - 1) * 100
                    
                    print(f"\n📈 SPACE EFFICIENCY METRICS (long_* parameters only):")
                    print(f"   🎯 Encapsulating vs Normalize: {efficiency_vs_normalize:.1f}% reduction")
                    print(f"   📏 Encapsulating vs Suggested: {expansion_vs_suggested:.1f}% expansion")
                    print(f"   ⚖️  Balance: Maintains exploration while improving efficiency")
                    print(f"   🔒 Short parameters: Fixed at normalize bounds (no optimization)")
            
        except Exception as e:
            print(f"❌ Error generating bounds file: {e}")
        
        return encapsulating_bounds, comparison_data

    def _assess_analysis_reliability(self):
        """Assess and warn about analysis reliability issues"""
        print(f"\n🔍 ANALYSIS RELIABILITY ASSESSMENT:")
        print("-" * 50)
        
        warnings = []
        
        # Check sensitivity scores
        max_sensitivity = max(s['sensitivity'] for s in self.sensitivity_scores)
        avg_sensitivity = np.mean([s['sensitivity'] for s in self.sensitivity_scores])
        
        if max_sensitivity < 0.05:
            warnings.append("⚠️  Very low sensitivity scores suggest weak parameter-fitness relationships")
            warnings.append("   → Fitness data may be noisy or parameters may not significantly affect outcomes")
        
        if avg_sensitivity < 0.01:
            warnings.append("⚠️  Extremely low average sensitivity - consider focusing on convergence metrics only")
        
        # Check if most importance comes from convergence rather than sensitivity
        high_conv_low_sens = 0
        for param in self.importance_ranking[:10]:
            if param['optimization_quality'] > 0.7 and param['sensitivity'] < 0.01:
                high_conv_low_sens += 1
        
        if high_conv_low_sens > 7:
            warnings.append("⚠️  Most 'important' parameters show convergence but not fitness sensitivity")
            warnings.append("   → This suggests parameters converged to similar values, not that they're important")
        
        # Check score clustering
        scores = [p['combined_score'] for p in self.importance_ranking]
        score_range = max(scores) - min(scores)
        if score_range < 0.1:
            warnings.append("⚠️  All parameters have very similar importance scores")
            warnings.append("   → Difficulty distinguishing truly important parameters")
        
        if warnings:
            print("❌ RELIABILITY CONCERNS:")
            for warning in warnings:
                print(f"   {warning}")
            
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   • Focus on parameters with highest convergence ratios")
            print(f"   • Consider that most parameters may be equally (un)important")
            print(f"   • Use variance analysis to identify parameters that were actively explored")
            print(f"   • Consider fixing parameters that converged early and stayed stable")
        else:
            print("✅ Analysis appears reliable - sensitivity and convergence metrics align well")
        
        return len(warnings) == 0

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting PSO Parameter Analysis...")
        
        if not self.load_data():
            return False
        
        # Run core analyses (simplified output)
        print("\n🔍 Running correlation analysis...")
        high_corr = self.analyze_parameter_correlations()
        
        print("\n🎯 Running sensitivity analysis...")
        self.analyze_parameter_sensitivity()
        
        print("\n📊 Running variance analysis...")
        self.analyze_parameter_variance()
        
        print("\n🏆 Creating importance ranking...")
        self.create_importance_ranking()
        
        # Assess reliability before making recommendations
        is_reliable = self._assess_analysis_reliability()
        
        # Generate actionable recommendations
        recommendations = self.generate_actionable_recommendations()
        
        # Show highly correlated parameters
        if high_corr:
            print(f"\n🔗 HIGHLY CORRELATED PARAMETERS:")
            print("-" * 50)
            for corr in high_corr[:5]:
                print(f"   {corr['param1']} ↔ {corr['param2']}: r = {corr['correlation']:.3f}")
                print(f"   💡 Consider removing one of these parameters")
        
        if not is_reliable:
            print(f"\n⚠️  IMPORTANT: Due to reliability concerns, use these recommendations cautiously.")
            print(f"   Consider focusing on parameters that show clear convergence patterns.")
        
        print("\n✅ Analysis complete!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Analyze PSO history data for dimensionality reduction')
    parser.add_argument('--history_file', default='pso_history_data.pkl',
                       help='Path to PSO history data file or pattern (e.g., pso_history_*.pkl)')
    parser.add_argument('--merge_files', action='store_true',
                       help='Merge multiple history files if pattern is provided')
    
    args = parser.parse_args()
    
    # Handle multiple files if pattern is provided
    if '*' in args.history_file or args.merge_files:
        import glob
        history_files = glob.glob(args.history_file)
        if len(history_files) > 1:
            print(f"🔍 Found {len(history_files)} history files to merge:")
            for f in history_files:
                print(f"   - {f}")
            
            # Merge multiple history files
            merged_file = "merged_pso_history.pkl"
            merge_history_files(history_files, merged_file)
            args.history_file = merged_file
    
    # Run analysis
    analyzer = PSOAnalyzer(args.history_file)
    success = analyzer.run_full_analysis()
    
    if success:
        print(f"\n🎉 Analysis completed successfully!")
    else:
        print(f"\n❌ Analysis failed. Check your history data file.")

def merge_history_files(file_list, output_file):
    """Merge multiple PSO history files into one"""
    print(f"🔄 Merging {len(file_list)} history files...")
    
    merged_data = {
        'positions': [],
        'fitness_scores': [],
        'iteration': [],
        'global_best_pos': [],
        'global_best_cost': [],
        'parameter_names': None
    }
    
    iteration_offset = 0
    
    for i, file_path in enumerate(sorted(file_list)):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"   📁 Loading {file_path}: {len(data.get('iteration', []))} iterations")
            
            # Merge data with iteration offset
            for key in ['positions', 'fitness_scores', 'global_best_pos', 'global_best_cost']:
                if key in data:
                    merged_data[key].extend(data[key])
            
            # Handle iterations with offset to avoid conflicts
            if 'iteration' in data:
                offset_iterations = [it + iteration_offset for it in data['iteration']]
                merged_data['iteration'].extend(offset_iterations)
                iteration_offset = max(offset_iterations) + 1
            
            # Use parameter names from first file that has them
            if merged_data['parameter_names'] is None and data.get('parameter_names'):
                merged_data['parameter_names'] = data['parameter_names']
                
        except Exception as e:
            print(f"   ⚠️ Failed to load {file_path}: {e}")
    
    # Save merged data
    with open(output_file, 'wb') as f:
        pickle.dump(merged_data, f)
    
    total_iterations = len(merged_data['iteration'])
    print(f"✅ Merged data saved to {output_file}: {total_iterations} total iterations")

if __name__ == "__main__":
    main()