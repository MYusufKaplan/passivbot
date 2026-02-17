"""
History tracking and statistics for DEAP optimizer.

This module provides functionality for tracking detailed optimization history
and computing evolutionary statistics.
"""

import os
import pickle
import logging
import numpy as np


class HistoryTracker:
    """
    Track detailed optimization history for analysis.
    
    Attributes
    ----------
    history_data : dict
        Dictionary containing history data:
        - generation: List of generation numbers
        - population_positions: List of population position arrays
        - fitness_scores: List of fitness score arrays
        - best_individual: List of best individuals per generation
        - best_fitness: List of best fitness values per generation
        - statistics: List of statistics dictionaries
        - parameter_names: List of parameter names
        - generation_times: List of generation times
        - improvement_rates: List of improvement rates
        - convergence_metrics: List of convergence metrics
        - diversity_metrics: List of diversity metrics
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    
    def __init__(self, parameter_names=None):
        """
        Initialize history tracker.
        
        Parameters
        ----------
        parameter_names : list of str, optional
            Names of parameters being optimized
        """
        self.history_data = {
            'generation': [],
            'population_positions': [],
            'fitness_scores': [],
            'best_individual': [],
            'best_fitness': [],
            'statistics': [],
            'parameter_names': parameter_names,
            'generation_times': [],
            'improvement_rates': [],
            'convergence_metrics': [],
            'diversity_metrics': []
        }
        self.in_memory = True
        self.file_path = None
    
    def set_file_storage(self, file_path):
        """
        Enable file-based storage.
        
        Parameters
        ----------
        file_path : str
            Path to save history data
        
        Requirements: 4.4
        """
        self.file_path = file_path
        self.in_memory = False
    
    def record_generation(self, generation, population, fitness_scores, 
                         best_individual, best_fitness, generation_time):
        """
        Record data for a generation.
        
        Parameters
        ----------
        generation : int
            Generation number
        population : list
            List of individuals
        fitness_scores : array-like
            Fitness scores for population
        best_individual : Individual
            Best individual in generation
        best_fitness : float
            Best fitness value
        generation_time : float
            Time taken for generation
        
        Requirements: 4.1, 4.2, 4.3
        """
        self.history_data['generation'].append(generation)
        self.history_data['population_positions'].append(np.array([list(ind) for ind in population]))
        self.history_data['fitness_scores'].append(np.array(fitness_scores))
        self.history_data['best_individual'].append(list(best_individual))
        self.history_data['best_fitness'].append(best_fitness)
        self.history_data['generation_times'].append(generation_time)
        
        # Calculate improvement rate
        if len(self.history_data['best_fitness']) > 1:
            prev_fitness = self.history_data['best_fitness'][-2]
            if prev_fitness != 0:
                improvement_rate = (prev_fitness - best_fitness) / abs(prev_fitness)
            else:
                improvement_rate = 0.0
        else:
            improvement_rate = 0.0
        self.history_data['improvement_rates'].append(improvement_rate)
        
        # Calculate diversity metric (population spread)
        if len(population) > 1:
            positions = np.array([list(ind) for ind in population])
            diversity = np.mean(np.std(positions, axis=0))
        else:
            diversity = 0.0
        self.history_data['diversity_metrics'].append(diversity)
        
        # Calculate convergence metric (fitness variance)
        if len(fitness_scores) > 1:
            convergence = np.std(fitness_scores)
        else:
            convergence = 0.0
        self.history_data['convergence_metrics'].append({'fitness_std': convergence})
        
        # Save to file if file-based storage is enabled
        if not self.in_memory and self.file_path:
            self.save_to_file()
    
    def save_to_file(self):
        """
        Save history data to file.
        
        Requirements: 4.2, 4.4
        """
        if not self.file_path:
            return
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            with open(self.file_path, 'wb') as f:
                pickle.dump(self.history_data, f)
            
            logging.debug(f"History data saved to {self.file_path}")
        except Exception as e:
            logging.error(f"Failed to save history data: {e}")
    
    def load_from_file(self, file_path):
        """
        Load history data from file.
        
        Parameters
        ----------
        file_path : str
            Path to history data file
        
        Requirements: 4.4
        """
        if not os.path.exists(file_path):
            logging.warning(f"History file not found: {file_path}")
            return
        
        try:
            with open(file_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Merge loaded data with current data
            for key in self.history_data.keys():
                if key in loaded_data and loaded_data[key]:
                    if isinstance(self.history_data[key], list):
                        self.history_data[key].extend(loaded_data[key])
                    elif key == 'parameter_names' and not self.history_data[key]:
                        self.history_data[key] = loaded_data[key]
            
            logging.info(f"History data loaded from {file_path}")
        except Exception as e:
            logging.error(f"Failed to load history data: {e}")
    
    def get_statistics(self):
        """
        Get comprehensive statistics from history.
        
        Returns
        -------
        dict
            Dictionary containing statistics:
            - total_generations: Total number of generations
            - best_fitness: Best fitness achieved
            - average_generation_time: Average time per generation
            - total_improvement: Total fitness improvement
            - convergence_rate: Rate of convergence
            - diversity_trend: Trend in population diversity
        
        Requirements: 4.5
        """
        if not self.history_data['generation']:
            return {}
        
        stats = {
            'total_generations': len(self.history_data['generation']),
            'best_fitness': min(self.history_data['best_fitness']) if self.history_data['best_fitness'] else float('inf'),
            'average_generation_time': np.mean(self.history_data['generation_times']) if self.history_data['generation_times'] else 0,
        }
        
        # Calculate total improvement
        if len(self.history_data['best_fitness']) > 1:
            initial_fitness = self.history_data['best_fitness'][0]
            final_fitness = self.history_data['best_fitness'][-1]
            if initial_fitness != 0:
                stats['total_improvement'] = (initial_fitness - final_fitness) / abs(initial_fitness)
            else:
                stats['total_improvement'] = 0.0
        else:
            stats['total_improvement'] = 0.0
        
        # Calculate convergence rate (average improvement per generation)
        if self.history_data['improvement_rates']:
            stats['convergence_rate'] = np.mean(self.history_data['improvement_rates'])
        else:
            stats['convergence_rate'] = 0.0
        
        # Calculate diversity trend (slope of diversity over time)
        if len(self.history_data['diversity_metrics']) > 1:
            x = np.arange(len(self.history_data['diversity_metrics']))
            y = np.array(self.history_data['diversity_metrics'])
            # Simple linear regression slope
            slope = np.polyfit(x, y, 1)[0]
            stats['diversity_trend'] = slope
        else:
            stats['diversity_trend'] = 0.0
        
        return stats
