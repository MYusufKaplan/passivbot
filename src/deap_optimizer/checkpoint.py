"""
Checkpointing functionality for DEAP optimizer.

This module provides checkpoint save/load functionality using pickle serialization.
Handles missing and corrupted checkpoint files gracefully, and includes toolbox state
and random state for complete reproducibility.
"""

import os
import pickle
import logging
import random
import numpy as np


def save_checkpoint(checkpoint_path, checkpoint_data):
    """
    Save checkpoint to file using pickle serialization.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to save checkpoint file
    checkpoint_data : dict
        Dictionary containing checkpoint data:
        - generation: Current generation number
        - population: List of individuals
        - logbook: DEAP logbook (optional)
        - best_individual: Best individual found
        - best_fitness: Best fitness value
        - generation_times: List of generation times
        - parameter_names: List of parameter names
        - options: Algorithm options
        - toolbox_state: Toolbox configuration (optional)
        - random_state: Random number generator state (optional)
    
    Returns
    -------
    bool
        True if checkpoint saved successfully, False otherwise
    
    Requirements: 3.1, 3.4, 7.2
    """
    try:
        # Ensure directory exists
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Capture random state if not provided
        if 'random_state' not in checkpoint_data:
            checkpoint_data['random_state'] = {
                'python_random': random.getstate(),
                'numpy_random': np.random.get_state()
            }
        
        # Save to temporary file first, then rename for atomic write
        temp_path = checkpoint_path + '.tmp'
        with open(temp_path, 'wb') as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Atomic rename
        os.replace(temp_path, checkpoint_path)
        
        logging.info(f"Checkpoint saved to {checkpoint_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")
        # Clean up temporary file if it exists
        temp_path = checkpoint_path + '.tmp'
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False


def load_checkpoint(checkpoint_path):
    """
    Load checkpoint from file.
    
    Handles missing and corrupted checkpoint files gracefully by returning None,
    allowing the optimizer to start fresh.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file
    
    Returns
    -------
    dict or None
        Checkpoint data if file exists and is valid, None otherwise.
        Returns None for missing files, corrupted files, or any loading errors.
    
    Requirements: 3.2, 3.5, 7.2
    """
    if not os.path.exists(checkpoint_path):
        logging.info(f"No checkpoint found at {checkpoint_path}")
        return None
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Validate checkpoint data structure
        required_keys = ['generation', 'best_individual', 'best_fitness']
        if not all(key in checkpoint_data for key in required_keys):
            logging.warning(f"Checkpoint file is missing required keys. Starting fresh optimization.")
            return None
        
        # Restore random state if present
        if 'random_state' in checkpoint_data:
            try:
                if 'python_random' in checkpoint_data['random_state']:
                    random.setstate(checkpoint_data['random_state']['python_random'])
                if 'numpy_random' in checkpoint_data['random_state']:
                    np.random.set_state(checkpoint_data['random_state']['numpy_random'])
                logging.info("Random state restored from checkpoint")
            except Exception as e:
                logging.warning(f"Failed to restore random state: {e}")
        
        logging.info(f"Checkpoint loaded from {checkpoint_path} (generation {checkpoint_data.get('generation', 0)})")
        return checkpoint_data
    except (pickle.UnpicklingError, EOFError, AttributeError) as e:
        logging.error(f"Corrupted checkpoint file: {e}")
        logging.warning("Starting fresh optimization")
        return None
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        logging.warning("Starting fresh optimization")
        return None


def validate_checkpoint(checkpoint_data):
    """
    Validate checkpoint data structure.
    
    Parameters
    ----------
    checkpoint_data : dict
        Checkpoint data to validate
    
    Returns
    -------
    bool
        True if checkpoint data is valid, False otherwise
    """
    if not isinstance(checkpoint_data, dict):
        return False
    
    required_keys = ['generation', 'best_individual', 'best_fitness']
    return all(key in checkpoint_data for key in required_keys)
