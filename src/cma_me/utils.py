"""
Utility functions for CMA-ME optimizer.

This module provides helper functions for logging, progress bars, and
checkpoint management.
"""

import os
import pickle
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    TaskProgressColumn,
)


def log_archive_stats(archive, generation, best_fitness, console):
    """
    Log archive coverage, QD-score, and best fitness.
    
    Args:
        archive: pyribs GridArchive instance
        generation: Current generation number
        best_fitness: Best fitness value found
        console: Rich console for logging
    """
    # Get archive statistics
    stats = archive.stats
    
    # Calculate coverage percentage
    coverage = stats.coverage * 100  # Convert to percentage
    
    # Get QD-score (sum of all elite objectives)
    qd_score = stats.obj_mean * stats.num_elites  # Approximation of sum
    
    # Get number of filled cells
    num_elites = stats.num_elites
    
    # Get best elite from archive
    best_elite = archive.best_elite
    if best_elite is not None:
        # best_elite is a dict with keys: 'solution', 'objective', 'measures', 'index', 'metadata'
        best_measures = best_elite['measures']
        measures_str = ", ".join([f"{m:.3f}" for m in best_measures])
    else:
        measures_str = "N/A"
    
    # Create formatted message
    message = f"""🌟 Gen {generation}
    📊 Archive coverage: {coverage:.1f}% ({num_elites} elites)
    🌍 Best fitness: {best_fitness:.6e}
    📈 QD-Score: {qd_score:.6e}
    🎯 Best measures: [{measures_str}]"""
    
    # Create panel
    panel = Panel(message, title="CMA-ME Stats", border_style="cyan")
    console.print(panel)


def create_progress_bar(total, description, console=None):
    """
    Create Rich progress bar matching existing optimizer style.
    
    Args:
        total: Total number of items to process
        description: Description text for progress bar
        console: Optional Rich Console instance to write to (for file logging)
        
    Returns:
        Rich Progress instance
    """
    # When writing to file, don't use transient mode so progress is logged
    transient = console is None
    
    progress = Progress(
        SpinnerColumn(spinner_name="dots12"),
        TextColumn(f"🧬 [progress.description]{{task.description}}"),
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
        transient=transient,  # Don't use transient when logging to file
        console=console,  # Use provided console for file logging
    )
    
    return progress


def save_checkpoint(checkpoint_path, archive, scheduler, generation, best_fitness, logbook):
    """
    Save optimization state to disk.
    
    Args:
        checkpoint_path: Path to save checkpoint file
        archive: pyribs GridArchive instance
        scheduler: pyribs Scheduler instance
        generation: Current generation number
        best_fitness: Best fitness value found
        logbook: List of generation statistics
        
    Raises:
        IOError: If checkpoint save fails (logged but not raised)
    """
    try:
        checkpoint_data = {
            'archive': archive,
            'scheduler': scheduler,
            'generation': generation,
            'best_fitness': best_fitness,
            'logbook': logbook,
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
    except IOError as e:
        # Log error but don't crash - optimization can continue
        import sys
        print(f"⚠️ Checkpoint save failed: {e}, continuing optimization", file=sys.stderr)
    except Exception as e:
        # Catch any other serialization errors
        import sys
        print(f"⚠️ Checkpoint save failed: {e}, continuing optimization", file=sys.stderr)


def load_checkpoint(checkpoint_path):
    """
    Load optimization state from disk.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary containing checkpoint data or None if file doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        return checkpoint_data
    except Exception as e:
        import sys
        print(f"⚠️ Failed to load checkpoint: {e}, starting fresh", file=sys.stderr)
        return None
