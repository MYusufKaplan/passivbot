"""
Checkpoint save/load functionality for CMA-ES optimization.
"""

import os
import pickle
from typing import Optional
from rich.console import Console
from .data_structures import Checkpoint

console = Console(
    force_terminal=True,
    no_color=False,
    log_path=False,
    width=191,
    color_system="truecolor",
    legacy_windows=False,
)


def save_checkpoint(
    checkpoint_path: str,
    restart_number: int,
    global_best_solution,
    global_best_fitness: float,
    logbook: list = None
):
    """
    Save CMA-ES restart checkpoint to disk.
    
    This function saves the global best solution and restart state to enable
    resuming optimization after interruption. It handles save failures gracefully
    by logging errors but not crashing, allowing optimization to continue.
    
    Note: CMA-ES internal state is NOT saved because each restart is independent
    with fresh state (identity covariance, configured sigma0).
    
    Args:
        checkpoint_path: Path to save checkpoint file
        restart_number: Current restart number
        global_best_solution: Best solution found across all restarts (denormalized)
        global_best_fitness: Best fitness value found
        logbook: Optional list of generation statistics
        
    Requirements: 6.2, 6.6
    """
    try:
        # Create checkpoint data
        checkpoint_data = Checkpoint(
            restart_number=restart_number,
            global_best_solution=global_best_solution,
            global_best_fitness=global_best_fitness,
            logbook=logbook if logbook is not None else [],
        )
        
        # Save to disk
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        console.print(
            f"[green]💾 Checkpoint saved: restart #{restart_number}, "
            f"best fitness: {global_best_fitness:.6e}[/green]"
        )
        
    except IOError as e:
        # Log error but don't crash - optimization can continue
        console.print(
            f"[yellow]⚠️ Checkpoint save failed (IOError): {e}, continuing optimization[/yellow]"
        )
    except Exception as e:
        # Catch any other serialization errors
        console.print(
            f"[yellow]⚠️ Checkpoint save failed: {e}, continuing optimization[/yellow]"
        )


def load_checkpoint(checkpoint_path: str) -> Optional[Checkpoint]:
    """
    Load CMA-ES restart checkpoint from disk.
    
    This function attempts to load a checkpoint file to resume optimization.
    If the file doesn't exist or is corrupted, it returns None and logs a
    warning, allowing optimization to start fresh.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Checkpoint object if successful, None if file missing or corrupted
        
    Requirements: 6.3, 6.4, 6.5, 12.1
    """
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        console.print(
            f"[cyan]ℹ️ No checkpoint found at {checkpoint_path}, starting fresh[/cyan]"
        )
        return None
    
    try:
        # Load checkpoint from disk
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Verify it's a Checkpoint object
        if not isinstance(checkpoint_data, Checkpoint):
            console.print(
                f"[yellow]⚠️ Checkpoint file has invalid format, starting fresh[/yellow]"
            )
            return None
        
        console.print(
            f"[green]✅ Checkpoint loaded: restart #{checkpoint_data.restart_number}, "
            f"best fitness: {checkpoint_data.global_best_fitness:.6e}[/green]"
        )
        
        return checkpoint_data
        
    except (pickle.UnpicklingError, EOFError) as e:
        # Corrupted pickle file
        console.print(
            f"[yellow]⚠️ Checkpoint file corrupted: {e}, starting fresh[/yellow]"
        )
        return None
    except Exception as e:
        # Any other error
        console.print(
            f"[yellow]⚠️ Failed to load checkpoint: {e}, starting fresh[/yellow]"
        )
        return None
