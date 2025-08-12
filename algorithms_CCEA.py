#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`algorithms` module is intended to contain some specific algorithms
in order to execute very common evolutionary algorithms. The method used here
are more for convenience than reference as the implementation of every
evolutionary algorithm may vary infinitely. Most of the algorithms in this
module use operators registered in the toolbox. Generally, the keyword used are
:meth:`mate` for crossover, :meth:`mutate` for mutation, :meth:`~deap.select`
for selection and :meth:`evaluate` for evaluation.

You are encouraged to write your own algorithms in order to make them do what
you really want them to do.
"""

import random

from . import tools



import os
import pickle
import datetime
import time
import json
import numpy as np
from collections import deque
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.rule import Rule
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, TaskProgressColumn
from multiprocessing import Pool, cpu_count

RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Initialize the console for rich output
console = console = Console(
            force_terminal=True, 
            no_color=False, 
            log_path=False, 
            width=159,
            color_system="truecolor",  # Force truecolor support
            legacy_windows=False
        )

def population_diversity(pop):
    gene_matrix = np.array(pop)
    return np.sum(np.std(gene_matrix, axis=0))

def log_message(message, emoji=None, panel=False, timestamp=True):
    """Utility function to print logs with Rich panels and rules"""
    if timestamp:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        timestamp = ""
    emoji_str = f" {emoji}" if emoji else ""
    log_text = f"{timestamp}{emoji_str} {message}"

    # Using Rule for major transitions
    if panel:
        panel_message = Panel(log_text, title="Stats", border_style="cyan")
        console_wrapper(panel_message)
    else:
        console_wrapper(log_text)

import contextlib

LOG_PATH = "logs/evaluation_output.log"
WATCH_PATH = "logs/evaluation.log"
BEST_LOG_PATH = "logs/evaluation_output_best.log"

def evaluate_solution(args):
    evaluator, ind, showMe = args
    if showMe:
        with open(BEST_LOG_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            print("\n" + "=" * 60)  # separator between evaluations
            # Return a tuple (as DEAP expects)
            return evaluator.evaluate(ind)
    with open(LOG_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        print("\n" + "=" * 60)  # separator between evaluations
        # Return a tuple (as DEAP expects)
        return evaluator.evaluate(ind)



def console_wrapper(msg):
    with open(WATCH_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        console.print(msg)

def mutate_specialist(specialist, bounds, mutation_strength=0.1, is_integer=False):
    """Step 3: Mutate a single specialist value"""
    min_val, max_val = bounds
    range_size = max_val - min_val
    mutated = specialist + random.gauss(0, mutation_strength * range_size)
    mutated = max(min_val, min(max_val, mutated))  # Clamp to bounds
    
    if is_integer:
        mutated = round(mutated)
        mutated = max(int(min_val), min(int(max_val), int(mutated)))  # Ensure integer bounds
    
    return mutated

def evaluate_all_islands(islands, global_best_vector, evaluator, toolbox):
    """Step 2 & 6: Evaluate all specialists by creating full individuals"""
    pool = Pool(processes=(cpu_count() - 1))
    all_args = []
    island_ranges = []
    start_idx = 0
    with open(WATCH_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):

        # Prepare evaluation arguments
        for island in islands:
            param_indices = island['param_indices']
            for specialist in island['specialists']:
                # Create full individual by combining specialist group with global vector
                full_solution = global_best_vector.copy()
                for i, param_idx in enumerate(param_indices):
                    full_solution[param_idx] = specialist[i]
                
                individual = toolbox.individual()
                individual[:] = full_solution
                all_args.append((evaluator, individual, False))
            
            island_ranges.append((start_idx, start_idx + len(island['specialists'])))
            start_idx += len(island['specialists'])
        
        # Initialize results tracking
        all_fitnesses = [None] * len(all_args)
        island_best_fitnesses = [float("inf")] * len(islands)
        completed_counts = [0] * len(islands)
        
        # Evaluate with individual progress bars for each island
        with Progress(
            SpinnerColumn(spinner_name="dots12"),
            TextColumn("🏝️ [progress.description]{task.description}"),
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
            
            # Create progress tasks for each island
            island_tasks = []
            for island_id, island in enumerate(islands):
                group_name = island['group_name']
                
                task = progress.add_task(
                    f"🏝️ {group_name} | Best: {island_best_fitnesses[island_id]:.6e}", 
                    total=len(island['specialists'])
                )
                island_tasks.append(task)
            
            # Process results as they complete
            result_idx = 0
            for fitness in pool.imap_unordered(evaluate_solution, all_args):
                all_fitnesses[result_idx] = fitness[0]  # Extract fitness value
                
                # Find which island this result belongs to
                for island_id, (start, end) in enumerate(island_ranges):
                    if start <= result_idx < end:
                        completed_counts[island_id] += 1
                        island_best_fitnesses[island_id] = min(island_best_fitnesses[island_id], fitness[0])
                        
                        # Update progress for this island
                        group_name = islands[island_id]['group_name']
                        
                        progress.update(
                            island_tasks[island_id], 
                            advance=1,
                            description=f"🏝️ {group_name} | Best: {island_best_fitnesses[island_id]:.6e}"
                        )
                        break
                
                result_idx += 1
        
        pool.close()
        pool.join()
        
        # Assign fitnesses back to islands and find best specialists
        global_best_fitness = float('inf')
        
        for island_id, island in enumerate(islands):
            start, end = island_ranges[island_id]
            island_fitnesses = all_fitnesses[start:end]
            
            # 1) Best fitness within an island in the current generation
            best_idx = island_fitnesses.index(min(island_fitnesses))
            current_gen_island_best_specialist = island['specialists'][best_idx]
            current_gen_island_best_fitness = island_fitnesses[best_idx]
            
            # 2) Best fitness within an island in all generations so far (all-time island best)
            if current_gen_island_best_fitness < island['best_fitness']:
                # New all-time best for this island
                island['best_specialist'] = current_gen_island_best_specialist
                island['best_fitness'] = current_gen_island_best_fitness
                # Note: stagnation will be managed in main loop based on global vector improvement
            else:
                # No improvement for this island
                # Note: stagnation will be managed in main loop based on global vector improvement
                # Keep the all-time best specialist and fitness (don't update)
                
                # Debug: Check if the all-time best specialist is actually in the current generation
                if island['best_specialist'] not in island['specialists']:
                    log_message(f"⚠️ ERROR: All-time best specialist {island['best_specialist']} not found in current specialists for {island['group_name']}", emoji="⚠️")
                    log_message(f"Current specialists: {island['specialists'][:5]}...", emoji="🔍")
            
            # 3) Best fitness within all islands in all generations so far (global best)
            # Use current generation's island best to update global best
            if current_gen_island_best_fitness < global_best_fitness:
                global_best_fitness = current_gen_island_best_fitness
            
            # Store fitness mapping for selection (convert lists to tuples for hashing)
            island['fitness_map'] = dict(zip([tuple(s) for s in island['specialists']], island_fitnesses))
        
        return global_best_fitness

def update_global_vector(islands, global_best_vector):
    """Step 4 & 7: Update global vector with best specialists (legacy function)"""
    for island in islands:
        param_idx = island['param_idx']
        global_best_vector[param_idx] = island['best_specialist']








def create_individual_from_list(values):
    """Helper function to create individual from list of values"""
    individual = list(values)
    return individual

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, evaluator,
                       stats=None, halloffame=None, verbose=__debug__,
                       checkpoint_path="checkpoint.pkl", checkpoint_interval=1,
                       parameter_bounds=None):
    
    """
    CCEA (Cooperative Coevolutionary Algorithm) Implementation
    
    Terminology:
    - individuals: toolbox.individual() objects with full parameter vectors
    - specialists: single float values representing one parameter
    """
    start_time = time.time()
    
    if not parameter_bounds:
        raise ValueError("parameter_bounds is required for CCEA")
    
    # Filter optimizable vs fixed parameters and identify integer parameters
    optimizable_bounds = {}
    fixed_params = {}
    integer_params = set()
    
    for param_name, bounds in parameter_bounds.items():
        min_val, max_val = bounds
        if "short" not in param_name :  # Parameter has range to optimize
            optimizable_bounds[param_name] = bounds
            # Check if parameter should be integer (like n_positions, positions, etc.)
            if any(keyword in param_name.lower() for keyword in ['n_positions']):
                integer_params.add(param_name)
        else:  # Fixed parameter
            fixed_params[param_name] = min_val
    
    all_param_names = list(parameter_bounds.keys())
    num_parameters = len(optimizable_bounds)
    island_size = mu // num_parameters if num_parameters > 0 else 0
    
    log_message(f"🧬 CCEA: {num_parameters} optimizable parameters, {len(fixed_params)} fixed", emoji="🔬")
    
    # Load initial values from optimize.json
    initial_values = {}
    try:
        with open("configs/optimize.json", "r") as f:
            config = json.load(f)
            long_config = config.get("bot", {}).get("long", {})
            for param_name in optimizable_bounds.keys():
                if param_name.replace("long_","") in long_config:
                    initial_values[param_name] = long_config[param_name.replace("long_","")]
        log_message(f"📋 Loaded {len(initial_values)} initial values from optimize.json", emoji="📋")
    except Exception as e:
        log_message(f"⚠️ Could not load optimize.json, using middle values: {e}", emoji="⚠️")
    
    # Step 1: Try loading from checkpoint
    islands = []
    global_best_vector = []
    global_best_fitness = float('inf')
    start_gen = 1
    logbook = tools.Logbook()
    
    if os.path.exists(checkpoint_path):
        log_message("📦 Loading checkpoint...", emoji="📦")
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)
            islands = checkpoint_data.get("islands", [])
            global_best_vector = checkpoint_data.get("global_best_vector", [])
            global_best_fitness = checkpoint_data.get("global_best_fitness", float('inf'))
            start_gen = checkpoint_data.get("generation", 0) + 1
            logbook = checkpoint_data.get("logbook", tools.Logbook())
        
        log_message(f"✅ Resumed from generation {start_gen-1}, best fitness: {global_best_fitness:.6e}", emoji="✅")
        
        # Ensure integer flags are preserved after checkpoint loading
        for island in islands:
            # Initialize stagnation counter if not present (for old checkpoints)
            if 'stagnation' not in island:
                island['stagnation'] = 0
            
            # Ensure integer flags are set correctly for each parameter in the group
            if 'integer_flags' not in island:
                island['integer_flags'] = [param_name in integer_params for param_name in island['param_names']]
            
            # Ensure current best_specialist values are integers where needed
            for i, param_name in enumerate(island['param_names']):
                if param_name in integer_params:
                    island['best_specialist'][i] = round(island['best_specialist'][i])
        
        # Ensure global vector has integer values for integer parameters
        for i, param_name in enumerate(all_param_names):
            if param_name in integer_params:
                global_best_vector[i] = round(global_best_vector[i])
        
        # Go to step 5 (main loop)
    else:
        # Step 2: Initialize islands and global vector
        log_message("🚀 No checkpoint, starting fresh", emoji="🚀")
        
        # Initialize global best vector with initial values from config, fallback to middle values
        for param_name in all_param_names:
            if param_name in fixed_params:
                global_best_vector.append(fixed_params[param_name])
            else:
                # Use initial value from config if available, otherwise use middle value
                if initial_values and param_name in initial_values:
                    initial_val = initial_values[param_name]
                    if param_name in integer_params:
                        initial_val = round(initial_val)
                    global_best_vector.append(initial_val)
                else:
                    # Fallback to middle value
                    min_val, max_val = optimizable_bounds[param_name]
                    middle_val = (min_val + max_val) / 2
                    if param_name in integer_params:
                        middle_val = round(middle_val)
                    global_best_vector.append(middle_val)
        
        # Group parameters into functional islands
        parameter_groups = {
            'entry': [p for p in optimizable_bounds.keys() if 'entry' in p],
            'close': [p for p in optimizable_bounds.keys() if 'close' in p],
            'ema': [p for p in optimizable_bounds.keys() if 'ema' in p],
            'filter': [p for p in optimizable_bounds.keys() if 'filter' in p],
            'unstuck': [p for p in optimizable_bounds.keys() if 'unstuck' in p],
            'general': [p for p in optimizable_bounds.keys() if not any(keyword in p for keyword in ['entry', 'close', 'ema', 'filter', 'unstuck'])]
        }
        
        # Remove empty groups
        parameter_groups = {name: params for name, params in parameter_groups.items() if params}
        
        # Calculate island size based on number of groups
        num_groups = len(parameter_groups)
        island_size = mu // num_groups if num_groups > 0 else mu
        
        log_message(f"🏝️ Created {num_groups} parameter groups: {list(parameter_groups.keys())}", emoji="🏝️")
        
        # Create islands - one per parameter group
        for group_name, param_names in parameter_groups.items():
            # Get parameter indices and bounds for this group
            param_indices = [all_param_names.index(p) for p in param_names]
            param_bounds = [optimizable_bounds[p] for p in param_names]
            
            # Create specialists (full parameter vectors for this group)
            specialists = []
            
            # Create initial specialist from JSON values
            initial_specialist = []
            for param_name in param_names:
                if initial_values and param_name in initial_values:
                    initial_val = initial_values[param_name]
                    if param_name in integer_params:
                        initial_val = round(initial_val)
                    initial_specialist.append(initial_val)
                else:
                    # Fallback to middle value
                    min_val, max_val = optimizable_bounds[param_name]
                    middle_val = (min_val + max_val) / 2
                    if param_name in integer_params:
                        middle_val = round(middle_val)
                    initial_specialist.append(middle_val)
            
            # Add initial specialist
            specialists.append(initial_specialist.copy())
            
            # Generate random specialists for this group
            for _ in range(island_size - 1):
                specialist = []
                for i, param_name in enumerate(param_names):
                    min_val, max_val = param_bounds[i]
                    if param_name in integer_params:
                        specialist.append(random.randint(int(min_val), int(max_val)))
                    else:
                        specialist.append(random.uniform(min_val, max_val))
                specialists.append(specialist)
            
            islands.append({
                'group_name': group_name,
                'param_names': param_names,
                'param_indices': param_indices,
                'param_bounds': param_bounds,
                'specialists': specialists,
                'best_specialist': initial_specialist.copy(),
                'best_fitness': float('inf'),
                'integer_flags': [p in integer_params for p in param_names],
                'stagnation': 0
            })
            
            log_message(f"  🏝️ {group_name}: {len(param_names)} params, {island_size} specialists", emoji="🏝️")
        
        # Step 3: Set initial global vector to JSON values and evaluate
        log_message("🔍 Initial evaluation with JSON values...", emoji="🔍")
        
        # The global vector is already initialized with JSON values above
        # Just evaluate it to get the initial global fitness
        initial_individual = toolbox.individual()
        initial_individual[:] = global_best_vector
        global_best_fitness = evaluator.evaluate(initial_individual)[0]
        
        # Set each island's best specialist to the corresponding global vector values
        for island in islands:
            param_indices = island['param_indices']
            best_specialist = [global_best_vector[idx] for idx in param_indices]
            island['best_specialist'] = best_specialist
            island['best_fitness'] = global_best_fitness  # All start with the same global fitness
        
        logbook.record(gen=0, nevals=mu, best=global_best_fitness)
    
    log_message(f"🧬 Starting CCEA evolution from generation {start_gen}", emoji="🧬")

    # Initialize tracking variables
    stagnation = 0
    best_fitness_so_far = global_best_fitness
    generation_times = []

    # Main CCEA Evolution Loop (Steps 5-7)
    for gen in range(start_gen, ngen + 1):
        gen_start_time = time.time()
        console_wrapper(Rule(f"Generation {gen}", style="bold blue"))
        
        # Step 5: Get best specialists and mutate them
        for island in islands:
            best_specialist = island['best_specialist']
            param_bounds = island['param_bounds']
            integer_flags = island['integer_flags']
            
            # Generate mu/island_count mutated specialists
            new_specialists = []
            
            # First, preserve the best specialist from last generation (elitism)
            new_specialists.append(best_specialist.copy())
            
            # Then generate (island_size - 1) mutated specialists
            for _ in range(island_size - 1):
                mutated_specialist = []
                for i, (param_val, bounds, is_integer) in enumerate(zip(best_specialist, param_bounds, integer_flags)):
                    mutated_val = mutate_specialist(param_val, bounds, mutation_strength=mutpb, is_integer=is_integer)
                    mutated_specialist.append(mutated_val)
                new_specialists.append(mutated_specialist)
            
            island['specialists'] = new_specialists
        
        # Step 6: Evaluate all specialists and validate global improvements
        current_best_fitness = evaluate_all_islands(islands, global_best_vector, evaluator, toolbox)
        
        # Step 7: Validate and update global vector with proper CCEA logic
        proposed_global_vector = global_best_vector.copy()
        any_improvement = False
        
        # For each island, check if its best specialist actually improves the global vector
        for island in islands:
            param_indices = island['param_indices']
            current_gen_best_specialist = None
            current_gen_best_fitness = float('inf')
            
            # Find the specialist that gives the best GLOBAL fitness when combined with current global vector
            for specialist in island['specialists']:
                specialist_fitness = island['fitness_map'][tuple(specialist)]
                if specialist_fitness < current_gen_best_fitness:
                    current_gen_best_fitness = specialist_fitness
                    current_gen_best_specialist = specialist.copy()
            
            # Test if this specialist improves the global vector
            test_global_vector = global_best_vector.copy()
            for i, param_idx in enumerate(param_indices):
                test_global_vector[param_idx] = current_gen_best_specialist[i]
            
            # Create and evaluate the test individual
            test_individual = toolbox.individual()
            test_individual[:] = test_global_vector
            test_global_fitness = evaluator.evaluate(test_individual)[0]
            
            # Only accept if it improves the global fitness
            if test_global_fitness < global_best_fitness:
                # This specialist improves the global vector
                island['best_specialist'] = current_gen_best_specialist
                island['best_fitness'] = current_gen_best_fitness
                island['stagnation'] = 0  # Reset island stagnation
                for i, param_idx in enumerate(param_indices):
                    proposed_global_vector[param_idx] = current_gen_best_specialist[i]
                any_improvement = True
                log_message(f"✅ {island['group_name']}: improves global fitness to {test_global_fitness:.6e}", emoji="✅")
            else:
                # No improvement for this island
                island['stagnation'] += 1  # Increment island stagnation
                log_message(f"❌ {island['group_name']}: No improvement (stagnation: {island['stagnation']})", emoji="❌")
        
        # Update global state if any island improved
        if any_improvement:
            global_best_vector[:] = proposed_global_vector
            # Re-evaluate the new global vector to get the exact fitness
            final_individual = toolbox.individual()
            final_individual[:] = global_best_vector
            global_best_fitness = evaluator.evaluate(final_individual)[0]
            best_fitness_so_far = global_best_fitness
            log_message(f"🌟 New global best: {global_best_fitness:.6e}", emoji="🌟")
            stagnation = 0  # Reset global stagnation counter
        else:
            stagnation += 1  # Increment global stagnation counter
        
        # Logging and stats
        gen_time = time.time() - gen_start_time
        generation_times.append(gen_time)
        avg_gen_time = sum(generation_times) / len(generation_times)
        logbook.record(gen=gen, nevals=mu, best=global_best_fitness, time=gen_time)
        
        if verbose:
            # Show generation best line if improved
            gen_best_line = ""
            if current_best_fitness < best_fitness_so_far:
                gen_best_line = f"\n🎯 Gen Best fitness: {current_best_fitness:.6e}"
            
            # Show global vector with island stagnation info (skip short_* params)
            global_vector_info = "\n🌐 Global Vector:"
            for i, (param_name, value) in enumerate(zip(all_param_names, global_best_vector)):
                # Skip short_* parameters (fixed parameters)
                if "short" in param_name:
                    continue
                    
                # Find corresponding island for this parameter
                island_stagnation = 0
                group_name = ""
                for island in islands:
                    if param_name in island['param_names']:
                        island_stagnation = island['stagnation']
                        group_name = island['group_name']
                        break
                
                if param_name in integer_params:
                    global_vector_info += f"\n                  📊 {param_name}: {int(value)} ({group_name}, stag: {island_stagnation})"
                else:
                    global_vector_info += f"\n                  📊 {param_name}: {value:.6f} ({group_name}, stag: {island_stagnation})"
            
            log_message(
                f"""{CYAN}🌟 Gen {gen}{RESET}
⏱️ Stagnation: {stagnation}
🌍 Global Best fitness: {best_fitness_so_far:.6e}{gen_best_line}
⏱️ Generation time: {gen_time:.2f} sec / {(gen_time/60):.2f} min
📆 Avg gen time: {avg_gen_time:.2f} sec / {(avg_gen_time/60):.2f} min{global_vector_info}""",
                panel=True, timestamp=False
            )
        
        # Checkpoint saving
        if gen % checkpoint_interval == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "islands": islands,
                    "global_best_vector": global_best_vector,
                    "global_best_fitness": global_best_fitness,
                    "generation": gen,
                    "logbook": logbook
                }, f)
            log_message(f"💾 Checkpoint saved at generation {gen}", emoji="💾")
    
    # Create final population for return
    final_population = []
    for i in range(mu):
        individual = toolbox.individual()
        individual[:] = global_best_vector
        individual.fitness.values = (global_best_fitness,)
        final_population.append(individual)
    
    total_time = time.time() - start_time
    log_message(f"🕒 Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)", emoji="🕒")
    
    return final_population, logbook
