import os
import json
import random
import numpy as np
import asyncio
import dill as pickle
from rich.console import Console
from rich.traceback import install
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from deap import base, creator, tools, algorithms
from optimize import initEvaluator
import contextlib
import time
import multiprocessing

install()
console = Console(force_terminal=True, no_color=False, log_path=False)

CHECKPOINT_FILE = "de_checkpoints/de_state.pkl"
LOG_PATH = "logs/evaluation_output.log"
os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)

# DEAP Setup (Custom Differential Evolution)
def setup_deap(bounds, evaluator):
    dim = len(bounds)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: random.uniform(0, 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(ind):
        return (evaluate_solution((ind, bounds, evaluator)),)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.2, indpb=0.2)  # Mutation
    toolbox.register("select", tools.selBest)  # Selection
    
    return toolbox

def load_output_bounds(json_path):
    console.log(f"[bold cyan]📂 Loading output bounds from:[/bold cyan] {json_path}")
    with open(json_path) as f:
        data = json.load(f)
    return list(data["optimize"]["bounds"].values())

def map_output(raw_vals, bounds):
    return [min_ + val * (max_ - min_) for val, (min_, max_) in zip(raw_vals, bounds)]

def evaluate_solution(args):
    sol, bounds, evaluator = args
    sol_clipped = np.clip(sol, 0, 1)
    mapped = map_output(sol_clipped, bounds)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    try:
        with open(LOG_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            print("\n" + "=" * 60)
            raw_fitness = evaluator.evaluate(mapped)[0]
            return np.log1p(raw_fitness)  # Compressed fitness
    except Exception as e:
        with open(LOG_PATH, "a") as f:
            f.write(f"\n[Evaluation Error] {e}\n")
        return float("inf")

def save_checkpoint(pop, gen, logbook):
    os.makedirs("de_checkpoints", exist_ok=True)
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump((pop, gen, logbook), f)
    console.log(f"💾 [green]Checkpoint saved at generation {gen}[/green]")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            pop, gen, logbook = pickle.load(f)
        console.log(f"🔄 [yellow]Resuming from generation {gen + 1}[/yellow]")
        return pop, gen, logbook
    return None, 0, None

# Main DE Optimization Loop
async def run_de(bounds, evaluator):
    console.rule("[bold green]🚀 Starting Differential Evolution")

    # Set up the DEAP algorithm
    toolbox = setup_deap(bounds, evaluator)
    population = toolbox.population(n=100)

    checkpoint_pop, gen, logbook = load_checkpoint()

    if checkpoint_pop:
        population = checkpoint_pop
    else:
        gen = 0

    # Define callback to save checkpoint
    history = {"best": [], "time": []}
    
    def callback(gen, pop, logbook):
        fitness = [ind.fitness.values[0] for ind in pop]
        best_individual = tools.selBest(pop, 1)[0]
        history["best"].append((best_individual, best_individual.fitness.values[0]))
        history["time"].append(time.time())
        save_checkpoint(pop, gen, logbook)
        console.log(f"📌 Intermediate Best Fitness: {best_individual.fitness.values[0]:.6e}")

    # Set up multiprocessing pool
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    toolbox.register("map", pool.map)

    # Optimization loop
    while gen < 300:  # You can adjust max generations here
        gen_start = time.time()
        console.rule(f"[bold green]📈 Generation {gen + 1}")

        # Evaluate population in parallel using multiprocessing
        fitnesses = list(toolbox.map(toolbox.evaluate, population))

        # Apply the Differential Evolution algorithm
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:  # Crossover probability
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:  # Mutation probability
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate new fitness
        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        toolbox.map(toolbox.evaluate, invalid_individuals)

        # Replace old population with the new one
        population[:] = offspring

        # Log statistics for the generation
        best_individual = tools.selBest(population, 1)[0]
        best_fitness = best_individual.fitness.values[0]
        mean_fitness = np.mean([ind.fitness.values[0] for ind in population])

        gen_time = time.time() - gen_start
        console.print(f"🎖️ [cyan]Best fitness:[/cyan] {best_fitness:.6e}")
        console.print(f"📊 [cyan]Mean fitness:[/cyan] {mean_fitness:.6e}")
        console.print(f"⏱️ [magenta]Generation time:[/magenta] {gen_time:.2f} sec / {(gen_time/60):.2f} min")
        save_checkpoint(population, gen, logbook)

        gen += 1
        callback(gen, population, logbook)

    # Return the best parameters
    best_individual = tools.selBest(population, 1)[0]
    best_params = map_output(np.clip(best_individual, 0, 1), bounds)
    return best_params, best_individual.fitness.values[0]

async def main():
    pb_config = "/home/myusuf/Projects/passivbot/configs/optimize.json"
    evaluator = await initEvaluator(pb_config)

    console.print(Panel("🤖 [bold]Differential Evolution Optimizer[/bold] initialized", style="bold blue"))
    output_bounds = load_output_bounds(pb_config)

    try:
        result, fitness = await run_de(output_bounds, evaluator)
    except Exception as e:
        console.log(f"🔥 [bold red]Differential Evolution failed: {e}[/bold red]")
        return

    console.print("\n🎯 [cyan]Final Outputs:[/cyan]")
    for i, val in enumerate(result, 1):
        console.print(f"[{i:02d}] {val:.4f}", style="bright_white")

    console.print(f"\n🏆 [bold green]Final Fitness: {fitness:.4f}[/bold green]")

if __name__ == "__main__":
    asyncio.run(main())
