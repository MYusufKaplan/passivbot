import os
import json
import cma
import numpy as np
import asyncio
import dill as pickle
from rich.console import Console
from rich.traceback import install
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from optimize import initEvaluator
import multiprocessing
import time
import contextlib

install()
console = Console(force_terminal=True, no_color=False, log_path=False)
CHECKPOINT_FILE = "cma_checkpoints/cma_state.pkl"
LOG_PATH = "logs/evaluation_output.log"

def load_output_bounds(json_path):
    console.log(f"[bold cyan]📂 Loading output bounds from:[/bold cyan] {json_path}")
    with open(json_path) as f:
        data = json.load(f)
    return list(data["optimize"]["bounds"].values())


def map_output(raw_vals, bounds):
    return [min_ + val * (max_ - min_) for val, (min_, max_) in zip(raw_vals, bounds)]


def save_checkpoint(es, generation):
    os.makedirs("cma_checkpoints", exist_ok=True)
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump((es, generation), f)
    console.log(f"💾 [green]Checkpoint saved at generation {generation}[/green]")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            es, generation = pickle.load(f)
        console.log(f"🔄 [yellow]Resuming from generation {generation}[/yellow]")
        return es, generation
    return None, 0

def evaluate_solution(args):
    sol, bounds, evaluator = args
    sol_clipped = np.clip(sol, 0, 1)
    mapped = map_output(sol_clipped, bounds)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    try:
        with open(LOG_PATH, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            print("\n" + "=" * 60)  # separator between evaluations
            print(f"Evaluating solution: {mapped}")
            fitness, _ = evaluator.evaluate(mapped)
    except Exception as e:
        with open(LOG_PATH, "a") as f:
            f.write(f"\n[Evaluation Error] {e}\n")
        return float("inf")

    return fitness

def wrapped_eval(args):
    sol, bounds, evaluator = args
    fitness = evaluate_solution((sol, bounds, evaluator))  # Evaluate fitness
    return fitness

async def run_cma_es(bounds, evaluator):
    console.rule("[bold green]🚀 Starting CMA-ES Optimization")

    dim = len(bounds)
    sigma = 0.3
    x0 = [0.5] * dim  # normalized to [0, 1]

    es, start_gen = load_checkpoint()
    if not es:
        es = cma.CMAEvolutionStrategy(x0, sigma, {
            'popsize': 100,
            'bounds': [0, 1],
            'verb_disp': 0
        })

    gen = start_gen
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    console.log(f"🧠 Using [magenta]{multiprocessing.cpu_count()}[/magenta] CPU cores")
    gen_runtimes = []

    while not es.stop():
        gen_start = time.time()

        solutions = es.ask()
        args = [(sol, bounds, evaluator) for sol in solutions]
        fitnesses = []

        # Progress bar inside the main loop
        with Progress(
            TextColumn("🔍 [progress.description]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,         # Make sure console = Console(force_terminal=True)
            transient=False          # Keep progress bar visible in logs
        ) as progress:
            task = progress.add_task(f"Generation {gen + 1}", total=len(args))

            # Use pool.imap_unordered with wrapped_eval to update the progress bar during execution
            for fitness in pool.imap_unordered(wrapped_eval, args):
                fitnesses.append(fitness)
                progress.update(task, advance=1)  # Update progress bar for each evaluated solution


        es.tell(solutions, fitnesses)
        es.logger.add()
        es.disp()
        gen += 1

        # Extract statistics
        best_fitness = np.min(fitnesses)
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)
        best_index = np.argmin(fitnesses)
        best_solution = solutions[best_index]

        # Time tracking
        gen_time = time.time() - gen_start
        gen_runtimes.append(gen_time)
        avg_gen_time = sum(gen_runtimes) / len(gen_runtimes)

        console.rule(f"[bold green]📈 Generation {gen}")
        console.print(f"🏆 [cyan]Best fitness:[/cyan] {best_fitness:.6e}")
        console.print(f"📊 [cyan]Mean fitness:[/cyan] {mean_fitness:.6e}")
        console.print(f"📉 [cyan]Fitness std dev:[/cyan] {std_fitness:.6e}")
        console.print(f"⏱️ [magenta]Generation time:[/magenta] {gen_time:.2f} sec")
        console.print(f"📆 [magenta]Avg gen time:[/magenta] {avg_gen_time:.2f} sec")

        save_checkpoint(es, gen)

        # if gen >= 300:
        #     break
    pool.close()
    pool.join()

    best = es.result.xbest
    best_params = map_output(np.clip(best, 0, 1), bounds)
    return best_params, es.result.fbest


async def main():
    pb_config = "/home/myusuf/Projects/passivbot/configs/optimize.json"
    evaluator = await initEvaluator(pb_config)

    console.print(Panel("🤖 [bold]CMA-ES Optimizer[/bold] initialized", style="bold blue"))
    output_bounds = load_output_bounds(pb_config)

    try:
        result, fitness = await run_cma_es(output_bounds, evaluator)
    except Exception as e:
        console.log(f"🔥 [bold red]CMA-ES failed during execution: {e}[/bold red]")
        return

    console.print("\n🎯 [cyan]Final 50 Outputs:[/cyan]")
    for i, val in enumerate(result, 1):
        console.print(f"[{i:02d}] {val:.4f}", style="bright_white")

    console.print(f"\n🏆 [bold green]Final Fitness: {fitness:.4f}[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
