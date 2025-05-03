import neat
import numpy as np
import os
import json
import asyncio
import multiprocessing
import dill as pickle
from rich.console import Console
from rich.traceback import install
from rich.panel import Panel
from neat_tools import evaluate
from optimize import initEvaluator
import re

install()
console = Console()

def load_output_bounds(json_path):
    console.log(f"~[bold cyan]📂 Loading output bounds from:[/bold cyan] {json_path}")
    with open(json_path) as f:
        data = json.load(f)
    return list(data["optimize"]["bounds"].values())

def map_output(value, min_val, max_val):
    return min_val + (value + 1) / 2 * (max_val - min_val)

class SafeEvaluator:
    def __init__(self, output_bounds, evaluator):
        self.output_bounds = output_bounds
        self.evaluator = evaluator

    def __call__(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        raw_output = net.activate([1])  # in [-1, 1] due to tanh
        bounded_output = [
            map_output(val, *bound) for val, bound in zip(raw_output, self.output_bounds)
        ]
        fit, discard = self.evaluator.evaluate(bounded_output)
        genome.fitness = -1 * fit
        console.log(f"~[bold cyan]🧠 Genome Fitness:[/bold cyan] {genome.fitness}")
        return genome.fitness


def run_neat(config_path, output_bounds, evaluator, resume=True):
    console.rule("[bold green]🚀 Starting NEAT Optimization")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    def latest_checkpoint():
        files = [f for f in os.listdir("neat_checkpoints") if f.startswith("neat-checkpoint-")]
        if not files:
            return None
        return os.path.join("neat_checkpoints", max(files, key=lambda f: int(re.search(r"\d+", f).group())))

    ckpt = latest_checkpoint()

    if resume and ckpt:
        console.log(f"~[yellow]🪄 Resuming from last checkpoint: {ckpt} ...")
        pop = neat.Checkpointer.restore_checkpoint(ckpt)
    else:
        console.log("~[blue]🧬 Creating new NEAT population...")
        pop = neat.Population(config)
    # console.log("[blue]🧬 Creating new NEAT population...")
    # pop = neat.Population(config)
    console.log("~[green]📊 Adding reporters and checkpointers...")
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix="neat_checkpoints/neat-checkpoint-"))

    cpu_count = multiprocessing.cpu_count()
    console.log(f"~[magenta]🧠 Using {cpu_count} CPU cores for parallel evaluation")

    safe_eval = SafeEvaluator(output_bounds, evaluator)
    pe = neat.ParallelEvaluator(cpu_count, safe_eval)

    console.log("~[white]🔁 Running NEAT evolution (max 300 generations)...")
    winner = pop.run(pe.evaluate, 300)

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    console.log("~[bold green]🏁 Evolution completed!")
    return net.activate([]), winner.fitness

async def main():
    pb_config = "/home/myusuf/Projects/passivbot/configs/optimize.json"
    neat_config = "/home/myusuf/Projects/passivbot/src/neat_config.txt"
    evaluator = await initEvaluator(pb_config)
    console.print(Panel("🤖 [bold]NEAT Optimizer[/bold] initialized", style="bold blue"))

    output_bounds = load_output_bounds(pb_config)

    try:
        result, fitness = run_neat(neat_config, output_bounds, evaluator)
    except Exception as e:
        console.log(f"~[bold red]🔥 NEAT failed during execution: {e}")
        return

    console.print("\n🎯 ~[bold]Final 50 Outputs:[/bold]", style="cyan")
    for i, val in enumerate(result, 1):
        console.print(f"[{i:02d}] {val:.4f}", style="bright_white")

    console.print(f"\n🏆 ~[bold green]Final Fitness: {fitness:.4f}", style="bold green")

if __name__ == "__main__":
    asyncio.run(main())
