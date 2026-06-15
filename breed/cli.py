"""Click-based CLI for the agentbreed evolutionary breeding framework.

Provides commands for initialising projects, running breeding loops,
inspecting results, tracing lineage, diffing generations, and exporting
genomes.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import click
import yaml

from breed import __version__

# ---------------------------------------------------------------------------
# Optional-dependency guards
# ---------------------------------------------------------------------------

_RICH_AVAILABLE = False
try:
    from rich.columns import Columns  # noqa: F401
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree

    _RICH_AVAILABLE = True
except ImportError:
    pass

_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


# Web UI removed -- breed is terminal-only.


# ---------------------------------------------------------------------------
# Emoji / Unicode safety for Windows legacy terminals
# ---------------------------------------------------------------------------

def _can_use_emoji() -> bool:
    """Check if the terminal supports emoji output."""
    if sys.platform != "win32":
        return True
    # If stdout encoding is utf-8, emoji should work
    enc = getattr(sys.stdout, "encoding", "") or ""
    return "utf" in enc.lower()


_USE_EMOJI = _can_use_emoji()

# Map of emoji to ASCII-safe alternatives
_ICONS: dict[str, str] = {
    "baby": "\U0001f476" if _USE_EMOJI else "+",
    "skull": "\U0001f480" if _USE_EMOJI else "x",
    "crown": "\U0001f451" if _USE_EMOJI else "*",
    "chart": "\U0001f4ca" if _USE_EMOJI else "#",
    "dna": "\U0001f9ec" if _USE_EMOJI else "~",
    "trophy": "\U0001f3c6" if _USE_EMOJI else "!",
    "magnifier": "\U0001f50d" if _USE_EMOJI else "?",
    "shuffle": "\U0001f500" if _USE_EMOJI else "%",
    "radioactive": "\u2622\ufe0f" if _USE_EMOJI else "^",
    "arriving": "\U0001f6ec" if _USE_EMOJI else ">",
    "target": "\U0001f3af" if _USE_EMOJI else ">",
    "bullet": "\u2022" if _USE_EMOJI else "-",
    "check": "\u2713" if _USE_EMOJI else "Y",
    "cross": "\u2717" if _USE_EMOJI else "N",
    "up": "\u25b2" if _USE_EMOJI else "^",
    "down": "\u25bc" if _USE_EMOJI else "v",
    "diamond": "\u25c6" if _USE_EMOJI else "*",
    "block": "\u2588" if _USE_EMOJI else "#",
    "light_block": "\u2591" if _USE_EMOJI else ".",
}


_BANNER_RICH = """
[bright_red]  _                        _[/]
[bright_red] | |__  _ __ ___  ___  __| |[/]
[bright_yellow] | '_ \\| '__/ _ \\/ _ \\/ _` |[/]
[bright_yellow] | |_) | | |  __/  __/ (_| |[/]
[bright_green] |_.__/|_|  \\___|\\___|\\__,_|[/]
[dim]  Darwinian selection for AI agents  v{ver}[/dim]
"""

_BANNER_PLAIN = """
  _                        _
 | |__  _ __ ___  ___  __| |
 | '_ \\| '__/ _ \\/ _ \\/ _` |
 | |_) | | |  __/  __/ (_| |
 |_.__/|_|  \\___|\\___|\\__,_|
  Darwinian selection for AI agents  v{ver}
"""


def _print_banner(console: "Console | None" = None) -> None:
    """Print the ASCII art banner."""
    if console is not None and _RICH_AVAILABLE:
        console.print(_BANNER_RICH.format(ver=__version__))
    else:
        click.echo(_BANNER_PLAIN.format(ver=__version__))


def _require_rich() -> Console:
    """Return a Rich Console or exit with an install hint."""
    if not _RICH_AVAILABLE:
        click.echo(
            "Error: 'rich' is required for terminal display.\n"
            "  pip install rich\n"
            "  -- or --\n"
            "  pip install agentbreed[all]",
            err=True,
        )
        raise SystemExit(1)
    return Console()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict[str, Any]:
    """Load and return a YAML config, exiting on error."""
    config_path = Path(path)
    if not config_path.exists():
        click.echo(f"Error: config file not found: {config_path}", err=True)
        raise SystemExit(1)
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        click.echo(f"Error: config file is not a valid YAML mapping: {config_path}", err=True)
        raise SystemExit(1)
    return data


def _resolve_lineage_path(config: dict[str, Any], config_dir: Path) -> Path:
    """Derive the lineage JSONL path from config and its parent directory."""
    logging_cfg = config.get("logging", {})
    lineage_file = logging_cfg.get("lineage_file", "lineage.jsonl")
    return config_dir / lineage_file


def _resolve_results_dir(config: dict[str, Any], config_dir: Path) -> Path:
    """Derive the results directory from config and its parent directory."""
    logging_cfg = config.get("logging", {})
    results_dir = logging_cfg.get("results_dir", "results/")
    return config_dir / results_dir


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(__version__, prog_name="breed")
def main() -> None:
    """breed -- Evolve AI agents through Darwinian selection."""


# ===================================================================
# breed init
# ===================================================================

_CONFIG_TEMPLATES: dict[str, str] = {
    "forecasting": "forecasting.yaml",
    "coding": "coding.yaml",
}


@main.command()
@click.option(
    "--arena",
    type=click.Choice(["forecasting", "coding", "custom"]),
    default="forecasting",
    help="Arena type to configure.",
)
@click.option(
    "--adapter",
    type=click.Choice(["messages", "callable", "openai"]),
    default="callable",
    help="Adapter type to configure.",
)
def init(arena: str, adapter: str) -> None:
    """Initialise a new breeding project in the current directory."""
    console = _require_rich()
    project_dir = Path.cwd()

    # --- Create directory structure ----------------------------------------
    genomes_dir = project_dir / "genomes"
    results_dir = project_dir / "results"
    genomes_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # --- Copy or generate breed.yaml --------------------------------------
    config_dest = project_dir / "breed.yaml"

    template_name = _CONFIG_TEMPLATES.get(arena)
    configs_dir = Path(__file__).resolve().parent.parent / "configs"

    if template_name and (configs_dir / template_name).exists():
        src = configs_dir / template_name
        with open(src, encoding="utf-8") as f:
            config_text = f.read()

        # Patch adapter type into the copied config
        adapter_type_map = {
            "messages": "anthropic_messages",
            "callable": "callable",
            "openai": "openai",
        }
        config_data = yaml.safe_load(config_text)
        config_data.setdefault("adapter", {})["type"] = adapter_type_map.get(adapter, adapter)
        config_text = yaml.dump(config_data, default_flow_style=False, sort_keys=False)
    else:
        # Generate a minimal config for custom arenas
        config_data = {
            "arena": {"type": arena},
            "population": {"size": 20, "elite_count": 4},
            "breeding": {
                "generations": 25,
                "crossover_rate": 0.7,
                "mutation_rate": 0.15,
                "immigration_rate": 0.1,
                "selection": "tournament",
                "tournament_size": 3,
            },
            "adapter": {"type": adapter},
            "logging": {
                "lineage_file": "lineage.jsonl",
                "results_dir": "results/",
            },
        }
        config_text = yaml.dump(config_data, default_flow_style=False, sort_keys=False)

    with open(config_dest, "w", encoding="utf-8") as f:
        f.write(config_text)

    # --- Welcome banner ---------------------------------------------------
    _print_banner(console)

    tree = Tree("[bold green]Project initialised[/]", guide_style="dim")
    tree.add("[dim]breed.yaml[/]       -- configuration")
    tree.add("[dim]genomes/[/]         -- saved genomes")
    tree.add("[dim]results/[/]         -- output & charts")
    console.print(tree)

    console.print()
    console.print(Panel(
        "[bold]Next steps:[/]\n\n"
        "  1. Edit [cyan]breed.yaml[/] to tune parameters\n"
        "  2. Run [bold green]breed run[/] to start evolution\n"
        "  3. Run [bold green]breed results[/] to view stats\n"
        "  4. Run [bold green]breed champion[/] to see the winner",
        title="[bold yellow]What now?[/]",
        border_style="yellow",
        padding=(1, 2),
    ))


# ===================================================================
# breed run
# ===================================================================

async def _demo_agent(genome_dict: dict[str, Any], task: str) -> str:
    """Demo agent that returns random probabilities -- no API key needed.

    Uses genome genes to deterministically seed the RNG so different
    genomes produce different (but reproducible) outputs.  This lets
    anyone experience the full breeding loop without any external API.
    """
    # Build a seed from the genome_id for reproducibility per genome+task
    genome_id = genome_dict.get("genome_id", "")
    seed_material = f"{genome_id}:{task}"
    seed = hash(seed_material) & 0xFFFFFFFF
    rng = random.Random(seed)

    # The temperature gene biases how extreme the prediction is
    genes = genome_dict.get("genes", {})
    temp_gene = genes.get("temperature", {})
    temperature = temp_gene.get("value", 0.7) if isinstance(temp_gene, dict) else 0.7

    # Higher temperature => more extreme predictions
    base = rng.random()
    if temperature > 0.8:
        base = base ** 0.5 if base > 0.5 else 1.0 - (1.0 - base) ** 0.5
    probability = max(0.01, min(0.99, base))

    return f"{probability:.4f}"


class _CliRichObserver:
    """Observer that collects events for the Rich live display."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.generation_stats: list[dict[str, Any]] = []
        self.current_gen: int = 0
        self.champion_id: str = ""
        self.best_fitness: float = 0.0

    async def on_event(self, event: Any) -> None:
        event_dict = event.to_dict()
        self.events.append(event_dict)

        et = event_dict.get("event_type", "")
        if et == "generation_complete":
            self.generation_stats.append(event_dict)
            self.current_gen = event_dict.get("generation", 0)
            self.champion_id = event_dict.get("champion_id", "")
            self.best_fitness = event_dict.get("best_fitness", 0.0)
        elif et == "champion_changed":
            self.champion_id = event_dict.get("new_champion_id", "")
            self.best_fitness = event_dict.get("fitness_score", 0.0)


def _build_rich_display(
    observer: _CliRichObserver,
    total_generations: int,
    population_size: int,
    current_generation: int,
    elapsed: float,
) -> Table:
    """Build the full Rich renderable for the live display."""
    # --- outer layout ---
    outer = Table.grid(padding=(0, 0))
    outer.add_column()

    # --- banner ---
    outer.add_row(Text.from_markup(_BANNER_RICH.format(ver=__version__)))

    # --- progress info ---
    pct = (current_generation / total_generations * 100) if total_generations > 0 else 0
    bar_width = 30
    filled = int(bar_width * pct / 100)
    bar = "[bold green]" + _ICONS["block"] * filled + "[/][dim]" + _ICONS["light_block"] * (bar_width - filled) + "[/]"
    elapsed_str = f"{elapsed:.1f}s"
    progress_text = (
        f"  Generation [bold]{current_generation}[/]/{total_generations}  "
        f"{bar}  {pct:.0f}%  "
        f"[dim]elapsed {elapsed_str}[/]"
    )
    outer.add_row(Text.from_markup(progress_text))
    outer.add_row(Text(""))

    # --- stats table ---
    if observer.generation_stats:
        stats_table = Table(
            title="[bold]Generation Stats[/]",
            border_style="bright_blue",
            show_lines=False,
            padding=(0, 1),
        )
        stats_table.add_column("Gen", style="bold", justify="right")
        stats_table.add_column("Best", style="bold green", justify="right")
        stats_table.add_column("Avg", style="cyan", justify="right")
        stats_table.add_column("Worst", style="red", justify="right")
        stats_table.add_column("Pop", justify="right")
        stats_table.add_column("Champion", style="bright_yellow", max_width=16)

        # Show last 8 generations
        recent = observer.generation_stats[-8:]
        for gs in recent:
            champion_short = gs.get("champion_id", "")[:12]
            stats_table.add_row(
                str(gs.get("generation", "?")),
                f"{gs.get('best_fitness', 0):.4f}",
                f"{gs.get('avg_fitness', 0):.4f}",
                f"{gs.get('worst_fitness', 0):.4f}",
                str(gs.get("population_size", "?")),
                champion_short + "...",
            )
        outer.add_row(stats_table)
        outer.add_row(Text(""))

    # --- event log ---
    event_styles: dict[str, str] = {
        "agent_born": "green",
        "agent_died": "red",
        "champion_changed": "bold bright_yellow",
        "generation_complete": "bold bright_blue",
        "breeding_started": "bold magenta",
        "breeding_complete": "bold bright_magenta",
        "agent_evaluated": "dim cyan",
        "crossover_event": "dim",
        "mutation_event": "dim",
        "immigrants_injected": "bright_cyan",
    }
    event_icons: dict[str, str] = {
        "agent_born": _ICONS["baby"],
        "agent_died": _ICONS["skull"],
        "champion_changed": _ICONS["crown"],
        "generation_complete": _ICONS["chart"],
        "breeding_started": _ICONS["dna"],
        "breeding_complete": _ICONS["trophy"],
        "agent_evaluated": _ICONS["magnifier"],
        "crossover_event": _ICONS["shuffle"],
        "mutation_event": _ICONS["radioactive"],
        "immigrants_injected": _ICONS["arriving"],
    }

    recent_events = observer.events[-10:]
    if recent_events:
        log_lines: list[Text] = []
        for ev in recent_events:
            et = ev.get("event_type", "unknown")
            style = event_styles.get(et, "dim")
            icon = event_icons.get(et, _ICONS["bullet"])
            line = Text()
            line.append(f"  {icon} ", style=style)

            if et == "agent_born":
                gid = ev.get("genome_id", "")[:10]
                gen = ev.get("generation", "?")
                line.append(f"Born: {gid}... ", style=style)
                line.append(f"(gen {gen})", style="dim")
            elif et == "agent_died":
                gid = ev.get("genome_id", "")[:10]
                fitness = ev.get("fitness_score", 0)
                line.append(f"Died: {gid}... ", style=style)
                line.append(f"(fitness {fitness:.4f})", style="dim")
            elif et == "champion_changed":
                new_id = ev.get("new_champion_id", "")[:10]
                fitness = ev.get("fitness_score", 0)
                line.append(f"New champion! {new_id}... ", style=style)
                line.append(f"(fitness {fitness:.4f})", style="bold green")
            elif et == "generation_complete":
                gen = ev.get("generation", "?")
                best = ev.get("best_fitness", 0)
                line.append(f"Gen {gen} complete ", style=style)
                line.append(f"(best {best:.4f})", style="green")
            elif et == "breeding_started":
                total_gen = ev.get("total_generations", "?")
                pop = ev.get("population_size", "?")
                line.append(f"Breeding started: {total_gen} gens, pop {pop}", style=style)
            elif et == "breeding_complete":
                cid = ev.get("champion_id", "")[:10]
                cf = ev.get("champion_fitness", 0)
                line.append(f"Complete! Champion {cid}... ", style=style)
                line.append(f"(fitness {cf:.4f})", style="bold green")
            elif et == "agent_evaluated":
                gid = ev.get("genome_id", "")[:10]
                fitness = ev.get("fitness_score", 0)
                line.append(f"Eval: {gid}... ", style=style)
                line.append(f"({fitness:.4f})", style="dim")
            elif et == "immigrants_injected":
                cnt = ev.get("count", 0)
                gen = ev.get("generation", "?")
                line.append(f"Injected {cnt} immigrants (gen {gen})", style=style)
            else:
                line.append(f"{et}", style=style)

            log_lines.append(line)

        event_panel = Panel(
            Columns([Text("\n").join(log_lines)], padding=(0, 0)),
            title="[bold]Recent Events[/]",
            border_style="dim",
            padding=(0, 1),
        )
        outer.add_row(event_panel)

    # --- current champion ---
    if observer.champion_id:
        champ_text = Text()
        champ_text.append(f"{_ICONS['crown']} Champion: ", style="bold yellow")
        champ_text.append(observer.champion_id[:16], style="bright_yellow")
        champ_text.append("   fitness: ", style="dim")
        champ_text.append(f"{observer.best_fitness:.4f}", style="bold green")
        outer.add_row(Panel(champ_text, border_style="yellow", padding=(0, 1)))

    return outer


@main.command()
@click.option("--config", default="breed.yaml", help="Path to breed config file.")
@click.option("--generations", type=int, default=None, help="Override generation count.")
@click.option("--population", type=int, default=None, help="Override population size.")
@click.option("--headless", is_flag=True, help="JSON-only output (no Rich display).")
def run(
    config: str,
    generations: int | None,
    population: int | None,
    headless: bool,
) -> None:
    """Run the evolutionary breeding loop."""
    asyncio.run(_run_async(config, generations, population, headless))


async def _run_async(
    config_path: str,
    generations_override: int | None,
    population_override: int | None,
    headless: bool,
) -> None:
    """Async core of the ``breed run`` command."""
    from breed.adapters.callable_adapter import CallableAdapter
    from breed.arenas.coding import CodingArena
    from breed.arenas.forecasting import ForecastingArena
    from breed.events import (
        AgentBorn,
        AgentDied,
        AgentEvaluated,
        BreedingComplete,
        BreedingStarted,
        ChampionChanged,
        EventBus,
        GenerationComplete,
        JSONObserver,
    )
    from breed.lineage import LineageTracker
    from breed.population import Population

    cfg = _load_config(config_path)
    config_dir = Path(config_path).resolve().parent

    # --- resolve parameters ------------------------------------------------
    arena_cfg = cfg.get("arena", {})
    arena_type = arena_cfg.get("type", "forecasting")
    pop_cfg = cfg.get("population", {})
    breed_cfg = cfg.get("breeding", {})

    total_generations = generations_override or breed_cfg.get("generations", 25)
    pop_size = population_override or pop_cfg.get("size", 20)
    elite_count = pop_cfg.get("elite_count", max(2, pop_size // 5))
    mutation_rate = breed_cfg.get("mutation_rate", 0.15)
    immigration_rate = breed_cfg.get("immigration_rate", 0.1)
    num_questions = arena_cfg.get("num_questions", 10)
    selection_method = breed_cfg.get("selection", "tournament")
    tournament_size = breed_cfg.get("tournament_size", 3)

    # --- arena setup -------------------------------------------------------
    if arena_type == "forecasting":
        arena = ForecastingArena()
    elif arena_type == "coding":
        arena = CodingArena()
    else:
        # Default to forecasting for demo
        arena = ForecastingArena()

    # --- adapter setup (demo mode if no API key) ---------------------------
    adapter_cfg = cfg.get("adapter", {})
    adapter_type = adapter_cfg.get("type", "callable")

    # Always use demo agent for callable adapter, or when no API key present
    use_demo = adapter_type == "callable" or not os.environ.get("ANTHROPIC_API_KEY")
    if use_demo:
        adapter = CallableAdapter(_demo_agent)
    else:
        # Attempt to load the real adapter
        try:
            if adapter_type == "anthropic_messages":
                from breed.adapters.anthropic_adapter import AnthropicMessagesAdapter

                model = adapter_cfg.get("model", "claude-sonnet-4-6")
                adapter = AnthropicMessagesAdapter(model=model)
            elif adapter_type == "openai":
                from breed.adapters.openai_adapter import OpenAIAdapter

                model = adapter_cfg.get("model", "gpt-4o")
                adapter = OpenAIAdapter(model=model)
            else:
                adapter = CallableAdapter(_demo_agent)
        except (ImportError, Exception):
            adapter = CallableAdapter(_demo_agent)

    # --- event bus & observers ---------------------------------------------
    event_bus = EventBus()

    lineage_path = _resolve_lineage_path(cfg, config_dir)
    results_dir = _resolve_results_dir(cfg, config_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Events go to a separate file; lineage.jsonl is reserved for LineageTracker
    events_path = results_dir / "events.jsonl"
    json_observer = JSONObserver(events_path)
    await event_bus.add_observer(json_observer)

    rich_observer: _CliRichObserver | None = None
    if not headless:
        rich_observer = _CliRichObserver()
        await event_bus.add_observer(rich_observer)

    # --- lineage tracker ---------------------------------------------------
    lineage = LineageTracker(lineage_path)

    # --- spawn population --------------------------------------------------
    pop = Population.spawn(pop_size, seed=42)

    await event_bus.emit(BreedingStarted(
        total_generations=total_generations,
        population_size=pop_size,
        arena_type=arena_type,
    ))

    # Emit born events for gen-0
    for genome in pop.members:
        await event_bus.emit(AgentBorn(
            genome_id=genome.genome_id,
            generation=0,
            parents=[],
        ))

    # --- breeding loop -----------------------------------------------------
    overall_champion_id: str | None = None
    overall_best_fitness: float = -1.0
    total_evaluations = 0
    start_time = time.monotonic()

    console: Console | None = None
    live: Live | None = None

    if not headless and _RICH_AVAILABLE:
        console = Console()
        live = Live(
            _build_rich_display(
                rich_observer,  # type: ignore[arg-type]
                total_generations,
                pop_size,
                0,
                0.0,
            ),
            console=console,
            refresh_per_second=4,
            transient=False,
        )
        live.start()

    try:
        for gen in range(total_generations):
            # Generate tasks
            tasks = await arena.generate_tasks(min(num_questions, 10), seed=gen)

            # Evaluate population
            eval_results = await arena.evaluate_population(pop.members, adapter, tasks)
            total_evaluations += len(eval_results)

            fitness_scores: list[float] = []
            for er in eval_results:
                fitness_scores.append(er.fitness_score)

                # Log evaluation event
                await event_bus.emit(AgentEvaluated(
                    genome_id=er.genome_id,
                    generation=gen,
                    fitness_score=er.fitness_score,
                    fitness_breakdown=er.fitness_breakdown,
                ))

                # Record in lineage
                genome_match = next(
                    (g for g in pop.members if g.genome_id == er.genome_id), None
                )
                if genome_match is not None:
                    lineage.log(
                        genome=genome_match,
                        fitness_score=er.fitness_score,
                        fitness_breakdown=er.fitness_breakdown,
                    )

            # Attach fitness to genomes
            for genome, score in zip(pop.members, fitness_scores):
                genome.fitness_history.append(score)

            # Generation stats
            best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            best_fitness = fitness_scores[best_idx]
            avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
            worst_fitness = min(fitness_scores) if fitness_scores else 0.0
            gen_champion = pop.members[best_idx]

            await event_bus.emit(GenerationComplete(
                generation=gen,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                worst_fitness=worst_fitness,
                population_size=pop.size,
                champion_id=gen_champion.genome_id,
            ))

            # Track overall champion
            if best_fitness > overall_best_fitness:
                old_champion = overall_champion_id
                overall_champion_id = gen_champion.genome_id
                overall_best_fitness = best_fitness
                await event_bus.emit(ChampionChanged(
                    old_champion_id=old_champion,
                    new_champion_id=gen_champion.genome_id,
                    generation=gen,
                    fitness_score=best_fitness,
                ))

            # Update live display
            if live is not None and rich_observer is not None:
                elapsed = time.monotonic() - start_time
                live.update(_build_rich_display(
                    rich_observer,
                    total_generations,
                    pop_size,
                    gen + 1,
                    elapsed,
                ))

            # --- Selection ---
            pre_select = set(g.genome_id for g in pop.members)
            pop.select(
                fitness_scores,
                top_k=elite_count,
                method=selection_method,
                tournament_size=tournament_size,
                seed=gen,
            )
            post_select = set(g.genome_id for g in pop.members)

            # Emit death events
            for dead_id in pre_select - post_select:
                dead_score = 0.0
                for er in eval_results:
                    if er.genome_id == dead_id:
                        dead_score = er.fitness_score
                        break
                await event_bus.emit(AgentDied(
                    genome_id=dead_id,
                    generation=gen,
                    fitness_score=dead_score,
                ))

            # --- Breed offspring ---
            # Build per-survivor fitness list aligned with post-selection members
            id_to_fitness = {er.genome_id: er.fitness_score for er in eval_results}
            elite_fitness = [id_to_fitness.get(g.genome_id, 0.0) for g in pop.members]
            pop.breed(target_size=pop_size, seed=gen, fitness_scores=elite_fitness)

            # Emit born events for new children
            for genome in pop.members:
                if genome.genome_id not in post_select:
                    await event_bus.emit(AgentBorn(
                        genome_id=genome.genome_id,
                        generation=gen + 1,
                        parents=genome.parents,
                    ))

            # --- Mutate ---
            pop.mutate(rate=mutation_rate, seed=gen)

            # --- Immigration ---
            immigration_count = max(1, int(pop_size * immigration_rate))
            if gen < total_generations - 1:  # no immigration on last gen
                pop.add_immigrants(immigration_count, seed=gen)

    finally:
        if live is not None:
            live.stop()

    # --- Breeding complete -------------------------------------------------
    await event_bus.emit(BreedingComplete(
        final_generation=total_generations - 1,
        champion_id=overall_champion_id or "",
        champion_fitness=overall_best_fitness,
        total_evaluations=total_evaluations,
    ))

    # --- Final summary -----------------------------------------------------
    if not headless and _RICH_AVAILABLE:
        console = console or Console()
        elapsed = time.monotonic() - start_time

        console.print()
        console.print(Rule("[bold bright_magenta]Breeding Complete[/]", style="bright_magenta"))
        console.print()

        summary = Table(show_header=False, border_style="green", padding=(0, 2))
        summary.add_column("Key", style="bold")
        summary.add_column("Value")
        summary.add_row("Generations", str(total_generations))
        summary.add_row("Population size", str(pop_size))
        summary.add_row("Total evaluations", str(total_evaluations))
        summary.add_row("Elapsed", f"{elapsed:.1f}s")
        summary.add_row(
            f"{_ICONS['crown']} Champion",
            f"[bold bright_yellow]{(overall_champion_id or 'N/A')[:20]}[/]",
        )
        summary.add_row(
            f"{_ICONS['target']} Best fitness",
            f"[bold green]{overall_best_fitness:.4f}[/]",
        )
        console.print(Panel(summary, title="[bold]Summary[/]", border_style="bright_green"))
        console.print()
        console.print(
            "[dim]Results saved to:[/] "
            f"[cyan]{lineage_path}[/]"
        )
        console.print(
            "[dim]View results:[/]    "
            "[bold green]breed results[/]  |  "
            "[bold green]breed champion[/]  |  "
            "[bold green]breed lineage[/]"
        )
        console.print()
    elif headless:
        click.echo(json.dumps({
            "status": "complete",
            "generations": total_generations,
            "champion_id": overall_champion_id,
            "champion_fitness": overall_best_fitness,
            "total_evaluations": total_evaluations,
        }))


# ===================================================================
# breed results
# ===================================================================

@main.command()
@click.option("--config", default="breed.yaml", help="Path to breed config file.")
def results(config: str) -> None:
    """Display summary statistics and optionally generate charts."""
    console = _require_rich()
    cfg = _load_config(config)
    config_dir = Path(config).resolve().parent

    from breed.lineage import LineageTracker

    lineage_path = _resolve_lineage_path(cfg, config_dir)
    tracker = LineageTracker(lineage_path)
    records = tracker.load()

    if not records:
        console.print("[yellow]No lineage records found.[/] Run [bold]breed run[/] first.")
        return

    # --- Group by generation -----------------------------------------------
    gen_data: dict[int, list[float]] = defaultdict(list)
    for r in records:
        gen_data[r.generation].append(r.fitness_score)

    # --- Summary table -----------------------------------------------------
    console.print()
    console.print(Rule("[bold bright_magenta]Breeding Results[/]", style="bright_magenta"))
    console.print()

    table = Table(
        title="[bold]Per-Generation Summary[/]",
        border_style="bright_blue",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Gen", style="bold", justify="right")
    table.add_column("Best", style="bold green", justify="right")
    table.add_column("Avg", style="cyan", justify="right")
    table.add_column("Worst", style="red", justify="right")
    table.add_column("Count", justify="right")

    for gen_num in sorted(gen_data.keys()):
        scores = gen_data[gen_num]
        table.add_row(
            str(gen_num),
            f"{max(scores):.4f}",
            f"{sum(scores) / len(scores):.4f}",
            f"{min(scores):.4f}",
            str(len(scores)),
        )

    console.print(table)

    # --- Overall stats -----------------------------------------------------
    overall_best = max(records, key=lambda r: r.fitness_score)

    console.print()
    info = Table(show_header=False, border_style="green", padding=(0, 2))
    info.add_column("Key", style="bold")
    info.add_column("Value")
    info.add_row("Total records", str(len(records)))
    info.add_row("Generations", str(len(gen_data)))
    info.add_row("Overall best fitness", f"[bold green]{overall_best.fitness_score:.4f}[/]")
    info.add_row(
        f"{_ICONS['crown']} Overall champion",
        f"[bold bright_yellow]{overall_best.genome_id[:20]}[/]",
    )
    console.print(Panel(info, title="[bold]Overall[/]", border_style="bright_green"))

    # --- Charts (optional) -------------------------------------------------
    results_dir = _resolve_results_dir(cfg, config_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if _MATPLOTLIB_AVAILABLE:
        gens_sorted = sorted(gen_data.keys())
        bests = [max(gen_data[g]) for g in gens_sorted]
        avgs = [sum(gen_data[g]) / len(gen_data[g]) for g in gens_sorted]
        worsts = [min(gen_data[g]) for g in gens_sorted]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(gens_sorted, bests, "g-o", label="Best", markersize=4)
        ax.plot(gens_sorted, avgs, "b-s", label="Average", markersize=3)
        ax.plot(gens_sorted, worsts, "r-^", label="Worst", markersize=3)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness Over Generations")
        ax.legend()
        ax.grid(True, alpha=0.3)

        chart_path = results_dir / "fitness_over_generations.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.print(f"\n[dim]Chart saved:[/] [cyan]{chart_path}[/]")
    else:
        console.print(
            "\n[dim]Install matplotlib for charts:[/] "
            "[cyan]pip install agentbreed[charts][/]"
        )

    console.print()


# ===================================================================
# breed champion
# ===================================================================

@main.command()
@click.option("--config", default="breed.yaml", help="Path to breed config file.")
def champion(config: str) -> None:
    """Display the overall champion genome."""
    console = _require_rich()
    cfg = _load_config(config)
    config_dir = Path(config).resolve().parent

    from breed.lineage import LineageTracker

    lineage_path = _resolve_lineage_path(cfg, config_dir)
    tracker = LineageTracker(lineage_path)
    records = tracker.load()

    if not records:
        console.print("[yellow]No lineage records found.[/] Run [bold]breed run[/] first.")
        return

    best = max(records, key=lambda r: r.fitness_score)

    console.print()
    console.print(Rule("[bold bright_yellow]Champion Genome[/]", style="bright_yellow"))
    console.print()

    # --- Info panel ---------------------------------------------------------
    info = Table(show_header=False, border_style="yellow", padding=(0, 2))
    info.add_column("Key", style="bold")
    info.add_column("Value")
    info.add_row("Genome ID", f"[bright_yellow]{best.genome_id}[/]")
    info.add_row("Generation", str(best.generation))
    info.add_row("Fitness", f"[bold green]{best.fitness_score:.4f}[/]")
    info.add_row("Parents", ", ".join(best.parents) if best.parents else "seed (Gen-0)")
    if best.fitness_breakdown:
        breakdown_str = "  ".join(
            f"{k}: {v:.4f}" for k, v in best.fitness_breakdown.items()
        )
        info.add_row("Breakdown", breakdown_str)
    console.print(Panel(info, title=f"[bold]{_ICONS['crown']} Champion[/]", border_style="bright_yellow"))

    # --- Genome YAML -------------------------------------------------------
    snapshot = best.genome_snapshot
    if snapshot:
        genome_yaml = yaml.dump(
            snapshot.get("genes", snapshot),
            default_flow_style=False,
            sort_keys=False,
        )
        console.print()
        console.print(Panel(
            genome_yaml,
            title="[bold]Genome (YAML)[/]",
            border_style="cyan",
            padding=(1, 2),
        ))

    # --- Fitness history ---------------------------------------------------
    # Collect all records for this genome to show its history
    history = [r for r in records if r.genome_id == best.genome_id]
    if len(history) > 1:
        console.print()
        hist_table = Table(
            title="[bold]Fitness History[/]",
            border_style="green",
        )
        hist_table.add_column("Generation", justify="right")
        hist_table.add_column("Fitness", style="green", justify="right")
        hist_table.add_column("Survived", justify="center")

        for h in sorted(history, key=lambda r: r.generation):
            survived_mark = f"[green]{_ICONS['check']}[/]" if h.survived else f"[red]{_ICONS['cross']}[/]"
            hist_table.add_row(str(h.generation), f"{h.fitness_score:.4f}", survived_mark)
        console.print(hist_table)

    console.print()


# ===================================================================
# breed lineage
# ===================================================================

@main.command()
@click.option("--genome-id", default=None, help="Trace ancestry of a specific genome.")
@click.option(
    "--format", "fmt",
    type=click.Choice(["tree", "table"]),
    default="table",
    help="Output format.",
)
@click.option("--config", default="breed.yaml", help="Path to breed config file.")
def lineage(genome_id: str | None, fmt: str, config: str) -> None:
    """Trace the ancestry of a genome (default: the champion)."""
    console = _require_rich()
    cfg = _load_config(config)
    config_dir = Path(config).resolve().parent

    from breed.lineage import LineageTracker

    lineage_path = _resolve_lineage_path(cfg, config_dir)
    tracker = LineageTracker(lineage_path)
    records = tracker.load()

    if not records:
        console.print("[yellow]No lineage records found.[/] Run [bold]breed run[/] first.")
        return

    # Default to the champion
    if genome_id is None:
        best = max(records, key=lambda r: r.fitness_score)
        genome_id = best.genome_id

    ancestry = tracker.get_lineage(genome_id)
    if not ancestry:
        console.print(f"[yellow]No lineage found for genome {genome_id}[/]")
        return

    console.print()
    console.print(Rule(
        f"[bold bright_cyan]Lineage: {genome_id[:16]}...[/]",
        style="bright_cyan",
    ))
    console.print()

    if fmt == "tree":
        _print_lineage_tree(console, ancestry, genome_id)
    else:
        _print_lineage_table(console, ancestry)

    console.print()


def _print_lineage_tree(
    console: Console,
    ancestry: list[Any],
    target_id: str,
) -> None:
    """Render lineage as an ASCII tree via Rich."""
    root = Tree(
        f"[bold bright_yellow]{_ICONS['crown']} {target_id[:16]}...[/]",
        guide_style="bright_cyan",
    )

    # Build lookup
    by_id: dict[str, Any] = {}
    for rec in ancestry:
        by_id[rec.genome_id] = rec

    # The target is the root; parents branch out
    target = by_id.get(target_id)
    if target is not None:
        _add_parent_branches(root, target, by_id, visited=set())

    console.print(root)


def _add_parent_branches(
    node: Tree,
    record: Any,
    by_id: dict[str, Any],
    visited: set[str],
) -> None:
    """Recursively add parent branches to the tree."""
    if record.genome_id in visited:
        return
    visited.add(record.genome_id)

    for parent_id in record.parents:
        parent_rec = by_id.get(parent_id)
        if parent_rec is not None:
            label = (
                f"[dim]{parent_id[:12]}...[/]  "
                f"gen={parent_rec.generation}  "
                f"fitness=[green]{parent_rec.fitness_score:.4f}[/]"
            )
            child_node = node.add(label)
            _add_parent_branches(child_node, parent_rec, by_id, visited)
        else:
            node.add(f"[dim]{parent_id[:12]}... (no record)[/]")


def _print_lineage_table(console: Console, ancestry: list[Any]) -> None:
    """Render lineage as a Rich table."""
    table = Table(
        title="[bold]Ancestry Chain[/]",
        border_style="bright_cyan",
    )
    table.add_column("Genome ID", style="bright_yellow", max_width=20)
    table.add_column("Gen", justify="right")
    table.add_column("Fitness", style="green", justify="right")
    table.add_column("Parents", style="dim", max_width=30)
    table.add_column("Survived", justify="center")

    for rec in sorted(ancestry, key=lambda r: r.generation):
        parents_str = ", ".join(p[:10] + "..." for p in rec.parents) if rec.parents else "seed"
        survived_mark = f"[green]{_ICONS['check']}[/]" if rec.survived else f"[red]{_ICONS['cross']}[/]"
        table.add_row(
            rec.genome_id[:18] + "...",
            str(rec.generation),
            f"{rec.fitness_score:.4f}",
            parents_str,
            survived_mark,
        )

    console.print(table)


# ===================================================================
# breed diff
# ===================================================================

@main.command()
@click.option("--gen-a", type=int, required=True, help="First generation to compare.")
@click.option("--gen-b", type=int, required=True, help="Second generation to compare.")
@click.option("--config", default="breed.yaml", help="Path to breed config file.")
def diff(gen_a: int, gen_b: int, config: str) -> None:
    """Compare the champion of two different generations."""
    console = _require_rich()
    cfg = _load_config(config)
    config_dir = Path(config).resolve().parent

    from breed.lineage import LineageTracker

    lineage_path = _resolve_lineage_path(cfg, config_dir)
    tracker = LineageTracker(lineage_path)

    champ_a = tracker.get_champion(gen_a)
    champ_b = tracker.get_champion(gen_b)

    if champ_a is None:
        console.print(f"[red]No records found for generation {gen_a}[/]")
        return
    if champ_b is None:
        console.print(f"[red]No records found for generation {gen_b}[/]")
        return

    console.print()
    console.print(Rule(
        f"[bold]Diff: Gen {gen_a} vs Gen {gen_b}[/]",
        style="bright_magenta",
    ))
    console.print()

    # --- Fitness comparison ------------------------------------------------
    fitness_table = Table(
        title="[bold]Fitness Comparison[/]",
        border_style="bright_blue",
    )
    fitness_table.add_column("Metric", style="bold")
    fitness_table.add_column(f"Gen {gen_a}", justify="right")
    fitness_table.add_column(f"Gen {gen_b}", justify="right")
    fitness_table.add_column("Delta", justify="right")

    delta_f = champ_b.fitness_score - champ_a.fitness_score
    delta_color = "green" if delta_f > 0 else ("red" if delta_f < 0 else "yellow")
    delta_sign = "+" if delta_f > 0 else ""
    fitness_table.add_row(
        "Overall fitness",
        f"{champ_a.fitness_score:.4f}",
        f"{champ_b.fitness_score:.4f}",
        f"[{delta_color}]{delta_sign}{delta_f:.4f}[/]",
    )

    # Breakdown comparison
    all_metrics = set(champ_a.fitness_breakdown.keys()) | set(champ_b.fitness_breakdown.keys())
    for metric in sorted(all_metrics):
        val_a = champ_a.fitness_breakdown.get(metric, 0.0)
        val_b = champ_b.fitness_breakdown.get(metric, 0.0)
        d = val_b - val_a
        dc = "green" if d > 0 else ("red" if d < 0 else "yellow")
        ds = "+" if d > 0 else ""
        fitness_table.add_row(
            metric,
            f"{val_a:.4f}",
            f"{val_b:.4f}",
            f"[{dc}]{ds}{d:.4f}[/]",
        )

    console.print(fitness_table)

    # --- Gene-by-gene diff -------------------------------------------------
    genes_a = champ_a.genome_snapshot.get("genes", {})
    genes_b = champ_b.genome_snapshot.get("genes", {})
    all_genes = set(genes_a.keys()) | set(genes_b.keys())

    if all_genes:
        console.print()
        gene_table = Table(
            title="[bold]Gene-by-Gene Diff[/]",
            border_style="bright_cyan",
        )
        gene_table.add_column("Gene", style="bold")
        gene_table.add_column(f"Gen {gen_a}", max_width=35)
        gene_table.add_column(f"Gen {gen_b}", max_width=35)
        gene_table.add_column("Status", justify="center")

        for gene_name in sorted(all_genes):
            ga = genes_a.get(gene_name, {})
            gb = genes_b.get(gene_name, {})
            val_a = ga.get("value", "N/A") if isinstance(ga, dict) else ga
            val_b = gb.get("value", "N/A") if isinstance(gb, dict) else gb

            val_a_str = _format_gene_value(val_a)
            val_b_str = _format_gene_value(val_b)

            if val_a == val_b:
                status = "[dim]=[/]"
            elif isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                status = (
                    f"[green]{_ICONS['up']} improved[/]" if val_b > val_a
                    else f"[red]{_ICONS['down']} degraded[/]"
                )
            else:
                status = f"[yellow]{_ICONS['diamond']} changed[/]"

            gene_table.add_row(gene_name, val_a_str, val_b_str, status)

        console.print(gene_table)

    console.print()


def _format_gene_value(val: Any) -> str:
    """Format a gene value for display, truncating long strings."""
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, list):
        items = ", ".join(str(v) for v in val)
        if len(items) > 30:
            return items[:27] + "..."
        return items
    s = str(val)
    if len(s) > 35:
        return s[:32] + "..."
    return s


# ===================================================================
# breed export
# ===================================================================

@main.command()
@click.option("--genome-id", default=None, help="Genome to export (default: champion).")
@click.option(
    "--format", "fmt",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    help="Export format.",
)
@click.option("--config", default="breed.yaml", help="Path to breed config file.")
def export(genome_id: str | None, fmt: str, config: str) -> None:
    """Export a genome as YAML or JSON."""
    cfg = _load_config(config)
    config_dir = Path(config).resolve().parent

    from breed.lineage import LineageTracker

    lineage_path = _resolve_lineage_path(cfg, config_dir)
    tracker = LineageTracker(lineage_path)
    records = tracker.load()

    if not records:
        click.echo("Error: No lineage records found. Run 'breed run' first.", err=True)
        raise SystemExit(1)

    # Find the target genome
    if genome_id is None:
        target = max(records, key=lambda r: r.fitness_score)
    else:
        matches = [r for r in records if r.genome_id == genome_id]
        if not matches:
            click.echo(f"Error: genome {genome_id} not found in lineage.", err=True)
            raise SystemExit(1)
        target = matches[0]

    snapshot = target.genome_snapshot
    if not snapshot:
        click.echo("Error: no genome snapshot available for this record.", err=True)
        raise SystemExit(1)

    if fmt == "yaml":
        output = yaml.dump(snapshot, default_flow_style=False, sort_keys=False)
    else:
        output = json.dumps(snapshot, indent=2, default=str)

    click.echo(output)




# ===================================================================
# __main__ support
# ===================================================================

if __name__ == "__main__":
    main()
