"""Visualization utilities for Genetic Algorithm runs.

All functions return the Axes object so callers can compose subplots or
apply further customisation without additional imports.

Usage example:
    from genetic_algorithms.visualization import plot_fitness_history
    ga = BinaryGA(...)
    ga.evolve()
    ax = plot_fitness_history(ga, title="BinaryGA on sin(x)*cos(y)")
    import matplotlib.pyplot as plt
    plt.show()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .adaptive import AdaptiveGA
    from .base import GeneticAlgorithmBase
    from .island import IslandModelGA


def plot_fitness_history(
    ga: "GeneticAlgorithmBase",
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    color: str = "steelblue",
) -> plt.Axes:
    """Plot best fitness score versus generation number.

    Works for any GA variant that populates a `.history` list of
    (generation, best_score) tuples during `evolve()`.

    Parameters
    ----------
    ga : GeneticAlgorithmBase or IslandModelGA
        A GA instance that has already been evolved.
    ax : matplotlib.axes.Axes or None
        Axes to draw on. A new figure/axes is created if None.
    title : str or None
        Plot title. Defaults to "Fitness History".
    color : str
        Line colour.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if not ga.history:
        raise ValueError("No history found. Run ga.evolve() first.")

    generations, scores = zip(*ga.history)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(generations, scores, color=color, linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness Score")
    ax.set_title(title or "Fitness History")
    ax.grid(True, alpha=0.3)
    return ax


def plot_population_2d(
    ga: "GeneticAlgorithmBase",
    *,
    ax: Optional[plt.Axes] = None,
    show_best: bool = True,
    title: Optional[str] = None,
    alpha: float = 0.4,
) -> plt.Axes:
    """Scatter plot of the current population in 2-D space.

    Only meaningful for problems with exactly 2 dimensions.

    Parameters
    ----------
    ga : GeneticAlgorithmBase
        A GA instance (before or after evolve).
    ax : matplotlib.axes.Axes or None
    show_best : bool
        If True, overlay the best individual found as a gold star.
    title : str or None
    alpha : float
        Point transparency.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ga.n_dims != 2:
        raise ValueError(
            f"plot_population_2d requires 2 dimensions, got {ga.n_dims}."
        )

    decoded = [ga._decode(ind) for ind in ga.population]
    xs = [d[0] for d in decoded]
    ys = [d[1] for d in decoded]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(xs, ys, alpha=alpha, s=20, color="steelblue", label="Population")

    if show_best and ga.best_solution is not None:
        bx, by = ga.best_solution
        ax.scatter(
            [bx], [by],
            marker="*", s=300, color="gold", edgecolors="black",
            linewidths=0.8, zorder=5, label=f"Best ({ga.best_score:.4f})"
        )

    lo0, hi0 = ga.bounds[0]
    lo1, hi1 = ga.bounds[1]
    ax.set_xlim(lo0, hi0)
    ax.set_ylim(lo1, hi1)
    ax.set_xlabel("x₀")
    ax.set_ylabel("x₁")
    ax.set_title(title or "Population Distribution")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    return ax


def plot_convergence_comparison(
    *gas: "GeneticAlgorithmBase",
    labels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Overlay fitness history curves for multiple GA runs.

    Useful for comparing BinaryGA vs RealValuedGA vs AdaptiveGA on the same
    problem.

    Parameters
    ----------
    *gas : GeneticAlgorithmBase instances
        Two or more evolved GA objects.
    labels : list of str or None
        Legend labels for each GA. Defaults to GA class names.
    ax : matplotlib.axes.Axes or None
    title : str or None

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if len(gas) < 2:
        raise ValueError("Provide at least two GA instances for comparison.")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, ga in enumerate(gas):
        if not ga.history:
            raise ValueError(
                f"GA {i} has no history. Run ga.evolve() first."
            )
        label = (labels[i] if labels else None) or type(ga).__name__
        generations, scores = zip(*ga.history)
        ax.plot(
            generations, scores,
            label=label,
            color=colors[i % len(colors)],
            linewidth=2,
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness Score")
    ax.set_title(title or "Convergence Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_sigma_adaptation(
    adaptive_ga: "AdaptiveGA",
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Dual-axis plot: fitness history (left) and σ adaptation (right).

    Only valid for AdaptiveGA instances.

    Parameters
    ----------
    adaptive_ga : AdaptiveGA
        An evolved AdaptiveGA instance.
    ax : matplotlib.axes.Axes or None
        Primary axes. A secondary axes is created automatically.
    title : str or None

    Returns
    -------
    ax : matplotlib.axes.Axes
        The primary (fitness) axes.
    """
    if not adaptive_ga.history:
        raise ValueError("No history found. Run adaptive_ga.evolve() first.")
    if not adaptive_ga.sigma_history:
        raise ValueError(
            "No sigma history. Ensure this is an AdaptiveGA instance."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    # --- Fitness (left axis) ---
    gen_f, scores = zip(*adaptive_ga.history)
    ax.plot(
        gen_f, scores,
        color="steelblue", linewidth=2, marker="o", markersize=3,
        label="Best fitness",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness Score", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")

    # --- Sigma (right axis) ---
    ax2 = ax.twinx()
    gen_s, sigmas = zip(*adaptive_ga.sigma_history)
    ax2.plot(
        gen_s, sigmas,
        color="coral", linewidth=2, linestyle="--", marker="s", markersize=3,
        label="Mean σ",
    )
    ax2.set_ylabel("Mutation σ", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax.set_title(title or "Fitness & Adaptive σ Over Generations")
    ax.grid(True, alpha=0.2)
    return ax
