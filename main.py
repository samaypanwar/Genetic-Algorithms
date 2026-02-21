"""Demonstration of all Genetic Algorithm variants.

Fitness function (same as the original project):
    z = sin(x) * cos(y) + sin(y²) * x

All four variants maximise this function over [-10, 10]².
"""

import time

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend — change to "TkAgg" for pop-up windows
import matplotlib.pyplot as plt

from genetic_algorithms import AdaptiveGA, BinaryGA, IslandModelGA, RealValuedGA
from genetic_algorithms.visualization import (
    plot_convergence_comparison,
    plot_fitness_history,
    plot_population_2d,
    plot_sigma_adaptation,
)

BOUNDS = [(-10.0, 10.0), (-10.0, 10.0)]


def fitness(individual):
    x, y = individual[0], individual[1]
    return np.sin(x) * np.cos(y) + np.sin(y**2) * x


# ---------------------------------------------------------------------------
# 1. BinaryGA — fixed & improved version of the original
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("BinaryGA (fixed + improved)")
print("=" * 60)
t0 = time.time()
binary_ga = BinaryGA(
    fitness,
    BOUNDS,
    n_bits=16,
    pop_size=200,
    n_generations=50,
    maximize=True,
    elitism=2,
    patience=15,
    tol=1e-5,
)
binary_solution = binary_ga.evolve()
print(f"\nBest solution : x={binary_solution[0]:.4f}, y={binary_solution[1]:.4f}")
print(f"Best score    : {binary_ga.best_score:.6f}")
print(f"Time          : {time.time() - t0:.2f}s")

# ---------------------------------------------------------------------------
# 2. RealValuedGA — Extension 1
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RealValuedGA (float encoding, SBX crossover)")
print("=" * 60)
t0 = time.time()
real_ga = RealValuedGA(
    fitness,
    BOUNDS,
    pop_size=200,
    n_generations=50,
    sbx_eta=2.0,
    maximize=True,
    elitism=2,
    patience=15,
    tol=1e-5,
)
real_solution = real_ga.evolve()
print(f"\nBest solution : x={real_solution[0]:.4f}, y={real_solution[1]:.4f}")
print(f"Best score    : {real_ga.best_score:.6f}")
print(f"Time          : {time.time() - t0:.2f}s")

# ---------------------------------------------------------------------------
# 3. AdaptiveGA — Extension 4
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("AdaptiveGA (self-adaptive σ)")
print("=" * 60)
t0 = time.time()
adaptive_ga = AdaptiveGA(
    fitness,
    BOUNDS,
    pop_size=200,
    n_generations=50,
    maximize=True,
    elitism=2,
    patience=15,
    tol=1e-5,
    scale_up=1.2,
    scale_down=0.9,
)
adaptive_solution = adaptive_ga.evolve()
print(f"\nBest solution : x={adaptive_solution[0]:.4f}, y={adaptive_solution[1]:.4f}")
print(f"Best score    : {adaptive_ga.best_score:.6f}")
print(f"Time          : {time.time() - t0:.2f}s")

# ---------------------------------------------------------------------------
# 4. IslandModelGA — Extension 7
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("IslandModelGA (4 islands, ring migration)")
print("=" * 60)
t0 = time.time()
island_ga = IslandModelGA(
    fitness,
    BOUNDS,
    n_islands=4,
    pop_size_per_island=50,
    n_generations=40,
    migration_interval=10,
    n_migrants=2,
    maximize=True,
    elitism=2,
)
island_solution = island_ga.evolve()
print(f"\nBest solution : x={island_solution[0]:.4f}, y={island_solution[1]:.4f}")
print(f"Best score    : {island_ga.best_score:.6f}")
print(f"Time          : {time.time() - t0:.2f}s")

# ---------------------------------------------------------------------------
# Visualizations — Extension 9
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Generating plots ...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Genetic Algorithm Variants — Fitness History", fontsize=14)

plot_fitness_history(binary_ga, ax=axes[0, 0], title="BinaryGA", color="steelblue")
plot_fitness_history(real_ga, ax=axes[0, 1], title="RealValuedGA", color="seagreen")
plot_fitness_history(adaptive_ga, ax=axes[1, 0], title="AdaptiveGA", color="darkorange")

# Island GA history uses migration rounds on x-axis
gens, scores = zip(*island_ga.history)
axes[1, 1].plot(gens, scores, color="mediumpurple", linewidth=2, marker="o", markersize=4)
axes[1, 1].set_xlabel("Generation")
axes[1, 1].set_ylabel("Best Fitness Score")
axes[1, 1].set_title("IslandModelGA")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fitness_history.png", dpi=120)
print("Saved fitness_history.png")

# Convergence comparison (BinaryGA vs RealValuedGA vs AdaptiveGA)
fig2, ax2 = plt.subplots(figsize=(9, 5))
plot_convergence_comparison(
    binary_ga, real_ga, adaptive_ga,
    labels=["BinaryGA", "RealValuedGA", "AdaptiveGA"],
    ax=ax2,
    title="Convergence Comparison",
)
plt.tight_layout()
plt.savefig("convergence_comparison.png", dpi=120)
print("Saved convergence_comparison.png")

# Population scatter (RealValuedGA — float individuals can be plotted directly)
fig3, ax3 = plt.subplots(figsize=(6, 6))
plot_population_2d(real_ga, ax=ax3, title="RealValuedGA — Final Population")
plt.tight_layout()
plt.savefig("population_2d.png", dpi=120)
print("Saved population_2d.png")

# Adaptive sigma plot
fig4, ax4 = plt.subplots(figsize=(9, 5))
plot_sigma_adaptation(adaptive_ga, ax=ax4, title="AdaptiveGA — Fitness & sigma Adaptation")
plt.tight_layout()
plt.savefig("sigma_adaptation.png", dpi=120)
print("Saved sigma_adaptation.png")

print("\nDone.")
