"""genetic_algorithms — Modular Genetic Algorithm library.

Available classes
-----------------
BinaryGA
    Binary-encoded GA (fixed + improved version of the original).
RealValuedGA
    Float-encoded GA with SBX crossover and Gaussian mutation.
AdaptiveGA
    RealValuedGA with self-adaptive mutation σ based on population diversity.
IslandModelGA
    Multi-population island model with ring-topology migration.

Visualization helpers
---------------------
from genetic_algorithms.visualization import (
    plot_fitness_history,
    plot_population_2d,
    plot_convergence_comparison,
    plot_sigma_adaptation,
)
"""

from .adaptive import AdaptiveGA
from .binary import BinaryGA
from .island import IslandModelGA
from .real_valued import RealValuedGA

__all__ = [
    "BinaryGA",
    "RealValuedGA",
    "AdaptiveGA",
    "IslandModelGA",
]
