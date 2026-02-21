# Genetic-Algorithms

A modular Genetic Algorithm library for continuous function optimisation, supporting binary and real-valued encoding, adaptive mutation, island-model multi-population search, and built-in visualisation.

## Installation

Requires [Poetry](https://python-poetry.org/).

```bash
poetry install
```

## Quick Start

```python
from genetic_algorithms import RealValuedGA

bounds = [(-10.0, 10.0), (-10.0, 10.0)]

def fitness(individual):
    x, y = individual
    return x * y - x**2

ga = RealValuedGA(fitness, bounds, pop_size=200, n_generations=100, maximize=True)
solution = ga.evolve()
print(solution, ga.best_score)
```

## Available Classes

| Class | Description |
|---|---|
| `BinaryGA` | Binary bitstring encoding — fixed & improved original |
| `RealValuedGA` | Float encoding with SBX crossover + Gaussian mutation |
| `AdaptiveGA` | `RealValuedGA` with diversity-driven adaptive σ |
| `IslandModelGA` | Multi-population island model with ring migration |

## Common Parameters

All GA classes share these keyword arguments:

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 200 | Individuals per generation |
| `n_generations` | 100 | Maximum generations |
| `crossover_rate` | 0.9 | Probability of crossover |
| `maximize` | `True` | `False` to minimise |
| `elitism` | 2 | Best individuals preserved per generation |
| `tournament_size` | 10 | Tournament selection pool size |
| `patience` | `None` | Early stopping patience (generations) |
| `tol` | 1e-6 | Early stopping improvement threshold |
| `n_jobs` | 1 | Worker processes for fitness evaluation |

`BinaryGA` also accepts `n_bits` (default 16).
`RealValuedGA` also accepts `sigma` and `sbx_eta`.
`AdaptiveGA` also accepts `low_threshold`, `high_threshold`, `scale_up`, `scale_down`.

## Visualisation

```python
from genetic_algorithms.visualization import (
    plot_fitness_history,
    plot_population_2d,
    plot_convergence_comparison,
    plot_sigma_adaptation,
)
import matplotlib.pyplot as plt

plot_fitness_history(ga)
plt.show()
```

## Running the Demo

```bash
poetry run python main.py
```

Produces four PNG plots: `fitness_history.png`, `convergence_comparison.png`,
`population_2d.png`, and `sigma_adaptation.png`.
