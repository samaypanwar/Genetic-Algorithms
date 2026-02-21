from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from .base import GeneticAlgorithmBase


class BinaryGA(GeneticAlgorithmBase):
    """Genetic Algorithm with binary (bitstring) encoding.

    Individuals are represented as flat NumPy arrays of 0/1 bits.
    Each dimension uses `n_bits` bits, so an individual has length
    `n_bits * n_dims`.

    Bug fixes vs the original implementation:
      - Mutation now iterates over *all* bits (n_bits * n_dims), not
        just n_bits (which left second and higher dimensions unmutated).
      - best_individual is initialised to an actual individual copy,
        not the integer 0 (which would crash decode() if no improvement
        occurred during evolution).
      - select_parent returns a copy, preventing crossover from silently
        modifying the parent stored in self.population.

    Parameters
    ----------
    fitness_fn : callable
        Function that accepts a list of floats [x0, x1, ...] and returns
        a scalar fitness value.
    bounds : list of (float, float)
        Search-space bounds per dimension, e.g. [(-10, 10), (-10, 10)].
    n_bits : int
        Number of bits used to encode each dimension (resolution).
    pop_size : int
        Number of individuals in the population.
    n_generations : int
        Maximum number of generations to run.
    crossover_rate : float
        Probability that crossover occurs between two selected parents.
    maximize : bool
        True to maximise the fitness function, False to minimise.
    elitism : int
        Number of best individuals copied unchanged into each new generation.
    tournament_size : int
        Number of candidates considered per tournament selection event.
    patience : int or None
        Stop early if best score improves by less than `tol` over this many
        consecutive generations. None disables early stopping.
    tol : float
        Improvement threshold used with `patience`.
    n_jobs : int
        Number of worker processes for fitness evaluation. 1 = serial.
    """

    def __init__(
        self,
        fitness_fn: Callable,
        bounds: List[Tuple[float, float]],
        *,
        n_bits: int = 16,
        pop_size: int = 200,
        n_generations: int = 100,
        crossover_rate: float = 0.9,
        maximize: bool = True,
        elitism: int = 2,
        tournament_size: int = 10,
        patience: Optional[int] = None,
        tol: float = 1e-6,
        n_jobs: int = 1,
    ):
        self.n_bits = n_bits
        # mutation_rate: expected 1 bit flip per individual per generation
        self._mutation_rate = 1.0 / (n_bits * len(bounds))

        super().__init__(
            fitness_fn,
            bounds,
            pop_size=pop_size,
            n_generations=n_generations,
            crossover_rate=crossover_rate,
            maximize=maximize,
            elitism=elitism,
            tournament_size=tournament_size,
            patience=patience,
            tol=tol,
            n_jobs=n_jobs,
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _initialize_population(self) -> list:
        genome_length = self.n_bits * len(self.bounds)
        return [
            np.random.randint(0, 2, genome_length) for _ in range(self.pop_size)
        ]

    def _decode(self, individual) -> List[float]:
        """Map a bitstring individual to float values within bounds."""
        decoded = []
        largest = 2 ** self.n_bits
        for i, (lo, hi) in enumerate(self.bounds):
            start = i * self.n_bits
            end = start + self.n_bits
            substring = individual[start:end]
            chars = "".join(str(b) for b in substring)
            integer = int(chars, 2)
            value = lo + (integer / largest) * (hi - lo)
            decoded.append(value)
        return decoded

    def _crossover(self, parent1, parent2) -> Tuple:
        """Uniform crossover — each bit independently chosen from either parent."""
        child1, child2 = parent1.copy(), parent2.copy()
        if np.random.rand() < self.crossover_rate:
            mask = np.random.rand(len(child1)) < 0.5
            child1[mask] = parent2[mask]
            child2[mask] = parent1[mask]
        return child1, child2

    def _mutate(self, individual):
        """Bit-flip mutation applied to every bit in the genome (BUG FIX)."""
        # Fix: iterate over len(individual) = n_bits * n_dims, not just n_bits
        flip_mask = np.random.rand(len(individual)) < self._mutation_rate
        individual[flip_mask] = 1 - individual[flip_mask]
