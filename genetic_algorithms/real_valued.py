from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from .base import GeneticAlgorithmBase


class RealValuedGA(GeneticAlgorithmBase):
    """Genetic Algorithm with real-valued (float) encoding.

    Individuals are represented directly as NumPy float arrays — no binary
    encode/decode step is required. This is significantly more efficient for
    continuous optimisation problems.

    Crossover: Simulated Binary Crossover (SBX) with distribution index η.
      SBX mimics single-point crossover on binary strings but works in
      continuous space, producing children near the parents while
      occasionally exploring further away.

    Mutation: Gaussian perturbation — each gene is shifted by N(0, σ) and
      clipped back to its bounds.

    Parameters
    ----------
    fitness_fn : callable
        Function accepting a list of floats and returning a scalar score.
    bounds : list of (float, float)
        Search-space bounds per dimension.
    sigma : float or None
        Standard deviation for Gaussian mutation. Defaults to 10 % of each
        dimension's range (computed per-dimension at init time).
    sbx_eta : float
        Distribution index for SBX crossover. Higher values keep children
        closer to parents (more exploitative). Typical range: 2–20.
    pop_size : int
        Population size.
    n_generations : int
        Maximum number of generations.
    crossover_rate : float
        Probability crossover is applied to a parent pair.
    maximize : bool
        True to maximise fitness, False to minimise.
    elitism : int
        Number of elites carried forward unchanged.
    tournament_size : int
        Tournament selection pool size.
    patience : int or None
        Early stopping patience (generations without improvement).
    tol : float
        Early stopping improvement threshold.
    n_jobs : int
        Worker processes for fitness evaluation. 1 = serial.
    """

    def __init__(
        self,
        fitness_fn: Callable,
        bounds: List[Tuple[float, float]],
        *,
        sigma: Optional[float] = None,
        sbx_eta: float = 2.0,
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
        self.sbx_eta = sbx_eta
        # Per-dimension sigma: 10 % of each dimension's range
        ranges = np.array([hi - lo for lo, hi in bounds])
        self._sigma = sigma if sigma is not None else 0.1 * ranges
        self._bounds_lo = np.array([lo for lo, _ in bounds])
        self._bounds_hi = np.array([hi for _, hi in bounds])

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
        return [
            np.array(
                [
                    np.random.uniform(lo, hi)
                    for lo, hi in self.bounds
                ]
            )
            for _ in range(self.pop_size)
        ]

    def _decode(self, individual) -> List[float]:
        return individual.tolist()

    def _crossover(self, parent1, parent2) -> Tuple:
        """Simulated Binary Crossover (SBX)."""
        child1, child2 = parent1.copy(), parent2.copy()
        if np.random.rand() > self.crossover_rate:
            return child1, child2

        for i in range(self.n_dims):
            if np.random.rand() < 0.5:
                continue  # skip this gene with 50 % probability

            x1, x2 = parent1[i], parent2[i]
            lo, hi = self._bounds_lo[i], self._bounds_hi[i]

            if abs(x2 - x1) < 1e-12:
                continue  # parents identical on this gene

            # SBX spread factor β
            u = np.random.rand()
            eta = self.sbx_eta
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta + 1))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1))

            child1[i] = np.clip(0.5 * ((1 + beta) * x1 + (1 - beta) * x2), lo, hi)
            child2[i] = np.clip(0.5 * ((1 - beta) * x1 + (1 + beta) * x2), lo, hi)

        return child1, child2

    def _mutate(self, individual):
        """Gaussian perturbation, clipped to bounds."""
        noise = np.random.normal(0, self._sigma, size=self.n_dims)
        individual[:] = np.clip(
            individual + noise, self._bounds_lo, self._bounds_hi
        )

    # ------------------------------------------------------------------
    # Expose sigma as a property so AdaptiveGA can modify it
    # ------------------------------------------------------------------

    @property
    def sigma(self) -> np.ndarray:
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = np.asarray(value)
