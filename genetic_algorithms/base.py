from __future__ import annotations

import concurrent.futures
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np


class GeneticAlgorithmBase(ABC):
    """Abstract base class for all Genetic Algorithm variants.

    Provides shared infrastructure:
      - Tournament selection
      - Elitism (top N individuals survive each generation)
      - Convergence history tracking
      - Early stopping
      - Vectorised / multi-process fitness evaluation
    """

    def __init__(
        self,
        fitness_fn: Callable,
        bounds: List[Tuple[float, float]],
        *,
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
        self.fitness_fn = fitness_fn
        self.bounds = bounds
        self.n_dims = len(bounds)
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.maximize = maximize
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.patience = patience
        self.tol = tol
        self.n_jobs = n_jobs

        # State populated during evolve()
        self.history: List[Tuple[int, float]] = []
        self._best_individual = None
        self._best_score: Optional[float] = None
        self.population = self._initialize_population()

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _initialize_population(self) -> list:
        """Return an initial population of pop_size individuals."""

    @abstractmethod
    def _decode(self, individual) -> List[float]:
        """Convert an encoded individual to a list of float values."""

    @abstractmethod
    def _crossover(self, parent1, parent2) -> Tuple:
        """Produce two children from two parents."""

    @abstractmethod
    def _mutate(self, individual):
        """Apply in-place mutation to an individual."""

    # ------------------------------------------------------------------
    # Shared logic
    # ------------------------------------------------------------------

    def _is_better(self, score_a: float, score_b: float) -> bool:
        return score_a > score_b if self.maximize else score_a < score_b

    def _evaluate_population(self, decoded: List[List[float]]) -> List[float]:
        """Evaluate fitness for every individual.

        Uses ProcessPoolExecutor when n_jobs > 1, otherwise falls back to
        a plain list comprehension (which also handles non-vectorisable fns).
        """
        if self.n_jobs > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.n_jobs
            ) as executor:
                return list(executor.map(self.fitness_fn, decoded))
        return [self.fitness_fn(ind) for ind in decoded]

    def _select_parent(self, scores: List[float]):
        """Tournament selection — returns a *copy* of the winning individual."""
        selected_idx = np.random.randint(0, self.pop_size)
        candidates = np.random.randint(0, self.pop_size, self.tournament_size - 1)
        for idx in candidates:
            if self._is_better(scores[idx], scores[selected_idx]):
                selected_idx = idx
        # Always return a copy so crossover/mutation cannot corrupt the parent
        ind = self.population[selected_idx]
        return ind.copy() if hasattr(ind, "copy") else list(ind)

    def _get_elite_indices(self, scores: List[float]) -> List[int]:
        """Return indices of the top `elitism` individuals."""
        sorted_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=self.maximize,
        )
        return sorted_indices[: self.elitism]

    def _converged(self) -> bool:
        """True if recent improvement is below tol for `patience` generations."""
        if self.patience is None or len(self.history) < self.patience:
            return False
        recent = [s for _, s in self.history[-self.patience :]]
        return abs(recent[-1] - recent[0]) < self.tol

    # ------------------------------------------------------------------
    # Main evolution loop
    # ------------------------------------------------------------------

    def evolve(self):
        """Run the genetic algorithm and return the best decoded solution."""
        # Evaluate initial population
        decoded = [self._decode(ind) for ind in self.population]
        scores = self._evaluate_population(decoded)

        self._best_individual = self.population[0].copy() if hasattr(self.population[0], "copy") else list(self.population[0])
        self._best_score = scores[0]
        for i, score in enumerate(scores):
            if self._is_better(score, self._best_score):
                self._best_score = score
                self._best_individual = (
                    self.population[i].copy()
                    if hasattr(self.population[i], "copy")
                    else list(self.population[i])
                )

        for generation in range(self.n_generations):
            decoded = [self._decode(ind) for ind in self.population]
            scores = self._evaluate_population(decoded)

            # Track global best
            for i, score in enumerate(scores):
                if self._is_better(score, self._best_score):
                    self._best_score = score
                    self._best_individual = (
                        self.population[i].copy()
                        if hasattr(self.population[i], "copy")
                        else list(self.population[i])
                    )

            self.history.append((generation + 1, self._best_score))
            print(
                f"Generation {generation + 1:>4d} | Best score: {self._best_score:.6f}"
            )

            # Elitism — carry forward the best individuals unchanged
            elite_indices = self._get_elite_indices(scores)
            next_population = [
                (
                    self.population[i].copy()
                    if hasattr(self.population[i], "copy")
                    else list(self.population[i])
                )
                for i in elite_indices
            ]

            # Fill remaining slots with crossover + mutation offspring
            n_parents = self.pop_size - len(next_population)
            parents = [self._select_parent(scores) for _ in range(n_parents)]

            for i in range(0, n_parents - 1, 2):
                child1, child2 = self._crossover(parents[i], parents[i + 1])
                self._mutate(child1)
                self._mutate(child2)
                next_population.extend([child1, child2])

            # Handle odd remainder
            if len(next_population) < self.pop_size:
                extra = self._select_parent(scores)
                self._mutate(extra)
                next_population.append(extra)

            self.population = next_population[: self.pop_size]

            self._on_generation_end(generation, scores)

            if self._converged():
                print(f"  Early stopping at generation {generation + 1}.")
                break

        return self._decode(self._best_individual)

    def _on_generation_end(self, generation: int, scores: List[float]):
        """Hook for subclasses to run extra logic at end of each generation."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_score(self) -> Optional[float]:
        return self._best_score

    @property
    def best_solution(self) -> Optional[List[float]]:
        if self._best_individual is None:
            return None
        return self._decode(self._best_individual)
