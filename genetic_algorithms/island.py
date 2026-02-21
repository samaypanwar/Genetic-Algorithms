from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Type

import numpy as np

from .real_valued import RealValuedGA


class IslandModelGA:
    """Multi-population Genetic Algorithm using an island topology.

    Maintains `n_islands` independent sub-populations (islands). Each island
    runs its own GA for `migration_interval` generations, after which the top
    `n_migrants` individuals from each island migrate to the *next* island in
    a ring:

        island 0 → island 1 → island 2 → ... → island (n-1) → island 0

    Receiving islands replace their worst `n_migrants` individuals with the
    migrants from the previous island, injecting diversity and allowing good
    solutions to spread across the topology.

    Parameters
    ----------
    fitness_fn : callable
        Fitness function shared by all islands.
    bounds : list of (float, float)
        Search-space bounds.
    n_islands : int
        Number of independent sub-populations.
    pop_size_per_island : int
        Number of individuals on each island.
    n_generations : int
        Total generations *per island* across all migration rounds combined.
        The actual number of rounds = n_generations // migration_interval.
    migration_interval : int
        Generations each island runs independently before migration occurs.
    n_migrants : int
        Number of individuals exchanged between adjacent islands per
        migration event.
    island_cls : type
        GA class to use for each island (must be RealValuedGA or a subclass).
    maximize : bool
        Optimisation direction shared by all islands.
    island_kwargs : dict
        Extra keyword arguments forwarded to each island's constructor
        (e.g. sbx_eta, sigma, tournament_size, …).
    """

    def __init__(
        self,
        fitness_fn: Callable,
        bounds: List[Tuple[float, float]],
        *,
        n_islands: int = 4,
        pop_size_per_island: int = 50,
        n_generations: int = 100,
        migration_interval: int = 10,
        n_migrants: int = 2,
        island_cls: Type[RealValuedGA] = RealValuedGA,
        maximize: bool = True,
        **island_kwargs,
    ):
        self.fitness_fn = fitness_fn
        self.bounds = bounds
        self.n_islands = n_islands
        self.pop_size_per_island = pop_size_per_island
        self.n_generations = n_generations
        self.migration_interval = migration_interval
        self.n_migrants = n_migrants
        self.maximize = maximize

        # Build islands — each with its own independent RNG state
        self.islands: List[RealValuedGA] = [
            island_cls(
                fitness_fn,
                bounds,
                pop_size=pop_size_per_island,
                n_generations=migration_interval,
                maximize=maximize,
                **island_kwargs,
            )
            for _ in range(n_islands)
        ]

        # Global tracking
        self.history: List[Tuple[int, float]] = []
        self._best_solution: Optional[List[float]] = None
        self._best_score: Optional[float] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_better(self, a: float, b: float) -> bool:
        return a > b if self.maximize else a < b

    def _island_scores(self, island: RealValuedGA) -> List[float]:
        decoded = [island._decode(ind) for ind in island.population]
        return island._evaluate_population(decoded)

    def _get_top_indices(self, scores: List[float], n: int) -> List[int]:
        return sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=self.maximize,
        )[:n]

    def _get_bottom_indices(self, scores: List[float], n: int) -> List[int]:
        return sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=not self.maximize,
        )[:n]

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def _migrate(self):
        """Ring-topology migration: island i → island (i+1) % n_islands."""
        # Collect emigrants from each island first (before any replacements)
        migrants_per_island = []
        for island in self.islands:
            scores = self._island_scores(island)
            top_idx = self._get_top_indices(scores, self.n_migrants)
            emigrants = [island.population[i].copy() for i in top_idx]
            migrants_per_island.append(emigrants)

        # Inject into the *next* island in the ring
        for i, island in enumerate(self.islands):
            incoming = migrants_per_island[(i - 1) % self.n_islands]
            scores = self._island_scores(island)
            bottom_idx = self._get_bottom_indices(scores, self.n_migrants)
            for slot, migrant in zip(bottom_idx, incoming):
                island.population[slot] = migrant

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def evolve(self) -> List[float]:
        """Run the island model and return the globally best solution found."""
        n_rounds = max(1, self.n_generations // self.migration_interval)
        round_generation = 0

        for round_idx in range(n_rounds):
            print(
                f"\n=== Migration round {round_idx + 1}/{n_rounds} "
                f"(gens {round_generation + 1}–"
                f"{round_generation + self.migration_interval}) ==="
            )

            # Evolve each island independently
            for island_idx, island in enumerate(self.islands):
                print(f"\n  [Island {island_idx + 1}]")
                island.n_generations = self.migration_interval
                island.evolve()

            round_generation += self.migration_interval

            # Update global best
            for island in self.islands:
                if island.best_score is not None:
                    if self._best_score is None or self._is_better(
                        island.best_score, self._best_score
                    ):
                        self._best_score = island.best_score
                        self._best_solution = island.best_solution

            self.history.append((round_generation, self._best_score))
            print(
                f"\n  Global best after round {round_idx + 1}: "
                f"{self._best_score:.6f}"
            )

            # Migrate between islands (skip after the last round)
            if round_idx < n_rounds - 1:
                self._migrate()
                print(
                    f"  Migrated {self.n_migrants} individual(s) "
                    f"between each pair of islands."
                )

        return self._best_solution

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_score(self) -> Optional[float]:
        return self._best_score

    @property
    def best_solution(self) -> Optional[List[float]]:
        return self._best_solution
