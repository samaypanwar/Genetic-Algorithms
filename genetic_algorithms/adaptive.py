from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from .real_valued import RealValuedGA


class AdaptiveGA(RealValuedGA):
    """Real-valued GA with self-adaptive mutation standard deviation (σ).

    After every generation the population diversity is measured as the mean
    standard deviation across all dimensions:

        diversity = mean(std(population, axis=0))

    The mutation σ is then adjusted:
      - If diversity < low_threshold  → increase σ by `scale_up`
        (population is converging / stagnating — explore more)
      - If diversity > high_threshold → decrease σ by `scale_down`
        (population is spread out — exploit more)

    The sigma history is stored in `sigma_history` so you can observe the
    adaptation over time (e.g. with visualization.plot_sigma_adaptation).

    Parameters
    ----------
    fitness_fn : callable
    bounds : list of (float, float)
    low_threshold : float or None
        Diversity below which σ is increased. Defaults to 1 % of the mean
        dimension range.
    high_threshold : float or None
        Diversity above which σ is decreased. Defaults to 50 % of the mean
        dimension range.
    scale_up : float
        Multiplicative factor applied to σ when diversity is too low (>1).
    scale_down : float
        Multiplicative factor applied to σ when diversity is too high (<1).
    sigma_min : float
        Hard lower bound on σ to avoid premature convergence.
    sigma_max : float or None
        Hard upper bound on σ. Defaults to the mean dimension range.
    All other kwargs are forwarded to RealValuedGA.
    """

    def __init__(
        self,
        fitness_fn: Callable,
        bounds: List[Tuple[float, float]],
        *,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        scale_up: float = 1.2,
        scale_down: float = 0.9,
        sigma_min: float = 1e-6,
        sigma_max: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(fitness_fn, bounds, **kwargs)

        mean_range = float(
            np.mean([hi - lo for lo, hi in bounds])
        )
        self._low_threshold = low_threshold if low_threshold is not None else 0.01 * mean_range
        self._high_threshold = high_threshold if high_threshold is not None else 0.5 * mean_range
        self._scale_up = scale_up
        self._scale_down = scale_down
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max if sigma_max is not None else mean_range

        # sigma_history stores (generation, mean_sigma) tuples
        self.sigma_history: List[Tuple[int, float]] = []

    def _on_generation_end(self, generation: int, scores: List[float]):
        """Measure diversity and adapt σ accordingly."""
        if len(self.population) == 0:
            return

        pop_array = np.array(self.population)
        diversity = float(np.mean(np.std(pop_array, axis=0)))

        if diversity < self._low_threshold:
            self.sigma = np.clip(
                self.sigma * self._scale_up, self._sigma_min, self._sigma_max
            )
        elif diversity > self._high_threshold:
            self.sigma = np.clip(
                self.sigma * self._scale_down, self._sigma_min, self._sigma_max
            )

        mean_sigma = float(np.mean(self.sigma))
        self.sigma_history.append((generation + 1, mean_sigma))
        print(
            f"  Diversity: {diversity:.4f} | σ: {mean_sigma:.6f}"
        )
