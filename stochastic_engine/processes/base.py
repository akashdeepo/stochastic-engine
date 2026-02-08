"""Abstract base class for stochastic processes."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


class StochasticProcess(ABC):
    """
    Abstract base class for all stochastic processes.

    All stochastic process implementations should inherit from this class
    and implement the required methods for simulation and sampling.

    Attributes
    ----------
    seed : int | None
        Random seed for reproducibility.

    Methods
    -------
    simulate(T, steps, n_paths)
        Generate sample paths over time horizon T.
    sample(t, n_samples)
        Sample the process at a specific time point.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the stochastic process.

        Parameters
        ----------
        seed : int | None, optional
            Random seed for reproducibility. Default is None.
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _get_rng(self) -> np.random.Generator:
        """Get the random number generator."""
        return self._rng

    def reset_rng(self, seed: int | None = None) -> None:
        """
        Reset the random number generator.

        Parameters
        ----------
        seed : int | None, optional
            New seed value. If None, uses the original seed.
        """
        self._rng = np.random.default_rng(seed if seed is not None else self.seed)

    @abstractmethod
    def simulate(
        self,
        T: float,
        steps: int,
        n_paths: int = 1,
        method: Literal["euler", "exact"] = "exact",
    ) -> np.ndarray:
        """
        Generate sample paths of the process.

        Parameters
        ----------
        T : float
            Time horizon (in years).
        steps : int
            Number of time steps.
        n_paths : int, optional
            Number of paths to simulate. Default is 1.
        method : {"euler", "exact"}, optional
            Simulation method. Default is "exact".

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, steps + 1) containing simulated paths.
            Each row is a path, columns are time points from 0 to T.
        """
        pass

    @abstractmethod
    def sample(self, t: float, n_samples: int = 1) -> np.ndarray:
        """
        Sample the process at a specific time point.

        Parameters
        ----------
        t : float
            Time point to sample at.
        n_samples : int, optional
            Number of samples to generate. Default is 1.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) containing samples at time t.
        """
        pass

    @abstractmethod
    def mean(self, t: float) -> float:
        """
        Expected value of the process at time t.

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        float
            Expected value E[X_t].
        """
        pass

    @abstractmethod
    def variance(self, t: float) -> float:
        """
        Variance of the process at time t.

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        float
            Variance Var[X_t].
        """
        pass
