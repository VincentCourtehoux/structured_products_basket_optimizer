"""
Shared base class for structured product pricing results.

Factors out the pv/price/price_pct computation and the cashflow
dataframe / total payoff helpers.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd


class BaseProductResult(ABC):
    def __init__(self, params: Any, cashflow_matrix: np.ndarray, obs_dates: list[float]):
        self.params = params
        self.cashflow_matrix = cashflow_matrix  # (n_sim, n_obs)
        self.obs_dates = obs_dates
        self.notional = params.notional

        self.n_sim = cashflow_matrix.shape[0]
        self.n_obs = cashflow_matrix.shape[1]

        # Financial metrics common to every product
        self.pv = float(np.mean(np.sum(self.cashflow_matrix, axis=1)))
        self.price = self.pv
        self.price_pct = (self.price / self.notional) * 100.0 if self.notional > 0 else 0.0

    def cashflow_dataframe(self) -> pd.DataFrame:
        cols = [f"t={t:.2f}y" for t in self.obs_dates]
        return pd.DataFrame(self.cashflow_matrix, columns=cols)

    def total_undiscounted_payoff(self) -> np.ndarray:
        return np.sum(self.cashflow_matrix, axis=1)

    @abstractmethod
    def summary(self) -> dict:
        """Each product returns its own dict of headline + product-specific metrics."""
        ...

    @abstractmethod
    def print_summary(self) -> None:
        """Each product keeps its own bespoke console formatting."""
        ...