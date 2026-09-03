"""
Delta One Structured Product Engine.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class DeltaOneParams:
    index_name: str
    maturity: float = 2.0
    observation_dates: list[float] = field(default_factory=list)
    fixed_coupon: float = 0.06  # e.g. 6%
    notional: float = 1.0
    name: str = "Delta One"

    def __post_init__(self):
        if not self.observation_dates:
            self.observation_dates = [round(float(self.maturity), 6)]


class DeltaOneResult:
    def __init__(
        self,
        params: DeltaOneParams,
        cashflow_matrix: np.ndarray,
        obs_dates: list[float],
        index_perfs: np.ndarray,
        loss_mask: np.ndarray,
    ):
        self.params = params
        self.cashflow_matrix = cashflow_matrix  # (n_sim, 1)
        self.obs_dates = obs_dates
        self.index_perfs = index_perfs
        self.loss_mask = loss_mask
        self.notional = params.notional

        self.n_sim = cashflow_matrix.shape[0]
        self.n_obs = cashflow_matrix.shape[1]

        # Metrics
        self.pv = float(np.mean(np.sum(self.cashflow_matrix, axis=1)))
        self.price = self.pv
        self.price_pct = (self.price / self.notional) * 100.0 if self.notional > 0 else 0.0

        self.avg_index_perf = float(np.mean(self.index_perfs))
        self.prob_loss = float(np.mean(self.loss_mask))

    def cashflow_dataframe(self) -> pd.DataFrame:
        cols = [f"t={t:.2f}y" for t in self.obs_dates]
        return pd.DataFrame(self.cashflow_matrix, columns=cols)

    def total_undiscounted_payoff(self) -> np.ndarray:
        return np.sum(self.cashflow_matrix, axis=1)

    def summary(self) -> dict:
        return {
            "name": self.params.name,
            "type": "Delta One",
            "price": self.price,
            "price_pct": self.price_pct,
            "avg_index_perf": self.avg_index_perf,
            "prob_loss": self.prob_loss,
            "fixed_coupon": self.params.fixed_coupon,
            "notional": self.notional,
            "maturity": self.params.maturity,
        }

    def print_summary(self):
        s = self.summary()
        print(f"=== {s['name']} Summary ===")
        print(f"Prix (% notional):       {s['price_pct']:.2f}%")
        print(f"Perf sous-jacent moy.:   {s['avg_index_perf']*100:.2f}%")
        print(f"Probabilité de perte:    {s['prob_loss']*100:.2f}%")


class DeltaOnePricer:
    def __init__(self, params: DeltaOneParams):
        self.params = params

    def compute_payoffs(self, paths: np.ndarray, dt: float, index_names: list[str]) -> DeltaOneResult:
        n_sim, n_path_steps, n_assets = paths.shape

        if self.params.index_name not in index_names:
            raise ValueError(f"Underlying index '{self.params.index_name}' not found in simulation paths.")
        asset_idx = index_names.index(self.params.index_name)

        maturity_step = min(int(round(self.params.maturity / dt)), n_path_steps - 1)
        s0 = paths[:, 0, asset_idx]
        st = paths[:, maturity_step, asset_idx]

        index_perf = (st / s0) - 1.0
        payoffs = self.params.notional * (1.0 + self.params.fixed_coupon + index_perf)

        cashflows = np.zeros((n_sim, 1), dtype=np.float64)
        cashflows[:, 0] = payoffs

        loss_mask = payoffs < self.params.notional
        obs_dates = [round(float(self.params.maturity), 6)]

        return DeltaOneResult(
            params=self.params,
            cashflow_matrix=cashflows,
            obs_dates=obs_dates,
            index_perfs=index_perf,
            loss_mask=loss_mask,
        )
