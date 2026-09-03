"""
Capital Guaranteed Lookback Structured Product Engine.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class CapitalGuaranteedParams:
    index_name: str
    maturity: float = 2.0
    obs_freq: str = "semi-annual"  # "monthly" | "quarterly" | "semi-annual" | "annual"
    observation_dates: list[float] = field(default_factory=list)
    floor: float = 0.04  # e.g. 4%
    participation: float = 1.15  # e.g. 115%
    lookback_type: str = "max"  # "max" | "avg"
    notional: float = 1.0
    name: str = "Capital Garanti"

    def __post_init__(self):
        if not self.observation_dates:
            freq_map = {
                "monthly": 12,
                "quarterly": 4,
                "semi-annual": 2,
                "annual": 1,
            }
            n_per_year = freq_map.get(str(self.obs_freq).lower(), 2)
            dt_step = 1.0 / n_per_year
            n_obs = int(round(self.maturity * n_per_year))
            self.observation_dates = [round((i + 1) * dt_step, 6) for i in range(n_obs)]


class CapitalGuaranteedResult:
    def __init__(
        self,
        params: CapitalGuaranteedParams,
        cashflow_matrix: np.ndarray,
        obs_dates: list[float],
        lookback_perfs: np.ndarray,
        effective_perfs: np.ndarray,
        floor_binding_mask: np.ndarray,
    ):
        self.params = params
        self.cashflow_matrix = cashflow_matrix  # (n_sim, n_obs)
        self.obs_dates = obs_dates
        self.lookback_perfs = lookback_perfs
        self.effective_perfs = effective_perfs
        self.floor_binding_mask = floor_binding_mask
        self.notional = params.notional

        self.n_sim = cashflow_matrix.shape[0]
        self.n_obs = cashflow_matrix.shape[1]

        # Metrics
        self.pv = float(np.mean(np.sum(self.cashflow_matrix, axis=1)))
        self.price = self.pv
        self.price_pct = (self.price / self.notional) * 100.0 if self.notional > 0 else 0.0

        self.prob_floor_binding = float(np.mean(self.floor_binding_mask))
        self.avg_lookback_perf = float(np.mean(self.lookback_perfs))
        self.avg_effective_perf = float(np.mean(self.effective_perfs))

    def cashflow_dataframe(self) -> pd.DataFrame:
        cols = [f"t={t:.2f}y" for t in self.obs_dates]
        return pd.DataFrame(self.cashflow_matrix, columns=cols)

    def total_undiscounted_payoff(self) -> np.ndarray:
        return np.sum(self.cashflow_matrix, axis=1)

    def summary(self) -> dict:
        return {
            "name": self.params.name,
            "type": "Capital Guaranteed",
            "price": self.price,
            "price_pct": self.price_pct,
            "prob_floor_binding": self.prob_floor_binding,
            "avg_lookback_perf": self.avg_lookback_perf,
            "avg_effective_perf": self.avg_effective_perf,
            "floor": self.params.floor,
            "participation": self.params.participation,
            "lookback_type": self.params.lookback_type,
            "notional": self.notional,
            "maturity": self.params.maturity,
        }

    def print_summary(self):
        s = self.summary()
        print(f"=== {s['name']} Summary ===")
        print(f"Prix (% notional):       {s['price_pct']:.2f}%")
        print(f"Perf lookback moyenne:   {s['avg_lookback_perf']*100:.2f}%")
        print(f"Perf effective moyenne:  {s['avg_effective_perf']*100:.2f}%")
        print(f"Probabilité plancher:    {s['prob_floor_binding']*100:.2f}%")


class CapitalGuaranteedPricer:
    def __init__(self, params: CapitalGuaranteedParams):
        self.params = params

    def compute_payoffs(self, paths: np.ndarray, dt: float, index_names: list[str]) -> CapitalGuaranteedResult:
        n_sim, n_path_steps, n_assets = paths.shape

        if self.params.index_name not in index_names:
            raise ValueError(f"Underlying index '{self.params.index_name}' not found in simulation paths.")
        asset_idx = index_names.index(self.params.index_name)

        obs_dates = self.params.observation_dates
        n_obs = len(obs_dates)
        obs_step_indices = [min(int(round(t / dt)), n_path_steps - 1) for t in obs_dates]

        # Extract underlying prices at obs dates
        s0 = paths[:, 0, asset_idx]
        obs_prices = paths[:, obs_step_indices, asset_idx]  # (n_sim, n_obs)
        obs_perfs = (obs_prices / s0[:, None]) - 1.0

        if self.params.lookback_type.lower() == "max":
            lookback_perfs = np.max(obs_perfs, axis=1)
        else:
            lookback_perfs = np.mean(obs_perfs, axis=1)

        participated_perfs = self.params.participation * lookback_perfs
        effective_perfs = np.maximum(self.params.floor, participated_perfs)
        floor_binding = participated_perfs <= self.params.floor

        # One cashflow at maturity (last observation date)
        cashflows = np.zeros((n_sim, n_obs), dtype=np.float64)
        cashflows[:, -1] = self.params.notional * (1.0 + effective_perfs)

        return CapitalGuaranteedResult(
            params=self.params,
            cashflow_matrix=cashflows,
            obs_dates=obs_dates,
            lookback_perfs=lookback_perfs,
            effective_perfs=effective_perfs,
            floor_binding_mask=floor_binding,
        )
