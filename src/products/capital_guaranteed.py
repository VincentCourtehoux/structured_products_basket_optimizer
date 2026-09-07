"""
Capital Guaranteed Lookback Structured Product Engine.
"""

from dataclasses import dataclass, field
import numpy as np

from src.products.base import BaseProductResult
from src.products.schedule_utils import build_observation_schedule, map_dates_to_steps, resolve_index


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
            self.observation_dates = build_observation_schedule(
                self.maturity, self.obs_freq, default_freq="semi-annual"
            )


class CapitalGuaranteedResult(BaseProductResult):
    def __init__(
        self,
        params: CapitalGuaranteedParams,
        cashflow_matrix: np.ndarray,
        obs_dates: list[float],
        lookback_perfs: np.ndarray,
        effective_perfs: np.ndarray,
        floor_binding_mask: np.ndarray,
    ):
        super().__init__(params, cashflow_matrix, obs_dates)

        self.lookback_perfs = lookback_perfs
        self.effective_perfs = effective_perfs
        self.floor_binding_mask = floor_binding_mask

        # Product-specific metrics
        self.prob_floor_binding = float(np.mean(self.floor_binding_mask))
        self.avg_lookback_perf = float(np.mean(self.lookback_perfs))
        self.avg_effective_perf = float(np.mean(self.effective_perfs))

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

        asset_idx = resolve_index(self.params.index_name, index_names)

        obs_dates = self.params.observation_dates
        n_obs = len(obs_dates)
        obs_step_indices = map_dates_to_steps(obs_dates, dt, n_path_steps)

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