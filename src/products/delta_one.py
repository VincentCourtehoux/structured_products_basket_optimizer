"""
Delta One Structured Product Engine.
"""

from dataclasses import dataclass, field
import numpy as np

from src.products.base import BaseProductResult
from src.products.schedule_utils import map_dates_to_steps, resolve_index


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
            # Delta One has a single cashflow at maturity, not a regular
            # frequency-based schedule, so this stays product-specific
            # rather than going through build_observation_schedule.
            self.observation_dates = [round(float(self.maturity), 6)]


class DeltaOneResult(BaseProductResult):
    def __init__(
        self,
        params: DeltaOneParams,
        cashflow_matrix: np.ndarray,
        obs_dates: list[float],
        index_perfs: np.ndarray,
        loss_mask: np.ndarray,
    ):
        super().__init__(params, cashflow_matrix, obs_dates)

        self.index_perfs = index_perfs
        self.loss_mask = loss_mask

        # Product-specific metrics
        self.avg_index_perf = float(np.mean(self.index_perfs))
        self.prob_loss = float(np.mean(self.loss_mask))

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

        asset_idx = resolve_index(self.params.index_name, index_names)

        obs_dates = self.params.observation_dates
        maturity_step = map_dates_to_steps(obs_dates, dt, n_path_steps)[0]

        s0 = paths[:, 0, asset_idx]
        st = paths[:, maturity_step, asset_idx]

        index_perf = (st / s0) - 1.0
        payoffs = self.params.notional * (1.0 + self.params.fixed_coupon + index_perf)

        cashflows = np.zeros((n_sim, 1), dtype=np.float64)
        cashflows[:, 0] = payoffs

        loss_mask = payoffs < self.params.notional

        return DeltaOneResult(
            params=self.params,
            cashflow_matrix=cashflows,
            obs_dates=obs_dates,
            index_perfs=index_perf,
            loss_mask=loss_mask,
        )