"""
Autocall & Phoenix Structured Product Engine.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class AutocallParams:
    index_names: list[str]
    basket_type: str = "worst-of"  # "single" | "worst-of" | "best-of" | "average"
    maturity: float = 8.0
    obs_freq: str = "quarterly"  # "monthly" | "quarterly" | "semi-annual" | "annual"
    observation_dates: list[float] = field(default_factory=list)
    autocall_barrier_initial: float = 1.00
    barrier_step_down: float = 0.005
    barrier_step_freq: int | str = "quarterly"
    barrier_step_start_year: float = 1.25
    first_call_year: float = 1.0
    coupon_rate: float = 0.025  # per observation period
    coupon_barrier: float | None = None
    memory_coupon: bool = True
    capital_barrier: float = 0.60
    capital_barrier_type: str = "european"  # "european" | "american"
    notional: float = 1.0
    name: str = "Autocall"

    def __post_init__(self):
        if not self.observation_dates:
            freq_map = {
                "monthly": 12,
                "quarterly": 4,
                "semi-annual": 2,
                "annual": 1,
            }
            n_per_year = freq_map.get(str(self.obs_freq).lower(), 4)
            dt_step = 1.0 / n_per_year
            n_obs = int(round(self.maturity * n_per_year))
            self.observation_dates = [round((i + 1) * dt_step, 6) for i in range(n_obs)]


class AutocallResult:
    def __init__(
        self,
        params: AutocallParams,
        cashflow_matrix: np.ndarray,
        obs_dates: list[float],
        called_mask: np.ndarray,
        call_times: np.ndarray,
        loss_mask: np.ndarray,
        loss_amounts: np.ndarray,
        total_coupons: np.ndarray,
    ):
        self.params = params
        self.cashflow_matrix = cashflow_matrix  # (n_sim, n_obs)
        self.obs_dates = obs_dates
        self.called_mask = called_mask
        self.call_times = call_times
        self.loss_mask = loss_mask
        self.loss_amounts = loss_amounts
        self.total_coupons = total_coupons
        self.notional = params.notional

        self.n_sim = cashflow_matrix.shape[0]
        self.n_obs = cashflow_matrix.shape[1]

        # Financial metrics
        self.pv = float(np.mean(np.sum(self.cashflow_matrix, axis=1)))
        self.price = self.pv
        self.price_pct = (self.price / self.notional) * 100.0 if self.notional > 0 else 0.0

        self.prob_autocall = float(np.mean(self.called_mask))
        self.prob_capital_loss = float(np.mean(self.loss_mask))
        self.expected_capital_loss = float(np.mean(self.loss_amounts))
        called_times_subset = self.call_times[self.called_mask]
        self.avg_autocall_time = float(np.mean(called_times_subset)) if len(called_times_subset) > 0 else float(params.maturity)
        self.avg_coupons = float(np.mean(self.total_coupons))

    def cashflow_dataframe(self) -> pd.DataFrame:
        cols = [f"t={t:.2f}y" for t in self.obs_dates]
        return pd.DataFrame(self.cashflow_matrix, columns=cols)

    def total_undiscounted_payoff(self) -> np.ndarray:
        return np.sum(self.cashflow_matrix, axis=1)

    def autocall_time_distribution(self) -> dict[float, float]:
        dist = {}
        for t in self.obs_dates:
            dist[t] = float(np.mean(self.call_times == t))
        dist["uncalled"] = float(np.mean(~self.called_mask))
        return dist

    def summary(self) -> dict:
        return {
            "name": self.params.name,
            "type": "Autocall",
            "price": self.price,
            "price_pct": self.price_pct,
            "prob_autocall": self.prob_autocall,
            "prob_capital_loss": self.prob_capital_loss,
            "expected_capital_loss": self.expected_capital_loss,
            "avg_autocall_time": self.avg_autocall_time,
            "avg_coupons": self.avg_coupons,
            "notional": self.notional,
            "maturity": self.params.maturity,
            "obs_freq": self.params.obs_freq,
        }

    def print_summary(self):
        s = self.summary()
        print(f"=== {s['name']} Summary ===")
        print(f"Price pct:       {s['price_pct']:.2f}%")
        print(f"Autocall probability:  {s['prob_autocall']*100:.2f}%")
        print(f"Average autocall time:  {s['avg_autocall_time']:.2f} ans")
        print(f"Capital loss probability:  {s['prob_capital_loss']*100:.2f}%")
        print(f"Expected capital loss:   {s['expected_capital_loss']*100:.2f}%")
        print(f"Average coupons:   {s['avg_coupons']:.4f}")


class AutocallPricer:
    def __init__(self, params: AutocallParams):
        self.params = params

    def _get_barrier_schedule(self) -> np.ndarray:
        dates = np.array(self.params.observation_dates)
        barriers = np.full(len(dates), np.inf)

        # Map step freq
        freq_map = {"monthly": 1 / 12, "quarterly": 0.25, "semi-annual": 0.5, "annual": 1.0}
        if isinstance(self.params.barrier_step_freq, str):
            step_dt = freq_map.get(self.params.barrier_step_freq.lower(), 0.25)
        elif isinstance(self.params.barrier_step_freq, (int, float)):
            step_dt = float(self.params.barrier_step_freq)
        else:
            step_dt = 0.25

        for i, t in enumerate(dates):
            if t < self.params.first_call_year - 1e-5:
                barriers[i] = np.inf
            else:
                if self.params.barrier_step_down <= 0:
                    barriers[i] = self.params.autocall_barrier_initial
                elif t >= self.params.barrier_step_start_year - 1e-5:
                    n_steps = int(np.floor(round((t - self.params.barrier_step_start_year) / step_dt, 4))) + 1
                    barriers[i] = max(0.0, self.params.autocall_barrier_initial - n_steps * self.params.barrier_step_down)
                else:
                    barriers[i] = self.params.autocall_barrier_initial
        return barriers

    def compute_payoffs(self, paths: np.ndarray, dt: float, index_names: list[str]) -> AutocallResult:
        """
        paths: shape (n_sim, n_steps + 1, n_assets)
        dt: step size in years (e.g. 1/252)
        index_names: list of index names matching paths asset dimension
        """
        n_sim, n_path_steps, n_assets = paths.shape

        # Match indices
        matched_idx = []
        for name in self.params.index_names:
            if name in index_names:
                matched_idx.append(index_names.index(name))
            else:
                raise ValueError(f"Underlying index '{name}' not found in simulation paths.")

        if self.params.basket_type == "single" and len(matched_idx) != 1:
            raise ValueError(f"Single basket type requires exactly 1 index, got {len(matched_idx)}")

        # Selected paths normalized to S(0)
        # shape: (n_sim, n_path_steps, len(matched_idx))
        s0 = paths[:, 0:1, matched_idx]
        rel_paths = paths[:, :, matched_idx] / s0

        # Basket performance across all time steps
        if self.params.basket_type == "single" or len(matched_idx) == 1:
            basket_perf_all = rel_paths[:, :, 0]
        elif self.params.basket_type == "worst-of":
            basket_perf_all = np.min(rel_paths, axis=2)
        elif self.params.basket_type == "best-of":
            basket_perf_all = np.max(rel_paths, axis=2)
        elif self.params.basket_type == "average":
            basket_perf_all = np.mean(rel_paths, axis=2)
        else:
            basket_perf_all = np.min(rel_paths, axis=2)

        # Observation dates & indices in path
        obs_dates = self.params.observation_dates
        n_obs = len(obs_dates)
        obs_step_indices = [min(int(round(t / dt)), n_path_steps - 1) for t in obs_dates]

        # Barrier schedule
        autocall_barriers = self._get_barrier_schedule()
        coupon_barrier = self.params.coupon_barrier if self.params.coupon_barrier is not None else self.params.autocall_barrier_initial
        is_phoenix = (self.params.coupon_barrier is not None and self.params.coupon_barrier < self.params.autocall_barrier_initial)

        # American barrier check
        american_breached = np.zeros(n_sim, dtype=bool)
        if self.params.capital_barrier_type.lower() == "american":
            min_perf_over_time = np.min(basket_perf_all, axis=1)
            american_breached = (min_perf_over_time < self.params.capital_barrier)

        cashflows = np.zeros((n_sim, n_obs), dtype=np.float64)
        active_mask = np.ones(n_sim, dtype=bool)
        called_mask = np.zeros(n_sim, dtype=bool)
        call_times = np.full(n_sim, np.nan)
        accumulated_coupons = np.zeros(n_sim, dtype=np.float64)
        total_coupons_paid = np.zeros(n_sim, dtype=np.float64)

        for obs_i, (t_obs, step_idx) in enumerate(zip(obs_dates, obs_step_indices)):
            if not np.any(active_mask):
                break

            perf_t = basket_perf_all[:, step_idx]
            ac_bar = autocall_barriers[obs_i]

            # 1. Coupon Evaluation
            if is_phoenix:
                # Phoenix logic
                coupon_due = self.params.coupon_rate * self.params.notional
                can_pay_coupon = (perf_t >= coupon_barrier) & active_mask

                # Pay current + memory
                pay_amt = np.where(
                    can_pay_coupon,
                    coupon_due + accumulated_coupons,
                    0.0
                )
                cashflows[:, obs_i] += pay_amt
                total_coupons_paid += pay_amt

                # Update memory
                accumulated_coupons = np.where(
                    can_pay_coupon,
                    0.0,
                    np.where(
                        active_mask & self.params.memory_coupon,
                        accumulated_coupons + coupon_due,
                        0.0
                    )
                )
            else:
                # Standard autocall coupon accumulation handled at autocall or uncalled
                pass

            # 2. Autocall Evaluation
            is_autocalled = (perf_t >= ac_bar) & active_mask

            if np.any(is_autocalled):
                if not is_phoenix:
                    # Standard autocall pays notional + coupon (+ memory if enabled)
                    coupon_due = self.params.coupon_rate * self.params.notional
                    if self.params.memory_coupon:
                        coupon_to_pay = coupon_due + accumulated_coupons
                    else:
                        coupon_to_pay = coupon_due

                    pay_coupon = np.where(is_autocalled, coupon_to_pay, 0.0)
                    total_coupons_paid += pay_coupon
                    cashflows[:, obs_i] += np.where(is_autocalled, self.params.notional + pay_coupon, 0.0)
                else:
                    # Phoenix: coupon already added, just add notional
                    cashflows[:, obs_i] += np.where(is_autocalled, self.params.notional, 0.0)

                called_mask |= is_autocalled
                call_times = np.where(is_autocalled & np.isnan(call_times), t_obs, call_times)
                active_mask &= ~is_autocalled

            # If not called and standard autocall with memory
            if not is_phoenix:
                not_called_active = active_mask & (~is_autocalled)
                if self.params.memory_coupon:
                    accumulated_coupons = np.where(
                        not_called_active,
                        accumulated_coupons + (self.params.coupon_rate * self.params.notional),
                        accumulated_coupons
                    )

        # 3. Maturity Payoff for uncalled scenarios
        uncalled_mask = active_mask
        if np.any(uncalled_mask):
            last_obs_idx = n_obs - 1
            final_step = obs_step_indices[-1]
            final_perf = basket_perf_all[:, final_step]

            if self.params.capital_barrier_type.lower() == "american":
                capital_loss_condition = american_breached & (final_perf < 1.0)
            else:
                capital_loss_condition = (final_perf < self.params.capital_barrier)

            # Redemption amount
            redemption = np.where(
                capital_loss_condition,
                np.maximum(0.0, final_perf * self.params.notional),
                self.params.notional
            )

            cashflows[:, last_obs_idx] += np.where(uncalled_mask, redemption, 0.0)

        # Loss metrics
        final_payoff_total = np.sum(cashflows, axis=1)
        loss_mask = final_payoff_total < self.params.notional
        loss_amounts = np.maximum(0.0, self.params.notional - final_payoff_total)

        return AutocallResult(
            params=self.params,
            cashflow_matrix=cashflows,
            obs_dates=obs_dates,
            called_mask=called_mask,
            call_times=call_times,
            loss_mask=loss_mask,
            loss_amounts=loss_amounts,
            total_coupons=total_coupons_paid,
        )
