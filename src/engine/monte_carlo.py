import os
import numpy as np
import pandas as pd
from typing import Any, Optional

class MonteCarloSimulator:
    """
    Encapsulates the data ingestion, calibration, and execution of 
    a Monte Carlo simulation for correlated financial assets.
    """
    
    def __init__(self, filepath: str, trading_days: int = 252, max_history_years: float = 10.0):
        self.filepath = filepath
        self.trading_days = trading_days
        self.max_history_years = max_history_years
        
        self.tracks: dict[str, pd.Series] = {}
        self.index_names: list[str] = []
        self.stats_dict: dict[str, dict[str, float]] = {}
        self.corr_df: pd.DataFrame = pd.DataFrame()
        self.cholesky_matrix: np.ndarray = np.array([])
        self.paths: Optional[np.ndarray] = None
        
    def prepare_data(self) -> None:
        """Loads data, computes returns, volatility, drift, and correlations."""
        self._load_price_tracks()
        self.index_names = list(self.tracks.keys())
        
        if not self.index_names:
            raise ValueError("No valid indices found in the provided Excel file.")
            
        indiv_rets = self._compute_individual_log_returns()
        self._compute_drift_volatility(indiv_rets)
        self._compute_pairwise_correlation()
        self.cholesky_matrix = self._cholesky_decomposition(self.corr_df.to_numpy())

    def simulate(self, horizon: float = 8.0, n_sim: int = 1000, seed: int = 42) -> dict[str, Any]:
        """
        Runs the simulation using the calibrated data.
        Can be called multiple times with different parameters without reloading the Excel file.
        """
        if not self.index_names:
            self.prepare_data()

        s0 = np.array([self.tracks[name].iloc[-1] for name in self.index_names], dtype=np.float64)
        drift = np.array([self.stats_dict[name]["drift"] for name in self.index_names], dtype=np.float64)
        vol = np.array([self.stats_dict[name]["volatility"] for name in self.index_names], dtype=np.float64)
        
        n_steps = int(round(horizon * self.trading_days))
        dt = 1.0 / self.trading_days
        
        self.paths = self._simulate_correlated_paths(
            S0=s0, drift=drift, volatility=vol, L=self.cholesky_matrix,
            n_simulations=n_sim, n_steps=n_steps, dt=dt, seed=seed
        )

        return self._generate_summary_payload(horizon, n_sim, n_steps, dt)

    def _load_price_tracks(self) -> None:
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        xls = pd.ExcelFile(self.filepath)
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if df.empty or len(df.columns) < 2:
                continue
            
            date_col = df.columns[0]
            price_col = df.columns[1]
            
            df = df.dropna(subset=[date_col]).set_index(date_col)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            series = pd.to_numeric(df[price_col], errors="coerce").dropna()
            
            if len(series) < 10:
                continue

            series = series.ffill().dropna()

            if len(series) >= 10:
                self.tracks[str(sheet_name).strip()] = series

        if not self.tracks:
            raise ValueError(f"No valid time series could be extracted from {self.filepath}.")

    def _compute_individual_log_returns(self) -> dict[str, pd.Series]:
        returns_dict: dict[str, pd.Series] = {}
        target_range = int(self.max_history_years * self.trading_days)
        
        for name, series in self.tracks.items():
            n_obs = len(series)
            start_index = max(0, n_obs - target_range - 1)
            sub_series = series.iloc[start_index:]

            log_ret = np.log(sub_series / sub_series.shift(1)).dropna()
            
            if len(log_ret) > self.trading_days:
                returns_dict[name] = log_ret

        return returns_dict

    def _compute_drift_volatility(self, individual_returns: dict[str, pd.Series]) -> None:
        for name, rets in individual_returns.items():
            if len(rets) < self.trading_days:
                continue
                
            vol_ann = float(rets.std(ddof=1) * np.sqrt(self.trading_days))
            drift_ann = float(rets.mean() * self.trading_days + 0.5 * (vol_ann**2))
            
            self.stats_dict[name] = {
                "volatility": vol_ann,
                "drift": drift_ann,
                "mean_daily": float(rets.mean()),
                "std_daily": float(rets.std(ddof=1)),
                "skewness": float(rets.skew()),
                "kurtosis": float(rets.kurtosis()),
                "n_obs": len(rets),
            }

    def _compute_pairwise_correlation(self) -> None:
        n = len(self.index_names)
        corr_mat = np.eye(n, dtype=np.float64)
        target_range = int(self.max_history_years * self.trading_days)
        
        for i in range(n):
            name_i = self.index_names[i]
            for j in range(i + 1, n):
                name_j = self.index_names[j]

                common_idx = self.tracks[name_i].index.intersection(self.tracks[name_j].index)
                n_common = len(common_idx)
                
                start_index = max(0, n_common - target_range - 1)
                common_idx = common_idx[start_index:]

                rho = 0.0
                if len(common_idx) >= self.trading_days:
                    sub_i = self.tracks[name_i].loc[common_idx]
                    sub_j = self.tracks[name_j].loc[common_idx]

                    ind_tracks = {name_i: sub_i, name_j: sub_j}
                    
                    # Temporarily swapping properties to reuse internal logic for a subset
                    temp_tracks = self.tracks
                    self.tracks = ind_tracks
                    rets = self._compute_individual_log_returns()
                    self.tracks = temp_tracks 
                    
                    if name_i in rets and name_j in rets:
                        vals_i = rets[name_i].to_numpy()
                        vals_j = rets[name_j].to_numpy()
                        
                        std_i = np.std(vals_i, ddof=1)
                        std_j = np.std(vals_j, ddof=1)
                        
                        if std_i > 1e-12 and std_j > 1e-12:
                            rho_val = np.corrcoef(vals_i, vals_j)[0, 1]
                            if not np.isnan(rho_val):
                                rho = rho_val

                corr_mat[i, j] = rho
                corr_mat[j, i] = rho

        self.corr_df = pd.DataFrame(corr_mat, index=self.index_names, columns=self.index_names)

    def _cholesky_decomposition(self, corr_matrix: np.ndarray) -> np.ndarray:
        mat = np.array(corr_matrix, dtype=np.float64)

        try:
            return np.linalg.cholesky(mat)
        except np.linalg.LinAlgError:
            pass

        epsilons = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 9e-3]
        eye = np.eye(mat.shape[0], dtype=np.float64)

        for eps in epsilons:
            try:
                reg_mat = mat + eps * eye
                d = np.sqrt(np.diag(reg_mat))
                reg_mat = reg_mat / np.outer(d, d)
                return np.linalg.cholesky(reg_mat)
            except np.linalg.LinAlgError:
                continue

        raise ValueError("Unable to compute a valid Cholesky decomposition, even after regularization.")

    def _simulate_correlated_paths(
        self, S0: np.ndarray, drift: np.ndarray, volatility: np.ndarray, 
        L: np.ndarray, n_simulations: int, n_steps: int, dt: float, seed: int
    ) -> np.ndarray:
        
        n_assets = len(S0)
        rng = np.random.default_rng(seed)

        paths = np.zeros((n_simulations, n_steps + 1, n_assets), dtype=np.float64)
        paths[:, 0, :] = S0

        nu_dt = (drift - 0.5 * (volatility**2)) * dt
        vol_sqrt_dt = volatility * np.sqrt(dt)

        eps = rng.standard_normal((n_simulations, n_steps, n_assets))
        z = np.matmul(eps, L.T)

        log_increments = nu_dt + vol_sqrt_dt * z
        cumulative_log_rets = np.cumsum(log_increments, axis=1)

        paths[:, 1:, :] = S0 * np.exp(cumulative_log_rets)

        return paths

    def _generate_summary_payload(self, horizon: float, n_sim: int, n_steps: int, dt: float) -> dict[str, Any]:
        hist_stats_records = []
        for name in self.index_names:
            st = self.stats_dict[name]
            hist_stats_records.append({
                "Index": name,
                "Observations": f"{st['n_obs']:,}",
                "Window": f"{self.max_history_years:.0f} years",
                "Ann. vol.": f"{st['volatility']*100:.2f}%",
                "Ann. drift (μ)": f"{st['drift']*100:+.2f}%",
                "Skewness": f"{st['skewness']:.2f}",
                "Kurtosis": f"{st['kurtosis']:.2f}",
                "Initial spot": f"{self.tracks[name].iloc[-1]:,.2f}",
            })
            
        df_hist_stats = pd.DataFrame(hist_stats_records)

        return {
            "index_names": self.index_names,
            "corr": self.corr_df.to_dict(),
            "stats_dict": self.stats_dict,
            "hist_stats_table": df_hist_stats.to_dict("records"),
            "dt": dt,
            "n_steps": n_steps,
            "horizon": horizon,
            "n_sim": n_sim,
            "trading_days": self.trading_days,
            "shape": list(self.paths.shape) if self.paths is not None else [],
        }