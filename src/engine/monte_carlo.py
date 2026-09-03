import os
import numpy as np
import pandas as pd
from typing import Any

# Global cache for server-side persistence
_SERVER_CACHE: dict[str, Any] = {}


def load_price_tracks(filepath: str) -> dict[str, pd.Series]:
    """
    Loads and cleans price tracks from an Excel workbook.
    Assumes one sheet per index, with the first column as Date and the second as Price.
    
    Args:
        filepath: Path to the Excel file.
        
    Returns:
        Dictionary mapping sheet names to cleaned, business-day-reindexed price Series.
        
    Raises:
        FileNotFoundError: If the specified Excel file does not exist.
        ValueError: If no valid time series can be extracted.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    xls = pd.ExcelFile(filepath)
    cleaned_tracks: dict[str, pd.Series] = {}

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if df.empty or len(df.columns) < 2:
            continue
        
        date_col = df.columns[0]
        price_col = df.columns[1]
        
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Ensure price column is numeric, dropping unparseable rows
        series = pd.to_numeric(df[price_col], errors="coerce").dropna()
        
        if len(series) < 10:
            continue

        series = series.ffill().dropna()

        if len(series) >= 10:
            cleaned_tracks[str(sheet_name).strip()] = series

    if not cleaned_tracks:
        raise ValueError(f"No valid time series could be extracted from {filepath}.")

    return cleaned_tracks


def compute_individual_log_returns(
    tracks: dict[str, pd.Series],
    trading_days: int = 252,
    max_history_years: float = 10.0,
) -> dict[str, pd.Series]:
    """
    Computes daily logarithmic returns over a specified maximum historical window.
    
    Args:
        tracks: Dictionary of price series.
        trading_days: Number of trading days in a year.
        max_history_years: Maximum number of years to include in the lookback window.
        
    Returns:
        Dictionary of logarithmic return series.
    """
    returns_dict: dict[str, pd.Series] = {}
    target_range = int(max_history_years * trading_days)
    
    for name, series in tracks.items():
        n_obs = len(series)
        start_index = max(0, n_obs - target_range - 1)
        sub_series = series.iloc[start_index:]

        log_ret = np.log(sub_series / sub_series.shift(1)).dropna()
        
        if len(log_ret) > trading_days:
            returns_dict[name] = log_ret

    return returns_dict


def compute_drift_volatility(
    individual_returns: dict[str, pd.Series],
    trading_days: int = 252,
) -> dict[str, dict[str, float]]:
    """
    Calculates annualized drift, volatility, and higher-order moments for each return series.
    
    Args:
        individual_returns: Dictionary of log return series.
        trading_days: Number of trading days in a year.
        
    Returns:
        Nested dictionary containing statistical moments per index.
    """
    stats: dict[str, dict[str, float]] = {}
    
    for name, rets in individual_returns.items():
        if len(rets) < trading_days:
            continue
            
        vol_ann = float(rets.std(ddof=1) * np.sqrt(trading_days))
        drift_ann = float(rets.mean() * trading_days + 0.5 * (vol_ann**2))
        
        stats[name] = {
            "volatility": vol_ann,
            "drift": drift_ann,
            "mean_daily": float(rets.mean()),
            "std_daily": float(rets.std(ddof=1)),
            "skewness": float(rets.skew()),
            "kurtosis": float(rets.kurtosis()),
            "n_obs": len(rets),
        }
        
    return stats


def compute_pairwise_correlation(
    tracks: dict[str, pd.Series],
    trading_days: int = 252,
    max_history_years: float = 10.0,
) -> pd.DataFrame:
    """
    Computes a symmetric correlation matrix based on pairwise date intersections.
    Yields a 0.0 correlation for pairs with insufficient overlapping history.
    
    Args:
        tracks: Dictionary of price series.
        trading_days: Number of trading days in a year.
        max_history_years: Maximum number of years to consider.
        
    Returns:
        DataFrame representing the correlation matrix.
    """
    names = list(tracks.keys())
    n = len(names)
    corr_mat = np.eye(n, dtype=np.float64)
    target_range = int(max_history_years * trading_days)
    
    for i in range(n):
        name_i = names[i]
        for j in range(i + 1, n):
            name_j = names[j]

            common_idx = tracks[name_i].index.intersection(tracks[name_j].index)
            n_common = len(common_idx)
            
            start_index = max(0, n_common - target_range - 1)
            common_idx = common_idx[start_index:]

            rho = 0.0
            if len(common_idx) >= trading_days:
                sub_i = tracks[name_i].loc[common_idx]
                sub_j = tracks[name_j].loc[common_idx]

                ind_tracks = {name_i: sub_i, name_j: sub_j}
                rets = compute_individual_log_returns(
                    tracks=ind_tracks, 
                    trading_days=trading_days, 
                    max_history_years=max_history_years
                )
                
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

    return pd.DataFrame(corr_mat, index=names, columns=names)


def cholesky_decomposition(corr_matrix: np.ndarray | pd.DataFrame) -> np.ndarray:
    """
    Performs Cholesky decomposition with automatic Tikhonov regularization 
    if the input correlation matrix is not perfectly positive definite.
    
    Args:
        corr_matrix: Correlation matrix.
        
    Returns:
        Lower triangular matrix L such that L @ L.T = correlation_matrix.
        
    Raises:
        ValueError: If decomposition fails even after maximum regularization.
    """
    mat = np.array(corr_matrix, dtype=np.float64)

    try:
        return np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        pass

    # Regularization search grid
    epsilons = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 9e-3]
    eye = np.eye(mat.shape[0], dtype=np.float64)

    for eps in epsilons:
        try:
            reg_mat = mat + eps * eye
            # Re-normalize to ensure unit diagonal
            d = np.sqrt(np.diag(reg_mat))
            reg_mat = reg_mat / np.outer(d, d)
            return np.linalg.cholesky(reg_mat)
        except np.linalg.LinAlgError:
            continue

    raise ValueError("Unable to compute a valid Cholesky decomposition, even after regularization.")


def simulate_correlated_paths(
    S0: np.ndarray,
    drift: np.ndarray,
    volatility: np.ndarray,
    L: np.ndarray,
    n_simulations: int,
    n_steps: int,
    dt: float = 1.0 / 252.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulates correlated Geometric Brownian Motion (GBM) paths via Euler-Maruyama discretization.
    
    Args:
        S0: Array of initial spot prices.
        drift: Array of annualized drift (mu) values.
        volatility: Array of annualized volatility (sigma) values.
        L: Lower triangular Cholesky decomposition of the correlation matrix.
        n_simulations: Number of Monte Carlo paths.
        n_steps: Number of simulation steps.
        dt: Time increment per step (typically 1/252).
        seed: Random seed for reproducibility.
        
    Returns:
        3D NumPy array of simulated paths with shape (n_simulations, n_steps + 1, n_assets).
    """
    n_assets = len(S0)
    rng = np.random.default_rng(seed)

    paths = np.zeros((n_simulations, n_steps + 1, n_assets), dtype=np.float64)
    paths[:, 0, :] = S0

    # Pre-calculate deterministic drift per step
    nu_dt = (drift - 0.5 * (volatility**2)) * dt
    vol_sqrt_dt = volatility * np.sqrt(dt)

    # Generate standard normal random increments and correlate them
    eps = rng.standard_normal((n_simulations, n_steps, n_assets))
    z = np.matmul(eps, L.T)

    # Compute geometric log increments and cumulative sum
    log_increments = nu_dt + vol_sqrt_dt * z
    cumulative_log_rets = np.cumsum(log_increments, axis=1)

    # Broadcast initial spots against cumulative returns
    paths[:, 1:, :] = S0 * np.exp(cumulative_log_rets)

    return paths


def run_simulation(
    filepath: str,
    max_history_years: float = 10.0,
    n_sim: int = 1000,
    horizon: float = 8.0,
    trading_days: int = 252,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Executes the end-to-end Monte Carlo simulation pipeline: data ingestion,
    statistical calibration, and correlated path generation.
    
    Args:
        filepath: Source Excel file containing price tracks.
        max_history_years: Maximum lookback window for calibration.
        n_sim: Number of paths to simulate.
        horizon: Simulation horizon in years.
        trading_days: Number of trading days per year.
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary payload containing calibration statistics, matrix states, and metadata.
    """
    tracks = load_price_tracks(filepath)
    index_names = list(tracks.keys())
    
    if not index_names:
        raise ValueError("No valid indices found in the provided Excel file.")

    # 1. Log returns on max_history_years
    indiv_rets = compute_individual_log_returns(
        tracks, 
        trading_days=trading_days, 
        max_history_years=max_history_years
    )
    stats_dict = compute_drift_volatility(indiv_rets, trading_days=trading_days)

    # 2. Pairwise correlation on full overlapping history
    corr_df = compute_pairwise_correlation(tracks, trading_days=trading_days)

    # 3. Prepare parameters for simulation arrays
    s0 = np.array([tracks[name].iloc[-1] for name in index_names], dtype=np.float64)
    drift = np.array([stats_dict[name]["drift"] for name in index_names], dtype=np.float64)
    vol = np.array([stats_dict[name]["volatility"] for name in index_names], dtype=np.float64)

    # 4. Cholesky decomposition
    L = cholesky_decomposition(corr_df.to_numpy())

    # 5. Simulate correlated paths
    n_steps = int(round(horizon * trading_days))
    dt = 1.0 / trading_days
    
    paths = simulate_correlated_paths(
        S0=s0,
        drift=drift,
        volatility=vol,
        L=L,
        n_simulations=n_sim,
        n_steps=n_steps,
        dt=dt,
        seed=seed,
    )

    # 6. Store state in server cache
    _SERVER_CACHE.update({
        "paths": paths,
        "index_names": index_names,
        "tracks": tracks,
        "corr_df": corr_df,
        "stats_dict": stats_dict,
        "params": {
            "n_sim": n_sim,
            "horizon": horizon,
            "trading_days": trading_days,
            "dt": dt,
            "n_steps": n_steps,
            "seed": seed,
        }
    })

    # 7. Prepare summary table for historical stats
    hist_stats_records = []
    for name in index_names:
        st = stats_dict[name]
        hist_stats_records.append({
            "Index": name,
            "Observations": f"{st['n_obs']:,}",
            "Window": f"{max_history_years:.0f} years",
            "Ann. vol.": f"{st['volatility']*100:.2f}%",
            "Ann. drift (μ)": f"{st['drift']*100:+.2f}%",
            "Skewness": f"{st['skewness']:.2f}",
            "Kurtosis": f"{st['kurtosis']:.2f}",
            "Initial spot": f"{tracks[name].iloc[-1]:,.2f}",
        })
        
    df_hist_stats = pd.DataFrame(hist_stats_records)

    return {
        "index_names": index_names,
        "corr": corr_df.to_dict(),
        "stats_dict": stats_dict,
        "hist_stats_table": df_hist_stats.to_dict("records"),
        "dt": dt,
        "n_steps": n_steps,
        "horizon": horizon,
        "n_sim": n_sim,
        "trading_days": trading_days,
        "shape": list(paths.shape),
    }