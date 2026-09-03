"""
Cashflow evaluation pipeline for configured structured products on Monte Carlo paths.
"""

from typing import Any
import numpy as np
import pandas as pd

from src.engine.products_config import create_pricer_from_config, get_default_preset
from src.engine.monte_carlo import _SERVER_CACHE


def evaluate_product_cashflows(
    product_configs: list[dict[str, Any]],
    paths: np.ndarray,
    dt: float,
    index_names: list[str],
) -> dict[str, Any]:
    """
    Evaluates payoffs and cashflows for configured structured products against Monte Carlo paths.
    
    Args:
        product_configs: List of configuration dictionaries for each product.
        paths: 3D NumPy array of simulated market paths (simulations, steps, assets).
        dt: Time increment per simulation step in years.
        index_names: List of underlying asset tickers.
        
    Returns:
        Dictionary containing detailed product results, individual summaries, 
        and an aggregated DataFrame of all summaries.
    """
    results: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []

    for cfg in product_configs:
        pid = str(cfg.get("pid", cfg.get("name", "Product")))
        name = str(cfg.get("name", f"Product_{pid}"))
        
        pricer = create_pricer_from_config(cfg)
        result = pricer.compute_payoffs(paths, dt, index_names)
        summary = result.summary()

        results[pid] = {
            "result": result,
            "cashflows": result.cashflow_matrix,
            "obs_dates": result.obs_dates,
            "summary": summary,
            "name": name,
            "notional": result.notional,
        }
        summaries.append(summary)

    return {
        "results": results,
        "summaries": summaries,
        "df_summary": pd.DataFrame(summaries),
    }


def main() -> dict[str, Any]:
    """
    CLI runner for standalone execution.
    Retrieves cached simulation paths and evaluates default product presets.
    
    Returns:
        Evaluation results payload.
        
    Raises:
        RuntimeError: If the Monte Carlo simulation cache is missing or empty.
    """
    if "paths" not in _SERVER_CACHE:
        raise RuntimeError("No simulation found in cache. Please execute monte_carlo_simulation first.")

    paths = _SERVER_CACHE["paths"]
    dt = float(_SERVER_CACHE["params"]["dt"])
    index_names = _SERVER_CACHE["index_names"]

    p1_config = get_default_preset("p1", available_indices=index_names)
    p3_config = get_default_preset("p3", available_indices=index_names)

    evaluation_results = evaluate_product_cashflows(
        product_configs=[p1_config, p3_config], 
        paths=paths, 
        dt=dt, 
        index_names=index_names
    )
    
    return evaluation_results


if __name__ == "__main__":
    main()