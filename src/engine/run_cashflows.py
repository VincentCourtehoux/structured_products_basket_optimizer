"""
Cashflow evaluation pipeline for configured structured products on Monte Carlo paths.
"""

from typing import Any
import numpy as np
import pandas as pd
from src.engine.monte_carlo import MonteCarloSimulator
from src.engine.products_config import ProductFactory


class CashflowEngine:
    """
    Engine for evaluating structured product payoffs and cashflows 
    across simulated Monte Carlo paths.
    """

    def __init__(self, factory: Any):
        """
        Args:
            factory: An instance of ProductFactory used to generate pricers.
        """
        self.factory = factory

    def evaluate_portfolio(
        self,
        product_configs: list[dict[str, Any]],
        paths: np.ndarray,
        dt: float,
        index_names: list[str],
    ) -> dict[str, Any]:
        """
        Evaluates payoffs and cashflows for configured structured products.
        
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
            
            pricer = self.factory.create_pricer(cfg)
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
    Demonstrates the seamless interaction between the Simulator, 
    the Factory, and the Cashflow Engine without relying on global caches.
    
    Returns:
        Evaluation results payload.
    """
    # 1. Initialize and run the simulation engine
    # (Assuming tracks.xlsx is available in the run directory)
    simulator = MonteCarloSimulator(filepath="tracks.xlsx")
    simulator.prepare_data()
    simulator.simulate(horizon=8.0, n_sim=1000, seed=42)
    
    if simulator.paths is None:
        raise RuntimeError("Simulation failed to generate paths.")

    # 2. Initialize the product factory with the available indices
    factory = ProductFactory(available_indices=simulator.index_names)

    # 3. Retrieve product configurations
    p1_config = factory.get_preset("p1")
    p3_config = factory.get_preset("p3")

    # 4. Initialize the cashflow engine and evaluate
    engine = CashflowEngine(factory=factory)
    
    dt = 1.0 / simulator.trading_days
    
    evaluation_results = engine.evaluate_portfolio(
        product_configs=[p1_config, p3_config],
        paths=simulator.paths,
        dt=dt,
        index_names=simulator.index_names
    )
    
    return evaluation_results