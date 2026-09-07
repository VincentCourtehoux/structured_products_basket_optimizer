"""
Main execution pipeline for the Structured Product Basket Optimization.
Runs Monte Carlo simulations, optimizes portfolio weights via CVaR minimization, 
and exports publication-ready academic figures.
"""

from loguru import logger
from pathlib import Path

# Local module imports
from src.engine.monte_carlo import MonteCarloSimulator
from src.engine.products_config import ProductFactory
from src.engine.run_cashflows import CashflowEngine   
from src.optimization.optimizer import BasketOptimizer
from src.visualization.plots import (
    plot_product_irr_distributions,
    plot_cvar_efficient_frontier,
    plot_optimal_basket_composition,
    plot_simulated_paths,
    plot_correlation_matrix
)


def main() -> None:
    """
    Executes the end-to-end optimization and visualization pipeline.
    """
    # 1. Pipeline Configuration
    tracks_path = "tracks.xlsx"
    output_directory = Path("output_figures")
    
    # Optimization constraints
    n_combinations = 1000
    cvar_level = 0.01
    irr_targets = [i / 100.0 for i in range(4, 11)]  # Target IRRs from 4% to 10%
    irr_band = 0.001
    seed = 42
    n_products_per_basket = 4
    irr_metric = "median"

    # 2. Monte Carlo Simulation Engine
    logger.info("Initializing Monte Carlo simulation engine...")
    
    # Instantiate the class and run the data preparation and simulation
    simulator = MonteCarloSimulator(filepath=tracks_path)
    simulator.prepare_data()
    simulator.simulate(seed=seed)
    
    # Safely extract variables from the instance instead of the global cache
    paths = simulator.paths
    dt = 1.0 / simulator.trading_days
    index_names = simulator.index_names
    stats_dict = simulator.stats_dict
    corr_df = simulator.corr_df
    
    # 3. Product Configuration Parsing
    logger.info("Loading structured product configurations...")
    
    # Initialize the factory with the indices available from the simulation
    factory = ProductFactory(available_indices=index_names)
    
    # Load structured product configuration presets via the factory
    product_configs = [
        factory.get_preset("p1", index_names[1]),
        factory.get_preset("p2", index_names[0]),
        factory.get_preset("p3", index_names[0]),
        factory.get_preset("p4", index_names[1]),
    ]
    
    # Create your own custom structured product
    p5 = {
        "name": "P5 - Phoenix Autocall",
        "type": "autocall",
        "index_names": [index_names[0]],
        "basket_type": "single",
        "maturity": 8.0,
        "obs_freq": "quarterly",
        "autocall_barrier_initial": 1.00,
        "barrier_step_down": 0.0,
        "barrier_step_freq": "annual",
        "first_call_year": 1.0,
        "barrier_step_start_year": 0.0,
        "coupon_rate": 0.08,
        "coupon_barrier": None,
        "memory_coupon": True,
        "capital_barrier": 0.60,
        "capital_barrier_type": "european",
        "notional": 1.0,
    }
    
    # Validate the custom product before adding it to the optimization pool
    is_valid, msg = factory.validate_config(p5)
    if not is_valid:
        logger.error(f"Configuration error for P5: {msg}")
        raise ValueError(f"Failed to validate P5: {msg}")
        
    # product_configs.append(p5)

    # 4. Basket Optimization Execution
    logger.info("Executing Conditional Value-at-Risk (CVaR) basket optimization...")
    
    # Instantiate the Cashflow Engine with the Factory
    engine = CashflowEngine(factory=factory)
    
    optimizer = BasketOptimizer(
        cashflow_engine=engine,
        n_combinations=n_combinations,
        cvar_level=cvar_level,
        seed=seed,
        irr_band=irr_band
    )
    
    results = optimizer.optimize(
        product_configs=product_configs,
        paths=paths,
        dt=dt,
        index_names=index_names,
        irr_targets=irr_targets,
        n_products_per_basket=n_products_per_basket,
        irr_metric=irr_metric
    )

    # 5. Data Visualization Generation
    logger.info("Generating optimization landscape and simulation visualizations...")
    
    fig_sim_paths = plot_simulated_paths(
        paths=paths,
        index_names=index_names,
        stats_dict=stats_dict,
        n_paths_to_plot=50
    )
    
    fig_corr_matrix = plot_correlation_matrix(
        corr_df=corr_df
    )
    
    fig_distributions = plot_product_irr_distributions(
        standalone_irrs=results["standalone_irrs"]
    )
    
    fig_frontier = plot_cvar_efficient_frontier(
        all_evaluated_combos=results["all_evaluated"],
        best_per_target=results["best_per_target"]
    )
    
    fig_composition = plot_optimal_basket_composition(
        best_per_target=results["best_per_target"],
        product_names=results["product_names"]
    )

    # Display figures in local environment
    fig_sim_paths.show()
    fig_corr_matrix.show()
    fig_distributions.show()
    fig_frontier.show()
    fig_composition.show()
    
    logger.info("Optimization pipeline completed successfully.")


if __name__ == "__main__":
    main()