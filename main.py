"""
Main execution pipeline for the Structured Product Basket Optimization.
Runs Monte Carlo simulations, optimizes portfolio weights via CVaR minimization, 
and exports publication-ready academic figures.
"""

from loguru import logger
from pathlib import Path

# Local module imports
from src.engine.monte_carlo import run_simulation, _SERVER_CACHE
from src.engine.products_config import get_default_preset
from src.optimization.optimizer import run_basket_optimization
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
    irr_targets = [i / 100.0 for i in range(5, 15)]  # Target IRRs from 5% to 14%
    irr_band = 0.001
    seed = 42
    n_products_per_basket = 5
    irr_metric = "median"

    # 2. Monte Carlo Simulation Engine
    logger.info("Initializing Monte Carlo simulation engine...")
    run_simulation(filepath=tracks_path)
    
    paths = _SERVER_CACHE["paths"]
    dt = _SERVER_CACHE["params"]["dt"]
    index_names = _SERVER_CACHE["index_names"]
    stats_dict = _SERVER_CACHE["stats_dict"]
    corr_df = _SERVER_CACHE["corr_df"]
    
    # 3. Product Configuration Parsing
    logger.info("Loading structured product configurations...")
    # Load stuctured product configuration presets
    product_configs = [
        get_default_preset("p1", index=None, available_indices=index_names),
        get_default_preset("p2", index=None, available_indices=index_names),
        get_default_preset("p3", index=None, available_indices=index_names),
        get_default_preset("p4", index=None, available_indices=index_names),
    ]
    # Or create your own structured product
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
    product_configs.append(p5)

    # 4. Basket Optimization Execution
    logger.info("Executing Conditional Value-at-Risk (CVaR) basket optimization...")
    results = run_basket_optimization(
        product_configs=product_configs,
        paths=paths,
        dt=dt,
        index_names=index_names,
        n_combinations=n_combinations,
        n_products_per_basket=n_products_per_basket,
        irr_metric=irr_metric,
        cvar_level=cvar_level,
        seed=seed,
        irr_targets=irr_targets,
        irr_band=irr_band,
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

    # 6. Academic Output Export
    logger.info("Exporting high-resolution figures for academic publication...")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    fig_sim_paths.write_image(output_directory / "fig_simulated_paths.pdf")
    fig_corr_matrix.write_image(output_directory / "fig_correlation_matrix.pdf")
    fig_distributions.write_image(output_directory / "fig_standalone_distributions.pdf")
    fig_frontier.write_image(output_directory / "fig_efficient_frontier.pdf")
    fig_composition.write_image(output_directory / "fig_basket_composition.pdf")
    
    logger.info("Optimization pipeline completed successfully. Figures saved to /%s", output_directory.name)


if __name__ == "__main__":
    main()