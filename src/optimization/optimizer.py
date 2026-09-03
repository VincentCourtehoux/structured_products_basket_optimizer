"""
Multi-Product Basket Optimizer Engine (IRR Vectorization & CVaR Frontier).

Optimization Logic:
1. Compute L combinations of weights.
2. Calculate the mean expected cashflows for each generated weight combination (n_combinations * n_dates).
3. Compute the IRR of these mean cashflows and filter those falling within a specific target band.
4. For the matching combinations, reconstruct the full simulation cashflow matrix 
   (n_simulations * n_dates * n_target_combinations).
5. Compute the individual path IRRs for these specific combinations.
6. Determine the Value at Risk (VaR) at the alpha level for each weights combination. 
7. Filter paths with an IRR below the VaR, average their cashflows, and compute the final Conditional VaR (CVaR).
8. Select the weight configuration that maximizes CVaR for each given target IRR.
"""

from typing import Any
import numpy as np
import pandas as pd

from src.engine.run_cashflows import evaluate_product_cashflows


def build_global_date_grid(product_cashflows: list[dict[str, Any]]) -> np.ndarray:
    """
    Builds a unified, sorted array of unique observation dates across all products.
    
    Args:
        product_cashflows: List of product result dictionaries containing 'obs_dates'.
        
    Returns:
        1D NumPy array of unique observation dates.
    """
    unique_dates = set()
    for item in product_cashflows:
        for d in item["obs_dates"]:
            unique_dates.add(round(float(d), 6))
            
    if not unique_dates:
        unique_dates.add(1.0)
        
    return np.array(sorted(list(unique_dates)), dtype=np.float64)


def build_product_matrices(
    product_cashflows: list[dict[str, Any]],
    global_dates: np.ndarray,
    n_sim: int,
) -> list[np.ndarray]:
    """
    Maps each product's cashflow matrix onto the standardized global date grid.
    
    Args:
        product_cashflows: List of product cashflow data.
        global_dates: Standardized date grid.
        n_sim: Number of Monte Carlo simulations.
        
    Returns:
        List of 2D arrays, each of shape (n_sim, len(global_dates)).
    """
    matrices = []
    n_dates = len(global_dates)

    for item in product_cashflows:
        cf_orig = item["cashflows"]
        obs_dates = item["obs_dates"]
        cf_grid = np.zeros((n_sim, n_dates), dtype=np.float64)

        for local_idx, t_obs in enumerate(obs_dates):
            diffs = np.abs(global_dates - round(float(t_obs), 6))
            global_idx = int(np.argmin(diffs))
            if diffs[global_idx] < 1e-4:
                cf_grid[:, global_idx] += cf_orig[:, local_idx]

        matrices.append(cf_grid)

    return matrices


def irr_vectorized(
    cf_matrix: np.ndarray,
    dates: np.ndarray,
    initial_investment: float = 1.0,
    n_iter: int = 50,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    Vectorized Newton-Raphson solver for Annualized IRR across all scenarios.
    Solves: -initial_investment + sum_j [ cf_j / (1 + r)^t_j ] = 0
    
    Args:
        cf_matrix: 2D array of cashflows (scenarios, dates).
        dates: 1D array of observation dates.
        initial_investment: Outflow at t=0.
        n_iter: Maximum Newton-Raphson iterations.
        tol: Convergence tolerance.
        
    Returns:
        1D array of computed IRRs. Returns NaN for non-converging or invalid paths.
    """
    n_sim = cf_matrix.shape[0]
    total_cf = np.sum(cf_matrix, axis=1)
    max_t = np.max(dates) if len(dates) > 0 and np.max(dates) > 0 else 1.0

    valid_mask = total_cf > 1e-6
    r = np.full(n_sim, np.nan, dtype=np.float64)

    # Initial guess formulation
    ratio = np.maximum(1e-6, total_cf[valid_mask] / initial_investment)
    r_guess = np.power(ratio, 1.0 / max_t) - 1.0
    r[valid_mask] = np.clip(r_guess, -0.5, 5.0)

    active = valid_mask.copy()
    dates_arr = dates.reshape(1, -1)

    for _ in range(n_iter):
        if not np.any(active):
            break

        r_curr = r[active].reshape(-1, 1)
        one_plus_r = np.maximum(1e-5, 1.0 + r_curr)

        # Discount factor and cashflow subset
        disc = np.power(one_plus_r, -dates_arr)
        cf_sub = cf_matrix[active, :]

        # Function and derivative for Newton-Raphson
        f_val = np.sum(cf_sub * disc, axis=1) - initial_investment
        f_prime = np.sum(-dates_arr * cf_sub * (disc / one_plus_r), axis=1)

        # Handle gradient singularity
        singular = np.abs(f_prime) < 1e-12
        f_prime = np.where(singular, -1e-6, f_prime)

        step = f_val / f_prime
        r_next = np.clip(r[active] - step, -0.9999, 50.0)

        converged = np.abs(step) < tol
        indices_active = np.where(active)[0]
        r[indices_active] = r_next

        active[indices_active[converged]] = False
        active[indices_active[singular]] = False

    # Final residual check to invalidate diverging paths
    r_final = r.reshape(-1, 1)
    one_plus_r_final = np.maximum(1e-5, 1.0 + r_final)
    disc_final = np.power(one_plus_r_final, -dates_arr)
    residual = np.abs(np.sum(cf_matrix * disc_final, axis=1) - initial_investment)
    
    return np.where(residual < 1e-3, r, np.nan)


def evaluate_basket_combinations(
    product_matrices: list[np.ndarray],
    global_dates: np.ndarray,
    n_combinations: int = 2000,
    initial_investment: float = 1.0,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates Dirichlet-distributed random weight combinations, calculates mean 
    cashflows per combination, and computes their corresponding expected IRR.
    
    Args:
        product_matrices: List of (n_sim, n_dates) arrays per product.
        global_dates: 1D array of global observation dates.
        n_combinations: Number of weight configurations to simulate.
        initial_investment: Initial investment amount.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple containing weights matrix (K, M), mean basket cashflows (K, T), 
        and combination IRRs (K,).
    """
    product_cf_3d = np.array(product_matrices)
    m_products = product_cf_3d.shape[0]
    
    rng = np.random.default_rng(seed)
    alpha = np.ones(m_products)
    weights_matrix = rng.dirichlet(alpha, size=n_combinations)
    
    # Normalize rounded weights to prevent floating point drift
    weights_matrix = np.round(weights_matrix, 4)
    weights_matrix = weights_matrix / weights_matrix.sum(axis=1, keepdims=True)
    
    mean_product_cf = np.mean(product_cf_3d, axis=1)
    mean_basket_cf = np.dot(weights_matrix, mean_product_cf)
    
    combination_irrs = irr_vectorized(
        cf_matrix=mean_basket_cf,
        dates=global_dates,
        initial_investment=initial_investment
    )
    
    return weights_matrix, mean_basket_cf, combination_irrs


def run_basket_optimization(
    product_configs: list[dict[str, Any]],
    paths: np.ndarray,
    dt: float,
    index_names: list[str],
    n_combinations: int = 2000,
    n_products_per_basket: int = 4,
    irr_metric: str = "median",
    cvar_level: float = 0.05,
    seed: int = 42,
    irr_targets: list[float] | None = None,
    irr_band: float = 0.005,
) -> dict[str, Any]:
    """
    Executes the CVaR optimization engine across randomized product weightings.
    
    Args:
        product_configs: Product configuration schemas.
        paths: Monte Carlo paths tensor.
        dt: Time delta per step.
        index_names: Underlying index identifiers.
        n_combinations: Total random combinations to test.
        n_products_per_basket: Maximum products permitted (currently unused logic constraint).
        irr_metric: Evaluation metric constraint ('median' or 'mean').
        cvar_level: Alpha level for Expected Shortfall calculation.
        seed: PRNG seed.
        irr_targets: Target expected IRRs to optimize around.
        irr_band: Tolerance band around targets (e.g., +/- 0.5%).
        
    Returns:
        Dictionary payload containing frontier results, optimized combinations, and metadata.
    """
    if irr_targets is None:
        irr_targets = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

    if len(product_configs) < 2:
        raise ValueError("At least 2 products are required for basket optimization.")

    evaluated = evaluate_product_cashflows(product_configs, paths, dt, index_names)
    product_cashflows = [evaluated["results"][cfg.get("pid", cfg.get("name"))] for cfg in product_configs]
    product_names = [cfg.get("name", f"P_{i}") for i, cfg in enumerate(product_configs)]

    n_sim = paths.shape[0]
    global_dates = build_global_date_grid(product_cashflows)
    product_matrices = build_product_matrices(product_cashflows, global_dates, n_sim)
    
    product_cf_3d = np.array(product_matrices)

    standalone_irrs = {}
    for p_name, p_matrix in zip(product_names, product_matrices):
        standalone_irrs[p_name] = irr_vectorized(
            cf_matrix=p_matrix, 
            dates=global_dates, 
            initial_investment=1.0
        )

    weights_matrix, _, combination_irrs = evaluate_basket_combinations(
        product_matrices=product_matrices,
        global_dates=global_dates,
        n_combinations=n_combinations,
        seed=seed
    )

    best_per_target = {}
    target_results = []
    all_evaluated_combos = []

    for tgt in irr_targets:
        min_irr = tgt - irr_band
        max_irr = tgt + irr_band

        valid_mask = (combination_irrs >= min_irr) & (combination_irrs <= max_irr)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        matched_weights = weights_matrix[valid_indices]
        matched_sim_cfs = np.einsum('km,mnt->knt', matched_weights, product_cf_3d)

        best_cvar = -np.inf
        best_combo_data = None

        for idx, original_combo_id in enumerate(valid_indices):
            sim_cf = matched_sim_cfs[idx]
            sim_irrs = irr_vectorized(sim_cf, global_dates, initial_investment=1.0)
            
            finite_irrs = sim_irrs[np.isfinite(sim_irrs)]
            if len(finite_irrs) < 10:
                continue

            var_val = float(np.percentile(finite_irrs, cvar_level * 100.0))
            tail_mask = np.nan_to_num(sim_irrs, nan=np.inf) <= var_val
            
            if not np.any(tail_mask):
                continue

            tail_cfs = sim_cf[tail_mask]
            mean_tail_cf = np.mean(tail_cfs, axis=0)
            
            mean_tail_cf_2d = mean_tail_cf.reshape(1, -1)
            cvar_arr = irr_vectorized(mean_tail_cf_2d, global_dates, initial_investment=1.0)
            cvar_val = float(cvar_arr[0])

            if np.isfinite(cvar_val):
                all_evaluated_combos.append({
                    "cvar": cvar_val,
                    "irr": float(combination_irrs[original_combo_id])
                })

                if cvar_val > best_cvar:
                    best_cvar = cvar_val
                    best_combo_data = {
                        "combo_id": original_combo_id,
                        "target_irr": tgt,
                        "mean_irr": float(combination_irrs[original_combo_id]),
                        "cvar": cvar_val,
                        "var": var_val,
                        "weights": weights_matrix[original_combo_id].tolist(),
                        "sim_irrs": finite_irrs,
                    }

        if best_combo_data is not None:
            best_per_target[tgt] = best_combo_data
            
            comp_strs = [
                f"{product_names[i]} ({w*100:.1f}%)" 
                for i, w in enumerate(best_combo_data["weights"]) if w > 0.001
            ]
            
            target_results.append({
                "Target IRR": f"{tgt*100:.1f}%",
                "Composition": " + ".join(comp_strs),
                "Mean IRR": f"{best_combo_data['mean_irr']*100:.2f}%",
                "CVaR": f"{best_combo_data['cvar']*100:.2f}%",
                "VaR": f"{best_combo_data['var']*100:.2f}%"
            })

    return {
        "best_per_target": best_per_target,
        "target_results": target_results,
        "global_dates": global_dates.tolist(),
        "product_names": product_names,
        "all_evaluated": all_evaluated_combos,
        "initial_combination_irrs": combination_irrs.tolist(),
        "standalone_irrs": standalone_irrs
    }