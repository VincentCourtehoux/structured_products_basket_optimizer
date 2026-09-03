# Structured Product Basket Optimizer: CVaR & IRR Efficient Frontier

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)

## Overview
This repository contains a quantitative pricing and optimization engine designed for multi-asset structured products. It simulates market environments using Correlated Geometric Brownian Motion (GBM), evaluates structured products (e.g., Autocalls, Capital Guaranteed products), and optimizes portfolio allocation by maximizing the Conditional Value-at-Risk (CVaR) for a given target Expected Internal Rate of Return (IRR).

The engine is built for performance, utilizing highly vectorized NumPy operations, including a custom vectorized Newton-Raphson IRR solver, and produces publication-ready visualizations.

## Key Features
- **Synthetic Decrement Indices:** Reconstructs synthetic decrement indices (percentage or fixed points) from historical Total Return (TR) market data.
- **Correlated Market Simulation:** Generates Monte Carlo price paths using GBM, incorporating historical drift, volatility, and Cholesky decomposition with automatic Tikhonov regularization.
- **Exotic Payoff Pricing:** Modular pricer for complex structured products (Autocallable notes with memory coupons, Lookback Capital Guaranteed, Delta One).
- **Tail-Risk Optimization:** Constructs an efficient frontier by maximizing CVaR (Expected Shortfall) across Dirichlet-distributed random portfolio weights.
- **Publication-Ready Visualizations:** Automated generation of academic-grade Plotly charts (Simulated paths, Correlation matrices, IRR density, and Efficient Frontiers).

## Methodology

The pipeline executes in four distinct phases:

### 1. Data Ingestion & Decrement Construction
Standard Price Return indices do not reflect true performance, and Total Return indices are heavily impacted by dividend taxation. The engine ingests TR indices (e.g., `^SP500TR`, `^GDAXI`) and applies a synthetic backward decrement.
* **Mechanism:** Subtracts a fixed synthetic dividend (in percentage or index points) linearly on a daily basis from the historical TR series.

### 2. Market Calibration & Monte Carlo Simulation
Historical log-returns are computed to extract annualized drift ($\mu$), volatility ($\sigma$), and the pairwise correlation matrix ($\rho$). 
* **Simulation:** Assets are simulated under the physical measure ($\mathbb{P}$) using a Correlated Geometric Brownian Motion:
  $$S_{i}(t+\Delta t) = S_{i}(t) \exp \left( \left( \mu_i - \frac{\sigma_i^2}{2} \right)\Delta t + \sigma_i \sqrt{\Delta t} Z_i \right)$$
  Where $Z$ is generated via $Z = \epsilon \cdot L^T$, with $\epsilon \sim \mathcal{N}(0, I)$ and $L$ being the lower triangular Cholesky matrix of $\rho$.

### 3. Cashflow Generation
For each structured product configuration, the engine maps the product's term sheet rules (observation dates, coupon barriers, memory effects, capital protection) against the simulated market paths to generate a cashflow matrix of shape `(N_simulations, N_dates)`.

### 4. Portfolio Optimization (CVaR / IRR)
Instead of relying on standard Mean-Variance optimization, the engine optimizes for tail risk:
1. **Weight Generation:** Generates thousands of portfolio configurations using a Dirichlet distribution ($\sum w_i = 1$).
2. **Target IRR Filtering:** Computes the mean expected cashflows for each combination using a vectorized Newton-Raphson solver and filters configurations that fall within specific target IRR bands.
3. **Tail Risk (CVaR) Evaluation:** For matching combinations, the engine calculates the individual scenario IRRs, determines the Value-at-Risk (VaR) at the $\alpha$% level, and averages the cashflows of the worst-case scenarios to find the Conditional VaR (CVaR).
4. **Frontier Extraction:** Selects the weight composition that maximizes the CVaR for each target Expected IRR.


## Usage Guide

The pipeline is highly modular. Below is the step-by-step guide to configuring and running the optimizer.

### Step 1: Market Data Preparation
You can provide data to the engine in two ways:

**Option A: Automated Fetch & Decrement (via Retriever)**
Use the built-in retriever to pull Total Return indices from Yahoo Finance and apply a synthetic decrement before exporting to Excel.
```python
from src.data.retriever import fetch_historical_tracks, apply_backward_decrement, export_results_to_excel

raw_tracks = fetch_historical_tracks(["^SP500TR", "^GDAXI"], "2015-01-01", "2026-08-31")
dec_tracks = apply_backward_decrement(raw_tracks, decrement_value=50.0, decrement_type="points")
export_results_to_excel(dec_tracks, "tracks.xlsx")
```

**Option B: Custom Excel File**
Directly plug in your own tracks.xlsx file. The engine requires exactly one sheet per index, where the first column is the Date and the second column is the Price.

### Step 2: Monte Carlo Simulation Setup
Once tracks.xlsx is ready, trigger the simulation engine. This calibrates historical parameters and caches the correlated paths.

```python
from src.engine.monte_carlo import run_simulation, _SERVER_CACHE

# Run simulation (defaults to 10-year lookback, 1000 paths, 8-year horizon)
run_simulation(filepath="tracks.xlsx", max_history_years=10.0, n_sim=2000, horizon=8.0)

# Retrieve generated parameters from cache
paths = _SERVER_CACHE["paths"]
dt = _SERVER_CACHE["params"]["dt"]
index_names = _SERVER_CACHE["index_names"]
```

### Step 3: Structuring the basket
Select and configure the structured products you want to include in the optimization universe. You can use preset configurations or define custom dictionaries.

```python
from src.engine.products_config import get_default_preset

product_configs = [
    get_default_preset("p1", index="¨SP500TR", available_indices=index_names), # Step-down Autocall
    get_default_preset("p2", index=None, available_indices=index_names), # ATM Autocall
    get_default_preset("p3", index=None, available_indices=index_names), # Capital Guaranteed
    get_default_preset("p4", index=None, available_indices=index_names), # Delta One Tracker
]
```

### Step 4: Configuring and Running the Optimizer
Pass the simulated paths and product configurations into the optimizer. Key parameters to tune:
- irr_targets: The expected returns you want to target on the efficient frontier.
- irr_band: The tolerance around the target (e.g., 0.001 allows portfolios within $\pm 0.1\\%$ of the target).
- cvar_level: The tail risk percentile (e.g., 0.01 optimizes based on the worst 1% of scenarios).
- irr_metric: Determines if portfolio expected IRR is calculated via the mean or median of scenario IRRs.

```python
from src.optimization.optimizer import run_basket_optimization

results = run_basket_optimization(
    product_configs=product_configs,
    paths=paths,
    dt=dt,
    index_names=index_names,
    n_combinations=5000,            # Number of random Dirichlet weights to test
    cvar_level=0.01,                # 99% CVaR optimization
    irr_targets=[0.05, 0.06, 0.07, 0.08, 0.09, 0.10], 
    irr_band=0.001,                 # +/- 10 bps tolerance around target
    seed=42
)
```

For full execution and academic PDF generation, refer to main.py.







## Project Architecture

```text
├── src/
│   ├── data/
│   │   └── retriever.py            # Tracks retriever and decrement computation
│   ├── engine/
│   │   ├── monte_carlo.py          # GBM simulation and statistical calibration
│   │   ├── products_config.py      # Dataclasses and term sheet presets
│   │   └── run_cashflows.py        # Product cashflow aggregation
│   ├── products/
│   │   ├── autocall.py             # Autocall pricing logic
│   │   ├── capital_guaranteed.py   # Capital protected pricing logic
│   │   └── delta_one.py            # Linear payoff pricing logic
│   ├── optimization/
│   │   └── optimizer.py            # CVaR frontier and vectorized IRR solver
│   └── visualization/
│       └── plots.py                # Academic Plotly rendering functions
├── output_figures/                 # Generated high-res PDF visualizations
├── main.py                         # Main execution pipeline
├── requirements.txt                # Project dependencies
└── README.md