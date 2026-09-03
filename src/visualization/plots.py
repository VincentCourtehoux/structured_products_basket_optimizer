import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any

# Colorblind-safe color palette suitable for academic publications
_ACADEMIC_COLORS = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion
    "#009E73",  # Bluish Green
    "#E69F00",  # Orange
    "#CC79A7",  # Reddish Purple
    "#56B4E9",  # Sky Blue
]


def _apply_academic_layout(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    """
    Applies a publication-ready academic aesthetic to the figure.
    Enforces Serif fonts, white backgrounds, explicit axis borders, and external legends.
    """
    fig.update_layout(
        title=dict(
            text=title, 
            font=dict(family="Times New Roman, Serif", size=16, color="black"), 
            x=0.5, 
            xanchor="center"
        ),
        xaxis_title=dict(text=x_title, font=dict(family="Times New Roman, Serif", size=14, color="black")),
        yaxis_title=dict(text=y_title, font=dict(family="Times New Roman, Serif", size=14, color="black")),
        font=dict(family="Times New Roman, Serif", size=12, color="black"),
        template="simple_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bordercolor="black",
            borderwidth=1,
            bgcolor="white"
        ),
        margin=dict(l=70, r=160, t=60, b=60)
    )
    
    # Enforce standard scientific axis styling (mirrored borders, outside ticks)
    axis_styling = dict(
        showgrid=True, gridwidth=0.5, gridcolor="#E5E5E5",
        showline=True, linewidth=1, linecolor="black", mirror=True,
        ticks="outside", ticklen=5, tickwidth=1, tickcolor="black",
        zeroline=False
    )
    
    fig.update_xaxes(**axis_styling)
    fig.update_yaxes(**axis_styling)
    
    return fig


def plot_simulated_paths(
    paths: np.ndarray, 
    index_names: list[str], 
    stats_dict: dict[str, dict[str, float]], 
    n_paths_to_plot: int = 50
) -> go.Figure:
    """
    Generates a grid of line charts showing a subset of Monte Carlo simulated paths 
    for the underlying assets, annotated with their respective drift and volatility.
    
    Args:
        paths: 3D array of simulated paths (n_simulations, n_steps + 1, n_assets).
        index_names: List of asset names corresponding to the last dimension of paths.
        stats_dict: Dictionary containing 'drift' and 'volatility' parameters per asset.
        n_paths_to_plot: Number of scenario paths to render (to avoid visual clutter).
        
    Returns:
        Formatted Plotly Figure.
    """
    n_assets = min(len(index_names), 4)
    rows = 2 if n_assets > 2 else 1
    cols = 2 if n_assets > 1 else 1
    
    fig = make_subplots(
        rows=rows, cols=cols, 
        subplot_titles=index_names[:n_assets],
        vertical_spacing=0.15,
        horizontal_spacing=0.10
    )
    
    steps = np.arange(paths.shape[1])
    
    for i in range(n_assets):
        row = (i // 2) + 1
        col = (i % 2) + 1
        asset_name = index_names[i]
        asset_paths = paths[:n_paths_to_plot, :, i]
        
        for j in range(n_paths_to_plot):
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=asset_paths[j],
                    mode='lines',
                    line=dict(color=_ACADEMIC_COLORS[i % len(_ACADEMIC_COLORS)], width=0.5),
                    opacity=0.3,
                    showlegend=False
                ),
                row=row, col=col
            )
            
        drift = stats_dict[asset_name]["drift"] * 100.0
        vol = stats_dict[asset_name]["volatility"] * 100.0
        
        # Plotly syntax constraint: axis 1 is "x", axis 2 is "x2", etc.
        x_axis_ref = f"x{i+1 if i > 0 else ''} domain"
        y_axis_ref = f"y{i+1 if i > 0 else ''} domain"
        
        fig.add_annotation(
            x=0.05, y=0.95, 
            xref=x_axis_ref, 
            yref=y_axis_ref,
            text=f"μ: {drift:+.2f}%<br>σ: {vol:.2f}%",
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black", borderwidth=1,
            font=dict(family="Times New Roman, Serif", size=12, color="black")
        )
        
        fig.update_xaxes(title_text="Time Steps", row=row, col=col)
        fig.update_yaxes(title_text="Asset Level", row=row, col=col)

    fig = _apply_academic_layout(
        fig, 
        title="Monte Carlo Simulated Asset Trajectories", 
        x_title="", 
        y_title=""
    )
    
    for annotation in fig['layout']['annotations']:
        if annotation['text'] in index_names:
            annotation['font'] = dict(family="Times New Roman, Serif", size=14, color="black")
            
    return fig

def plot_correlation_matrix(corr_df: pd.DataFrame) -> go.Figure:
    """
    Generates a heatmap representation of the asset correlation matrix.
    
    Args:
        corr_df: Pandas DataFrame representing the symmetric correlation matrix.
        
    Returns:
        Formatted Plotly Figure.
    """
    labels = corr_df.columns.tolist()
    z_values = corr_df.to_numpy()
    
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmin=-1.0, zmax=1.0,
        text=np.round(z_values, 2),
        texttemplate="%{text:.2f}",
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Correlation (ρ)",
                font=dict(family="Times New Roman, Serif", size=14, color="black")
            ),
            tickfont=dict(family="Times New Roman, Serif", size=12, color="black")
        )
    ))
    
    fig = _apply_academic_layout(
        fig,
        title="Underlying Assets Historical Correlation Matrix",
        x_title="",
        y_title=""
    )
    
    # Heatmaps require specific layout overrides to look professional
    fig.update_layout(
        yaxis=dict(autorange="reversed"),  # Matrix ordering convention
        width=650, height=600,
        margin=dict(l=80, r=80, t=80, b=80)
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="", mirror=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, ticks="", mirror=False)
    
    return fig

def plot_product_irr_distributions(standalone_irrs: dict[str, np.ndarray]) -> go.Figure:
    """
    Generates a 2x2 grid of probability density histograms for standalone product IRRs.
    
    Args:
        standalone_irrs: Dictionary mapping product names to a 1D array of simulated IRRs.
        
    Returns:
        Formatted Plotly Figure.
    """
    product_names = list(standalone_irrs.keys())[:4]
        
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=product_names,
        vertical_spacing=0.15,
        horizontal_spacing=0.10
    )
    
    for i, p_name in enumerate(product_names):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        irrs = standalone_irrs[p_name]
        valid_irrs = irrs[np.isfinite(irrs)]
        
        fig.add_trace(
            go.Histogram(
                x=valid_irrs,
                histnorm='probability density',
                name=p_name,
                marker_color=_ACADEMIC_COLORS[i % len(_ACADEMIC_COLORS)],
                opacity=0.8,
                showlegend=False,
                marker_line=dict(color="black", width=0.5)
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="IRR", tickformat=".1%", row=row, col=col)
        fig.update_yaxes(title_text="Density", row=row, col=col)

    fig = _apply_academic_layout(
        fig, 
        title="Standalone Product IRR Distributions Across Scenarios", 
        x_title="", 
        y_title=""
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(family="Times New Roman, Serif", size=14, color="black")
        
    return fig


def plot_cvar_efficient_frontier(
    all_evaluated_combos: list[dict[str, Any]], 
    best_per_target: dict[float, dict[str, Any]]
) -> go.Figure:
    """
    Plots the scatter of all evaluated combinations (Expected IRR vs CVaR) 
    and overlays the optimal Pareto frontier.
    
    Args:
        all_evaluated_combos: List of all simulated basket metrics.
        best_per_target: Dictionary of the optimized baskets per target return.
        
    Returns:
        Formatted Plotly Figure.
    """
    fig = go.Figure()

    cloud_cvar = [combo["cvar"] for combo in all_evaluated_combos if np.isfinite(combo["cvar"])]
    cloud_irr = [combo["irr"] for combo in all_evaluated_combos if np.isfinite(combo["irr"])]

    fig.add_trace(go.Scatter(
        x=cloud_cvar,
        y=cloud_irr,
        mode='markers',
        marker=dict(
            size=4,
            color='rgba(180, 180, 180, 0.4)', 
            line=dict(width=0)
        ),
        name='Evaluated Portfolios',
        showlegend=True
    ))

    targets = sorted(list(best_per_target.keys()))
    optimal_cvar = [best_per_target[t]["cvar"] for t in targets]
    optimal_irr = [best_per_target[t]["mean_irr"] for t in targets]

    fig.add_trace(go.Scatter(
        x=optimal_cvar,
        y=optimal_irr,
        mode='lines',
        line=dict(color='black', width=1.5, dash='dash'),
        name='Efficient Frontier',
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=optimal_cvar,
        y=optimal_irr,
        mode='markers',
        marker=dict(
            size=10,
            color='#0072B2',
            symbol='diamond',
            line=dict(color='black', width=1)
        ),
        name='Optimal Target Baskets',
        showlegend=True
    ))

    fig = _apply_academic_layout(
        fig,
        title="Optimization Landscape: Expected IRR vs. Tail Risk (CVaR)",
        x_title="Conditional Value-at-Risk (Worst-α Scenarios)",
        y_title="Expected Internal Rate of Return (IRR)"
    )
    
    fig.update_xaxes(tickformat=".1%")
    fig.update_yaxes(tickformat=".1%")
    
    return fig


def plot_optimal_basket_composition(
    best_per_target: dict[float, dict[str, Any]], 
    product_names: list[str]
) -> go.Figure:
    """
    Creates a stacked bar chart illustrating the capital allocation weight evolution 
    across the optimized target IRRs.
    
    Args:
        best_per_target: Dictionary mapping target IRRs to their optimal parameters.
        product_names: List of product identifiers.
        
    Returns:
        Formatted Plotly Figure.
    """
    targets = sorted(list(best_per_target.keys()))
    target_labels = [f"{t*100:.1f}%" for t in targets]
    
    weights_evolution = {p_name: [] for p_name in product_names}
    
    for t in targets:
        weights = best_per_target[t]["weights"]
        for p_idx, p_name in enumerate(product_names):
            weights_evolution[p_name].append(weights[p_idx])
            
    fig = go.Figure()
    
    for idx, p_name in enumerate(product_names):
        fig.add_trace(go.Bar(
            x=target_labels,
            y=weights_evolution[p_name],
            name=p_name,
            marker_color=_ACADEMIC_COLORS[idx % len(_ACADEMIC_COLORS)],
            marker_line=dict(color="black", width=0.5)
        ))

    fig.update_layout(barmode='stack')
    
    fig = _apply_academic_layout(
        fig,
        title="Optimal Portfolio Allocation Weights by Target Return",
        x_title="Target Portfolio Return (IRR)",
        y_title="Capital Allocation Weight"
    )
    
    fig.update_yaxes(tickformat=".0%", range=[0, 1.01])
    
    return fig