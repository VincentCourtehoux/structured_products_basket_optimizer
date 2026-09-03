import re
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Literal

def fetch_historical_tracks(
    tickers: list[str], 
    start_date: str, 
    end_date: str
) -> dict[str, pd.DataFrame]:
    """
    Fetches historical daily close prices for a specified list of tickers.
    
    Args:
        tickers: List of Yahoo Finance ticker symbols.
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        
    Returns:
        Dictionary mapping each ticker to a DataFrame containing a 'Price' column 
        and a timezone-naive datetime index.
    """
    tracks = {}
    
    for ticker in tickers:
        track_data = yf.download(
            tickers=ticker, 
            start=start_date, 
            end=end_date, 
            progress=False
        )
        
        if track_data.empty:
            continue
            
        # Standardize index
        track_data.index = track_data.index.tz_localize(None)
        track_data.index.name = "Date"
        
        # Handle potential MultiIndex columns returned by newer yfinance versions
        if isinstance(track_data.columns, pd.MultiIndex):
            close_prices = track_data['Close'][ticker]
        else:
            close_prices = track_data['Close']
            
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]
            
        tracks[ticker] = close_prices.to_frame(name="Price")
            
    return tracks


def apply_backward_decrement(
    tracks: dict[str, pd.DataFrame],
    decrement_value: float,
    final_level: float = 1000.0,
    decrement_type: Literal["percentage", "points"] = "percentage"
) -> dict[str, pd.DataFrame]:
    """
    Applies a backward synthetic decrement to total return (TR) indices.
    
    Args:
        tracks: Dictionary of DataFrames containing TR prices.
        decrement_value: The decrement amount (annualized percentage or fixed points).
        final_level: The target final index level.
        decrement_type: The methodology applied ('percentage' or 'points').
        
    Returns:
        Dictionary of DataFrames with the decremented price series.
    """
    decremented_results = {}
    
    for ticker, df in tracks.items():
        n = len(df)
        if n < 2:
            decremented_results[ticker] = df.copy()
            continue
            
        dates = df.index
        tr_values = df["Price"].to_numpy()
        
        # Vectorized time deltas (in days)
        delta_t = np.zeros(n)
        delta_t[1:] = (dates[1:] - dates[:-1]).days
        
        # Pre-calculate TR ratios to minimize operations inside the loop
        tr_ratios = np.zeros(n)
        tr_ratios[1:] = tr_values[1:] / tr_values[:-1]
        
        dec_values = np.zeros(n)
        dec_values[-1] = final_level
        
        if decrement_type == "percentage":
            step_factors = tr_ratios - (decrement_value * delta_t) / 365.0
            for i in range(n - 1, 0, -1):
                dec_values[i-1] = dec_values[i] / step_factors[i]
                
        elif decrement_type == "points":
            tr_ratios_inv = np.zeros(n)
            tr_ratios_inv[1:] = 1.0 / tr_ratios[1:]
            day_factors = (decrement_value * delta_t) / 360.0
            
            for i in range(n - 1, 0, -1):
                dec_values[i-1] = (dec_values[i] + day_factors[i]) * tr_ratios_inv[i]
                
        else:
            raise ValueError("decrement_type must be either 'percentage' or 'points'")
            
        decremented_results[ticker] = pd.DataFrame(data={"Price": dec_values}, index=dates)
        
    return decremented_results


def export_results_to_excel(
    tracks: dict[str, pd.DataFrame], 
    output_filename: str = "tracks_output.xlsx"
) -> None:
    """
    Exports the processed tracks to an Excel workbook, assigning one sheet per ticker.
    
    Args:
        tracks: Dictionary of DataFrames to export.
        output_filename: Destination file path.
    """
    with pd.ExcelWriter(output_filename) as writer:
        for ticker, df in tracks.items():
            # Sanitize sheet name to comply with Excel constraints
            sanitized_name = re.sub(r'[\\/*?:\[\]]', '', str(ticker))
            sheet_name = sanitized_name[:31]
            df.to_excel(writer, sheet_name=sheet_name)


if __name__ == "__main__":
    tr_indices = ["^SP500TR", "^RUTTR"]
    
    raw_tracks = fetch_historical_tracks(
        tickers=tr_indices,
        start_date="2010-01-01",
        end_date="2026-08-31"
    )
    
    processed_tracks = apply_backward_decrement(
        tracks=raw_tracks,
        decrement_value=50.0,
        final_level=1000.0,
        decrement_type="points"
    )
    
    export_results_to_excel(
        tracks=processed_tracks,
        output_filename="tracks.xlsx"
    )