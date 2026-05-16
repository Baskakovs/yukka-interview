# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.19.6",
#     "numpy>=2.4.0",
#     "yukka-interview",
#     "cvxpy>=1.8.2",
# ]
# [tool.uv.sources]
# yukka-interview = { path = "../../..", editable = true }
# ///
"""Experiment 1: Momentum Strategy."""

import marimo

__generated_with = "0.23.6"
app = marimo.App()

with app.setup:
    import cvxpy as cp
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl

    from interview.data import YukkaRepository


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Introduction

    In this notebook you will build a momentum strategy on European equities (STOXX 600 universe).

    The data layer is already provided. Your task is to:

    1. Construct a **momentum signal** from the price data.
    2. Evaluate signal quality using the **information coefficient (IC)**.
    3. Build a **long-only Markowitz portfolio** using `cvxpy`.
    4. Compute the **annualised Sharpe ratio** of the resulting strategy.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data

    We use the *YukkaRepository()* class to import the price data for the
    STOXX 600 companies, then filter to stocks that were ever in the
    STOXX 100 (by rank <= 100). This gives ~130 assets, ensuring the
    entire pipeline operates on a well-conditioned universe.
    The data ranges from January 2016 to December 2025. The price data
    is resampled on the last trading day each month.
    """)
    return


@app.cell
def _():
    from yukka.data import Index

    repo = YukkaRepository(index=Index.STOXX600)
    assets = repo.assets

    # Full STOXX 600 prices (membership-masked)
    prices_all = repo.prices(assets=assets, mask=True)

    # Filter to STOXX 100 constituents only (rank 1-100 by market cap)
    prices = repo.prices(assets=assets, mask=True, rank_range=(1, 100))
    return prices, prices_all


@app.cell
def _(prices, prices_all):
    # Resample daily prices to month-end (last available trading day per month) for later IC analysis
    prices_monthly = (
        prices.sort("date")
        .group_by(pl.col("date").dt.year().alias("_y"), pl.col("date").dt.month().alias("_m"), maintain_order=True)
        .last()
        .drop("_y", "_m")
    )
    # Full STOXX 600 month-end prices for the old benchmark momentum strategy
    prices_all_monthly = (
        prices_all.sort("date")
        .group_by(pl.col("date").dt.year().alias("_y"), pl.col("date").dt.month().alias("_m"), maintain_order=True)
        .last()
        .drop("_y", "_m")
    )
    return prices_all_monthly, prices_monthly


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Your Task

    Using the `prices_monthly` and `prices_all_monthly` DataFrames above, implement the following:

    ### Part 1: Signal & IC Analysis

    1. **Momentum signal**: compute a cross-sectional momentum signal from prices
       (e.g. 12-month return, or 12-1 month return skipping the most recent month).
    2. **Information Coefficient (IC)**: measure the rank correlation between your signal
       and forward 1-month returns. Report the mean IC across all months.

    ### Part 2: Portfolio Construction

    3. **Markowitz optimisation**: use `cvxpy` to build a long-only portfolio that
       maximises expected return (using your signal as the alpha forecast) subject to
       a risk budget (e.g. constrain portfolio variance using a sample covariance matrix).
    4. **Backtest**: compute the monthly portfolio returns and report the
       **annualised Sharpe ratio**.

    ### Hints

    - `prices_monthly` contains ~130 STOXX 100 stocks (month-end prices).
    - Returns: `price[t] / price[t-1] - 1` for simple returns.
    - IC: `scipy.stats.spearmanr` or polars rank correlation.
    - For Markowitz, you can use a rolling or expanding sample covariance.
    - Sharpe ratio: `mean(excess_returns) / std(excess_returns) * sqrt(12)` for monthly data.
    """)
    return


if __name__ == "__main__":
    app.run()
