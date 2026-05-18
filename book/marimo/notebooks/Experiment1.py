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
    from interview.strategy import Strategy
    import cvxpy as cp
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl

    from interview.data import YukkaRepository


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

    repo = YukkaRepository()
    assets = repo.index.STOXX600.assets

    # Full STOXX 600 prices (membership-masked)
    prices_all = repo.prices(assets=assets, mask=True)

    # Filter to STOXX 100 constituents only (rank 1-100 by market cap)
    prices = repo.prices(assets=assets, mask=True, rank_range=(1, 100))
    prices
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
