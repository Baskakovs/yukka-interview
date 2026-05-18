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
    from interview.strategy import Strategy


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data

    We use the *YukkaRepository()* class to import the price data for the
    STOXX 600 companies, then show how to request a rank-masked top-100
    universe through ``rank_range=(1, 100)``. This notebook is a lightweight
    data-loading scratchpad; the momentum benchmark itself is implemented in
    ``Experiment2.py``.
    """)
    return


@app.cell
def _():
    from yukka.data import Index

    repo = YukkaRepository()
    assets = repo.index.STOXX600.assets

    # Full STOXX 600 prices (membership-masked)
    prices_all = repo.prices(assets=assets, mask=True)

    # Filter to top-100 STOXX 600 names by rank.
    prices = repo.prices(assets=assets, mask=True, rank_range=(1, 100))
    prices
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
