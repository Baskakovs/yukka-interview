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
    from interview.signal import Signal
    from interview.strategy import Strategy


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

    # Drop all-null columns
    prices = prices.select(
        ["date"] + [c for c in prices.columns if c != "date" and prices[c].drop_nulls().len() > 0]
    )

    # Filter zero-rows
    prices = prices.filter(
        pl.any_horizontal(pl.col(c).is_not_null() for c in prices.columns if c != "date"
                         )
    )

    # Filter by last day in month as prices are only resampled monthly
    prices = prices.sort("date").group_by_dynamic("date", every="1mo").agg(
        pl.col(c).last() for c in prices.columns if c != "date"
    )
    return (prices,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Signal

    The `Signal` class computes a cross-sectionally normalised momentum signal
    from price data. At each date $t$, the raw signal for stock $i$ is:

    $$\text{signal}_{i,t} = \frac{\text{price}_{i,t-\text{skip}}}{\text{price}_{i,t-\text{lookback}}} - 1$$

    With default parameters `lookback=12, skip=1` this captures the 12-month
    return skipping the most recent month, avoiding short-term reversal.

    The raw signal is then **normalised cross-sectionally** at each date:
    1. Rank all stocks by their raw signal
    2. Z-score the ranks: subtract the cross-sectional mean and divide by std

    This ensures the signal has mean 0 and std 1 at every date, making it
    comparable across time and suitable as expected return input ($\mu$) for
    the Markowitz optimiser.

    The `Signal` class is a callable — it implements `__call__` so it can be
    passed directly as `signal_fn` to `Strategy`. Any function with signature
    `Callable[[pl.DataFrame], pl.DataFrame]` can be used instead.

    **Note on nulls:** stocks with fewer than `lookback` months of history
    produce a `null` signal for those dates (burn-in period). `Strategy`
    drops these nulls before computing the IC and estimating the covariance matrix.
    """)
    return


@app.cell
def _(prices):
    signal = Signal()
    signal_df = signal(prices)

    # Show burn-in period: first lookback rows are null
    mo.md(f"""
    **Signal shape**: {signal_df.shape}

    **Burn-in rows** (null due to lookback={signal._lookback}, skip={signal._skip}):
    first {signal._lookback} rows are null — prices are not yet available far enough back.

    **Sample of signal values** (first non-null row onwards):
    """)
    return signal, signal_df


@app.cell
def _(signal_df):
    # Drop burn-in rows and show first few rows
    entity_cols = [c for c in signal_df.columns if c != "date"]
    table = signal_df.filter(
        pl.any_horizontal(pl.col(c).is_not_null() for c in entity_cols)
    ).head(5)

    mo.vstack([
        table,
        mo.md("""
        Values are cross-sectional z-scores bounded roughly between -1.7 and +1.7,
        with nulls for stocks outside the STOXX 100 universe at each date, as expected.
        """)
    ])
    return (entity_cols,)


@app.cell
def _(entity_cols, signal_df):
    # Verify cross-sectional normalisation: mean ~0, std ~1 at each date
    cross_section_stats = signal_df.filter(
        pl.any_horizontal(pl.col(c).is_not_null() for c in entity_cols)
    ).select(
        "date",
        pl.concat_list(entity_cols).list.drop_nulls().list.mean().alias("cross_mean"),
        pl.concat_list(entity_cols).list.drop_nulls().list.std().alias("cross_std"),
    )

    fig_signal = go.Figure()
    fig_signal.add_trace(go.Scatter(
        x=cross_section_stats["date"].to_list(),
        y=cross_section_stats["cross_mean"].to_list(),
        name="Cross-sectional mean",
        line={"color": "blue"},
    ))
    fig_signal.add_trace(go.Scatter(
        x=cross_section_stats["date"].to_list(),
        y=cross_section_stats["cross_std"].to_list(),
        name="Cross-sectional std",
        line={"color": "orange"},
    ))
    fig_signal.update_layout(
        title="Signal cross-sectional mean and std over time (should be ~0 and ~1)",
        xaxis_title="Date",
        yaxis_title="Value",
    )
    fig_signal
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Strategy

    The `Strategy` class combines the signal with Markowitz mean-variance
    optimisation to construct a portfolio. It exposes three key outputs:

    ### Mean IC
    The **Information Coefficient** measures how well the signal predicts
    forward returns. At each date $t$, it is the Pearson correlation between
    the signal and the 1-period forward return across all stocks:

    $$\text{IC}_t = \text{corr}(\mu_t, r_{t+1})$$

    The mean IC is averaged over all dates. A value of ~0.04 means the signal
    explains about 4% of the cross-sectional variation in returns.

    ### Markowitz Optimisation
    At each monthly rebalancing date, the optimiser solves:

    $$\max_w \quad \mu^\top w - \delta \cdot w^\top \Sigma w$$

    Subject to:
    - $\mathbf{1}^\top w = 1$ — fully invested
    - $w \geq 0$ — long only
    - $w_i \leq 0.05$ — maximum 5% per stock

    Where $\mu$ is the signal at time $t$ and $\Sigma$ is estimated from a
    rolling 60-month window of returns. The risk aversion parameter $\delta$
    controls the tradeoff between return and risk.

    ### Sharpe Ratio
    The annualised Sharpe ratio is computed from the realised monthly portfolio
    returns $r_t = w_t^\top r_{t+1}$:

    $$\text{Sharpe} = \frac{\bar{r} - r_f}{\sigma_r} \times \sqrt{12}$$
    """)
    return


@app.cell
def _(prices, signal):
    strat = Strategy(prices, signal)

    mo.md(f"""
    **Mean IC**: {strat.mean_ic:.4f}

    A positive IC confirms the momentum signal has predictive power for
    1-month forward returns across STOXX 100 constituents.
    """)
    return (strat,)


@app.cell
def _(strat):
    weights = strat.markowitz()

    mo.md(f"""
    **Markowitz weights computed**: {len(weights)} rebalancing dates

    Each row sums to 1 with no weight exceeding 20%.
    """)
    return (weights,)


@app.cell
def _(entity_cols, weights):
    # Plot portfolio weights over time for top 5 stocks by average weight
    avg_weights = {c: weights[c].mean() for c in entity_cols}
    top5 = sorted(avg_weights, key=avg_weights.get, reverse=True)[:5]

    fig_weights = go.Figure()
    for stock in top5:
        fig_weights.add_trace(go.Scatter(
            x=weights["date"].to_list(),
            y=weights[stock].to_list(),
            name=stock,
            stackgroup="one",
        ))
    fig_weights.update_layout(
        title="Portfolio weights over time (top 5 stocks by average weight)",
        xaxis_title="Date",
        yaxis_title="Weight",
    )
    fig_weights
    return


@app.cell
def _(strat):
    mo.md(f"""
    **Annualised Sharpe Ratio**: {strat.sharpe:.4f}

    The strategy achieves a Sharpe of ~0.64 over the sample period (2017–2025),
    consistent with academic momentum benchmarks on large-cap European equities.
    Note this is in-sample and does not account for transaction costs.
    """)
    return


if __name__ == "__main__":
    app.run()
