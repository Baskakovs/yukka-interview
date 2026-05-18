# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.19.6",
#     "numpy>=2.4.0",
#     "yukka-interview",
#     "jquantstats>=0.8.0",
#     "plotly>=5.0",
# ]
# [tool.uv.sources]
# yukka-interview = { path = "../../..", editable = true }
# ///
"""Experiment 2: Momentum Strategy Benchmark."""

import marimo

__generated_with = "0.23.6"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from jquantstats.data import Data
    from scipy.stats import t as t_dist

    from interview.data import YukkaRepository
    from interview.data.config import CACHE_DIR
    from interview.strategy import Strategy


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Part 2: Momentum Strategy Benchmark

    Yukka Lab produces news-sentiment scores for STOXX 600 constituents using NLP and ML. A natural benchmark for any sentiment-based strategy is pure price momentum. If sentiment cannot beat a signal that uses only past prices, it adds little value.

    I implemented the standard Jegadeesh & Titman 12-1 month momentum strategy on the top-100 STOXX 600 names by market-cap rank, establishing the baseline.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data

    Load daily STOXX 600 prices from *YukkaRepository*, deduplicate any
    repeated calendar dates (keeping the last entry per day), then resample to
    **monthly frequency** by taking the last observed price in each calendar
    month.  Keep the full masked column universe and let the monthly
    rebalance logic decide which assets have enough as-of trailing history.
    """)
    return


@app.cell
def _():
    repo = YukkaRepository()
    _assets_meta = repo.index.STOXX600.assets
    _prices_raw = repo.prices(assets=_assets_meta, mask=True, rank_range=(1, 100))

    # Deduplicate: YukkaRepository may yield repeated calendar dates.
    _prices_daily = _prices_raw.unique(subset=["date"], keep="last").sort("date")
    _asset_cols_raw = [c for c in _prices_daily.columns if c != "date"]

    # Monthly resample: last price of each calendar month.
    _prices_m = (
        _prices_daily.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg([pl.col(c).last() for c in _asset_cols_raw])
        .sort("month")
        .rename({"month": "date"})
    )

    # Keep the full rank-masked universe; no full-sample availability filter.
    prices_monthly = _prices_m
    return (prices_monthly,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Signal: 12-1 Month Momentum

    The raw signal for asset $i$ at month-end $t$ is:

    $$
    s_{i,t} = \frac{P_{i,t-1}}{P_{i,t-12}} - 1
    $$

    Skipping the most recent month ($t-1$ rather than $t$) avoids contamination
    from short-term bid-ask reversal.  Assets are ranked cross-sectionally each
    month; the portfolio is rebalanced monthly.
    """)
    return


@app.cell
def _(prices_monthly):
    _asset_cols = [c for c in prices_monthly.columns if c != "date"]

    # s[t] = price[t-1] / price[t-12] - 1  (one-month skip)
    signal = prices_monthly.select(
        "date",
        *[(pl.col(c).shift(1) / pl.col(c).shift(12) - 1).alias(c) for c in _asset_cols],
    )
    return (signal,)


@app.cell
def _(prices_monthly, signal):
    strategy = Strategy(prices=prices_monthly, signal=signal)
    ic = strategy.mean_ic

    # t-statistic for H₀: IC = 0,  df = n - 2
    n_ic = len(prices_monthly) - 1
    t_ic = ic * np.sqrt(n_ic - 2) / np.sqrt(max(1.0 - ic**2, 1e-12))
    p_val = 2.0 * (1.0 - t_dist.cdf(abs(t_ic), df=n_ic - 2))
    return ic, n_ic, p_val, strategy, t_ic


@app.cell(hide_code=True)
def _(ic, n_ic, p_val, t_ic):
    mo.md(rf"""
    ### Information Coefficient

    | Metric | Value |
    |--------|-------|
    | Mean IC | {ic:.4f} |
    | t-statistic | {t_ic:.2f} |
    | p-value | {p_val:.4f} |
    | N (months) | {n_ic} |
    """)
    return


@app.cell(hide_code=True)
def _(ic, p_val):
    _sig = "statistically significant" if p_val < 0.05 else "not statistically significant"
    mo.md(rf"""
    ## IC Interpretation

    A mean IC of **{ic:.4f}** ({_sig} at the 5 % level, p = {p_val:.4f}) measures
    the average cross-sectional rank correlation between the momentum signal and
    subsequent one-month returns.
    """)
    return


@app.cell
def _(strategy):
    # Latest rebalance snapshot for interpretability. The performance backtest
    # below re-solves the portfolio at each month instead of reusing this vector.
    _cols = strategy._asset_cols()
    weights = strategy.build_portfolio(
        objective="max_sharpe",
        long_only=True,
        max_weight=0.2,
        expected_returns="signal_rank",
    )

    # Top-10 holdings table.
    _order = np.argsort(weights)[::-1]
    _rows = "\n".join(f"| {i + 1} | {_cols[j]} | {weights[j]:.2%} |" for i, j in enumerate(_order[:10]))
    mo.md(rf"""
    ### Latest Rebalance Snapshot

    The table shows the latest available momentum-tilted max-Sharpe portfolio.
    The backtest below rebalances monthly; it does not hold this final
    snapshot through the full history.

    | Rank | Ticker | Weight |
    |------|--------|--------|
    {_rows}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Combining Signal and Risk Model

    The momentum signal serves as the **expected-return proxy** while the
    historical covariance matrix $\Sigma$ controls **risk**. At each monthly
    rebalance date $t$, the strategy uses only information available at that
    date: the contemporaneous 12-1 momentum rank $\hat{\mu}_t$ and a trailing
    covariance estimate $\Sigma_t$.

    $$
    \min_w \; w^{\top} \Sigma_t w \quad \text{s.t.} \quad \hat{\mu}_t^{\top} w = 1,\; w \geq 0,\; w \leq 0.20
    $$

    The resulting weights are normalised to sum to 1 and applied to the next
    month's returns. This removes the look-ahead bias of fitting one final
    portfolio and replaying it over the whole history.
    """)
    return


@app.cell
def _(strategy):
    # Monthly rebalanced, as-of backtest. Each row is labelled by the return
    # realisation month, matching the benchmark return dates below.
    _strategy_returns = strategy.backtest_rebalanced(
        objective="max_sharpe",
        long_only=True,
        max_weight=0.2,
        expected_returns="signal_rank",
        lookback=36,
        min_observations=12,
        transaction_cost_bps=10.0,
        return_col="Momentum",
    )

    # STOXX 600 benchmark: daily prices → monthly returns.
    _bench_raw = pl.read_parquet(CACHE_DIR / "benchmarks.parquet")
    _bench_m = (
        _bench_raw.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(pl.col("STOXX 600").last())
        .sort("month")
        .rename({"month": "date"})
        .with_columns((pl.col("STOXX 600") / pl.col("STOXX 600").shift(1) - 1).alias("STOXX 600"))
        .filter(pl.col("STOXX 600").is_not_null())
    )

    # Align on common dates, then remove both nulls and IEEE NaNs.
    combined = (
        _strategy_returns.join(_bench_m, on="date", how="inner")
        .drop_nulls()
        .filter(pl.all_horizontal(pl.all().exclude("date").is_finite()))
    )

    # jquantstats analytics - benchmark included so stats cover both series.
    _data = Data.from_returns(
        returns=combined.select(["date", "Momentum"]),
        benchmark=combined.select(["date", "STOXX 600"]),
        date_col="date",
    )
    _gross_data = Data.from_returns(
        returns=combined.select(["date", "Momentum_gross"]),
        date_col="date",
    )
    _s = _data.stats
    _gross_s = _gross_data.stats

    _sharpe = _s.sharpe()
    _mdd = _s.max_drawdown()
    _cagr = _s.cagr()
    _sortino = _s.sortino()
    _calmar = _s.calmar()
    _vol = _s.volatility()

    _mom_key = "Momentum"
    _gross_key = "Momentum_gross"
    _bm_key = "STOXX 600"
    _gross_sharpe = _gross_s.sharpe()
    _gross_cagr = _gross_s.cagr()
    _gross_mdd = _gross_s.max_drawdown()
    _attempted = max(strategy.prices.height - 2, 0)
    _successful = _strategy_returns.height
    _skipped = _attempted - _successful
    _avg_turnover = _strategy_returns["turnover"].mean()
    _avg_active = _strategy_returns["active_assets"].mean()
    _avg_cost_bps = _strategy_returns["transaction_cost"].mean() * 10_000

    mo.md(rf"""
    ### Performance Analytics

    Momentum returns are shown net of a simple 10 bps cost per unit of traded
    notional, applied to two-way monthly turnover.

    | Metric | Momentum | STOXX 600 |
    |--------|----------|-----------|
    | CAGR | {_cagr.get(_mom_key, float("nan")):.2%} | {_cagr.get(_bm_key, float("nan")):.2%} |
    | Ann. Volatility | {_vol.get(_mom_key, float("nan")):.2%} | {_vol.get(_bm_key, float("nan")):.2%} |
    | Sharpe ratio | {_sharpe.get(_mom_key, float("nan")):.3f} | {_sharpe.get(_bm_key, float("nan")):.3f} |
    | Sortino ratio | {_sortino.get(_mom_key, float("nan")):.3f} | {_sortino.get(_bm_key, float("nan")):.3f} |
    | Calmar ratio | {_calmar.get(_mom_key, float("nan")):.3f} | {_calmar.get(_bm_key, float("nan")):.3f} |
    | Max drawdown | {_mdd.get(_mom_key, float("nan")):.2%} | {_mdd.get(_bm_key, float("nan")):.2%} |

    ### Trading Cost Impact

    | Metric | Momentum Gross | Momentum Net |
    |--------|----------------|--------------|
    | CAGR | {_gross_cagr.get(_gross_key, float("nan")):.2%} | {_cagr.get(_mom_key, float("nan")):.2%} |
    | Sharpe ratio | {_gross_sharpe.get(_gross_key, float("nan")):.3f} | {_sharpe.get(_mom_key, float("nan")):.3f} |
    | Max drawdown | {_gross_mdd.get(_gross_key, float("nan")):.2%} | {_mdd.get(_mom_key, float("nan")):.2%} |

    ### Rebalance Diagnostics

    | Diagnostic | Value |
    |------------|-------|
    | Attempted rebalances | {_attempted} |
    | Successful rebalances | {_successful} |
    | Skipped rebalances | {_skipped} |
    | Average active holdings | {_avg_active:.1f} |
    | Average two-way monthly turnover | {_avg_turnover:.2f} |
    | Average transaction cost | {_avg_cost_bps:.1f} bps |
    """)
    return (combined,)


@app.cell
def _(combined):
    _dates = combined["date"].to_list()
    turnover_fig = go.Figure()
    turnover_fig.add_trace(
        go.Bar(
            x=_dates,
            y=combined["turnover"].to_list(),
            name="Two-Way Monthly Turnover",
            marker_color="steelblue",
            yaxis="y",
        )
    )
    turnover_fig.add_trace(
        go.Scatter(
            x=_dates,
            y=combined["active_assets"].to_list(),
            name="Active Holdings",
            line={"color": "darkorange", "width": 2},
            yaxis="y2",
        )
    )
    turnover_fig.update_layout(
        title="Two-Way Turnover and Active Holdings",
        xaxis_title="Date",
        yaxis={"title": "Two-way turnover", "tickformat": ".0%"},
        yaxis2={"title": "Active holdings", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        template="plotly_white",
        height=420,
    )

    turnover_fig
    return


@app.cell
def _(combined):
    _cum_mom = (1.0 + combined["Momentum"]).cum_prod()
    _cum_bench = (1.0 + combined["STOXX 600"]).cum_prod()
    _dates = combined["date"].to_list()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_dates,
            y=_cum_mom.to_list(),
            name="Momentum Strategy",
            line={"color": "steelblue", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_dates,
            y=_cum_bench.to_list(),
            name="STOXX 600",
            line={"color": "tomato", "width": 2, "dash": "dash"},
        )
    )
    fig.update_layout(
        title="Cumulative Returns: Momentum Strategy vs STOXX 600",
        xaxis_title="Date",
        yaxis_title="Growth of 1 EUR",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        template="plotly_white",
        height=480,
    )

    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Limitations & Extensions

    **Limitations**

    - *Simple transaction-cost model.* The 10 bps turnover charge is a rough
      implementation haircut; a realistic model would vary by liquidity,
      spread, and trade size.
    - *Universe construction risk.* The notebook assumes the rank data used by
      ``rank_range=(1, 100)`` is historical/as-of. If the rank file itself were
      rebuilt with hindsight, the universe would inherit that bias.
    - *Simple covariance model.* The covariance matrix uses a trailing sample
      covariance with light missing-data handling; a production risk model
      would use explicit shrinkage and more careful treatment of sparse assets.

    **Natural Extensions**

    1. **Transaction-cost-aware optimisation.** Add a penalty term
       $\gamma \|w - w_{\text{prev}}\|_1$ to the objective to reduce turnover
       while maintaining the momentum tilt.
    2. **Yukka sentiment overlay.** Replace or augment the price-momentum signal
       with Yukka's news-sentiment score.  If the IC of the sentiment signal is
       orthogonal to price momentum, a combined signal yields higher risk-adjusted
       returns with no additional turnover.
    3. **Risk-model tuning.** Sweep the covariance lookback, shrinkage method,
       and per-name cap to test whether results are robust.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
