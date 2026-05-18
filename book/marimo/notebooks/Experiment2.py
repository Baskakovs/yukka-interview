# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.19.6",
#     "numpy>=2.4.0",
#     "yukka-interview",
#     "cvxpy>=1.8.2",
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
    import cvxpy as cp
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

    **12-1 month cross-sectional momentum** on the STOXX 600 universe (top-100
    constituents by market-cap rank), benchmarked against the STOXX 600 index.

    Jegadeesh & Titman (1993) documented that past 12-month winners continue to
    outperform past losers over the following 3–12 months.  On STOXX 600 data,
    the same effect provides a natural benchmark: it represents the *structural*
    return that markets reward for holding recent winners, independent of any
    external information signal.  Comparing the Yukka news-sentiment alpha
    against pure price momentum reveals whether the sentiment signal is additive
    or merely a noisy proxy for what prices already reflect.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data

    We load daily STOXX 600 prices from *YukkaRepository*, deduplicate any
    repeated calendar dates (keeping the last entry per day), then resample to
    **monthly frequency** by taking the last observed price in each calendar
    month.  Stocks with more than 80 % missing values — illiquid micro-caps that
    rarely appear in the top-100 — are dropped so the covariance matrix stays
    well-conditioned.
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

    # Drop columns with > 80 % nulls (stocks rarely in top-100).
    _n = len(_prices_m)
    _good = ["date"] + [c for c in _asset_cols_raw if _prices_m[c].null_count() / _n <= 0.80]
    prices_monthly = _prices_m.select(_good)

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

    return strategy, ic, n_ic, t_ic, p_val


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
    subsequent one-month returns.  An IC in the range 0.02–0.05 is considered
    economically meaningful in live equity strategies — modest per-period
    predictive skill compounds into substantial alpha when the portfolio is
    well-diversified and turnover costs are managed.
    """)
    return


@app.cell
def _(strategy):
    # Min-variance optimisation with 10 % per-asset weight cap.
    _cols = strategy._asset_cols()
    _n = len(_cols)
    _ret = strategy._historical_returns().select(_cols).to_numpy().astype(float)
    _valid = np.all(np.isfinite(_ret), axis=1)
    _ret = _ret[_valid]
    _sigma = np.cov(_ret, rowvar=False) + 1e-6 * np.eye(_n)

    _w = cp.Variable(_n)
    cp.Problem(
        cp.Minimize(cp.quad_form(_w, _sigma)),
        [cp.sum(_w) == 1, _w >= 0, _w <= 0.10],
    ).solve()
    weights = _w.value

    # Top-10 holdings table.
    _order = np.argsort(weights)[::-1]
    _rows = "\n".join(f"| {i + 1} | {_cols[j]} | {weights[j]:.2%} |" for i, j in enumerate(_order[:10]))
    mo.md(rf"""
    ### Portfolio: Min-Variance, 10 % Weight Cap

    Markowitz minimum-variance optimisation constrained to
    $w_i \in [0,\, 0.10]$ and $\sum_i w_i = 1$.

    | Rank | Ticker | Weight |
    |------|--------|--------|
    {_rows}
    """)

    return (weights,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Signal vs Optimiser

    The momentum signal ranks assets by recent price performance; the optimiser
    selects weights to minimise portfolio variance.  These two objectives are
    **orthogonal**: the optimiser does not tilt toward high-IC momentum names.
    A natural extension is *signal-tilted* optimisation, which adds
    $-\lambda\,\hat{\mu}^{\top} w$ to the objective (where $\hat{\mu}$ is the
    signal-implied expected return), blending alpha capture with risk control
    along the full mean-variance frontier.
    """)
    return


@app.cell
def _(strategy, weights):
    # Portfolio forward returns aligned with the price dates.
    _cols = strategy._asset_cols()
    _fwd = strategy._forward_returns().select(_cols).to_numpy().astype(float)
    _port_ret = _fwd @ weights
    _dates = strategy.prices["date"].to_list()
    _port_df = pl.DataFrame({"date": _dates, "Momentum": _port_ret.tolist()})

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

    # Align on common dates, drop any remaining nulls.
    combined = _port_df.join(_bench_m, on="date", how="inner").drop_nulls()

    # jquantstats analytics — benchmark included so stats cover both series.
    _data = Data.from_returns(
        returns=combined.select(["date", "Momentum"]),
        benchmark=combined.select(["date", "STOXX 600"]),
        date_col="date",
    )
    _s = _data.stats

    _sharpe = _s.sharpe()
    _mdd = _s.max_drawdown()
    _cagr = _s.cagr()
    _sortino = _s.sortino()
    _calmar = _s.calmar()
    _vol = _s.volatility()

    _mom_key = "Momentum"
    _bm_key = "STOXX 600"

    mo.md(rf"""
    ### Performance Analytics

    | Metric | Momentum | STOXX 600 |
    |--------|----------|-----------|
    | CAGR | {_cagr.get(_mom_key, float("nan")):.2%} | {_cagr.get(_bm_key, float("nan")):.2%} |
    | Ann. Volatility | {_vol.get(_mom_key, float("nan")):.2%} | {_vol.get(_bm_key, float("nan")):.2%} |
    | Sharpe ratio | {_sharpe.get(_mom_key, float("nan")):.3f} | {_sharpe.get(_bm_key, float("nan")):.3f} |
    | Sortino ratio | {_sortino.get(_mom_key, float("nan")):.3f} | {_sortino.get(_bm_key, float("nan")):.3f} |
    | Calmar ratio | {_calmar.get(_mom_key, float("nan")):.3f} | {_calmar.get(_bm_key, float("nan")):.3f} |
    | Max drawdown | {_mdd.get(_mom_key, float("nan")):.2%} | {_mdd.get(_bm_key, float("nan")):.2%} |
    """)

    return (combined,)


@app.cell
def _(combined):
    _cum_mom = (1.0 + combined["Momentum"]).cumprod()
    _cum_bench = (1.0 + combined["STOXX 600"]).cumprod()
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
    return (fig,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Limitations & Extensions

    **Limitations**

    - *No transaction-cost model.* Monthly rebalancing of ~100 names incurs
      meaningful slippage; a realistic backtest should deduct at least 5–10 bps
      per trade, reducing reported returns noticeably.
    - *Survivorship-adjacent bias.* Filtering by STOXX 100 rank uses the
      full-sample rank file; stocks that later dropped out of the index may be
      underrepresented in the early period.
    - *Single covariance window.* The covariance matrix is estimated over the
      entire history, mixing volatility regimes.  Weights optimised on a calm
      period may be dangerously concentrated when a crisis arrives.

    **Natural Extensions**

    1. **Signal-tilted optimisation.** Add $-\lambda\,\hat{\mu}^{\top} w$ to the
       objective to align the optimiser with the momentum ranking, sweeping
       $\lambda$ to trace the mean-variance frontier.
    2. **Yukka sentiment overlay.** Replace or augment the price-momentum signal
       with Yukka's news-sentiment score.  If the IC of the sentiment signal is
       orthogonal to price momentum, a combined signal yields higher risk-adjusted
       returns with no additional turnover.
    3. **Rolling covariance.** Use a 36- or 60-month expanding or rolling window
       to track regime changes rather than relying on the full-sample estimate.
    4. **Transaction-cost-aware rebalancing.** Add a penalty term
       $\gamma \|w - w_{\text{prev}}\|_1$ to the objective to reduce turnover
       while maintaining the momentum tilt.
    """)
    return


if __name__ == "__main__":
    app.run()
