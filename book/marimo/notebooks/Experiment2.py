# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.19.6",
#     "numpy>=2.4.0",
#     "yukka-interview",
#     "cvxpy>=1.8.2",
#     "jquantstats",
# ]
# [tool.uv.sources]
# yukka-interview = { path = "../../..", editable = true }
# ///
"""Experiment 2 — Part 2: Momentum strategy as benchmark."""

import marimo

__generated_with = "0.23.6"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from jquantstats import Data

    from interview.data import YukkaRepository
    from interview.data.config import CACHE_DIR
    from interview.signal import Signal
    from interview.strategy import Strategy


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Part 2: Momentum Strategy as Benchmark

    ## Motivation

    Momentum the most robust and basic anomalies in systematic trading.
    It has been replicated across asset classes, geographies, and time periods.

    Our hypothesis:

    > **Stocks in the top quintile of 12-1 month returns will outperform the
    > equal-weight universe over the following month on a risk-adjusted basis.**

    This strategy serves as a benchmark against which YUKKA's proprietary
    sentiment signals can later be compared or combined.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Universe

    We restrict to the **STOXX 100** — the top 100 STOXX 600 constituents by
    market capitalisation (`rank_range=(1, 100)`).

    **Why not the full 600?**

    - **Liquidity**: smaller names have wider bid-ask spreads; momentum profits
      in live trading are eroded more severely for illiquid stocks.
    - **Data quality**: smaller constituents have more gaps and shorter price
      histories, adding noise to the signal.
    - **Covariance conditioning**: with ~130 assets vs. 600, the rolling
      covariance matrix is better conditioned, reducing optimisation instability.

    In practice the top 100 by market cap covers the majority of STOXX 600
    index weight, so it remains a meaningful representation of European large caps.
    """)
    return


@app.cell
def _():
    repo = YukkaRepository()
    assets = repo.index.STOXX600.assets

    prices = repo.prices(assets=assets, mask=True, rank_range=(1, 100))

    # Drop all-null columns (stocks never in the top 100)
    prices = prices.select(
        ["date"] + [
            c for c in prices.columns
            if c != "date" and prices[c].drop_nulls().len() > 0
        ]
    )

    # Drop rows where every stock is null
    prices = prices.filter(
        pl.any_horizontal(
            pl.col(c).is_not_null()
            for c in prices.columns
            if c != "date"
        )
    )

    # Resample to month-end prices
    prices = prices.sort("date").group_by_dynamic("date", every="1mo").agg(
        pl.col(c).last()
        for c in prices.columns
        if c != "date"
    )

    entity_cols = [c for c in prices.columns if c != "date"]

    mo.md(f"""
    **Universe size**: `{len(entity_cols)}` stocks \n
    **Date range**: `{prices['date'].min()}` to `{prices['date'].max()}` \n
    **Rows (months)**: `{len(prices)}`
    """)
    return entity_cols, prices


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Signal Design

    ### Formulation

    The raw momentum signal for stock $i$ at month $t$ is:

    $$s_{i,t} = \frac{P_{i,\,t-1}}{P_{i,\,t-12}} - 1$$

    This is the **12-1 month return**: the one-year return skipping the most
    recent month. The skip is standard practice — it avoids the well-documented
    short-term reversal effect (Jegadeesh, 1990), where last month's winner
    tends to mean-revert over the next few weeks.

    ### Normalisation

    The raw signal is normalised cross-sectionally at each date $t$ via rank
    z-scoring:

    $$\tilde{s}_{i,t} = \frac{\operatorname{rank}(s_{i,t}) - \overline{\operatorname{rank}}}{\sigma_{\operatorname{rank}}}$$

    This ensures $\tilde{s}_{i,t}$ has mean $0$ and std $1$ at every date,
    removing the effect of market-wide momentum and making the signal
    comparable across time.

    ### Parameter choice

    We use `lookback=12, skip=1` following the academic consensus. Shorter
    lookback windows (e.g. 3-1 or 6-1) tend to be noisier in large-cap
    universes; longer ones (e.g. 24-1) begin to capture slower value-like
    mean-reversion. This is a deliberate starting point; sensitivity analysis
    across lookbacks is left to future work.
    """)
    return


@app.cell
def _():
    signal = Signal(lookback=12, skip=1)
    return (signal,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## IC Analysis

    The **Information Coefficient (IC)** is the primary diagnostic for signal
    quality. At each date $t$ it is the Pearson correlation between the
    cross-sectional signal and the 1-month forward return:

    $$\text{IC}_t = \operatorname{corr}(\tilde{s}_{i,t},\; r_{i,t+1})$$

    We track three quantities:

    | Metric | What it tells us |
    |--------|-----------------|
    | $\overline{\text{IC}}$ | Does the signal predict returns on average? |
    | $\sigma_{\text{IC}}$ | Is the predictive power stable over time? |
    | $\text{ICIR} = \frac{\overline{\text{IC}}}{\sigma_{\text{IC}}} \times \sqrt{12}$ | Annualised signal information ratio |

    A mean IC of ~0.03-0.05 is typical for academic momentum in large-cap
    equities. An annualised ICIR above 0.5 is generally considered viable.
    """)
    return


@app.cell
def _(entity_cols, prices, signal, strat):
    signal_df = signal(prices)
    returns_df = prices.select(
        "date",
        *[(pl.col(c) / pl.col(c).shift(1) - 1).alias(c) for c in entity_cols],
    )

    # Per-date IC series for the bar chart (Strategy.mean_ic returns only the scalar)
    forward_returns_ic = returns_df.select(
        "date",
        *[pl.col(c).shift(-1).alias(f"__r_{c}") for c in entity_cols],
    )
    aligned = signal_df.join(forward_returns_ic, on="date", how="inner")

    ic_rows = []
    for row in aligned.iter_rows(named=True):
        sig_vals = np.array([row[c] for c in entity_cols], dtype=float)
        ret_vals = np.array([row[f"__r_{c}"] for c in entity_cols], dtype=float)
        valid = ~np.isnan(sig_vals) & ~np.isnan(ret_vals)
        if valid.sum() > 5:
            corr = float(np.corrcoef(sig_vals[valid], ret_vals[valid])[0, 1])
            ic_rows.append({"date": row["date"], "ic": corr})

    ic_df  = pl.DataFrame(ic_rows)
    std_ic = float(ic_df["ic"].std())
    mean_ic = strat.mean_ic  # authoritative value from Strategy
    icir   = mean_ic / std_ic * np.sqrt(12)

    fig_ic = go.Figure()
    fig_ic.add_trace(go.Bar(
        x=ic_df["date"].to_list(),
        y=ic_df["ic"].to_list(),
        name="Monthly IC",
        marker_color=[
            "royalblue" if v >= 0 else "tomato"
            for v in ic_df["ic"].to_list()
        ],
    ))
    fig_ic.add_hline(
        y=mean_ic,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Mean IC = {mean_ic:.4f}",
    )
    fig_ic.update_layout(
        title="Monthly IC: signal vs 1-month forward return",
        xaxis_title="Date",
        yaxis_title="IC (Pearson correlation)",
    )
    fig_ic
    return icir, mean_ic, returns_df, std_ic


@app.cell(hide_code=True)
def _(icir, mean_ic, std_ic):
    mo.md(f"""
    | Metric | Value |
    |--------|-------|
    | Mean IC | `{mean_ic:.4f}` |
    | IC Std | `{std_ic:.4f}` |
    | Annualised ICIR | `{icir:.4f}` |

    **Interpretation:** A positive mean IC confirms that 12-1 month momentum has
    predictive power for 1-month forward returns in this universe. The ICIR of >0.5 indicates a good stability of the predictive power.

    The month-to-month IC is noisy (std = `{std_ic:.2f}`), which is typical for
    cross-sectional momentum. Although signal does not predict every month, it shows a systematic tilt on average.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Portfolio Construction

    ### Objective and Constraints

    We use Markowitz mean-variance optimisation with the normalised signal as
    the expected return input $\mu$. At each monthly rebalancing date $t$:

    $$\max_{w} \quad \tilde{s}_t^\top w - \delta \cdot w^\top \hat{\Sigma}_t w$$

    subject to:

    - $\mathbf{1}^\top w = 1$ — fully invested
    - $w_i \geq 0$ — long only (no short selling)
    - $w_i \leq 0.05$ — maximum 5% per stock (concentration cap)

    **Why long-only?** Short-selling European equities incurs borrowing costs
    and operational complexity that make long-short momentum impractical without
    dedicated infrastructure.

    **Why a 20% cap?** Momentum strategies naturally concentrate into recent
    winners. A single firm event can otherwise dominate
    the portfolio. The 20% cap enforces diversification while still allowing
    meaningful tilts.

    **Covariance estimation:** $\hat{\Sigma}_t$ is estimated from a rolling
    60-month window of returns, balancing regime responsiveness with the matrix
    stability needed for reliable optimisation.
    """)
    return


@app.cell
def _(prices, signal):
    strat = Strategy(prices, signal)
    weights = strat.markowitz(max_weight=0.2)

    entity_cols_strat = [c for c in weights.columns if c != "date"]
    avg_weights = {c: float(weights[c].mean()) for c in entity_cols_strat}
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
    return entity_cols_strat, strat, weights


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Performance vs Benchmark

    We compare against the STOXX 600 total return index from `benchmarks.parquet`.
    The file stores index *levels* $L_t$, so we first convert to monthly returns:

    $$r^b_t = \frac{L_t}{L_{t-1}} - 1$$

    Key metrics:

    | Metric | Definition |
    |--------|-----------|
    | Sharpe | $\bar{r}^p / \sigma^p \times \sqrt{12}$ (no risk-free rate) |
    | Alpha | $12 \times \overline{(r^p - r^b)}$ (annualised) |
    | Tracking Error | $\sigma(r^p - r^b) \times \sqrt{12}$ |
    | Information Ratio | $\alpha \;/\; \text{TE}$ |
    | Max Drawdown | $\min_t \left( \text{NAV}_t \;/\; \max_{s \leq t} \text{NAV}_s - 1 \right)$ |
    """)
    return


@app.cell
def _(entity_cols_strat, returns_df, weights):
    # Build per-date portfolio return series for the benchmark comparison and charts.
    # Strategy.sharpe returns the scalar; we still need the time series here.
    forward_rets = returns_df.select(
        "date",
        *[
            pl.col(c).shift(-1).alias(c)
            for c in entity_cols_strat
            if c in returns_df.columns
        ],
    ).filter(pl.col("date").is_in(weights["date"].implode()))

    weights_aligned = (
        weights
        .sort("date")
        .filter(pl.col("date").is_in(forward_rets["date"].implode()))
    )

    w_np = weights_aligned.select(entity_cols_strat).to_numpy()
    r_np = forward_rets.sort("date").select(entity_cols_strat).to_numpy()

    portfolio_returns_np = np.nansum(w_np * r_np, axis=1)
    port_dates = weights_aligned["date"].to_list()

    portfolio_returns_df = pl.DataFrame({
        "date": port_dates,
        "momentum": portfolio_returns_np,
    })
    return port_dates, portfolio_returns_df, portfolio_returns_np


@app.cell
def _(port_dates, portfolio_returns_np, strat):
    # Load benchmark levels, resample to month-end, then convert to monthly returns
    benchmark_raw = pl.read_parquet(CACHE_DIR / "benchmarks.parquet")
    benchmark_col = next(c for c in benchmark_raw.columns if c != "date")

    benchmark_all = (
        benchmark_raw
        .sort("date")
        .group_by_dynamic("date", every="1mo")
        .agg(pl.col(benchmark_col).last())
        .select(
            "date",
            (pl.col(benchmark_col) / pl.col(benchmark_col).shift(1) - 1)
            .alias(benchmark_col),
        )
        .drop_nulls()
    )

    # Aligned series: used for relative metrics (alpha, TE, IR) and cumulative chart.
    benchmark_aligned = benchmark_all.filter(pl.col("date").is_in(port_dates)).sort("date")
    bench_r_aligned = benchmark_aligned[benchmark_col].to_numpy()
    n       = min(len(portfolio_returns_np), len(bench_r_aligned))
    port_r  = portfolio_returns_np[:n]
    bench_r = bench_r_aligned[:n]

    # Full monthly series: for standalone benchmark stats (Sharpe, MDD).
    bench_r_full = benchmark_all[benchmark_col].to_numpy()

    def _sharpe(r: np.ndarray) -> float:
        return float(r.mean() / r.std() * np.sqrt(12))

    def _max_drawdown(r: np.ndarray) -> float:
        cum  = np.cumprod(1 + r)
        peak = np.maximum.accumulate(cum)
        return float(((cum - peak) / peak).min())

    sharpe_port  = strat.sharpe
    sharpe_bench = _sharpe(bench_r_full)
    excess       = port_r - bench_r
    alpha        = float(excess.mean() * 12)
    te           = float(excess.std() * np.sqrt(12))
    ir           = alpha / te
    mdd_port     = _max_drawdown(port_r)
    mdd_bench    = _max_drawdown(bench_r_full)

    mo.md(f"""
    | Metric | Momentum Strategy | STOXX 600 Benchmark |
    |--------|:-----------------:|:-------------------:|
    | Annualised Sharpe | `{sharpe_port:.3f}` | `{sharpe_bench:.3f}` |
    | Alpha (ann.) | `{alpha:.3f}` | — |
    | Tracking Error (ann.) | `{te:.3f}` | — |
    | Information Ratio | `{ir:.3f}` | — |
    | Max Drawdown | `{mdd_port:.2%}` | `{mdd_bench:.2%}` |

    > **Note on drawdown:** All drawdown figures are computed on month-end prices
    > and therefore understate the true intraday/intramonth peak-to-trough decline.
    > The STOXX 600 fell approximately 35% intramonth during the COVID crash (Feb–Mar 2020),
    > versus ~22% on a month-end basis. Daily data would give a more accurate picture.
    """)
    return bench_r, mdd_port, n, port_r, sharpe_port


@app.cell
def _(bench_r, n, port_dates, port_r):
    cum_port  = np.cumprod(1 + port_r)
    cum_bench = np.cumprod(1 + bench_r)

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=port_dates[:n],
        y=cum_port.tolist(),
        name="Momentum Strategy",
        line={"color": "royalblue"},
    ))
    fig_cum.add_trace(go.Scatter(
        x=port_dates[:n],
        y=cum_bench.tolist(),
        name="STOXX 600 Benchmark",
        line={"color": "tomato", "dash": "dash"},
    ))
    fig_cum.update_layout(
        title="Cumulative returns: momentum strategy vs STOXX 600",
        xaxis_title="Date",
        yaxis_title="Cumulative return (1 = starting value)",
    )
    fig_cum
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Analytical Tearsheet

    We use `jquantstats` for a richer set of risk-adjusted statistics and to
    inspect rolling performance (important for understanding whether returns
    are stable across regimes or concentrated in a single period).
    """)
    return


@app.cell
def _(portfolio_returns_df):
    jqs = Data.from_returns(returns=portfolio_returns_df, date_col="date")

    def _scalar(v) -> float:
        """Extract scalar value from jquantstats result.

        Returns dicts keyed by column name even for a single
        series, so we always extract the first value as a plain float.
        """
        if isinstance(v, dict):
            return float(next(iter(v.values())))
        return float(v)

    stats = {
        # --- Returns ---
        "CAGR":                  _scalar(jqs.stats.cagr()),
        "Avg Monthly Return":    _scalar(jqs.stats.avg_return()),
        "Best Month":            _scalar(jqs.stats.best()),
        "Worst Month":           _scalar(jqs.stats.worst()),
        "Win Rate":              _scalar(jqs.stats.win_rate()),
        # --- Risk ---
        "Volatility (ann.)":     _scalar(jqs.stats.volatility()),
        "Max Drawdown":          _scalar(jqs.stats.max_drawdown()),
        "VaR (95%, monthly)":    _scalar(jqs.stats.value_at_risk()),
        "ES (95%, monthly)":   _scalar(jqs.stats.conditional_value_at_risk()),
        "Skewness":              _scalar(jqs.stats.skew()),
        "Kurtosis":              _scalar(jqs.stats.kurtosis()),
        # --- Risk-adjusted ---
        "Sharpe (ann.)":         _scalar(jqs.stats.sharpe()),
        "Sortino (ann.)":        _scalar(jqs.stats.sortino()),
        "Calmar":                _scalar(jqs.stats.calmar()),
        # --- Trade stats ---
        "Avg Win":               _scalar(jqs.stats.avg_win()),
        "Avg Loss":              _scalar(jqs.stats.avg_loss()),
        "Payoff Ratio":          _scalar(jqs.stats.payoff_ratio()),
    }

    sections = {
        "Returns":        ["CAGR", "Avg Monthly Return", "Best Month", "Worst Month", "Win Rate"],
        "Risk":           ["Volatility (ann.)", "Max Drawdown", "VaR (95%, monthly)", "ES (95%, monthly)", "Skewness", "Kurtosis"],
        "Risk-adjusted":  ["Sharpe (ann.)", "Sortino (ann.)", "Calmar"],
        "Trade stats":    ["Avg Win", "Avg Loss", "Payoff Ratio"],
    }

    table = "| Section | Metric | Value |\n|---------|--------|-------|\n"
    for section, keys in sections.items():
        for k in keys:
            table += f"| {section} | {k} | `{stats[k]:.4f}` |\n"
        section = ""  # blank section label after first row

    mo.md(table)
    return jqs, stats


@app.cell
def _(jqs):
    jqs.plots.rolling_sharpe(rolling_period=12)
    return


@app.cell(hide_code=True)
def _(mdd_port, sharpe_port, stats):
    mo.md(f"""
    **Interpretation:**

    - **Sharpe of `{sharpe_port:.3f}`** — a Sharpe in the range 0.3-0.7 is consistent
      with academic momentum benchmarks on European large caps. This figure is gross
      of transaction costs; monthly rebalancing of ~130 stocks would incur meaningful
      turnover and erode net returns.

    - **Max drawdown of `{mdd_port:.2%}`** — momentum strategies are exposed to sharp
      crashes during market reversals (e.g. the COVID pandemic in 2020), when recent
      losers outperform recent winners.

    - **Calmar of `{stats['Calmar']:.3f}`** — return per unit of maximum drawdown.
      A Calmar above 1 indicates the strategy earns more than it risks in its worst
      drawdown; below 1 suggests the drawdown risk is not fully compensated.

    - **Rolling 12-month Sharpe** — periods where the rolling Sharpe turns negative
      correspond to momentum crashes. These tend to be brief but severe, which is why
      managing drawdown (e.g. via volatility scaling) matters for live deployment.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conclusions and Next Steps

    ### What worked

    - The 12-1 momentum signal shows a positive mean IC over the sample period,
      confirming the anomaly is present in the STOXX 100 universe.
    - The Markowitz portfolio with a 20% weight cap produces a diversified factor
      tilt without excessive concentration in a single sector or name.
    - The strategy achieves a Sharpe broadly comparable to the STOXX 600, with a
      small positive alpha — a reasonable starting point before any refinement.

    ### Limitations

    - **No transaction cost modelling.** Monthly rebalancing of ~130 positions
      generates meaningful turnover. Incorporating a turnover constraint into the
      optimiser, or extending the rebalancing frequency to quarterly, would bring
      simulated performance closer to what is achievable in practice.
    - **In-sample only.** No walk-forward or out-of-sample evaluation has been
      performed. The Sharpe and alpha figures should be treated as upper bounds.
    - **Momentum crash risk.** The strategy's max drawdown of -27% exceeds the
      benchmark's -23%, consistent with the well-known vulnerability of momentum
      to sharp market reversals. The strategy currently has no explicit defence
      against these episodes.
    - **Constant risk exposure.** Markowitz implicitly reduces weights when volatility rises, but never pins the portfolio to an explicit volatility target, aggregate portfolio vol still drifts through time, making the strategy harder to size consistently within a broader portfolio.

    ### Next steps

    1. **Size of the Universe.** Adjust size to incorporate the entire
       universe of 600 stocks. You could account for illiquid stocks by double
       sorting with respect to momentum and size and bound the part of small
       stocks to 50% of the portfolio.
    2. **GARCH volatility scaling.** Scale the portfolio's aggregate exposure by
       $k_t = \sigma^* / \hat{\sigma}_t$, where $\hat{\sigma}_t$ is the one-step-ahead
       conditional volatility from a rolling GARCH (or eGARCH) and $\sigma^*$ is the
       unconditional target. In prior work on US equities this materially improved the drawdown
       profile without sacrificing Sharpe.

    3. **Last-month return filter.** The skip-1 convention already avoids the
       short-term reversal, but stocks with very strong last-month returns can still
       enter the portfolio at inflated prices. Double-sorting by 12-1 momentum and
       penalising or excluding stocks whose most recent monthly return is in the top
       quintile of the cross-section can reduce this effect.

    4. **Sector neutrality.** The current 20% weight cap provides diversification at
       the stock level but not at the sector level — a single macro event can still dominate. Enforcing equal sector exposure
       via PCA-based orthogonalisation or explicit sector weight constraints would
       make the factor tilt cleaner and more replicable.

    5. **Lookback sensitivity.** Compare 12-1 vs 6-1 vs 3-1 lookbacks and report
       IC and Sharpe for each, to understand robustness to parameter choice.

    6. **Incorporate YUKKA sentiment data.** Use sentiment as a second signal
       alongside momentum, either independently (blended $\mu$ in the Markowitz
       objective) or as a filter (e.g. only hold momentum longs where sentiment is
       also positive).
    """)
    return


if __name__ == "__main__":
    app.run()
