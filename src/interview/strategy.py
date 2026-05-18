"""Portfolio strategy: IC computation and Markowitz optimisation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import polars as pl
from scipy.stats import spearmanr


@dataclass
class Strategy:
    """Portfolio strategy combining signal IC and Markowitz optimisation.

    Args:
        prices: Wide-format DataFrame with a ``date`` column and one column per asset.
        signal: Same shape as ``prices``; values are the raw alpha signal.
    """

    prices: pl.DataFrame
    signal: pl.DataFrame

    def _asset_cols(self) -> list[str]:
        """Return column names excluding the date column."""
        return [c for c in self.prices.columns if c != "date"]

    def _forward_returns(self) -> pl.DataFrame:
        """Return fwd_ret[t] = price[t+1] / price[t] - 1 (last row is null)."""
        cols = self._asset_cols()
        return self.prices.select(
            "date",
            *[(pl.col(c).shift(-1) / pl.col(c) - 1).alias(c) for c in cols],
        )

    def _historical_returns(self) -> pl.DataFrame:
        """Return ret[t] = price[t] / price[t-1] - 1 with the first (null) row dropped."""
        cols = self._asset_cols()
        return self.prices.select(
            "date",
            *[(pl.col(c) / pl.col(c).shift(1) - 1).alias(c) for c in cols],
        ).slice(1)

    @property
    def mean_ic(self) -> float:
        """Mean cross-sectional Spearman IC between signal and forward returns."""
        fwd = self._forward_returns()
        cols = self._asset_cols()
        ics: list[float] = []
        for sig_row, fwd_row in zip(self.signal.iter_rows(named=True), fwd.iter_rows(named=True), strict=True):
            sig_vals = np.array([sig_row[c] for c in cols], dtype=float)
            fwd_vals = np.array([fwd_row[c] for c in cols], dtype=float)
            mask = np.isfinite(sig_vals) & np.isfinite(fwd_vals)
            if mask.sum() < 2:
                continue
            ic, _ = spearmanr(sig_vals[mask], fwd_vals[mask])
            ics.append(float(ic))
        return float(np.mean(ics)) if ics else float("nan")

    def build_portfolio(
        self,
        objective: str = "min_variance",
        long_only: bool = True,
    ) -> np.ndarray:
        """Solve Markowitz optimisation and return asset weights.

        Args:
            objective: ``"min_variance"`` or ``"max_sharpe"``.
            long_only: If ``True``, constrain all weights to be non-negative.

        Returns:
            1-D NumPy array of weights summing to 1.

        Raises:
            ValueError: If *objective* is not recognised.
        """
        cols = self._asset_cols()
        n = len(cols)
        ret_np = self._historical_returns().select(cols).to_numpy().astype(float)
        valid = np.all(np.isfinite(ret_np), axis=1)
        ret_np = ret_np[valid]
        mu = ret_np.mean(axis=0)
        sigma = np.cov(ret_np, rowvar=False) + 1e-6 * np.eye(n)

        if objective == "min_variance":
            w = cp.Variable(n)
            constraints = [cp.sum(w) == 1]
            if long_only:
                constraints.append(w >= 0)
            cp.Problem(cp.Minimize(cp.quad_form(w, sigma)), constraints).solve()
            return w.value

        if objective == "max_sharpe":
            # Charnes-Cooper transformation: fix μᵀy = 1, minimise yᵀΣy, then normalise to sum(w) = 1.
            y = cp.Variable(n)
            constraints = [mu @ y == 1]
            if long_only:
                constraints.append(y >= 0)
            cp.Problem(cp.Minimize(cp.quad_form(y, sigma)), constraints).solve()
            y_val = y.value
            return y_val / y_val.sum()

        raise ValueError(f"Unknown objective: {objective!r}")  # noqa: TRY003

    def sharpe_ratio(
        self,
        objective: str = "min_variance",
        long_only: bool = True,
    ) -> float:
        """Annualised Sharpe ratio of the optimised portfolio.

        Args:
            objective: Forwarded to :meth:`build_portfolio`.
            long_only: Forwarded to :meth:`build_portfolio`.

        Returns:
            Annualised Sharpe ratio scaled by ``sqrt(12)`` for monthly data.
        """
        weights = self.build_portfolio(objective=objective, long_only=long_only)
        fwd_np = self._forward_returns().select(self._asset_cols()).to_numpy().astype(float)
        port_ret = fwd_np @ weights
        port_ret = port_ret[np.isfinite(port_ret)]
        if len(port_ret) < 2:
            return float("nan")
        return float(port_ret.mean() / port_ret.std(ddof=1) * math.sqrt(12))
