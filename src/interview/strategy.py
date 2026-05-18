"""Portfolio strategy: IC computation and Markowitz optimisation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import cvxpy as cp
import numpy as np
import polars as pl
from cvxpy.constraints.constraint import Constraint
from scipy.stats import spearmanr

_OPTIMAL_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


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
        # Return column names excluding the date column.
        return [c for c in self.prices.columns if c != "date"]

    def _forward_returns(self) -> pl.DataFrame:
        # Return fwd_ret[t] = price[t+1] / price[t] - 1 (last row is null).
        cols = self._asset_cols()
        return self.prices.select(
            "date",
            *[(pl.col(c).shift(-1) / pl.col(c) - 1).alias(c) for c in cols],
        )

    def _historical_returns(self) -> pl.DataFrame:
        # Return ret[t] = price[t] / price[t-1] - 1 with the first (null) row dropped.
        cols = self._asset_cols()
        return self.prices.select(
            "date",
            *[(pl.col(c) / pl.col(c).shift(1) - 1).alias(c) for c in cols],
        ).slice(1)

    def _return_estimates(self, min_observations: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate mean returns and covariance from a sparse returns panel.

        Returns:
            Tuple of ``(active_mask, mu, sigma)``. ``active_mask`` maps the
            estimated assets back to the original asset column order.

        Raises:
            ValueError: If there is not enough finite return history.
        """
        cols = self._asset_cols()
        ret_np = self._historical_returns().select(cols).to_numpy().astype(float)
        finite = np.isfinite(ret_np)
        active_mask = finite.sum(axis=0) >= min_observations
        if not active_mask.any():
            raise ValueError("Not enough finite return history.")  # noqa: TRY003

        active_ret = ret_np[:, active_mask]
        row_mask = np.any(np.isfinite(active_ret), axis=1)
        active_ret = active_ret[row_mask]
        if active_ret.shape[0] < min_observations:
            raise ValueError("Not enough finite return history.")  # noqa: TRY003

        mu = np.nanmean(active_ret, axis=0)
        filled_ret = np.where(np.isfinite(active_ret), active_ret, mu)
        if filled_ret.shape[1] == 1:
            sigma = np.array([[np.var(filled_ret[:, 0], ddof=1)]])
        else:
            sigma = np.cov(filled_ret, rowvar=False)
        sigma = np.atleast_2d(sigma)
        sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = (sigma + sigma.T) / 2
        sigma = sigma + 1e-6 * np.eye(active_mask.sum())

        if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
            raise ValueError("Return estimates are not finite.")  # noqa: TRY003
        return active_mask, mu, sigma

    def _portfolio_forward_returns(self, weights: np.ndarray) -> np.ndarray:
        """Return NaN-aware one-period-forward portfolio returns."""
        fwd_np = self._forward_returns().select(self._asset_cols()).to_numpy().astype(float)
        finite = np.isfinite(fwd_np)
        weighted_returns = np.where(finite, fwd_np * weights, 0.0).sum(axis=1)
        invested = finite & (np.abs(weights) > 0)
        weighted_returns[~invested.any(axis=1)] = np.nan
        return weighted_returns

    @property
    def mean_ic(self) -> float:  # noqa: D102
        # Mean cross-sectional Spearman IC between signal and forward returns.
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
            ics.append(float(np.asarray(ic).item()))
        return float(np.mean(ics)) if ics else float("nan")

    def build_portfolio(
        self,
        objective: str = "min_variance",
        long_only: bool = True,
        max_weight: float | None = None,
    ) -> np.ndarray:
        """Solve Markowitz optimisation and return asset weights.

        Args:
            objective: ``"min_variance"`` or ``"max_sharpe"``.
            long_only: If ``True``, constrain all weights to be non-negative.
            max_weight: Optional per-asset upper bound on the final weights.

        Returns:
            1-D NumPy array of weights summing to 1.

        Raises:
            ValueError: If *objective* is not recognised.
        """
        cols = self._asset_cols()
        n = len(cols)
        active_mask, mu, sigma = self._return_estimates()
        active_n = int(active_mask.sum())

        if max_weight is not None and max_weight <= 0:
            raise ValueError("max_weight must be positive.")  # noqa: TRY003

        def _expand(active_weights: np.ndarray) -> np.ndarray:
            weights = np.zeros(n)
            weights[active_mask] = active_weights
            return weights

        if objective == "min_variance":
            w = cp.Variable(active_n)
            constraints: list[Constraint] = [cast(Constraint, cp.sum(w) == 1)]
            if long_only:
                constraints.append(w >= 0)
            if max_weight is not None:
                constraints.append(w <= max_weight)
            problem = cp.Problem(cp.Minimize(cp.quad_form(w, sigma)), constraints)
            problem.solve()
            if problem.status not in _OPTIMAL_STATUSES or w.value is None:
                raise RuntimeError("Optimisation failed.")  # noqa: TRY003
            return _expand(np.asarray(w.value, dtype=float))

        if objective == "max_sharpe":
            # Charnes-Cooper transformation: mu.T @ y = 1, minimise y.T @ sigma @ y.
            y = cp.Variable(active_n)
            constraints: list[Constraint] = [cast(Constraint, mu @ y == 1)]
            if long_only:
                constraints.append(y >= 0)
            if max_weight is not None:
                constraints.append(y <= max_weight * cp.sum(y))
            problem = cp.Problem(cp.Minimize(cp.quad_form(y, sigma)), constraints)
            problem.solve()
            if problem.status not in _OPTIMAL_STATUSES or y.value is None:
                raise RuntimeError("Optimisation failed.")  # noqa: TRY003
            y_val = np.asarray(y.value, dtype=float)
            return _expand(y_val / y_val.sum())

        raise ValueError(f"Unknown objective: {objective!r}")  # noqa: TRY003

    def sharpe_ratio(
        self,
        objective: str = "min_variance",
        long_only: bool = True,
        max_weight: float | None = None,
    ) -> float:
        """Annualised Sharpe ratio of the optimised portfolio.

        Args:
            objective: Forwarded to :meth:`build_portfolio`.
            long_only: Forwarded to :meth:`build_portfolio`.
            max_weight: Forwarded to :meth:`build_portfolio`.

        Returns:
            Annualised Sharpe ratio scaled by ``sqrt(12)`` for monthly data.
        """
        weights = self.build_portfolio(objective=objective, long_only=long_only, max_weight=max_weight)
        port_ret = self._portfolio_forward_returns(weights)
        port_ret = port_ret[np.isfinite(port_ret)]
        if len(port_ret) < 2:
            return float("nan")
        return float(port_ret.mean() / port_ret.std(ddof=1) * math.sqrt(12))
