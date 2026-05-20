"""Portfolio strategy construction using mean-variance optimisation."""

from __future__ import annotations

import datetime
from collections.abc import Callable

import cvxpy as cp
import numpy as np
import polars as pl

ANNUALISATION_FACTOR = np.sqrt(12)


class Strategy:
    """Mean-variance portfolio strategy driven by a user-defined signal.

    Constructs a portfolio using a user-defined signal and Markowitz
    mean-variance optimisation. Exposes mean IC, optimal weights,
    and annualised Sharpe ratio.

    Parameters
    ----------
    prices:
        Wide-format DataFrame of monthly prices (date x stock).
    signal_fn:
        Callable that takes prices and returns a signal DataFrame
        of the same shape (predicted returns).
    risk_free_rate:
        Monthly risk-free rate for Sharpe calculation. Defaults to 0.
    """

    def __init__(
        self,
        prices: pl.DataFrame,
        signal_fn: Callable[[pl.DataFrame], pl.DataFrame],
        risk_free_rate: float = 0.0,
    ) -> None:
        """Initialise Strategy with prices, signal function, and risk-free rate."""
        self._prices = prices
        self._date_col = "date"
        self._entity_cols = [c for c in prices.columns if c != self._date_col]
        self._returns = prices.select(
            self._date_col,
            *[(pl.col(c) / pl.col(c).shift(1) - 1).alias(c) for c in self._entity_cols],
        )
        self._signal = signal_fn(prices)
        self._risk_free_rate = risk_free_rate
        self._weights: pl.DataFrame | None = None
        self._valid_cols: np.ndarray | None = None

    @property
    def mean_ic(self) -> float:
        """Mean Pearson IC between signal and 1-period forward return.

        Computed cross-sectionally at each date and averaged over time.
        """
        signal_long = self._signal.unpivot(
            index=self._date_col, variable_name="stock", value_name="signal"
        )
        returns_long = self._returns.unpivot(
            index=self._date_col, variable_name="stock", value_name="forward_return"
        )
        return float(
            signal_long
            .join(returns_long, on=[self._date_col, "stock"], how="inner")
            .filter(pl.col("signal").is_not_null() & pl.col("forward_return").is_not_null())
            .group_by(self._date_col)
            .agg(pl.corr("signal", "forward_return", method="pearson").alias("ic"))
            ["ic"].mean()
        )

    def _get_sigma(
        self,
        date: datetime.date,
        window: int,
        regularisation: float,
    ) -> np.ndarray | None:
        """Estimate covariance matrix from a rolling window of returns.

        Also sets self._valid_cols — the boolean mask of stocks with
        sufficient data — for use in _get_mu.
        """
        returns_window = (
            self._returns
            .filter(pl.col(self._date_col) <= date)
            .tail(window)
            .select(self._entity_cols)
            .to_numpy()
            .astype(float)
        )
        if returns_window.shape[0] < 2:
            return None

        self._valid_cols = ~np.all(np.isnan(returns_window), axis=0)
        returns_window = returns_window[:, self._valid_cols]

        col_means = np.nanmean(returns_window, axis=0)
        nan_mask = np.isnan(returns_window)
        returns_window[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        sigma = np.cov(returns_window.T)
        sigma = (sigma + sigma.T) / 2
        return sigma + np.eye(returns_window.shape[1]) * regularisation

    def _get_mu(self, date: datetime.date) -> np.ndarray | None:
        """Extract and filter signal at date t to valid stocks only.

        Requires _get_sigma to have been called first for this date,
        as it depends on self._valid_cols.
        """
        row = self._signal.filter(pl.col(self._date_col) == date)
        if row.is_empty():
            return None
        mu = np.array([row[c][0] for c in self._entity_cols], dtype=float)
        if np.all(np.isnan(mu)):
            return None
        mu = mu[self._valid_cols]
        return np.where(np.isnan(mu), 0.0, mu)

    @staticmethod
    def _solve_mean_variance(
        mu: np.ndarray,
        sigma: np.ndarray,
        delta: float,
        max_weight: float,
    ) -> np.ndarray | None:
        """Solve the mean-variance optimisation problem.

        Maximises mu^T w - delta * w^T sigma w subject to budget,
        long-only, and max weight constraints.
        """
        w = cp.Variable(len(mu))
        prob = cp.Problem(
            cp.Maximize(mu @ w - delta * cp.quad_form(w, sigma)),
            [cp.sum(w) == 1, w >= 0, w <= max_weight],
        )
        prob.solve(solver=cp.CLARABEL)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return None
        return w.value

    def markowitz(
        self,
        delta: float = 1.0,
        window: int = 60,
        max_weight: float = 0.2,
        regularisation: float = 1e-8,
    ) -> pl.DataFrame:
        """Run rolling Markowitz optimisation and return portfolio weights.

        Parameters
        ----------
        delta:
            Risk aversion parameter. Higher = more conservative.
        window:
            Rolling window in months for covariance estimation.
        max_weight:
            Maximum weight per stock (diversification constraint).
        regularisation:
            Ridge regularisation added to covariance diagonal for
            numerical stability.
        """
        dates = self._prices.sort(self._date_col)[self._date_col].to_list()
        all_weights = []

        for date in dates:
            sigma = self._get_sigma(date, window, regularisation)
            if sigma is None:
                continue

            mu = self._get_mu(date)
            if mu is None:
                continue

            weights_valid = self._solve_mean_variance(mu, sigma, delta, max_weight)
            if weights_valid is None:
                continue

            weights_full = np.zeros(len(self._entity_cols))
            weights_full[self._valid_cols] = weights_valid
            weights_full = np.clip(weights_full, 0, max_weight)
            weights_full /= weights_full.sum()

            all_weights.append(
                {self._date_col: date, **dict(zip(self._entity_cols, weights_full, strict=True))}
            )

        self._weights = pl.DataFrame(all_weights)
        return self._weights

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe ratio of the strategy.

        Calls markowitz() automatically if weights have not been computed.
        Assumes monthly data and annualises by sqrt(12).
        """
        if self._weights is None:
            self.markowitz()

        weights_matrix = self._weights.sort(self._date_col).select(self._entity_cols).to_numpy()

        returns_shifted = (
            self._returns
            .sort(self._date_col)
            .select(
                self._date_col,
                *[pl.col(c).shift(-1).alias(c) for c in self._entity_cols],
            )
            .filter(pl.col(self._date_col).is_in(self._weights[self._date_col].implode()))
            .select(self._entity_cols)
            .to_numpy()
        )

        portfolio_returns = np.nansum(weights_matrix * returns_shifted, axis=1)
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        return float((mean_return - self._risk_free_rate) / std_return * ANNUALISATION_FACTOR)
