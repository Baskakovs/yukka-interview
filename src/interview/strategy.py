"""Portfolio strategy: IC computation and Markowitz optimisation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import cvxpy as cp
import numpy as np
import polars as pl
from cvxpy.constraints.constraint import Constraint
from scipy.stats import rankdata, spearmanr

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

    def _estimate_from_return_matrix(
        self,
        ret_np: np.ndarray,
        min_observations: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate mean returns and covariance from a sparse NumPy return matrix.

        Returns:
            Tuple of ``(active_mask, mu, sigma)``. ``active_mask`` maps the
            estimated assets back to the input matrix column order.

        Raises:
            ValueError: If there is not enough finite return history.
        """
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

    def _return_estimates(self, min_observations: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate mean returns and covariance from the full returns panel."""
        cols = self._asset_cols()
        ret_np = self._historical_returns().select(cols).to_numpy().astype(float)
        return self._estimate_from_return_matrix(ret_np=ret_np, min_observations=min_observations)

    def _portfolio_forward_returns(self, weights: np.ndarray) -> np.ndarray:
        """Return NaN-aware one-period-forward portfolio returns."""
        fwd_np = self._forward_returns().select(self._asset_cols()).to_numpy().astype(float)
        finite = np.isfinite(fwd_np)
        available_weights = np.where(finite, weights, 0.0)
        exposure = available_weights.sum(axis=1)
        weighted_returns = np.full(fwd_np.shape[0], np.nan)
        has_exposure = np.abs(exposure) > 1e-12
        period_returns = np.where(finite, fwd_np * weights, 0.0).sum(axis=1)
        weighted_returns[has_exposure] = period_returns[has_exposure] / exposure[has_exposure]
        return weighted_returns

    def _latest_signal(self) -> np.ndarray:
        """Return the most recent finite cross-section of the signal.

        Trailing months can have no signal if the underlying prices are partial
        or fully masked. Those rows are skipped. Null entries in the selected
        row are mapped to ``NaN`` so callers can mask them out.

        Raises:
            ValueError: If the signal has no finite entries in any row.
        """
        cols = self._asset_cols()
        for row_idx in range(self.signal.height - 1, -1, -1):
            row = self.signal.row(row_idx, named=True)
            signal_values = np.array(
                [row[c] if row[c] is not None else np.nan for c in cols],
                dtype=float,
            )
            if np.isfinite(signal_values).any():
                return signal_values
        raise ValueError("Signal has no finite entries.")  # noqa: TRY003

    def _signal_to_expected_returns(self, signal_values: np.ndarray) -> np.ndarray:
        """Map a raw signal cross-section to positive rank scores."""
        finite = np.isfinite(signal_values)
        expected_returns = np.full_like(signal_values, np.nan, dtype=float)
        if finite.any():
            expected_returns[finite] = rankdata(signal_values[finite], method="average") / finite.sum()
        return expected_returns

    def _restrict_expected_returns(
        self,
        mu_full: np.ndarray,
        active_mask: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Restrict expected returns and covariance to finite active assets."""
        mu_active = mu_full[active_mask]
        finite = np.isfinite(mu_active)
        if not finite.any():
            raise ValueError("No assets have finite expected returns.")  # noqa: TRY003
        if finite.all():
            return active_mask, mu_active, sigma

        active_indices = np.where(active_mask)[0]
        new_active_mask = np.zeros_like(active_mask)
        new_active_mask[active_indices[finite]] = True
        keep_idx = np.where(finite)[0]
        sigma = sigma[np.ix_(keep_idx, keep_idx)]
        return new_active_mask, mu_active[finite], sigma

    def _apply_expected_returns(
        self,
        expected_returns: str | np.ndarray,
        active_mask: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve *expected_returns* and tighten the active universe to finite entries.

        Returns:
            Tuple of ``(active_mask, mu, sigma)`` restricted to assets with finite
            expected returns. The returned ``active_mask`` may be a strict subset
            of the input.
        """
        n = len(self._asset_cols())
        if isinstance(expected_returns, str):
            if expected_returns == "historical":
                # No override; recover the historical mu via _return_estimates.
                _, mu_hist, _ = self._return_estimates()
                return active_mask, mu_hist, sigma
            if expected_returns == "signal":
                mu_full = self._latest_signal()
            elif expected_returns == "signal_rank":
                mu_full = self._signal_to_expected_returns(self._latest_signal())
            else:
                raise ValueError(  # noqa: TRY003
                    f"Unknown expected_returns: {expected_returns!r}. "
                    "Use 'historical', 'signal', 'signal_rank', or a NumPy array."
                )
        elif isinstance(expected_returns, np.ndarray):
            if expected_returns.shape != (n,):
                raise ValueError(  # noqa: TRY003
                    f"expected_returns array must have shape ({n},), got {expected_returns.shape}."
                )
            mu_full = expected_returns.astype(float)
        else:
            kind = type(expected_returns).__name__
            raise TypeError(  # noqa: TRY003
                f"expected_returns must be 'historical', 'signal', 'signal_rank', or a NumPy array; got {kind}."
            )

        return self._restrict_expected_returns(mu_full=mu_full, active_mask=active_mask, sigma=sigma)

    def _solve_portfolio(
        self,
        active_mask: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        objective: str,
        long_only: bool,
        max_weight: float | None,
    ) -> np.ndarray:
        """Solve the Markowitz problem for a pre-estimated active universe."""
        n = len(self._asset_cols())
        active_n = int(active_mask.sum())
        if active_n == 0:
            raise ValueError("No active assets remain.")  # noqa: TRY003
        if max_weight is not None and max_weight <= 0:
            raise ValueError("max_weight must be positive.")  # noqa: TRY003
        if max_weight is not None and active_n * max_weight < 1.0 - 1e-10:
            raise ValueError("max_weight is too low for the active universe.")  # noqa: TRY003

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
            if long_only and np.nanmax(mu) <= 0:
                raise ValueError("Long-only max_sharpe requires at least one positive expected return.")  # noqa: TRY003
            # Charnes-Cooper transformation: mu.T @ y = 1, minimise y.T @ sigma @ y.
            y = cp.Variable(active_n)
            constraints = [cast(Constraint, mu @ y == 1)]
            if long_only:
                constraints.append(y >= 0)
            if max_weight is not None:
                constraints.append(y <= max_weight * cp.sum(y))
            problem = cp.Problem(cp.Minimize(cp.quad_form(y, sigma)), constraints)
            problem.solve()
            if problem.status not in _OPTIMAL_STATUSES or y.value is None:
                raise RuntimeError("Optimisation failed.")  # noqa: TRY003
            y_val = np.asarray(y.value, dtype=float)
            if abs(y_val.sum()) <= 1e-12:
                raise RuntimeError("Optimisation returned zero-sum weights.")  # noqa: TRY003
            return _expand(y_val / y_val.sum())

        raise ValueError(f"Unknown objective: {objective!r}")  # noqa: TRY003

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
            if np.unique(sig_vals[mask]).size < 2 or np.unique(fwd_vals[mask]).size < 2:
                continue
            ic, _ = spearmanr(sig_vals[mask], fwd_vals[mask])
            ic_value = float(np.asarray(ic).item())
            if np.isfinite(ic_value):
                ics.append(ic_value)
        return float(np.mean(ics)) if ics else float("nan")

    def build_portfolio(
        self,
        objective: str = "min_variance",
        long_only: bool = True,
        max_weight: float | None = None,
        expected_returns: str | np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve Markowitz optimisation and return asset weights.

        Args:
            objective: ``"min_variance"`` or ``"max_sharpe"``.
            long_only: If ``True``, constrain all weights to be non-negative.
            max_weight: Optional per-asset upper bound on the final weights.
            expected_returns: Override for the expected-return vector consumed
                by ``"max_sharpe"`` (ignored for ``"min_variance"``).

                - ``None`` / ``"historical"`` (default): historical sample mean
                  of per-asset returns.
                - ``"signal"``: latest cross-section of ``self.signal``.
                - ``"signal_rank"``: positive ranks of the latest signal
                  cross-section.
                - 1-D ``np.ndarray`` of length ``len(asset_cols)``: used as-is.

                Assets with NaN expected returns are excluded from the
                optimisation universe.

        Returns:
            1-D NumPy array of weights summing to 1, in original asset order.

        Raises:
            ValueError: If *objective* or *expected_returns* is invalid, or if
                no active assets remain after masking.
        """
        active_mask, mu, sigma = self._return_estimates()

        if objective == "max_sharpe" and expected_returns is not None:
            active_mask, mu, sigma = self._apply_expected_returns(
                expected_returns,
                active_mask=active_mask,
                sigma=sigma,
            )

        return self._solve_portfolio(
            active_mask=active_mask,
            mu=mu,
            sigma=sigma,
            objective=objective,
            long_only=long_only,
            max_weight=max_weight,
        )

    def backtest_rebalanced(
        self,
        objective: str = "max_sharpe",
        long_only: bool = True,
        max_weight: float | None = None,
        expected_returns: str = "signal_rank",
        lookback: int = 36,
        min_observations: int = 12,
        transaction_cost_bps: float = 0.0,
        return_col: str = "strategy",
    ) -> pl.DataFrame:
        """Backtest a monthly rebalanced strategy without look-ahead.

        At each date ``t``, the method estimates risk from returns observed up
        to ``t``, uses the signal cross-section available at ``t`` as the
        expected-return proxy, solves the requested portfolio, and applies those
        weights to returns from ``t`` to ``t+1``.

        Args:
            objective: ``"min_variance"`` or ``"max_sharpe"``.
            long_only: If ``True``, constrain all weights to be non-negative.
            max_weight: Optional per-asset upper bound.
            expected_returns: ``"historical"``, ``"signal"``, or
                ``"signal_rank"``. ``"signal_rank"`` is the default because it
                keeps the expected-return proxy positive for long-only
                max-Sharpe optimisation.
            lookback: Number of trailing periods used to estimate covariance.
            min_observations: Minimum finite trailing returns required per asset.
            transaction_cost_bps: One-way transaction cost in basis points,
                charged as ``turnover * transaction_cost_bps / 10_000``.
            return_col: Name of the strategy return column.

        Returns:
            DataFrame with net returns and rebalance diagnostics.
        """
        if lookback < min_observations:
            raise ValueError("lookback must be >= min_observations.")  # noqa: TRY003
        if expected_returns not in {"historical", "signal", "signal_rank"}:
            raise ValueError(f"Unknown expected_returns: {expected_returns!r}")  # noqa: TRY003
        if transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be non-negative.")  # noqa: TRY003

        cols = self._asset_cols()
        dates = self.prices["date"].to_list()
        hist_ret = self._historical_returns().select(cols).to_numpy().astype(float)
        fwd_ret = self._forward_returns().select(cols).to_numpy().astype(float)
        signal_np = self.signal.select(cols).to_numpy().astype(float)
        cost_rate = transaction_cost_bps / 10_000.0
        previous_weights = np.zeros(len(cols))

        out_dates = []
        out_returns = []
        out_gross_returns = []
        out_costs = []
        out_turnover = []
        out_active_assets = []
        for row_idx in range(1, len(dates) - 1):
            window_start = max(0, row_idx - lookback)
            trailing_ret = hist_ret[window_start:row_idx]
            try:
                active_mask, mu, sigma = self._estimate_from_return_matrix(
                    ret_np=trailing_ret,
                    min_observations=min_observations,
                )
                if objective == "max_sharpe" and expected_returns != "historical":
                    if expected_returns == "signal":
                        mu_full = signal_np[row_idx]
                    else:
                        mu_full = self._signal_to_expected_returns(signal_np[row_idx])
                    active_mask, mu, sigma = self._restrict_expected_returns(
                        mu_full=mu_full,
                        active_mask=active_mask,
                        sigma=sigma,
                    )

                weights = self._solve_portfolio(
                    active_mask=active_mask,
                    mu=mu,
                    sigma=sigma,
                    objective=objective,
                    long_only=long_only,
                    max_weight=max_weight,
                )
            except (RuntimeError, ValueError):
                continue

            period_ret = fwd_ret[row_idx]
            tradable = np.isfinite(period_ret) & (np.abs(weights) > 1e-12)
            if not tradable.any():
                continue
            realised_weights = np.where(tradable, weights, 0.0)
            weight_sum = realised_weights.sum()
            if abs(weight_sum) <= 1e-12:
                continue
            realised_weights = realised_weights / weight_sum
            turnover = float(np.abs(realised_weights - previous_weights).sum())
            transaction_cost = turnover * cost_rate
            gross_return = float(period_ret[tradable] @ realised_weights[tradable])
            out_dates.append(dates[row_idx + 1])
            out_returns.append(gross_return - transaction_cost)
            out_gross_returns.append(gross_return)
            out_costs.append(transaction_cost)
            out_turnover.append(turnover)
            out_active_assets.append(int(tradable.sum()))
            previous_weights = realised_weights

        return pl.DataFrame(
            {
                "date": out_dates,
                return_col: out_returns,
                f"{return_col}_gross": out_gross_returns,
                "transaction_cost": out_costs,
                "turnover": out_turnover,
                "active_assets": out_active_assets,
            }
        )

    def sharpe_ratio(
        self,
        objective: str = "min_variance",
        long_only: bool = True,
        max_weight: float | None = None,
        expected_returns: str | np.ndarray | None = None,
    ) -> float:
        """Annualised Sharpe ratio of the optimised portfolio.

        Args:
            objective: Forwarded to :meth:`build_portfolio`.
            long_only: Forwarded to :meth:`build_portfolio`.
            max_weight: Forwarded to :meth:`build_portfolio`.
            expected_returns: Forwarded to :meth:`build_portfolio`.

        Returns:
            Annualised Sharpe ratio scaled by ``sqrt(12)`` for monthly data.
        """
        weights = self.build_portfolio(
            objective=objective,
            long_only=long_only,
            max_weight=max_weight,
            expected_returns=expected_returns,
        )
        port_ret = self._portfolio_forward_returns(weights)
        port_ret = port_ret[np.isfinite(port_ret)]
        if len(port_ret) < 2:
            return float("nan")
        return float(port_ret.mean() / port_ret.std(ddof=1) * math.sqrt(12))
