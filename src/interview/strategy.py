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
            else:
                raise ValueError(  # noqa: TRY003
                    f"Unknown expected_returns: {expected_returns!r}. Use 'historical', 'signal', or a NumPy array."
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
                f"expected_returns must be 'historical', 'signal', or a NumPy array; got {kind}."
            )

        mu_active = mu_full[active_mask]
        finite = np.isfinite(mu_active)
        if not finite.any():
            raise ValueError("No assets have finite expected returns.")  # noqa: TRY003
        if finite.all():
            return active_mask, mu_active, sigma

        # Tighten active_mask to drop assets with non-finite expected returns.
        active_indices = np.where(active_mask)[0]
        new_active_mask = np.zeros_like(active_mask)
        new_active_mask[active_indices[finite]] = True
        keep_idx = np.where(finite)[0]
        sigma = sigma[np.ix_(keep_idx, keep_idx)]
        return new_active_mask, mu_active[finite], sigma

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
                - 1-D ``np.ndarray`` of length ``len(asset_cols)``: used as-is.

                Assets with NaN expected returns are excluded from the
                optimisation universe.

        Returns:
            1-D NumPy array of weights summing to 1, in original asset order.

        Raises:
            ValueError: If *objective* or *expected_returns* is invalid, or if
                no active assets remain after masking.
        """
        cols = self._asset_cols()
        n = len(cols)
        active_mask, mu, sigma = self._return_estimates()

        if max_weight is not None and max_weight <= 0:
            raise ValueError("max_weight must be positive.")  # noqa: TRY003

        if objective == "max_sharpe" and expected_returns is not None:
            active_mask, mu, sigma = self._apply_expected_returns(
                expected_returns,
                active_mask=active_mask,
                sigma=sigma,
            )

        active_n = int(active_mask.sum())

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
