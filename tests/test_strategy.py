"""Tests for interview.strategy — Strategy class."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from interview.strategy import Strategy


@pytest.fixture
def strategy() -> Strategy:
    # Strategy backed by small deterministic price and signal data.
    n_dates = 30
    assets = ["A", "B", "C", "D"]
    rng = np.random.default_rng(42)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    price_data: dict = {"date": dates}
    signal_data: dict = {"date": dates}
    for asset in assets:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.01, 0.02, n_dates)))
        price_data[asset] = prices.tolist()
        signal_data[asset] = rng.normal(0, 1, n_dates).tolist()
    return Strategy(prices=pl.DataFrame(price_data), signal=pl.DataFrame(signal_data))


class TestMeanIC:
    # Tests for Strategy.mean_ic.

    def test_returns_float(self, strategy: Strategy) -> None:
        # mean_ic returns a Python float.
        assert isinstance(strategy.mean_ic, float)

    def test_in_range(self, strategy: Strategy) -> None:
        # mean_ic is a valid Spearman correlation, bounded in [-1, 1].
        assert -1.0 <= strategy.mean_ic <= 1.0


class TestBuildPortfolio:
    # Tests for Strategy.build_portfolio.

    def test_weights_sum_to_one_min_variance(self, strategy: Strategy) -> None:
        # min_variance weights sum to 1 within tolerance.
        w = strategy.build_portfolio(objective="min_variance")
        assert abs(w.sum() - 1.0) < 1e-4

    def test_weights_sum_to_one_max_sharpe(self, strategy: Strategy) -> None:
        # max_sharpe weights sum to 1 within tolerance.
        w = strategy.build_portfolio(objective="max_sharpe")
        assert abs(w.sum() - 1.0) < 1e-4

    def test_long_only_min_variance(self, strategy: Strategy) -> None:
        # long_only=True gives non-negative weights for min_variance.
        w = strategy.build_portfolio(objective="min_variance", long_only=True)
        assert np.all(w >= -1e-6)

    def test_long_only_max_sharpe(self, strategy: Strategy) -> None:
        # long_only=True gives non-negative weights for max_sharpe.
        w = strategy.build_portfolio(objective="max_sharpe", long_only=True)
        assert np.all(w >= -1e-6)

    def test_unknown_objective_raises(self, strategy: Strategy) -> None:
        # An unrecognised objective raises ValueError.
        with pytest.raises(ValueError, match="Unknown objective"):
            strategy.build_portfolio(objective="invalid")


class TestSharpeRatio:
    # Tests for Strategy.sharpe_ratio.

    def test_returns_float_min_variance(self, strategy: Strategy) -> None:
        # sharpe_ratio returns a float for min_variance.
        assert isinstance(strategy.sharpe_ratio(objective="min_variance"), float)

    def test_returns_float_max_sharpe(self, strategy: Strategy) -> None:
        # sharpe_ratio returns a float for max_sharpe.
        assert isinstance(strategy.sharpe_ratio(objective="max_sharpe"), float)
