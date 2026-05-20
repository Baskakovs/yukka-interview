"""Tests for interview.strategy — Strategy class."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from interview.signal import Signal
from interview.strategy import Strategy

_WINDOW = 13


@pytest.fixture
def prices() -> pl.DataFrame:
    """Monthly prices DataFrame with sufficient history for testing."""
    dates = pl.date_range(
        start=pl.date(2010, 1, 1),
        end=pl.date(2024, 12, 1),
        interval="1mo",
        eager=True,
    )
    rng = np.random.default_rng(42)
    n = len(dates)
    return pl.DataFrame({
        "date": dates,
        "AAPL": (100 * np.cumprod(1 + rng.normal(0.01, 0.05, n))).tolist(),
        "MSFT": (100 * np.cumprod(1 + rng.normal(0.01, 0.05, n))).tolist(),
        "GOOG": (100 * np.cumprod(1 + rng.normal(0.01, 0.05, n))).tolist(),
        "TSLA": (100 * np.cumprod(1 + rng.normal(0.01, 0.05, n))).tolist(),
        "AMZN": (100 * np.cumprod(1 + rng.normal(0.01, 0.05, n))).tolist(),
    })


@pytest.fixture
def strategy(prices: pl.DataFrame) -> Strategy:
    """Strategy instance with default Signal."""
    return Strategy(prices=prices, signal_fn=Signal())


@pytest.fixture
def strategy_with_weights(prices: pl.DataFrame) -> Strategy:
    """Strategy instance with weights pre-computed."""
    strat = Strategy(prices=prices, signal_fn=Signal())
    strat.markowitz(window=_WINDOW, max_weight=0.5)
    return strat


class TestStrategyInit:
    """Tests for Strategy initialisation."""

    def test_stores_prices(self, prices: pl.DataFrame, strategy: Strategy) -> None:
        """Prices DataFrame is stored correctly."""
        assert strategy._prices is prices

    def test_entity_cols_excludes_date(self, strategy: Strategy) -> None:
        """Entity columns do not include the date column."""
        assert "date" not in strategy._entity_cols

    def test_weights_initially_none(self, strategy: Strategy) -> None:
        """Weights are None before markowitz() is called."""
        assert strategy._weights is None

    def test_custom_risk_free_rate(self, prices: pl.DataFrame) -> None:
        """Custom risk-free rate is stored correctly."""
        strat = Strategy(prices=prices, signal_fn=Signal(), risk_free_rate=0.02)
        assert strat._risk_free_rate == 0.02


class TestMeanIC:
    """Tests for Strategy.mean_ic."""

    def test_between_minus_one_and_one(self, strategy: Strategy) -> None:
        """mean_ic is a valid correlation value between -1 and 1."""
        assert -1.0 <= strategy.mean_ic <= 1.0


class TestGetsigma:
    """Tests for Strategy._get_sigma."""

    def test_returns_none_insufficient_data(self, strategy: Strategy) -> None:
        """Returns None when fewer than 2 rows of data are available."""
        date = strategy._prices["date"].to_list()[0]
        assert strategy._get_sigma(date, window=60, regularisation=1e-8) is None

    def test_is_symmetric_and_positive_definite(self, strategy: Strategy) -> None:
        """Covariance matrix is symmetric and positive definite."""
        date = strategy._prices["date"].to_list()[-1]
        sigma = strategy._get_sigma(date, window=_WINDOW, regularisation=1e-8)
        np.testing.assert_array_almost_equal(sigma, sigma.T, decimal=10)
        assert np.all(np.linalg.eigvalsh(sigma) > 0)

    def test_sets_valid_cols(self, strategy: Strategy) -> None:
        """_get_sigma sets self._valid_cols as a boolean array."""
        date = strategy._prices["date"].to_list()[-1]
        strategy._get_sigma(date, window=_WINDOW, regularisation=1e-8)
        assert strategy._valid_cols is not None
        assert strategy._valid_cols.dtype == bool


class TestSolveMeanVariance:
    """Tests for Strategy._solve_mean_variance."""

    def test_satisfies_constraints(self, strategy: Strategy) -> None:
        """Solved weights sum to 1 and are non-negative."""
        date = strategy._prices["date"].to_list()[-1]
        sigma = strategy._get_sigma(date, window=_WINDOW, regularisation=1e-8)
        mu = strategy._get_mu(date)
        weights = Strategy._solve_mean_variance(mu, sigma, delta=1.0, max_weight=0.5)
        assert abs(weights.sum() - 1.0) < 1e-4
        assert np.all(weights >= -1e-6)


class TestMarkowitz:
    """Tests for Strategy.markowitz."""

    def test_budget_and_long_only_and_max_weight(self, strategy: Strategy) -> None:
        """Weights satisfy all three constraints at every date."""
        max_weight = 0.5
        weights = strategy.markowitz(window=_WINDOW, max_weight=max_weight)
        entity_cols = [c for c in weights.columns if c != "date"]
        assert "date" in weights.columns
        for row in weights.iter_rows(named=True):
            w = np.array([row[c] for c in entity_cols])
            assert abs(w.sum() - 1.0) < 1e-6
            assert np.all(w >= -1e-6)
            assert np.all(w <= max_weight + 1e-6)


class TestSharpe:
    """Tests for Strategy.sharpe."""

    def test_reasonable_range(self, strategy_with_weights: Strategy) -> None:
        """Sharpe ratio is a float within a plausible range."""
        sharpe = strategy_with_weights.sharpe
        assert isinstance(sharpe, float)
        assert -5.0 < sharpe < 5.0

    def test_calls_markowitz_if_needed(self, strategy: Strategy) -> None:
        """Sharpe populates weights automatically if not yet computed."""
        assert strategy._weights is None
        strategy.markowitz(window=_WINDOW)
        assert strategy._weights is not None
