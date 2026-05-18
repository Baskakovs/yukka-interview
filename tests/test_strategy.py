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


@pytest.fixture
def sparse_strategy() -> Strategy:
    # Sparse panel where not every asset has finite returns on the same dates.
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(6)]
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "B": [None, None, 50.0, 51.0, 52.0, 53.0],
            "C": [10.0, 10.1, None, None, None, None],
        },
    )
    signal = prices.select("date", pl.lit(1.0).alias("A"), pl.lit(0.5).alias("B"), pl.lit(0.0).alias("C"))
    return Strategy(prices=prices, signal=signal)


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

    def test_sparse_panel_drops_assets_with_insufficient_history(self, sparse_strategy: Strategy) -> None:
        # Sparse masked panels should still produce finite weights in the original asset order.
        w = sparse_strategy.build_portfolio(objective="min_variance", long_only=True)
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(np.isfinite(w))
        assert w[2] == 0.0

    def test_unknown_objective_raises(self, strategy: Strategy) -> None:
        # An unrecognised objective raises ValueError.
        with pytest.raises(ValueError, match="Unknown objective"):
            strategy.build_portfolio(objective="invalid")


class TestExpectedReturns:
    # Tests for the expected_returns parameter on build_portfolio.

    def test_historical_default_matches_explicit(self, strategy: Strategy) -> None:
        # expected_returns=None and "historical" produce identical weights.
        w_default = strategy.build_portfolio(objective="max_sharpe", long_only=True)
        w_explicit = strategy.build_portfolio(objective="max_sharpe", long_only=True, expected_returns="historical")
        np.testing.assert_allclose(w_default, w_explicit, atol=1e-6)

    def test_signal_produces_valid_weights(self, strategy: Strategy) -> None:
        # max_sharpe with expected_returns="signal" returns valid weights.
        w = strategy.build_portfolio(
            objective="max_sharpe",
            long_only=True,
            max_weight=0.5,
            expected_returns="signal",
        )
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(w >= -1e-6)
        assert np.all(w <= 0.5 + 1e-6)

    def test_array_produces_valid_weights(self, strategy: Strategy) -> None:
        # max_sharpe with an explicit mu array returns valid weights.
        cols = strategy._asset_cols()
        mu = np.linspace(0.01, 0.05, len(cols))
        w = strategy.build_portfolio(objective="max_sharpe", long_only=True, expected_returns=mu)
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(w >= -1e-6)

    def test_min_variance_ignores_expected_returns(self, strategy: Strategy) -> None:
        # min_variance result is unchanged by expected_returns parameter.
        w_plain = strategy.build_portfolio(objective="min_variance", long_only=True)
        w_with_mu = strategy.build_portfolio(objective="min_variance", long_only=True, expected_returns="signal")
        np.testing.assert_allclose(w_plain, w_with_mu, atol=1e-6)

    def test_nan_signal_entries_excluded(self) -> None:
        # Assets with NaN signal at the latest cross-section get zero weight.
        n_dates = 20
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rng = np.random.default_rng(7)
        prices = {"date": dates}
        signal: dict = {"date": dates}
        for name in ["A", "B", "C"]:
            prices[name] = (100.0 * np.exp(np.cumsum(rng.normal(0.01, 0.02, n_dates)))).tolist()
            signal[name] = rng.normal(0, 1, n_dates).tolist()
        # Force C's latest signal to None — should be masked out.
        signal["C"][-1] = None
        s = Strategy(prices=pl.DataFrame(prices), signal=pl.DataFrame(signal))
        w = s.build_portfolio(objective="max_sharpe", long_only=True, expected_returns="signal")
        assert abs(w.sum() - 1.0) < 1e-4
        assert w[2] == 0.0

    def test_trailing_empty_signal_row_uses_latest_finite_row(self) -> None:
        # Trailing all-null signal rows can appear when the latest price month is partial.
        n_dates = 20
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rng = np.random.default_rng(17)
        prices = {"date": dates}
        signal: dict = {"date": dates}
        for name in ["A", "B", "C"]:
            prices[name] = (100.0 * np.exp(np.cumsum(rng.normal(0.01, 0.02, n_dates)))).tolist()
            signal[name] = rng.normal(0, 1, n_dates).tolist()
            signal[name][-1] = None
        s = Strategy(prices=pl.DataFrame(prices), signal=pl.DataFrame(signal))
        w = s.build_portfolio(objective="max_sharpe", long_only=True, expected_returns="signal")
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(np.isfinite(w))

    def test_invalid_string_raises(self, strategy: Strategy) -> None:
        # An unrecognised expected_returns string raises ValueError.
        with pytest.raises(ValueError, match="Unknown expected_returns"):
            strategy.build_portfolio(objective="max_sharpe", expected_returns="bogus")

    def test_wrong_shape_array_raises(self, strategy: Strategy) -> None:
        # A NumPy array of the wrong shape raises ValueError.
        with pytest.raises(ValueError, match="expected_returns array must have shape"):
            strategy.build_portfolio(objective="max_sharpe", expected_returns=np.array([1.0, 2.0]))


class TestSharpeRatio:
    # Tests for Strategy.sharpe_ratio.

    def test_returns_float_min_variance(self, strategy: Strategy) -> None:
        # sharpe_ratio returns a float for min_variance.
        assert isinstance(strategy.sharpe_ratio(objective="min_variance"), float)

    def test_returns_float_max_sharpe(self, strategy: Strategy) -> None:
        # sharpe_ratio returns a float for max_sharpe.
        assert isinstance(strategy.sharpe_ratio(objective="max_sharpe"), float)

    def test_sparse_panel_returns_finite_min_variance(self, sparse_strategy: Strategy) -> None:
        # Sparse forward returns should not become all-NaN through nan * zero weights.
        sharpe = sparse_strategy.sharpe_ratio(objective="min_variance", long_only=True)
        assert np.isfinite(sharpe)
