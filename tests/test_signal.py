"""Tests for interview.signal — Signal class."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from interview.signal import Signal


@pytest.fixture
def prices() -> pl.DataFrame:
    """Small monthly prices DataFrame for testing."""
    dates = pl.date_range(
        start=pl.date(2016, 1, 1),
        end=pl.date(2021, 12, 1),
        interval="1mo",
        eager=True,
    )
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "date": dates,
        "AAPL": (100 * np.cumprod(1 + rng.normal(0.01, 0.05, len(dates)))).tolist(),
        "MSFT": (100 * np.cumprod(1 + rng.normal(0.01, 0.05, len(dates)))).tolist(),
        "GOOG": (100 * np.cumprod(1 + rng.normal(0.01, 0.05, len(dates)))).tolist(),
    })


@pytest.fixture
def signal() -> Signal:
    """Default Signal instance."""
    return Signal()


class TestSignalInit:
    """Tests for Signal initialisation."""

    def test_default_lookback(self, signal: Signal) -> None:
        """Default lookback is 12."""
        assert signal._lookback == 12

    def test_default_skip(self, signal: Signal) -> None:
        """Default skip is 1."""
        assert signal._skip == 1

    def test_custom_lookback(self) -> None:
        """Custom lookback is stored correctly."""
        s = Signal(lookback=6)
        assert s._lookback == 6

    def test_custom_skip(self) -> None:
        """Custom skip is stored correctly."""
        s = Signal(skip=2)
        assert s._skip == 2


class TestSignalCall:
    """Tests for Signal.__call__."""

    def test_returns_dataframe(self, signal: Signal, prices: pl.DataFrame) -> None:
        """Signal returns a Polars DataFrame."""
        result = signal(prices)
        assert isinstance(result, pl.DataFrame)

    def test_same_shape_as_prices(self, signal: Signal, prices: pl.DataFrame) -> None:
        """Output has same shape as input prices."""
        result = signal(prices)
        assert result.shape == prices.shape

    def test_date_column_preserved(self, signal: Signal, prices: pl.DataFrame) -> None:
        """Output contains a date column."""
        result = signal(prices)
        assert "date" in result.columns

    def test_burn_in_period_is_null(self, signal: Signal, prices: pl.DataFrame) -> None:
        """First lookback rows are null due to burn-in period."""
        result = signal(prices)
        entity_cols = [c for c in result.columns if c != "date"]
        for i in range(signal._lookback):
            row_values = [result[c][i] for c in entity_cols]
            assert all(v is None for v in row_values), f"Row {i} should be null"

    def test_non_null_after_burn_in(self, signal: Signal, prices: pl.DataFrame) -> None:
        """Rows after burn-in period contain non-null values."""
        result = signal(prices)
        entity_cols = [c for c in result.columns if c != "date"]
        last_row = {c: result[c][-1] for c in entity_cols}
        assert any(v is not None for v in last_row.values())

    def test_cross_sectional_mean_near_zero(self, signal: Signal, prices: pl.DataFrame) -> None:
        """Normalised signal has approximately zero cross-sectional mean at each date."""
        result = signal(prices)
        entity_cols = [c for c in result.columns if c != "date"]
        for row in result.iter_rows(named=True):
            values = [row[c] for c in entity_cols if row[c] is not None]
            if len(values) > 1:
                assert abs(np.mean(values)) < 1e-6, f"Cross-sectional mean not zero: {np.mean(values)}"

    def test_longer_lookback_increases_burn_in(self, prices: pl.DataFrame) -> None:
        """A longer lookback period increases the number of null rows."""
        s6 = Signal(lookback=6)
        s12 = Signal(lookback=12)
        result6 = s6(prices)
        result12 = s12(prices)
        entity_cols = [c for c in result6.columns if c != "date"]
        nulls6 = sum(1 for i in range(len(result6)) if all(result6[c][i] is None for c in entity_cols))
        nulls12 = sum(1 for i in range(len(result12)) if all(result12[c][i] is None for c in entity_cols))
        assert nulls12 > nulls6
