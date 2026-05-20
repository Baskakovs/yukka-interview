"""Basic momentum signal for cross-sectional portfolio strategies. Serves as signal function for Strategy."""

from __future__ import annotations

import polars as pl

__all__ = ["Signal"]


class Signal:
    """Normalised cross-sectional momentum signal.

    Computes the return from t-lookback to t-skip, then normalises
    cross-sectionally at each date via rank z-scoring.

    Parameters
    ----------
    lookback:
        Number of periods over which to measure past performance.
    skip:
        Number of most recent periods to exclude (avoids short-term reversal).
    """

    def __init__(self, lookback: int = 12, skip: int = 1) -> None:
        """Initialise Signal with lookback and skip periods."""
        self._lookback = lookback
        self._skip = skip
        self._date_col = "date"

    def __call__(self, prices: pl.DataFrame) -> pl.DataFrame:
        """Compute signal from prices and return wide-format DataFrame.

        Parameters
        ----------
        prices:
            Wide-format DataFrame of monthly prices (date x stock).

        Returns:
        -------
        pl.DataFrame
            Wide-format DataFrame of normalised momentum scores (date x stock).
        """
        entity_cols = [c for c in prices.columns if c != self._date_col]

        raw = prices.select(
            self._date_col,
            *[
                (pl.col(c).shift(self._skip) / pl.col(c).shift(self._lookback) - 1).alias(c)
                for c in entity_cols
            ],
        )

        long = raw.unpivot(index=self._date_col, variable_name="stock", value_name="signal")

        normalised_long = long.with_columns(
            pl.col("signal").rank().over(self._date_col).alias("ranked")
        ).with_columns(
            (
                (pl.col("ranked") - pl.col("ranked").mean().over(self._date_col))
                / pl.col("ranked").std(ddof=1).over(self._date_col)
            ).alias("normalised")
        )

        return normalised_long.select(self._date_col, "stock", "normalised").pivot(
            on="stock", index=self._date_col, values="normalised"
        )
