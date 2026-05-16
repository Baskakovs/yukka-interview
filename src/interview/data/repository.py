"""Abstract repository class and asset descriptor."""

from __future__ import annotations

import contextlib
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl

# Importing yukka.data triggers Index enum initialization which prints to stdout; suppress it.
with contextlib.redirect_stdout(io.StringIO()):
    from yukka.data import Asset


@dataclass
class Repository(ABC):
    """Abstract repository defining the data access interface."""

    @property
    @abstractmethod
    def assets(self) -> list[Asset]:
        """List available assets with metadata."""
        ...

    @abstractmethod
    def prices(self, assets: list[Asset] | None = None, **kwargs) -> pl.DataFrame:
        """Load raw price data for the given assets."""
        ...

    @abstractmethod
    def returns(self, assets: list[Asset] | None = None, **kwargs) -> pl.DataFrame:
        """Compute returns from price data for the given assets."""
        ...
