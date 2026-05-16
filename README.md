# Yukka Interview Take-Home

Portfolio construction and analysis take-home for summer intern candidates.

## Setup

1. **Fork** this repository on GitHub.
2. **Clone** your fork:

   ```bash
   git clone git@github.com:<your-username>/yukka-interview.git
   cd yukka-interview
   ```

3. **Install dependencies** (requires [uv](https://docs.astral.sh/uv/)):

   ```bash
   uv sync --all-extras --all-groups
   ```

4. **Open the notebook**:

   ```bash
   make marimo
   ```

## Project Structure

```text
yukka-interview/
  src/interview/
    __init__.py             # Package init (loads .env)
    data/
      config.py             # Cache directory path
      repository.py         # Repository ABC and Asset dataclass
      returns.py            # Returns class with preprocessing
      yukka_repository.py   # Concrete repository (prices, returns)
      cache/                # Pre-computed parquet files (committed)
        prices_all.parquet
        ranks_wide.parquet
        benchmarks.parquet
  book/marimo/notebooks/
    Experiment1.py          # Your working notebook
  tests/
    test_repository.py      # Data layer tests
    test_returns.py         # Returns class tests
```

## Data Layer

The `YukkaRepository` class provides access to STOXX 600 price data:

```python
from interview.data import YukkaRepository
from yukka.data import Index

repo = YukkaRepository(index=Index.STOXX600)
assets = repo.assets                          # list of Asset objects
prices = repo.prices(assets, mask=True)       # wide DataFrame: date + one column per asset
returns = repo.returns(assets)                # same shape, simple returns
```

All data is cached locally in parquet files -- no API key is needed.

## Submitting Your Work

1. Commit your changes to your fork.
2. Push to GitHub.
3. Share the link to your fork with us.

## Contact

If you have any questions, reach out to Peter at [peter@yukkalab.com](mailto:peter@yukkalab.com).
