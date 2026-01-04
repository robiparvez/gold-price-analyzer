
# 🥇 Gold Price Analyzer

Professional Streamlit application for gold price forecasting, investment tracking, and portfolio management. Powered by DuckDB storage, Prophet/RandomForest/XGBoost ML models, and comprehensive backtesting.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-ffa94d?style=for-the-badge)
![UV](https://img.shields.io/badge/UV-Package%20Manager-8A2BE2?style=for-the-badge)

## 🔥 Key Features

- **ML-Powered Forecasting**: Prophet + RandomForest + XGBoost ensemble with Optuna hyperparameter tuning
- **Investment Tracker**: Complete portfolio management with transactions, goals, and P/L tracking
- **Jewelry Pricing Calculator**: VAT and making charges calculator for all jewelry types
- **Comprehensive Backtesting**: Rolling window and time series split validation across all models
- **Golden/Death Cross Signals**: MA crossover detection for trading signals
- **DuckDB Storage**: High-performance unified database for all data types
- **Historical Data Sync**: Automated Sakib.dev integration with gap detection
- **External Data Integration**: USD/BDT FX rates and global gold spot prices
- **Professional UI**: 8-tab Streamlit interface with candlestick charts and PDF export

## 🏗️ Architecture

1) **Scraping**: `scraper.py` pulls BAJUS live prices → saves CSV + DuckDB.
2) **Historical sync**: `historical_scraper.py` pulls/patches history (Sakib.dev) → DuckDB + CSV backup.
3) **External feeds**: `external_data_fetcher.py` ingests FX + spot gold.
4) **Analysis/ML**: `analyzer.py` trains Prophet/RF/XGBoost, ensembles, evaluates.
5) **UI**: `app.py` renders comprehensive analytics with 8 tabs including investment tracking, jewelry pricing, and backtesting.

## 🚀 Quick Start

```bash
# install uv if needed
pip install uv

# install deps
uv sync

# run Streamlit app
uv run streamlit run app.py
# open http://localhost:8501
```

### Common commands

```bash
# scrape latest BAJUS prices
uv run python scraper.py

# run historical sync (gap-aware)
uv run python historical_scraper.py

# run tests
uv run pytest
```

## ⚙️ Configuration

- `data/gold_prices.db` — DuckDB database (prices, historical_prices, external_data, investment_tracking, investment_goals, portfolio).
- `.env` (optional) — override defaults such as timeouts or cache TTLs.

## 📂 Project Layout

```text
ml-gold-price-analyzer/
├── app.py                    # Primary Streamlit application with 8-tab interface
├── analyzer.py               # Advanced ML + ensemble logic
├── scraper.py                # BAJUS live scraper → DuckDB/CSV
├── historical_scraper.py     # Hybrid sync + gap fill → DuckDB/CSV
├── external_data_fetcher.py  # FX + global spot gold
├── database_schema.py        # DuckDB schema + helpers
├── hyperparameter_tuner.py   # Optuna + XGBoost tuning
├── data/                     # DuckDB file + CSV exports
├── logs/                     # Log outputs
├── docs/                     # Additional guides
└── pyproject.toml            # UV/dep config
```

## 🧠 Models

- **Prophet**: seasonality-aware baseline.
- **Random Forest**: tree-based regression for short horizons.
- **XGBoost**: tuned via Optuna; participates in ensemble.
- **Ensemble**: weighted blend of Prophet + RF (+ XGB where available).

## 🗄️ Database (DuckDB)

Tables created via `database_schema.py`:

- `prices` — live BAJUS prices.
- `historical_prices` — Sakib.dev history with gap repairs.
- `external_data` — FX + spot gold snapshots.
- `investment_tracking` — buy/sell transactions.
- `investment_goals` — targets and status.
- `portfolio` — aggregated holdings and P/L.

## 🛠️ Development

```bash
uv run black .
uv run isort .
uv run mypy .
uv run pytest
```

## 🔄 Migration Notes

- Modern architecture using `analyzer.py` for ML models and DuckDB for efficient data storage.
- Existing SQLite data should be exported to CSV then imported into `data/gold_prices.db` if needed.

## 📜 License

MIT License. See `LICENSE`.

## 🙋 Support

Built with ❤️ using Python, Streamlit, and UV
