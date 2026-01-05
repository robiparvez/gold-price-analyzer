# 🥇 Gold Price Analyzer

A comprehensive, professional-grade gold price analysis platform built with machine learning, featuring real-time price tracking, advanced forecasting, investment portfolio management, jewelry pricing calculator, and comprehensive backtesting capabilities.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-0.9+-ffa94d?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-0.1.0-blue?style=for-the-badge)

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Functional Architecture](#️-functional-architecture)
- [Installation Guide](#-installation-guide)
- [Usage Examples](#-usage-examples)
- [Configuration Options](#️-configuration-options)
- [API Documentation](#-api-documentation)
- [Contribution Guidelines](#-contribution-guidelines)

## 🎯 Project Overview

The Gold Price Analyzer is a sophisticated financial analysis platform designed for Bangladesh's gold market. It combines machine learning forecasting, real-time data collection, investment tracking, and professional reporting to provide comprehensive insights for investors, jewelers, and financial analysts.

**Core Purpose**: Enable data-driven decision making in gold investment and jewelry pricing through advanced analytics, ML-powered forecasting, and comprehensive market analysis.

**Target Audience**:

- Individual gold investors
- Jewelry retailers and manufacturers
- Financial analysts
- Investment advisors
- Market researchers

## 🔥 Key Features

### 🤖 Machine Learning & Forecasting

- **Multi-Model Ensemble**: Prophet, Random Forest, XGBoost with weighted blending
- **7-Day Price Predictions**: Short-term forecasting with confidence intervals
- **Hyperparameter Optimization**: Optuna-powered model tuning
- **Feature Importance Analysis**: Understand what drives price movements

### 📊 Real-Time Data & Analytics

- **Live Price Scraping**: Automated collection from BAJUS (Bangladesh Jewellers Association)
- **Historical Data Generation**: Synthetic data with realistic market trends
- **External Data Integration**: USD/BDT exchange rates and global spot prices
- **Interactive Charts**: Candlestick charts, trend analysis, and technical indicators

### 💰 Investment Management

- **Portfolio Tracking**: Buy/sell transaction logging with P&L calculations
- **Investment Goals**: Target-based savings tracking
- **Performance Analytics**: ROI, volatility, and risk metrics
- **Asset Allocation**: Multi-purity gold portfolio management

### 💍 Jewelry Pricing Calculator

- **VAT Integration**: Automatic 5% VAT calculation
- **Making Charges**: Type-specific pricing (rings, necklaces, bracelets, etc.)
- **Multi-Purity Support**: 18K, 21K, 22K gold calculations
- **Price Breakdown**: Transparent cost structure display

### 🧪 Professional Backtesting

- **Rolling Window Validation**: Historical model performance testing
- **Time Series Split**: Chronological validation methods
- **Error Metrics**: MAE, RMSE, R² score analysis
- **Model Comparison**: Side-by-side performance evaluation

### 📄 Reporting & Export

- **PDF Report Generation**: Professional market analysis reports
- **Interactive Dashboards**: 8-tab comprehensive interface
- **Data Export**: CSV and database backups
- **Visual Analytics**: Plotly and Altair chart integration

## 🏗️ Functional Architecture

### Core Components

```text
Gold Price Analyzer
├── 🎨 User Interface (app.py)
│   ├── Current Analysis Dashboard
│   ├── 7-Day ML Forecasting
│   ├── Historical Trends & Charts
│   ├── Model Performance Metrics
│   ├── Custom Prediction Tool
│   ├── Investment Portfolio Tracker
│   ├── Jewelry Pricing Calculator
│   └── Backtesting Suite
│
├── 🤖 ML Engine (analyzer.py)
│   ├── Prophet Time Series Model
│   ├── Random Forest Regressor
│   ├── XGBoost with Optuna Tuning
│   └── Ensemble Model Blending
│
├── 📡 Data Pipeline
│   ├── Live Scraper (scraper.py)
│   │   └── BAJUS Price Collection
│   ├── Historical Generator (historical_scraper.py)
│   │   └── Synthetic Market Data
│   └── External Feeds (external_data_fetcher.py)
│       └── FX Rates & Global Spot
│
├── 🗄️ Data Storage (database_schema.py)
│   ├── DuckDB High-Performance Database
│   ├── Prices Table (Live Data)
│   ├── Historical_Prices Table (Time Series)
│   ├── External_Data Table (Market Feeds)
│   ├── Investment_Tracking Table (Transactions)
│   ├── Investment_Goals Table (Targets)
│   └── Portfolio Table (Holdings)
│
├── 🧮 Business Logic
│   ├── Jewelry Calculator (jewelry_pricing.py)
│   ├── Backtesting Engine (backtesting.py)
│   ├── PDF Reports (pdf_report_generator.py)
│   └── Utilities (utils.py)
│
└── 🔧 Infrastructure
    ├── Logging System (logging_config.py)
    ├── Configuration Management
    └── Development Tools (pytest, black, mypy)
```

### Data Flow Architecture

```text
External Sources → Scrapers → DuckDB → ML Models → Analysis → UI → Reports
                      ↓
               Validation → Backtesting → Optimization
```

## 🚀 Installation Guide

### Prerequisites

- **Python**: 3.10 or higher
- **UV Package Manager**: Modern Python package installer
- **Git**: Version control system

### Quick Start Installation

```bash
# Clone the repository
git clone https://github.com/robiparvez/gold-price-analyzer.git
cd gold-price-analyzer

# Install dependencies using UV (recommended)
uv sync

# Alternative: Install with pip
pip install -r requirements.txt
```

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Run code quality checks
uv run black .
uv run isort .
uv run mypy .
uv run ruff check .

# Run tests
uv run pytest
```

### First-Time Setup

```bash
# Initialize database and scrape initial data
uv run python scraper.py
uv run python historical_scraper.py

# Launch the application
uv run streamlit run app.py
```

## 💻 Usage Examples

### Basic Price Analysis

```python
from analyzer import AdvancedGoldPriceAnalyzer

# Initialize analyzer
analyzer = AdvancedGoldPriceAnalyzer()

# Generate 7-day forecast for 22K gold
forecast = analyzer.generate_forecast(days=7, purity="22K")
print(f"Predicted price: {forecast['prediction']} BDT/gram")
```

### Jewelry Price Calculation

```python
from jewelry_pricing import JewelryPricingCalculator

# Initialize calculator
calc = JewelryPricingCalculator()

# Calculate price for 10g 22K necklace
price = calc.calculate_jewelry_price(
    weight_grams=10,
    purity="22K",
    item_type="necklace",
    base_price_per_gram=8500
)

print(f"Total price: {price.total_price} BDT")
print(f"Making charges: {price.making_charges} BDT")
print(f"VAT (5%): {price.vat_amount} BDT")
```

### Backtesting Models

```python
from backtesting import GoldPriceBacktester

# Initialize backtester
backtester = GoldPriceBacktester()

# Run rolling window backtest
results = backtester.rolling_window_backtest(
    df=historical_data,
    window_size=30,
    forecast_horizon=7,
    model_type="ensemble"
)

print(f"MAE: {results['mae']:.2f}")
print(f"RMSE: {results['rmse']:.2f}")
print(f"R² Score: {results['r2']:.3f}")
```

### Database Operations

```python
from database_schema import DatabaseSchema

# Initialize database
db = DatabaseSchema()

# Get latest prices
prices = db.get_latest_prices(purity="22K", limit=10)

# Record investment transaction
db.record_investment(
    purity="22K",
    quantity_grams=50,
    price_per_gram=8500,
    transaction_type="buy"
)
```

## ⚙️ Configuration Options

### Environment Variables

```bash
# Database Configuration
GOLD_DB_PATH=data/gold_prices.db

# Scraping Configuration
SCRAPER_TIMEOUT=30
BAJUS_BASE_URL=https://www.bajus.org/gold-price

# ML Model Configuration
MODEL_CACHE_DIR=models/
DEFAULT_FORECAST_DAYS=7

# External Data Configuration
EXTERNAL_DATA_API_KEY=your_api_key_here
FX_UPDATE_INTERVAL=3600

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### Configuration Files

#### `pyproject.toml` - Project Configuration

```toml
[project]
name = "gold-price-analyzer"
version = "0.1.0"
dependencies = [
    "streamlit==1.50.0",
    "duckdb>=0.9.0",
    "pandas>=2.1.0",
    # ... other dependencies
]

[tool.black]
line-length = 88
target-version = ['py310']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

### Runtime Configuration

```python
# Custom analyzer configuration
analyzer = AdvancedGoldPriceAnalyzer(
    data_dir="custom_data/",
    models_dir="custom_models/"
)

# Custom jewelry calculator
calculator = JewelryPricingCalculator(
    vat_rate=0.05,  # 5% VAT
    custom_making_charges={
        "ring": {"22K": 800, "21K": 750}
    }
)
```

## 📚 API Documentation

### Core Classes

#### `AdvancedGoldPriceAnalyzer`

**Main ML analysis engine supporting multiple forecasting models.**

```python
class AdvancedGoldPriceAnalyzer:
    def __init__(self, data_dir: str = "data", models_dir: str = "models")

    def generate_forecast(self, days: int = 7, purity: str = "22K") -> dict
    def train_prophet_model(self, df: pd.DataFrame) -> dict
    def train_random_forest_model(self, df: pd.DataFrame) -> dict
    def evaluate_model_accuracy(self, purity: str = "22K", test_days: int = 30) -> dict
```

#### `JewelryPricingCalculator`

**Professional jewelry pricing with VAT and making charges.**

```python
class JewelryPricingCalculator:
    def __init__(self, vat_rate: float = 0.05, custom_making_charges: dict = None)

    def calculate_jewelry_price(self, weight_grams: float, purity: str,
                               item_type: str, base_price_per_gram: float) -> JewelryPrice
    def get_making_charge(self, item_type: str, purity: str) -> float
```

#### `GoldPriceBacktester`

**Comprehensive model validation and performance testing.**

```python
class GoldPriceBacktester:
    def __init__(self)

    def rolling_window_backtest(self, df: pd.DataFrame, window_size: int = 30,
                               forecast_horizon: int = 7, model_type: str = "ensemble") -> dict
    def time_series_split_backtest(self, df: pd.DataFrame, n_splits: int = 5) -> dict
```

#### `DatabaseSchema`

**DuckDB database management for all application data.**

```python
class DatabaseSchema:
    def __init__(self, db_path: str = "data/gold_prices.db")

    def create_all_tables(self) -> bool
    def get_latest_prices(self, purity: str = "22K", limit: int = 100) -> pd.DataFrame
    def record_investment(self, purity: str, quantity_grams: float,
                         price_per_gram: float, transaction_type: str) -> bool
```

### Data Models

#### `JewelryPrice` (dataclass)

```python
@dataclass
class JewelryPrice:
    base_price: float      # Base gold cost
    vat_amount: float      # 5% VAT amount
    making_charges: float  # Jeweler's charges
    total_price: float     # Final price
    weight_grams: float    # Item weight
    purity: str           # Gold purity (18K/21K/22K)
    item_type: str        # Jewelry type
```

### Utility Functions

#### Price Formatting

```python
def format_price_bdt(price: float, decimals: int = 2) -> str
# Format prices in Bangladeshi Taka with proper separators

def format_price_bdt_short(price: float) -> str
# Compact formatting for charts (K/M notation)
```

## 🤝 Contribution Guidelines

### Development Workflow

1. **Fork the Repository**

   ```bash
   git clone https://github.com/your-username/gold-price-analyzer.git
   cd gold-price-analyzer
   ```

2. **Set Up Development Environment**

   ```bash
   uv sync --dev
   ```

3. **Create Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Code Quality Standards**

   ```bash
   # Run all quality checks
   uv run black .
   uv run isort .
   uv run mypy .
   uv run pytest
   ```

5. **Testing Requirements**
   - Unit tests for new features
   - Integration tests for data pipelines
   - Performance tests for ML models

### Code Style Guidelines

- **Black**: Automatic code formatting (88 character line length)
- **isort**: Import sorting and organization
- **MyPy**: Type hints required for all functions
- **Docstrings**: Google-style docstrings for all public methods

### Pull Request Process

1. **Update Documentation**: Ensure README and docstrings are updated
2. **Add Tests**: Include comprehensive test coverage
3. **Update Changelog**: Document changes in CHANGELOG.md
4. **Squash Commits**: Clean commit history before merging

### Areas for Contribution

- **ML Model Improvements**: New algorithms, better feature engineering
- **Data Sources**: Additional price feeds, alternative data
- **UI Enhancements**: Better visualizations, mobile responsiveness
- **API Development**: REST API for external integrations
- **Internationalization**: Multi-language support, additional currencies

### Testing Strategy

```bash
# Run full test suite
uv run pytest --cov=src --cov-report=html

# Run specific test categories
uv run pytest tests/test_ml_models.py
uv run pytest tests/test_data_pipeline.py

# Performance testing
uv run pytest tests/test_backtesting.py --durations=10
```

### Documentation Standards

- **README.md**: Keep updated with new features
- **Code Comments**: Explain complex algorithms
- **API Docs**: Document all public interfaces
- **Examples**: Provide usage examples for new features

---

Built with ❤️ for Bangladesh's gold market | MIT License | Python 3.10+
