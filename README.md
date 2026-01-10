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

#### **Advanced 9-Model Ensemble System**

- **Classical Models (2)**: ARIMA, ETS with automatic parameter tuning
- **ML Enhanced Models (3)**: LightGBM, CatBoost, SVR with feature engineering
- **Deep Learning Models (3)**: LSTM, GRU, TCN with sequence modeling
- **Hybrid Model (1)**: LSTM-ARIMA combining neural networks with classical statistics

#### **Intelligent Ensemble Methods**

- **Weighted Mean**: Variance-based dynamic weighting for optimal accuracy
- **Equal Weight**: Simple averaging for robustness
- **Median**: Outlier-resistant aggregation
- **Optuna-Optimized**: Bayesian hyperparameter optimization (30-50 trials)

#### **Forecasting Capabilities**

- **Flexible Horizon**: 1-30 day forecasting with configurable window
- **Confidence Intervals**: 95% prediction bounds with ensemble uncertainty
- **Model Comparison**: Side-by-side performance metrics (RMSE, MAE, R²)
- **Caching System**: SQLite-based forecast caching for instant retrieval
- **Legacy Support**: Prophet + Random Forest fallback option

#### **Service Architecture**

- **GoldPriceService**: Unified API wrapping orchestrator and optimizer
- **ModelOrchestrator**: Manages all 9 models with auto-selection
- **EnsembleOptimizer**: Optuna-powered weight optimization
- **Performance Tracking**: Cache hit rates, error monitoring, request metrics

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
│   ├── Advanced 9-Model Forecasting
│   ├── Historical Trends & Charts
│   ├── Model Performance Metrics
│   ├── Custom Prediction Tool
│   ├── Investment Portfolio Tracker
│   ├── Jewelry Pricing Calculator
│   └── Backtesting Suite
│
├── 🤖 ML Engine (9-Model Ensemble)
│   ├── Classical Models
│   │   ├── ARIMA (Auto-ARIMA)
│   │   └── ETS (Error-Trend-Seasonal)
│   ├── ML Enhanced Models
│   │   ├── LightGBM
│   │   ├── CatBoost
│   │   └── SVR (Support Vector Regression)
│   ├── Deep Learning Models
│   │   ├── LSTM (Long Short-Term Memory)
│   │   ├── GRU (Gated Recurrent Unit)
│   │   └── TCN (Temporal Convolutional Network)
│   ├── Hybrid Models
│   │   └── LSTM-ARIMA
│   └── Orchestration Layer
│       ├── ModelOrchestrator (9-Model Management)
│       ├── EnsembleOptimizer (Optuna-Powered)
│       └── GoldPriceService (Unified API)
│
├── 🧠 Legacy ML Engine (analyzer.py)
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
├── 🗄️ Data Storage
│   ├── DuckDB (Primary Database - database_schema.py)
│   │   ├── Prices Table (Live Data)
│   │   ├── Historical_Prices Table (Time Series)
│   │   ├── External_Data Table (Market Feeds)
│   │   ├── Investment_Tracking Table (Transactions)
│   │   ├── Investment_Goals Table (Targets)
│   │   └── Portfolio Table (Holdings)
│   └── SQLite (Forecast Cache - services/gold_price_service.py)
│       └── Cached forecast results with TTL
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
                      ↓                     ↓
               Validation           9-Model Ensemble
                                           ↓
                                    Optuna Optimizer → Weighted Predictions
                                           ↓
                                    SQLite Cache → Instant Retrieval
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

## 🖥️ Streamlit UI User Guide

### Advanced Forecasting Tab (9-Model System)

#### Step 1: Configure Forecast Parameters

1. **Forecast Horizon**: Select 1-30 days using the slider
2. **Optimization**: Toggle "Optimize ensemble weights with Optuna"
   - ✅ Enabled: 30-trial Bayesian optimization for best weights
   - ❌ Disabled: Use equal-weight ensemble
3. **Legacy Fallback**: Toggle "Use legacy forecast if advanced fails"
   - Automatically uses Prophet + Random Forest if 9-model system encounters errors

#### Step 2: Generate Forecast

1. Click **"Generate Advanced Forecast"** button
2. Watch progress indicators:
   - 🔄 Loading historical data
   - 🎯 Training 9 models with 85/15 train/val split
   - 🔧 Optimizing ensemble weights (if enabled)
   - 📈 Generating forecast with confidence intervals

#### Step 3: Review Results

**Training Results Section:**

- ✅ Green checkmarks: Successfully trained models
- ❌ Red X marks: Failed models (excluded from ensemble)
- View which of the 9 models are contributing

**Model Comparison Table:**

- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **R² Score**: Coefficient of determination (higher is better)
- **Sort**: Click column headers to sort by performance

**Ensemble Metadata:**

- **Best Individual Model**: Highest-performing single model
- **Ensemble RMSE**: Combined ensemble performance
- **Optimized Weights**: Weight distribution across models (if optimization enabled)

**Service Metrics:**

- **Cache Hits**: Forecasts retrieved from cache
- **Cache Misses**: Forecasts computed fresh
- **Hit Rate**: Cache efficiency percentage
- **Total Requests**: All forecast requests
- **Errors**: Failed forecast attempts

#### Step 4: Visualize & Export

**Interactive Chart:**

- Blue line: Historical prices
- Red line: Ensemble forecast
- Shaded area: 95% confidence interval
- Hover for detailed values

**Export Options:**

1. **📊 Download Forecast CSV**: Date, forecast, lower/upper bounds
2. **📋 Download Comparison CSV**: Full model comparison table
3. **📄 Export to PDF**: Comprehensive report with all metrics

### Model Information Reference

**Classical Models (2):**

- **ARIMA**: Statistical model for trending data
- **ETS**: Error-Trend-Seasonal decomposition

**ML Enhanced (3):**

- **LightGBM**: Fast gradient boosting (Microsoft)
- **CatBoost**: Categorical data specialist (Yandex)
- **SVR**: Support Vector Regression for non-linear patterns

**Deep Learning (3):**

- **LSTM**: Sequence modeling with long-term memory
- **GRU**: Efficient gated recurrent network
- **TCN**: Temporal convolutions with dilations

**Hybrid (1):**

- **LSTM-ARIMA**: Neural network + statistical fusion

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
