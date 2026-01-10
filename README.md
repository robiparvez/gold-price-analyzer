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
- **Caching System**: DuckDB-based forecast caching for instant retrieval

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
├── 🧠 Utility Functions (analyzer.py)
│   ├── Data Loading & Statistics
│   ├── Trend Analysis & Indicators
│   └── Chart Generation
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

## 💻 Usage Examples

### Advanced 9-Model Forecasting

```python
from services.gold_price_service import GoldPriceService

# Initialize service with all 9 models
service = GoldPriceService()

# Load and prepare historical data
data = service.load_historical_data(purity="22K")

# Train all models with 85/15 train/validation split
service.train_models(data, val_split=0.15)

# Optimize ensemble weights using Optuna (30 trials)
service.optimize_ensemble(n_trials=30, timeout=300)

# Generate 14-day forecast with optimized ensemble
forecast_result = service.generate_forecast(
    forecast_days=14,
    use_cache=True
)

print(f"Ensemble RMSE: {forecast_result['ensemble_rmse']:.2f}")
print(f"Best Model: {forecast_result['best_model']}")
print(f"Optimized Weights: {forecast_result['ensemble_weights']}")

# Access individual model predictions
for model_name, predictions in forecast_result['model_forecasts'].items():
    print(f"{model_name}: {predictions['forecast'][:3]}...")
```

### Model Comparison & Selection

```python
from models.orchestrator import ModelOrchestrator

# Initialize orchestrator with all 9 models
orchestrator = ModelOrchestrator()

# Train and evaluate models
train_results = orchestrator.train_all_models(
    train_data=train_df,
    val_data=val_df
)

# Compare model performance
comparison_df = orchestrator.get_model_comparison()
print(comparison_df[['model', 'rmse', 'mae', 'r2']].sort_values('rmse'))

# Generate ensemble forecast with multiple methods
ensemble_pred = orchestrator.get_ensemble_forecast(
    forecast_days=7,
    method='weighted_mean'  # Options: weighted_mean, equal_weight, median
)
```

### Optuna-Powered Optimization

```python
from models.ensemble_optimizer import EnsembleOptimizer

# Initialize optimizer with trained orchestrator
optimizer = EnsembleOptimizer(orchestrator)

# Run Bayesian optimization (50 trials, 10 min timeout)
best_weights = optimizer.optimize_weights(
    n_trials=50,
    timeout=600,
    pruner='median',
    sampler='tpe'
)

print(f"Optimized RMSE: {optimizer.best_rmse:.2f}")
print("Best Ensemble Weights:")
for model, weight in best_weights.items():
    print(f"  {model}: {weight:.3f}")
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

## 🖥️ Streamlit UI User Guide

### Advanced Forecasting Tab (9-Model System)

#### Step 1: Configure Forecast Parameters

1. **Forecast Horizon**: Select 1-30 days using the slider
2. **Optimization**: Toggle "Use Optuna Optimization"
   - ✅ Enabled: 30-trial Bayesian optimization for best weights
   - ❌ Disabled: Use weighted-mean ensemble based on variance

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

#### `GoldPriceService`

**Unified service layer for 9-model forecasting with optimization and caching.**

```python
class GoldPriceService:
    def __init__(self, cache_enabled: bool = True, cache_ttl: int = 3600)

    def load_historical_data(self, purity: str = "22K", days_back: int = 365) -> pd.DataFrame
        """Load historical price data from DuckDB."""

    def train_models(self, data: pd.DataFrame, val_split: float = 0.15) -> dict
        """Train all 9 models with train/validation split.
        Returns: {model_name: success_status}"""

    def optimize_ensemble(self, n_trials: int = 30, timeout: int = 300) -> dict
        """Optimize ensemble weights using Optuna.
        Returns: {weights: dict, rmse: float}"""

    def generate_forecast(self, forecast_days: int = 7, use_cache: bool = True) -> dict
        """Generate ensemble forecast with all trained models.
        Returns: {
            'forecast': list[float],
            'lower_bound': list[float],
            'upper_bound': list[float],
            'model_forecasts': dict,
            'ensemble_weights': dict,
            'ensemble_rmse': float,
            'best_model': str
        }"""

    def get_service_metrics(self) -> dict
        """Get service performance metrics.
        Returns: {cache_hits, cache_misses, hit_rate, total_requests, errors}"""

    def clear_cache(self) -> None
        """Clear all cached forecasts."""
```

#### `ModelOrchestrator`

**Manages all 9 forecasting models with ensemble methods.**

```python
class ModelOrchestrator:
    def __init__(self)

    def train_all_models(self, train_data: pd.DataFrame,
                        val_data: pd.DataFrame) -> dict[str, bool]
        """Train all 9 models and return success status for each."""

    def get_model_comparison(self) -> pd.DataFrame
        """Get comparison table with RMSE, MAE, R² for all models."""

    def get_ensemble_forecast(self, forecast_days: int = 7,
                            method: str = 'weighted_mean') -> dict
        """Generate ensemble forecast.
        Methods: 'weighted_mean', 'equal_weight', 'median'"""

    def get_available_models(self) -> list[str]
        """List all 9 available model names."""
```

#### `EnsembleOptimizer`

**Optuna-powered Bayesian optimization for ensemble weights.**

```python
class EnsembleOptimizer:
    def __init__(self, orchestrator: ModelOrchestrator)

    def optimize_weights(self, n_trials: int = 30, timeout: int = 300,
                        pruner: str = 'median', sampler: str = 'tpe') -> dict[str, float]
        """Optimize ensemble weights using Optuna.
        Pruners: 'median', 'threshold', 'percentile'
        Samplers: 'tpe', 'random', 'cmaes'"""

    @property
    def best_rmse(self) -> float
        """Get best RMSE achieved during optimization."""

    def get_optimization_history(self) -> pd.DataFrame
        """Get trial-by-trial optimization results."""
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
