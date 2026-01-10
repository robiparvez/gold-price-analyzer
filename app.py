"""
Gold Price Analyzer - Refactored Streamlit Application

This is the refactored version using modular view components.
All tabs are now handled by separate view modules for better maintainability.
"""

import asyncio
import logging
import os
import warnings

import pandas as pd
import streamlit as st

from analyzer import AdvancedGoldPriceAnalyzer
from backtesting import GoldPriceBacktester
from database_schema import DatabaseSchema
from historical_scraper import HistoricalGoldPriceScraper
from jewelry_pricing import JewelryPricingCalculator, format_price_breakdown
from logging_config import setup_logging
from scraper import GoldPriceScraper
from services.forecast_validation_service import ForecastValidationService
from services.gold_price_service import GoldPriceService
from services.model_retraining_service import ModelRetrainingService
from utils import format_price_bdt

# Import refactored views
from views import (
    render_backtesting_tab,
    render_current_analysis_tab,
    render_forecast_tab,
    render_historical_trends_tab,
    render_investment_tracker_tab,
    render_jewelry_pricing_tab,
    render_model_performance_tab,
    render_user_prediction_tab,
    render_validation_tab,
)

# Suppress TensorFlow deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="GP Analyzer",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/robiparvez/gold-price-analyzer",
        "Report a bug": "https://github.com/robiparvez/gold-price-analyzer/issues",
        "About": "# Gold Price Analyzer\nProfessional ML-powered gold price forecasting.",
    },
)

# Custom CSS (simplified)
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #FFD700;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .tab-content { padding: 1rem 0; }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #cccccc;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Cached Data Loading Functions
# =============================================================================


@st.cache_data(ttl=300)
def load_price_data(days: int = 365) -> pd.DataFrame:
    """Load price data with caching."""
    analyzer = AdvancedGoldPriceAnalyzer()
    return analyzer.load_data(days=days)


@st.cache_data(ttl=300)
def load_historical_data(days: int = 365) -> pd.DataFrame:
    """Load historical data with caching."""
    scraper = HistoricalGoldPriceScraper()
    return scraper.load_historical_data(days=days)


@st.cache_data(ttl=1800)
def scrape_latest_prices() -> dict:
    """Scrape latest prices with caching."""
    try:
        scraper = GoldPriceScraper()
        price_data, _ = scraper.scrape_and_save()
        return price_data
    except Exception as e:
        logger.error(f"Error scraping prices: {e}")
        return {}


@st.cache_data(ttl=3600)
def fetch_and_save_historical_data(days: int = 365) -> pd.DataFrame:
    """Fetch historical data from external sources."""

    async def _fetch_data():
        scraper = HistoricalGoldPriceScraper()
        df = await scraper.fetch_all_historical_data(days=days)
        if not df.empty:
            scraper.save_historical_data(df)
        return df

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_fetch_data())
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()


@st.cache_resource
def get_forecasting_service():
    """Get or create the forecasting service singleton."""
    return GoldPriceService(cache_enabled=True)


# =============================================================================
# Sidebar Controls
# =============================================================================


def sidebar_controls() -> dict:
    """Create sidebar controls and return settings."""
    st.sidebar.title("⚙️ Settings")

    # Metal selection
    metal = st.sidebar.selectbox("Metal", ["gold"], format_func=lambda x: x.title())

    # Purity selection
    purity = st.sidebar.selectbox("Purity", ["22K", "21K", "18K", "Traditional"])

    # Forecast settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔮 Forecast Settings")

    enable_forecast = st.sidebar.checkbox("Enable Forecasting", value=True)

    forecast_days = st.sidebar.slider(
        "Forecast Days",
        min_value=1,
        max_value=14,
        value=7,
        disabled=not enable_forecast,
    )

    use_optimization = st.sidebar.checkbox(
        "Use Optuna Optimization",
        value=True,
        disabled=not enable_forecast,
        help="Enable hyperparameter optimization for better accuracy",
    )

    # Historical data settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Historical Data")

    use_historical = st.sidebar.checkbox("Use Historical Data", value=True)

    days = st.sidebar.slider(
        "Historical Days",
        min_value=30,
        max_value=365,
        value=90,
        disabled=not use_historical,
    )

    # Data refresh
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    return {
        "metal": metal,
        "purity": purity,
        "enable_forecast": enable_forecast,
        "forecast_days": forecast_days,
        "use_optimization": use_optimization,
        "use_historical": use_historical,
        "days": days,
    }


# =============================================================================
# Helper Functions for Views
# =============================================================================


def generate_advanced_forecast(
    historical_df: pd.DataFrame,
    purity: str,
    forecast_days: int = 7,
    use_optimization: bool = True,
) -> dict:
    """Generate forecast using advanced 9-model ensemble."""
    try:
        service = get_forecasting_service()

        if historical_df.empty:
            return {"error": "No historical data available"}

        # Filter by purity if needed
        if "purity" in historical_df.columns:
            df_filtered = historical_df[historical_df["purity"] == purity].copy()
        else:
            df_filtered = historical_df.copy()

        if df_filtered.empty:
            return {"error": f"No data available for purity {purity}"}

        # Prepare features and target
        # Use price as both feature and target (time series)
        if "price_bdt_per_gram" not in df_filtered.columns:
            return {"error": "Missing price_bdt_per_gram column in data"}

        # Sort by date and set date as index
        df_filtered = df_filtered.sort_values("date")
        if "date" in df_filtered.columns:
            df_filtered = df_filtered.set_index("date")

        # Create features (lagged values) and target
        X = df_filtered[["price_bdt_per_gram"]].iloc[:-1]  # All but last
        y = df_filtered["price_bdt_per_gram"].iloc[1:]  # Shifted by 1

        # Train models
        service.train(X, y)

        # Generate forecast
        forecast_result = service.forecast(
            steps=forecast_days, use_optimized=use_optimization
        )

        # Convert predictions to list
        predictions = (
            forecast_result.predictions.tolist()
            if hasattr(forecast_result.predictions, "tolist")
            else list(forecast_result.predictions)
        )

        # Build forecasts list with date, predicted_price, bounds
        forecasts = []
        for i, (date, price) in enumerate(zip(forecast_result.dates, predictions)):
            forecast_item = {
                "date": date,
                "predicted_price": float(price),
            }

            # Add confidence bounds if available
            if (
                hasattr(forecast_result, "confidence_lower")
                and forecast_result.confidence_lower is not None
            ):
                forecast_item["lower_bound"] = float(
                    forecast_result.confidence_lower[i]
                    if hasattr(forecast_result.confidence_lower, "__getitem__")
                    else forecast_result.confidence_lower.tolist()[i]
                )

            if (
                hasattr(forecast_result, "confidence_upper")
                and forecast_result.confidence_upper is not None
            ):
                forecast_item["upper_bound"] = float(
                    forecast_result.confidence_upper[i]
                    if hasattr(forecast_result.confidence_upper, "__getitem__")
                    else forecast_result.confidence_upper.tolist()[i]
                )

            forecasts.append(forecast_item)

        # Return in format expected by views
        return {
            "model_name": "9-Model Ensemble",
            "forecasts": forecasts,
            "purity": purity,
        }

    except Exception as e:
        logger.error(f"Error in generate_advanced_forecast: {e}")
        return {"error": str(e)}


def create_ohlc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create OHLC data from price data for candlestick charts."""
    if df.empty:
        return pd.DataFrame()

    ohlc = df.groupby(pd.Grouper(key="date", freq="W")).agg(
        {
            "price_bdt_per_gram": ["first", "max", "min", "last"],
        }
    )
    ohlc.columns = ["open", "high", "low", "close"]
    ohlc = ohlc.reset_index()
    return ohlc


def evaluate_model_performance(purity: str) -> dict:
    """Evaluate model performance for the given purity."""
    try:
        service = get_forecasting_service()
        df = service.get_model_comparison()

        # Convert DataFrame to dict format expected by views
        if df.empty:
            return {"error": "No model comparison data available"}

        models_dict = {}
        for idx, row in df.iterrows():
            model_name = row.get("model_name", idx)
            models_dict[model_name] = {
                "accuracy_percentage": (
                    (1 - row.get("mape", 100) / 100) * 100 if "mape" in row else 0
                ),
                "mape": row.get("mape", 0),
                "mae": row.get("mae", 0),
                "rmse": row.get("rmse", 0),
                "r2": row.get("r2", 0),
            }

        return {"models": models_dict}

    except Exception as e:
        logger.error(f"Error evaluating models: {e}")
        return {"error": str(e)}


# =============================================================================
# Main Application
# =============================================================================


def main():
    """Main application entry point."""
    # Header
    st.markdown(
        '<h1 class="main-header">🥇 Gold Price Analyzer</h1>',
        unsafe_allow_html=True,
    )

    # Initialize database
    db = DatabaseSchema()

    # Initialize services
    analyzer = AdvancedGoldPriceAnalyzer()

    # Get sidebar settings
    settings = sidebar_controls()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        [
            "📊 Current Analysis",
            "🔮 7-Day Forecast",
            "📈 Historical Trends",
            "🎯 Model Performance",
            "🎲 User Input Prediction",
            "💰 Investment Tracker",
            "💍 Jewelry Pricing",
            "🧪 Backtesting",
            "✅ Accuracy & Validation",
        ]
    )

    # Tab 1: Current Analysis
    with tab1:
        render_current_analysis_tab(
            analyzer=analyzer,
            load_price_data=load_price_data,
            load_historical_data=load_historical_data,
            settings=settings,
        )

    # Tab 2: Forecast
    with tab2:
        render_forecast_tab(
            generate_advanced_forecast=generate_advanced_forecast,
            load_historical_data=load_historical_data,
            fetch_historical_data=fetch_and_save_historical_data,
            get_forecasting_service=get_forecasting_service,
            settings=settings,
        )

    # Tab 3: Historical Trends
    with tab3:
        render_historical_trends_tab(
            settings=settings,
            analyzer=analyzer,
            load_historical_func=load_historical_data,
            fetch_historical_func=fetch_and_save_historical_data,
            create_ohlc_func=create_ohlc_data,
            format_price_func=format_price_bdt,
        )

    # Tab 4: Model Performance
    with tab4:
        render_model_performance_tab(
            settings=settings,
            evaluate_func=evaluate_model_performance,
            format_price_func=format_price_bdt,
        )

    # Tab 5: User Prediction
    with tab5:
        render_user_prediction_tab(
            settings=settings,
            format_price_func=format_price_bdt,
        )

    # Tab 6: Investment Tracker
    with tab6:
        render_investment_tracker_tab(
            settings=settings,
            db=db,
            format_price_func=format_price_bdt,
            logger=logger,
        )

    # Tab 7: Jewelry Pricing
    with tab7:
        calculator = JewelryPricingCalculator()
        render_jewelry_pricing_tab(
            settings=settings,
            calculator=calculator,
            format_price_func=format_price_bdt,
            format_breakdown_func=format_price_breakdown,
        )

    # Tab 8: Backtesting
    with tab8:
        render_backtesting_tab(
            settings=settings,
            backtester_class=GoldPriceBacktester,
            load_historical_func=load_historical_data,
            logger=logger,
        )

    # Tab 9: Validation
    with tab9:
        validation_service = ForecastValidationService()
        retraining_service = ModelRetrainingService()
        render_validation_tab(
            settings=settings,
            validation_service=validation_service,
            retraining_service=retraining_service,
            scraper_class=GoldPriceScraper,
            logger=logger,
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "Gold Price Analyzer | ML Forecasting • Investment Tracking • "
        "Jewelry Pricing • Backtesting | Built with Streamlit"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
