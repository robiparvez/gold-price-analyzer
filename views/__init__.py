"""Views package for Streamlit UI components.

This package contains modular view components extracted from app.py
to improve maintainability and separation of concerns.
"""

from views.backtesting_view import BacktestingView, render_backtesting_tab
from views.base import (
    BaseView,
    create_section_header,
    display_metric_card,
    format_price_display,
    wrap_in_tab_content,
)
from views.current_analysis_view import CurrentAnalysisView, render_current_analysis_tab
from views.forecast_view import ForecastView, render_forecast_tab
from views.historical_trends_view import (
    HistoricalTrendsView,
    render_historical_trends_tab,
)
from views.investment_tracker_view import (
    InvestmentTrackerView,
    render_investment_tracker_tab,
)
from views.jewelry_pricing_view import JewelryPricingView, render_jewelry_pricing_tab
from views.model_performance_view import (
    ModelPerformanceView,
    render_model_performance_tab,
)
from views.user_prediction_view import UserPredictionView, render_user_prediction_tab
from views.validation_view import ValidationView, render_validation_tab

__all__ = [
    # Base classes and utilities
    "BaseView",
    "create_section_header",
    "display_metric_card",
    "format_price_display",
    "wrap_in_tab_content",
    # View classes
    "BacktestingView",
    "CurrentAnalysisView",
    "ForecastView",
    "HistoricalTrendsView",
    "InvestmentTrackerView",
    "JewelryPricingView",
    "ModelPerformanceView",
    "UserPredictionView",
    "ValidationView",
    # Functional interfaces
    "render_backtesting_tab",
    "render_current_analysis_tab",
    "render_forecast_tab",
    "render_historical_trends_tab",
    "render_investment_tracker_tab",
    "render_jewelry_pricing_tab",
    "render_model_performance_tab",
    "render_user_prediction_tab",
    "render_validation_tab",
]
