"""Forecast View - 7-day price forecast with 9-model ensemble.

This view provides:
- Advanced 9-model forecasting configuration
- Optuna optimization options
- Forecast visualization and metrics
"""

import logging
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from views.base import BaseView, create_section_header

logger = logging.getLogger(__name__)


class ForecastView(BaseView):
    """View for 7-day price forecast."""

    def __init__(
        self,
        generate_advanced_forecast: callable,
        load_historical_data: callable,
        fetch_historical_data: callable,
        get_forecasting_service: callable,
    ):
        """Initialize the forecast view.

        Args:
            generate_advanced_forecast: 9-model forecast function.
            load_historical_data: Function to load historical data.
            fetch_historical_data: Function to fetch/save historical data.
            get_forecasting_service: Function to get the forecasting service.
        """
        super().__init__()
        self.generate_advanced_forecast = generate_advanced_forecast
        self.load_historical_data = load_historical_data
        self.fetch_historical_data = fetch_historical_data
        self.get_forecasting_service = get_forecasting_service

    def render(self, settings: dict[str, Any]) -> None:
        """Render the forecast view.

        Args:
            settings: User settings dictionary.
        """
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        if not settings.get("enable_forecast", False):
            st.info("🔮 Enable forecasting in the sidebar to view 7-day predictions")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        create_section_header("Advanced 9-Model Forecasting System", "🔮")

        # Forecast configuration
        forecast_days, use_optimization = self._render_config_controls()

        # Model information expander
        self._render_model_info()

        # Generate forecast button
        if st.button("🚀 Generate Forecast", type="primary"):
            self._generate_and_display_forecast(
                settings, forecast_days, use_optimization
            )

        # Display cached results if available
        if st.session_state.get("forecast_generated", False):
            self._display_forecast_results(st.session_state.forecast_results)

        st.markdown("</div>", unsafe_allow_html=True)

    def _render_config_controls(self) -> tuple[int, bool]:
        """Render forecast configuration controls.

        Returns:
            Tuple of (forecast_days, use_optimization).
        """
        col1, col2 = st.columns(2)

        with col1:
            forecast_days = st.slider(
                "Forecast Horizon (days)",
                min_value=1,
                max_value=30,
                value=7,
                help="Number of days to forecast ahead",
            )

        with col2:
            use_optimization = st.checkbox(
                "Use Optuna Optimization",
                value=True,
                help="Optimize ensemble weights using Bayesian optimization (30 trials)",
            )

        return forecast_days, use_optimization

    def _render_model_info(self) -> None:
        """Render the model information expander."""
        with st.expander("ℹ️ About the 9-Model System"):
            st.markdown(
                """
            **Classical Models (2):**
            - 🎯 **ARIMA**: Auto-regressive Integrated Moving Average for time series
            - 🎯 **ETS**: Exponential Smoothing with trend and seasonality

            **ML Enhanced Models (3):**
            - 🌲 **LightGBM**: Gradient boosting with fast training
            - 🐱 **CatBoost**: Gradient boosting optimized for categorical features
            - 🔮 **SVR**: Support Vector Regression for non-linear patterns

            **Deep Learning Models (3):**
            - 🧠 **LSTM**: Long Short-Term Memory networks for sequential patterns
            - 🧠 **GRU**: Gated Recurrent Units for efficient time series modeling
            - 🧠 **TCN**: Temporal Convolutional Networks for long-range dependencies

            **Hybrid Model (1):**
            - ⚡ **LSTM-ARIMA**: Combines deep learning with classical statistics

            **Ensemble Methods:**
            - 📊 **Weighted Mean**: Inversely proportional to prediction variance
            - 📊 **Equal Weight**: Simple average of all models
            - 📊 **Median**: Robust to outliers
            - 🔬 **Optuna-Optimized**: Bayesian hyperparameter optimization
            """
            )

    def _generate_and_display_forecast(
        self,
        settings: dict[str, Any],
        forecast_days: int,
        use_optimization: bool,
    ) -> None:
        """Generate forecast and display results.

        Args:
            settings: User settings dictionary.
            forecast_days: Number of days to forecast.
            use_optimization: Whether to use Optuna optimization.
        """
        with st.spinner("Training 9 models and generating forecast..."):
            try:
                # Load historical data
                historical_df = self.load_historical_data(settings["days"])

                if historical_df.empty:
                    st.warning("Fetching historical data...")
                    historical_df = self.fetch_historical_data(settings["days"])

                if historical_df.empty:
                    st.error("Unable to fetch historical data")
                    return

                # Generate forecast
                forecast_results = self.generate_advanced_forecast(
                    historical_df,
                    settings["purity"],
                    forecast_days=forecast_days,
                    use_optimization=use_optimization,
                )

                if "error" in forecast_results:
                    st.error(f"Forecast error: {forecast_results['error']}")
                    return

                st.success(
                    f"✅ Forecast generated using {forecast_results.get('model_name', 'ensemble')} model!"
                )

                # Store in session state
                st.session_state.forecast_results = forecast_results
                st.session_state.forecast_generated = True

                # Display results immediately
                self._display_forecast_results(forecast_results)

            except Exception as e:
                self.logger.exception("Forecast generation failed")
                st.error(f"Error generating forecast: {str(e)}")

    def _display_forecast_results(self, results: dict[str, Any]) -> None:
        """Display forecast results.

        Args:
            results: Forecast results dictionary.
        """
        create_section_header("Forecast Results", "📈")

        forecasts = results.get("forecasts", [])
        if not forecasts:
            st.warning("No forecast data available")
            st.info(f"Results keys: {list(results.keys())}")
            if "error" in results:
                st.error(f"Error in results: {results['error']}")
            return

        # Create forecast dataframe
        forecast_df = pd.DataFrame(forecasts)

        # Display forecast table
        st.dataframe(
            forecast_df,
            column_config={
                "date": st.column_config.DateColumn("Date"),
                "predicted_price": st.column_config.NumberColumn(
                    "Predicted Price", format="৳%.0f"
                ),
                "lower_bound": st.column_config.NumberColumn(
                    "Lower Bound", format="৳%.0f"
                ),
                "upper_bound": st.column_config.NumberColumn(
                    "Upper Bound", format="৳%.0f"
                ),
            },
            use_container_width=True,
        )

        # Create forecast chart
        self._create_forecast_chart(forecast_df)

        # Display metrics if available
        if "metrics" in results:
            self._display_model_metrics(results["metrics"])

    def _create_forecast_chart(self, forecast_df: pd.DataFrame) -> None:
        """Create and display forecast chart.

        Args:
            forecast_df: DataFrame with forecast data.
        """
        fig = go.Figure()

        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=list(forecast_df["date"]) + list(forecast_df["date"][::-1]),
                y=list(forecast_df["upper_bound"])
                + list(forecast_df["lower_bound"][::-1]),
                fill="toself",
                fillcolor="rgba(0, 100, 80, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% Confidence Interval",
            )
        )

        # Add predicted values
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["predicted_price"],
                mode="lines+markers",
                name="Predicted Price",
                line=dict(color="rgb(0, 100, 80)", width=2),
                marker=dict(size=8),
            )
        )

        fig.update_layout(
            title="7-Day Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (BDT/gram)",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    def _display_model_metrics(self, metrics: dict[str, Any]) -> None:
        """Display model metrics.

        Args:
            metrics: Dictionary with model metrics.
        """
        with st.expander("📊 Model Metrics"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("MAE", f"৳{metrics.get('mae', 0):.2f}")
            with col2:
                st.metric("RMSE", f"৳{metrics.get('rmse', 0):.2f}")
            with col3:
                st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")


def render_forecast_tab(
    generate_advanced_forecast: callable,
    load_historical_data: callable,
    fetch_historical_data: callable,
    get_forecasting_service: callable,
    settings: dict[str, Any],
) -> None:
    """Render the forecast tab (functional interface).

    Args:
        generate_advanced_forecast: 9-model forecast function.
        load_historical_data: Function to load historical data.
        fetch_historical_data: Function to fetch/save historical data.
        get_forecasting_service: Function to get forecasting service.
        settings: User settings dictionary.
    """
    view = ForecastView(
        generate_advanced_forecast,
        load_historical_data,
        fetch_historical_data,
        get_forecasting_service,
    )
    view.render(settings)
