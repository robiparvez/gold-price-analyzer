"""User Prediction View - Tab 5 of the dashboard.

Allows users to input custom parameters for gold price prediction.
"""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from views.base import BaseView

# Seasonal adjustment factors
SEASONAL_FACTORS: dict[str, float] = {
    "Normal": 1.0,
    "Festive Season": 1.05,
    "Wedding Season": 1.08,
    "Economic Downturn": 0.92,
}


class UserPredictionView(BaseView):
    """View for custom user predictions."""

    def __init__(self, format_price_func: Any):
        """Initialize the user prediction view.

        Args:
            format_price_func: Function to format price display.
        """
        self.format_price_bdt = format_price_func

    def render(self, settings: dict[str, Any]) -> None:
        """Render the user prediction tab.

        Args:
            settings: Dashboard settings dictionary.
        """
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        if not settings["enable_forecast"]:
            st.info("🎲 Enable forecasting in the sidebar to use user input prediction")
        else:
            self._render_prediction_content()

        st.markdown("</div>", unsafe_allow_html=True)

    def _render_prediction_content(self) -> None:
        """Render the main prediction content."""
        st.markdown("### 🎲 Custom Prediction with User Inputs")

        # Get user inputs
        inputs = self._get_user_inputs()

        # Generate prediction button
        if st.button(
            "🔮 Generate Custom Prediction",
            type="primary",
            use_container_width=True,
        ):
            self._generate_and_display_prediction(inputs)

        # Help information
        self._display_help_section()

    def _get_user_inputs(self) -> dict[str, Any]:
        """Get user input parameters.

        Returns:
            Dictionary of user input values.
        """
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Input Parameters")
            base_price = st.number_input(
                "Current Gold Price (BDT/gram)",
                min_value=0,
                max_value=20000,
                value=7500,
                step=100,
                help="Enter the current gold price per gram",
            )

        with col2:
            st.markdown("#### Market Factors")
            exchange_rate = st.number_input(
                "USD to BDT Exchange Rate",
                min_value=50.0,
                max_value=200.0,
                value=110.0,
                step=0.5,
                help="Current USD to BDT exchange rate",
            )

        with col3:
            st.markdown("#### Prediction Settings")
            prediction_days = st.slider(
                "Prediction Horizon (Days)",
                min_value=1,
                max_value=30,
                value=7,
                help="How many days ahead to predict",
            )

        # Technical indicators input
        volatility, momentum, seasonality = self._get_technical_inputs()

        return {
            "base_price": base_price,
            "exchange_rate": exchange_rate,
            "prediction_days": prediction_days,
            "volatility": volatility,
            "momentum": momentum,
            "seasonality": seasonality,
        }

    def _get_technical_inputs(self) -> tuple[float, int, str]:
        """Get technical indicator inputs.

        Returns:
            Tuple of (volatility, momentum, seasonality).
        """
        with st.expander("📊 Advanced Technical Indicators (Optional)"):
            col1, col2, col3 = st.columns(3)

            with col1:
                volatility = st.number_input(
                    "Price Volatility (%)",
                    min_value=0.0,
                    max_value=20.0,
                    value=2.5,
                    step=0.1,
                    help="Expected price volatility",
                )

            with col2:
                momentum = st.slider(
                    "Market Momentum",
                    min_value=-10,
                    max_value=10,
                    value=0,
                    help="Current market momentum (-10 bearish to +10 bullish)",
                )

            with col3:
                seasonality = st.selectbox(
                    "Seasonal Factor",
                    list(SEASONAL_FACTORS.keys()),
                    help="Select current market season",
                )

        return volatility, momentum, seasonality

    def _generate_and_display_prediction(self, inputs: dict[str, Any]) -> None:
        """Generate and display the prediction.

        Args:
            inputs: User input parameters.
        """
        with st.spinner("Generating custom prediction..."):
            try:
                prediction_df = self._calculate_predictions(inputs)
                self._display_prediction_results(prediction_df, inputs)

            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")

    def _calculate_predictions(self, inputs: dict[str, Any]) -> pd.DataFrame:
        """Calculate price predictions.

        Args:
            inputs: User input parameters.

        Returns:
            DataFrame with predictions.
        """
        base_price = inputs["base_price"]
        prediction_days = inputs["prediction_days"]
        volatility = inputs["volatility"]
        momentum = inputs["momentum"]
        seasonality = inputs["seasonality"]

        seasonal_multiplier = SEASONAL_FACTORS[seasonality]

        predictions = []
        dates = []
        base_adjusted = base_price * seasonal_multiplier

        for day in range(1, prediction_days + 1):
            # Add some randomness and trends
            daily_change = (momentum * 0.1) + (
                volatility * 0.05 * (1 if day % 2 == 0 else -1)
            )
            price = base_adjusted * (1 + daily_change * day / 10)

            predictions.append(price)
            dates.append(datetime.now() + timedelta(days=day))

        return pd.DataFrame({"date": dates, "predicted_price": predictions})

    def _display_prediction_results(
        self, prediction_df: pd.DataFrame, inputs: dict[str, Any]
    ) -> None:
        """Display prediction results.

        Args:
            prediction_df: DataFrame with predictions.
            inputs: User input parameters.
        """
        base_price = inputs["base_price"]
        volatility = inputs["volatility"]
        prediction_days = inputs["prediction_days"]

        st.markdown("### 📊 Custom Prediction Results")

        # Summary metrics
        self._display_summary_metrics(prediction_df, base_price)

        # Chart
        self._display_prediction_chart(
            prediction_df, base_price, volatility, prediction_days
        )

        # Detailed table
        self._display_prediction_table(prediction_df, base_price)

        # Disclaimer
        st.warning(
            """
        ⚠️ **Disclaimer**: These predictions are based on user-provided inputs and simplified models.
        They should not be used for making actual financial decisions. Consult with professional
        financial advisors before making any investment decisions based on gold prices.
        """
        )

    def _display_summary_metrics(
        self, prediction_df: pd.DataFrame, base_price: float
    ) -> None:
        """Display summary metrics.

        Args:
            prediction_df: DataFrame with predictions.
            base_price: Starting base price.
        """
        col1, col2, col3, col4 = st.columns(4)

        avg_price = prediction_df["predicted_price"].mean()
        max_price = prediction_df["predicted_price"].max()
        min_price = prediction_df["predicted_price"].min()

        with col1:
            st.metric("Starting Price", self.format_price_bdt(base_price, decimals=0))

        with col2:
            delta_pct = (avg_price - base_price) / base_price * 100
            st.metric(
                "Predicted Average",
                self.format_price_bdt(avg_price, decimals=0),
                delta=f"{delta_pct:.1f}%",
            )

        with col3:
            delta_pct = (max_price - base_price) / base_price * 100
            st.metric(
                "Highest Prediction",
                self.format_price_bdt(max_price, decimals=0),
                delta=f"{delta_pct:.1f}%",
            )

        with col4:
            delta_pct = (min_price - base_price) / base_price * 100
            st.metric(
                "Lowest Prediction",
                self.format_price_bdt(min_price, decimals=0),
                delta=f"{delta_pct:.1f}%",
            )

    def _display_prediction_chart(
        self,
        prediction_df: pd.DataFrame,
        base_price: float,
        volatility: float,
        prediction_days: int,
    ) -> None:
        """Display prediction chart.

        Args:
            prediction_df: DataFrame with predictions.
            base_price: Starting base price.
            volatility: Volatility percentage.
            prediction_days: Number of prediction days.
        """
        fig = go.Figure()

        # Current price marker
        fig.add_trace(
            go.Scatter(
                x=[datetime.now()],
                y=[base_price],
                mode="markers",
                name="Current Price",
                marker=dict(size=15, color="red", symbol="circle"),
            )
        )

        # Prediction line
        fig.add_trace(
            go.Scatter(
                x=prediction_df["date"],
                y=prediction_df["predicted_price"],
                mode="lines+markers",
                name="Predicted Price",
                line=dict(color="#00cc96", width=3),
                marker=dict(size=8),
            )
        )

        # Confidence bands
        upper_band = prediction_df["predicted_price"] * (1 + volatility / 100)
        lower_band = prediction_df["predicted_price"] * (1 - volatility / 100)

        fig.add_trace(
            go.Scatter(
                x=prediction_df["date"],
                y=upper_band,
                mode="lines",
                name="Upper Band",
                line=dict(color="rgba(0,0,0,0.2)", width=0),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=prediction_df["date"],
                y=lower_band,
                mode="lines",
                name="Confidence Band",
                line=dict(color="rgba(0,0,0,0.2)", width=0),
                fill="tonexty",
                fillcolor="rgba(0,204,150,0.2)",
            )
        )

        fig.update_layout(
            title=f"Custom Price Prediction - {prediction_days} Days",
            xaxis_title="Date",
            yaxis_title="Predicted Price (BDT/gram)",
            yaxis=dict(tickformat=",.0f", ticksuffix=" BDT"),
            height=500,
            hovermode="x unified",
        )

        st.plotly_chart(fig, config={"responsive": True})

    def _display_prediction_table(
        self, prediction_df: pd.DataFrame, base_price: float
    ) -> None:
        """Display detailed prediction table.

        Args:
            prediction_df: DataFrame with predictions.
            base_price: Starting base price.
        """
        st.markdown("### 📋 Detailed Predictions")

        display_df = prediction_df.copy()
        display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
        display_df["daily_change"] = display_df["predicted_price"].diff().fillna(0)
        display_df["daily_change_pct"] = (
            display_df["daily_change"]
            / display_df["predicted_price"].shift(1).fillna(base_price)
            * 100
        )

        display_df.columns = [
            "Date",
            "Predicted Price (BDT/gram)",
            "Daily Change",
            "Daily Change (%)",
        ]
        st.dataframe(display_df, use_container_width=True)

        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name=f"gold_price_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    def _display_help_section(self) -> None:
        """Display help information."""
        with st.expander("💡 How to Use This Feature"):
            st.markdown(
                """
            1. **Enter Current Price**: Input today's gold price per gram
            2. **Set Exchange Rate**: Provide the current USD to BDT exchange rate
            3. **Choose Prediction Horizon**: Select how many days ahead you want to predict
            4. **Adjust Technical Indicators** (Optional):
               - **Volatility**: How much the price typically fluctuates
               - **Momentum**: Current market trend (-10 bearish to +10 bullish)
               - **Seasonality**: Select current market season for better accuracy
            5. **Generate Prediction**: Click the button to see your custom forecast

            The prediction uses a combination of your inputs, seasonal factors, and volatility
            adjustments to generate an educated forecast. The confidence band shows the
            expected range of prices based on the volatility setting.
            """
            )


def render_user_prediction_tab(
    settings: dict[str, Any],
    format_price_func: Any,
) -> None:
    """Render the user prediction tab.

    Args:
        settings: Dashboard settings dictionary.
        format_price_func: Function to format price display.
    """
    view = UserPredictionView(format_price_func=format_price_func)
    view.render(settings)
