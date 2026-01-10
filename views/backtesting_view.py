"""Backtesting View - Tab 8 of the dashboard.

Model backtesting using historical data with rolling windows.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from views.base import BaseView


class BacktestingView(BaseView):
    """View for model backtesting functionality."""

    def __init__(
        self,
        backtester_class: Any,
        load_historical_func: Any,
        logger: Any,
    ):
        """Initialize the backtesting view.

        Args:
            backtester_class: GoldPriceBacktester class.
            load_historical_func: Function to load historical data.
            logger: Logger instance.
        """
        self.BacktesterClass = backtester_class
        self.load_historical_data = load_historical_func
        self.logger = logger

    def render(self, settings: dict[str, Any]) -> None:
        """Render the backtesting tab.

        Args:
            settings: Dashboard settings dictionary.
        """
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("## 🧪 Model Backtesting")

        st.markdown(
            """
        Evaluate model performance using historical data. Backtesting simulates how the model
        would have performed in the past by training on historical windows and testing predictions.
        """
        )

        historical_df = self.load_historical_data(days=365)

        if not historical_df.empty:
            inputs = self._render_configuration(historical_df)
            self._render_backtest_controls(historical_df, inputs)
            self._render_model_comparison(historical_df, inputs)
        else:
            st.warning("📭 No historical data available for backtesting.")

        st.markdown("</div>", unsafe_allow_html=True)

    def _render_configuration(self, historical_df: pd.DataFrame) -> dict[str, Any]:
        """Render backtest configuration inputs.

        Args:
            historical_df: Historical data.

        Returns:
            Dictionary of configuration values.
        """
        col1, col2 = st.columns(2)

        with col1:
            purity = st.selectbox(
                "Select Purity", ["22K", "21K", "18K"], key="backtest_purity"
            )

            window_size = st.slider(
                "Training Window (days)",
                min_value=14,
                max_value=90,
                value=30,
                step=7,
                help="Number of days to use for training",
            )

        with col2:
            forecast_horizon = st.slider(
                "Forecast Horizon (days)",
                min_value=1,
                max_value=14,
                value=7,
                step=1,
                help="Number of days ahead to predict",
            )

            model_type = st.selectbox(
                "Model Type",
                ["ensemble", "prophet", "random_forest", "xgboost"],
                help="Select model to backtest",
            )

        return {
            "purity": purity,
            "window_size": window_size,
            "forecast_horizon": forecast_horizon,
            "model_type": model_type,
        }

    def _render_backtest_controls(
        self, historical_df: pd.DataFrame, inputs: dict[str, Any]
    ) -> None:
        """Render backtest execution controls.

        Args:
            historical_df: Historical data.
            inputs: Configuration values.
        """
        if st.button("🚀 Run Backtest", type="primary"):
            self._run_backtest(historical_df, inputs)

    def _run_backtest(
        self, historical_df: pd.DataFrame, inputs: dict[str, Any]
    ) -> None:
        """Run the backtest.

        Args:
            historical_df: Historical data.
            inputs: Configuration values.
        """
        with st.spinner("Running backtest... This may take a few minutes..."):
            try:
                # Filter data
                backtest_df = historical_df[
                    (historical_df["purity"] == inputs["purity"])
                    & (historical_df["metal"] == "gold")
                ].copy()

                min_required = inputs["window_size"] + inputs["forecast_horizon"]
                if len(backtest_df) < min_required:
                    st.error(f"Insufficient data. Need at least {min_required} days.")
                    return

                backtester = self.BacktesterClass()

                results = backtester.rolling_window_backtest(
                    df=backtest_df,
                    window_size=inputs["window_size"],
                    forecast_horizon=inputs["forecast_horizon"],
                    model_type=inputs["model_type"],
                    step_size=inputs["forecast_horizon"],
                )

                if results and "metrics" in results:
                    st.success("✅ Backtest completed successfully!")
                    self._display_backtest_results(results, inputs, backtester)
                else:
                    st.error("Backtest failed to produce results.")

            except Exception as e:
                st.error(f"Error during backtesting: {str(e)}")
                self.logger.error(f"Backtesting error: {e}")

    def _display_backtest_results(
        self, results: dict[str, Any], inputs: dict[str, Any], backtester: Any
    ) -> None:
        """Display backtest results.

        Args:
            results: Backtest results.
            inputs: Configuration values.
            backtester: Backtester instance.
        """
        self._display_metrics(results)
        self._display_predictions_chart(results, inputs)
        self._display_error_distribution(results, backtester)
        self._display_detailed_results(results, inputs)

    def _display_metrics(self, results: dict[str, Any]) -> None:
        """Display performance metrics.

        Args:
            results: Backtest results.
        """
        st.markdown("### 📊 Performance Metrics")

        col1, col2, col3, col4, col5 = st.columns(5)

        metrics = results["metrics"]

        with col1:
            st.metric(
                "MAE", f"{metrics['mae']:.2f}", help="Mean Absolute Error (BDT/gram)"
            )

        with col2:
            st.metric(
                "RMSE",
                f"{metrics['rmse']:.2f}",
                help="Root Mean Squared Error (BDT/gram)",
            )

        with col3:
            st.metric(
                "MAPE", f"{metrics['mape']:.2f}%", help="Mean Absolute Percentage Error"
            )

        with col4:
            st.metric(
                "R² Score",
                f"{metrics['r2_score']:.3f}",
                help="Coefficient of determination",
            )

        with col5:
            st.metric(
                "Predictions",
                metrics["total_predictions"],
                help="Total number of predictions made",
            )

    def _display_predictions_chart(
        self, results: dict[str, Any], inputs: dict[str, Any]
    ) -> None:
        """Display predictions vs actual chart.

        Args:
            results: Backtest results.
            inputs: Configuration values.
        """
        st.markdown("### 📈 Predictions vs Actual Prices")

        results_df = results["results_df"]

        fig = go.Figure()

        # Actual prices
        fig.add_trace(
            go.Scatter(
                x=results_df["date"],
                y=results_df["actual"],
                mode="lines+markers",
                name="Actual Price",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=6),
            )
        )

        # Predicted prices
        fig.add_trace(
            go.Scatter(
                x=results_df["date"],
                y=results_df["predicted"],
                mode="lines+markers",
                name="Predicted Price",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                marker=dict(size=6, symbol="x"),
            )
        )

        fig.update_layout(
            title=f"Backtest Results: {inputs['model_type'].title()} Model",
            xaxis_title="Date",
            yaxis_title="Price (BDT/gram)",
            yaxis=dict(tickformat=",.0f", ticksuffix=" BDT"),
            height=500,
            hovermode="x unified",
        )

        st.plotly_chart(fig, config={"responsive": True})

    def _display_error_distribution(
        self, results: dict[str, Any], backtester: Any
    ) -> None:
        """Display error distribution.

        Args:
            results: Backtest results.
            backtester: Backtester instance.
        """
        st.markdown("### 📉 Error Distribution")

        col1, col2 = st.columns(2)

        with col1:
            results_df = results["results_df"]
            fig_hist = go.Figure(
                data=[
                    go.Histogram(
                        x=results_df["error"],
                        nbinsx=30,
                        marker_color="#2ca02c",
                    )
                ]
            )

            fig_hist.update_layout(
                title="Distribution of Prediction Errors",
                xaxis_title="Absolute Error (BDT/gram)",
                yaxis_title="Frequency",
                height=400,
            )

            st.plotly_chart(fig_hist, config={"responsive": True})

        with col2:
            error_stats = backtester.get_error_distribution()

            if error_stats:
                st.markdown("#### Error Statistics")
                st.metric("Mean Error", f"{error_stats['mean_error']:.2f} BDT/g")
                st.metric("Std Error", f"{error_stats['std_error']:.2f} BDT/g")
                st.metric("Median Error", f"{error_stats['median_error']:.2f} BDT/g")
                st.metric(
                    "95th Percentile", f"{error_stats['percentile_95']:.2f} BDT/g"
                )

    def _display_detailed_results(
        self, results: dict[str, Any], inputs: dict[str, Any]
    ) -> None:
        """Display detailed results table.

        Args:
            results: Backtest results.
            inputs: Configuration values.
        """
        with st.expander("📋 Detailed Results"):
            display_results = results["results_df"].copy()
            display_results["date"] = pd.to_datetime(
                display_results["date"]
            ).dt.strftime("%Y-%m-%d")
            display_results["error_pct"] = (
                display_results["error"] / display_results["actual"] * 100
            )
            display_results.columns = [
                "Date",
                "Actual Price",
                "Predicted Price",
                "Absolute Error",
                "Error %",
            ]
            st.dataframe(display_results, use_container_width=True)

            # Download button
            csv = display_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Backtest Results",
                data=csv,
                file_name=f"backtest_results_{inputs['model_type']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

    def _render_model_comparison(
        self, historical_df: pd.DataFrame, inputs: dict[str, Any]
    ) -> None:
        """Render model comparison section.

        Args:
            historical_df: Historical data.
            inputs: Configuration values.
        """
        st.markdown("---")
        st.markdown("### 🏆 Model Comparison")

        if st.button("📊 Compare All Models"):
            self._run_model_comparison(historical_df, inputs)

    def _run_model_comparison(
        self, historical_df: pd.DataFrame, inputs: dict[str, Any]
    ) -> None:
        """Run model comparison.

        Args:
            historical_df: Historical data.
            inputs: Configuration values.
        """
        with st.spinner("Comparing models... This will take several minutes..."):
            try:
                backtest_df = historical_df[
                    (historical_df["purity"] == inputs["purity"])
                    & (historical_df["metal"] == "gold")
                ].copy()

                backtester = self.BacktesterClass()

                comparison = backtester.compare_models(
                    df=backtest_df,
                    window_size=inputs["window_size"],
                    forecast_horizon=inputs["forecast_horizon"],
                )

                if comparison and "comparison_df" in comparison:
                    st.success("✅ Model comparison completed!")
                    self._display_comparison_results(comparison)
                else:
                    st.error("Model comparison failed.")

            except Exception as e:
                st.error(f"Error during model comparison: {str(e)}")
                self.logger.error(f"Model comparison error: {e}")

    def _display_comparison_results(self, comparison: dict[str, Any]) -> None:
        """Display comparison results.

        Args:
            comparison: Comparison results.
        """
        comp_df = comparison["comparison_df"]

        st.markdown("#### Performance Comparison")

        display_comp = comp_df.copy()
        display_comp.index.name = "Model"
        display_comp = display_comp.reset_index()

        st.dataframe(display_comp, use_container_width=True)

        st.info(f"🏆 Best Model: **{comparison['best_model'].upper()}** (Lowest MAE)")

        # Comparison chart
        fig_comp = go.Figure()

        metrics_to_plot = ["mae", "rmse", "mape"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for i, metric in enumerate(metrics_to_plot):
            fig_comp.add_trace(
                go.Bar(
                    name=metric.upper(),
                    x=comp_df.index,
                    y=comp_df[metric],
                    marker_color=colors[i],
                )
            )

        fig_comp.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Error Value",
            barmode="group",
            height=500,
        )

        st.plotly_chart(fig_comp, config={"responsive": True})


def render_backtesting_tab(
    settings: dict[str, Any],
    backtester_class: Any,
    load_historical_func: Any,
    logger: Any,
) -> None:
    """Render the backtesting tab.

    Args:
        settings: Dashboard settings dictionary.
        backtester_class: GoldPriceBacktester class.
        load_historical_func: Function to load historical data.
        logger: Logger instance.
    """
    view = BacktestingView(
        backtester_class=backtester_class,
        load_historical_func=load_historical_func,
        logger=logger,
    )
    view.render(settings)
