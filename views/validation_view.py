"""Validation View - Tab 9 of the dashboard.

Forecast accuracy tracking and model management.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from views.base import BaseView


class ValidationView(BaseView):
    """View for forecast validation and model management."""

    def __init__(
        self,
        validation_service: Any,
        retraining_service: Any,
        scraper_class: Any,
        logger: Any,
    ):
        """Initialize the validation view.

        Args:
            validation_service: ForecastValidationService instance.
            retraining_service: ModelRetrainingService instance.
            scraper_class: GoldPriceScraper class.
            logger: Logger instance.
        """
        self.validation_service = validation_service
        self.retraining_service = retraining_service
        self.ScraperClass = scraper_class
        self.logger = logger

    def render(self, settings: dict[str, Any]) -> None:
        """Render the validation tab.

        Args:
            settings: Dashboard settings dictionary.
        """
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("## ✅ Forecast Accuracy & Validation")

        st.markdown(
            """
        This section tracks forecast accuracy against actual BAJUS prices and provides
        tools for model maintenance and improvement.
        """
        )

        try:
            self._render_validation_tabs()
        except Exception as e:
            st.error(f"Error loading validation services: {e}")
            self.logger.error(f"Validation tab error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    def _render_validation_tabs(self) -> None:
        """Render validation sub-tabs."""
        val_tab1, val_tab2, val_tab3, val_tab4 = st.tabs(
            [
                "📊 Daily Validation",
                "📈 Accuracy Metrics",
                "⚠️ Alerts",
                "🔄 Model Management",
            ]
        )

        with val_tab1:
            self._render_daily_validation()

        with val_tab2:
            self._render_accuracy_metrics()

        with val_tab3:
            self._render_alerts()

        with val_tab4:
            self._render_model_management()

    def _render_daily_validation(self) -> None:
        """Render daily validation section."""
        st.markdown("### 📊 Daily Forecast vs BAJUS Comparison")

        col1, col2 = st.columns([2, 1])

        with col1:
            self._render_validation_form()

        with col2:
            self._render_quick_stats()

        st.markdown("---")
        self._render_validation_history()

    def _render_validation_form(self) -> None:
        """Render validation form."""
        st.markdown("#### Validate Today's Forecast")

        purity_to_validate = st.selectbox(
            "Select Purity",
            options=["22K", "21K", "18K", "Traditional"],
            index=0,
            key="validation_purity",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            manual_forecast = st.number_input(
                "Forecasted Price (BDT)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="Enter the forecasted price (leave 0 to use logged forecast)",
            )
        with col_b:
            manual_actual = st.number_input(
                "Actual BAJUS Price (BDT)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="Enter actual BAJUS price (leave 0 to fetch live)",
            )

        if st.button("🔍 Validate Forecast", type="primary"):
            self._validate_forecast(purity_to_validate, manual_forecast, manual_actual)

    def _validate_forecast(
        self, purity: str, manual_forecast: float, manual_actual: float
    ) -> None:
        """Validate a forecast.

        Args:
            purity: Purity to validate.
            manual_forecast: Manual forecast price.
            manual_actual: Manual actual price.
        """
        with st.spinner("Validating forecast against BAJUS..."):
            result = self.validation_service.validate_forecast(
                purity=purity,
                forecasted_price=manual_forecast if manual_forecast > 0 else None,
                actual_price=manual_actual if manual_actual > 0 else None,
            )

            if result:
                self._display_validation_result(result)
            else:
                st.warning("Could not validate forecast. Check data availability.")

    def _display_validation_result(self, result: Any) -> None:
        """Display validation result.

        Args:
            result: Validation result object.
        """
        if result.within_threshold:
            st.success(
                f"✅ Forecast within threshold! "
                f"Error: {result.absolute_error:.0f} BDT "
                f"({result.percentage_error:.2f}%)"
            )
        else:
            st.error(
                f"⚠️ Forecast exceeded threshold! "
                f"Error: {result.absolute_error:.0f} BDT "
                f"({result.percentage_error:.2f}%)"
            )

        comp_col1, comp_col2, comp_col3 = st.columns(3)
        with comp_col1:
            st.metric("Forecasted", f"{result.forecasted_price:,.0f} BDT")
        with comp_col2:
            st.metric("Actual (BAJUS)", f"{result.actual_price:,.0f} BDT")
        with comp_col3:
            delta_color = "normal" if result.within_threshold else "inverse"
            st.metric(
                "Difference",
                f"{result.absolute_error:,.0f} BDT",
                delta=f"{result.percentage_error:.2f}%",
                delta_color=delta_color,
            )

    def _render_quick_stats(self) -> None:
        """Render quick stats section."""
        st.markdown("#### Quick Stats")

        try:
            scraper = self.ScraperClass()
            current_prices = scraper.get_latest_prices()

            if current_prices:
                st.markdown("**Current BAJUS Prices:**")
                for purity, price in current_prices.items():
                    st.write(f"• {purity}: {price:,.0f} BDT/g")
            else:
                st.info("Unable to fetch current BAJUS prices")
        except Exception as e:
            st.warning(f"BAJUS connection issue: {e}")

    def _render_validation_history(self) -> None:
        """Render validation history section."""
        st.markdown("### 📜 Recent Validation History")

        history_days = st.slider(
            "Days to show",
            min_value=7,
            max_value=90,
            value=30,
            key="history_days",
        )

        history_df = self.validation_service.get_validation_history(days=history_days)

        if not history_df.empty:
            display_history = history_df.copy()
            display_history["date"] = pd.to_datetime(
                display_history["date"]
            ).dt.strftime("%Y-%m-%d")
            display_history["within_threshold"] = display_history[
                "within_threshold"
            ].map({True: "✅", False: "❌"})
            display_history.columns = [
                "Date",
                "Purity",
                "Forecast (BDT)",
                "Actual (BDT)",
                "Error (BDT)",
                "Error %",
                "Within Threshold",
                "Model",
            ]
            st.dataframe(display_history, use_container_width=True)

            csv = display_history.to_csv(index=False)
            st.download_button(
                "📥 Download History",
                data=csv,
                file_name=f"validation_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No validation history available yet.")

    def _render_accuracy_metrics(self) -> None:
        """Render accuracy metrics section."""
        st.markdown("### 📈 Accuracy Metrics")

        col1, col2 = st.columns(2)

        with col1:
            metrics_period = st.selectbox(
                "Period",
                options=[7, 14, 30, 60, 90],
                index=2,
                format_func=lambda x: f"Last {x} days",
            )

        metrics = self.validation_service.get_validation_metrics(days=metrics_period)

        if metrics.total_validations > 0:
            self._display_key_metrics(metrics)
            self._display_additional_metrics(metrics)
            self._display_improvement_trend()
        else:
            st.info(
                "No validation data available yet. Start validating forecasts to see metrics."
            )

    def _display_key_metrics(self, metrics: Any) -> None:
        """Display key metrics.

        Args:
            metrics: Metrics object.
        """
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Mean Absolute Error",
                f"{metrics.mae:.0f} BDT",
                help="Average prediction error in BDT",
            )
        with col2:
            st.metric(
                "MAPE",
                f"{metrics.mape:.2f}%",
                help="Mean Absolute Percentage Error",
            )
        with col3:
            st.metric(
                "Accuracy Rate",
                f"{metrics.accuracy_rate:.1f}%",
                help="% of forecasts within ±100 BDT threshold",
            )
        with col4:
            st.metric("Total Validations", f"{metrics.total_validations}")

    def _display_additional_metrics(self, metrics: Any) -> None:
        """Display additional metrics.

        Args:
            metrics: Metrics object.
        """
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Error Statistics")
            st.write(f"• **RMSE:** {metrics.rmse:.0f} BDT")
            st.write(f"• **Max Error:** {metrics.max_error:.0f} BDT")
            st.write(f"• **Min Error:** {metrics.min_error:.0f} BDT")
            st.write(
                f"• **Within Threshold:** {metrics.within_threshold_count}/{metrics.total_validations}"
            )

        with col2:
            st.markdown("#### Error by Purity")
            if metrics.errors_by_purity:
                for purity, error in metrics.errors_by_purity.items():
                    color = "🟢" if error <= 100 else "🟡" if error <= 200 else "🔴"
                    st.write(f"{color} **{purity}:** {error:.0f} BDT MAE")

    def _display_improvement_trend(self) -> None:
        """Display improvement trend."""
        st.markdown("---")
        st.markdown("#### 📉 Improvement Trend")

        report = self.validation_service.generate_daily_report()
        if "improvement_trend" in report:
            trend = report["improvement_trend"]
            trend_type = trend.get("trend", "unknown")

            if trend_type == "improving":
                st.success(f"📈 {trend.get('message', 'Model accuracy is improving!')}")
            elif trend_type == "declining":
                st.warning(f"📉 {trend.get('message', 'Model accuracy declining.')}")
            else:
                st.info(
                    trend.get("message", "Collecting more data for trend analysis...")
                )

    def _render_alerts(self) -> None:
        """Render alerts section."""
        st.markdown("### ⚠️ Validation Alerts")

        alert_col1, alert_col2 = st.columns([3, 1])

        with alert_col2:
            show_unacknowledged = st.checkbox("Unacknowledged only", value=True)
            alert_days = st.number_input("Days", min_value=1, max_value=30, value=7)

        alerts_df = self.validation_service.get_recent_alerts(
            days=alert_days, unacknowledged_only=show_unacknowledged
        )

        if not alerts_df.empty:
            st.warning(f"⚠️ {len(alerts_df)} alert(s) found")

            for _, alert in alerts_df.iterrows():
                with st.expander(
                    f"🚨 {alert['purity']} - {alert['discrepancy_bdt']:.0f} BDT discrepancy"
                ):
                    st.write(f"**Date:** {alert['alert_date']}")
                    st.write(f"**Forecasted:** {alert['forecasted_price']:,.0f} BDT")
                    st.write(f"**Actual:** {alert['actual_price']:,.0f} BDT")
                    st.write(f"**Message:** {alert['message']}")

                    if not alert["acknowledged"]:
                        if st.button("✓ Acknowledge", key=f"ack_{alert['id']}"):
                            self.validation_service.acknowledge_alert(alert["id"])
                            st.rerun()
        else:
            st.success("✅ No alerts! All forecasts within acceptable range.")

    def _render_model_management(self) -> None:
        """Render model management section."""
        st.markdown("### 🔄 Model Management")

        model_col1, model_col2 = st.columns(2)

        with model_col1:
            self._render_active_model()

        with model_col2:
            self._render_retraining_schedule()

        st.markdown("---")
        self._render_manual_retraining()
        st.markdown("---")
        self._render_version_history()

    def _render_active_model(self) -> None:
        """Render active model info."""
        st.markdown("#### Current Active Model")

        active_version = self.retraining_service.get_active_version()

        if active_version:
            st.success(f"**Version:** {active_version.version_id}")
            st.write(f"• Created: {active_version.created_at}")
            st.write(f"• Training samples: {active_version.training_samples}")
            st.write(f"• Validation MAE: {active_version.validation_mae:.0f} BDT")
            st.write(f"• Validation MAPE: {active_version.validation_mape:.2f}%")
        else:
            st.info("No active model version. Train a model to get started.")

    def _render_retraining_schedule(self) -> None:
        """Render retraining schedule."""
        st.markdown("#### Retraining Schedule")

        schedule = self.retraining_service.get_retraining_schedule()

        if schedule.get("is_enabled"):
            st.success(f"✅ Enabled ({schedule.get('frequency', 'weekly')})")
            if schedule.get("next_run"):
                st.write(f"Next run: {schedule['next_run']}")
            if schedule.get("last_run"):
                st.write(f"Last run: {schedule['last_run']}")
        else:
            st.warning("⚠️ No schedule configured")

            if st.button("📅 Enable Weekly Retraining"):
                self.retraining_service.set_retraining_schedule(
                    frequency="weekly", enabled=True
                )
                st.rerun()

    def _render_manual_retraining(self) -> None:
        """Render manual retraining section."""
        st.markdown("#### 🚀 Manual Retraining")

        retrain_col1, retrain_col2, retrain_col3 = st.columns(3)

        with retrain_col1:
            retrain_days = st.slider(
                "Training data (days)",
                min_value=60,
                max_value=365,
                value=180,
            )
        with retrain_col2:
            retrain_purity = st.selectbox(
                "Purity",
                options=["22K", "21K", "18K"],
                index=0,
                key="retrain_purity",
            )
        with retrain_col3:
            n_trials = st.slider(
                "Optimization trials",
                min_value=10,
                max_value=100,
                value=30,
            )

        if st.button("🔄 Retrain Models Now", type="primary"):
            self._retrain_models(retrain_days, retrain_purity, n_trials)

    def _retrain_models(self, days: int, purity: str, n_trials: int) -> None:
        """Retrain models.

        Args:
            days: Training data days.
            purity: Purity to train on.
            n_trials: Number of optimization trials.
        """
        with st.spinner("Retraining models... This may take several minutes."):
            result = self.retraining_service.retrain_models(
                days=days,
                purity=purity,
                trigger_type="manual",
                n_optimization_trials=n_trials,
            )

            if result.success:
                if result.improvement > 0:
                    st.success(
                        f"✅ Retraining complete! "
                        f"MAE improved by {result.improvement:.0f} BDT "
                        f"({result.improvement_percentage:.1f}%)"
                    )
                else:
                    st.info(
                        f"ℹ️ Retraining complete. "
                        f"New MAE: {result.new_mae:.0f} BDT (no improvement)"
                    )

                st.write(f"• Training time: {result.training_time_seconds:.1f}s")
                st.write(f"• Models trained: {result.models_trained}")
                st.write(f"• Version: {result.version_id}")
            else:
                st.error(f"❌ Retraining failed: {result.message}")

    def _render_version_history(self) -> None:
        """Render version history."""
        st.markdown("#### 📜 Model Version History")

        versions = self.retraining_service.get_version_history(limit=5)

        if versions:
            for version in versions:
                status = "🟢 Active" if version.is_active else "⚪ Inactive"
                with st.expander(
                    f"{version.version_id} - MAE: {version.validation_mae:.0f} BDT {status}"
                ):
                    st.write(f"• Created: {version.created_at}")
                    st.write(f"• Training samples: {version.training_samples}")
                    st.write(
                        f"• Training period: {version.training_date_range[0]} to {version.training_date_range[1]}"
                    )
                    st.write(f"• RMSE: {version.validation_rmse:.0f} BDT")

                    if not version.is_active:
                        if st.button(
                            "🔙 Rollback to this version",
                            key=f"rollback_{version.version_id}",
                        ):
                            if self.retraining_service.rollback_to_version(
                                version.version_id
                            ):
                                st.success(f"Rolled back to {version.version_id}")
                                st.rerun()
        else:
            st.info("No model versions saved yet.")


def render_validation_tab(
    settings: dict[str, Any],
    validation_service: Any,
    retraining_service: Any,
    scraper_class: Any,
    logger: Any,
) -> None:
    """Render the validation tab.

    Args:
        settings: Dashboard settings dictionary.
        validation_service: ForecastValidationService instance.
        retraining_service: ModelRetrainingService instance.
        scraper_class: GoldPriceScraper class.
        logger: Logger instance.
    """
    view = ValidationView(
        validation_service=validation_service,
        retraining_service=retraining_service,
        scraper_class=scraper_class,
        logger=logger,
    )
    view.render(settings)
