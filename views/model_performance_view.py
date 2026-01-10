"""Model Performance View - Tab 4 of the dashboard.

Displays model accuracy metrics, feature importance, and model comparisons.
"""

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from views.base import BaseView


class ModelPerformanceView(BaseView):
    """View for model performance evaluation."""

    def __init__(
        self,
        evaluate_func: Any,
        format_price_func: Any,
    ):
        """Initialize the model performance view.

        Args:
            evaluate_func: Function to evaluate model performance.
            format_price_func: Function to format price display.
        """
        self.evaluate_model_performance = evaluate_func
        self.format_price_bdt = format_price_func

    def render(self, settings: dict[str, Any]) -> None:
        """Render the model performance tab.

        Args:
            settings: Dashboard settings dictionary.
        """
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        if not settings["enable_forecast"]:
            st.info("🎯 Enable forecasting to view model performance metrics")
        else:
            self._render_performance_content(settings)

        st.markdown("</div>", unsafe_allow_html=True)

    def _render_performance_content(self, settings: dict[str, Any]) -> None:
        """Render the performance evaluation content.

        Args:
            settings: Dashboard settings dictionary.
        """
        with st.spinner("Evaluating model performance..."):
            try:
                evaluation_results = self.evaluate_model_performance(settings["purity"])

                if "error" in evaluation_results:
                    st.error(f"Evaluation error: {evaluation_results['error']}")
                elif "models" in evaluation_results:
                    self._display_model_metrics(evaluation_results)
                    self._display_model_comparison(evaluation_results)
                else:
                    st.warning("No model evaluation results available")

            except Exception as e:
                st.error(f"Error evaluating models: {str(e)}")

    def _display_model_metrics(self, evaluation_results: dict[str, Any]) -> None:
        """Display metrics for each model.

        Args:
            evaluation_results: Model evaluation results dictionary.
        """
        st.markdown("### 🎯 Model Accuracy Assessment")

        for model_name, metrics in evaluation_results["models"].items():
            if "error" not in metrics:
                self._display_single_model_metrics(model_name, metrics)

    def _display_single_model_metrics(
        self, model_name: str, metrics: dict[str, Any]
    ) -> None:
        """Display metrics for a single model.

        Args:
            model_name: Name of the model.
            metrics: Model metrics dictionary.
        """
        st.markdown(f"#### {model_name.title()} Model")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            accuracy = metrics.get("accuracy_percentage", 0)
            st.metric("Accuracy", f"{accuracy:.1f}%")

        with col2:
            mape = metrics.get("mape", 0)
            st.metric("MAPE", f"{mape:.2f}%")

        with col3:
            mae = metrics.get("mae", 0)
            st.metric("MAE", self.format_price_bdt(mae, decimals=0))

        with col4:
            r2 = metrics.get("r2_score", 0)
            st.metric("R² Score", f"{r2:.3f}")

        # Feature importance for Random Forest
        if model_name == "random_forest" and "feature_importance" in metrics:
            self._display_feature_importance(model_name, metrics)

    def _display_feature_importance(
        self, model_name: str, metrics: dict[str, Any]
    ) -> None:
        """Display feature importance chart.

        Args:
            model_name: Name of the model.
            metrics: Model metrics dictionary.
        """
        with st.expander(f"📊 {model_name.title()} Feature Importance"):
            importance_df = pd.DataFrame(
                [
                    {"Feature": k, "Importance": v}
                    for k, v in metrics["feature_importance"].items()
                ]
            ).sort_values("Importance", ascending=False)

            chart = (
                alt.Chart(importance_df)
                .mark_bar()
                .encode(
                    x=alt.X("Importance:Q", title="Importance"),
                    y=alt.Y("Feature:N", sort="-x", title="Feature"),
                )
                .properties(
                    title=f"{model_name.title()} Feature Importance",
                    height=400,
                )
            )
            st.altair_chart(chart, use_container_width=True)

    def _display_model_comparison(self, evaluation_results: dict[str, Any]) -> None:
        """Display model comparison table.

        Args:
            evaluation_results: Model evaluation results dictionary.
        """
        if len(evaluation_results["models"]) <= 1:
            return

        st.markdown("### 📊 Model Comparison")

        comparison_data = []
        for model_name, metrics in evaluation_results["models"].items():
            if "error" not in metrics:
                comparison_data.append(
                    {
                        "Model": model_name.title(),
                        "Accuracy (%)": f"{metrics.get('accuracy_percentage', 0):.1f}",
                        "MAPE (%)": f"{metrics.get('mape', 0):.2f}",
                        "MAE (BDT)": self.format_price_bdt(
                            metrics.get("mae", 0), decimals=0
                        ),
                        "R² Score": f"{metrics.get('r2_score', 0):.3f}",
                        "Test Samples": metrics.get("test_samples", 0),
                    }
                )

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)


def render_model_performance_tab(
    settings: dict[str, Any],
    evaluate_func: Any,
    format_price_func: Any,
) -> None:
    """Render the model performance tab.

    Args:
        settings: Dashboard settings dictionary.
        evaluate_func: Function to evaluate model performance.
        format_price_func: Function to format price display.
    """
    view = ModelPerformanceView(
        evaluate_func=evaluate_func,
        format_price_func=format_price_func,
    )
    view.render(settings)
