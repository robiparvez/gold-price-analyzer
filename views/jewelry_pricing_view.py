"""Jewelry Pricing View - Tab 7 of the dashboard.

Calculator for jewelry pricing including gold, VAT, and making charges.
"""

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from views.base import BaseView


class JewelryPricingView(BaseView):
    """View for jewelry pricing calculator."""

    def __init__(
        self,
        calculator: Any,
        format_price_func: Any,
        format_breakdown_func: Any,
    ):
        """Initialize the jewelry pricing view.

        Args:
            calculator: JewelryPricingCalculator instance.
            format_price_func: Function to format price display.
            format_breakdown_func: Function to format price breakdown.
        """
        self.calculator = calculator
        self.format_price_bdt = format_price_func
        self.format_price_breakdown = format_breakdown_func

    def render(self, settings: dict[str, Any]) -> None:
        """Render the jewelry pricing tab.

        Args:
            settings: Dashboard settings dictionary.
        """
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("## 💍 Jewelry Pricing Calculator")

        st.markdown(
            """
        Calculate the total cost of jewelry including base gold price, VAT (5%), and making charges.
        Making charges vary based on the type of jewelry and gold purity.
        """
        )

        # Main calculator interface
        inputs = self._render_calculator_inputs()

        # Price comparison section
        st.markdown("---")
        self._render_price_comparison(inputs)

        st.markdown("</div>", unsafe_allow_html=True)

    def _render_calculator_inputs(self) -> dict[str, Any]:
        """Render calculator input fields.

        Returns:
            Dictionary of input values.
        """
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Input Parameters")

            base_price = st.number_input(
                "Base Gold Price (BDT/gram)",
                min_value=0.0,
                value=8500.0,
                step=10.0,
                help="Current market price of gold per gram",
            )

            weight = st.number_input(
                "Weight (grams)",
                min_value=0.01,
                value=5.0,
                step=0.01,
                help="Weight of the jewelry item",
            )

            purity = st.selectbox(
                "Gold Purity",
                ["22K", "21K", "18K"],
                help="Purity of gold in the jewelry",
            )

            item_type = st.selectbox(
                "Jewelry Type",
                [
                    "ring",
                    "necklace",
                    "bracelet",
                    "earrings",
                    "chain",
                    "bangle",
                    "other",
                ],
                help="Type of jewelry item",
            )

            # Show making charge rate
            making_rate = self.calculator.get_making_charge_rate(item_type, purity)
            st.info(
                f"💎 Making Charge for {item_type.title()} ({purity}): "
                f"{self.format_price_bdt(making_rate)}/gram"
            )

            use_custom = st.checkbox("Use Custom Making Charge")
            custom_charge = None
            if use_custom:
                custom_charge = st.number_input(
                    "Custom Making Charge (BDT/gram)",
                    min_value=0.0,
                    value=float(making_rate),
                    step=10.0,
                )

        with col2:
            st.markdown("### Price Breakdown")

            if st.button("💰 Calculate Price", type="primary"):
                self._calculate_and_display_price(
                    base_price, weight, purity, item_type, custom_charge
                )

        return {
            "base_price": base_price,
            "weight": weight,
            "purity": purity,
            "item_type": item_type,
            "custom_charge": custom_charge,
        }

    def _calculate_and_display_price(
        self,
        base_price: float,
        weight: float,
        purity: str,
        item_type: str,
        custom_charge: float | None,
    ) -> None:
        """Calculate and display jewelry price.

        Args:
            base_price: Base gold price per gram.
            weight: Weight in grams.
            purity: Gold purity.
            item_type: Type of jewelry.
            custom_charge: Custom making charge (optional).
        """
        try:
            price = self.calculator.calculate_jewelry_price(
                base_price_per_gram=base_price,
                weight_grams=weight,
                purity=purity,
                item_type=item_type,
                custom_making_charge=custom_charge,
            )

            # Display in metrics
            st.metric(
                "Base Gold Price",
                self.format_price_bdt(price.base_price, decimals=2),
            )
            st.metric("VAT (5%)", self.format_price_bdt(price.vat_amount, decimals=2))
            st.metric(
                "Making Charges",
                self.format_price_bdt(price.making_charges, decimals=2),
            )
            st.markdown("---")
            st.metric(
                "**Total Price**",
                self.format_price_bdt(price.total_price, decimals=2),
                help="Final price including all charges",
            )

            # Detailed breakdown
            with st.expander("📋 Detailed Breakdown"):
                st.code(self.format_price_breakdown(price))

        except Exception as e:
            st.error(f"Error calculating price: {str(e)}")

    def _render_price_comparison(self, inputs: dict[str, Any]) -> None:
        """Render price comparison section.

        Args:
            inputs: Input values from calculator.
        """
        st.markdown("### 📊 Compare Prices Across Jewelry Types")

        if st.button("🔍 Compare All Types"):
            self._generate_comparison(inputs)

    def _generate_comparison(self, inputs: dict[str, Any]) -> None:
        """Generate and display price comparison.

        Args:
            inputs: Input values from calculator.
        """
        try:
            comparison = self.calculator.calculate_price_comparison(
                base_price_per_gram=inputs["base_price"],
                weight_grams=inputs["weight"],
                purity=inputs["purity"],
            )

            # Create comparison dataframe
            comp_data = []
            for item, price in comparison.items():
                comp_data.append(
                    {
                        "Jewelry Type": item.title(),
                        "Making Charge/g": self.format_price_bdt(
                            price.making_charges / inputs["weight"], decimals=0
                        ),
                        "Base Price": self.format_price_bdt(
                            price.base_price, decimals=2
                        ),
                        "VAT": self.format_price_bdt(price.vat_amount, decimals=2),
                        "Making Charges": self.format_price_bdt(
                            price.making_charges, decimals=2
                        ),
                        "Total Price": self.format_price_bdt(
                            price.total_price, decimals=2
                        ),
                    }
                )

            import pandas as pd

            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True)

            # Visualization
            self._display_comparison_chart(comparison, inputs)

        except Exception as e:
            st.error(f"Error generating comparison: {str(e)}")

    def _display_comparison_chart(
        self, comparison: dict[str, Any], inputs: dict[str, Any]
    ) -> None:
        """Display comparison bar chart.

        Args:
            comparison: Price comparison dictionary.
            inputs: Input values from calculator.
        """
        fig = go.Figure(
            data=[
                go.Bar(
                    name="Base Price",
                    x=[item.title() for item in comparison.keys()],
                    y=[price.base_price for price in comparison.values()],
                    marker_color="#FFD700",
                ),
                go.Bar(
                    name="Making Charges",
                    x=[item.title() for item in comparison.keys()],
                    y=[price.making_charges for price in comparison.values()],
                    marker_color="#CD7F32",
                ),
                go.Bar(
                    name="VAT",
                    x=[item.title() for item in comparison.keys()],
                    y=[price.vat_amount for price in comparison.values()],
                    marker_color="#C0C0C0",
                ),
            ]
        )

        fig.update_layout(
            barmode="stack",
            title=f"Price Comparison for {inputs['weight']}g {inputs['purity']} Jewelry",
            xaxis_title="Jewelry Type",
            yaxis_title="Price (BDT)",
            yaxis=dict(tickformat=",.0f", ticksuffix=" BDT"),
            height=500,
        )

        st.plotly_chart(fig, config={"responsive": True})


def render_jewelry_pricing_tab(
    settings: dict[str, Any],
    calculator: Any,
    format_price_func: Any,
    format_breakdown_func: Any,
) -> None:
    """Render the jewelry pricing tab.

    Args:
        settings: Dashboard settings dictionary.
        calculator: JewelryPricingCalculator instance.
        format_price_func: Function to format price display.
        format_breakdown_func: Function to format price breakdown.
    """
    view = JewelryPricingView(
        calculator=calculator,
        format_price_func=format_price_func,
        format_breakdown_func=format_breakdown_func,
    )
    view.render(settings)
