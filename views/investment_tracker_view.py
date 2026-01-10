"""Investment Tracker View - Tab 6 of the dashboard.

Manages investment transactions, portfolio, goals, and history.
"""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from views.base import BaseView


class InvestmentTrackerView(BaseView):
    """View for investment tracking functionality."""

    def __init__(self, db: Any, format_price_func: Any, logger: Any):
        """Initialize the investment tracker view.

        Args:
            db: Database connection instance.
            format_price_func: Function to format price display.
            logger: Logger instance.
        """
        self.db = db
        self.format_price_bdt = format_price_func
        self.logger = logger

    def render(self, settings: dict[str, Any]) -> None:
        """Render the investment tracker tab.

        Args:
            settings: Dashboard settings dictionary.
        """
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("## 💰 Investment Tracker")

        # Create sub-tabs for different functions
        inv_tab1, inv_tab2, inv_tab3, inv_tab4 = st.tabs(
            [
                "📝 Add Transaction",
                "📊 Portfolio Summary",
                "🎯 Investment Goals",
                "📈 Transaction History",
            ]
        )

        with inv_tab1:
            self._render_add_transaction()

        with inv_tab2:
            self._render_portfolio_summary()

        with inv_tab3:
            self._render_investment_goals()

        with inv_tab4:
            self._render_transaction_history()

        st.markdown("</div>", unsafe_allow_html=True)

    def _render_add_transaction(self) -> None:
        """Render the add transaction form."""
        st.markdown("### Add New Transaction")

        col1, col2 = st.columns(2)

        with col1:
            trans_type = st.selectbox(
                "Transaction Type",
                ["Buy", "Sell"],
                help="Select whether you're buying or selling gold",
            )

            trans_date = st.date_input(
                "Transaction Date",
                value=datetime.now(),
                help="Date of the transaction",
            )

            purity = st.selectbox(
                "Gold Purity", ["22K", "21K", "18K"], help="Purity of gold"
            )

        with col2:
            quantity = st.number_input(
                "Quantity (grams)",
                min_value=0.01,
                value=1.0,
                step=0.01,
                help="Weight in grams",
            )

            price_per_gram = st.number_input(
                "Price per Gram (BDT)",
                min_value=0.0,
                value=8500.0,
                step=10.0,
                help="Price per gram in BDT",
            )

            total_amount = quantity * price_per_gram
            st.metric("Total Amount", self.format_price_bdt(total_amount, decimals=2))

        notes = st.text_area(
            "Notes (Optional)",
            placeholder="Add any notes about this transaction...",
            help="Additional details about the transaction",
        )

        if st.button("💾 Save Transaction", type="primary"):
            self._save_transaction(
                trans_date,
                trans_type,
                purity,
                quantity,
                price_per_gram,
                total_amount,
                notes,
            )

    def _save_transaction(
        self,
        trans_date: Any,
        trans_type: str,
        purity: str,
        quantity: float,
        price_per_gram: float,
        total_amount: float,
        notes: str,
    ) -> None:
        """Save a new transaction.

        Args:
            trans_date: Transaction date.
            trans_type: Transaction type (buy/sell).
            purity: Gold purity.
            quantity: Quantity in grams.
            price_per_gram: Price per gram.
            total_amount: Total transaction amount.
            notes: Transaction notes.
        """
        try:
            transaction = {
                "transaction_date": trans_date,
                "transaction_type": trans_type.lower(),
                "purity": purity,
                "quantity_grams": quantity,
                "price_per_gram": price_per_gram,
                "total_amount": total_amount,
                "notes": notes,
            }

            if self.db.insert_investment_tracking(transaction):
                st.success("✅ Transaction saved successfully!")
                st.balloons()
            else:
                st.error("❌ Failed to save transaction. Please try again.")

        except Exception as e:
            st.error(f"Error saving transaction: {str(e)}")

    def _render_portfolio_summary(self) -> None:
        """Render the portfolio summary."""
        st.markdown("### 📊 Portfolio Summary")

        try:
            conn = self.db.get_connection()
            portfolio_df = conn.execute(
                """
                SELECT
                    purity,
                    total_quantity_grams,
                    average_cost_per_gram,
                    current_value_bdt,
                    profit_loss_bdt,
                    profit_loss_percent,
                    last_updated
                FROM portfolio
                ORDER BY purity
            """
            ).fetchdf()

            if not portfolio_df.empty:
                self._display_portfolio_metrics(portfolio_df)
                self._display_portfolio_chart(portfolio_df)
                self._display_portfolio_table(portfolio_df)
            else:
                st.info("📭 No portfolio data available. Start by adding transactions!")

        except Exception as e:
            st.error(f"Error loading portfolio: {str(e)}")
            self.logger.error(f"Portfolio error: {e}")

    def _display_portfolio_metrics(self, portfolio_df: pd.DataFrame) -> None:
        """Display portfolio summary metrics.

        Args:
            portfolio_df: Portfolio data.
        """
        total_value = portfolio_df["current_value_bdt"].sum()
        total_pl = portfolio_df["profit_loss_bdt"].sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Portfolio Value",
                self.format_price_bdt(total_value, decimals=2),
            )
        with col2:
            pl_pct = (total_pl / total_value * 100) if total_value > 0 else 0
            st.metric(
                "Total P/L",
                self.format_price_bdt(total_pl, decimals=2),
                delta=f"{pl_pct:.2f}%",
            )
        with col3:
            st.metric(
                "Total Holdings",
                f"{portfolio_df['total_quantity_grams'].sum():.2f}g",
            )

    def _display_portfolio_chart(self, portfolio_df: pd.DataFrame) -> None:
        """Display portfolio distribution chart.

        Args:
            portfolio_df: Portfolio data.
        """
        st.markdown("#### Holdings by Purity")

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=portfolio_df["purity"],
                    values=portfolio_df["current_value_bdt"],
                    hole=0.3,
                    marker_colors=["#FFD700", "#C0C0C0", "#CD7F32"],
                )
            ]
        )

        fig.update_layout(title="Portfolio Distribution by Value", height=400)

        st.plotly_chart(fig, config={"responsive": True})

    def _display_portfolio_table(self, portfolio_df: pd.DataFrame) -> None:
        """Display detailed portfolio breakdown.

        Args:
            portfolio_df: Portfolio data.
        """
        st.markdown("#### Detailed Breakdown")
        display_df = portfolio_df.copy()
        display_df.columns = [
            "Purity",
            "Quantity (g)",
            "Avg Cost (BDT/g)",
            "Current Value (BDT)",
            "P/L (BDT)",
            "P/L (%)",
            "Last Updated",
        ]
        st.dataframe(display_df, use_container_width=True)

    def _render_investment_goals(self) -> None:
        """Render the investment goals section."""
        st.markdown("### 🎯 Investment Goals")

        # Add new goal
        self._render_add_goal_form()

        # Display existing goals
        self._display_existing_goals()

    def _render_add_goal_form(self) -> None:
        """Render the add goal form."""
        with st.expander("➕ Add New Goal"):
            goal_name = st.text_input("Goal Name", placeholder="e.g., Wedding Ring")

            col1, col2 = st.columns(2)
            with col1:
                target_amount = st.number_input(
                    "Target Amount (BDT)",
                    min_value=0.0,
                    value=100000.0,
                    step=1000.0,
                )
                goal_type = st.selectbox(
                    "Goal Type",
                    ["Savings", "Purchase", "Investment", "Emergency Fund"],
                )

            with col2:
                target_date = st.date_input(
                    "Target Date", value=datetime.now() + timedelta(days=365)
                )
                current_amount = st.number_input(
                    "Current Amount (BDT)", min_value=0.0, value=0.0, step=1000.0
                )

            if st.button("💾 Save Goal"):
                self._save_goal(
                    goal_name, target_amount, target_date, current_amount, goal_type
                )

    def _save_goal(
        self,
        goal_name: str,
        target_amount: float,
        target_date: Any,
        current_amount: float,
        goal_type: str,
    ) -> None:
        """Save a new investment goal.

        Args:
            goal_name: Name of the goal.
            target_amount: Target amount in BDT.
            target_date: Target date.
            current_amount: Current amount saved.
            goal_type: Type of goal.
        """
        try:
            goal = {
                "goal_name": goal_name,
                "target_amount": target_amount,
                "target_date": target_date,
                "current_amount": current_amount,
                "goal_type": goal_type.lower(),
            }

            if self.db.insert_investment_goal(goal):
                st.success("✅ Goal saved successfully!")
            else:
                st.error("❌ Failed to save goal.")
        except Exception as e:
            st.error(f"Error saving goal: {str(e)}")

    def _display_existing_goals(self) -> None:
        """Display existing investment goals."""
        try:
            conn = self.db.get_connection()
            goals_df = conn.execute(
                """
                SELECT
                    goal_name,
                    target_amount,
                    current_amount,
                    target_date,
                    goal_type,
                    status
                FROM investment_goals
                WHERE status = 'active'
                ORDER BY target_date
            """
            ).fetchdf()

            if not goals_df.empty:
                st.markdown("#### Active Goals")

                for idx, goal in goals_df.iterrows():
                    progress = (
                        (goal["current_amount"] / goal["target_amount"] * 100)
                        if goal["target_amount"] > 0
                        else 0
                    )

                    with st.container():
                        st.markdown(
                            f"**{goal['goal_name']}** ({goal['goal_type'].title()})"
                        )

                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.progress(min(progress / 100, 1.0))
                        with col2:
                            current = self.format_price_bdt(
                                goal["current_amount"], decimals=0
                            )
                            target = self.format_price_bdt(
                                goal["target_amount"], decimals=0
                            )
                            st.write(f"{current} / {target}")
                        with col3:
                            st.write(f"Target: {goal['target_date']}")

                        st.markdown("---")
            else:
                st.info("📭 No active goals. Add your first goal above!")

        except Exception as e:
            st.error(f"Error loading goals: {str(e)}")

    def _render_transaction_history(self) -> None:
        """Render the transaction history."""
        st.markdown("### 📈 Transaction History")

        try:
            conn = self.db.get_connection()
            transactions_df = conn.execute(
                """
                SELECT
                    transaction_date,
                    transaction_type,
                    purity,
                    quantity_grams,
                    price_per_gram,
                    total_amount,
                    notes
                FROM investment_tracking
                ORDER BY transaction_date DESC
                LIMIT 100
            """
            ).fetchdf()

            if not transactions_df.empty:
                filtered_df = self._apply_transaction_filters(transactions_df)
                self._display_transaction_summary(filtered_df)
                self._display_transaction_table(filtered_df)
            else:
                st.info("📭 No transactions yet. Add your first transaction!")

        except Exception as e:
            st.error(f"Error loading transactions: {str(e)}")
            self.logger.error(f"Transaction history error: {e}")

    def _apply_transaction_filters(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Apply filters to transaction data.

        Args:
            transactions_df: Raw transaction data.

        Returns:
            Filtered DataFrame.
        """
        col1, col2 = st.columns(2)
        with col1:
            filter_type = st.multiselect(
                "Filter by Type",
                options=["buy", "sell"],
                default=["buy", "sell"],
            )
        with col2:
            filter_purity = st.multiselect(
                "Filter by Purity",
                options=transactions_df["purity"].unique().tolist(),
                default=transactions_df["purity"].unique().tolist(),
            )

        return transactions_df[
            (transactions_df["transaction_type"].isin(filter_type))
            & (transactions_df["purity"].isin(filter_purity))
        ]

    def _display_transaction_summary(self, filtered_df: pd.DataFrame) -> None:
        """Display transaction summary metrics.

        Args:
            filtered_df: Filtered transaction data.
        """
        col1, col2, col3 = st.columns(3)
        with col1:
            total_bought = filtered_df[filtered_df["transaction_type"] == "buy"][
                "quantity_grams"
            ].sum()
            st.metric("Total Bought", f"{total_bought:.2f}g")
        with col2:
            total_sold = filtered_df[filtered_df["transaction_type"] == "sell"][
                "quantity_grams"
            ].sum()
            st.metric("Total Sold", f"{total_sold:.2f}g")
        with col3:
            net_position = total_bought - total_sold
            st.metric("Net Position", f"{net_position:.2f}g")

    def _display_transaction_table(self, filtered_df: pd.DataFrame) -> None:
        """Display transaction table.

        Args:
            filtered_df: Filtered transaction data.
        """
        st.markdown("#### Recent Transactions")
        display_df = filtered_df.copy()
        display_df["transaction_date"] = pd.to_datetime(
            display_df["transaction_date"]
        ).dt.strftime("%Y-%m-%d")
        display_df.columns = [
            "Date",
            "Type",
            "Purity",
            "Quantity (g)",
            "Price/g (BDT)",
            "Total (BDT)",
            "Notes",
        ]
        st.dataframe(display_df, use_container_width=True)

        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Transaction History",
            data=csv,
            file_name=f"investment_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


def render_investment_tracker_tab(
    settings: dict[str, Any],
    db: Any,
    format_price_func: Any,
    logger: Any,
) -> None:
    """Render the investment tracker tab.

    Args:
        settings: Dashboard settings dictionary.
        db: Database connection instance.
        format_price_func: Function to format price display.
        logger: Logger instance.
    """
    view = InvestmentTrackerView(
        db=db,
        format_price_func=format_price_func,
        logger=logger,
    )
    view.render(settings)
