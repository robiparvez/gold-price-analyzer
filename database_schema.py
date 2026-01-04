"""
Database Schema Management for Gold Price Analyzer
Handles table creation and schema updates for DuckDB databases
"""

import logging
from datetime import datetime
from pathlib import Path

import duckdb

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseSchema:
    """
    Manages database schema for gold price analysis system.
    Creates and maintains tables for prices, external data, tracking history, and investment goals.
    """

    def __init__(self, db_path: str = "data/gold_prices.db"):
        """
        Initialize database schema manager.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)

    def get_connection(self):
        """
        Get a DuckDB connection to the database.

        Returns:
            DuckDB connection object
        """
        return duckdb.connect(str(self.db_path))

    def create_all_tables(self) -> bool:
        """
        Create all required tables in the database.

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = duckdb.connect(str(self.db_path))

            # Create prices table
            self._create_prices_table(conn)

            # Create historical prices table
            self._create_historical_prices_table(conn)

            # Create external data table
            self._create_external_data_table(conn)

            # Create investment tracking table
            self._create_investment_tracking_table(conn)

            # Create investment goals table
            self._create_investment_goals_table(conn)

            # Create portfolio table
            self._create_portfolio_table(conn)

            conn.commit()
            conn.close()
            logger.info("All database tables created successfully")
            return True

        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            return False

    def _create_prices_table(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create live prices table."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY,
                metal TEXT NOT NULL,
                purity TEXT NOT NULL,
                purity_raw TEXT,
                price_bdt_per_gram REAL NOT NULL,
                price_raw TEXT,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, metal, purity)
            )
        """
        )
        logger.info("Created/verified prices table")

    def _create_historical_prices_table(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create historical prices table."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_prices (
                id INTEGER PRIMARY KEY,
                date DATE NOT NULL,
                metal TEXT NOT NULL DEFAULT 'gold',
                purity TEXT NOT NULL DEFAULT '22K',
                price_bdt_per_gram REAL NOT NULL,
                source TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, metal, purity, source)
            )
        """
        )
        logger.info("Created/verified historical_prices table")

    def _create_external_data_table(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create external data table for currency rates and global prices."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS external_data (
                id INTEGER PRIMARY KEY,
                fetch_date TIMESTAMP NOT NULL,
                usd_bdt_rate REAL NOT NULL,
                gold_spot_price_usd REAL NOT NULL,
                gold_price_usd_per_gram REAL NOT NULL,
                source TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(fetch_date)
            )
        """
        )
        logger.info("Created/verified external_data table")

    def _create_investment_tracking_table(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """Create investment tracking table."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS investment_tracking (
                id INTEGER PRIMARY KEY,
                transaction_date DATE NOT NULL,
                transaction_type TEXT NOT NULL,
                purity TEXT NOT NULL,
                quantity_grams REAL NOT NULL,
                price_per_gram REAL NOT NULL,
                total_amount REAL NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        logger.info("Created/verified investment_tracking table")

    def _create_investment_goals_table(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create investment goals table."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS investment_goals (
                id INTEGER PRIMARY KEY,
                goal_name TEXT NOT NULL,
                target_amount REAL NOT NULL,
                target_date DATE,
                current_amount REAL DEFAULT 0.0,
                goal_type TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        logger.info("Created/verified investment_goals table")

    def _create_portfolio_table(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create portfolio summary table."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY,
                purity TEXT NOT NULL,
                total_quantity_grams REAL DEFAULT 0.0,
                average_cost_per_gram REAL DEFAULT 0.0,
                current_value_bdt REAL DEFAULT 0.0,
                profit_loss_bdt REAL DEFAULT 0.0,
                profit_loss_percent REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(purity)
            )
        """
        )
        logger.info("Created/verified portfolio table")

    def insert_external_data(self, data: dict) -> bool:
        """
        Insert external data (currency rates, spot prices).

        Args:
            data: Dictionary with external data

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = duckdb.connect(str(self.db_path))
            conn.execute(
                """
                INSERT OR REPLACE INTO external_data
                (fetch_date, usd_bdt_rate, gold_spot_price_usd, gold_price_usd_per_gram, source)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    data.get("fetch_time", datetime.now().isoformat()),
                    data.get("usd_bdt_rate", 0.0),
                    data.get("gold_spot_price_usd", 0.0),
                    data.get("gold_price_usd_per_gram", 0.0),
                    "external_api",
                ),
            )
            conn.commit()
            conn.close()
            logger.info("Inserted external data")
            return True

        except Exception as e:
            logger.error(f"Error inserting external data: {e}")
            return False

    def insert_investment_tracking(self, transaction: dict) -> bool:
        """
        Insert investment transaction.

        Args:
            transaction: Dictionary with transaction data

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = duckdb.connect(str(self.db_path))
            conn.execute(
                """
                INSERT INTO investment_tracking
                (transaction_date, transaction_type, purity, quantity_grams,
                 price_per_gram, total_amount, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    transaction.get("transaction_date"),
                    transaction.get("transaction_type"),
                    transaction.get("purity"),
                    transaction.get("quantity_grams"),
                    transaction.get("price_per_gram"),
                    transaction.get("total_amount"),
                    transaction.get("notes"),
                ),
            )
            conn.commit()
            conn.close()
            logger.info("Inserted investment transaction")
            return True

        except Exception as e:
            logger.error(f"Error inserting investment transaction: {e}")
            return False

    def insert_investment_goal(self, goal: dict) -> bool:
        """
        Insert investment goal.

        Args:
            goal: Dictionary with goal data

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = duckdb.connect(str(self.db_path))
            conn.execute(
                """
                INSERT INTO investment_goals
                (goal_name, target_amount, target_date, goal_type)
                VALUES (?, ?, ?, ?)
            """,
                (
                    goal.get("goal_name"),
                    goal.get("target_amount"),
                    goal.get("target_date"),
                    goal.get("goal_type"),
                ),
            )
            conn.commit()
            conn.close()
            logger.info("Inserted investment goal")
            return True

        except Exception as e:
            logger.error(f"Error inserting investment goal: {e}")
            return False

    def update_portfolio(self, purity: str, portfolio_data: dict) -> bool:
        """
        Update portfolio summary.

        Args:
            purity: Gold purity level
            portfolio_data: Dictionary with portfolio data

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = duckdb.connect(str(self.db_path))
            conn.execute(
                """
                INSERT OR REPLACE INTO portfolio
                (purity, total_quantity_grams, average_cost_per_gram,
                 current_value_bdt, profit_loss_bdt, profit_loss_percent)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    purity,
                    portfolio_data.get("total_quantity_grams", 0.0),
                    portfolio_data.get("average_cost_per_gram", 0.0),
                    portfolio_data.get("current_value_bdt", 0.0),
                    portfolio_data.get("profit_loss_bdt", 0.0),
                    portfolio_data.get("profit_loss_percent", 0.0),
                ),
            )
            conn.commit()
            conn.close()
            logger.info(f"Updated portfolio for {purity}")
            return True

        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
            return False

    def get_portfolio_summary(self) -> dict:
        """
        Get complete portfolio summary.

        Returns:
            Dictionary with portfolio data
        """
        try:
            conn = duckdb.connect(str(self.db_path))
            cursor = conn.execute("SELECT * FROM portfolio")
            columns = [description[0] for description in cursor.description]

            portfolio_data = {}
            for row in cursor.fetchall():
                data = dict(zip(columns, row))
                portfolio_data[data["purity"]] = data

            conn.close()
            return portfolio_data

        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}

    def calculate_pl(self, purity: str) -> dict:
        """
        Calculate profit/loss for a specific purity.

        Args:
            purity: Gold purity level

        Returns:
            Dictionary with P/L calculations
        """
        try:
            conn = duckdb.connect(str(self.db_path))

            # Get investment transactions
            cursor = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN transaction_type='buy' THEN quantity_grams ELSE -quantity_grams END) as total_grams,
                    SUM(CASE WHEN transaction_type='buy' THEN total_amount ELSE -total_amount END) as total_spent
                FROM investment_tracking
                WHERE purity = ?
            """,
                (purity,),
            )

            result = cursor.fetchone()
            total_grams = result[0] if result and result[0] is not None else 0.0
            total_spent = result[1] if result and result[1] is not None else 0.0

            # Get current price
            cursor = conn.execute(
                """
                SELECT price_bdt_per_gram FROM prices
                WHERE purity = ? AND date = current_date
                ORDER BY timestamp DESC LIMIT 1
            """,
                (purity,),
            )

            current_price_row = cursor.fetchone()
            current_price = current_price_row[0] if current_price_row else 0.0

            current_value = total_grams * current_price
            profit_loss = current_value - total_spent
            profit_loss_pct = (profit_loss / total_spent * 100) if total_spent else 0.0

            conn.close()

            return {
                "purity": purity,
                "total_grams": total_grams,
                "average_cost": total_spent / total_grams if total_grams else 0.0,
                "current_price": current_price,
                "current_value": current_value,
                "total_investment": total_spent,
                "profit_loss": profit_loss,
                "profit_loss_percent": profit_loss_pct,
            }

        except Exception as e:
            logger.error(f"Error calculating P/L: {e}")
            return {}
