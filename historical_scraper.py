"""
Historical Gold Price Scraper for Bangladesh Sources
Extends the existing scraper with historical data fetching capabilities
"""

import logging
from datetime import datetime, timedelta

import duckdb
import httpx
import pandas as pd

from scraper import GoldPriceScraper

# Configure logging
logger = logging.getLogger(__name__)


class HistoricalGoldPriceScraper(GoldPriceScraper):
    """
    Extended scraper that fetches historical gold price data from multiple Bangladesh sources.
    Supports Sakib.dev API as primary source.
    """

    def __init__(self, data_dir: str = "data", timeout: int = 30):
        """
        Initialize the historical scraper.

        Args:
            data_dir: Directory to store data files
            timeout: Request timeout in seconds
        """
        super().__init__(data_dir, timeout)

        # Historical data sources
        self.historical_sources = {
            "sakib_dev": {
                "url": "https://gold-price.sakib.dev/api/gold-price",
                "type": "api",
                "description": "Sakib.dev Gold Price API",
            }
        }

    async def fetch_sakib_dev_data(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch historical data from Sakib.dev API.

        Args:
            days: Number of days of historical data to fetch

        Returns:
            DataFrame with historical price data
        """
        try:
            logger.info("Fetching historical data from Sakib.dev API...")

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Try to fetch data from Sakib.dev API
                response = await client.get(
                    self.historical_sources["sakib_dev"]["url"],
                    headers=self.headers,
                    params={
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "format": "json",
                    },
                )

                if response.status_code == 200:
                    # Check if response has content
                    if not response.text or response.text.strip() == "":
                        logger.warning("Sakib.dev API returned empty response")
                        return pd.DataFrame()

                    try:
                        data = response.json()
                    except ValueError:
                        logger.warning("Sakib.dev API returned invalid JSON")
                        return pd.DataFrame()

                    # Convert to DataFrame
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict) and "data" in data:
                        df = pd.DataFrame(data["data"])
                    else:
                        logger.warning("Unexpected API response format from Sakib.dev")
                        return pd.DataFrame()

                    # Standardize column names
                    df = self._standardize_historical_data(df, source="sakib_dev")
                    logger.info(
                        f"Successfully fetched {len(df)} records from Sakib.dev"
                    )
                    return df
                else:
                    logger.warning(
                        f"Sakib.dev API returned status {response.status_code}"
                    )

        except Exception as e:
            logger.error(f"Error fetching Sakib.dev data: {e}")

        return pd.DataFrame()

    def generate_synthetic_historical_data(self, days: int = 365) -> pd.DataFrame:
        """
        Generate synthetic historical data based on current prices and market trends.
        This is a fallback when external sources are unavailable.

        Args:
            days: Number of days of historical data to generate

        Returns:
            DataFrame with synthetic historical price data
        """
        logger.info("Generating synthetic historical data as fallback...")

        try:
            # Get current prices as baseline
            current_data, _ = self.scrape_and_save(save_csv=False, save_duckdb=False)

            if not current_data.get("gold"):
                logger.warning(
                    "No current gold price data available for synthetic generation"
                )
                return pd.DataFrame()

            # Use 22K gold price as baseline
            baseline_price = None
            for gold_entry in current_data["gold"]:
                if gold_entry["purity"] == "22K":
                    baseline_price = gold_entry["price_bdt_per_gram"]
                    break

            if baseline_price is None:
                baseline_price = current_data["gold"][0]["price_bdt_per_gram"]

            # Generate synthetic data
            historical_data = []
            current_date = datetime.now()

            for i in range(days):
                date = current_date - timedelta(days=i)

                # Add some realistic variation (±2% daily volatility)
                import random

                daily_change = random.uniform(-0.02, 0.02)
                trend_factor = 1 + (
                    i * random.uniform(-0.0001, 0.0001)
                )  # Long-term trend

                price = baseline_price * trend_factor * (1 + daily_change)

                historical_data.append(
                    {
                        "date": date.date(),
                        "price_bdt_per_gram": round(price, 2),
                        "source": "synthetic",
                        "purity": "22K",
                        "metal": "gold",
                    }
                )

            df = pd.DataFrame(historical_data)
            df = df.sort_values("date").reset_index(drop=True)

            logger.info(f"Generated {len(df)} synthetic historical records")
            return df

        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return pd.DataFrame()

    def _parse_date(self, date_text: str) -> datetime | None:
        """Parse various date formats into datetime object."""
        date_formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%Y-%m-%d %H:%M:%S",
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_text.strip(), fmt).date()  # type: ignore[return-value]
            except (ValueError, AttributeError):
                pass

        return None

    def _standardize_historical_data(
        self, df: pd.DataFrame, source: str
    ) -> pd.DataFrame:
        """
        Standardize historical data format across different sources.

        Args:
            df: Raw DataFrame from source
            source: Source identifier

        Returns:
            Standardized DataFrame
        """
        if df.empty:
            return df

        # Ensure required columns exist
        # Validate data structure
        _ = ["date", "price_bdt_per_gram"]  # For reference

        # Map common column variations
        column_mapping = {
            "Date": "date",
            "date": "date",
            "Date Time": "date",
            "Price": "price_bdt_per_gram",
            "price": "price_bdt_per_gram",
            "Gold Price": "price_bdt_per_gram",
            "price_per_gram": "price_bdt_per_gram",
        }

        # Apply column mapping
        df = df.rename(columns=column_mapping)

        # Add missing columns with defaults
        if "source" not in df.columns:
            df["source"] = source
        if "purity" not in df.columns:
            df["purity"] = "22K"
        if "metal" not in df.columns:
            df["metal"] = "gold"

        # Convert date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date

        # Clean price data
        if "price_bdt_per_gram" in df.columns:
            df["price_bdt_per_gram"] = pd.to_numeric(
                df["price_bdt_per_gram"], errors="coerce"
            )
            df = df.dropna(subset=["price_bdt_per_gram"])

        # Remove duplicates
        df = df.drop_duplicates(subset=["date", "purity", "metal"])

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        return df

    def detect_and_fill_gaps(
        self, df: pd.DataFrame, gap_threshold: int = 3
    ) -> pd.DataFrame:
        """
        Detect data gaps and fill them with interpolation.
        Implements hybrid sync engine logic for gap detection.

        Args:
            df: DataFrame with historical data
            gap_threshold: Number of days to consider as a gap

        Returns:
            DataFrame with gaps filled
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for gap detection")
            return df

        try:
            logger.info("Detecting and filling data gaps...")

            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])

            # Group by purity and metal to detect gaps for each
            filled_dfs = []

            for (purity, metal), group in df.groupby(["purity", "metal"]):
                group = group.sort_values("date").reset_index(drop=True)

                # Detect gaps
                date_diffs = group["date"].diff().dt.days  # type: ignore[attr-defined]
                gaps = date_diffs[date_diffs > gap_threshold]

                if not gaps.empty:
                    logger.info(f"Detected {len(gaps)} gaps for {metal} ({purity})")

                    # Fill gaps using linear interpolation
                    date_range = pd.date_range(
                        start=group["date"].min(), end=group["date"].max(), freq="D"
                    )

                    group_reindexed = group.set_index("date").reindex(date_range)
                    group_reindexed["price_bdt_per_gram"] = group_reindexed[
                        "price_bdt_per_gram"
                    ].interpolate(method="linear")

                    # Restore purity and metal columns
                    group_reindexed["purity"] = purity
                    group_reindexed["metal"] = metal
                    group_reindexed = group_reindexed.reset_index()
                    group_reindexed = group_reindexed.rename(columns={"index": "date"})

                    filled_dfs.append(group_reindexed)
                else:
                    filled_dfs.append(group)

            if filled_dfs:
                result_df = pd.concat(filled_dfs, ignore_index=True)
                result_df = result_df.sort_values("date").reset_index(drop=True)
                logger.info(f"Filled {len(result_df)} total records after gap filling")
                return result_df

            return df

        except Exception as e:
            logger.error(f"Error detecting and filling gaps: {e}")
            return df

    async def sync_sakib_dev_daily(self) -> bool:
        """
        Perform daily sync with Sakib.dev API and auto-fill gaps.
        Part of the hybrid sync engine.

        Returns:
            True if sync successful, False otherwise
        """
        try:
            logger.info("Starting daily Sakib.dev sync with auto-fill...")

            # Fetch latest data
            new_data = await self.fetch_sakib_dev_data(days=30)

            if new_data.empty:
                logger.warning("No new data from Sakib.dev")
                return False

            # Load existing data
            existing_data = self.load_historical_data(source="csv", days=90)

            if not existing_data.empty:
                # Combine with existing data
                combined = pd.concat([existing_data, new_data], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["date", "purity", "metal"], keep="last"
                )
            else:
                combined = new_data

            # Detect and fill gaps
            filled_data = self.detect_and_fill_gaps(combined)

            # Save updated data
            self.save_historical_data(filled_data)

            logger.info("Daily sync and auto-fill complete")
            return True

        except Exception as e:
            logger.error(f"Error in daily sync: {e}")
            return False

    async def fetch_all_historical_data(
        self, days: int = 365, use_synthetic: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data from all available sources and combine them.

        Args:
            days: Number of days of historical data to fetch
            use_synthetic: Whether to generate synthetic data as fallback

        Returns:
            Combined DataFrame with historical data
        """
        logger.info("Fetching historical data from all sources...")

        all_data = []

        # Try Sakib.dev API first
        try:
            sakib_data = await self.fetch_sakib_dev_data(days)
            if not sakib_data.empty:
                all_data.append(sakib_data)
        except Exception as e:
            logger.error(f"Failed to fetch Sakib.dev data: {e}")

        # Combine data from all sources
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(
                subset=["date", "purity", "metal"]
            )
            combined_df = combined_df.sort_values("date").reset_index(drop=True)

            # Apply gap detection and auto-fill (hybrid sync engine)
            combined_df = self.detect_and_fill_gaps(combined_df)

            logger.info(
                f"Combined {len(combined_df)} historical records from {len(all_data)} sources with gap filling"
            )
        else:
            combined_df = pd.DataFrame()

        # Use synthetic data if no real data available and allowed
        if combined_df.empty and use_synthetic:
            logger.info(
                "No historical data available from external sources, generating synthetic data as fallback"
            )
            combined_df = self.generate_synthetic_historical_data(days)

        return combined_df

    def load_historical_data(
        self, source: str = "csv", days: int = 365
    ) -> pd.DataFrame:
        """
        Load historical data from storage.

        Args:
            source: Data source ('csv' or 'duckdb')
            days: Number of days to load

        Returns:
            DataFrame with historical data
        """
        try:
            if source == "duckdb":
                db_path = self.data_dir / "gold_prices.db"
                if db_path.exists():
                    conn = duckdb.connect(str(db_path))
                    query = f"""
                        SELECT date, metal, purity, price_bdt_per_gram, source
                        FROM historical_prices
                        WHERE date >= current_date - INTERVAL '{days} days'
                        ORDER BY date ASC
                    """
                    df = conn.execute(query).df()
                    conn.close()
                    if not df.empty:
                        df["date"] = pd.to_datetime(df["date"])
                        return df

            if source == "csv":
                csv_files = list(self.data_dir.glob("historical_gold_prices_*.csv"))
                if not csv_files:
                    csv_files = list(self.data_dir.glob("historical_gold_prices.csv"))

                if csv_files:
                    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                    df = pd.read_csv(latest_csv)

                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        cutoff_date = datetime.now() - timedelta(days=days)
                        df = df[df["date"] >= cutoff_date]

                    return df

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")

        return pd.DataFrame()

    def save_historical_data(
        self, df: pd.DataFrame, filename: str = "historical_gold_prices.csv"
    ) -> str:
        """
        Save historical data to CSV and DuckDB.

        Args:
            df: Historical data DataFrame
            filename: CSV filename

        Returns:
            Path to saved CSV file
        """
        if df.empty:
            logger.warning("No historical data to save")
            return ""

        # Save to CSV
        csv_path = self.data_dir / filename
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} historical records to {csv_path}")

        # Save to DuckDB
        try:
            db_path = self.data_dir / "gold_prices.db"
            conn = duckdb.connect(str(db_path))

            # Create historical prices table if not exists
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_prices (
                    id INTEGER PRIMARY KEY,
                    date DATE NOT NULL,
                    metal TEXT NOT NULL,
                    purity TEXT NOT NULL,
                    price_bdt_per_gram REAL NOT NULL,
                    source TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, metal, purity, source)
                )
            """
            )

            # Insert historical data
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO historical_prices
                        (date, metal, purity, price_bdt_per_gram, source)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            row["date"],
                            row.get("metal", "gold"),
                            row.get("purity", "22K"),
                            row["price_bdt_per_gram"],
                            row.get("source", "unknown"),
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Failed to insert row: {e}")
                    continue

            conn.commit()
            conn.close()
            logger.info(f"Saved historical data to DuckDB database: {db_path}")

        except Exception as e:
            logger.error(f"Error saving to DuckDB: {e}")

        return str(csv_path)
