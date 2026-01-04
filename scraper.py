"""
Gold Price Scraper for BAJUS (Bangladesh Jewellers Association)
Fetches live gold price data from https://www.bajus.org/gold-price
"""

import logging
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoldPriceScraper:
    """
    A comprehensive scraper for gold prices from BAJUS.
    Supports both CSV and DuckDB storage with automatic data validation.
    """

    def __init__(self, data_dir: str = "data", timeout: int = 30):
        """
        Initialize the scraper with data directory and timeout settings.

        Args:
            data_dir: Directory to store data files
            timeout: Request timeout in seconds
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.timeout = timeout
        self.base_url = "https://www.bajus.org/gold-price"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def fetch_webpage(self) -> str | None:
        """
        Fetch the HTML content from BAJUS gold price page.

        Returns:
            HTML content as string or None if failed
        """
        try:
            logger.info(f"Fetching data from {self.base_url}")
            response = requests.get(
                self.base_url, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch webpage: {e}")
            return None

    def parse_price_data(
        self, html_content: str
    ) -> dict[str, list[dict[str, str | float]]]:
        """
        Parse HTML content to extract gold price data.

        Args:
            html_content: HTML content from the webpage

        Returns:
            Dictionary containing 'gold' price list
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Initialize data structure
        price_data = {"gold": []}

        try:
            # Find gold table by CSS class
            gold_table = soup.find("table", class_="gold-table")

            # Process gold table
            if gold_table:
                self._process_metal_table(gold_table, "gold", price_data)

            logger.info(f"Parsed {len(price_data['gold'])} gold prices")

        except Exception as e:
            logger.error(f"Error parsing price data: {e}")

        return price_data

    def _process_metal_table(self, table, metal_type: str, price_data: dict) -> None:
        """Process a specific metal table and extract price data."""
        rows = table.find_all("tr")

        for row in rows[1:]:  # Skip header row
            cells = row.find_all(["td", "th"])

            if len(cells) >= 3:
                # Extract product info (karat)
                product_cell = cells[0]
                product_text = product_cell.get_text(strip=True)

                # Extract description
                description_cell = cells[1]
                description_text = description_cell.get_text(strip=True)

                # Extract price
                price_cell = cells[2]
                price_text = price_cell.get_text(strip=True)

                # Skip header rows
                if "Product" in product_text or "Price" in price_text:
                    continue

                # Extract price value
                price_clean = self._extract_price(price_text)

                if price_clean > 0:
                    purity = self._extract_purity_from_product(product_text)

                    price_entry = {
                        "purity": purity,
                        "purity_raw": product_text,
                        "description": description_text,
                        "price_bdt_per_gram": price_clean,
                        "price_raw": price_text,
                        "timestamp": datetime.now().isoformat(),
                        "date": datetime.now().strftime("%Y-%m-%d"),
                    }

                    price_data[metal_type].append(price_entry)

    def _extract_purity_from_product(self, product_text: str) -> str:
        """Extract purity information from product text."""
        product_lower = product_text.lower()
        if "22" in product_text and "karat" in product_lower:
            return "22K"
        elif "21" in product_text and "karat" in product_lower:
            return "21K"
        elif "18" in product_text and "karat" in product_lower:
            return "18K"
        elif "traditional" in product_lower or "সনাতন" in product_text:
            return "Traditional"
        else:
            return "Unknown"

    def _extract_price(self, price_text: str) -> float:
        """Extract numeric price from text."""
        try:
            # Remove common text and extract numbers
            price_clean = (
                price_text.replace("BDT/GRAM", "")
                .replace("৳", "")
                .replace(",", "")
                .strip()
            )
            # Extract numbers (handle both English and Bengali numerals)
            import re

            numbers = re.findall(r"[\d,]+", price_clean)
            if numbers:
                # Take the largest number (main price)
                number_str = max(numbers, key=len).replace(",", "")
                return float(number_str)
        except (ValueError, IndexError):
            pass
        return 0.0

    def _extract_purity(self, purity_text: str) -> str:
        """Extract purity information from text."""
        # Common purity patterns
        if "22" in purity_text:
            return "22K"
        elif "21" in purity_text:
            return "21K"
        elif "18" in purity_text:
            return "18K"
        elif "সনাতন" in purity_text or "traditional" in purity_text.lower():
            return "Traditional"
        else:
            return "Unknown"

    def save_to_csv(
        self, price_data: dict[str, list[dict]], filename: str | None = None
    ) -> str:
        """
        Save price data to CSV file.

        Args:
            price_data: Parsed price data
            filename: Optional custom filename

        Returns:
            Path to saved CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gold_prices_{timestamp}.csv"

        csv_path = self.data_dir / filename

        try:
            # Combine gold and silver data
            all_data = []
            for metal, prices in price_data.items():
                for price_entry in prices:
                    price_entry["metal"] = metal
                    all_data.append(price_entry)

            if all_data:
                df = pd.DataFrame(all_data)
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {len(all_data)} price entries to {csv_path}")
            else:
                logger.warning("No data to save to CSV")

        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")

        return str(csv_path)

    def save_to_duckdb(
        self, price_data: dict[str, list[dict]], db_name: str = "gold_prices.db"
    ) -> str:
        """
        Save price data to DuckDB database.

        Args:
            price_data: Parsed price data
            db_name: Database filename

        Returns:
            Path to database file
        """
        db_path = self.data_dir / db_name

        try:
            conn = duckdb.connect(str(db_path))

            # Create table if not exists
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prices (
                    id BIGINT,
                    metal TEXT NOT NULL,
                    purity TEXT NOT NULL,
                    purity_raw TEXT,
                    price_bdt_per_gram REAL NOT NULL,
                    price_raw TEXT,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Compute next id to support older tables without identity defaults
            max_id_row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM prices"
            ).fetchone()
            next_id = (
                max_id_row[0] if max_id_row and max_id_row[0] is not None else 0
            ) + 1

            # Insert data
            for metal, prices in price_data.items():
                for price_entry in prices:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO prices
                        (id, metal, purity, purity_raw, price_bdt_per_gram, price_raw, timestamp, date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            next_id,
                            metal,
                            price_entry["purity"],
                            price_entry["purity_raw"],
                            price_entry["price_bdt_per_gram"],
                            price_entry["price_raw"],
                            price_entry["timestamp"],
                            price_entry["date"],
                        ),
                    )
                    next_id += 1

            conn.commit()
            logger.info(f"Saved price data to DuckDB database: {db_path}")

            conn.close()

        except Exception as e:
            logger.error(f"Error saving to DuckDB: {e}")

        return str(db_path)

    def load_historical_data(
        self, source: str = "duckdb", days: int = 30
    ) -> pd.DataFrame:
        """
        Load historical price data from storage.

        Args:
            source: Data source ('duckdb' or 'csv')
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
                        SELECT * FROM prices
                        WHERE date >= current_date - INTERVAL '{days} days'
                        ORDER BY timestamp DESC
                    """
                    df = conn.execute(query).df()
                    conn.close()
                    return df

            elif source == "csv":
                # Load most recent CSV files
                csv_files = list(self.data_dir.glob("gold_prices_*.csv"))
                if csv_files:
                    # Sort by modification time and take the most recent
                    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                    return pd.read_csv(latest_csv)

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")

        return pd.DataFrame()

    def scrape_and_save(
        self, save_csv: bool = True, save_duckdb: bool = True
    ) -> tuple[dict, list[str]]:
        """
        Complete scraping workflow: fetch, parse, and save data.

        Args:
            save_csv: Whether to save to CSV
            save_duckdb: Whether to save to DuckDB

        Returns:
            Tuple of (price_data, saved_file_paths)
        """
        saved_files = []

        # Fetch webpage
        html_content = self.fetch_webpage()
        if not html_content:
            return {}, saved_files

        # Parse data
        price_data = self.parse_price_data(html_content)

        # Save data
        if save_csv:
            csv_path = self.save_to_csv(price_data)
            saved_files.append(csv_path)

        if save_duckdb:
            db_path = self.save_to_duckdb(price_data)
            saved_files.append(db_path)

        return price_data, saved_files

    def get_latest_prices(self) -> dict[str, float]:
        """
        Get the latest gold prices in a simplified format.

        Returns:
            Dictionary with latest prices by purity
        """
        price_data, _ = self.scrape_and_save(save_csv=False, save_duckdb=False)

        latest_prices = {}

        if "gold" in price_data:
            for price_entry in price_data["gold"]:
                purity = price_entry["purity"]
                price = price_entry["price_bdt_per_gram"]
                latest_prices[purity] = price

        return latest_prices
