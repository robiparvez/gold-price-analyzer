"""
External Data Fetcher for Currency Rates and Global Gold Prices
Fetches USD/BDT exchange rates and global spot prices for accurate price conversions
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class ExternalDataFetcher:
    """
    Fetcher for external financial data including currency rates and global spot prices.
    Uses Currency API for exchange rates and Yahoo Finance for global gold prices.
    """

    def __init__(self, timeout: int = 30, data_dir: str = "data"):
        """
        Initialize the external data fetcher.

        Args:
            timeout: Request timeout in seconds
            data_dir: Directory to store fetched data
        """
        self.timeout = timeout
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # API endpoints
        self.currency_api_url = "https://open.er-api.com/v6/latest"  # Free currency API
        self.yahoo_finance_url = "https://query1.finance.yahoo.com/v8/finance/chart"

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    async def fetch_usd_bdt_rate(self) -> float | None:
        """
        Fetch current USD/BDT exchange rate.

        Returns:
            Exchange rate as float or None if failed
        """
        try:
            logger.info("Fetching USD/BDT exchange rate...")

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self.currency_api_url, params={"base": "USD"}, headers=self.headers
                )

                if response.status_code == 200:
                    data = response.json()
                    rate = data.get("rates", {}).get("BDT")

                    if rate:
                        logger.info(f"USD/BDT rate: {rate}")
                        return float(rate)
                else:
                    logger.warning(
                        f"Currency API returned status {response.status_code}"
                    )

        except Exception as e:
            logger.error(f"Error fetching USD/BDT rate: {e}")

        return None

    async def fetch_global_gold_spot_price(self) -> float | None:
        """
        Fetch current global gold spot price in USD per troy ounce.

        Returns:
            Gold spot price in USD or None if failed
        """
        try:
            logger.info("Fetching global gold spot price...")

            # Using Yahoo Finance GC=F (Gold Futures)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.yahoo_finance_url}/GC=F",
                    params={"interval": "1d", "range": "1d"},
                    headers=self.headers,
                )

                if response.status_code == 200:
                    data = response.json()

                    if "chart" in data and "result" in data["chart"]:
                        result = data["chart"]["result"][0]
                        if "regularMarketPrice" in result:
                            price = result["regularMarketPrice"]
                            logger.info(f"Gold spot price: ${price}/oz")
                            return float(price)

        except Exception as e:
            logger.error(f"Error fetching global gold spot price: {e}")

        return None

    async def fetch_gold_price_usd_per_gram(self) -> float | None:
        """
        Fetch current gold price converted to USD per gram.

        Returns:
            Gold price in USD per gram or None if failed
        """
        spot_price = await self.fetch_global_gold_spot_price()

        if spot_price:
            # 1 troy ounce = 31.1035 grams
            price_per_gram = spot_price / 31.1035
            logger.info(f"Gold price: ${price_per_gram:.2f}/gram")
            return price_per_gram

        return None

    async def fetch_all_external_data(self) -> dict[str, float | None]:
        """
        Fetch all external data sources concurrently.

        Returns:
            Dictionary with all fetched data
        """
        logger.info("Fetching all external data sources...")

        results = {
            "usd_bdt_rate": None,
            "gold_spot_price_usd": None,
            "gold_price_usd_per_gram": None,
            "fetch_time": datetime.now().isoformat(),
        }

        try:
            # Fetch all data concurrently
            usd_bdt, spot_price = await asyncio.gather(
                self.fetch_usd_bdt_rate(),
                self.fetch_global_gold_spot_price(),
                return_exceptions=True,
            )

            # Handle results
            if isinstance(usd_bdt, Exception):
                logger.error(f"Error fetching USD/BDT: {usd_bdt}")
            else:
                results["usd_bdt_rate"] = usd_bdt

            if isinstance(spot_price, Exception):
                logger.error(f"Error fetching spot price: {spot_price}")
            else:
                results["gold_spot_price_usd"] = spot_price

            # Calculate USD per gram
            if spot_price and not isinstance(spot_price, Exception):
                results["gold_price_usd_per_gram"] = spot_price / 31.1035  # type: ignore[operator]

            logger.info(f"Successfully fetched external data: {results}")

        except Exception as e:
            logger.error(f"Error fetching all external data: {e}")

        return results

    def save_external_data(self, data: dict) -> str:
        """
        Save fetched external data to CSV.

        Args:
            data: Dictionary with external data

        Returns:
            Path to saved CSV file
        """
        try:
            csv_path = (
                self.data_dir
                / f"external_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            df = pd.DataFrame([data])
            df.to_csv(csv_path, index=False)

            logger.info(f"Saved external data to {csv_path}")
            return str(csv_path)

        except Exception as e:
            logger.error(f"Error saving external data: {e}")
            return ""

    def get_latest_external_data(self) -> dict | None:
        """
        Load latest external data from CSV.

        Returns:
            Dictionary with latest external data or None
        """
        try:
            csv_files = list(self.data_dir.glob("external_data_*.csv"))

            if csv_files:
                latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_csv(latest_csv)

                if not df.empty:
                    return df.iloc[-1].to_dict()

        except Exception as e:
            logger.error(f"Error loading external data: {e}")

        return None

    def calculate_bdt_price_with_spot(
        self, spot_price_usd: float, usd_bdt_rate: float
    ) -> float:
        """
        Calculate BDT price per gram using USD spot price and exchange rate.

        Args:
            spot_price_usd: Gold spot price in USD per troy ounce
            usd_bdt_rate: USD to BDT exchange rate

        Returns:
            Gold price in BDT per gram
        """
        # Convert USD/oz to USD/gram
        price_usd_per_gram = spot_price_usd / 31.1035

        # Convert to BDT per gram
        price_bdt_per_gram = price_usd_per_gram * usd_bdt_rate

        return price_bdt_per_gram


async def main():
    """Example usage of the ExternalDataFetcher."""
    fetcher = ExternalDataFetcher()

    # Fetch all external data
    data = await fetcher.fetch_all_external_data()

    print("External Data Fetched:")
    for key, value in data.items():
        print(f"  {key}: {value}")

    # Save to CSV
    csv_path = fetcher.save_external_data(data)
    print(f"\nData saved to: {csv_path}")

    # Calculate BDT price
    if data["usd_bdt_rate"] and data["gold_spot_price_usd"]:
        bdt_price = fetcher.calculate_bdt_price_with_spot(
            data["gold_spot_price_usd"], data["usd_bdt_rate"]
        )
        print(f"\nCalculated BDT Price: ৳{bdt_price:,.2f}/gram")


if __name__ == "__main__":
    asyncio.run(main())
