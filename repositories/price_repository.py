"""Repository for gold price data access."""

from typing import Any

import duckdb

from models.base import BaseRepository
from models.price import HistoricalPrice, Price


class PriceRepository(BaseRepository):
    """Repository for accessing gold price data."""

    def get_by_id(self, id: Any) -> Price | None:
        """Get price by date and purity.

        Args:
            id: Tuple of (date, purity).

        Returns:
            Price instance or None.
        """
        if not isinstance(id, tuple) or len(id) != 2:
            return None

        date, purity = id
        with duckdb.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT * FROM prices WHERE date = ? AND purity = ? LIMIT 1",
                [date, purity],
            ).fetchone()

            if result:
                return Price(
                    date=result[0],
                    price_bdt_per_gram=result[1],
                    purity=result[2],
                    metal=result[3],
                    source=result[4],
                )
        return None

    def get_all(self, limit: int | None = None) -> list[Price]:
        """Get all prices.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of Price instances.
        """
        query = "SELECT * FROM prices ORDER BY date DESC"
        if limit:
            query += f" LIMIT {limit}"

        with duckdb.connect(self.db_path) as conn:
            results = conn.execute(query).fetchall()
            return [
                Price(
                    date=row[0],
                    price_bdt_per_gram=row[1],
                    purity=row[2],
                    metal=row[3],
                    source=row[4],
                )
                for row in results
            ]

    def get_latest_by_purity(self, purity: str) -> Price | None:
        """Get latest price for given purity.

        Args:
            purity: Gold purity (18K, 21K, 22K, 24K).

        Returns:
            Latest Price instance or None.
        """
        with duckdb.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT * FROM prices WHERE purity = ? ORDER BY date DESC LIMIT 1",
                [purity],
            ).fetchone()

            if result:
                return Price(
                    date=result[0],
                    price_bdt_per_gram=result[1],
                    purity=result[2],
                    metal=result[3],
                    source=result[4],
                )
        return None

    def get_by_date_range(
        self, start_date: str, end_date: str, purity: str | None = None
    ) -> list[Price]:
        """Get prices within date range.

        Args:
            start_date: Start date (ISO format).
            end_date: End date (ISO format).
            purity: Optional purity filter.

        Returns:
            List of Price instances.
        """
        query = "SELECT * FROM prices WHERE date BETWEEN ? AND ?"
        params = [start_date, end_date]

        if purity:
            query += " AND purity = ?"
            params.append(purity)

        query += " ORDER BY date ASC"

        with duckdb.connect(self.db_path) as conn:
            results = conn.execute(query, params).fetchall()
            return [
                Price(
                    date=row[0],
                    price_bdt_per_gram=row[1],
                    purity=row[2],
                    metal=row[3],
                    source=row[4],
                )
                for row in results
            ]

    def create(self, price: Price) -> Price:
        """Create new price record.

        Args:
            price: Price instance to create.

        Returns:
            Created Price instance.
        """
        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO prices (date, price_bdt_per_gram, purity, metal, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        price.date.isoformat()
                        if hasattr(price.date, "isoformat")
                        else price.date
                    ),
                    price.price_bdt_per_gram,
                    price.purity,
                    price.metal,
                    price.source,
                ],
            )
            conn.commit()
        return price

    def update(self, price: Price) -> bool:
        """Update existing price record.

        Args:
            price: Price instance with updated data.

        Returns:
            True if successful.
        """
        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE prices
                SET price_bdt_per_gram = ?, metal = ?, source = ?
                WHERE date = ? AND purity = ?
                """,
                [
                    price.price_bdt_per_gram,
                    price.metal,
                    price.source,
                    (
                        price.date.isoformat()
                        if hasattr(price.date, "isoformat")
                        else price.date
                    ),
                    price.purity,
                ],
            )
            conn.commit()
        return True

    def delete(self, id: Any) -> bool:
        """Delete price record.

        Args:
            id: Tuple of (date, purity).

        Returns:
            True if successful.
        """
        if not isinstance(id, tuple) or len(id) != 2:
            return False

        date, purity = id
        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM prices WHERE date = ? AND purity = ?", [date, purity]
            )
            conn.commit()
        return True


class HistoricalPriceRepository(BaseRepository):
    """Repository for accessing historical gold price data."""

    def get_by_id(self, id: Any) -> HistoricalPrice | None:
        """Get historical price by date and purity.

        Args:
            id: Tuple of (date, purity).

        Returns:
            HistoricalPrice instance or None.
        """
        if not isinstance(id, tuple) or len(id) != 2:
            return None

        date, purity = id
        with duckdb.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT * FROM historical_prices WHERE date = ? AND purity = ? LIMIT 1",
                [date, purity],
            ).fetchone()

            if result:
                return HistoricalPrice(
                    date=result[0],
                    price_bdt_per_gram=result[1],
                    purity=result[2],
                    metal=result[3],
                    source=result[4],
                )
        return None

    def get_all(self, limit: int | None = None) -> list[HistoricalPrice]:
        """Get all historical prices.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of HistoricalPrice instances.
        """
        query = "SELECT * FROM historical_prices ORDER BY date DESC"
        if limit:
            query += f" LIMIT {limit}"

        with duckdb.connect(self.db_path) as conn:
            results = conn.execute(query).fetchall()
            return [
                HistoricalPrice(
                    date=row[0],
                    price_bdt_per_gram=row[1],
                    purity=row[2],
                    metal=row[3],
                    source=row[4],
                )
                for row in results
            ]

    def create(self, price: HistoricalPrice) -> HistoricalPrice:
        """Create new historical price record.

        Args:
            price: HistoricalPrice instance to create.

        Returns:
            Created HistoricalPrice instance.
        """
        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO historical_prices (date, price_bdt_per_gram, purity, metal, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        price.date.isoformat()
                        if hasattr(price.date, "isoformat")
                        else price.date
                    ),
                    price.price_bdt_per_gram,
                    price.purity,
                    price.metal,
                    price.source,
                ],
            )
            conn.commit()
        return price

    def update(self, price: HistoricalPrice) -> bool:
        """Update existing historical price record.

        Args:
            price: HistoricalPrice instance with updated data.

        Returns:
            True if successful.
        """
        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE historical_prices
                SET price_bdt_per_gram = ?, metal = ?, source = ?
                WHERE date = ? AND purity = ?
                """,
                [
                    price.price_bdt_per_gram,
                    price.metal,
                    price.source,
                    (
                        price.date.isoformat()
                        if hasattr(price.date, "isoformat")
                        else price.date
                    ),
                    price.purity,
                ],
            )
            conn.commit()
        return True

    def delete(self, id: Any) -> bool:
        """Delete historical price record.

        Args:
            id: Tuple of (date, purity).

        Returns:
            True if successful.
        """
        if not isinstance(id, tuple) or len(id) != 2:
            return False

        date, purity = id
        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM historical_prices WHERE date = ? AND purity = ?",
                [date, purity],
            )
            conn.commit()
        return True
