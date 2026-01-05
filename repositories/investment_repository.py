"""Repository for investment data access."""

import duckdb

from models.base import BaseRepository
from models.investment import Investment, InvestmentGoal


class InvestmentRepository(BaseRepository):
    """Repository for accessing investment transaction data."""

    def get_by_id(self, id: int) -> Investment | None:
        """Get investment by ID.

        Args:
            id: Investment ID.

        Returns:
            Investment instance or None.
        """
        with duckdb.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT * FROM investments WHERE id = ? LIMIT 1", [id]
            ).fetchone()

            if result:
                return Investment(
                    id=result[0],
                    date=result[1],
                    weight_grams=result[2],
                    purity=result[3],
                    price_per_gram=result[4],
                    total_cost=result[5],
                    notes=result[6] if len(result) > 6 else "",
                )
        return None

    def get_all(self, limit: int | None = None) -> list[Investment]:
        """Get all investments.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of Investment instances.
        """
        query = "SELECT * FROM investments ORDER BY date DESC"
        if limit:
            query += f" LIMIT {limit}"

        with duckdb.connect(self.db_path) as conn:
            results = conn.execute(query).fetchall()
            return [
                Investment(
                    id=row[0],
                    date=row[1],
                    weight_grams=row[2],
                    purity=row[3],
                    price_per_gram=row[4],
                    total_cost=row[5],
                    notes=row[6] if len(row) > 6 else "",
                )
                for row in results
            ]

    def create(self, investment: Investment) -> Investment:
        """Create new investment record.

        Args:
            investment: Investment instance to create.

        Returns:
            Created Investment instance with ID.
        """
        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO investments (date, weight_grams, purity, price_per_gram, total_cost, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        investment.date.isoformat()
                        if hasattr(investment.date, "isoformat")
                        else investment.date
                    ),
                    investment.weight_grams,
                    investment.purity,
                    investment.price_per_gram,
                    investment.total_cost,
                    investment.notes,
                ],
            )
            conn.commit()

            # Get the last inserted ID
            result = conn.execute("SELECT lastval()").fetchone()
            if result:
                investment.id = result[0]

        return investment

    def update(self, investment: Investment) -> bool:
        """Update existing investment record.

        Args:
            investment: Investment instance with updated data.

        Returns:
            True if successful.
        """
        if investment.id is None:
            return False

        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE investments
                SET date = ?, weight_grams = ?, purity = ?, price_per_gram = ?, total_cost = ?, notes = ?
                WHERE id = ?
                """,
                [
                    (
                        investment.date.isoformat()
                        if hasattr(investment.date, "isoformat")
                        else investment.date
                    ),
                    investment.weight_grams,
                    investment.purity,
                    investment.price_per_gram,
                    investment.total_cost,
                    investment.notes,
                    investment.id,
                ],
            )
            conn.commit()
        return True

    def delete(self, id: int) -> bool:
        """Delete investment record.

        Args:
            id: Investment ID.

        Returns:
            True if successful.
        """
        with duckdb.connect(self.db_path) as conn:
            conn.execute("DELETE FROM investments WHERE id = ?", [id])
            conn.commit()
        return True

    def get_by_purity(self, purity: str) -> list[Investment]:
        """Get investments by purity.

        Args:
            purity: Gold purity.

        Returns:
            List of Investment instances.
        """
        with duckdb.connect(self.db_path) as conn:
            results = conn.execute(
                "SELECT * FROM investments WHERE purity = ? ORDER BY date DESC",
                [purity],
            ).fetchall()
            return [
                Investment(
                    id=row[0],
                    date=row[1],
                    weight_grams=row[2],
                    purity=row[3],
                    price_per_gram=row[4],
                    total_cost=row[5],
                    notes=row[6] if len(row) > 6 else "",
                )
                for row in results
            ]


class InvestmentGoalRepository(BaseRepository):
    """Repository for accessing investment goal data."""

    def get_by_id(self, id: int) -> InvestmentGoal | None:
        """Get investment goal by ID.

        Args:
            id: Goal ID.

        Returns:
            InvestmentGoal instance or None.
        """
        with duckdb.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT * FROM investment_goals WHERE id = ? LIMIT 1", [id]
            ).fetchone()

            if result:
                return InvestmentGoal(
                    id=result[0],
                    target_weight_grams=result[1],
                    target_date=result[2],
                    description=result[3],
                    completed=bool(result[4]),
                )
        return None

    def get_all(self, limit: int | None = None) -> list[InvestmentGoal]:
        """Get all investment goals.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of InvestmentGoal instances.
        """
        query = "SELECT * FROM investment_goals ORDER BY target_date ASC"
        if limit:
            query += f" LIMIT {limit}"

        with duckdb.connect(self.db_path) as conn:
            results = conn.execute(query).fetchall()
            return [
                InvestmentGoal(
                    id=row[0],
                    target_weight_grams=row[1],
                    target_date=row[2],
                    description=row[3],
                    completed=bool(row[4]),
                )
                for row in results
            ]

    def create(self, goal: InvestmentGoal) -> InvestmentGoal:
        """Create new investment goal.

        Args:
            goal: InvestmentGoal instance to create.

        Returns:
            Created InvestmentGoal instance with ID.
        """
        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO investment_goals (target_weight_grams, target_date, description, completed)
                VALUES (?, ?, ?, ?)
                """,
                [
                    goal.target_weight_grams,
                    (
                        goal.target_date.isoformat()
                        if hasattr(goal.target_date, "isoformat")
                        else goal.target_date
                    ),
                    goal.description,
                    goal.completed,
                ],
            )
            conn.commit()

            # Get the last inserted ID
            result = conn.execute("SELECT lastval()").fetchone()
            if result:
                goal.id = result[0]

        return goal

    def update(self, goal: InvestmentGoal) -> bool:
        """Update existing investment goal.

        Args:
            goal: InvestmentGoal instance with updated data.

        Returns:
            True if successful.
        """
        if goal.id is None:
            return False

        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE investment_goals
                SET target_weight_grams = ?, target_date = ?, description = ?, completed = ?
                WHERE id = ?
                """,
                [
                    goal.target_weight_grams,
                    (
                        goal.target_date.isoformat()
                        if hasattr(goal.target_date, "isoformat")
                        else goal.target_date
                    ),
                    goal.description,
                    goal.completed,
                    goal.id,
                ],
            )
            conn.commit()
        return True

    def delete(self, id: int) -> bool:
        """Delete investment goal.

        Args:
            id: Goal ID.

        Returns:
            True if successful.
        """
        with duckdb.connect(self.db_path) as conn:
            conn.execute("DELETE FROM investment_goals WHERE id = ?", [id])
            conn.commit()
        return True
