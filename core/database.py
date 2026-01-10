"""Database connection management for DuckDB.

This module provides a centralized connection manager to eliminate
repeated connection patterns across the codebase.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from core.constants import DATABASE_PATH
from core.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class DatabaseConnectionManager:
    """Centralized database connection manager for DuckDB.

    Provides thread-safe connection handling with automatic
    resource cleanup via context managers.

    Example:
        >>> db_manager = DatabaseConnectionManager()
        >>> with db_manager.get_connection() as conn:
        ...     result = conn.execute("SELECT * FROM prices").fetchall()
    """

    def __init__(self, db_path: str | Path = DATABASE_PATH):
        """Initialize the connection manager.

        Args:
            db_path: Path to DuckDB database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Get a database connection as a context manager.

        Yields:
            DuckDB connection object.

        Raises:
            DatabaseError: If connection cannot be established.

        Example:
            >>> with db_manager.get_connection() as conn:
            ...     conn.execute("SELECT 1").fetchone()
        """
        conn = None
        try:
            conn = duckdb.connect(str(self.db_path))
            yield conn
        except duckdb.Error as e:
            logger.error(f"Database connection error: {e}")
            raise DatabaseError(f"Failed to connect to database: {e}") from e
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")

    @contextmanager
    def transaction(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Get a database connection with transaction support.

        Automatically commits on success, rolls back on exception.

        Yields:
            DuckDB connection object.

        Raises:
            DatabaseError: If transaction fails.
        """
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                # DuckDB auto-rollbacks on connection close without commit
                logger.error(f"Transaction failed, rolling back: {e}")
                raise

    def execute_query(self, query: str, params: tuple | list | None = None) -> list:
        """Execute a query and return results.

        Args:
            query: SQL query string.
            params: Query parameters.

        Returns:
            List of result rows.

        Raises:
            DatabaseError: If query execution fails.
        """
        try:
            with self.get_connection() as conn:
                if params:
                    result = conn.execute(query, params).fetchall()
                else:
                    result = conn.execute(query).fetchall()
                return result
        except duckdb.Error as e:
            logger.error(f"Query execution error: {e}")
            raise DatabaseError(f"Query failed: {e}") from e

    def execute_many(self, query: str, params_list: list[tuple | list]) -> int:
        """Execute a query with multiple parameter sets (batch insert).

        Args:
            query: SQL query string with placeholders.
            params_list: List of parameter tuples.

        Returns:
            Number of rows affected.

        Raises:
            DatabaseError: If batch execution fails.
        """
        try:
            with self.transaction() as conn:
                conn.executemany(query, params_list)
                return len(params_list)
        except duckdb.Error as e:
            logger.error(f"Batch execution error: {e}")
            raise DatabaseError(f"Batch operation failed: {e}") from e

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of the table to check.

        Returns:
            True if table exists, False otherwise.
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(
                    "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
                    [table_name],
                ).fetchone()
                return result is not None
        except duckdb.Error:
            return False


# Global singleton instance
_db_manager: DatabaseConnectionManager | None = None


def get_db_manager(db_path: str | Path = DATABASE_PATH) -> DatabaseConnectionManager:
    """Get the global database connection manager instance.

    Args:
        db_path: Path to database file.

    Returns:
        DatabaseConnectionManager instance.
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseConnectionManager(db_path)
    return _db_manager
