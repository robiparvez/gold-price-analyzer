"""Base classes for MVC architecture."""

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BaseModel:
    """Base class for all data models.

    All data models should inherit from this class and use @dataclass decorator.
    Provides common serialization methods.
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary.

        Returns:
            Dictionary representation of the model.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseModel":
        """Create model instance from dictionary.

        Args:
            data: Dictionary containing model data.

        Returns:
            New model instance.
        """
        return cls(**data)


class BaseRepository(ABC):
    """Base class for all repository classes.

    Repositories handle data access and persistence operations.
    Each repository should manage a single entity type.
    """

    def __init__(self, db_path: str = "data/gold_prices.db"):
        """Initialize repository with database connection.

        Args:
            db_path: Path to DuckDB database file.
        """
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get_by_id(self, id: Any) -> Any:
        """Get entity by ID.

        Args:
            id: Entity identifier.

        Returns:
            Entity instance or None if not found.
        """
        pass

    @abstractmethod
    def get_all(self, limit: int | None = None) -> list[Any]:
        """Get all entities.

        Args:
            limit: Maximum number of entities to return.

        Returns:
            List of entity instances.
        """
        pass

    @abstractmethod
    def create(self, entity: Any) -> Any:
        """Create new entity.

        Args:
            entity: Entity instance to create.

        Returns:
            Created entity with ID.
        """
        pass

    @abstractmethod
    def update(self, entity: Any) -> bool:
        """Update existing entity.

        Args:
            entity: Entity instance with updated data.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def delete(self, id: Any) -> bool:
        """Delete entity by ID.

        Args:
            id: Entity identifier.

        Returns:
            True if successful, False otherwise.
        """
        pass


class BaseService(ABC):
    """Base class for all service classes.

    Services contain business logic and orchestrate operations
    across multiple repositories.
    """

    def __init__(self):
        """Initialize service."""
        self.logger = logging.getLogger(self.__class__.__name__)


class BaseController(ABC):
    """Base class for all controller classes.

    Controllers handle request/response logic and validation.
    They delegate business logic to services.
    """

    def __init__(self, service: BaseService):
        """Initialize controller with service.

        Args:
            service: Service instance to handle business logic.
        """
        self.service = service
        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_request(self, request: dict[str, Any]) -> tuple[bool, str]:
        """Validate incoming request data.

        Args:
            request: Request data dictionary.

        Returns:
            Tuple of (is_valid, error_message).
        """
        return True, ""

    def _format_response(
        self, data: Any, success: bool = True, error: str | None = None
    ) -> dict[str, Any]:
        """Format response data.

        Args:
            data: Response data.
            success: Whether operation was successful.
            error: Error message if unsuccessful.

        Returns:
            Formatted response dictionary.
        """
        return {
            "success": success,
            "data": data if success else None,
            "error": error if not success else None,
        }


class BaseView(ABC):
    """Base class for all view classes.

    Views handle UI presentation using Streamlit.
    They call controllers to get data and render it.
    """

    def __init__(self, controller: BaseController):
        """Initialize view with controller.

        Args:
            controller: Controller instance to handle requests.
        """
        self.controller = controller
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def render(self) -> None:
        """Render the view.

        This method should contain the Streamlit UI components.
        """
        pass
