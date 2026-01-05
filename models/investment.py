"""Data models for investments and portfolio tracking."""

from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime

from models.base import BaseModel


@dataclass
class Investment(BaseModel):
    """Investment transaction model."""

    id: int | None = None
    date: date_type | str = None
    weight_grams: float = 0.0
    purity: str = "22K"
    price_per_gram: float = 0.0
    total_cost: float = 0.0
    notes: str = ""

    def __post_init__(self):
        """Convert string dates to date objects and calculate total cost."""
        if isinstance(self.date, str):
            self.date = datetime.fromisoformat(self.date).date()
        elif self.date is None:
            self.date = datetime.now().date()

        # Auto-calculate total cost if not provided
        if self.total_cost == 0.0 and self.weight_grams > 0 and self.price_per_gram > 0:
            self.total_cost = self.weight_grams * self.price_per_gram


@dataclass
class InvestmentGoal(BaseModel):
    """Investment goal model."""

    id: int | None = None
    target_weight_grams: float = 0.0
    target_date: date_type | str = None
    description: str = ""
    completed: bool = False

    def __post_init__(self):
        """Convert string dates to date objects."""
        if isinstance(self.target_date, str):
            self.target_date = datetime.fromisoformat(self.target_date).date()


@dataclass
class PortfolioSummary(BaseModel):
    """Portfolio summary statistics."""

    total_weight_grams: float
    total_investment_bdt: float
    current_value_bdt: float
    profit_loss_bdt: float
    profit_loss_percent: float
    average_buy_price: float
    current_price: float
    number_of_transactions: int
