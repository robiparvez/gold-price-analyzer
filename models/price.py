"""Data models for gold prices."""

from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime

from models.base import BaseModel


@dataclass
class Price(BaseModel):
    """Gold price data model."""

    date: date_type | str
    price_bdt_per_gram: float
    purity: str  # 18K, 21K, 22K, 24K
    metal: str = "gold"  # gold, silver, etc.
    source: str = "BAJUS"

    def __post_init__(self):
        """Convert string dates to date objects."""
        if isinstance(self.date, str):
            self.date = datetime.fromisoformat(self.date).date()


@dataclass
class HistoricalPrice(BaseModel):
    """Historical gold price data model."""

    date: date_type | str
    price_bdt_per_gram: float
    purity: str
    metal: str = "gold"
    source: str = "BAJUS"
    volume: float | None = None
    change_percent: float | None = None

    def __post_init__(self):
        """Convert string dates to date objects."""
        if isinstance(self.date, str):
            self.date = datetime.fromisoformat(self.date).date()


@dataclass
class PricePrediction(BaseModel):
    """Price prediction model."""

    date: date_type | str
    predicted_price: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.95
    model_name: str = "ensemble"
    purity: str = "22K"

    def __post_init__(self):
        """Convert string dates to date objects."""
        if isinstance(self.date, str):
            self.date = datetime.fromisoformat(self.date).date()
