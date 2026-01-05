"""Data models package for Gold Price Analyzer.

This package contains dataclass models representing business entities.
For ML models, see ml_models/ directory.
"""

from models.base import BaseController, BaseModel, BaseRepository, BaseService, BaseView
from models.investment import Investment, InvestmentGoal, PortfolioSummary
from models.jewelry import JewelryPrice, MakingCharges
from models.price import HistoricalPrice, Price, PricePrediction

__all__ = [
    # Base classes
    "BaseModel",
    "BaseRepository",
    "BaseService",
    "BaseController",
    "BaseView",
    # Price models
    "Price",
    "HistoricalPrice",
    "PricePrediction",
    # Investment models
    "Investment",
    "InvestmentGoal",
    "PortfolioSummary",
    # Jewelry models
    "JewelryPrice",
    "MakingCharges",
]
