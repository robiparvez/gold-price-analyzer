"""Data models for jewelry pricing."""

from dataclasses import dataclass

from models.base import BaseModel


@dataclass
class JewelryPrice(BaseModel):
    """Jewelry price breakdown model."""

    base_price: float
    vat_amount: float
    making_charges: float
    total_price: float
    weight_grams: float
    purity: str
    item_type: str
    vat_rate: float = 0.05

    @property
    def price_per_gram(self) -> float:
        """Calculate price per gram."""
        return self.total_price / self.weight_grams if self.weight_grams > 0 else 0.0


@dataclass
class MakingCharges(BaseModel):
    """Making charges configuration model."""

    item_type: str
    purity: str
    charge_per_gram: float
    min_charge: float | None = None
    max_charge: float | None = None
