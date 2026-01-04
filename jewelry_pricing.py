"""
Jewelry pricing calculator with VAT and making charges.

This module provides functionality to calculate final jewelry prices
including base gold price, VAT, and making charges.
"""

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class JewelryPrice:
    """Container for jewelry price breakdown."""

    base_price: float
    vat_amount: float
    making_charges: float
    total_price: float
    weight_grams: float
    purity: str
    item_type: str


class JewelryPricingCalculator:
    """Calculate jewelry prices with VAT and making charges."""

    # Standard VAT rate in Bangladesh
    DEFAULT_VAT_RATE = 0.05  # 5%

    # Making charges by jewelry type (per gram)
    MAKING_CHARGES = {
        "ring": {"22k": 800, "21k": 750, "18k": 600},
        "necklace": {"22k": 1200, "21k": 1100, "18k": 900},
        "bracelet": {"22k": 900, "21k": 850, "18k": 700},
        "earrings": {"22k": 700, "21k": 650, "18k": 550},
        "chain": {"22k": 1000, "21k": 950, "18k": 800},
        "bangle": {"22k": 850, "21k": 800, "18k": 650},
        "other": {"22k": 800, "21k": 750, "18k": 600},
    }

    def __init__(
        self,
        vat_rate: float = DEFAULT_VAT_RATE,
        custom_making_charges: dict[str, dict[str, float]] | None = None,
    ):
        """
        Initialize pricing calculator.

        Args:
            vat_rate: VAT percentage (default 5% = 0.05)
            custom_making_charges: Optional custom making charges per jewelry type
        """
        self.vat_rate = vat_rate
        self.making_charges = custom_making_charges or self.MAKING_CHARGES
        logger.info(
            f"Initialized JewelryPricingCalculator with VAT rate: {vat_rate * 100}%"
        )

    def calculate_jewelry_price(
        self,
        base_price_per_gram: float,
        weight_grams: float,
        purity: Literal["22k", "21k", "18k"] = "22k",
        item_type: Literal[
            "ring", "necklace", "bracelet", "earrings", "chain", "bangle", "other"
        ] = "other",
        custom_making_charge: float | None = None,
    ) -> JewelryPrice:
        """
        Calculate total jewelry price including VAT and making charges.

        Args:
            base_price_per_gram: Base gold price per gram (BDT)
            weight_grams: Weight of jewelry in grams
            purity: Gold purity (22k, 21k, or 18k)
            item_type: Type of jewelry item
            custom_making_charge: Optional custom making charge per gram

        Returns:
            JewelryPrice object with detailed breakdown
        """
        # Calculate base price
        base_price = base_price_per_gram * weight_grams

        # Calculate VAT
        vat_amount = base_price * self.vat_rate

        # Get making charges
        if custom_making_charge is not None:
            making_charge_per_gram = custom_making_charge
        else:
            making_charge_per_gram = self.making_charges.get(item_type, {}).get(
                purity, 800
            )

        making_charges = making_charge_per_gram * weight_grams

        # Calculate total
        total_price = base_price + vat_amount + making_charges

        result = JewelryPrice(
            base_price=base_price,
            vat_amount=vat_amount,
            making_charges=making_charges,
            total_price=total_price,
            weight_grams=weight_grams,
            purity=purity,
            item_type=item_type,
        )

        logger.debug(
            f"Calculated price for {weight_grams}g {purity} {item_type}: "
            f"Base={base_price:.2f}, VAT={vat_amount:.2f}, "
            f"Making={making_charges:.2f}, Total={total_price:.2f}"
        )

        return result

    def get_making_charge_rate(
        self, item_type: str = "other", purity: str = "22k"
    ) -> float:
        """
        Get making charge rate for specific jewelry type and purity.

        Args:
            item_type: Type of jewelry
            purity: Gold purity

        Returns:
            Making charge per gram in BDT
        """
        return self.making_charges.get(item_type, {}).get(purity, 800)

    def update_vat_rate(self, new_rate: float) -> None:
        """
        Update VAT rate.

        Args:
            new_rate: New VAT rate (e.g., 0.05 for 5%)
        """
        self.vat_rate = new_rate
        logger.info(f"Updated VAT rate to {new_rate * 100}%")

    def update_making_charges(
        self, item_type: str, purity: str, new_charge: float
    ) -> None:
        """
        Update making charge for specific item type and purity.

        Args:
            item_type: Jewelry type
            purity: Gold purity
            new_charge: New making charge per gram
        """
        if item_type not in self.making_charges:
            self.making_charges[item_type] = {}

        self.making_charges[item_type][purity] = new_charge
        logger.info(
            f"Updated making charge for {item_type} ({purity}): {new_charge} BDT/g"
        )

    def calculate_price_comparison(
        self,
        base_price_per_gram: float,
        weight_grams: float,
        purity: str = "22k",
    ) -> dict[str, JewelryPrice]:
        """
        Calculate prices for all jewelry types for comparison.

        Args:
            base_price_per_gram: Base gold price per gram
            weight_grams: Weight in grams
            purity: Gold purity

        Returns:
            Dictionary mapping item type to JewelryPrice
        """
        comparison = {}

        for item_type in self.making_charges.keys():
            comparison[item_type] = self.calculate_jewelry_price(
                base_price_per_gram=base_price_per_gram,
                weight_grams=weight_grams,
                purity=purity,  # type: ignore[arg-type]
                item_type=item_type,  # type: ignore[arg-type]
            )

        return comparison


def format_price_breakdown(price: JewelryPrice) -> str:
    """
    Format jewelry price breakdown as readable string.

    Args:
        price: JewelryPrice object

    Returns:
        Formatted string with price details
    """
    return f"""
Jewelry Price Breakdown
=======================
Item Type: {price.item_type.title()}
Purity: {price.purity}
Weight: {price.weight_grams}g

Base Gold Price: ৳{price.base_price:,.2f}
VAT (5%): ৳{price.vat_amount:,.2f}
Making Charges: ৳{price.making_charges:,.2f}
------------------------
Total Price: ৳{price.total_price:,.2f}
""".strip()


if __name__ == "__main__":
    # Example usage
    calculator = JewelryPricingCalculator()

    # Calculate price for a 22k gold ring weighing 5 grams
    base_price = 8500  # BDT per gram
    weight = 5  # grams

    price = calculator.calculate_jewelry_price(
        base_price_per_gram=base_price,
        weight_grams=weight,
        purity="22k",
        item_type="ring",
    )

    print(format_price_breakdown(price))
