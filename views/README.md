# Views Package

This directory will contain view classes for the MVC architecture.

## Purpose

Views handle UI presentation using Streamlit. They:

- Render UI components
- Handle user interactions
- Call controllers to get/update data
- Display results to users

## Structure

```text
views/
├── __init__.py
├── price_view.py          # Current price analysis UI
├── forecast_view.py       # 7-day forecast UI
├── historical_view.py     # Historical trends UI
├── investment_view.py     # Investment tracking UI
├── jewelry_view.py        # Jewelry pricing calculator UI
└── backtest_view.py       # Backtesting UI
```

## Example Usage

```python
import streamlit as st
from models import BaseView
from controllers import PriceController

class PriceView(BaseView):
    def __init__(self, controller: PriceController):
        super().__init__(controller)

    def render(self) -> None:
        st.header("Current Gold Prices")

        # User input
        purity = st.selectbox("Select Purity", ["18K", "21K", "22K", "24K"])

        # Get data from controller
        if st.button("Get Price"):
            response = self.controller.get_current_price({"purity": purity})

            if response['success']:
                price_data = response['data']
                st.metric("Price", f"৳{price_data['price_bdt_per_gram']:,.2f}/gram")
            else:
                st.error(response['error'])
```

## Next Steps

Gradually extract UI components from app.py into view classes as the MVC migration progresses.
