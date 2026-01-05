# Controllers Package

This directory will contain controller classes for the MVC architecture.

## Purpose

Controllers handle request/response logic and validation. They:

- Receive requests from views
- Validate input data
- Delegate business logic to services
- Format responses for views

## Structure

```text
controllers/
├── __init__.py
├── price_controller.py       # Handle price-related requests
├── investment_controller.py  # Handle investment tracking requests
├── jewelry_controller.py     # Handle jewelry pricing requests
└── forecast_controller.py    # Handle ML forecast requests
```

## Example Usage

```python
from models import BaseController
from services import PriceService

class PriceController(BaseController):
    def __init__(self, service: PriceService):
        super().__init__(service)

    def get_current_price(self, request: dict) -> dict:
        # Validate request
        is_valid, error = self._validate_request(request)
        if not is_valid:
            return self._format_response(None, success=False, error=error)

        # Get data from service
        try:
            price = self.service.get_current_price(request['purity'])
            return self._format_response(price.to_dict())
        except Exception as e:
            return self._format_response(None, success=False, error=str(e))
```

## Next Steps

Gradually refactor app.py logic into controllers as the MVC migration progresses.
