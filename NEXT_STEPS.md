# Next Steps - MVC Migration Guide

## Quick Start for Next Developer

This guide helps you continue the MVC refactoring where it was left off.

## Current Status: Phase 1 Complete ✅

The foundation is complete:

- ✅ Base classes created (`models/base.py`)
- ✅ Data models created (`models/*.py`)
- ✅ Repositories created (`repositories/*.py`)
- ✅ DuckDB standardization complete
- ✅ Code quality improvements (logging, tests)

## Next: Phase 2 - Service Layer

### Step 1: Create Price Service

Create `services/price_service.py`:

```python
from models import Price, HistoricalPrice, BaseService
from repositories import PriceRepository, HistoricalPriceRepository

class PriceService(BaseService):
    def __init__(self, db_path: str = "data/gold_prices.db"):
        super().__init__()
        self.price_repo = PriceRepository(db_path)
        self.historical_repo = HistoricalPriceRepository(db_path)

    def get_current_price(self, purity: str = "22K") -> Price | None:
        """Get current price for given purity."""
        return self.price_repo.get_latest_by_purity(purity)

    def calculate_price_change(self, purity: str, days: int = 7) -> dict:
        """Calculate price change over specified days."""
        # Business logic here
        pass

    def get_price_statistics(self, purity: str, days: int = 30) -> dict:
        """Get price statistics (avg, min, max, volatility)."""
        # Business logic here
        pass
```

### Step 2: Create Investment Service

Create `services/investment_service.py`:

```python
from models import Investment, InvestmentGoal, PortfolioSummary, BaseService
from repositories import InvestmentRepository, InvestmentGoalRepository
from services.price_service import PriceService

class InvestmentService(BaseService):
    def __init__(self, db_path: str = "data/gold_prices.db"):
        super().__init__()
        self.investment_repo = InvestmentRepository(db_path)
        self.goal_repo = InvestmentGoalRepository(db_path)
        self.price_service = PriceService(db_path)

    def add_investment(self, investment: Investment) -> Investment:
        """Add new investment transaction."""
        return self.investment_repo.create(investment)

    def calculate_portfolio_value(self, purity: str = "22K") -> PortfolioSummary:
        """Calculate current portfolio value and P&L."""
        # Business logic here
        pass

    def track_goal_progress(self, goal_id: int) -> dict:
        """Track progress toward investment goal."""
        # Business logic here
        pass
```

### Step 3: Create Jewelry Service

Create `services/jewelry_service.py`:

```python
from models import JewelryPrice, MakingCharges, BaseService
from services.price_service import PriceService

class JewelryService(BaseService):
    # Making charges by jewelry type (per gram)
    MAKING_CHARGES = {
        "ring": {"22k": 800, "21k": 750, "18k": 600},
        "necklace": {"22k": 1200, "21k": 1100, "18k": 900},
        # ... rest from jewelry_pricing.py
    }

    DEFAULT_VAT_RATE = 0.05

    def __init__(self, db_path: str = "data/gold_prices.db"):
        super().__init__()
        self.price_service = PriceService(db_path)

    def calculate_jewelry_price(
        self,
        weight_grams: float,
        purity: str,
        item_type: str
    ) -> JewelryPrice:
        """Calculate final jewelry price with VAT and making charges."""
        # Move logic from jewelry_pricing.py here
        pass
```

### Step 4: Update Service __init__.py

Update `services/__init__.py`:

```python
from services.gold_price_service import GoldPriceService
from services.price_service import PriceService
from services.investment_service import InvestmentService
from services.jewelry_service import JewelryService

__all__ = [
    "GoldPriceService",  # Existing ML service
    "PriceService",
    "InvestmentService",
    "JewelryService",
]
```

## Phase 3 - Controller Layer

### Step 1: Create Price Controller

Create `controllers/price_controller.py`:

```python
from models import BaseController
from services import PriceService

class PriceController(BaseController):
    def __init__(self, service: PriceService):
        super().__init__(service)

    def get_current_price(self, request: dict) -> dict:
        """Handle get current price request."""
        # Validate
        is_valid, error = self._validate_request(request)
        if not is_valid:
            return self._format_response(None, success=False, error=error)

        # Execute
        try:
            purity = request.get('purity', '22K')
            price = self.service.get_current_price(purity)

            if price:
                return self._format_response(price.to_dict())
            else:
                return self._format_response(
                    None,
                    success=False,
                    error=f"No price found for {purity}"
                )
        except Exception as e:
            self.logger.error(f"Error getting price: {e}")
            return self._format_response(None, success=False, error=str(e))

    def _validate_request(self, request: dict) -> tuple[bool, str]:
        """Validate price request."""
        purity = request.get('purity', '22K')
        valid_purities = ['18K', '21K', '22K', '24K']

        if purity not in valid_purities:
            return False, f"Invalid purity. Must be one of {valid_purities}"

        return True, ""
```

### Step 2: Create Other Controllers

Follow the same pattern for:

- `investment_controller.py`
- `jewelry_controller.py`
- `forecast_controller.py`

## Phase 4 - View Layer

### Step 1: Create Price View

Create `views/price_view.py`:

```python
import streamlit as st
from models import BaseView
from controllers import PriceController

class PriceView(BaseView):
    def __init__(self, controller: PriceController):
        super().__init__(controller)

    def render(self) -> None:
        """Render current price analysis view."""
        st.header("📊 Current Gold Prices")

        # User inputs
        col1, col2 = st.columns(2)
        with col1:
            purity = st.selectbox(
                "Select Purity",
                ["18K", "21K", "22K", "24K"],
                index=2  # Default to 22K
            )

        # Get data
        response = self.controller.get_current_price({"purity": purity})

        # Display results
        if response['success']:
            price_data = response['data']
            st.metric(
                f"Gold Price ({purity})",
                f"৳{price_data['price_bdt_per_gram']:,.2f}/gram"
            )
            st.caption(f"Source: {price_data['source']}")
            st.caption(f"Date: {price_data['date']}")
        else:
            st.error(response['error'])
```

### Step 2: Extract Views from app.py

Gradually extract each tab from `app.py`:

1. Current Analysis → `price_view.py`
2. 7-Day Forecast → `forecast_view.py`
3. Historical Trends → `historical_view.py`
4. Model Performance → `performance_view.py`
5. User Input Prediction → `prediction_view.py`
6. Investment Tracker → `investment_view.py`
7. Jewelry Pricing → `jewelry_view.py`
8. Backtesting → `backtest_view.py`

## Phase 5 - Refactor app.py

### Final app.py Structure

```python
import streamlit as st
from logging_config import setup_logging
from database_schema import DatabaseSchema

# Import all views, controllers, services
from services import PriceService, InvestmentService, JewelryService
from controllers import PriceController, InvestmentController, JewelryController
from views import PriceView, InvestmentView, JewelryView

# Setup
setup_logging()
st.set_page_config(...)

class GoldPriceAnalyzerApp:
    """Main application class."""

    def __init__(self):
        # Initialize database
        self.db = DatabaseSchema()
        self.db.create_all_tables()

        # Initialize services
        self.price_service = PriceService()
        self.investment_service = InvestmentService()
        self.jewelry_service = JewelryService()

        # Initialize controllers
        self.price_controller = PriceController(self.price_service)
        self.investment_controller = InvestmentController(self.investment_service)
        self.jewelry_controller = JewelryController(self.jewelry_service)

        # Initialize views
        self.price_view = PriceView(self.price_controller)
        self.investment_view = InvestmentView(self.investment_controller)
        self.jewelry_view = JewelryView(self.jewelry_controller)

    def run(self):
        """Run the application."""
        # Header
        st.markdown('<h1 class="main-header">🥇 GP Analyzer</h1>', unsafe_allow_html=True)

        # Tabs
        tabs = st.tabs([
            "📊 Current Analysis",
            "💰 Investment Tracker",
            "💍 Jewelry Pricing",
            # ... other tabs
        ])

        with tabs[0]:
            self.price_view.render()

        with tabs[1]:
            self.investment_view.render()

        with tabs[2]:
            self.jewelry_view.render()

if __name__ == "__main__":
    app = GoldPriceAnalyzerApp()
    app.run()
```

## Testing Strategy

### 1. Test Each Layer

```bash
# Test repositories
uv run pytest tests/test_repositories.py -v

# Test services
uv run pytest tests/test_services.py -v

# Test controllers
uv run pytest tests/test_controllers.py -v

# Test views (integration)
uv run pytest tests/test_views.py -v
```

### 2. Create Test Files

Create these test files as you build each layer:

- `tests/test_price_service.py`
- `tests/test_investment_service.py`
- `tests/test_jewelry_service.py`
- `tests/test_price_controller.py`
- etc.

## Migration Checklist

Use this checklist to track progress:

- [ ] __Phase 2: Services__
  - [ ] PriceService created
  - [ ] InvestmentService created
  - [ ] JewelryService created
  - [ ] ForecastService created (wraps ML models)
  - [ ] BacktestService created
  - [ ] All services tested

- [ ] __Phase 3: Controllers__
  - [ ] PriceController created
  - [ ] InvestmentController created
  - [ ] JewelryController created
  - [ ] ForecastController created
  - [ ] BacktestController created
  - [ ] All controllers tested

- [ ] __Phase 4: Views__
  - [ ] PriceView extracted from app.py
  - [ ] ForecastView extracted
  - [ ] HistoricalView extracted
  - [ ] PerformanceView extracted
  - [ ] InvestmentView extracted
  - [ ] JewelryView extracted
  - [ ] BacktestView extracted

- [ ] __Phase 5: App Refactoring__
  - [ ] GoldPriceAnalyzerApp class created
  - [ ] Dependency injection implemented
  - [ ] app.py simplified to < 200 lines
  - [ ] All tabs working with new structure

- [ ] __Phase 6: Cleanup__
  - [ ] Remove legacy CSV files
  - [ ] Archive old analyzer.py (replaced by ml_models)
  - [ ] Archive old jewelry_pricing.py (replaced by service)
  - [ ] Update all documentation
  - [ ] Run full test suite
  - [ ] Run code quality checks (black, isort, mypy)

## Common Patterns

### Error Handling

```python
try:
    result = self.service.operation()
    return self._format_response(result.to_dict())
except ValueError as e:
    return self._format_response(None, success=False, error=f"Validation error: {e}")
except Exception as e:
    self.logger.error(f"Unexpected error: {e}")
    return self._format_response(None, success=False, error="Internal server error")
```

### Logging

```python
self.logger.info(f"Processing request for {purity} gold")
self.logger.warning(f"No data found for {purity}")
self.logger.error(f"Database error: {e}")
```

### Type Hints

```python
def get_price(self, purity: str) -> Price | None:
    """Always include return type hints."""
    pass

def calculate_stats(self, data: list[Price]) -> dict[str, float]:
    """Be explicit about nested types."""
    pass
```

## Resources

- __Architecture Guide__: See `models/base.py` for base class examples
- __Data Models__: See `models/` for dataclass examples
- __Repositories__: See `repositories/` for CRUD patterns
- __Progress Tracking__: See `MVC_REFACTORING.md`
- __Legacy Code__: See `LEGACY_FILES.md`

## Questions?

If you're stuck:

1. Check existing code in `repositories/` for patterns
2. Review base classes in `models/base.py`
3. Look at test examples in `tests/`
4. Follow the type hints - they guide you to correct usage

## Tips

1. __Work incrementally__ - Complete one service/controller/view at a time
2. __Test as you go__ - Write tests before refactoring
3. __Keep app.py running__ - Don't break existing functionality
4. __Use type hints__ - They catch errors early
5. __Log everything__ - Makes debugging easier

---

__Last Updated:__ January 5, 2026
__Current Phase:__ Phase 1 Complete, Ready for Phase 2
__Next Task:__ Create PriceService in `services/price_service.py`
