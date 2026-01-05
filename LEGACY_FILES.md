# Legacy Files Documentation

This document tracks legacy files and their migration status.

## Files Pending Refactor

### app.py (2500+ lines)

**Status**: Needs MVC refactoring
**Issue**: Monolithic file containing UI, business logic, and data access
**Plan**: Extract into proper MVC components:

- Views → `views/` directory
- Controllers → `controllers/` directory
- Business logic → `services/` directory
- Data models → `models/` directory (✅ Created)

### analyzer.py

**Status**: Legacy ML implementation
**Issue**: Original analyzer before ml_models/ restructure
**Plan**: Gradually migrate to use `ml_models/` orchestrator

### jewelry_pricing.py

**Status**: Standalone module
**Plan**: Refactor into:

- Model: `models/jewelry.py` (✅ Created)
- Service: `services/jewelry_service.py`
- Controller: `controllers/jewelry_controller.py`
- View: `views/jewelry_view.py`

### backtesting.py

**Status**: Standalone module
**Plan**: Refactor into:

- Service: `services/backtest_service.py`
- Controller: `controllers/backtest_controller.py`
- View: `views/backtest_view.py`

## CSV Files in data/

### gold_prices_20260105.csv

**Status**: Legacy dated backup
**Action**: Can be removed (data is in DuckDB)

### historical_gold_prices.csv

**Status**: Legacy CSV export
**Action**: Can be removed (data is in DuckDB)

## Empty/Placeholder Files

### ai_review_template.md

**Status**: Review checklist template
**Action**: Keep as reference

### .pre-commit/ai_review_local.py

**Status**: Pre-commit hook script
**Action**: Keep if used in CI/CD

## Migration Checklist

- [x] Replace print() with logger calls
- [x] Move test code from utils.py to tests/
- [x] Standardize on DuckDB (remove SQLite)
- [x] Restructure models/ folder
- [x] Create MVC base classes
- [x] Create data models
- [x] Create repository layer
- [ ] Create service layer
- [ ] Create controller layer
- [ ] Create view layer
- [ ] Refactor app.py
- [ ] Update all imports
- [ ] Run tests
- [ ] Update documentation

## Deprecated Patterns

1. **CSV as primary storage** → Use DuckDB exclusively
2. **print() statements** → Use logger
3. **Direct DB queries in app.py** → Use repositories
4. **Business logic in app.py** → Move to services
5. **UI mixed with logic** → Separate into views

## Notes

This is a living document. Update as refactoring progresses.
