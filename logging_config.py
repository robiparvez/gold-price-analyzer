"""
Logging configuration for the Gold Price Analyzer application
"""

import logging
import logging.config
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str = "app.log") -> None:
    """
    Setup centralized logging configuration for the application.

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Log file name
    """

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_path = log_dir / log_file

    config: dict[str, any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "level": "WARNING",
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": str(log_path),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "": {  # Root logger
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "scraper": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "analyzer": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "streamlit": {
                "level": "WARNING",  # Reduce streamlit verbosity
                "handlers": ["file"],
                "propagate": False,
            },
            "prophet": {
                "level": "WARNING",  # Reduce prophet verbosity
                "handlers": ["file"],
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(config)

    # Log the initialization
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized with level {log_level}")
    logger.info(f"Log file: {log_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Custom exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions by logging them."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log keyboard interrupts
        return

    logger = get_logger("exception_handler")
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


# Setup exception handler
sys.excepthook = handle_exception
