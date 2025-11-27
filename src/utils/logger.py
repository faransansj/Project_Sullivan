"""
Project Sullivan - Logging Infrastructure

This module provides structured logging for the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from datetime import datetime


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    LEVEL_COLORS = {
        "DEBUG": Colors.CYAN,
        "INFO": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
        "CRITICAL": Colors.RED + Colors.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Save original values
        original_levelname = record.levelname
        original_msg = record.msg

        # Add color to level name
        if record.levelname in self.LEVEL_COLORS:
            record.levelname = (
                f"{self.LEVEL_COLORS[record.levelname]}"
                f"{record.levelname}{Colors.RESET}"
            )

        # Format message
        formatted = super().format(record)

        # Restore original values
        record.levelname = original_levelname
        record.msg = original_msg

        return formatted


def setup_logger(
    name: str = "project_sullivan",
    level: str | int = "INFO",
    log_file: Optional[str | Path] = None,
    console: bool = True,
    file_level: Optional[str | int] = None,
) -> logging.Logger:
    """
    Set up a logger with console and/or file handlers.

    Args:
        name: Logger name
        level: Console logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None = no file logging)
        console: Whether to enable console logging
        file_level: File logging level (None = same as console level)

    Returns:
        logging.Logger: Configured logger

    Example:
        >>> logger = setup_logger("preprocessing", level="DEBUG")
        >>> logger.info("Processing started")
        >>> logger.warning("Low quality frame detected")
        >>> logger.error("Failed to load data")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels, filter at handlers
    logger.handlers.clear()  # Remove existing handlers

    # Convert string levels to logging constants
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if isinstance(file_level, str):
        file_level = getattr(logging, file_level.upper())
    elif file_level is None:
        file_level = level

    # Console handler with color
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        console_format = (
            f"{Colors.BOLD}[%(asctime)s]{Colors.RESET} "
            f"%(levelname)s - "
            f"{Colors.BOLD}%(name)s{Colors.RESET}: "
            f"%(message)s"
        )
        console_formatter = ColoredFormatter(
            console_format, datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler without color
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level)

        file_format = (
            "[%(asctime)s] %(levelname)-8s - %(name)s - "
            "%(funcName)s:%(lineno)d - %(message)s"
        )
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "project_sullivan") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name

    Returns:
        logging.Logger: Logger instance

    Example:
        >>> logger = get_logger("preprocessing.alignment")
        >>> logger.info("Alignment completed")
    """
    logger = logging.getLogger(name)

    # Set up logger if it doesn't have handlers
    if not logger.handlers:
        logger = setup_logger(name)

    return logger


def create_experiment_logger(
    experiment_name: str,
    log_dir: str | Path = "logs",
    level: str = "INFO",
) -> logging.Logger:
    """
    Create a logger for an experiment with timestamped log file.

    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
        level: Logging level

    Returns:
        logging.Logger: Configured experiment logger

    Example:
        >>> logger = create_experiment_logger("preprocessing_run1")
        >>> logger.info("Starting preprocessing")
    """
    # Create timestamped log file with microseconds to avoid collisions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = Path(log_dir) / f"{experiment_name}_{timestamp}.log"

    # Set up logger with unique name including timestamp
    logger = setup_logger(
        name=f"exp_{experiment_name}_{timestamp}",
        level=level,
        log_file=log_file,
        console=True,
    )

    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Timestamp: {timestamp}")

    return logger


class LoggerContext:
    """Context manager for temporary log level changes."""

    def __init__(self, logger: logging.Logger, level: str | int):
        """
        Initialize context manager.

        Args:
            logger: Logger to modify
            level: Temporary logging level
        """
        self.logger = logger
        self.new_level = level if isinstance(level, int) else getattr(logging, level.upper())
        self.original_level = logger.level

    def __enter__(self):
        """Enter context: set new level."""
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: restore original level."""
        self.logger.setLevel(self.original_level)


def log_function_call(func):
    """
    Decorator to log function calls.

    Example:
        >>> @log_function_call
        ... def process_data(data):
        ...     return data * 2
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise

    return wrapper


# ============================================================================
# Convenience Functions
# ============================================================================


def log_info(message: str, logger_name: str = "project_sullivan") -> None:
    """Log info message."""
    get_logger(logger_name).info(message)


def log_warning(message: str, logger_name: str = "project_sullivan") -> None:
    """Log warning message."""
    get_logger(logger_name).warning(message)


def log_error(message: str, logger_name: str = "project_sullivan") -> None:
    """Log error message."""
    get_logger(logger_name).error(message)


def log_debug(message: str, logger_name: str = "project_sullivan") -> None:
    """Log debug message."""
    get_logger(logger_name).debug(message)


# Initialize default logger
default_logger = setup_logger()
