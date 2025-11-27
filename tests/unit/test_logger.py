"""
Unit tests for src/utils/logger.py
"""

import pytest
import logging
from pathlib import Path

from src.utils.logger import (
    setup_logger,
    get_logger,
    create_experiment_logger,
    LoggerContext,
)


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_default_setup(self):
        """Test default logger setup."""
        logger = setup_logger("test_logger")

        assert logger.name == "test_logger"
        assert len(logger.handlers) > 0

    def test_console_only(self):
        """Test logger with console handler only."""
        logger = setup_logger("test_console", level="INFO", console=True, log_file=None)

        assert logger.name == "test_console"
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_file_only(self, tmp_path):
        """Test logger with file handler only."""
        log_file = tmp_path / "test.log"

        logger = setup_logger(
            "test_file",
            level="INFO",
            console=False,
            log_file=log_file,
        )

        assert logger.name == "test_file"
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.FileHandler)

    def test_console_and_file(self, tmp_path):
        """Test logger with both console and file handlers."""
        log_file = tmp_path / "test.log"

        logger = setup_logger(
            "test_both",
            level="INFO",
            console=True,
            log_file=log_file,
        )

        assert len(logger.handlers) == 2

    def test_logging_levels(self, tmp_path):
        """Test different logging levels."""
        log_file = tmp_path / "test.log"

        # Create logger with DEBUG level
        logger = setup_logger(
            "test_levels",
            level="DEBUG",
            console=False,
            log_file=log_file,
        )

        # Log messages at different levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Read log file
        log_content = log_file.read_text()

        assert "Debug message" in log_content
        assert "Info message" in log_content
        assert "Warning message" in log_content
        assert "Error message" in log_content

    def test_file_level_different_from_console(self, tmp_path):
        """Test different log levels for console and file."""
        log_file = tmp_path / "test.log"

        logger = setup_logger(
            "test_diff_levels",
            level="WARNING",  # Console level
            console=True,
            log_file=log_file,
            file_level="DEBUG",  # File level
        )

        # File handler should capture DEBUG
        file_handler = [h for h in logger.handlers if isinstance(h, logging.FileHandler)][0]
        assert file_handler.level == logging.DEBUG

        # Console handler should be WARNING
        console_handler = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)][0]
        assert console_handler.level == logging.WARNING

    def test_log_file_creates_parent_dir(self, tmp_path):
        """Test that log file creation creates parent directories."""
        log_file = tmp_path / "nested" / "dir" / "test.log"

        logger = setup_logger(
            "test_nested",
            console=False,
            log_file=log_file,
        )

        logger.info("Test message")

        assert log_file.exists()
        assert log_file.parent.exists()


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_creates_if_not_exists(self):
        """Test that get_logger creates logger if it doesn't exist."""
        logger = get_logger("test_get_logger")

        assert logger.name == "test_get_logger"
        assert len(logger.handlers) > 0

    def test_get_logger_returns_existing(self):
        """Test that get_logger returns existing logger."""
        # Create logger
        logger1 = setup_logger("test_existing")

        # Get same logger
        logger2 = get_logger("test_existing")

        assert logger1 is logger2


class TestCreateExperimentLogger:
    """Tests for create_experiment_logger function."""

    def test_create_experiment_logger(self, tmp_path):
        """Test experiment logger creation."""
        logger = create_experiment_logger(
            "test_experiment",
            log_dir=tmp_path,
            level="INFO",
        )

        # Logger name now includes timestamp for uniqueness
        assert logger.name.startswith("exp_test_experiment_")

        # Check that log file was created
        log_files = list(tmp_path.glob("test_experiment_*.log"))
        assert len(log_files) == 1

        # Check log file contains experiment info
        log_content = log_files[0].read_text()
        assert "Experiment: test_experiment" in log_content

    def test_experiment_logger_timestamp(self, tmp_path):
        """Test that experiment logger includes timestamp in filename."""
        create_experiment_logger("exp1", log_dir=tmp_path)
        create_experiment_logger("exp1", log_dir=tmp_path)

        # Should create two different log files
        log_files = list(tmp_path.glob("exp1_*.log"))
        assert len(log_files) == 2


class TestLoggerContext:
    """Tests for LoggerContext context manager."""

    def test_logger_context_changes_level(self):
        """Test that LoggerContext temporarily changes log level."""
        logger = setup_logger("test_context", level="INFO")

        original_level = logger.level

        # Change level in context
        with LoggerContext(logger, "DEBUG"):
            assert logger.level == logging.DEBUG

        # Level should be restored
        assert logger.level == original_level

    def test_logger_context_with_int_level(self):
        """Test LoggerContext with integer log level."""
        logger = setup_logger("test_context_int", level="INFO")

        with LoggerContext(logger, logging.ERROR):
            assert logger.level == logging.ERROR

    def test_logger_context_restores_on_exception(self):
        """Test that LoggerContext restores level even on exception."""
        logger = setup_logger("test_context_exc", level="INFO")
        original_level = logger.level

        try:
            with LoggerContext(logger, "DEBUG"):
                assert logger.level == logging.DEBUG
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Level should still be restored
        assert logger.level == original_level
