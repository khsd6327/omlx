# SPDX-License-Identifier: Apache-2.0
"""
Logging configuration for oMLX.

This module provides centralized logging configuration with support for:
- Standard logging with configurable levels
- Structured JSON logging (optional)
- Request context tracking
- File logging with daily rotation
- Consistent formatting across all modules
"""

import logging
from contextvars import ContextVar
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

# Context variable for request ID tracking
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class RequestContextFilter(logging.Filter):
    """
    Add request_id to log records.

    This filter adds the current request ID (if set) to all log records,
    enabling request-level log correlation.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id attribute to log record."""
        record.request_id = _request_id.get() or "-"
        return True


class AdminStatsAccessFilter(logging.Filter):
    """Suppress repetitive uvicorn access logs for admin polling endpoints."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "/admin/api/stats" in msg:
            return False
        if "/admin/api/login" in msg:
            return False
        if "/admin/api/hf/tasks" in msg:
            return False
        if "/admin/api/oq/tasks" in msg:
            return False
        return True


class RequestLogContext:
    """
    Context manager for request-scoped logging.

    Usage:
        with RequestLogContext(request_id="abc123"):
            logger.info("Processing request")
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.previous_id: Optional[str] = None

    def __enter__(self) -> "RequestLogContext":
        self.previous_id = _request_id.get()
        _request_id.set(self.request_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _request_id.set(self.previous_id)


def configure_file_logging(
    log_dir: Path,
    level: str = "INFO",
    include_request_id: bool = True,
    retention_days: int = 7,
) -> None:
    """
    Configure file logging with daily rotation.

    Adds a file handler to the root logger that writes to {log_dir}/server.log
    with automatic daily rotation. Old log files are automatically deleted
    after retention_days.

    Args:
        log_dir: Directory to store log files.
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        include_request_id: Whether to include request_id in log format.
        retention_days: Number of days to retain old logs.
    """
    # Ensure log directory exists
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    level_name = level.upper()
    log_level = 5 if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)

    # Build format string (no colors for file)
    if include_request_id:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"
    else:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create file handler with daily rotation
    # File: server.log, rotated files: server.log.YYYY-MM-DD
    log_file = log_dir / "server.log"

    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"  # Results in server.log.2024-01-15
    file_handler.setLevel(log_level)

    formatter = logging.Formatter(format_str)
    file_handler.setFormatter(formatter)

    # Add request context filter
    if include_request_id:
        file_handler.addFilter(RequestContextFilter())

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
