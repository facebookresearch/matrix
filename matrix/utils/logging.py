# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Logging utility that supports both standard logging and Scuba/Hive logging via frogger2.
Automatically uses Scuba/Hive when running in Meta cloud environments.
"""

import logging
import os
from enum import Enum, auto, unique
from typing import Optional

# Try to import frogger2, fall back to None if not available
FROGGER_AVAILABLE: bool = False
OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None
try:
    from frogger2.applications.matrix import log  # type: ignore

    matrix_logger = log
    FROGGER_AVAILABLE = True
    OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
except ImportError:
    matrix_logger = None
    FROGGER_AVAILABLE = False
    OTEL_EXPORTER_OTLP_ENDPOINT = None


@unique
class MatrixLogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class MatrixLogger:
    """Logger that supports both standard logging and Scuba logging."""

    def __init__(self, name: Optional[str] = None, level: int = logging.INFO):
        """
        Initialize a Matrix logger.

        Args:
            name: Logger name (defaults to __name__)
            level: Logging level (default: INFO)
        """
        self.logger = logging.getLogger(name or __name__)
        self.logger.setLevel(level)
        self.use_frogger = FROGGER_AVAILABLE and OTEL_EXPORTER_OTLP_ENDPOINT

    @property
    def handlers(self):
        """Access underlying logger's handlers for configuration."""
        return self.logger.handlers

    def addHandler(self, handler):
        """Add a handler to the underlying logger."""
        self.logger.addHandler(handler)

    def removeHandler(self, handler):
        """Remove a handler from the underlying logger."""
        self.logger.removeHandler(handler)

    def setLevel(self, level):
        """Set logging level."""
        self.logger.setLevel(level)

    def transmit(self, level: MatrixLogLevel, message: str, **kwargs):
        """Log to Scuba if available."""
        try:
            matrix_logger(log_level=level, log_message=message, **kwargs)
        except Exception as exn:
            self.logger.error(f"MatrixLogger failed to transmit payload: {exn}")

    def _log(self, level: MatrixLogLevel, message: str, **kwargs) -> None:
        """Log info message to both standard logger and Hive/Scuba."""
        local_log = getattr(self.logger, level.name.lower())

        if self.use_frogger:
            # Extract structured fields for OpenTelemetry payload
            extra = {
                k: v
                for k, v in kwargs.items()
                if k in ["job_id", "duration_seconds", "num_samples", "status"]
            }
            local_log(message, extra=extra)
            self.transmit(level=level, message=message, **extra)
        else:
            # Standard logging fallback
            if kwargs:
                # Format kwargs as `key=value` pairs
                kwargs_str: str = " ".join(f"{k}={v}" for k, v in kwargs.items())
                formatted_message: str = f"{message} [{kwargs_str}]"
                local_log(formatted_message, extra=kwargs)
            else:
                local_log(message)

    def debug(self, message: str, **kwargs) -> None:
        self._log(level=MatrixLogLevel.DEBUG, message=message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self._log(level=MatrixLogLevel.INFO, message=message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self._log(level=MatrixLogLevel.WARNING, message=message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self._log(level=MatrixLogLevel.ERROR, message=message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self._log(level=MatrixLogLevel.CRITICAL, message=message, **kwargs)


def get_logger(
    name: Optional[str] = None, level: int = logging.INFO
) -> MatrixLogger | logging.Logger:
    """
    Get a logger instance

    Args:
        name: Logger name (defaults to calling module's `__name__`)
        level: Logger level (default: `INFO`)

    Returns:
        `MatrixLogger` or `logging.Logger`
    """

    if FROGGER_AVAILABLE:
        return MatrixLogger(name=name, level=level)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger(name)
        logger.setLevel(level)
        return logger
