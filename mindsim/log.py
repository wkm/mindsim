"""Structured logging configuration for MindSim.

Configures structlog wrapping stdlib logging so that:
- Console output uses colored key-value format (human-readable)
- File output writes JSON lines to logs/mindsim.log (rotating, 5MB, 3 backups)
- All existing ``logging.getLogger(__name__)`` calls continue to work
- Unhandled exceptions are captured to the log
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

import structlog

_setup_done = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structlog + stdlib logging for the entire process.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    Path("logs").mkdir(exist_ok=True)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(),
        ],
    )

    json_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        "logs/mindsim.log",
        maxBytes=5_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(json_formatter)
    root.addHandler(file_handler)

    _original_excepthook = sys.excepthook

    def _logging_excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: object,
    ) -> None:
        if not issubclass(exc_type, KeyboardInterrupt):
            logging.critical(
                "Unhandled exception", exc_info=(exc_type, exc_value, exc_tb)
            )
        _original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _logging_excepthook
