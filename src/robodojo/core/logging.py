"""Application-owned logging configuration."""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

LOGGER_NAME: Final = "robodojo"
LOG_LEVEL_ENV: Final = "ROBODOJO_LOG_LEVEL"
DEFAULT_LOG_LEVEL: Final = "INFO"
LOG_FORMAT: Final = "%(levelname)s %(name)s: %(message)s"

_LEVELS: Final = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
_HANDLER_MARKER: Final = "_robodojo_console_handler"


def parse_log_level(value: str | int | None) -> int:
    """Resolve a supported log level or raise an actionable error."""
    if isinstance(value, int):
        if value in _LEVELS.values():
            return value
        raise ValueError(f"unsupported log level: {value}")
    normalized = (value or os.environ.get(LOG_LEVEL_ENV) or DEFAULT_LOG_LEVEL).strip().upper()
    try:
        return _LEVELS[normalized]
    except KeyError as exc:
        expected = ", ".join(_LEVELS)
        raise ValueError(f"unsupported log level {normalized!r}; expected one of: {expected}") from exc


def configure_logging(level: str | int | None = None) -> int:
    """Configure the ``robodojo`` logger hierarchy exactly once.

    Repeated calls update the owned console handler instead of accumulating
    duplicate handlers. Third-party and root logger configuration is left
    untouched.
    """
    resolved = parse_log_level(level)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolved)
    logger.propagate = False

    handler = next(
        (candidate for candidate in logger.handlers if getattr(candidate, _HANDLER_MARKER, False)),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)
    else:
        handler.stream = sys.stderr
    handler.setLevel(resolved)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return resolved
