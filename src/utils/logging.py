"""Logging helpers."""

from __future__ import annotations

import logging
import sys
from typing import Optional


class _MaxLevelFilter(logging.Filter):
	"""Allow records up to and including a maximum log level."""

	def __init__(self, max_level: int) -> None:
		super().__init__()
		self.max_level = max_level

	def filter(self, record: logging.LogRecord) -> bool:
		return record.levelno <= self.max_level


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
	"""Configure and return the root project logger."""

	logger = logging.getLogger("fire_forecasting")
	logger.setLevel(getattr(logging, level.upper(), logging.INFO))
	logger.handlers.clear()

	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

	stdout_handler = logging.StreamHandler(sys.stdout)
	stdout_handler.setFormatter(formatter)
	stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))
	logger.addHandler(stdout_handler)

	stderr_handler = logging.StreamHandler(sys.stderr)
	stderr_handler.setLevel(logging.WARNING)
	stderr_handler.setFormatter(formatter)
	logger.addHandler(stderr_handler)

	if log_file:
		file_handler = logging.FileHandler(log_file)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

	logger.propagate = False
	return logger
