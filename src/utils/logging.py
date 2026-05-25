"""Logging helpers."""

from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
	"""Configure and return the root project logger."""

	logger = logging.getLogger("fire_forecasting")
	logger.setLevel(getattr(logging, level.upper(), logging.INFO))
	logger.handlers.clear()

	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

	if log_file:
		file_handler = logging.FileHandler(log_file)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

	logger.propagate = False
	return logger
