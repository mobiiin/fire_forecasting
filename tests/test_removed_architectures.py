"""Regression tests for removed architecture handling."""

from __future__ import annotations

import pytest

from src.models.architecture_registry import get_architecture_spec
from src.models.model_factory import build_model_from_config


_REMOVED_MESSAGE = "The old CAWFE-Latte-Lite implementation has been removed. A new design will be added later."


def test_new_cawfe_latte_is_available_again() -> None:
	pytest.importorskip("torch")
	from src.models.cawfe_latte import CAWFELatte

	spec = get_architecture_spec("cawfe_latte")
	assert spec.name == "cawfe_latte"
	model = build_model_from_config({"model": {"architecture": "cawfe_latte", "input_channels": 129}, "cawfe_latte": {"input_channels": 129}}, input_channels=129)
	assert isinstance(model, CAWFELatte)


def test_removed_cawfe_latte_lite_registry_error_is_clear() -> None:
	with pytest.raises(KeyError, match="CAWFE-Latte-Lite implementation has been removed"):
		get_architecture_spec("cawfe_latte_lite")


def test_removed_cawfe_latte_lite_factory_error_is_clear() -> None:
	config = {"model": {"architecture": "cawfe_latte_lite", "output_channels": 4}}
	with pytest.raises(ValueError, match="CAWFE-Latte-Lite implementation has been removed"):
		build_model_from_config(config, input_channels=129)
