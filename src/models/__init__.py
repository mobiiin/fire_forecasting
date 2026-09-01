"""Model definitions for wildfire forecasting."""

from src.models.architecture_registry import ARCHITECTURE_REGISTRY, get_architecture_spec, resolve_model_architecture
from src.models.cawfe_latte import CAWFELatte
from src.models.convlstm_unet import ConvLSTMUNet
from src.models.earthformer_lite import EarthformerLite
from src.models.model_factory import build_model_from_config
from src.models.st_mamba_lite import STMamba
from src.models.weatherformer_lite import WeatherFormerLite

__all__ = [
	"ARCHITECTURE_REGISTRY",
	"CAWFELatte",
	"ConvLSTMUNet",
	"EarthformerLite",
	"STMamba",
	"WeatherFormerLite",
	"build_model_from_config",
	"get_architecture_spec",
	"resolve_model_architecture",
]
