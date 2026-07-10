"""Non-neural wildfire forecasting baselines."""

from src.baselines.evaluator import evaluate_baseline
from src.baselines.linear_extrapolation import predict_linear_extrapolation_for_sample
from src.baselines.persistence import predict_persistence_for_sample

__all__ = [
	"evaluate_baseline",
	"predict_linear_extrapolation_for_sample",
	"predict_persistence_for_sample",
]
