"""
Utility functions and classes for POLARIS.

Contains device management, encoding, I/O, metrics, and mathematical utilities.
"""

from .device import get_best_device, set_device, get_device_info
from .encoding import encode_observation, decode_observation
from .io import save_model, load_model, save_metrics, load_metrics
from .metrics import calculate_learning_rate, calculate_accuracy
from .math import softmax, log_softmax, kl_divergence

__all__ = [
    "get_best_device",
    "set_device", 
    "get_device_info",
    "encode_observation",
    "decode_observation",
    "save_model",
    "load_model", 
    "save_metrics",
    "load_metrics",
    "calculate_learning_rate",
    "calculate_accuracy",
    "softmax",
    "log_softmax",
    "kl_divergence",
]
