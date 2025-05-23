"""
Utility functions and classes for POLARIS.

Contains device management, encoding, I/O, metrics, and mathematical utilities.
"""

from .device import get_best_device
from .encoding import encode_observation
from .io import setup_random_seeds, create_output_directory, save_checkpoint_models, save_final_models
from .metrics import initialize_metrics, update_metrics, save_metrics_to_file, calculate_theoretical_bounds
from .math import calculate_learning_rate

__all__ = [
    "get_best_device",
    "encode_observation",
    "setup_random_seeds",
    "create_output_directory",
    "save_checkpoint_models",
    "save_final_models",
    "initialize_metrics",
    "update_metrics", 
    "save_metrics_to_file",
    "calculate_theoretical_bounds",
    "calculate_learning_rate",
]
