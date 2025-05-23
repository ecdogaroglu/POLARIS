"""
Regularization algorithms for continual learning.

Includes Synaptic Intelligence (SI) and Elastic Weight Consolidation (EWC).
"""

try:
    from .si import SILoss, calculate_path_integral_from_replay_buffer, calculate_si_loss
except ImportError:
    SILoss = None
    calculate_path_integral_from_replay_buffer = None
    calculate_si_loss = None

try:
    from .ewc import ElasticWeightConsolidation, calculate_ewc_loss
except ImportError:
    ElasticWeightConsolidation = None
    calculate_ewc_loss = None

__all__ = [
    "SILoss",
    "calculate_path_integral_from_replay_buffer",
    "calculate_si_loss",
    "ElasticWeightConsolidation",
    "calculate_ewc_loss",
] 