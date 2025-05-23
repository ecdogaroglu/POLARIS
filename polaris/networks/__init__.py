"""
Neural network architectures for POLARIS.

Contains GNNs, MLPs, Transformers and other neural network components.
"""

from .mlp import (
    EncoderNetwork,
    DecoderNetwork,
    PolicyNetwork,
    ContinuousPolicyNetwork,
    QNetwork,
)

try:
    from .transformer import TransformerBeliefProcessor
except ImportError:
    TransformerBeliefProcessor = None

try:
    from .gnn import TemporalGNN, SocialLearningGNN
except ImportError:
    TemporalGNN = None
    SocialLearningGNN = None

__all__ = [
    "EncoderNetwork",
    "DecoderNetwork",
    "PolicyNetwork",
    "ContinuousPolicyNetwork",
    "QNetwork",
    "TransformerBeliefProcessor",
    "TemporalGNN",
    "SocialLearningGNN",
]
