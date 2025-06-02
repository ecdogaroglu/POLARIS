"""
Neural network architectures for POLARIS.

Contains GNNs, Transformers and other neural network components.
GNN is the default architecture for all inference and policy tasks.
"""

# Import GNN-based networks (now the default)
from .policy import PolicyNetwork, ContinuousPolicyNetwork
from .qnetwork import QNetwork
from .temporal_gnn import TemporalGNN, AttentionProcessingError, TemporalMemoryError

# Import transformer components
try:
    from .transformer import TransformerBeliefProcessor, InvertibleBeliefHead
except ImportError:
    TransformerBeliefProcessor = None
    InvertibleBeliefHead = None

__all__ = [
    # Core network components (GNN-based)
    "PolicyNetwork",
    "ContinuousPolicyNetwork",
    "QNetwork",
    
    # Advanced GNN components
    "TemporalGNN",
    "AttentionProcessingError", 
    "TemporalMemoryError",
    
    # Transformer components (optional)
    "TransformerBeliefProcessor",
    "InvertibleBeliefHead",
]
