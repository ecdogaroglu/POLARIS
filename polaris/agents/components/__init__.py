"""
Agent components for POLARIS.

Contains modular components for belief processing, policies, critics, and inference.
"""

try:
    from .belief import BeliefComponent
except ImportError:
    BeliefComponent = None

try:
    from .critics import CriticComponent
except ImportError:
    CriticComponent = None

try:
    from .inference import InferenceComponent
except ImportError:
    InferenceComponent = None

try:
    from .policy import PolicyComponent
except ImportError:
    PolicyComponent = None

__all__ = [
    "BeliefComponent",
    "CriticComponent", 
    "InferenceComponent",
    "PolicyComponent",
]
