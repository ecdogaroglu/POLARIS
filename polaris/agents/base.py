from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class BaseAgent(ABC):
    """Abstract base class for all agents in POLARIS."""

    def __init__(
        self,
        agent_id: int,
        num_agents: int,
        num_states: int,
        device: Union[str, torch.device] = "cpu",
    ):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.num_states = num_states
        self.device = torch.device(device) if isinstance(device, str) else device

    @abstractmethod
    def observe(
        self, signal: torch.Tensor, neighbor_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process observation and update internal state."""
        pass

    @abstractmethod
    def select_action(self) -> Tuple[Any, Union[torch.Tensor, np.ndarray]]:
        """Select an action based on current state."""
        pass

    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update agent parameters."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent state."""
        pass

    @abstractmethod
    def load(self, path: str, evaluation_mode: bool = False) -> None:
        """Load agent state."""
        pass

    def set_train_mode(self) -> None:
        """Set agent to training mode."""
        for module in self.modules():
            if isinstance(module, nn.Module):
                module.train()

    def set_eval_mode(self) -> None:
        """Set agent to evaluation mode."""
        for module in self.modules():
            if isinstance(module, nn.Module):
                module.eval()
