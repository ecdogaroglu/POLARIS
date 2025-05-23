from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from polaris.networks.mlp import ContinuousPolicyNetwork, PolicyNetwork


class PolicyComponent:
    """Handles action selection for POLARIS agents."""

    def __init__(
        self,
        belief_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: torch.device,
        continuous_actions: bool = False,
        min_action: float = 0.0,
        max_action: float = 1.0,
    ):
        self.device = device
        self.continuous_actions = continuous_actions
        self.action_dim = action_dim

        if continuous_actions:
            self.network = ContinuousPolicyNetwork(
                belief_dim=belief_dim,
                latent_dim=latent_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                device=device,
                min_action=min_action,
                max_action=max_action,
            ).to(device)
        else:
            self.network = PolicyNetwork(
                belief_dim=belief_dim,
                latent_dim=latent_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                device=device,
            ).to(device)

    def select_action(
        self, belief: torch.Tensor, latent: torch.Tensor
    ) -> Tuple[Union[int, float], np.ndarray]:
        """Select action based on current state."""
        if self.continuous_actions:
            action, log_prob, mean = self.network.sample_action(belief, latent)
            action_value = action.squeeze().detach().cpu().numpy()
            return action_value.item(), np.array([action_value.item()])
        else:
            action_logits = self.network(belief, latent)
            action_probs = F.softmax(action_logits, dim=-1)

            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()

            return action, action_probs.squeeze(0).detach().cpu().numpy()

    def get_action_distribution(
        self, belief: torch.Tensor, latent: torch.Tensor
    ) -> Union[torch.distributions.Normal, torch.distributions.Categorical]:
        """Get action distribution."""
        if self.continuous_actions:
            mean, log_std = self.network(belief, latent)
            std = torch.exp(log_std)
            return torch.distributions.Normal(mean, std)
        else:
            action_logits = self.network(belief, latent)
            action_probs = F.softmax(action_logits, dim=-1)
            return torch.distributions.Categorical(action_probs)
