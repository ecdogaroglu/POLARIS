from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from polaris.networks.gnn import TemporalGNN
from polaris.networks.mlp import DecoderNetwork, EncoderNetwork


class InferenceComponent:
    """Handles latent state inference for POLARIS agents."""

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        latent_dim: int,
        num_agents: int,
        num_belief_states: int,
        device: torch.device,
        use_gnn: bool = True,
        **kwargs,
    ):
        self.latent_dim = latent_dim
        self.device = device
        self.use_gnn = use_gnn

        if use_gnn:
            self.module = TemporalGNN(
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                num_agents=num_agents,
                device=device,
                num_belief_states=num_belief_states,
                **kwargs,
            ).to(device)
        else:
            self.encoder = EncoderNetwork(
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_agents=num_agents,
                device=device,
                num_belief_states=num_belief_states,
            ).to(device)

            self.decoder = DecoderNetwork(
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_agents=num_agents,
                num_belief_states=num_belief_states,
                device=device,
            ).to(device)

        # Initialize latent state
        self.current_latent = torch.zeros(1, latent_dim, device=device)
        self.current_mean = torch.zeros(1, latent_dim, device=device)
        self.current_logvar = torch.zeros(1, latent_dim, device=device)

    def infer(
        self,
        signal: torch.Tensor,
        neighbor_actions: torch.Tensor,
        reward: torch.Tensor,
        next_signal: torch.Tensor,
        current_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Infer latent state from observations."""
        if current_latent is None:
            current_latent = self.current_latent

        if self.use_gnn:
            mean, logvar, opponent_belief = self.module(
                signal, neighbor_actions, reward, next_signal, current_latent
            )
        else:
            mean, logvar, opponent_belief = self.encoder(
                signal, neighbor_actions, reward, next_signal, current_latent
            )

        # Sample latent state
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-6, max=1e6)

        distribution = torch.distributions.Normal(mean, std)
        new_latent = distribution.rsample()

        # Update state
        self.current_latent = new_latent.unsqueeze(0)
        self.current_mean = mean
        self.current_logvar = logvar

        return new_latent, mean, logvar

    def predict_actions(
        self, signal: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        """Predict neighbor actions."""
        if self.use_gnn:
            return self.module.predict_actions(signal, latent)
        else:
            return self.decoder(signal, latent)

    def reset(self):
        """Reset latent state."""
        self.current_latent.zero_()
        self.current_mean.zero_()
        self.current_logvar.zero_()

        if self.use_gnn:
            self.module.reset_memory()
