import torch
import torch.nn as nn
from typing import Tuple
from polaris.networks.mlp import QNetwork

class CriticComponent:
    """Handles value estimation for POLARIS agents."""
    
    def __init__(
        self,
        belief_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_agents: int,
        device: torch.device
    ):
        self.device = device
        
        # Twin Q-networks for SAC
        self.q_network1 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        self.q_network2 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        # Target networks
        self.target_q_network1 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        self.target_q_network2 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        # Copy parameters to targets
        self.update_targets(tau=1.0)
        
    def get_q_values(
        self,
        belief: torch.Tensor,
        latent: torch.Tensor,
        neighbor_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from both networks."""
        q1 = self.q_network1(belief, latent, neighbor_actions)
        q2 = self.q_network2(belief, latent, neighbor_actions)
        return q1, q2
        
    def get_target_q_values(
        self,
        belief: torch.Tensor,
        latent: torch.Tensor,
        neighbor_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from target networks."""
        q1 = self.target_q_network1(belief, latent, neighbor_actions)
        q2 = self.target_q_network2(belief, latent, neighbor_actions)
        return q1, q2
        
    def update_targets(self, tau: float = 0.005):
        """Soft update of target networks."""
        for target_param, param in zip(
            self.target_q_network1.parameters(), 
            self.q_network1.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
            
        for target_param, param in zip(
            self.target_q_network2.parameters(), 
            self.q_network2.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )