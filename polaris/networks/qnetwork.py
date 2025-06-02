import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.device import get_best_device


class QNetwork(nn.Module):
    """Q-function network for evaluating state-action values."""

    def __init__(
        self, belief_dim, latent_dim, action_dim, hidden_dim, num_agents=10, device=None
    ):
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        super(QNetwork, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.num_agents = num_agents

        # Combined input: belief, latent, and neighbor actions (one-hot encoded for all neighbors)
        # We use action_dim * num_agents to represent all possible neighbor actions
        input_dim = belief_dim + latent_dim + action_dim * num_agents

        # Q-network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Initialize parameters
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, belief, latent, neighbor_actions=None):
        """Compute Q-values given belief, latent, and neighbor actions."""
        # Handle different dimensions
        if belief.dim() == 3:  # [batch_size, 1, belief_dim]
            belief = belief.squeeze(0)
        if latent.dim() == 3:  # [batch_size, 1, latent_dim]
            latent = latent.squeeze(0)

        # Combine inputs
        combined = torch.cat([belief, latent, neighbor_actions], dim=1)

        # Forward pass
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values 