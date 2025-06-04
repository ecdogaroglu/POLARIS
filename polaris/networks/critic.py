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


class ContinuousQNetwork(nn.Module):
    """Q-function network for evaluating continuous actions."""

    def __init__(
        self, belief_dim, latent_dim, action_dim, hidden_dim, num_agents=2, device=None
    ):
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        super(ContinuousQNetwork, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.num_agents = num_agents

        # Combined input: belief, latent, continuous action, and neighbor actions
        # neighbor_actions is now [batch_size, 1] so we add 1 instead of num_agents
        input_dim = belief_dim + latent_dim + 1 + 1

        # Q-network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 1)  # Single Q-value output

        # Initialize parameters
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.q_head.weight)

    def forward(self, belief, latent, action, neighbor_actions=None):
        """Compute Q-value given belief, latent, action, and neighbor actions."""
        # Handle different dimensions
        if belief.dim() == 3:  # [1, batch_size, belief_dim]
            belief = belief.squeeze(0)
        if latent.dim() == 3:  # [1, batch_size, latent_dim]
            latent = latent.squeeze(0)
        
        # Get batch size from belief tensor
        batch_size = belief.size(0)
        
        # Ensure action is properly shaped [batch_size, 1]
        if action.dim() == 1:
            action = action.unsqueeze(1)
        elif action.dim() == 2 and action.size(0) == 1:
            # Handle case where action is [1, batch_size] - transpose it
            action = action.transpose(0, 1)
            
        # Handle neighbor actions - expect [batch_size, 1] format with actual values
        if neighbor_actions is None:
            neighbor_actions = torch.zeros(batch_size, 1, device=self.device)
        else:
            # Ensure correct shape [batch_size, 1]
            if neighbor_actions.dim() == 1:
                neighbor_actions = neighbor_actions.unsqueeze(1)
            elif neighbor_actions.dim() != 1:
                # If multi-agent format [batch_size, num_agents], extract neighbor values
                # Sum to get the non-zero neighbor action value
                neighbor_actions = neighbor_actions.sum(dim=1, keepdim=True)

        # Combine inputs - all tensors should now have matching batch sizes
        combined = torch.cat([belief, latent, action, neighbor_actions], dim=1)

        # Forward pass
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_head(x)

        return q_value 