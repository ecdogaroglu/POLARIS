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
        self, belief_dim, latent_dim, action_dim, hidden_dim, num_agents=10, device=None
    ):
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        super(ContinuousQNetwork, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.num_agents = num_agents

        # Combined input: belief, latent, continuous action, and neighbor actions
        input_dim = belief_dim + latent_dim + action_dim + num_agents

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
        if belief.dim() == 3:  # [batch_size, 1, belief_dim]
            belief = belief.squeeze(0)
        if latent.dim() == 3:  # [batch_size, 1, latent_dim]
            latent = latent.squeeze(0)
        
        # Get batch size from belief tensor
        batch_size = belief.size(0)
        
        # Ensure action is properly shaped
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if action.dim() == 2 and action.size(1) != self.action_dim:
            # If action is [batch_size, 1], keep it as is
            pass
        
        # Handle batch size mismatch for action
        if action.size(0) != batch_size:
            if action.size(0) == 1:
                # Broadcast single action to batch
                action = action.repeat(batch_size, 1)
            else:
                # Create zero actions with correct batch size
                action = torch.zeros(batch_size, self.action_dim, device=self.device)
        
        # Handle neighbor actions - ensure correct batch size and shape
        if neighbor_actions is None:
            neighbor_actions = torch.zeros(batch_size, self.num_agents, device=self.device)
        else:
            # Handle batch size mismatch
            if neighbor_actions.size(0) != batch_size:
                if neighbor_actions.size(0) == 1:
                    # Broadcast single set of neighbor actions to batch
                    neighbor_actions = neighbor_actions.repeat(batch_size, 1)
                else:
                    # Create zero neighbor actions with correct batch size
                    neighbor_actions = torch.zeros(batch_size, self.num_agents, device=self.device)
            # Ensure correct number of neighbor dimensions
            elif neighbor_actions.dim() == 1:
                neighbor_actions = neighbor_actions.unsqueeze(0).repeat(batch_size, 1)
            elif neighbor_actions.size(1) != self.num_agents:
                if neighbor_actions.size(1) < self.num_agents:
                    # Pad with zeros
                    padding = torch.zeros(batch_size, self.num_agents - neighbor_actions.size(1), device=self.device)
                    neighbor_actions = torch.cat([neighbor_actions, padding], dim=1)
                else:
                    # Truncate to correct size
                    neighbor_actions = neighbor_actions[:, :self.num_agents]

        # Combine inputs - all tensors should now have matching batch sizes
        combined = torch.cat([belief, latent, action, neighbor_actions], dim=1)

        # Forward pass
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_head(x)

        return q_value 