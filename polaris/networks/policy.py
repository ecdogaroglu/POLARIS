import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.device import get_best_device


class PolicyNetwork(nn.Module):
    """Policy network for deciding actions."""

    def __init__(self, belief_dim, latent_dim, action_dim, hidden_dim, device=None):
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        super(PolicyNetwork, self).__init__()
        self.device = device
        self.action_dim = action_dim

        # Combined input: belief and latent
        input_dim = belief_dim + latent_dim

        # Policy network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Initialize parameters
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, belief, latent):
        """Compute action logits given belief and latent."""
        # Handle both batched and single inputs
        if belief.dim() == 3:  # [batch_size, 1, belief_dim]
            belief = belief.squeeze(0)
        if latent.dim() == 3:  # [batch_size, 1, latent_dim]
            latent = latent.squeeze(0)

        # For batched inputs
        if belief.dim() == 2 and latent.dim() == 2:
            # Combine inputs along feature dimension
            combined = torch.cat([belief, latent], dim=1)
        # For single inputs
        else:
            # Ensure we have batch dimension
            if belief.dim() == 1:
                belief = belief.unsqueeze(0)
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
            combined = torch.cat([belief, latent], dim=1)

        # Forward pass
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)

        return action_logits


class ContinuousPolicyNetwork(nn.Module):
    """Policy network for continuous actions in strategic experimentation."""

    def __init__(
        self,
        belief_dim,
        latent_dim,
        action_dim,
        hidden_dim,
        device=None,
        min_action=0.0,
        max_action=1.0,
    ):
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        super(ContinuousPolicyNetwork, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.min_action = min_action
        self.max_action = max_action
        self.belief_dim = belief_dim
        self.latent_dim = latent_dim

        # Support both continuous and discrete belief representations
        # For continuous case, belief dimension will be the hidden dimension from transformer
        # For discrete case, belief dimension will be the transformer's hidden dimension

        # In either case, we can use the same network since the belief processor
        # produces a fixed-dimension belief representation regardless of input type

        # Combined input: belief and latent state
        # For strategic experimentation, we can use both belief and latent if available
        input_dim = belief_dim + latent_dim

        # Policy network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Mean and log_std output heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Initialize parameters
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.mean_head.weight)
        nn.init.xavier_normal_(self.log_std_head.weight)

        # Initialize log_std bias to produce small initial std
        nn.init.constant_(self.log_std_head.bias, -2)

    def forward(self, belief, latent):
        """Compute action distribution parameters (mean and log std) given belief and latent."""
        # Handle different input dimensions
        if belief.dim() == 3:  # [seq_len, batch_size, belief_dim]
            belief = belief.squeeze(0)  # Remove sequence dimension
        if belief.dim() == 1:  # [belief_dim]
            belief = belief.unsqueeze(0)  # Add batch dimension

        # Same for latent
        if latent.dim() == 3:  # [seq_len, batch_size, latent_dim]
            latent = latent.squeeze(0)
        if latent.dim() == 1:  # [latent_dim]
            latent = latent.unsqueeze(0)

        # Ensure all inputs are on the correct device
        belief = belief.to(self.device)
        latent = latent.to(self.device)

        # Combine belief and latent states
        combined = torch.cat([belief, latent], dim=1)

        # Forward pass
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))

        # Extract mean and log_std
        mean = torch.sigmoid(self.mean_head(x))  # Sigmoid for [0,1] range
        log_std = torch.sigmoid(self.log_std_head(x))

        # Scale mean to action range
        scaled_mean = self.min_action + (self.max_action - self.min_action) * mean

        # Scale log_std to a range
        min_log_std = -4
        max_log_std = 1
        scaled_log_std = min_log_std + (max_log_std - min_log_std) * log_std

        return scaled_mean, scaled_log_std

    def sample_action(self, belief, latent):
        """Sample an action from the policy distribution."""
        # Ensure both belief and latent are properly formatted
        belief = belief.to(self.device)
        latent = latent.to(self.device)

        # Get distribution parameters
        mean, log_std = self.forward(belief, latent)
        std = log_std.exp()

        # Sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick

        # Apply clipping to ensure actions are bounded
        action = torch.clamp(x_t, self.min_action, self.max_action)

        # Calculate log probability
        log_prob = normal.log_prob(x_t)

        # Return action, log probability, and mean
        return action, log_prob, mean 