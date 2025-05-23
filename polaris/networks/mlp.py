import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: nn.Module = nn.ReLU,
    output_activation: Optional[nn.Module] = None
) -> nn.Sequential:
    """Build a multi-layer perceptron."""
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim
        
    layers.append(nn.Linear(prev_dim, output_dim))
    
    if output_activation is not None:
        layers.append(output_activation())
        
    return nn.Sequential(*layers)

class EncoderNetwork(nn.Module):
    """Encoder network for inference."""
    def __init__(
        self,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_agents: int,
        device: torch.device,
        num_belief_states: int
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.num_belief_states = num_belief_states
        
        # Input dimensions
        discrete_input_dim = num_belief_states + action_dim * num_agents + 1 + num_belief_states + latent_dim
        continuous_input_dim = 1 + action_dim * num_agents + 1 + 1 + latent_dim
        
        # Networks
        self.fc1_discrete = nn.Linear(discrete_input_dim, hidden_dim)
        self.fc1_continuous = nn.Linear(continuous_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.opponent_belief_head = nn.Linear(hidden_dim, num_belief_states)
        
        self._init_parameters()
        
    def _init_parameters(self):
        for module in [self.fc1_discrete, self.fc1_continuous, self.fc2, 
                      self.fc_mean, self.fc_logvar, self.opponent_belief_head]:
            if hasattr(module, 'weight'):
                nn.init.xavier_normal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, signal, actions, reward, next_signal, current_latent):
        """Encode observations into latent distribution."""
        # Ensure proper dimensions
        signal = signal.to(self.device)
        actions = actions.to(self.device)
        reward = reward.to(self.device)
        next_signal = next_signal.to(self.device)
        current_latent = current_latent.to(self.device)
        
        # Handle dimensions
        if current_latent.dim() == 3:
            current_latent = current_latent.squeeze(1)
            
        # Ensure batch dimensions
        for tensor in [signal, actions, next_signal, current_latent]:
            if tensor.dim() == 1:
                tensor.unsqueeze_(0)
                
        # Handle reward
        if isinstance(reward, (int, float)):
            reward = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        elif reward.dim() == 0:
            reward = reward.unsqueeze(0).unsqueeze(0)
        elif reward.dim() == 1:
            reward = reward.unsqueeze(1)
            
        # Check if continuous
        is_continuous = signal.size(1) == 1 and next_signal.size(1) == 1
        
        # Concatenate inputs
        if is_continuous:
            combined = torch.cat([signal, actions, reward, next_signal, current_latent], dim=1)
            x = F.relu(self.fc1_continuous(combined))
        else:
            combined = torch.cat([signal, actions, reward, next_signal, current_latent], dim=1)
            x = F.relu(self.fc1_discrete(combined))
            
        x = F.relu(self.fc2(x))
        
        # Output distributions
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        
        # Opponent belief
        logits = self.opponent_belief_head(x)
        opponent_belief = F.softmax(logits, dim=-1)
        
        return mean, logvar, opponent_belief

class DecoderNetwork(nn.Module):
    """Decoder network for action prediction."""
    def __init__(
        self,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_agents: int,
        num_belief_states: int,
        device: torch.device
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        input_dim = num_belief_states + latent_dim
        
        self.network = build_mlp(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=action_dim * num_agents
        )
        
    def forward(self, signal, latent):
        """Predict actions from signal and latent."""
        signal = signal.to(self.device)
        latent = latent.to(self.device)
        
        if latent.dim() == 3:
            latent = latent.squeeze(0)
            
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            
        combined = torch.cat([signal, latent], dim=1)
        return self.network(combined)

class PolicyNetwork(nn.Module):
    """Policy network for discrete actions."""
    def __init__(
        self,
        belief_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: torch.device
    ):
        super().__init__()
        self.device = device
        
        self.network = build_mlp(
            input_dim=belief_dim + latent_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=action_dim
        )
        
    def forward(self, belief, latent):
        """Compute action logits."""
        if belief.dim() == 3:
            belief = belief.squeeze(0)
        if latent.dim() == 3:
            latent = latent.squeeze(0)
            
        if belief.dim() == 1:
            belief = belief.unsqueeze(0)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            
        combined = torch.cat([belief, latent], dim=1)
        return self.network(combined)

class ContinuousPolicyNetwork(nn.Module):
    """Policy network for continuous actions."""
    def __init__(
        self,
        belief_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: torch.device,
        min_action: float = 0.0,
        max_action: float = 1.0
    ):
        super().__init__()
        self.device = device
        self.min_action = min_action
        self.max_action = max_action
        
        input_dim = belief_dim + latent_dim
        
        self.shared = build_mlp(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=hidden_dim
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize log_std to produce small initial std
        nn.init.constant_(self.log_std_head.bias, -2)
        
    def forward(self, belief, latent):
        """Compute mean and log_std of action distribution."""
        if belief.dim() == 3:
            belief = belief.squeeze(0)
        if latent.dim() == 3:
            latent = latent.squeeze(0)
            
        if belief.dim() == 1:
            belief = belief.unsqueeze(0)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            
        combined = torch.cat([belief, latent], dim=1).to(self.device)
        
        features = self.shared(combined)
        
        # Compute mean and log_std
        mean = torch.sigmoid(self.mean_head(features))
        log_std = torch.sigmoid(self.log_std_head(features))
        
        # Scale outputs
        scaled_mean = self.min_action + (self.max_action - self.min_action) * mean
        scaled_log_std = -3 + 4 * log_std  # Maps to [-3, 1]
        
        return scaled_mean, scaled_log_std
        
    def sample_action(self, belief, latent):
        """Sample action from the policy."""
        mean, log_std = self.forward(belief, latent)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        
        # Clamp action
        action = torch.clamp(action, self.min_action, self.max_action)
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, mean

class QNetwork(nn.Module):
    """Q-network for value estimation."""
    def __init__(
        self,
        belief_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_agents: int,
        device: torch.device
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        
        input_dim = belief_dim + latent_dim + action_dim * num_agents
        
        self.network = build_mlp(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=action_dim
        )
        
    def forward(self, belief, latent, neighbor_actions):
        """Compute Q-values."""
        if belief.dim() == 3:
            belief = belief.squeeze(0)
        if latent.dim() == 3:
            latent = latent.squeeze(0)
            
        combined = torch.cat([belief, latent, neighbor_actions], dim=1)
        return self.network(combined)