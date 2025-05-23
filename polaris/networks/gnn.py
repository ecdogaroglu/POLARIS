import torch
import torch.nn as nn
import torch.nn.functional as F

# Add new imports for graph operations
import torch_geometric
from torch_geometric.nn import GATConv, GCNConv

from ..utils.device import get_best_device


class TransformerBeliefProcessor(nn.Module):
    """Transformer-based belief state processor for POLARIS."""

    def __init__(
        self,
        hidden_dim,
        action_dim,
        device=None,
        num_belief_states=None,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    ):
        super(TransformerBeliefProcessor, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_belief_states = num_belief_states
        self.nhead = nhead
        self.num_layers = num_layers

        # For continuous signals, the signal will be a single value
        # For discrete signals, the signal will be one-hot encoded with num_belief_states
        # We'll handle both cases by detecting the input dimension
        self.supports_continuous_signal = True

        # Signal-only input dimensions (no neighbor actions)
        self.continuous_signal_dim = 1  # Just the continuous signal
        self.discrete_signal_dim = num_belief_states  # Just the discrete signal

        # Input projections to match hidden dimension (for signal-only processing)
        self.signal_only_projection = nn.Linear(self.discrete_signal_dim, hidden_dim)
        self.continuous_signal_projection = nn.Linear(
            self.continuous_signal_dim, hidden_dim
        )

        # Positional encoding for transformer
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

        # Add softmax head for belief distribution
        self.belief_head = nn.Linear(hidden_dim, num_belief_states)

        # Initialize parameters
        nn.init.xavier_normal_(self.signal_only_projection.weight)
        nn.init.constant_(self.signal_only_projection.bias, 0)
        nn.init.xavier_normal_(self.continuous_signal_projection.weight)
        nn.init.constant_(self.continuous_signal_projection.bias, 0)
        nn.init.xavier_normal_(self.belief_head.weight)
        nn.init.constant_(self.belief_head.bias, 0)
        nn.init.normal_(self.pos_encoder, mean=0.0, std=0.02)

    def standardize_belief_state(self, belief):
        """Ensure belief state has consistent shape [1, batch_size, hidden_dim]."""
        if belief is None:
            return None

        # Add batch dimension if missing
        if belief.dim() == 1:  # [hidden_dim]
            belief = belief.unsqueeze(0)  # [1, hidden_dim]

        # Add sequence dimension if missing
        if belief.dim() == 2:  # [batch_size, hidden_dim]
            belief = belief.unsqueeze(0)  # [1, batch_size, hidden_dim]

        # Transpose if dimensions are in wrong order
        if belief.dim() == 3 and belief.size(0) != 1:
            belief = belief.transpose(0, 1).contiguous()

        return belief

    def forward(self, signal, neighbor_actions=None, current_belief=None):
        """Update belief state based on new observation. Only uses signals, ignores neighbor_actions."""
        # Handle both batched and single inputs

        # Ensure we have batch dimension for signal
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        batch_size = signal.size(0)

        # Detect if signal is continuous (1-dimensional) or discrete (one-hot encoded)
        is_continuous_signal = signal.size(1) == 1

        # Process signal based on type (continuous or discrete)
        if is_continuous_signal:
            # Signal is continuous (1-dimensional)
            # Use only the signal without neighbor actions
            combined = signal

            # Add sequence dimension
            combined = combined.unsqueeze(1).to(self.device)  # [batch_size, 1, 1]

            # Project input using the projection for continuous signals
            # We'll need a separate projection for just the continuous signal
            projected = self.continuous_signal_projection(combined)
        else:
            # Signal is discrete (one-hot encoded)
            # Ensure signal has correct dimensions
            if signal.size(1) != self.num_belief_states:
                signal = signal[:, : self.num_belief_states]

            # Use only the signal without neighbor actions
            combined = signal

            # Add sequence dimension (Transformer expects [batch, seq_len, features])
            combined = combined.unsqueeze(1).to(
                self.device
            )  # [batch_size, 1, num_belief_states]

            # Project input using the projection for discrete signals
            projected = self.signal_only_projection(combined)

        # Initialize or standardize current_belief
        if current_belief is None:
            current_belief = torch.zeros(
                1, batch_size, self.hidden_dim, device=self.device
            )
        else:
            current_belief = self.standardize_belief_state(current_belief)

        # Add positional encoding
        projected = projected + self.pos_encoder

        # If we have a current belief, we can use it as context
        if current_belief is not None:
            # Reshape current_belief to [batch_size, 1, hidden_dim]
            context = current_belief.transpose(0, 1)
            # Concatenate with projected input to form sequence
            sequence = torch.cat([context, projected], dim=1)
        else:
            sequence = projected

        # Process through transformer with appropriate mode (training or evaluation)
        # In evaluation mode, this will use different behavior for dropout
        with torch.set_grad_enabled(self.training):
            transformer_output = self.transformer_encoder(sequence)

            # Take the last token's output as the new belief state
            new_belief = transformer_output[:, -1:, :].transpose(0, 1)

            # Calculate belief distribution
            logits = self.belief_head(new_belief.squeeze(0))

            # Check if we're using continuous signal (determined earlier in forward pass)
            if is_continuous_signal:
                # For continuous signals, we need to output a single value
                # We'll use the first dimension as our continuous prediction
                belief_distribution = logits[:, :1]  # Just take the first element
            else:
                # For discrete signals, use softmax to get a distribution
                temperature = 1  # Temperature for softmax
                belief_distribution = F.softmax(logits / temperature, dim=-1)

            # Ensure new_belief maintains shape [1, batch_size, hidden_dim]
            new_belief = self.standardize_belief_state(new_belief)

        return new_belief, belief_distribution


class EncoderNetwork(nn.Module):
    """Encoder network for inference of other agents' policies."""

    def __init__(
        self,
        action_dim,
        latent_dim,
        hidden_dim,
        num_agents,
        device=None,
        num_belief_states=None,
    ):
        # Use the best available device if none is specified

        super(EncoderNetwork, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.num_belief_states = num_belief_states

        # Support for both discrete and continuous signals
        # For continuous signals, signal dimension is 1
        # For discrete signals, signal dimension is num_belief_states

        # Maximum input dim (for one-hot encoded signals)
        max_input_dim = (
            num_belief_states
            + action_dim * num_agents
            + 1
            + num_belief_states
            + latent_dim
        )

        # Minimum input dim (for continuous signals)
        min_input_dim = 1 + action_dim * num_agents + 1 + 1 + latent_dim

        # Create separate networks for each case
        self.fc1_discrete = nn.Linear(max_input_dim, hidden_dim)
        self.fc1_continuous = nn.Linear(min_input_dim, hidden_dim)

        # Shared layers after initial projection
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Initialize parameters
        nn.init.xavier_normal_(self.fc1_discrete.weight)
        nn.init.constant_(self.fc1_discrete.bias, 0)
        nn.init.xavier_normal_(self.fc1_continuous.weight)
        nn.init.constant_(self.fc1_continuous.bias, 0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc_mean.weight)
        nn.init.xavier_normal_(self.fc_logvar.weight)

        # Add softmax head for opponent belief distribution
        self.opponent_belief_head = nn.Linear(hidden_dim, num_belief_states)
        nn.init.xavier_normal_(self.opponent_belief_head.weight)
        nn.init.constant_(self.opponent_belief_head.bias, 0)

    def forward(self, signal, actions, reward, next_signal, current_latent):
        """Encode the state into a latent distribution."""
        signal = signal.to(self.device)
        actions = actions.to(self.device)
        next_signal = next_signal.to(self.device)
        current_latent = current_latent.to(self.device)

        # Handle different dimensions
        if current_latent.dim() == 3:  # [batch_size, 1, latent_dim]
            current_latent = current_latent.squeeze(1)

        # Ensure all inputs have batch dimension and are 2D tensors
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        # Handle reward which could be a scalar or tensor
        if isinstance(reward, (int, float)):
            reward = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        elif isinstance(reward, torch.Tensor):
            if reward.dim() == 0:
                reward = reward.unsqueeze(0).unsqueeze(0)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1)

        if next_signal.dim() == 1:
            next_signal = next_signal.unsqueeze(0)

        if current_latent.dim() == 1:
            current_latent = current_latent.unsqueeze(0)

        # Detect if signals are continuous (1D) or discrete (one-hot)
        is_continuous = signal.size(1) == 1 and next_signal.size(1) == 1

        # Forward pass based on input type
        if is_continuous:
            # For continuous signals
            combined = torch.cat(
                [
                    signal,  # 1D continuous signal
                    actions,
                    reward,
                    next_signal,  # 1D continuous next signal
                    current_latent,
                ],
                dim=1,
            ).to(self.device)

            # Use the network for continuous inputs
            x = F.relu(self.fc1_continuous(combined))
        else:
            # For discrete (one-hot encoded) signals
            combined = torch.cat(
                [
                    signal,  # One-hot encoded signal
                    actions,
                    reward,
                    next_signal,  # One-hot encoded next signal
                    current_latent,
                ],
                dim=1,
            ).to(self.device)

            # Use the network for discrete inputs
            x = F.relu(self.fc1_discrete(combined))

        # Shared forward pass for the rest of the network
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        # Calculate opponent belief distribution
        logits = self.opponent_belief_head(x)
        temperature = 1  # Temperature for softmax
        opponent_belief_distribution = F.softmax(logits / temperature, dim=-1)

        return mean, logvar, opponent_belief_distribution


class DecoderNetwork(nn.Module):
    """Decoder network for predicting other agents' actions."""

    def __init__(
        self,
        action_dim,
        latent_dim,
        hidden_dim,
        num_agents,
        num_belief_states,
        device=None,
    ):
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        super(DecoderNetwork, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.num_belief_states = num_belief_states

        # Store dimensions for debugging
        self.observation_dim = num_belief_states
        self.latent_dim = latent_dim

        # Combined input: observation and latent
        input_dim = num_belief_states + latent_dim

        # Decoder network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim * num_agents)

        # Initialize parameters
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, signal, latent):
        signal = signal.to(self.device)
        latent = latent.to(self.device)

        """Predict peer actions from observation and latent."""
        # Handle different dimensions
        if latent.dim() == 3:  # [batch_size, 1, latent_dim]
            latent = latent.squeeze(0)

        # Create a new input tensor with the correct dimensions
        batch_size = signal.size(0) if signal.dim() > 1 else 1

        # Ensure latent has the right shape
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        # Combine inputs
        combined = torch.cat([signal, latent], dim=1)

        # Forward pass
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)

        return action_logits


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
        input_dim = belief_dim

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
            combined = torch.cat([belief], dim=1)
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
        min_log_std = -3
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


class TemporalGNN(nn.Module):
    """Graph Neural Network with Temporal Attention for neighbor action inference."""

    def __init__(
        self,
        hidden_dim,
        action_dim,
        latent_dim,
        num_agents,
        device=None,
        num_belief_states=None,
        num_gnn_layers=2,
        num_attn_heads=4,
        dropout=0.1,
        temporal_window_size=5,
    ):
        super(TemporalGNN, self).__init__()
        self.device = device if device is not None else get_best_device()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_agents = num_agents
        self.num_belief_states = num_belief_states
        self.temporal_window_size = temporal_window_size
        self.num_attn_heads = num_attn_heads

        # Support for both continuous and discrete signals
        # For continuous signals, the node feature dimension is 1 + action_dim
        # For discrete signals, the node feature dimension is num_belief_states + action_dim
        self.continuous_node_feat_dim = 1 + action_dim  # continuous signal + action
        self.discrete_node_feat_dim = (
            num_belief_states + action_dim
        )  # belief state + action

        # Track which node feature dimension is being used
        self.is_using_continuous = False

        # Create separate GNN layers for continuous and discrete signals
        # Graph layers for discrete signals (using Graph Attention Networks)
        self.discrete_gnn_layers = nn.ModuleList()
        self.discrete_gnn_layers.append(
            GATConv(
                self.discrete_node_feat_dim,
                hidden_dim,
                heads=num_attn_heads,
                dropout=dropout,
            )
        )

        # Additional GNN layers for discrete signals
        for i in range(num_gnn_layers - 1):
            self.discrete_gnn_layers.append(
                GATConv(
                    hidden_dim * num_attn_heads,
                    hidden_dim,
                    heads=num_attn_heads,
                    dropout=dropout,
                )
            )

        # Graph layers for continuous signals (using Graph Attention Networks)
        self.continuous_gnn_layers = nn.ModuleList()
        self.continuous_gnn_layers.append(
            GATConv(
                self.continuous_node_feat_dim,
                hidden_dim,
                heads=num_attn_heads,
                dropout=dropout,
            )
        )

        # Additional GNN layers for continuous signals
        for i in range(num_gnn_layers - 1):
            self.continuous_gnn_layers.append(
                GATConv(
                    hidden_dim * num_attn_heads,
                    hidden_dim,
                    heads=num_attn_heads,
                    dropout=dropout,
                )
            )

        # Temporal attention layer
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * num_attn_heads,
            num_heads=num_attn_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Calculate feature dimension after all GNN layers
        self.feature_dim = hidden_dim * num_attn_heads

        # Output projection for latent space
        self.latent_mean = nn.Linear(self.feature_dim, latent_dim)
        self.latent_logvar = nn.Linear(self.feature_dim, latent_dim)

        # Output projection for action prediction
        self.action_predictor = nn.Linear(latent_dim, action_dim * num_agents)

        # Belief distribution head
        self.belief_head = nn.Linear(self.feature_dim, num_belief_states)

        # Feature adapter for aligning dimensions when combining GNN output with latent
        self.feature_adapter = nn.Linear(self.feature_dim, latent_dim)

        # Temporal memory buffer for storing past node features and edge indices
        self.temporal_memory = {
            "node_features": [],
            "edge_indices": [],
            "attention_mask": None,
        }

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize network parameters."""
        # Initialize discrete GNN layers
        for layer in self.discrete_gnn_layers:
            if hasattr(layer, "lin"):
                nn.init.xavier_normal_(layer.lin.weight)
            if hasattr(layer, "att"):
                nn.init.xavier_normal_(layer.att)

        # Initialize continuous GNN layers
        for layer in self.continuous_gnn_layers:
            if hasattr(layer, "lin"):
                nn.init.xavier_normal_(layer.lin.weight)
            if hasattr(layer, "att"):
                nn.init.xavier_normal_(layer.att)

        nn.init.xavier_normal_(self.latent_mean.weight)
        nn.init.constant_(self.latent_mean.bias, 0)
        nn.init.xavier_normal_(self.latent_logvar.weight)
        nn.init.constant_(self.latent_logvar.bias, 0)
        nn.init.xavier_normal_(self.action_predictor.weight)
        nn.init.constant_(self.action_predictor.bias, 0)
        nn.init.xavier_normal_(self.belief_head.weight)
        nn.init.constant_(self.belief_head.bias, 0)
        nn.init.xavier_normal_(self.feature_adapter.weight)
        nn.init.zeros_(self.feature_adapter.bias)

    def _construct_graph(self, signals, neighbor_actions, agent_id=0):
        """
        Construct a graph from signals and neighbor actions.

        Args:
            signals: Tensor of shape [batch_size, num_belief_states] or [batch_size, 1] for continuous signals
            neighbor_actions: Tensor of shape [batch_size, num_agents] or [batch_size, num_agents * action_dim]
            agent_id: ID of the current agent

        Returns:
            node_features: Tensor of node features
            edge_index: Tensor of edge indices
            is_continuous: Whether the signal is continuous
        """
        batch_size = signals.size(0)

        # Check if signal is continuous (1D) or discrete (one-hot)
        is_continuous_signal = signals.size(1) == 1

        # Store this for later use
        self.is_using_continuous = is_continuous_signal

        # Check if we're using continuous or discrete actions based on neighbor_actions shape
        using_continuous_actions = neighbor_actions.size(1) == self.num_agents

        if using_continuous_actions:
            # Continuous actions format: [batch_size, num_agents]
            neighbor_actions_reshaped = neighbor_actions
        else:
            # Discrete actions format: [batch_size, num_agents * action_dim]
            # Reshape to [batch_size, num_agents, action_dim]
            neighbor_actions_reshaped = neighbor_actions.view(
                batch_size, self.num_agents, self.action_dim
            )

        # Create node features by concatenating belief state with actions
        # For each agent, we'll create a node with its own feature
        node_features = []

        # Add the current agent's node first
        for b in range(batch_size):
            # Current agent's features: concatenate signal with its own action
            if using_continuous_actions:
                # For continuous case, we expand the single value to match dimensions
                agent_action = torch.zeros(self.action_dim, device=signals.device)
                agent_action[0] = neighbor_actions_reshaped[b, agent_id]
            else:
                agent_action = neighbor_actions_reshaped[b, agent_id]

            # Concatenate signal with action based on signal type
            if is_continuous_signal:
                # If signal is continuous, we need to make sure it's treated properly
                # Convert to the right shape for concatenation
                agent_signal = signals[b].view(-1)  # Make sure it's flattened
                agent_features = torch.cat([agent_signal, agent_action], dim=-1)
            else:
                # Discrete signal case (one-hot encoded)
                agent_features = torch.cat([signals[b], agent_action], dim=-1)

            node_features.append(agent_features)

            # Add neighbor nodes
            for n in range(self.num_agents):
                if n != agent_id:
                    # For neighbor agents: concatenate zeros (no belief) with actions
                    if using_continuous_actions:
                        # For continuous case, expand the single value
                        neighbor_action = torch.zeros(
                            self.action_dim, device=signals.device
                        )
                        neighbor_action[0] = neighbor_actions_reshaped[b, n]
                    else:
                        neighbor_action = neighbor_actions_reshaped[b, n]

                    # Create zero belief for neighbors based on signal type
                    if is_continuous_signal:
                        # For continuous signal, just use zeros of the same shape
                        neighbor_belief = torch.zeros_like(signals[b])
                    else:
                        # For discrete signal, use zero one-hot encoding
                        neighbor_belief = torch.zeros_like(signals[b])

                    neighbor_features = torch.cat(
                        [neighbor_belief, neighbor_action], dim=-1
                    )
                    node_features.append(neighbor_features)

        # Stack node features
        node_features = torch.stack(node_features).to(self.device)

        # Create fully connected edge indices (each agent connects to every other agent)
        edge_indices = []
        nodes_per_batch = self.num_agents

        for b in range(batch_size):
            batch_offset = b * nodes_per_batch
            for i in range(nodes_per_batch):
                for j in range(nodes_per_batch):
                    if i != j:  # No self-loops
                        edge_indices.append([batch_offset + i, batch_offset + j])

        # Convert to tensor
        edge_index = (
            torch.tensor(edge_indices, dtype=torch.long)
            .t()
            .contiguous()
            .to(self.device)
        )

        return node_features, edge_index, is_continuous_signal

    def _update_temporal_memory(self, node_features, edge_index):
        """Update temporal memory with new graph data."""
        # Store batch size for this entry for debugging
        batch_size = node_features.size(0) // self.num_agents

        # Add new features and edges to memory
        self.temporal_memory["node_features"].append(
            node_features.detach()
        )  # Detach to avoid memory leak
        self.temporal_memory["edge_indices"].append(edge_index.detach())

        # Maintain fixed window size
        while len(self.temporal_memory["node_features"]) > self.temporal_window_size:
            self.temporal_memory["node_features"].pop(0)
            self.temporal_memory["edge_indices"].pop(0)

        # Update attention mask for temporal attention
        seq_len = len(self.temporal_memory["node_features"])
        self.temporal_memory["attention_mask"] = torch.ones(
            seq_len, seq_len, device=self.device
        )

        # Make it causal (can only attend to current and past frames)
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:  # Future frames
                    self.temporal_memory["attention_mask"][i, j] = 0

    def _apply_gnn(self, node_features, edge_index):
        """Apply GNN layers to node features."""
        x = node_features

        # Apply GNN layers based on signal type
        if self.is_using_continuous:
            # Use continuous GNN layers
            for layer in self.continuous_gnn_layers:
                x = layer(x, edge_index)
                x = F.relu(x)
        else:
            # Use discrete GNN layers
            for layer in self.discrete_gnn_layers:
                x = layer(x, edge_index)
                x = F.relu(x)

        return x

    def _apply_temporal_attention(self):
        """Apply temporal attention to sequence of GNN outputs."""
        # Stack temporal sequence of GNN outputs
        if len(self.temporal_memory["node_features"]) == 0:
            # If empty, return zero tensor
            batch_size = 1  # Default batch size
            return torch.zeros(
                batch_size,
                self.hidden_dim * self.gnn_layers[-1].heads,
                device=self.device,
            )

        # Process each frame with GNN
        temporal_gnn_outputs = []
        batch_sizes = []

        for i in range(len(self.temporal_memory["node_features"])):
            node_feats = self.temporal_memory["node_features"][i]
            edge_idx = self.temporal_memory["edge_indices"][i]

            # Apply GNN to get node embeddings
            gnn_output = self._apply_gnn(node_feats, edge_idx)

            # Extract only the ego agent's node representation (first node of each batch)
            local_batch_size = node_feats.size(0) // self.num_agents
            batch_sizes.append(local_batch_size)
            ego_indices = torch.arange(
                0, node_feats.size(0), self.num_agents, device=self.device
            )
            ego_output = gnn_output[ego_indices]
            temporal_gnn_outputs.append(ego_output)

        # Check if all batch sizes are the same
        if len(set(batch_sizes)) > 1:
            # Batch sizes are different, need to make them consistent
            # Use the latest batch size as the target
            target_batch_size = batch_sizes[-1]

            # Adjust tensors to match the target batch size
            for i in range(len(temporal_gnn_outputs)):
                if batch_sizes[i] != target_batch_size:
                    # If this tensor has a different batch size, we need to adapt it
                    if batch_sizes[i] == 1 and target_batch_size > 1:
                        # Repeat the single sample to match the batch size
                        temporal_gnn_outputs[i] = temporal_gnn_outputs[i].repeat(
                            target_batch_size, 1
                        )
                    elif batch_sizes[i] > 1 and target_batch_size == 1:
                        # Take the mean of the batch
                        temporal_gnn_outputs[i] = torch.mean(
                            temporal_gnn_outputs[i], dim=0, keepdim=True
                        )
                    else:
                        # For other cases, replace with zeros of the right size
                        # This is less ideal but prevents crashes
                        temporal_gnn_outputs[i] = torch.zeros(
                            target_batch_size,
                            temporal_gnn_outputs[i].size(1),
                            device=self.device,
                        )

        # Now all tensors have the same batch size and can be stacked
        sequence = torch.stack(
            temporal_gnn_outputs, dim=1
        )  # [batch_size, seq_len, hidden_dim]

        # Update attention mask if needed
        seq_len = len(temporal_gnn_outputs)
        if (
            self.temporal_memory["attention_mask"] is None
            or self.temporal_memory["attention_mask"].size(0) != seq_len
        ):
            self.temporal_memory["attention_mask"] = torch.ones(
                seq_len, seq_len, device=self.device
            )

            # Make it causal (can only attend to current and past frames)
            for i in range(seq_len):
                for j in range(seq_len):
                    if j > i:  # Future frames
                        self.temporal_memory["attention_mask"][i, j] = 0

        # Apply temporal self-attention
        attn_output, _ = self.temporal_attention(
            sequence,
            sequence,
            sequence,
            attn_mask=self.temporal_memory["attention_mask"],
        )

        # Return the most recent output
        return attn_output[:, -1]

    def forward(
        self, signal, neighbor_actions, reward, next_signal, current_latent=None
    ):
        """
        Forward pass through the Temporal GNN.

        Args:
            signal: Current signal/observation (can be continuous or discrete)
            neighbor_actions: Actions of all agents (can be continuous or discrete)
            reward: Reward received
            next_signal: Next signal/observation (can be continuous or discrete)
            current_latent: Current latent state (optional)

        Returns:
            mean: Mean of latent distribution
            logvar: Log variance of latent distribution
            belief_distribution: Belief distribution over states
        """
        # Ensure inputs have batch dimension
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        if neighbor_actions.dim() == 1:
            neighbor_actions = neighbor_actions.unsqueeze(0)
        if isinstance(reward, (int, float)):
            reward = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        elif isinstance(reward, torch.Tensor):
            if reward.dim() == 0:
                reward = reward.unsqueeze(0).unsqueeze(0)
            elif reward.dim() == 1:
                reward = reward.unsqueeze(1)
        if next_signal.dim() == 1:
            next_signal = next_signal.unsqueeze(0)

        # Make sure everything is on the correct device
        signal = signal.to(self.device)
        neighbor_actions = neighbor_actions.to(self.device)
        reward = reward.to(self.device)
        next_signal = next_signal.to(self.device)

        # Detect if we're using continuous signals
        is_continuous_signal = signal.size(1) == 1

        # Construct graph from current observation and actions
        node_features, edge_index, is_continuous = self._construct_graph(
            signal, neighbor_actions
        )

        # Update temporal memory
        self._update_temporal_memory(node_features, edge_index)

        # Apply GNN with temporal attention
        gnn_output = self._apply_temporal_attention()

        # Generate latent distribution parameters
        mean = self.latent_mean(gnn_output)
        logvar = self.latent_logvar(gnn_output)

        # Calculate belief distribution
        logits = self.belief_head(gnn_output)
        temperature = 0.5  # Temperature for softmax
        belief_distribution = F.softmax(logits / temperature, dim=-1)

        return mean, logvar, belief_distribution

    def predict_actions(self, signal, latent):
        """
        Predict neighbor actions based on current signal and latent state.

        Args:
            signal: Current signal/observation (can be continuous or discrete)
            latent: Current latent state

        Returns:
            action_logits: Logits for neighbor actions
        """
        # Ensure inputs have batch dimension
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        # Handle different latent dimensions
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)  # [1, latent_dim]
        elif latent.dim() == 3:
            # If latent is [batch_size, seq_len, latent_dim], take the last sequence element
            latent = latent[:, -1, :]  # [batch_size, latent_dim]

        # Make sure everything is on the correct device
        signal = signal.to(self.device)
        latent = latent.to(self.device)

        # Detect if signal is continuous (1D) or discrete (one-hot)
        is_continuous_signal = signal.size(1) == 1
        self.is_using_continuous = is_continuous_signal

        # Construct a dummy graph with just the signal
        # We'll use zeros for neighbor actions since we're trying to predict them
        batch_size = signal.size(0)
        dummy_actions = torch.zeros(
            batch_size, self.num_agents * self.action_dim, device=self.device
        )
        node_features, edge_index, _ = self._construct_graph(signal, dummy_actions)

        # Process through GNN
        gnn_output = self._apply_gnn(node_features, edge_index)

        # Extract only the ego agent's node
        batch_size = node_features.size(0) // self.num_agents
        ego_indices = torch.arange(
            0, node_features.size(0), self.num_agents, device=self.device
        )
        ego_output = gnn_output[ego_indices]

        # Ensure latent has the same batch size
        if latent.size(0) != ego_output.size(0):
            if latent.size(0) == 1 and ego_output.size(0) > 1:
                # Expand latent to match batch size
                latent = latent.expand(ego_output.size(0), -1)
            elif latent.size(0) > 1 and ego_output.size(0) == 1:
                # Take mean of latent
                latent = torch.mean(latent, dim=0, keepdim=True)

        # Project ego_output to latent dimension using the feature adapter
        ego_output = self.feature_adapter(ego_output)

        # Combine with latent
        combined = ego_output + latent  # Simple addition, could be more complex

        # Predict actions
        action_logits = self.action_predictor(combined)

        return action_logits

    def reset_memory(self):
        """Reset temporal memory."""
        self.temporal_memory = {
            "node_features": [],
            "edge_indices": [],
            "attention_mask": None,
        }
