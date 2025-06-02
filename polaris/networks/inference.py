import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import warnings

# Add new imports for graph operations
import torch_geometric
from torch_geometric.nn import GATConv, GCNConv

from ..utils.device import get_best_device
from .belief import InvertibleBeliefHead


class AttentionProcessingError(ValueError):
    """Raised when attention weights cannot be processed due to format issues."""
    pass


class TemporalMemoryError(RuntimeError):
    """Raised when temporal memory operations fail due to inconsistent state."""
    pass


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
        flow_layers=4,
    ):
        super(TemporalGNN, self).__init__()
        self.device = device if device is not None else get_best_device()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_agents = num_agents
        self.num_belief_states = num_belief_states
        self.num_gnn_layers = num_gnn_layers
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.temporal_window_size = temporal_window_size

        # Calculate feature dimensions based on input types
        # For discrete signals: num_belief_states + action_dim
        # For continuous signals: 1 + action_dim
        self.discrete_feature_dim = num_belief_states + action_dim
        self.continuous_feature_dim = 1 + action_dim

        # Track which signal type is being used
        self.is_using_continuous = False

        # Create separate GNN layers for discrete and continuous inputs
        # Each layer outputs hidden_dim features
        self.discrete_gnn_layers = nn.ModuleList([
            GATConv(
                in_channels=self.discrete_feature_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim // num_attn_heads,
                heads=num_attn_heads,
                dropout=dropout,
                concat=True,
            )
            for i in range(num_gnn_layers)
        ])

        self.continuous_gnn_layers = nn.ModuleList([
            GATConv(
                in_channels=self.continuous_feature_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim // num_attn_heads,
                heads=num_attn_heads,
                dropout=dropout,
                concat=True,
            )
            for i in range(num_gnn_layers)
        ])

        # After GNN layers, we have hidden_dim features
        self.feature_dim = hidden_dim

        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attn_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output layers
        self.latent_mean = nn.Linear(self.feature_dim, latent_dim)
        self.latent_logvar = nn.Linear(self.feature_dim, latent_dim)

        # Output projection for action prediction
        self.action_predictor = nn.Linear(latent_dim, action_dim * num_agents)

        # Invertible belief distribution head
        self.belief_head = InvertibleBeliefHead(
            hidden_dim=self.feature_dim,
            num_belief_states=num_belief_states,
            num_layers=flow_layers
        )

        # Feature adapter for aligning dimensions when combining GNN output with latent
        self.feature_adapter = nn.Linear(self.feature_dim, latent_dim)

        # Temporal memory buffer for storing past node features and edge indices
        self.temporal_memory = {
            "node_features": [],
            "edge_indices": [],
            "attention_mask": None,
        }

        # Store latest attention weights for analysis
        self.latest_attention_weights = None

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
        nn.init.xavier_normal_(self.feature_adapter.weight)
        nn.init.zeros_(self.feature_adapter.bias)

    def _construct_graph(self, signals, neighbor_actions, agent_id=0):
        """
        Construct a graph from signals and neighbor actions.
        
        The network topology is inferred from the neighbor_actions tensor:
        - Non-zero entries indicate actual neighbors (edges exist)
        - Zero entries indicate non-neighbors (no edges)

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

        # Create edge indices based on the network topology encoded in neighbor_actions
        # Only create edges where neighbor_actions has non-zero values
        edge_indices = []
        nodes_per_batch = self.num_agents

        for b in range(batch_size):
            batch_offset = b * nodes_per_batch
            
            # For each agent, check which neighbors have non-zero actions
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i != j:  # No self-loops
                        # Check if agent i can observe agent j (j has non-zero action in i's observation)
                        if using_continuous_actions:
                            # For continuous actions, check if the action value is non-zero
                            has_connection = neighbor_actions_reshaped[b, j].abs() > 1e-6
                        else:
                            # For discrete actions, check if any action dimension is non-zero
                            has_connection = neighbor_actions_reshaped[b, j].abs().sum() > 1e-6
                        
                        if has_connection:
                            edge_indices.append([batch_offset + i, batch_offset + j])

        # Convert to tensor
        if len(edge_indices) > 0:
            edge_index = (
                torch.tensor(edge_indices, dtype=torch.long)
                .t()
                .contiguous()
                .to(self.device)
            )
        else:
            # If no edges, create empty edge index tensor
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

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
        attention_weights = None

        # Apply GNN layers based on signal type
        if self.is_using_continuous:
            # Use continuous GNN layers
            for i, layer in enumerate(self.continuous_gnn_layers):
                if i == 0:  # Capture attention weights from the first layer
                    x, (edge_index_out, attention_weights) = layer(x, edge_index, return_attention_weights=True)
                else:
                    x = layer(x, edge_index)
                x = F.relu(x)
        else:
            # Use discrete GNN layers
            for i, layer in enumerate(self.discrete_gnn_layers):
                if i == 0:  # Capture attention weights from the first layer
                    x, (edge_index_out, attention_weights) = layer(x, edge_index, return_attention_weights=True)
                else:
                    x = layer(x, edge_index)
                x = F.relu(x)

        # Store attention weights for analysis
        if attention_weights is not None:
            self.latest_attention_weights = attention_weights.detach().cpu().numpy()

        return x

    def _apply_temporal_attention(self):
        """
        Apply temporal attention to sequence of GNN outputs.
        
        Returns:
            Tensor: Attended GNN output for the current timestep
            
        Note:
            Empty temporal memory at initialization is expected and handled gracefully.
            This is not a fallback behavior but proper initialization.
        """
        # Handle expected empty case during initialization
        if len(self.temporal_memory["node_features"]) == 0:
            # This is expected during the first forward pass before any memory is accumulated
            # Return properly shaped zero tensor for initialization
            batch_size = 1  # Default batch size for initialization
            return torch.zeros(
                batch_size,
                self.feature_dim,
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

        # Calculate belief distribution using invertible head
        if is_continuous_signal:
            # For continuous signals, take only the first component
            belief_distribution = self.belief_head(gnn_output)
            belief_distribution = belief_distribution[:, :1]
        else:
            belief_distribution = self.belief_head(gnn_output)

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

    def get_attention_weights(self):
        """Get the latest attention weights as a numpy array."""
        return self.latest_attention_weights

    def get_attention_matrix(self, edge_index, attention_weights):
        """Convert edge-based attention weights to a full adjacency matrix."""
        if attention_weights is None or edge_index is None:
            return None
            
        try:
            # Create attention matrix
            attention_matrix = np.zeros((self.num_agents, self.num_agents))
            
            # Convert edge_index and attention_weights to numpy if they're tensors
            if hasattr(edge_index, 'cpu'):
                edge_index = edge_index.cpu().numpy()
            if hasattr(attention_weights, 'cpu'):
                attention_weights = attention_weights.cpu().numpy()
            
            # Handle attention weights from multiple heads
            if attention_weights.ndim > 1:
                # For multiple heads, average across heads
                # Expected shape: [num_edges, num_heads] or [num_edges * num_heads]
                if attention_weights.shape[1] == self.num_attn_heads:
                    # Shape is [num_edges, num_heads] - average across heads
                    attention_weights = np.mean(attention_weights, axis=1)
                elif attention_weights.shape[0] % self.num_attn_heads == 0:
                    # Shape might be [num_edges * num_heads, 1] - reshape and average
                    num_edges = attention_weights.shape[0] // self.num_attn_heads
                    attention_weights = attention_weights.reshape(num_edges, self.num_attn_heads)
                    attention_weights = np.mean(attention_weights, axis=1)
                else:
                    # Raise error instead of fallback
                    raise AttentionProcessingError(
                        f"Cannot process attention weights with shape {attention_weights.shape}. "
                        f"Expected either [num_edges, {self.num_attn_heads}] or [num_edges * {self.num_attn_heads}, 1], "
                        f"but got incompatible dimensions."
                    )
                    
            # Ensure edge_index has the right shape
            if edge_index.ndim == 1:
                raise AttentionProcessingError(
                    f"Edge index has incompatible shape {edge_index.shape}. "
                    f"Expected 2D array with shape [2, num_edges]."
                )
                
            # Fill the attention matrix
            for i, (src, tgt) in enumerate(edge_index.T):
                if i < len(attention_weights):
                    # Convert batch-level indices to agent-level indices
                    src_agent = int(src) % self.num_agents
                    tgt_agent = int(tgt) % self.num_agents
                    
                    # Only fill if indices are within bounds
                    if (0 <= src_agent < self.num_agents and 
                        0 <= tgt_agent < self.num_agents):
                        attention_matrix[src_agent, tgt_agent] = attention_weights[i]
            
            # CRITICAL FIX: Properly normalize attention weights
            # Each row should represent attention weights from a source node
            # and should sum to 1 (proper attention mechanism)
            for i in range(self.num_agents):
                row_sum = np.sum(attention_matrix[i, :])
                if row_sum > 1e-8:  # Avoid division by zero
                    attention_matrix[i, :] = attention_matrix[i, :] / row_sum
                else:
                    # If no outgoing attention, set uniform attention to connected nodes
                    num_connections = np.sum(attention_matrix[i, :] > 0)
                    if num_connections > 0:
                        attention_matrix[i, attention_matrix[i, :] > 0] = 1.0 / num_connections
                    
            return attention_matrix
            
        except AttentionProcessingError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Convert any other error to our custom error type
            raise AttentionProcessingError(f"Could not process attention weights: {e}")

    def get_belief_log_prob(self, hidden_features: torch.Tensor) -> torch.Tensor:
        """Get log probability of belief transformation."""
        return self.belief_head.log_prob(hidden_features)
    
    def inverse_belief_transform(self, belief_distribution: torch.Tensor) -> torch.Tensor:
        """Inverse transform from belief distribution to hidden features."""
        return self.belief_head.inverse(belief_distribution) 