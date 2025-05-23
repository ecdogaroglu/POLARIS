import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBeliefProcessor(nn.Module):
    """Transformer-based belief state processor."""
    
    def __init__(
        self, 
        hidden_dim: int,
        action_dim: int,
        device: torch.device,
        num_belief_states: int,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_belief_states = num_belief_states
        
        # Input projections
        self.signal_projection = nn.Linear(num_belief_states, hidden_dim)
        self.continuous_signal_projection = nn.Linear(1, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Output head
        self.belief_head = nn.Linear(hidden_dim, num_belief_states)
        
        self._init_parameters()
        
    def _init_parameters(self):
        nn.init.xavier_normal_(self.signal_projection.weight)
        nn.init.xavier_normal_(self.continuous_signal_projection.weight)
        nn.init.xavier_normal_(self.belief_head.weight)
        nn.init.normal_(self.pos_encoder, mean=0.0, std=0.02)
        
    def standardize_belief_state(self, belief):
        """Ensure belief state has consistent shape."""
        if belief is None:
            return None
            
        if belief.dim() == 1:
            belief = belief.unsqueeze(0).unsqueeze(0)
        elif belief.dim() == 2:
            belief = belief.unsqueeze(0)
            
        return belief
        
    def forward(self, signal, neighbor_actions=None, current_belief=None):
        """Process signal and update belief."""
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
            
        batch_size = signal.size(0)
        is_continuous = signal.size(1) == 1
        
        # Project signal
        if is_continuous:
            projected = self.continuous_signal_projection(signal.unsqueeze(1))
        else:
            projected = self.signal_projection(signal.unsqueeze(1))
            
        # Add positional encoding
        projected = projected + self.pos_encoder
        
        # Include current belief as context
        if current_belief is not None:
            current_belief = self.standardize_belief_state(current_belief)
            context = current_belief.transpose(0, 1)
            sequence = torch.cat([context, projected], dim=1)
        else:
            sequence = projected
            
        # Process through transformer
        output = self.transformer_encoder(sequence)
        new_belief = output[:, -1:, :].transpose(0, 1)
        
        # Get belief distribution
        logits = self.belief_head(new_belief.squeeze(0))
        
        if is_continuous:
            belief_distribution = logits[:, :1]
        else:
            belief_distribution = F.softmax(logits, dim=-1)
            
        return self.standardize_belief_state(new_belief), belief_distribution