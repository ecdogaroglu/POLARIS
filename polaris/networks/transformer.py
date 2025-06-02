"""
Transformer-based belief processing with normalizing flows for POLARIS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class InverseTransformationError(RuntimeError):
    """Raised when inverse transformation fails due to numerical issues."""
    pass


class CouplingLayer(nn.Module):
    """Coupling layer for normalizing flows."""
    
    def __init__(self, input_dim: int, hidden_dim: int, mask: torch.Tensor):
        super().__init__()
        self.mask = mask
        
        # Networks for scale and translation
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Bounded output for stability
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through coupling layer."""
        if not reverse:
            # Forward transformation
            x_masked = x * self.mask
            scale = self.scale_net(x_masked)
            translate = self.translate_net(x_masked)
            
            # Apply transformation only to unmasked dimensions
            y = x_masked + (1 - self.mask) * (x * torch.exp(scale) + translate)
            
            # Log determinant of Jacobian
            log_det = torch.sum((1 - self.mask) * scale, dim=-1)
            
            return y, log_det
        else:
            # Inverse transformation
            y_masked = x * self.mask
            scale = self.scale_net(y_masked)
            translate = self.translate_net(y_masked)
            
            # Apply inverse transformation
            x = y_masked + (1 - self.mask) * (x - translate) * torch.exp(-scale)
            
            # Log determinant of Jacobian (negative for inverse)
            log_det = -torch.sum((1 - self.mask) * scale, dim=-1)
            
            return x, log_det


class InvertibleBeliefHead(nn.Module):
    """Invertible network for belief distribution using normalizing flows."""
    
    def __init__(self, hidden_dim: int, num_belief_states: int, num_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_belief_states = num_belief_states
        self.num_layers = num_layers
        
        # Create alternating masks for coupling layers
        self.masks = []
        for i in range(num_layers):
            mask = torch.zeros(hidden_dim)
            mask[::2] = 1 if i % 2 == 0 else 0
            mask[1::2] = 0 if i % 2 == 0 else 1
            self.register_buffer(f'mask_{i}', mask)
            self.masks.append(mask)
        
        # Coupling layers
        self.coupling_layers = nn.ModuleList([
            CouplingLayer(hidden_dim, hidden_dim // 2, mask) 
            for mask in self.masks
        ])
        
        # Final projection to belief space
        self.belief_projection = nn.Linear(hidden_dim, num_belief_states)
        
        # Base distribution (standard normal)
        self.register_buffer('base_loc', torch.zeros(hidden_dim))
        self.register_buffer('base_scale', torch.ones(hidden_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform hidden representation to belief distribution."""
        batch_size = x.size(0)
        
        # Apply coupling layers
        z = x
        total_log_det = torch.zeros(batch_size, device=x.device)
        
        for layer in self.coupling_layers:
            z, log_det = layer(z, reverse=False)
            total_log_det += log_det
        
        # Project to belief space and apply softmax
        belief_logits = self.belief_projection(z)
        belief_distribution = F.softmax(belief_logits, dim=-1)
        
        return belief_distribution
    
    def inverse(self, belief_distribution: torch.Tensor) -> torch.Tensor:
        """Inverse transformation from belief distribution to hidden representation."""
        # Ensure belief distribution is properly normalized
        eps = 1e-8
        belief_clamped = torch.clamp(belief_distribution, eps, 1 - eps)
        
        # Convert belief distribution back to logits
        # Since forward does: logits = self.belief_projection(z), then softmax(logits)
        # We need to invert this process
        
        # Step 1: Convert from probability to logits (inverse of softmax)
        # Use the log-sum-exp trick for numerical stability
        belief_logits = torch.log(belief_clamped)
        
        # Step 2: Invert the linear projection
        # Since logits = W @ z + b, we need to solve for z
        # For an over-determined system (hidden_dim > num_belief_states), we use least squares
        
        # Remove bias first: logits_no_bias = logits - bias
        logits_no_bias = belief_logits - self.belief_projection.bias.unsqueeze(0)
        
        # Solve: W @ z = logits_no_bias for z
        # z = W^T @ (W @ W^T)^(-1) @ logits_no_bias (Moore-Penrose pseudoinverse)
        W = self.belief_projection.weight  # [num_belief_states, hidden_dim]
        
        # Use torch.linalg.lstsq for numerical stability
        # We want to solve W @ z.T = logits_no_bias.T for z.T
        # So z.T = lstsq(W.T, logits_no_bias.T)
        try:
            z_T = torch.linalg.lstsq(W.T, logits_no_bias.T).solution
            z = z_T.T
        except Exception as e:
            # Raise error instead of fallback to pseudoinverse
            raise InverseTransformationError(
                f"Linear system solving failed during inverse belief transformation: {e}. "
                f"This may indicate numerical instability or singular matrix. "
                f"Try reducing learning rates or checking belief distribution values."
            )
        
        # Step 3: Apply inverse coupling layers
        for layer in reversed(self.coupling_layers):
            z, _ = layer(z, reverse=True)
        
        return z
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of the transformation."""
        batch_size = x.size(0)
        
        # Apply coupling layers and accumulate log determinants
        z = x
        total_log_det = torch.zeros(batch_size, device=x.device)
        
        for layer in self.coupling_layers:
            z, log_det = layer(z, reverse=False)
            total_log_det += log_det
        
        # Base distribution log probability
        base_dist = torch.distributions.Normal(self.base_loc, self.base_scale)
        base_log_prob = base_dist.log_prob(z).sum(dim=-1)
        
        # Total log probability
        return base_log_prob + total_log_det


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
        dropout: float = 0.1,
        flow_layers: int = 4,
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
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

        # Invertible belief head
        self.belief_head = InvertibleBeliefHead(
            hidden_dim=hidden_dim,
            num_belief_states=num_belief_states,
            num_layers=flow_layers
        )

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_normal_(self.signal_projection.weight)
        nn.init.xavier_normal_(self.continuous_signal_projection.weight)
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

        # Get belief distribution using invertible head
        hidden_features = new_belief.squeeze(0)
        
        if is_continuous:
            # For continuous signals, take only the first component
            belief_distribution = self.belief_head(hidden_features)
            belief_distribution = belief_distribution[:, :1]
        else:
            belief_distribution = self.belief_head(hidden_features)

        return self.standardize_belief_state(new_belief), belief_distribution

    def get_belief_log_prob(self, hidden_features: torch.Tensor) -> torch.Tensor:
        """Get log probability of belief transformation."""
        return self.belief_head.log_prob(hidden_features)
    
    def inverse_belief_transform(self, belief_distribution: torch.Tensor) -> torch.Tensor:
        """Inverse transform from belief distribution to hidden features."""
        return self.belief_head.inverse(belief_distribution)
