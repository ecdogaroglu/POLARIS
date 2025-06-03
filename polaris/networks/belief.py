"""
Transformer-based belief processing with normalizing flows for POLARIS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List


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
        
        # Separate projections for continuous and discrete cases
        self.discrete_belief_projection = nn.Linear(hidden_dim, num_belief_states)
        self.continuous_belief_projection = nn.Linear(hidden_dim, 1)  # 1D for continuous signals
        
        # Base distribution (standard normal)
        self.register_buffer('base_loc', torch.zeros(hidden_dim))
        self.register_buffer('base_scale', torch.ones(hidden_dim))
        
    def forward(self, x: torch.Tensor, is_continuous: bool = False) -> torch.Tensor:
        """Transform hidden representation to belief distribution."""
        batch_size = x.size(0)
        
        # Apply coupling layers
        z = x
        total_log_det = torch.zeros(batch_size, device=x.device)
        
        for layer in self.coupling_layers:
            z, log_det = layer(z, reverse=False)
            total_log_det += log_det
        
        # Project to belief space using appropriate projection layer
        if is_continuous:
            belief_logits = self.continuous_belief_projection(z)
        else:
            belief_logits = self.discrete_belief_projection(z)
        
        if is_continuous:
            # For continuous signals, use sigmoid to output signal values in [0,1] range
            belief_distribution = torch.sigmoid(belief_logits)
        else:
            # For discrete signals, use softmax to output probability distributions
            belief_distribution = F.softmax(belief_logits, dim=-1)
        
        return belief_distribution
    
    def inverse(self, belief_distribution: torch.Tensor) -> torch.Tensor:
        """Inverse transformation from belief distribution to hidden representation."""
        # Ensure belief distribution is properly normalized
        eps = 1e-8
        belief_clamped = torch.clamp(belief_distribution, eps, 1 - eps)
        
        # Determine if this is continuous or discrete based on output dimensions
        is_continuous = belief_distribution.size(-1) == 1
        
        # Convert belief distribution back to logits
        if is_continuous:
            # For continuous signals: inverse of sigmoid
            belief_logits = torch.logit(belief_clamped)
            projection_layer = self.continuous_belief_projection
        else:
            # For discrete signals: inverse of softmax
            belief_logits = torch.log(belief_clamped)
            projection_layer = self.discrete_belief_projection
        
        # Step 2: Invert the linear projection
        # Since logits = W @ z + b, we need to solve for z
        # For an over-determined system (hidden_dim > output_dim), we use least squares
        
        # Remove bias first: logits_no_bias = logits - bias
        logits_no_bias = belief_logits - projection_layer.bias.unsqueeze(0)
        
        # Solve: W @ z = logits_no_bias for z
        # z = W^T @ (W @ W^T)^(-1) @ logits_no_bias (Moore-Penrose pseudoinverse)
        W = projection_layer.weight  # [output_dim, hidden_dim]
        
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

        # Get belief distribution using different approaches for continuous vs discrete signals
        hidden_features = new_belief.squeeze(0)
        
        if is_continuous:
            # For continuous signals (strategic experimentation), use the belief head with continuous mode
            belief_distribution = self.belief_head(hidden_features, is_continuous=True)
            belief_distribution = belief_distribution[:, :1]  # Take only first component
        else:
            # For discrete signals, use the full invertible head with softmax
            belief_distribution = self.belief_head(hidden_features, is_continuous=False)

        return self.standardize_belief_state(new_belief), belief_distribution

    def get_belief_log_prob(self, hidden_features: torch.Tensor) -> torch.Tensor:
        """Get log probability of belief transformation."""
        return self.belief_head.log_prob(hidden_features)
    
    def inverse_belief_transform(self, belief_distribution: torch.Tensor) -> torch.Tensor:
        """Inverse transform from belief distribution to hidden features."""
        return self.belief_head.inverse(belief_distribution)
    
    def calculate_expected_signal(
        self, 
        belief_distribution: torch.Tensor,
        drift_rates: List[float],
        jump_rates: List[float], 
        jump_sizes: List[float],
        background_informativeness: float,
        time_step: float
    ) -> torch.Tensor:
        """
        Calculate expected signal increment under the current belief distribution.
        
        For strategic experimentation, the signal follows:
        signal_increment = background_informativeness * drift_rate * time_step + diffusion + jump
        
        Since we're dealing with expectations and diffusion has zero mean, we focus on:
        E[signal_increment] = background_informativeness * drift_rate * time_step + E[jump]
        where E[jump] = jump_rate * time_step * jump_size

        Only for continuous case.
        
        Args:
            belief_distribution: Current belief probabilities [batch_size, num_states] or [batch_size, 1] for continuous
            drift_rates: Drift rates for each state [bad_state, good_state]
            jump_rates: Jump rates for each state [bad_state, good_state] 
            jump_sizes: Jump sizes for each state [bad_state, good_state]
            background_informativeness: Informativeness parameter
            time_step: Time step size
            
        Returns:
            expected_signal: Expected signal increment [batch_size, 1]
        """
        device = belief_distribution.device
        batch_size = belief_distribution.size(0)
        
        # Check if this is continuous (1D) or discrete (2D) belief distribution
        is_continuous = belief_distribution.size(-1) == 1
        
        # For continuous case, belief_distribution represents the probability of good state
        belief_good = belief_distribution.squeeze(-1)  # [batch_size]
        belief_bad = 1.0 - belief_good

        
        # Convert to tensors
        drift_rates_tensor = torch.tensor(drift_rates, device=device, dtype=torch.float32)
        jump_rates_tensor = torch.tensor(jump_rates, device=device, dtype=torch.float32)
        jump_sizes_tensor = torch.tensor(jump_sizes, device=device, dtype=torch.float32)
        
        # Calculate expected signals for each state
        # State 0 (bad): drift_rates[0], jump_rates[0], jump_sizes[0]
        # State 1 (good): drift_rates[1], jump_rates[1], jump_sizes[1]
        
        expected_signal_bad = (
            background_informativeness * drift_rates_tensor[0] * time_step +
            jump_rates_tensor[0] * time_step * jump_sizes_tensor[0]
        )
        
        expected_signal_good = (
            background_informativeness * drift_rates_tensor[1] * time_step +
            jump_rates_tensor[1] * time_step * jump_sizes_tensor[1]
        )
        
        # Weighted average based on belief distribution
        expected_signal = (
            belief_bad * expected_signal_bad + 
            belief_good * expected_signal_good
        )
        
        return expected_signal.unsqueeze(-1)  # [batch_size, 1]
    
    def belief_signal_loss(
        self,
        belief_distribution: torch.Tensor,
        actual_signal: torch.Tensor,
        drift_rates: List[float],
        jump_rates: List[float],
        jump_sizes: List[float], 
        background_informativeness: float,
        time_step: float
    ) -> torch.Tensor:
        """
        Calculate adaptive loss between expected signal under belief and actual received signal.
        
        This implements a principled belief update mechanism with enhanced responsiveness to state changes.
        Uses higher penalties when beliefs strongly contradict observed signals.
        
        Args:
            belief_distribution: Current belief probabilities
            actual_signal: Actually received signal increment
            drift_rates: Drift rates for each state
            jump_rates: Jump rates for each state
            jump_sizes: Jump sizes for each state
            background_informativeness: Informativeness parameter
            time_step: Time step size
            
        Returns:
            loss: Adaptive loss between expected and actual signal
        """
        # Calculate expected signal under current belief
        expected_signal = self.calculate_expected_signal(
            belief_distribution,
            drift_rates,
            jump_rates,
            jump_sizes,
            background_informativeness,
            time_step
        )
        
        # Ensure actual_signal has the right shape
        if actual_signal.dim() == 1:
            actual_signal = actual_signal.unsqueeze(-1)
        
        # Calculate base MSE loss
        signal_error = expected_signal - actual_signal
        base_loss = F.mse_loss(expected_signal, actual_signal)
        
        # Add adaptive penalty for large contradictions
        # When belief is high (>0.5) but signal is very low (<0.01), or vice versa
        belief_good = belief_distribution.squeeze(-1)  # [batch_size]
        actual_signal_flat = actual_signal.squeeze(-1)  # [batch_size]
        
        # Detect contradictions:
        # 1. High belief (>0.5) but very low signal (<0.01)
        # 2. Low belief (<0.1) but high signal (>0.1)
        contradiction_penalty = torch.zeros_like(base_loss)
        
        high_belief_low_signal = (belief_good > 0.5) & (actual_signal_flat < 0.01)
        low_belief_high_signal = (belief_good < 0.1) & (actual_signal_flat > 0.1)
        
        if high_belief_low_signal.any():
            # Strong penalty when believing in good state but receiving zero signals
            penalty = 10.0 * belief_good[high_belief_low_signal].pow(2)
            contradiction_penalty = contradiction_penalty + penalty.mean()
        
        if low_belief_high_signal.any():
            # Strong penalty when believing in bad state but receiving positive signals
            penalty = 10.0 * (1.0 - belief_good[low_belief_high_signal]).pow(2)
            contradiction_penalty = contradiction_penalty + penalty.mean()
        
        # Total loss combines base MSE with contradiction penalty
        total_loss = base_loss + contradiction_penalty
        
        return total_loss
