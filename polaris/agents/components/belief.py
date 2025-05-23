import torch
import torch.nn as nn
from typing import Tuple, Optional
from polaris.networks.transformer import TransformerBeliefProcessor

class BeliefComponent:
    """Handles belief state processing for POLARIS agents."""
    
    def __init__(
        self,
        belief_dim: int,
        action_dim: int,
        num_belief_states: int,
        device: torch.device,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        self.belief_dim = belief_dim
        self.device = device
        
        self.processor = TransformerBeliefProcessor(
            hidden_dim=belief_dim,
            action_dim=action_dim,
            device=device,
            num_belief_states=num_belief_states,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        
        # Initialize belief state
        self.current_belief = torch.zeros(1, 1, belief_dim, device=device)
        self.current_belief_distribution = torch.ones(
            1, num_belief_states, device=device
        ) / num_belief_states
        
    def process_observation(
        self,
        signal: torch.Tensor,
        neighbor_actions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update belief based on observation."""
        new_belief, belief_dist = self.processor(
            signal, 
            neighbor_actions,
            self.current_belief
        )
        
        self.current_belief = new_belief
        self.current_belief_distribution = belief_dist
        
        return new_belief, belief_dist
        
    def reset(self):
        """Reset belief state."""
        self.current_belief.zero_()
        self.current_belief_distribution.fill_(1.0 / self.current_belief_distribution.size(1))
        
    def get_state_dict(self) -> dict:
        """Get state dict for saving."""
        return self.processor.state_dict()
        
    def load_state_dict(self, state_dict: dict):
        """Load state dict."""
        self.processor.load_state_dict(state_dict)