"""
Observation encoding utilities for POLARIS.
"""

from typing import Dict, Union, Tuple

import numpy as np
import torch


def encode_observation(
    signal: Union[int, float, torch.Tensor],
    neighbor_actions: Dict[int, Union[int, float, torch.Tensor]],
    num_agents: int,
    num_states: int,
    continuous_actions: bool = False,
    continuous_signal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode the observation (signal + neighbor actions) into a fixed-size vector.

    Args:
        signal: The private signal (can be an integer or a float)
        neighbor_actions: Dictionary of neighbor IDs to their actions
        num_agents: Total number of agents in the environment
        num_states: Number of possible states
        continuous_actions: If True, encode actions as continuous values
        continuous_signal: If True, use signal directly without one-hot encoding

    Returns:
        signal_one_hot: Encoded signal tensor
        action_encoding: Encoded neighbor actions tensor
    """
    # Handle signal encoding based on type
    if continuous_signal:
        # For continuous signals, create a 1-element tensor with the raw signal value
        if isinstance(signal, (int, float)):
            signal_one_hot = torch.tensor([float(signal)])
        elif isinstance(signal, torch.Tensor):
            signal_one_hot = signal.clone().detach().float().view(1)
        else:
            raise ValueError(f"Invalid signal type: {type(signal)}")
    else:
        # One-hot encode discrete signals
        signal_one_hot = torch.zeros(num_states)
        if isinstance(signal, (int, np.integer)) and 0 <= signal < num_states:
            signal_one_hot[signal] = 1.0
        elif isinstance(signal, torch.Tensor) and signal.dim() == 0:
            if 0 <= signal.item() < num_states:
                signal_one_hot[signal.item()] = 1.0

    # Encode neighbor actions
    if continuous_actions:
        # For continuous actions, create a vector with raw allocation values
        action_encoding = torch.zeros(num_agents)

        if neighbor_actions is not None:
            for neighbor_id, allocation in neighbor_actions.items():
                if (
                    isinstance(neighbor_id, (int, np.integer))
                    and 0 <= neighbor_id < num_agents
                ):
                    # Convert to float if needed
                    if isinstance(allocation, (int, float, np.number)):
                        action_encoding[neighbor_id] = float(allocation)
                    elif isinstance(allocation, torch.Tensor):
                        action_encoding[neighbor_id] = allocation.item()
    else:
        # For discrete actions, use one-hot encoding
        action_encoding = torch.zeros(num_agents * num_states)

        if neighbor_actions is not None:
            for neighbor_id, action in neighbor_actions.items():
                if (
                    isinstance(neighbor_id, (int, np.integer))
                    and 0 <= neighbor_id < num_agents
                ):
                    # Calculate the starting index for this neighbor's action encoding
                    start_idx = neighbor_id * num_states
                    # One-hot encode the action
                    if (
                        isinstance(action, (int, np.integer))
                        and 0 <= action < num_states
                    ):
                        action_encoding[start_idx + action] = 1.0
                    elif isinstance(action, torch.Tensor) and action.dim() == 0:
                        action_idx = action.item()
                        if 0 <= action_idx < num_states:
                            action_encoding[start_idx + action_idx] = 1.0

    return signal_one_hot, action_encoding


def calculate_observation_dimension(env) -> int:
    """Calculate the observation dimension based on environment properties."""
    return env.num_states + env.num_agents * env.num_states
