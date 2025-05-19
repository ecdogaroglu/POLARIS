"""
Synaptic Intelligence (SI) implementation for POLARIS.

This module provides functionality to calculate parameter importance based on the path
the parameters took during training and SI loss for mitigating catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, List, Tuple, Optional, Union, Set


class SILoss:
    """
    Synaptic Intelligence (SI) loss for preventing catastrophic forgetting.
    
    SI calculates parameter importance based on the path parameters take during training,
    rather than using Fisher Information Matrix (as in EWC).
    """
    
    def __init__(
        self,
        model: nn.Module,
        importance: float = 100.0,
        damping: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the SI loss.
        
        Args:
            model: The model to protect from catastrophic forgetting
            importance: The importance factor for the SI penalty (lambda)
            damping: Small constant to prevent division by zero when calculating importance
            device: Device to use for computations
        """
        self.model = model
        self.importance = importance
        self.damping = damping
        self.device = device if device is not None else torch.device('cpu')
        
        # Initialize dictionaries to store parameters, path integrals, and importances
        self.previous_params = {}
        self.importance_scores = {}
        self.param_path_integrals = {}  # Accumulated gradient * parameter change
        self.param_trajectories = {}    # Accumulated parameter change from start of training
        
        # Store parameter names for easier access
        self.param_names = [name for name, _ in model.named_parameters() if _.requires_grad]
        
        # Initialize param paths and importances
        self._init_param_trajectories()
    
    def _init_param_trajectories(self):
        """Initialize tracking of parameter trajectories."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Clone the parameter to avoid reference issues
                self.previous_params[name] = param.data.clone()
                # Initialize path integral to zero
                self.param_path_integrals[name] = torch.zeros_like(param.data).to(self.device)
                # Initialize importance to zero (or small value)
                self.importance_scores[name] = torch.zeros_like(param.data).to(self.device)
    
    def update_trajectory(self, optimizer_step: bool = True):
        """
        Update the parameter trajectories and accumulate path integrals.
        
        Args:
            optimizer_step: Whether an optimizer step has been performed since last call
        """
        if not optimizer_step:
            return
        
        # Update path integral and reset reference parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.previous_params:
                if param.grad is not None:
                    # Accumulate path integral: gradient * parameter change
                    delta = param.data - self.previous_params[name]
                    self.param_path_integrals[name] += -param.grad * delta
                    
                # Update previous parameter values
                self.previous_params[name] = param.data.clone()
    
    def register_task(self):
        """
        Register the end of a task, calculating importance scores from path integrals.
        """
        # Calculate importance scores from accumulated path integrals
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.param_path_integrals:
                # Get the path integral
                path_integral = self.param_path_integrals[name]
                
                # Get parameter change from start point
                delta = param.data - self.previous_params[name]
                
                # Calculate importance: path_integral / (delta^2 + damping)
                # Add small constant to avoid division by zero
                delta_squared = delta.pow(2) + self.damping
                new_importance = path_integral / delta_squared
                
                # Accumulate importance scores
                self.importance_scores[name] += new_importance
                
                # Reset path integral for next task
                self.param_path_integrals[name] = torch.zeros_like(param.data).to(self.device)
                
                # Update previous parameter values
                self.previous_params[name] = param.data.clone()
    
    def calculate_loss(self) -> torch.Tensor:
        """
        Calculate the SI loss based on parameter importances.
        
        Returns:
            The SI loss tensor
        """
        loss = torch.tensor(0.0, device=self.device)
        
        # Calculate SI loss
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.importance_scores and name in self.previous_params:
                # Get the importance scores
                importance = self.importance_scores[name]
                
                # Get the old parameter values
                old_param = self.previous_params[name]
                
                # Calculate squared difference weighted by importance scores
                param_diff = (param - old_param).pow(2)
                weighted_diff = (importance * param_diff).sum()
                
                # Add to total loss
                loss += weighted_diff
        
        # Apply importance factor
        loss *= self.importance / 2.0
        
        return loss


def calculate_path_integral(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_batches: int = 0,  # 0 means all batches
) -> Dict[str, torch.Tensor]:
    """
    Calculate path integrals for parameters using a dataset.
    
    Args:
        model: The model to calculate path integrals for
        data_loader: DataLoader containing the dataset
        loss_fn: Loss function to use for calculating gradients
        optimizer: Optimizer to use for parameter updates
        device: Device to use for computations
        num_batches: Number of batches to process (0 means all)
        
    Returns:
        Dictionary mapping parameter names to their path integrals
    """
    # Set model to training mode
    model.train()
    
    # Initialize SI tracker
    si_tracker = SILoss(model, device=device)
    
    # Process each batch
    for i, batch in enumerate(data_loader):
        if num_batches > 0 and i >= num_batches:
            break
            
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss = loss_fn(model, batch)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Update trajectories
        si_tracker.update_trajectory()
    
    return si_tracker.param_path_integrals


def calculate_path_integral_from_replay_buffer(
    model: nn.Module,
    replay_buffer,
    loss_fn: callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 1000,
    num_batches: int = 20
) -> Tuple[SILoss, Dict[str, torch.Tensor]]:
    """
    Calculate path integrals using samples from a replay buffer.
    
    Args:
        model: The model to calculate path integrals for
        replay_buffer: Replay buffer containing transitions
        loss_fn: Loss function to use for calculating gradients
        optimizer: Optimizer to use for parameter updates
        device: Device to use for computations
        batch_size: Batch size for sampling from replay buffer
        num_batches: Number of batches to process
        
    Returns:
        A tuple containing:
        - SILoss tracker instance with accumulated path integrals
        - Dictionary mapping parameter names to their path integrals
    """
    # Set model to training mode
    model.train()
    
    # Initialize SI tracker
    si_tracker = SILoss(model, device=device)
    
    # Process multiple batches
    valid_batches = 0
    max_attempts = num_batches * 2  # Allow more attempts to get valid batches
    attempts = 0
    
    while valid_batches < num_batches and attempts < max_attempts:
        attempts += 1
        
        # Sample batch from replay buffer
        batch = replay_buffer.sample(batch_size)
        if batch is None:
            continue
        
        try:
            # Zero gradients
            optimizer.zero_grad()
            
            # Make sure all tensors require gradients
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.requires_grad_(True)
            
            # Calculate loss
            loss = loss_fn(model, batch)
                
            # Backward pass
            loss.backward()
            
            # Check if gradients are valid (not NaN or Inf)
            valid_grads = True
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        valid_grads = False
                        break
            
            if not valid_grads:
                print("Skipping batch with NaN or Inf gradients")
                continue
            
            # Update parameters
            optimizer.step()
            
            # Update trajectories
            si_tracker.update_trajectory()
            
            valid_batches += 1
            if valid_batches % 10 == 0:
                print(f"Processed {valid_batches}/{num_batches} valid batches for path integral calculation")
                
        except Exception as e:
            print(f"Error calculating path integral for batch: {e}")
            continue
    
    if valid_batches > 0:
        print(f"Calculated path integrals using {valid_batches} valid batches")
    else:
        print("Warning: No valid batches for path integral calculation!")
    
    return si_tracker, si_tracker.param_path_integrals 