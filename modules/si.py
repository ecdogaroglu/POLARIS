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
import numpy as np
import matplotlib.pyplot as plt

from modules.utils import get_best_device


class SILoss:
    """
    Synaptic Intelligence (SI) regularization loss for continual learning.
    
    Based on the paper "Continual Learning Through Synaptic Intelligence" (Zenke et al., 2017).
    """
    
    def __init__(self, model, importance=1.0, damping=0.1, exclude_layers=None, device=None):
        """
        Initialize SI loss module.
        
        Args:
            model: Model parameters to protect
            importance: Importance hyperparameter (scaling factor for the SI loss)
            damping: Small constant to prevent division by zero
            exclude_layers: List of parameter names to exclude from regularization
        """
        self.model = model
        self.device = device if device is not None else get_best_device()
        self.importance = importance
        self.damping = damping
        
        # Initialize parameter regularization data structures
        self.omega = {n: torch.zeros_like(p).to(self.device) for n, p in model.named_parameters() if not self._is_excluded(n, exclude_layers)}
        self.prev_params = {n: torch.clone(p).detach() for n, p in model.named_parameters() if not self._is_excluded(n, exclude_layers)}
        
        # Path integral accumulation
        self.path_integral = {n: torch.zeros_like(p).to(self.device) for n, p in model.named_parameters() if not self._is_excluded(n, exclude_layers)}
        
        # Set model reference parameters for current task
        self.reference_params = {n: torch.clone(p).detach() for n, p in model.named_parameters() if not self._is_excluded(n, exclude_layers)}
        
        # Tracker for step count
        self.step_count = 0
        self.accumulated_loss = 0.0
        self.exclude_layers = exclude_layers
        
        # Debug info
        print(f"Initialized SILoss with importance={importance}, damping={damping}")
        if exclude_layers:
            print(f"Excluded layers: {exclude_layers}")
        
    def _is_excluded(self, name, exclude_layers):
        """Check if parameter should be excluded from regularization."""
        if exclude_layers is None:
            return False
        return any(excluded in name for excluded in exclude_layers)
        
    def update_tracker(self, loss_value):
        """
        Update the SI tracker with the current loss and accumulate path integrals.
        
        Args:
            loss_value: Current loss value
        """
        # Update accumulated loss
        self.accumulated_loss += loss_value
        
        # Update step count
        self.step_count += 1
        
        # Update path integrals
        for n, p in self.model.named_parameters():
            if n in self.path_integral:  # Only track non-excluded parameters
                if p.grad is not None:
                    # Calculate parameter movement
                    delta_theta = p.detach() - self.prev_params[n]
                    
                    # Accumulate the product of negative gradient and parameter update
                    # This approximates the path integral of the loss gradient
                    self.path_integral[n] -= p.grad.detach() * delta_theta
                    
                    # Update previous parameters
                    self.prev_params[n] = p.detach().clone()
    
    def calculate_importance(self):
        """
        Calculate parameter importance based on accumulated path integrals.
        
        This is typically called when a task is complete.
        """
        # For each parameter
        for n, omega in self.omega.items():
            # Get the parameter
            p = self.model.get_parameter(n)
            
            # Calculate parameter change from reference
            delta_theta = p.detach() - self.reference_params[n]
            
            # Update importance (omega) based on path integral
            # If parameter didn't change much (small delta), importance is high
            # If parameter changed a lot, importance is lower
            change_norm = delta_theta.pow(2) + self.damping
            omega.add_(self.path_integral[n] / change_norm)
            
        # Reset path integrals and set new reference parameters
        self.path_integral = {n: torch.zeros_like(p).to(self.device) for n, p in self.model.named_parameters() 
                              if n in self.path_integral}
        self.reference_params = {n: torch.clone(p).detach() for n, p in self.model.named_parameters() 
                                if n in self.reference_params}
        
        # Reset step count and accumulated loss
        step_count = self.step_count
        avg_loss = self.accumulated_loss / max(1, self.step_count)
        self.step_count = 0
        self.accumulated_loss = 0.0
        
        return step_count, avg_loss
        
    def calculate_loss(self):
        """Calculate the SI regularization loss."""
        loss = torch.tensor(0.0).to(self.device)
        
        # For each parameter
        for n, p in self.model.named_parameters():
            if n in self.omega:  # Only consider non-excluded parameters
                # Calculate change from reference parameters
                loss_param = self.omega[n] * (p - self.reference_params[n]).pow(2)
                loss += loss_param.sum()
                
        # Scale the loss by importance factor
        return self.importance * loss
        
    def get_parameter_importance(self, normalized=True):
        """
        Get the importance scores for all parameters.
        
        Args:
            normalized: Whether to normalize the importance scores
            
        Returns:
            Dict mapping parameter names to importance scores
        """
        importance_dict = {}
        
        if normalized:
            # Find max importance for normalization
            max_importance = 0.0
            for n, omega in self.omega.items():
                max_importance = max(max_importance, omega.abs().max().item())
                
            # Avoid division by zero
            if max_importance == 0.0:
                max_importance = 1.0
                
            # Normalize importance
            for n, omega in self.omega.items():
                importance_dict[n] = omega.abs() / max_importance
        else:
            # Return raw importance
            for n, omega in self.omega.items():
                importance_dict[n] = omega.abs()
                
        return importance_dict
    
    def register_task_change(self):
        """Register that a task has changed and calculate importance for previous task."""
        return self.calculate_importance()
        
    def save_state(self, path):
        """Save SI state to file."""
        state = {
            'omega': self.omega,
            'prev_params': self.prev_params,
            'path_integral': self.path_integral,
            'reference_params': self.reference_params,
            'importance': self.importance,
            'damping': self.damping,
            'step_count': self.step_count,
            'accumulated_loss': self.accumulated_loss
        }
        torch.save(state, path)
        
    def load_state(self, path):
        """Load SI state from file."""
        state = torch.load(path, map_location=self.device)
        self.omega = state['omega']
        self.prev_params = state['prev_params']
        self.path_integral = state['path_integral']
        self.reference_params = state['reference_params']
        self.importance = state['importance']
        self.damping = state['damping']
        self.step_count = state['step_count']
        self.accumulated_loss = state['accumulated_loss']


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
        si_tracker.update_tracker(loss.item())
    
    return si_tracker.path_integral


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
            si_tracker.update_tracker(loss.item())
            
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
    
    return si_tracker, si_tracker.path_integral 