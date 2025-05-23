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
    tracking it online during the normal optimization process, rather than requiring separate
    backward passes through a replay buffer.
    
    Based on the paper: "Continual Learning Through Synaptic Intelligence" (Zenke et al., 2017)
    """
    
    def __init__(
        self,
        model: nn.Module,
        importance: float = 100.0,
        damping: float = 0.1,
        device: Optional[torch.device] = None,
        excluded_layers: List[str] = None,
        specialization_strength: float = 0.8
    ):
        """
        Initialize the SI loss.
        
        Args:
            model: The model to protect from catastrophic forgetting
            importance: The importance factor for the SI penalty (lambda)
            damping: Small constant to prevent division by zero when calculating importance
            device: Device to use for computations
            excluded_layers: List of layer names to exclude from SI protection
            specialization_strength: Strength of parameter specialization (0-1, higher = more specialization)
        """
        self.model = model
        self.importance = importance
        self.damping = damping
        self.device = device if device is not None else torch.device('cpu')
        self.excluded_layers = excluded_layers or []
        self.specialization_strength = specialization_strength
        
        # Initialize dictionaries to store parameters, path integrals, and importances
        self.previous_params = {}
        self.importance_scores = {}
        self.param_path_integrals = {}  # Accumulated gradient * parameter change
        
        # For new online trajectory tracking
        self.prev_params_per_step = {}  # Parameter values at the previous step
        self.current_task_id = None
        
        # Track task-specific importance scores
        self.task_importance_scores = {}
        
        # Keep the original gradients for visualization
        self.original_gradients = {}
        self.masked_gradients = {}
        
        # Initialize param paths and importances
        self._init_param_trajectories()
    
    def _init_param_trajectories(self):
        """Initialize tracking of parameter trajectories."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Skip excluded layers
                if name in self.excluded_layers:
                    continue
                    
                # Clone the parameter to avoid reference issues
                self.previous_params[name] = param.data.clone()
                self.prev_params_per_step[name] = param.data.clone()
                
                # Initialize path integral to zero
                self.param_path_integrals[name] = torch.zeros_like(param.data).to(self.device)
                
                # Initialize importance to zero
                self.importance_scores[name] = torch.zeros_like(param.data).to(self.device)
    
    def set_task(self, task_id):
        """
        Set the current task for importance tracking. Call this whenever the task changes.
        
        Args:
            task_id: Identifier for the current task (e.g., environment state)
        """
        # If switching to a new task, compute importances for previous task before resetting
        if self.current_task_id is not None and self.current_task_id != task_id:
            self.register_task()
            
        # Update the current task
        self.current_task_id = task_id
        
        # Reset path integrals for the new task
        for name in self.param_path_integrals:
            self.param_path_integrals[name].zero_()
            
        # Update previous parameters to current values for the new task
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in self.excluded_layers:
                self.previous_params[name] = param.data.clone()
                self.prev_params_per_step[name] = param.data.clone()
    
    def update_trajectory(self, optimizer_step: bool = True):
        """
        Update the parameter trajectories and accumulate path integrals.
        This should be called after each backward pass but before optimizer.step()
        
        This method also applies gradient masking to promote task-specific parameter specialization.
        
        Args:
            optimizer_step: Whether an optimizer step will be performed (should be True in most cases)
        """
        if not optimizer_step:
            return
        
        # Skip gradient masking if we don't have task importance scores yet
        has_previous_tasks = len(self.task_importance_scores) > 0 and self.current_task_id is not None
        
        # Update path integral for each parameter and apply gradient masking
        for name, param in self.model.named_parameters():
            # Skip excluded layers
            if name in self.excluded_layers:
                continue
                
            if param.requires_grad and param.grad is not None and name in self.prev_params_per_step:
                # Store the original gradient for this step
                self.original_gradients[name] = param.grad.detach().clone()
                
                # Apply gradient masking if we have previous tasks
                if has_previous_tasks and self.specialization_strength > 0:
                    # Calculate a mask based on importance scores from other tasks
                    mask = torch.ones_like(param.grad)
                    
                    # Accumulate importance from all tasks except the current one
                    other_tasks_importance = torch.zeros_like(param.grad)
                    
                    for task_id, importances in self.task_importance_scores.items():
                        if task_id != self.current_task_id and name in importances:
                            other_tasks_importance += torch.abs(importances[name])
                    
                    # Normalize the importance
                    if torch.max(other_tasks_importance) > 0:
                        normalized_importance = other_tasks_importance / torch.max(other_tasks_importance)
                        
                        # Create a mask that reduces gradient flow to important parameters
                        # The stronger the specialization, the more we mask
                        # Apply exponential weighting to create sharper contrast
                        mask = torch.exp(-normalized_importance * self.specialization_strength * 3.0)
                        
                        # Apply the mask to the gradient
                        param.grad = param.grad * mask
                        
                        # Store masked gradient for visualization
                        self.masked_gradients[name] = param.grad.detach().clone()
    
    def update_trajectory_post_step(self):
        """
        Update parameter trajectories after optimizer.step() has been called.
        This should be called after optimizer.step() to complete the importance accumulation.
        """
        # Accumulate path integrals using saved gradients and parameter changes
        for name, param in self.model.named_parameters():
            # Skip excluded layers
            if name in self.excluded_layers:
                continue
                
            if param.requires_grad and name in self.prev_params_per_step:
                # Calculate parameter update (new - old)
                delta = param.data - self.prev_params_per_step[name]
                
                # Check if we have a gradient from the backward pass
                if param.grad is not None:
                    # Use the original (unmasked) gradient for path integral calculation
                    # This ensures the importance calculation is accurate
                    original_grad = self.original_gradients.get(name, param.grad.detach())
                    
                    # Accumulate path integral: -gradient * parameter change
                    # Note the negative sign, as the gradient points in the direction of increasing loss
                    self.param_path_integrals[name] -= original_grad * delta
                
                # Update previous step parameters to current values
                self.prev_params_per_step[name] = param.data.clone()
    
    def register_task(self):
        """
        Register the end of a task, calculating importance scores from path integrals.
        This should be called when transitioning to a new task.
        """
        if self.current_task_id is None:
            return
            
        # Store task-specific importance scores in a dictionary
        task_specific_importances = {}
            
        # Calculate importance scores from accumulated path integrals
        for name, param in self.model.named_parameters():
            # Skip excluded layers
            if name in self.excluded_layers:
                continue
                
            if param.requires_grad and name in self.param_path_integrals:
                # Get the path integral
                path_integral = self.param_path_integrals[name]
                
                # Get parameter change from start of task
                delta = param.data - self.previous_params[name]
                
                # Calculate importance: path_integral / (delta^2 + damping)
                # Add damping factor to avoid division by zero
                delta_squared = delta.pow(2) + self.damping
                new_importance = path_integral / delta_squared
                
                # Store task-specific importance
                task_specific_importances[name] = new_importance.detach().clone()
                
                # Normalize importance scores to encourage diversity
                # We apply softmax normalization across parameters to get a more sparse distribution
                # This helps different tasks to use different parameters
                
                # Accumulate importance scores
                self.importance_scores[name] += new_importance
                
                # Reset path integral for next task
                self.param_path_integrals[name] = torch.zeros_like(param.data).to(self.device)
                
                # Update previous parameter values
                self.previous_params[name] = param.data.clone()
        
        # Store task-specific importance scores
        self.task_importance_scores[self.current_task_id] = task_specific_importances
        
        # Apply importance balancing to encourage task specialization
        self._balance_importance_across_tasks()
        
        # Enhance task specialization
        self._enhance_task_specialization()
    
    def _balance_importance_across_tasks(self):
        """
        Balance importance scores across tasks to encourage different tasks
        to use different parameters. This promotes parameter specialization.
        """
        if len(self.task_importance_scores) <= 1:
            return  # Skip if we have only one task
            
        # Get all parameter names
        param_names = list(self.importance_scores.keys())
        
        # For each parameter, compute the diversity factor across tasks
        for name in param_names:
            # Skip if not important for any task
            if name not in self.importance_scores:
                continue
                
            # Collect importance scores for this parameter across all tasks
            task_scores = []
            for task_id, importances in self.task_importance_scores.items():
                if name in importances:
                    # Convert to scalar for easier computation
                    # We use mean importance across the parameter
                    task_scores.append((task_id, torch.mean(torch.abs(importances[name])).item()))
            
            # If we have importance scores from multiple tasks
            if len(task_scores) > 1:
                # Sort by importance
                task_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Get the most important task for this parameter
                most_important_task, highest_score = task_scores[0]
                
                # Enhance the contrast between tasks by scaling down importance
                # for tasks other than the most important one
                for task_id, _ in task_scores[1:]:  # Loop through less important tasks
                    if task_id in self.task_importance_scores and name in self.task_importance_scores[task_id]:
                        # Reduce importance for this parameter in less important tasks
                        # This encourages the model to use different parameters for different tasks
                        scale_factor = 0.2  # Reduce importance significantly more to create stronger specialization
                        self.task_importance_scores[task_id][name] *= scale_factor
                        
        # Recompute consolidated importance scores
        self._recompute_consolidated_importance()
    
    def _recompute_consolidated_importance(self):
        """Recompute consolidated importance scores from task-specific scores."""
        # Reset consolidated importance scores
        for name in self.importance_scores:
            self.importance_scores[name].zero_()
            
        # Sum up task-specific importance scores
        for task_id, importances in self.task_importance_scores.items():
            for name, importance in importances.items():
                if name in self.importance_scores:
                    self.importance_scores[name] += importance
    
    def calculate_loss(self) -> torch.Tensor:
        """
        Calculate the SI loss based on parameter importances.
        
        Returns:
            The SI loss tensor
        """
        loss = torch.tensor(0.0, device=self.device)
        
        # Calculate SI loss
        for name, param in self.model.named_parameters():
            # Skip excluded layers
            if name in self.excluded_layers:
                continue
                
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

    def _enhance_task_specialization(self):
        """
        Enhance task-specific parameter specialization by boosting importance
        for parameters that are already starting to specialize for the current task.
        """
        if self.current_task_id is None or len(self.task_importance_scores) <= 1:
            return  # Skip if we only have the current task
        
        # For each parameter
        for name in self.importance_scores.keys():
            if name not in self.task_importance_scores[self.current_task_id]:
                continue
                
            # Get importance for current task
            current_importance = self.task_importance_scores[self.current_task_id][name]
            
            # Calculate average importance for this parameter across other tasks
            other_importance_sum = torch.zeros_like(current_importance)
            other_task_count = 0
            
            for task_id, importances in self.task_importance_scores.items():
                if task_id != self.current_task_id and name in importances:
                    other_importance_sum += torch.abs(importances[name])
                    other_task_count += 1
            
            if other_task_count > 0:
                other_avg_importance = other_importance_sum / other_task_count
                
                # Calculate specialization potential: how much more important this parameter
                # is for the current task compared to other tasks
                specialization_potential = torch.abs(current_importance) - other_avg_importance
                
                # Apply boost where the parameter is more important for current task
                boost_mask = (specialization_potential > 0).float()
                boost_factor = 1.5  # Boost by 50%
                
                # Apply the boost
                boosted_importance = current_importance * (1.0 + boost_mask * (boost_factor - 1.0))
                
                # Update the task-specific importance
                self.task_importance_scores[self.current_task_id][name] = boosted_importance
        
        # Recompute consolidated importance scores
        self._recompute_consolidated_importance()


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
        
        # Record gradients before optimizer step
        si_tracker.update_trajectory()
        
        # Update parameters
        optimizer.step()
        
        # Update path integrals after optimizer step
        si_tracker.update_trajectory_post_step()
    
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
            
            # Record gradients before optimizer step
            si_tracker.update_trajectory()
            
            # Update parameters
            optimizer.step()
            
            # Update path integrals after optimizer step
            si_tracker.update_trajectory_post_step()
            
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