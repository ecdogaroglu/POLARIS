import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import copy

class SynapticIntelligence:
    """
    Implementation of Synaptic Intelligence for continual learning to prevent catastrophic forgetting.
    
    This implementation tracks parameter importance online during training without additional
    backpropagation passes. It maintains separate importance estimations for different tasks.
    
    Based on the paper:
    "Continual Learning Through Synaptic Intelligence" (Zenke et al., 2017)
    """
    
    def __init__(self, model, si_lambda=1.0, damping_factor=0.1, visualization_dir="results/si_visualizations"):
        """
        Initialize the Synaptic Intelligence tracker.
        
        Args:
            model: The neural network model to track
            si_lambda: Regularization strength for SI penalty
            damping_factor: Small constant to avoid division by zero
            visualization_dir: Directory to save importance visualizations
        """
        self.model = model
        self.si_lambda = si_lambda
        self.damping_factor = damping_factor
        self.visualization_dir = visualization_dir
        
        # Create the visualization directory if it doesn't exist
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Dictionary to store task-specific importance values
        self.importances = defaultdict(lambda: defaultdict(dict))
        
        # Dictionary to store parameter-specific consolidated importances across tasks
        self.consolidated_importances = {}
        
        # Previous parameter values to track changes
        self.prev_params = {}
        
        # Accumulated gradients times parameter changes
        self.omega_accumulation = {}
        
        # Small constant to add to denominator for numerical stability
        self.eps = 1e-8
        
        # Current task being learned
        self.current_task_id = None
        
        # Initialize parameter tracking
        self._init_tracking()
    
    def _init_tracking(self):
        """Initialize parameter tracking for the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Store initial parameter values
                self.prev_params[name] = param.detach().clone()
                
                # Initialize accumulation of parameter importance
                self.omega_accumulation[name] = torch.zeros_like(param.data)
                
                # Initialize consolidated importance across tasks
                self.consolidated_importances[name] = torch.zeros_like(param.data)
    
    def set_task(self, task_id):
        """
        Set the current task for importance tracking.
        
        Args:
            task_id: Identifier for the current task (e.g., environment state)
        """
        # If switching to a new task, reset accumulators
        if self.current_task_id != task_id:
            # Save accumulated importance for previous task before resetting
            if self.current_task_id is not None:
                self._update_task_importance()
            
            # Reset accumulation for new task
            for name in self.omega_accumulation:
                self.omega_accumulation[name].zero_()
            
            # Update previous parameters to current values
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.prev_params[name] = param.detach().clone()
            
            # Set new task
            self.current_task_id = task_id
    
    def _update_task_importance(self):
        """Update importance values for the current task."""
        if self.current_task_id is None:
            return
            
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Calculate parameter change for the current task
                delta = param.detach() - self.prev_params[name]
                
                # Calculate importance: omega / (delta^2 + damping)
                delta_squared = delta.pow(2) + self.damping_factor
                importance = self.omega_accumulation[name] / (delta_squared + self.eps)
                
                # Store importance for this task
                self.importances[self.current_task_id][name] = importance.detach().clone()
                
                # Update consolidated importance (sum across all tasks)
                if name not in self.consolidated_importances:
                    self.consolidated_importances[name] = importance.detach().clone()
                else:
                    self.consolidated_importances[name] += importance.detach().clone()
                
                # Update previous parameter values for next task
                self.prev_params[name] = param.detach().clone()
    
    def update_omega(self, loss_scale=1.0):
        """
        Update importance estimates after each gradient step.
        This should be called after optimizer.step() but before zero_grad().
        
        Args:
            loss_scale: Scaling factor for gradients (for mixed precision training)
        """
        if self.current_task_id is None:
            return
            
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Calculate parameter update (new - old)
                delta_theta = param.detach() - self.prev_params[name]
                
                # Accumulate "path integral" of gradient × parameter update
                # This estimates how much this parameter contributed to reducing loss
                self.omega_accumulation[name] -= param.grad.detach() * delta_theta * loss_scale
                
                # Update previous parameters to current values
                self.prev_params[name] = param.detach().clone()
    
    def compute_consolidation_loss(self):
        """
        Compute the regularization loss to preserve important parameters.
        This should be added to the task loss during training.
        
        Returns:
            The regularization loss term to be added to the task loss
        """
        if not self.importances:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
            
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # For each parameter, calculate its importance-weighted squared distance
                # from its value at the end of each previous task
                for task_id, importances in self.importances.items():
                    if task_id != self.current_task_id and name in importances:
                        # Get parameter values at the end of previous task
                        prev_task_param = self.prev_params[name]
                        
                        # Calculate squared parameter change
                        delta = param - prev_task_param
                        
                        # Add importance-weighted penalty to the loss
                        loss += torch.sum(importances[name] * delta.pow(2))
        
        return self.si_lambda * loss
    
    def visualize_importances(self, layer_names=None, max_tasks=5, top_n_parameters=100):
        """
        Visualize parameter importance across different tasks.
        
        Args:
            layer_names: List of layer names to visualize (None for all layers)
            max_tasks: Maximum number of tasks to include in visualization
            top_n_parameters: Number of top parameters to visualize per layer
        """
        if not self.importances:
            print("No importance values available for visualization")
            return
            
        # If no layer names specified, use all layers
        if layer_names is None:
            layer_names = list(self.consolidated_importances.keys())
        
        # Get task IDs (limit to max_tasks for readability)
        task_ids = list(self.importances.keys())
        if len(task_ids) > max_tasks:
            # Take the first and last few tasks if we have more than max_tasks
            half = max_tasks // 2
            task_ids = task_ids[:half] + task_ids[-half:]
        
        # Visualize importances for each requested layer
        for layer_name in layer_names:
            if layer_name not in self.consolidated_importances:
                continue
                
            # For 2D weights (e.g., fully connected layers)
            if len(self.consolidated_importances[layer_name].shape) == 2:
                self._visualize_matrix_importance(layer_name, task_ids)
                self._visualize_top_parameters(layer_name, task_ids, top_n_parameters)
            # For 1D weights (e.g., biases)
            elif len(self.consolidated_importances[layer_name].shape) == 1:
                self._visualize_vector_importance(layer_name, task_ids)
            # For 4D weights (e.g., convolutional layers)
            elif len(self.consolidated_importances[layer_name].shape) == 4:
                self._visualize_conv_importance(layer_name, task_ids)
    
    def _visualize_matrix_importance(self, layer_name, task_ids):
        """Visualize importance for matrix parameters (e.g., weight matrices)."""
        # Create heatmap for consolidated importance
        plt.figure(figsize=(10, 8))
        
        # Get importance data and convert to numpy
        importance_data = self.consolidated_importances[layer_name].cpu().numpy()
        
        # Create a heatmap
        ax = sns.heatmap(importance_data, cmap="viridis", 
                         xticklabels=20, yticklabels=20)
        
        plt.title(f"Consolidated Parameter Importance: {layer_name}")
        plt.xlabel("Output Dimension")
        plt.ylabel("Input Dimension")
        
        # Save the figure
        filename = os.path.join(self.visualization_dir, f"si_importance_heatmap_{layer_name.replace('.', '_')}.png")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def _visualize_vector_importance(self, layer_name, task_ids):
        """Visualize importance for vector parameters (e.g., biases)."""
        plt.figure(figsize=(12, 6))
        
        # Get importance data and convert to numpy
        importance_data = self.consolidated_importances[layer_name].cpu().numpy()
        
        # Plot the importance values
        plt.bar(range(len(importance_data)), importance_data)
        
        plt.title(f"Parameter Importance: {layer_name}")
        plt.xlabel("Parameter Index")
        plt.ylabel("Importance")
        
        # Save the figure
        filename = os.path.join(self.visualization_dir, f"si_parameter_importance_{layer_name.replace('.', '_')}.png")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def _visualize_conv_importance(self, layer_name, task_ids):
        """Visualize importance for convolutional parameters."""
        importance_data = self.consolidated_importances[layer_name].cpu().numpy()
        
        # Reshape to 2D for visualization
        out_channels, in_channels, kh, kw = importance_data.shape
        reshaped_data = importance_data.reshape(out_channels, in_channels * kh * kw)
        
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(reshaped_data, cmap="viridis", 
                         xticklabels=20, yticklabels=20)
        
        plt.title(f"Consolidated Parameter Importance: {layer_name}")
        plt.xlabel("Input Channel * Kernel Position")
        plt.ylabel("Output Channel")
        
        # Save the figure
        filename = os.path.join(self.visualization_dir, f"si_importance_heatmap_{layer_name.replace('.', '_')}.png")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def _visualize_top_parameters(self, layer_name, task_ids, top_n):
        """Visualize importance of top N parameters across tasks."""
        if len(task_ids) == 0:
            return
            
        plt.figure(figsize=(14, 8))
        
        # Get the flattened consolidated importance values
        cons_importance = self.consolidated_importances[layer_name].cpu().flatten()
        
        # Get indices of top N important parameters
        _, top_indices = torch.topk(torch.tensor(cons_importance), min(top_n, len(cons_importance)))
        top_indices = top_indices.numpy()
        
        # Create a task × parameters matrix
        importance_matrix = np.zeros((len(task_ids), len(top_indices)))
        
        # Fill the matrix with importance values for each task and parameter
        for i, task_id in enumerate(task_ids):
            if task_id in self.importances and layer_name in self.importances[task_id]:
                task_importance = self.importances[task_id][layer_name].cpu().flatten()
                for j, param_idx in enumerate(top_indices):
                    importance_matrix[i, j] = task_importance[param_idx]
        
        # Create heatmap
        ax = sns.heatmap(importance_matrix, cmap="viridis")
        
        plt.title(f"Top {len(top_indices)} Parameter Importances Across Tasks: {layer_name}")
        plt.xlabel("Parameter Index (sorted by overall importance)")
        plt.ylabel("Task ID")
        plt.yticks(np.arange(len(task_ids)) + 0.5, task_ids)
        
        # Save the figure
        filename = os.path.join(self.visualization_dir, 
                               f"si_cross_task_importance_{layer_name.replace('.', '_')}.png")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    def visualize_importance_evolution(self, layer_name=None, top_k=5):
        """
        Visualize how parameter importance evolves across tasks.
        
        Args:
            layer_name: The layer to visualize (None to select one automatically)
            top_k: Number of top parameters to track
        """
        if not self.importances or len(self.importances) < 2:
            print("Need at least 2 tasks to visualize importance evolution")
            return
            
        # If no layer specified, choose the first one with 2D parameters
        if layer_name is None:
            for name, importance in self.consolidated_importances.items():
                if len(importance.shape) == 2:
                    layer_name = name
                    break
                    
        if layer_name is None or layer_name not in self.consolidated_importances:
            print("No suitable layer found for visualization")
            return
            
        # Get sorted task IDs
        task_ids = sorted(self.importances.keys())
        
        # For 2D parameters (weights)
        if len(self.consolidated_importances[layer_name].shape) == 2:
            # Find top-k most important parameters overall
            cons_importance = self.consolidated_importances[layer_name].cpu()
            flat_importance = cons_importance.flatten()
            _, top_indices = torch.topk(flat_importance, min(top_k, len(flat_importance)))
            
            # Convert flat indices to 2D indices
            rows, cols = [], []
            for idx in top_indices:
                idx = idx.item()
                r = idx // cons_importance.shape[1]
                c = idx % cons_importance.shape[1]
                rows.append(r)
                cols.append(c)
                
            # Track these parameters across tasks
            plt.figure(figsize=(12, 8))
            for i in range(len(rows)):
                r, c = rows[i], cols[i]
                param_importance = []
                
                for task_id in task_ids:
                    if task_id in self.importances and layer_name in self.importances[task_id]:
                        task_imp = self.importances[task_id][layer_name].cpu()
                        param_importance.append(task_imp[r, c].item())
                    else:
                        param_importance.append(0)
                
                plt.plot(range(len(task_ids)), param_importance, marker='o', 
                         label=f"Param ({r},{c})")
            
            plt.xlabel("Task Sequence")
            plt.ylabel("Parameter Importance")
            plt.title(f"Evolution of Top Parameter Importances: {layer_name}")
            plt.xticks(range(len(task_ids)), task_ids, rotation=45)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save the figure
            filename = os.path.join(self.visualization_dir, 
                                   f"si_importance_evolution_{layer_name.replace('.', '_')}.png")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
    
    def visualize_layer_importance(self):
        """Visualize overall importance of different layers across tasks."""
        if not self.importances:
            print("No importance values available for visualization")
            return
            
        # Get task IDs
        task_ids = sorted(self.importances.keys())
        
        # Calculate average importance per layer for each task
        layer_importances = {}
        for task_id in task_ids:
            layer_importances[task_id] = {}
            for layer_name, importance in self.importances[task_id].items():
                layer_importances[task_id][layer_name] = torch.mean(importance).item()
        
        # Find common layers across all tasks
        common_layers = set()
        for task_id in task_ids:
            if task_id in self.importances:
                if not common_layers:
                    common_layers = set(self.importances[task_id].keys())
                else:
                    common_layers &= set(self.importances[task_id].keys())
        
        common_layers = sorted(list(common_layers))
        
        # Create a task × layer matrix
        if not common_layers or not task_ids:
            print("No common layers found across tasks")
            return
            
        importance_matrix = np.zeros((len(task_ids), len(common_layers)))
        
        for i, task_id in enumerate(task_ids):
            for j, layer_name in enumerate(common_layers):
                importance_matrix[i, j] = layer_importances[task_id][layer_name]
        
        # Create heatmap
        plt.figure(figsize=(max(10, len(common_layers)), max(8, len(task_ids))))
        ax = sns.heatmap(importance_matrix, cmap="viridis", annot=True, fmt=".2e")
        
        plt.title("Layer Importance Across Tasks")
        plt.xlabel("Layer")
        plt.ylabel("Task ID")
        plt.xticks(np.arange(len(common_layers)) + 0.5, common_layers, rotation=90)
        plt.yticks(np.arange(len(task_ids)) + 0.5, task_ids)
        
        # Save the figure
        filename = os.path.join(self.visualization_dir, "si_layer_importance.png")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_importances(self, file_path):
        """
        Save importance values to a file.
        
        Args:
            file_path: Path to save the importance values
        """
        save_dict = {
            'task_importances': {
                task_id: {
                    name: imp.cpu() for name, imp in importances.items()
                } for task_id, importances in self.importances.items()
            },
            'consolidated_importances': {
                name: imp.cpu() for name, imp in self.consolidated_importances.items()
            }
        }
        torch.save(save_dict, file_path)
    
    def load_importances(self, file_path):
        """
        Load importance values from a file.
        
        Args:
            file_path: Path to the saved importance values
        """
        checkpoint = torch.load(file_path)
        
        # Load task importances
        device = next(self.model.parameters()).device
        for task_id, importances in checkpoint['task_importances'].items():
            self.importances[task_id] = {
                name: imp.to(device) for name, imp in importances.items()
            }
        
        # Load consolidated importances
        self.consolidated_importances = {
            name: imp.to(device) for name, imp in checkpoint['consolidated_importances'].items()
        } 