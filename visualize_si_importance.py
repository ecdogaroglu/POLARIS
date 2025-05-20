"""
Visualize parameter importance in Synaptic Intelligence (SI).

This script creates visualizations to demonstrate how parameter importance 
is calculated and updated across different tasks in Synaptic Intelligence.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import sys
import os
from matplotlib.patches import Rectangle, Arrow, FancyArrowPatch
from matplotlib.lines import Line2D

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.si import SILoss

# Set up a simple neural network for demonstration
class SimpleNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=8, output_size=1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define simple task data
def generate_task_data(task_id, n_samples=1000, input_size=10):
    """Generate synthetic data for a task"""
    X = torch.randn(n_samples, input_size)
    # Different tasks will have different patterns
    if task_id == 0:
        # Task A: First half features are important
        w = torch.zeros(input_size)
        w[:input_size//2] = 1.0
    elif task_id == 1:
        # Task B: Second half features are important
        w = torch.zeros(input_size)
        w[input_size//2:] = 1.0
    else:
        # Task C: All features with alternating signs
        w = torch.ones(input_size) * torch.tensor([1.0, -1.0] * (input_size//2))
    
    # Generate labels based on task-specific pattern
    noise = torch.randn(n_samples) * 0.1
    y = torch.matmul(X, w) + noise
    # Convert to binary classification
    y = (y > 0).float()
    
    return X, y.unsqueeze(1)

# Define a simple loss function
def loss_fn(model, batch):
    X, y = batch
    outputs = model(X)
    return nn.functional.binary_cross_entropy_with_logits(outputs, y)

def train_and_track(model, task_data, task_id, si_tracker, optimizer, n_epochs=5):
    """Train model on a task and track parameter trajectories"""
    X, y = task_data
    
    # Store parameter trajectories
    param_history = {name: [] for name, _ in model.named_parameters() if _.requires_grad}
    importance_history = {name: [] for name, _ in model.named_parameters() if _.requires_grad}
    path_integral_history = {name: [] for name, _ in model.named_parameters() if _.requires_grad}
    
    # Training loop
    for epoch in range(n_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss = loss_fn(model, (X, y))
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Update SI trajectories
        si_tracker.update_trajectory()
        
        # Store parameter values
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_history[name].append(param.data.clone().flatten().detach().cpu().numpy())
                if name in si_tracker.importance_scores:
                    importance_history[name].append(si_tracker.importance_scores[name].clone().flatten().detach().cpu().numpy())
                if name in si_tracker.param_path_integrals:
                    path_integral_history[name].append(si_tracker.param_path_integrals[name].clone().flatten().detach().cpu().numpy())
        
        print(f"Task {task_id}, Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Register task end in SI
    si_tracker.register_task()
    
    # One more record after task registration
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_history[name].append(param.data.clone().flatten().detach().cpu().numpy())
            if name in si_tracker.importance_scores:
                importance_history[name].append(si_tracker.importance_scores[name].clone().flatten().detach().cpu().numpy())
            if name in si_tracker.param_path_integrals:
                path_integral_history[name].append(si_tracker.param_path_integrals[name].clone().flatten().detach().cpu().numpy())
    
    return param_history, importance_history, path_integral_history

def visualize_parameter_importance(all_param_history, all_importance_history, all_path_integral_history, param_name="fc1.weight"):
    """Create visualizations showing parameter importance across tasks"""
    n_tasks = len(all_param_history)
    
    # Set up the figure
    plt.figure(figsize=(18, 12))
    gs = GridSpec(3, n_tasks)
    
    # For selected parameter, visualize across all tasks
    param_data = [hist[param_name] for hist in all_param_history]
    importance_data = [hist[param_name] for hist in all_importance_history]
    path_integral_data = [hist[param_name] for hist in all_path_integral_history]
    
    # 1. Parameter Trajectories
    ax1 = plt.subplot(gs[0, :])
    for t in range(n_tasks):
        # Plot trajectories for the selected parameters in each task
        param_trajectories = np.array(param_data[t])
        # Select a subset of parameters to visualize
        n_params = min(10, param_trajectories.shape[1])
        for i in range(n_params):
            plt.plot(np.arange(t*param_trajectories.shape[0], (t+1)*param_trajectories.shape[0]), 
                    param_trajectories[:, i], 
                    label=f"Param {i}" if t == 0 else "", 
                    alpha=0.7)
        
        # Mark task boundaries
        plt.axvline(x=(t+1)*param_trajectories.shape[0]-0.5, color='r', linestyle='--')
    
    plt.title(f"Parameter Trajectories for {param_name}")
    plt.xlabel("Training Steps")
    plt.ylabel("Parameter Value")
    if n_params <= 10:
        plt.legend(loc='upper right')
    
    # 2. Path Integral Accumulation (for each task)
    for t in range(n_tasks):
        ax = plt.subplot(gs[1, t])
        path_integral_values = np.array(path_integral_data[t])
        # Plot final accumulated path integral at the end of the task
        if path_integral_values.shape[0] > 0:
            plt.bar(np.arange(min(20, path_integral_values.shape[1])), 
                   path_integral_values[-1, :min(20, path_integral_values.shape[1])],
                   alpha=0.7)
        plt.title(f"Path Integral (Task {t+1})")
        plt.xlabel("Parameter Index")
        plt.ylabel("Path Integral")
    
    # 3. Importance Scores (for each task)
    for t in range(n_tasks):
        ax = plt.subplot(gs[2, t])
        importance_values = np.array(importance_data[t])
        # Plot final importance at the end of the task
        if importance_values.shape[0] > 0:
            plt.bar(np.arange(min(20, importance_values.shape[1])), 
                   importance_values[-1, :min(20, importance_values.shape[1])],
                   alpha=0.7)
        plt.title(f"Importance Scores (After Task {t+1})")
        plt.xlabel("Parameter Index")
        plt.ylabel("Importance Score")
    
    plt.tight_layout()
    plt.savefig(f"si_parameter_importance_{param_name.replace('.', '_')}.png")
    plt.close()

def visualize_importance_heatmap(model, si_tracker, fig_path="si_importance_heatmap.png"):
    """Visualize importance scores as a heatmap for all parameters"""
    plt.figure(figsize=(12, 10))
    
    # Get all parameter importances
    param_importances = {}
    
    # Collect and reshape parameter importance values
    for name, param in model.named_parameters():
        if param.requires_grad and name in si_tracker.importance_scores:
            # Flatten the importance tensor
            importance = si_tracker.importance_scores[name].detach().cpu().numpy()
            param_shape = importance.shape
            
            # Store the mean importance for each parameter
            param_importances[name] = np.mean(np.abs(importance))
            
            # For parameters that are matrices (2D), create a heatmap
            if len(param_shape) == 2 and param_shape[0] * param_shape[1] < 1000:  # Only for manageable sizes
                plt.figure(figsize=(8, 6))
                sns.heatmap(np.abs(importance), cmap='viridis', xticklabels=10, yticklabels=10)
                plt.title(f"Importance Heatmap: {name}")
                plt.tight_layout()
                plt.savefig(f"si_importance_heatmap_{name.replace('.', '_')}.png")
                plt.close()
    
    # Create a bar chart of average importances per layer
    plt.figure(figsize=(10, 6))
    layers = list(param_importances.keys())
    importances = [param_importances[layer] for layer in layers]
    
    plt.bar(range(len(layers)), importances)
    plt.xticks(range(len(layers)), layers, rotation=45, ha='right')
    plt.title("Average Parameter Importance by Layer")
    plt.xlabel("Layer")
    plt.ylabel("Mean Importance Score")
    plt.tight_layout()
    plt.savefig("si_layer_importance.png")
    plt.close()

def visualize_importance_evolution(all_importance_history, param_name="fc1.weight"):
    """Visualize how importance evolves across tasks for specific parameters"""
    n_tasks = len(all_importance_history)
    
    # Extract data for the selected parameters
    importance_data = [hist[param_name] for hist in all_importance_history]
    
    # Get final importance after each task
    final_importances = []
    for t in range(n_tasks):
        if len(importance_data[t]) > 0:
            # Get the final importance values for this task
            final_imp = importance_data[t][-1]
            final_importances.append(final_imp)
    
    if not final_importances:
        print(f"No importance data found for {param_name}")
        return
        
    # Select a subset of parameters to visualize (at most 10)
    n_params = min(10, final_importances[0].shape[0])
    param_indices = np.arange(n_params)
    
    # Create a stacked bar chart showing importance accumulation
    plt.figure(figsize=(12, 8))
    
    # Initialize the bottom of each bar to zero
    bottoms = np.zeros(n_params)
    
    # Plot importance from each task as a stacked bar
    for t in range(len(final_importances)):
        # For the first task, the importance is just the final importance
        if t == 0:
            task_importance = final_importances[0][:n_params]
        # For subsequent tasks, we need the difference from previous accumulated importance
        else:
            prev_importance = final_importances[t-1][:n_params]
            curr_importance = final_importances[t][:n_params]
            task_importance = curr_importance - prev_importance
            
        plt.bar(param_indices, task_importance, bottom=bottoms, 
                label=f'Task {t+1}', alpha=0.7)
        bottoms += task_importance
    
    plt.title(f'Importance Evolution Across Tasks - {param_name}')
    plt.xlabel('Parameter Index')
    plt.ylabel('Accumulated Importance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'si_importance_evolution_{param_name.replace(".", "_")}.png')
    plt.close()

def create_si_mechanism_diagram():
    """Create a diagram explaining the Synaptic Intelligence mechanism"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Disable axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Set the title
    ax.set_title('Synaptic Intelligence (SI) Mechanism', fontsize=16)
    
    # Draw the main components
    task_box_1 = Rectangle((0.1, 0.7), 0.2, 0.2, fill=True, alpha=0.3, ec="black", fc="blue")
    task_box_2 = Rectangle((0.4, 0.7), 0.2, 0.2, fill=True, alpha=0.3, ec="black", fc="green")
    param_box = Rectangle((0.25, 0.4), 0.2, 0.2, fill=True, alpha=0.3, ec="black", fc="gray")
    importance_box = Rectangle((0.55, 0.4), 0.2, 0.2, fill=True, alpha=0.3, ec="black", fc="red")
    loss_box = Rectangle((0.4, 0.1), 0.2, 0.2, fill=True, alpha=0.3, ec="black", fc="purple")
    
    # Add boxes to the plot
    ax.add_patch(task_box_1)
    ax.add_patch(task_box_2)
    ax.add_patch(param_box)
    ax.add_patch(importance_box)
    ax.add_patch(loss_box)
    
    # Add text labels
    ax.text(0.2, 0.8, "Task 1", ha='center', va='center', fontsize=12)
    ax.text(0.5, 0.8, "Task 2", ha='center', va='center', fontsize=12)
    ax.text(0.35, 0.5, "Parameters\nθ", ha='center', va='center', fontsize=12)
    ax.text(0.65, 0.5, "Importance\nΩ", ha='center', va='center', fontsize=12)
    ax.text(0.5, 0.2, "SI Loss\nL = λ/2 * Ω(θ-θ*)²", ha='center', va='center', fontsize=12)
    
    # Add arrows
    arrow_style = dict(arrowstyle='->', linewidth=2, color='black')
    
    # Task 1 to Parameters
    task1_param = FancyArrowPatch((0.2, 0.7), (0.3, 0.6), connectionstyle="arc3,rad=-0.2", **arrow_style)
    ax.add_patch(task1_param)
    ax.text(0.2, 0.65, "Update θ", fontsize=10, ha='center')
    
    # Task 2 to Parameters
    task2_param = FancyArrowPatch((0.5, 0.7), (0.4, 0.6), connectionstyle="arc3,rad=0.2", **arrow_style)
    ax.add_patch(task2_param)
    ax.text(0.5, 0.65, "Update θ", fontsize=10, ha='center')
    
    # Parameters to Importance (calculation)
    param_imp = FancyArrowPatch((0.45, 0.5), (0.55, 0.5), **arrow_style)
    ax.add_patch(param_imp)
    ax.text(0.5, 0.52, "Calculate\nΩ = ∫g·dθ / (Δθ²+ξ)", fontsize=10, ha='center')
    
    # Parameters and Importance to Loss
    param_loss = FancyArrowPatch((0.35, 0.4), (0.45, 0.3), connectionstyle="arc3,rad=-0.2", **arrow_style)
    imp_loss = FancyArrowPatch((0.65, 0.4), (0.55, 0.3), connectionstyle="arc3,rad=0.2", **arrow_style)
    ax.add_patch(param_loss)
    ax.add_patch(imp_loss)
    
    # Loss back to parameters (regularization)
    loss_param = FancyArrowPatch((0.4, 0.2), (0.3, 0.4), connectionstyle="arc3,rad=0.2", **arrow_style)
    ax.add_patch(loss_param)
    ax.text(0.3, 0.3, "Regularize", fontsize=10, ha='center')
    
    # Add explanation text
    explanation = """
    Synaptic Intelligence (SI) Process:
    
    1. Train model on Task 1, updating parameters θ
    2. Calculate parameter importance Ω based on parameter trajectory:
       • Accumulate path integrals of gradient * parameter change
       • Compute importance as path integral divided by squared parameter change
    3. When training on Task 2:
       • Use regularization loss to penalize changes to important parameters
       • Loss = λ/2 * Ω(θ-θ*)², where θ* are parameter values after Task 1
    4. Importance scores accumulate across tasks - parameters important 
       for multiple tasks receive higher protection
    """
    
    plt.figtext(0.02, 0.02, explanation, fontsize=12, ha='left', va='bottom')
    
    plt.tight_layout()
    plt.savefig('si_mechanism_diagram.png', dpi=200, bbox_inches='tight')
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model
    input_size = 10
    model = SimpleNN(input_size=input_size)
    
    # Set up optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Set up SI tracker
    si_tracker = SILoss(model, importance=100.0)
    
    # Define tasks
    n_tasks = 3
    tasks = [generate_task_data(task_id, input_size=input_size) for task_id in range(n_tasks)]
    
    # Track parameter history across tasks
    all_param_history = []
    all_importance_history = []
    all_path_integral_history = []
    
    # Train on each task
    for task_id, task_data in enumerate(tasks):
        print(f"\nTraining on Task {task_id+1}...")
        param_history, importance_history, path_integral_history = train_and_track(
            model, task_data, task_id, si_tracker, optimizer
        )
        
        all_param_history.append(param_history)
        all_importance_history.append(importance_history)
        all_path_integral_history.append(path_integral_history)
        
        # After each task, calculate SI loss that would be applied
        si_loss = si_tracker.calculate_loss()
        print(f"SI Loss after Task {task_id+1}: {si_loss.item():.4f}")
    
    # Visualize parameter importance for specific parameters
    for param_name in ["fc1.weight", "fc2.weight"]:
        visualize_parameter_importance(
            all_param_history, all_importance_history, all_path_integral_history, param_name
        )
        # Add importance evolution visualization
        visualize_importance_evolution(all_importance_history, param_name)
    
    # Visualize importance as heatmap
    visualize_importance_heatmap(model, si_tracker)
    
    # Create explanatory diagram
    create_si_mechanism_diagram()
    
    print("Visualizations created successfully!")

if __name__ == "__main__":
    main() 