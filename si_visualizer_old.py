"""
Synaptic Intelligence Visualization Module for POLARIS.

This module provides functionality to visualize parameter importance in Synaptic Intelligence
when running POLARIS agents with SI enabled.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path
import os
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import signal
import time
import platform



def visualize_layer_importances_across_agents(agents, output_dir, layer_name):
    """
    Compare layer importance across different agents.
    
    Args:
        agents: Dictionary of agent_id to POLARISAgent instances
        output_dir: Directory to save visualizations
        layer_name: Layer name to compare across agents
    """
    # Create visualization directory
    vis_dir = Path(output_dir) / 'si_visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all agents with SI
    si_agents = {
        agent_id: agent for agent_id, agent in agents.items() 
        if hasattr(agent, 'use_si') and agent.use_si
    }
    
    if not si_agents:
        print("No agents with SI enabled found.")
        return
    
    # Components to visualize
    components = ['belief', 'policy']
    
    for component in components:
        # First check if this layer is excluded for any agent
        is_excluded = False
        for agent_id, agent in si_agents.items():
            if component == 'belief' and hasattr(agent, 'excluded_belief_layers'):
                if layer_name in agent.excluded_belief_layers:
                    print(f"Layer {layer_name} is excluded from belief SI for agent {agent_id}")
                    is_excluded = True
            elif component == 'policy' and hasattr(agent, 'excluded_policy_layers'):
                if layer_name in agent.excluded_policy_layers:
                    print(f"Layer {layer_name} is excluded from policy SI for agent {agent_id}")
                    is_excluded = True
        
        # Skip this layer if it's excluded for any agent
        if is_excluded:
            print(f"Skipping layer {layer_name} for {component} as it's excluded from SI")
            continue
        
        # Get the corresponding SI tracker for each agent
        si_trackers = {}
        for agent_id, agent in si_agents.items():
            if component == 'belief':
                si_trackers[agent_id] = agent.belief_si
            elif component == 'policy':
                si_trackers[agent_id] = agent.policy_si
        
        # Check if the layer exists in any agent
        layer_exists = False
        for agent_id, tracker in si_trackers.items():
            if layer_name in tracker.importance_scores:
                layer_exists = True
                break
        
        if not layer_exists:
            print(f"Layer {layer_name} not found in any agent's {component} tracker")
            continue
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # For each agent
        for agent_id, tracker in si_trackers.items():
            if layer_name in tracker.importance_scores:
                # Get importance scores
                importance = tracker.importance_scores[layer_name].detach().cpu().numpy()
                
                # Calculate mean absolute importance per input neuron
                if len(importance.shape) == 2:
                    # For 2D tensor, take mean across output dimension
                    mean_importance = np.mean(np.abs(importance), axis=1)
                else:
                    # Otherwise flatten
                    mean_importance = np.abs(importance.flatten())
                
                # Find top 20 values and their indices
                top_indices = np.argsort(mean_importance)[-20:]
                top_values = mean_importance[top_indices]
                
                # Plot top 20 values
                width = 0.35  # width of bars
                offset = 0.4 * (agent_id - list(si_agents.keys())[0])  # offset for agent
                ax.bar(
                    top_indices + offset,  # Position bars with offset
                    top_values,
                    width=width,
                    alpha=0.7,
                    label=f'Agent {agent_id}'
                )
        
        ax.set_title(f"Top Parameter Importances (Non-excluded) - {component} - {layer_name}")
        ax.set_xlabel("Parameter Index")
        ax.set_ylabel("Mean Absolute Importance")
        ax.legend()
        fig.savefig(vis_dir / f"si_importance_comparison_{component}_{layer_name.replace('.', '_')}.png",
                  bbox_inches='tight')
        plt.close(fig)
    
    print(f"Generated layer importance comparison visualizations in {vis_dir}")

def create_si_visualizations(agents, output_dir):
    """
    Create a complete set of SI visualizations.
    
    Args:
        agents: Dictionary of agent_id to POLARISAgent instances
        output_dir: Directory to save visualizations
    """
    try:        
        # Create visualization directory
        vis_dir = Path(output_dir) / 'si_visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Count how many agents use SI
        si_agents = 0
        for agent_id, agent in agents.items():
            if hasattr(agent, 'use_si') and agent.use_si:
                si_agents += 1
        
        if si_agents == 0:
            print("No agents with SI enabled found. Skipping SI visualizations.")
            return
            
        print(f"Generating SI visualizations for {si_agents} agents...")
        
        # First, print information about excluded layers for each agent
        for agent_id, agent in agents.items():
            if hasattr(agent, 'use_si') and agent.use_si:
                print(f"\nExamining SI configuration for agent {agent_id}:")
                if hasattr(agent, 'si_exclude_final_layers'):
                    print(f"  SI exclude final layers: {agent.si_exclude_final_layers}")
                    
                # Check excluded belief layers
                belief_excluded = []
                if hasattr(agent, 'excluded_belief_layers'):
                    belief_excluded = agent.excluded_belief_layers
                print(f"  Excluded belief layers: {belief_excluded}")
                
                # Check excluded policy layers
                policy_excluded = []
                if hasattr(agent, 'excluded_policy_layers'):
                    policy_excluded = agent.excluded_policy_layers
                print(f"  Excluded policy layers: {policy_excluded}")
        
        # Create task comparison visualizations
        for agent_id, agent in agents.items():
            if hasattr(agent, 'use_si') and agent.use_si:                
                try:
                    print(f"\nCreating visualizations for agent {agent_id}")
                    
                    # Create consolidated task comparison visualization
                    print(f"  Creating consolidated importance visualization (skipping excluded layers)...")
                    visualize_consolidated_importance_across_tasks(agent, output_dir)
                    print(f"  Completed consolidated importance visualization")
                except Exception as e:
                    print(f"Error creating visualizations for agent {agent_id}: {e}")
        
        # Find non-excluded key layers for comparison
        key_layers = [
            'transformer.transformer_encoder.layers.0.linear1.weight',
            'transformer.transformer_encoder.layers.0.self_attn.out_proj.weight',
            'fc_belief.weight',
            'policy_network.0.weight'
        ]
        
        # Filter out key layers that are excluded for any agent
        filtered_key_layers = []
        for layer in key_layers:
            excluded = False
            for agent_id, agent in agents.items():
                if hasattr(agent, 'use_si') and agent.use_si:
                    # Check if belief layer
                    if layer.startswith('fc_belief') or layer.startswith('transformer'):
                        if hasattr(agent, 'excluded_belief_layers') and layer in agent.excluded_belief_layers:
                            excluded = True
                            print(f"Skipping key layer {layer} as it's excluded from belief SI in agent {agent_id}")
                            break
                    # Check if policy layer
                    elif layer.startswith('policy_network'):
                        if hasattr(agent, 'excluded_policy_layers') and layer in agent.excluded_policy_layers:
                            excluded = True
                            print(f"Skipping key layer {layer} as it's excluded from policy SI in agent {agent_id}")
                            break
            
            if not excluded:
                filtered_key_layers.append(layer)
        
        print(f"\nCreating layer comparisons for non-excluded layers: {filtered_key_layers}")
        for layer in filtered_key_layers:
            try:
                print(f"Creating layer comparison for {layer}")
                visualize_layer_importances_across_agents(agents, output_dir, layer)
            except Exception as e:
                print(f"Error visualizing layer {layer}: {e}")
        
        print("\nSI visualization generation complete!")
                
    except Exception as e:
        import traceback
        print(f"Error creating SI visualizations: {e}")
        print("Detailed error:")
        traceback.print_exc()


def visualize_consolidated_importance_across_tasks(agent, output_dir):
    """
    Create a consolidated figure showing 1D parameter importance comparison 
    across tasks for all layers' weights and biases.
    
    Args:
        agent: A POLARISAgent instance with SI enabled
        output_dir: Directory to save visualizations
    """
    import signal
    import time
    import platform
    
    start_time = time.time()
    
    # Set up timeout only on platforms that support SIGALRM (not on Windows)
    use_timeout = platform.system() != "Windows" and hasattr(signal, 'SIGALRM')
    
    if use_timeout:
        # Define a timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError("Visualization timed out")
        
        # Set a timeout of 60 seconds
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
    
    try:
        if not hasattr(agent, 'use_si') or not agent.use_si:
            print(f"Agent {agent.agent_id} does not use Synaptic Intelligence, skipping visualization")
            return
            
        if not hasattr(agent, 'state_belief_si_trackers') or not agent.state_belief_si_trackers:
            print(f"Agent {agent.agent_id} has no state-specific SI trackers, skipping comparison")
            return
            
        # Create visualization directory
        vis_dir = Path(output_dir) / 'si_visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the true states we've seen
        true_states = sorted(list(agent.state_belief_si_trackers.keys()))
        num_states = len(true_states)
        
        if num_states < 2:
            print(f"Agent {agent.agent_id} has only seen {num_states} true states, not enough for comparison")
            return
        
        # Dictionary to store parameter average importance scores
        param_importance = {}
        
        # Get excluded belief layers for easy checking
        belief_excluded = set()
        if hasattr(agent, 'excluded_belief_layers'):
            belief_excluded = set(agent.excluded_belief_layers)
            print(f"Excluded belief layers: {belief_excluded}")
            
        # Get excluded policy layers for easy checking
        policy_excluded = set()
        if hasattr(agent, 'excluded_policy_layers'):
            policy_excluded = set(agent.excluded_policy_layers)
            print(f"Excluded policy layers: {policy_excluded}")
        
        # First, get belief processor parameters and calculate average importance
        for name, param in agent.belief_processor.named_parameters():
            # Skip excluded layers
            if name in belief_excluded:
                print(f"Skipping excluded belief layer: {name}")
                continue
                
            # Only include weights of main layers
            if not ('weight' in name) or param.numel() < 10:
                continue
                
            # Calculate average importance across states
            total_importance = 0.0
            count = 0
            
            for true_state in true_states:
                tracker = agent.state_belief_si_trackers.get(true_state)
                if tracker and name in tracker.importance_scores:
                    importance = tracker.importance_scores[name].detach().cpu().numpy()
                    total_importance += np.mean(np.abs(importance))
                    count += 1
            
            if count > 0:
                avg_importance = total_importance / count
                param_importance[name] = {
                    'network': 'belief',
                    'name': name,
                    'importance': avg_importance,
                    'shape': param.shape
                }
        
        # Then, get policy parameters and calculate average importance
        for name, param in agent.policy.named_parameters():
            # Skip excluded layers
            if name in policy_excluded:
                print(f"Skipping excluded policy layer: {name}")
                continue
                
            # Only include weights of main layers
            if not ('weight' in name) or param.numel() < 10:
                continue
                
            # Calculate average importance across states
            total_importance = 0.0
            count = 0
            
            for true_state in true_states:
                tracker = agent.state_policy_si_trackers.get(true_state)
                if tracker and name in tracker.importance_scores:
                    importance = tracker.importance_scores[name].detach().cpu().numpy()
                    total_importance += np.mean(np.abs(importance))
                    count += 1
            
            if count > 0:
                avg_importance = total_importance / count
                param_importance[name] = {
                    'network': 'policy',
                    'name': name,
                    'importance': avg_importance,
                    'shape': param.shape
                }
        
        if not param_importance:
            print(f"No parameters with importance scores found for agent {agent.agent_id}")
            return
            
        # Sort parameters by importance and select top N
        top_n = 8  # Show only the top 8 most important parameters
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1]['importance'], reverse=True)
        top_params = sorted_params[:top_n]
        
        print(f"Top {len(top_params)} most important parameters for agent {agent.agent_id}:")
        for i, (name, info) in enumerate(top_params):
            print(f"{i+1}. {name} ({info['network']}): {info['importance']:.6f}")
            
        # Create figure with one subplot per parameter
        fig, axes = plt.subplots(len(top_params), 1, figsize=(12, 4 * len(top_params)), constrained_layout=True)
        if len(top_params) == 1:
            axes = [axes]  # Make axes iterable if there's only one subplot
    
        # Get color mapping for states
        colors = plt.cm.tab10.colors[:num_states]
        color_map = {state: colors[i] for i, state in enumerate(true_states)}
    
        # For each top parameter
        for i, (param_name, param_info) in enumerate(top_params):
            ax = axes[i]
            network = param_info['network']
            
            # Set subplot title
            ax.set_title(f"Top {i+1}: {network.capitalize()} - {param_name} (Avg Importance: {param_info['importance']:.6f})", fontsize=10)
            
            # Get parameter shape and determine sampling
            param_shape = param_info['shape']
            param_size = np.prod(param_shape)
            
            # Sample the parameter elements
            max_elements = 100
            if param_size > max_elements:
                sample_rate = max(1, param_size // max_elements)
                x_pos = np.arange(0, param_size, sample_rate)
            else:
                x_pos = np.arange(param_size)
            
            # Bar width based on number of states
            bar_width = 0.8 / num_states
                
            # For each state, plot importance
            for state_idx, true_state in enumerate(true_states):
                # Get the appropriate SI tracker
                if network == 'belief':
                    tracker = agent.state_belief_si_trackers.get(true_state)
                else:
                    tracker = agent.state_policy_si_trackers.get(true_state)
                    
                if tracker is None or param_name not in tracker.importance_scores:
                    continue
                
                # Get importance scores
                importance = tracker.importance_scores[param_name].detach().cpu().numpy()
                
                # Flatten importance scores for 1D plotting
                flat_importance = importance.flatten()
                
                # Sample if necessary
                if param_size > max_elements:
                    sampled_importance = flat_importance[::sample_rate]
                else:
                    sampled_importance = flat_importance
                
                # Plot bar for each state with slight offset
                state_offset = state_idx * bar_width - 0.4 + (bar_width / 2)
                ax.bar(x_pos + state_offset, sampled_importance, width=bar_width, 
                      label=f"Task {true_state}",
                      color=color_map[true_state], alpha=0.7)
        
        # Add axis labels and grid
            ax.set_xlabel("Parameter Index", fontsize=9)
            ax.set_ylabel("Importance", fontsize=9)
        ax.grid(alpha=0.3)
    
            # Add legend for first subplot only
            if i == 0:
                ax.legend(fontsize=8)
        
        # Add overall title 
        fig.suptitle(f"Top {len(top_params)} Most Important Parameters (Non-excluded) - Agent {agent.agent_id}", fontsize=14)
    
        # Save the figure
        fig_path = vis_dir / f"top_parameter_importance_agent{agent.agent_id}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        elapsed_time = time.time() - start_time
        print(f"Saved top parameter importance comparison to {fig_path} in {elapsed_time:.2f} seconds")
    
    except TimeoutError:
        print(f"WARNING: Visualization for agent {agent.agent_id} timed out after 60 seconds. Skipping.")
        # Clean up any open plots
        plt.close('all')
    except Exception as e:
        print(f"Error in visualize_consolidated_importance_across_tasks for agent {agent.agent_id}: {e}")
        # Clean up any open plots
        plt.close('all')
    finally:
        # Cancel the alarm if timeout was used
        if use_timeout:
            signal.alarm(0) 