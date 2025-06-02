#!/usr/bin/env python3
"""
POLARIS Brandl Social Learning Agent Performance Sweep

A comprehensive analysis tool for studying individual agent learning performance
across different network topologies and sizes in the Brandl social learning environment.

This script provides detailed insights into:
- Individual agent learning trajectories and convergence rates
- Learning rate calculations using log-linear regression
- Performance disparities between slowest and fastest learners
- Network topology effects on learning dynamics
- Comparative analysis across different network sizes
- Network position analysis for slowest agents
- Signal analysis across episodes

Key Features:
- Calculates learning rates (r) for each agent using incorrect probability decay
- Generates comprehensive visualizations with confidence intervals
- Highlights extreme learners (slowest in red, fastest in green)
- Supports multiple episodes with proper agent resetting
- Produces both aggregate plots and detailed JSON results
- Supports multiple network types: complete, ring, star, random
- Analyzes network positions of slowest agents
- Tracks private signals received by agents over time

Usage:
    python experiments/brandl_sweep.py
    python experiments/brandl_sweep.py --agent-counts 1 2 4 6 8 10 --network-types complete ring star --episodes 5

Outputs:
- fastest_slowest_network_sizes_evolution.png: Fastest/slowest agent trajectories with CIs across network sizes
- fastest_slowest_network_types_evolution.png: Fastest/slowest agent trajectories with CIs across network types
- slowest_agent_network_positions.png: Network position frequency analysis for slowest agents
- average_private_signals.png: Average private signals received over time
- agent_performance_results.json: Complete numerical results with learning rates

Author: POLARIS Framework
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
import networkx as nx

from polaris.config.experiment_config import (
    ExperimentConfig, AgentConfig, TrainingConfig, BrandlConfig
)
from polaris.environments import SocialLearningEnvironment
from polaris.training.simulation import run_experiment
from polaris.utils.math import calculate_learning_rate

# --- FOCUSED PARAMETERS ---
AGENT_COUNTS = [1, 2, 4, 8]  # Network sizes to compare
NETWORK_TYPES = ['complete', 'ring', 'star', 'random']  # Network types to compare
EPISODES = 5  # Multiple episodes for confidence intervals
HORIZON = 50  # Shorter for faster execution
SIGNAL_ACCURACY = 0.75
RESULTS_DIR = Path("results/brandl_experiment/agent_performance")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_agent_experiment(num_agents, network_type, episodes, horizon, signal_accuracy, seed=0, device='cpu', use_gnn=True):
    """Run Brandl experiment for multiple episodes and return individual agent performance metrics with confidence intervals."""
    
    # Store results from all episodes
    all_episodes_data = []
    
    # Run each episode with fresh agents
    for episode_idx in range(episodes):
        episode_seed = seed + episode_idx
        print(f"    Episode {episode_idx + 1}/{episodes} (seed {episode_seed})...")
        
        # Configuration for agent performance tracking
        agent_config = AgentConfig(
            learning_rate=1e-3,
            discount_factor=0.99,
            use_gnn=use_gnn,
            use_si=False
        )
        
        training_config = TrainingConfig(
            batch_size=32,
            buffer_capacity=100,
            update_interval=1,  # Update networks every step instead of every 10 steps
            num_episodes=1,  # Single episode per run
            horizon=horizon
        )
        
        env_config = BrandlConfig(
            environment_type='brandl',
            num_agents=num_agents,
            seed=episode_seed,  # Different seed for each episode
            signal_accuracy=signal_accuracy,
            network_type=network_type,
            network_density=0.5
        )
        
        config = ExperimentConfig(
            agent=agent_config,
            training=training_config,
            environment=env_config,
            device=device,
            output_dir=str(RESULTS_DIR),
            exp_name=f"agents_{network_type}_{num_agents}_ep{episode_idx}",
            save_model=False,
            load_model=None,
            eval_only=False,
            plot_internal_states=False,  # Disable internal plotting to avoid conflicts
            plot_allocations=False,      # Disable internal plotting to avoid conflicts
            latex_style=False,
            use_tex=False
        )
        
        # Add a custom attribute to bypass internal plotting completely
        config.disable_plotting = True
        
        # Create fresh environment for this episode
        # Use a fixed seed for network generation (based on network_type and num_agents)
        # but different seed for agent learning and signal generation
        network_seed = hash((network_type, num_agents)) % (2**31)  # Fixed seed for network structure
        
        network_params = None
        if network_type == 'random':
            network_params = {'density': env_config.network_density}
        
        env = SocialLearningEnvironment(
            num_agents=num_agents,
            num_states=2,
            signal_accuracy=signal_accuracy,
            network_type=network_type,
            network_params=network_params,
            horizon=horizon,
            seed=network_seed  # Use fixed seed for consistent network structure
        )
        
        # After network creation, reseed the environment for episode-specific randomness
        env.seed(episode_seed)
        
        # Run experiment with fresh agents (this will automatically create new agents)
        episodic_metrics, processed_metrics = run_experiment(env, config)
        
        # Extract individual agent performance for this episode
        if 'episodes' in episodic_metrics and episodic_metrics['episodes']:
            episode_data = episodic_metrics['episodes'][0]
            
            if 'incorrect_probs' in episode_data:
                incorrect_probs = np.array(episode_data['incorrect_probs'])
                if len(incorrect_probs) > 0 and isinstance(incorrect_probs[0], np.ndarray):
                    # Extract time series for each agent in this episode
                    episode_trajectories = []
                    episode_learning_rates = {}
                    
                    for agent_id in range(num_agents):
                        # Extract this agent's incorrect probabilities over time
                        agent_trajectory = [float(step_probs[agent_id]) for step_probs in incorrect_probs]
                        episode_trajectories.append(agent_trajectory)
                        
                        # Calculate learning rate for this agent in this episode
                        episode_learning_rates[agent_id] = calculate_learning_rate(agent_trajectory)
                    
                    # Extract signal data if available
                    episode_signals = []
                    if 'signals' in episode_data:
                        signals_data = episode_data['signals']
                        if len(signals_data) > 0:
                            # signals_data should be a list of arrays, one per time step
                            for step_signals in signals_data:
                                if isinstance(step_signals, np.ndarray):
                                    episode_signals.append(step_signals.copy())
                                else:
                                    episode_signals.append(np.array(step_signals))
                    
                    # Extract attention weights if available
                    episode_attention_weights = []
                    if 'attention_weights' in episode_data:
                        attention_data = episode_data['attention_weights']
                        if len(attention_data) > 0:
                            # attention_data should be a list of attention matrices, one per time step
                            for step_attention in attention_data:
                                if isinstance(step_attention, np.ndarray):
                                    episode_attention_weights.append(step_attention.copy())
                                else:
                                    episode_attention_weights.append(np.array(step_attention))
                    
                    all_episodes_data.append({
                        'trajectories': episode_trajectories,
                        'learning_rates': episode_learning_rates,
                        'final_probs': [traj[-1] for traj in episode_trajectories],
                        'episode_seed': episode_seed,
                        'true_state': env.true_state,
                        'network_matrix': env.network.copy(),
                        'signals': episode_signals,
                        'attention_weights': episode_attention_weights,
                        'network_type': network_type,
                        'num_agents': num_agents
                    })
    
    # Identify fastest and slowest agents for each episode separately
    fastest_trajectories = []
    slowest_trajectories = []
    fastest_learning_rates = []
    slowest_learning_rates = []
    slowest_positions = []  # Track network positions of slowest agents
    
    for ep in all_episodes_data:
        # For this episode, find the fastest and slowest agents
        episode_learning_rates = ep['learning_rates']
        
        # Identify fastest and slowest for this specific episode
        episode_fastest_id = max(episode_learning_rates.keys(), key=lambda k: episode_learning_rates[k])
        episode_slowest_id = min(episode_learning_rates.keys(), key=lambda k: episode_learning_rates[k])
        
        # Store trajectories and learning rates for this episode
        fastest_trajectories.append(ep['trajectories'][episode_fastest_id])
        slowest_trajectories.append(ep['trajectories'][episode_slowest_id])
        fastest_learning_rates.append(episode_learning_rates[episode_fastest_id])
        slowest_learning_rates.append(episode_learning_rates[episode_slowest_id])
        
        # Store network position of slowest agent
        slowest_positions.append(episode_slowest_id)
    
    # Calculate overall statistics
    mean_fastest_lr = np.mean(fastest_learning_rates)
    mean_slowest_lr = np.mean(slowest_learning_rates)
    
    # For display purposes, use the most common fastest/slowest agent IDs
    fastest_agent_counts = Counter()
    slowest_agent_counts = Counter()
    
    for ep in all_episodes_data:
        episode_learning_rates = ep['learning_rates']
        episode_fastest_id = max(episode_learning_rates.keys(), key=lambda k: episode_learning_rates[k])
        episode_slowest_id = min(episode_learning_rates.keys(), key=lambda k: episode_learning_rates[k])
        fastest_agent_counts[episode_fastest_id] += 1
        slowest_agent_counts[episode_slowest_id] += 1
    
    # Most common fastest and slowest agents (for labeling)
    most_common_fastest = fastest_agent_counts.most_common(1)[0][0]
    most_common_slowest = slowest_agent_counts.most_common(1)[0][0]
    
    # Calculate mean trajectories and confidence intervals
    def calculate_trajectory_stats(trajectories):
        """Calculate mean and 95% confidence intervals for trajectories."""
        trajectories = np.array(trajectories)  # Shape: (episodes, time_steps)
        mean_traj = np.mean(trajectories, axis=0)
        
        # Calculate 95% confidence intervals
        if len(trajectories) > 1:
            sem = stats.sem(trajectories, axis=0)  # Standard error of mean
            ci = stats.t.interval(0.95, len(trajectories)-1, loc=mean_traj, scale=sem)
            ci_lower, ci_upper = ci
            
            # Clamp confidence intervals to valid probability range [0, 1]
            ci_lower = np.clip(ci_lower, 0.0, 1.0)
            ci_upper = np.clip(ci_upper, 0.0, 1.0)
        else:
            ci_lower = ci_upper = mean_traj
            
        return mean_traj, ci_lower, ci_upper
    
    slowest_mean, slowest_ci_lower, slowest_ci_upper = calculate_trajectory_stats(slowest_trajectories)
    fastest_mean, fastest_ci_lower, fastest_ci_upper = calculate_trajectory_stats(fastest_trajectories)
    
    agent_performance = {
        'slowest_trajectories': slowest_trajectories,
        'fastest_trajectories': fastest_trajectories,
        'slowest_mean': slowest_mean,
        'fastest_mean': fastest_mean,
        'slowest_ci': (slowest_ci_lower, slowest_ci_upper),
        'fastest_ci': (fastest_ci_lower, fastest_ci_upper),
        'slowest_agent_id': int(most_common_slowest),
        'fastest_agent_id': int(most_common_fastest),
        'learning_rates': {most_common_slowest: mean_slowest_lr, most_common_fastest: mean_fastest_lr},
        'num_episodes': episodes,
        'all_episodes_data': all_episodes_data,  # Keep for detailed analysis
        'slowest_positions': slowest_positions,  # Network positions of slowest agents
        'slowest_position_counts': dict(slowest_agent_counts)  # Frequency of each position being slowest
    }
    
    return agent_performance


def plot_network_positions(results, args):
    """Create network graph plots showing frequency of slowest agents at each position."""
    print("Creating network position analysis plots...")
    
    # Only analyze star and random networks (complete and ring don't have meaningful position differences)
    relevant_networks = ['star', 'random']
    available_networks = [nt for nt in relevant_networks if nt in args.network_types]
    
    if not available_networks:
        print("No relevant network types (star, random) found for position analysis. Skipping...")
        return
    
    # Create network graphs showing slowest agent frequencies
    n_networks = len(available_networks)
    n_cols = min(2, n_networks)
    n_rows = (n_networks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    if n_networks == 1:
        axes = [axes]
    elif n_networks <= 2:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    for network_type in available_networks:
        ax = axes[plot_idx]
        
        # Find the largest network size for this network type to use as the representative
        agent_counts = sorted([ac for ac in results[network_type].keys() if ac > 1])
        if not agent_counts:
            plot_idx += 1
            continue
            
        # Use the largest network size for visualization
        num_agents = agent_counts[-1]
        performance = results[network_type][num_agents]
        
        if performance['slowest_mean'] is None or not performance['all_episodes_data']:
            plot_idx += 1
            continue
        
        # Get network matrix from the first episode
        network_matrix = performance['all_episodes_data'][0]['network_matrix']
        position_counts = performance['slowest_position_counts']
        
        # Create NetworkX graph
        G = nx.from_numpy_array(network_matrix)
        
        # Calculate node positions based on network type with proper scaling
        if network_type == 'complete':
            pos = nx.circular_layout(G, scale=0.8)
        elif network_type == 'ring':
            pos = nx.circular_layout(G, scale=0.8)
        elif network_type == 'star':
            pos = nx.spring_layout(G, k=1.5, iterations=100, scale=0.8)
        elif network_type == 'random':
            pos = nx.spring_layout(G, k=1.0, iterations=100, scale=0.8)
        else:
            pos = nx.spring_layout(G, scale=0.8)
        
        # Calculate frequencies and normalize
        total_episodes = sum(position_counts.values())
        node_frequencies = {}
        for node in range(num_agents):
            freq = position_counts.get(node, 0)
            node_frequencies[node] = freq / total_episodes * 100 if total_episodes > 0 else 0
        
        # Create node colors based on frequency (red = high frequency of being slowest)
        node_colors = []
        max_freq = max(node_frequencies.values()) if node_frequencies.values() else 1
        for node in range(num_agents):
            freq = node_frequencies[node]
            # Color intensity based on frequency (white to red)
            intensity = freq / max_freq if max_freq > 0 else 0
            node_colors.append((1.0, 1.0 - intensity, 1.0 - intensity))  # RGB: white to red
        
        # Draw the network with larger nodes and black frames
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.7, width=3.5, edge_color='gray')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=2500, alpha=0.9, edgecolors='black', linewidths=2)
        
        # Add labels showing frequency percentages with better formatting
        labels = {}
        for node in range(num_agents):
            freq = node_frequencies[node]
            labels[node] = f'{freq:.0f}%'
        
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=18, font_weight='bold', 
                               font_color='black')
        
        ax.set_title(f'{network_type.capitalize()} Network ({num_agents} agents)\nFrequency of Being Slowest Agent', 
                    fontweight='bold', fontsize=16)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        
        plot_idx += 1
    
    # Hide unused subplots
    if len(axes) > plot_idx:
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
    
    # Add a centered color legend (only if we have plots)
    if plot_idx > 0:
        from matplotlib.colors import LinearSegmentedColormap
        
        # Position the legend below the plots with more space
        legend_x = 0.35  # Center horizontally
        legend_y = 0.15  # Position below the plots
        legend_width = 0.3
        legend_height = 0.04
        
        # Create gradient rectangle using figure coordinates
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        
        # Add the legend to the figure (not to a specific axis)
        fig_legend_ax = fig.add_axes([legend_x, legend_y, legend_width, legend_height])
        fig_legend_ax.imshow(gradient, aspect='auto', cmap='Reds', alpha=0.8)
        fig_legend_ax.set_xlim(0, 100)
        fig_legend_ax.set_ylim(0, 1)
        fig_legend_ax.set_xticks([0, 100])
        fig_legend_ax.set_xticklabels(['0%', 'Max%'], fontsize=12, fontweight='bold')
        fig_legend_ax.set_yticks([])
        fig_legend_ax.set_xlabel('Slowest Agent Frequency', fontsize=13, fontweight='bold', labelpad=10)
        
        # Remove the box around the legend
        for spine in fig_legend_ax.spines.values():
            spine.set_visible(False)
    
    plt.suptitle('Network Position Analysis: Frequency of Being Slowest Agent (Star & Random Networks)', 
                 fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    plot_path = RESULTS_DIR / "slowest_agent_network_positions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved network position analysis to {plot_path}")
    plt.close()


def plot_signal_analysis(results, args):
    """Create plots showing private signals received by slowest agents over time."""
    print("Creating signal analysis plots...")
    
    # Collect signal data across all configurations
    signal_data = {}
    
    for network_type in args.network_types:
        signal_data[network_type] = {}
        
        for num_agents in args.agent_counts:
            performance = results[network_type][num_agents]
            if performance['slowest_mean'] is None:
                continue
                
            # Extract signal data from all episodes, focusing on slowest agents
            all_episode_signals = []
            
            for ep_idx, ep_data in enumerate(performance['all_episodes_data']):
                if 'signals' in ep_data and len(ep_data['signals']) > 0:
                    # ep_data['signals'] is a list of arrays, one per time step
                    episode_signals = ep_data['signals']
                    true_state = ep_data['true_state']
                    
                    # Get the slowest agent ID for this episode
                    slowest_agent_id = performance['slowest_positions'][ep_idx]
                    
                    # Calculate signal correctness for the slowest agent over time
                    slowest_signal_correctness = []
                    for step_signals in episode_signals:
                        if len(step_signals) > slowest_agent_id:
                            # Check if the slowest agent received the correct signal
                            agent_signal = step_signals[slowest_agent_id]
                            correctness = 1.0 if agent_signal == true_state else 0.0
                            slowest_signal_correctness.append(correctness)
                    
                    if len(slowest_signal_correctness) > 0:
                        all_episode_signals.append(slowest_signal_correctness)
            
            if len(all_episode_signals) > 0:
                signal_data[network_type][num_agents] = all_episode_signals
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot 1: Signal correctness across network sizes (using first network type)
    if args.network_types:
        primary_network = args.network_types[0]
        ax = axes[0]
        
        for num_agents in sorted(signal_data.get(primary_network, {}).keys()):
            episode_signals = signal_data[primary_network][num_agents]
            
            if len(episode_signals) > 0:
                # Calculate mean and CI across episodes
                max_len = max(len(ep) for ep in episode_signals)
                padded_signals = []
                
                for ep_signals in episode_signals:
                    # Pad shorter episodes with the last value
                    if len(ep_signals) < max_len:
                        padded = ep_signals + [ep_signals[-1]] * (max_len - len(ep_signals))
                    else:
                        padded = ep_signals[:max_len]
                    padded_signals.append(padded)
                
                signals_array = np.array(padded_signals)
                mean_signals = np.mean(signals_array, axis=0)
                
                if len(episode_signals) > 1:
                    sem = stats.sem(signals_array, axis=0)
                    ci = stats.t.interval(0.95, len(episode_signals)-1, loc=mean_signals, scale=sem)
                    ci_lower, ci_upper = ci
                else:
                    ci_lower = ci_upper = mean_signals
                
                time_steps = range(len(mean_signals))
                
                if num_agents == 1:
                    label = 'Autarky'
                else:
                    label = f'{num_agents} agents'
                
                ax.plot(time_steps, mean_signals, label=label, linewidth=2)
                ax.fill_between(time_steps, ci_lower, ci_upper, alpha=0.2)
        
        ax.set_xlabel('Time Steps', fontweight='bold')
        ax.set_ylabel('Fraction of Correct Signals (Slowest Agents)', fontweight='bold')
        ax.set_title(f'{primary_network.capitalize()} Network\nSlowest Agents Signal Correctness Over Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=SIGNAL_ACCURACY, color='red', linestyle='--', alpha=0.7, label=f'Expected ({SIGNAL_ACCURACY})')
    
    # Plot 2: Signal correctness across network types (using middle agent count)
    if len(args.agent_counts) > 1:
        middle_idx = len(args.agent_counts) // 2
        fixed_agent_count = args.agent_counts[-1]
        ax = axes[1]
        
        for network_type in args.network_types:
            if fixed_agent_count in signal_data.get(network_type, {}):
                episode_signals = signal_data[network_type][fixed_agent_count]
                
                if len(episode_signals) > 0:
                    # Calculate mean and CI across episodes
                    max_len = max(len(ep) for ep in episode_signals)
                    padded_signals = []
                    
                    for ep_signals in episode_signals:
                        if len(ep_signals) < max_len:
                            padded = ep_signals + [ep_signals[-1]] * (max_len - len(ep_signals))
                        else:
                            padded = ep_signals[:max_len]
                        padded_signals.append(padded)
                    
                    signals_array = np.array(padded_signals)
                    mean_signals = np.mean(signals_array, axis=0)
                    
                    if len(episode_signals) > 1:
                        sem = stats.sem(signals_array, axis=0)
                        ci = stats.t.interval(0.95, len(episode_signals)-1, loc=mean_signals, scale=sem)
                        ci_lower, ci_upper = ci
                    else:
                        ci_lower = ci_upper = mean_signals
                    
                    time_steps = range(len(mean_signals))
                    
                    ax.plot(time_steps, mean_signals, label=network_type.capitalize(), linewidth=2)
                    ax.fill_between(time_steps, ci_lower, ci_upper, alpha=0.2)
        
        ax.set_xlabel('Time Steps', fontweight='bold')
        ax.set_ylabel('Fraction of Correct Signals (Slowest Agents)', fontweight='bold')
        ax.set_title(f'Slowest Agents Signal Correctness Across Network Types\n({fixed_agent_count} Agents)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=SIGNAL_ACCURACY, color='red', linestyle='--', alpha=0.7, label=f'Expected ({SIGNAL_ACCURACY})')
    
    # Hide unused subplots
    for i in range(2, 4):
        axes[i].set_visible(False)
    
    plt.suptitle('Slowest Agents Private Signal Analysis: Correctness Over Time', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / "average_private_signals.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved signal analysis to {plot_path}")
    plt.close()


def plot_signal_quality_vs_learning_performance(results, args):
    """Create dedicated plots showing signal quality vs learning performance with fastest/slowest distinction."""
    print("Creating dedicated signal quality vs learning performance plots...")
    
    # Collect data for analysis
    all_agent_data = []
    
    for network_type in args.network_types:
        for num_agents in args.agent_counts:
            if num_agents == 1:  # Skip autarky
                continue
                
            performance = results[network_type][num_agents]
            if performance['slowest_mean'] is None:
                continue
                
            for ep_idx, ep_data in enumerate(performance['all_episodes_data']):
                if 'signals' in ep_data and len(ep_data['signals']) > 0:
                    episode_signals = ep_data['signals']
                    true_state = ep_data['true_state']
                    learning_rates = ep_data['learning_rates']
                    
                    # Get fastest and slowest agent IDs for this episode
                    slowest_agent_id = performance['slowest_positions'][ep_idx]
                    fastest_agent_id = max(learning_rates.keys(), key=lambda k: learning_rates[k])
                    
                    # Calculate signal correctness for each agent
                    for agent_id in range(num_agents):
                        agent_signal_correctness = []
                        for step_signals in episode_signals:
                            if len(step_signals) > agent_id:
                                agent_signal = step_signals[agent_id]
                                correctness = 1.0 if agent_signal == true_state else 0.0
                                agent_signal_correctness.append(correctness)
                        
                        if len(agent_signal_correctness) > 0:
                            avg_signal_correctness = np.mean(agent_signal_correctness)
                            agent_lr = learning_rates.get(agent_id, 0.0)
                            
                            # Determine agent category
                            if agent_id == slowest_agent_id:
                                agent_category = 'slowest'
                            elif agent_id == fastest_agent_id:
                                agent_category = 'fastest'
                            else:
                                agent_category = 'other'
                            
                            all_agent_data.append({
                                'network_type': network_type,
                                'num_agents': num_agents,
                                'agent_id': agent_id,
                                'episode': ep_idx,
                                'signal_correctness': avg_signal_correctness,
                                'learning_rate': agent_lr,
                                'agent_category': agent_category,
                                'final_incorrect_prob': ep_data['trajectories'][agent_id][-1]
                            })
    
    if not all_agent_data:
        print("No signal data available for analysis")
        return
    
    # Create two plots: Size comparison and Type comparison (only learning rate)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Signal Quality vs Learning Rate - Size Comparison
    ax1 = axes[0]
    
    # Use the first network type for size comparison
    primary_network = args.network_types[0]
    size_data = [d for d in all_agent_data if d['network_type'] == primary_network]
    
    slowest_data = [d for d in size_data if d['agent_category'] == 'slowest']
    fastest_data = [d for d in size_data if d['agent_category'] == 'fastest']
    other_data = [d for d in size_data if d['agent_category'] == 'other']
    
    if slowest_data:
        slowest_signals = [d['signal_correctness'] for d in slowest_data]
        slowest_lrs = [d['learning_rate'] for d in slowest_data]
        ax1.scatter(slowest_signals, slowest_lrs, c='red', alpha=0.8, s=80, label='Slowest Agents', edgecolors='darkred', linewidth=1)
    
    if fastest_data:
        fastest_signals = [d['signal_correctness'] for d in fastest_data]
        fastest_lrs = [d['learning_rate'] for d in fastest_data]
        ax1.scatter(fastest_signals, fastest_lrs, c='green', alpha=0.8, s=80, label='Fastest Agents', edgecolors='darkgreen', linewidth=1)
    
    if other_data:
        other_signals = [d['signal_correctness'] for d in other_data]
        other_lrs = [d['learning_rate'] for d in other_data]
        ax1.scatter(other_signals, other_lrs, c='lightblue', alpha=0.6, s=50, label='Other Agents', edgecolors='blue', linewidth=0.5)
    
    # Add regression line and p-value for size comparison
    if len(size_data) > 2:
        all_signals = [d['signal_correctness'] for d in size_data]
        all_lrs = [d['learning_rate'] for d in size_data]
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_signals, all_lrs)
        
        # Create regression line
        x_range = np.linspace(min(all_signals), max(all_signals), 100)
        y_pred = slope * x_range + intercept
        
        ax1.plot(x_range, y_pred, 'k--', alpha=0.8, linewidth=2, 
                label=f'Regression (p={p_value:.3f})')
    
    ax1.set_xlabel('Average Signal Correctness', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Learning Rate', fontweight='bold', fontsize=12)
    ax1.set_title(f'Signal Quality vs Learning Rate\n({primary_network.capitalize()} Network - Size Comparison)', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=SIGNAL_ACCURACY, color='black', linestyle='--', alpha=0.7, label=f'Expected Signal Accuracy ({SIGNAL_ACCURACY})')
    
    # Plot 2: Signal Quality vs Learning Rate - Type Comparison
    ax2 = axes[1]
    
    # Use a fixed agent count for type comparison
    middle_idx = len(args.agent_counts) // 2
    fixed_agent_count = args.agent_counts[-1]
    type_data = [d for d in all_agent_data if d['num_agents'] == fixed_agent_count]
    
    slowest_data = [d for d in type_data if d['agent_category'] == 'slowest']
    fastest_data = [d for d in type_data if d['agent_category'] == 'fastest']
    other_data = [d for d in type_data if d['agent_category'] == 'other']
    
    if slowest_data:
        slowest_signals = [d['signal_correctness'] for d in slowest_data]
        slowest_lrs = [d['learning_rate'] for d in slowest_data]
        ax2.scatter(slowest_signals, slowest_lrs, c='red', alpha=0.8, s=80, label='Slowest Agents', edgecolors='darkred', linewidth=1)
    
    if fastest_data:
        fastest_signals = [d['signal_correctness'] for d in fastest_data]
        fastest_lrs = [d['learning_rate'] for d in fastest_data]
        ax2.scatter(fastest_signals, fastest_lrs, c='green', alpha=0.8, s=80, label='Fastest Agents', edgecolors='darkgreen', linewidth=1)
    
    if other_data:
        other_signals = [d['signal_correctness'] for d in other_data]
        other_lrs = [d['learning_rate'] for d in other_data]
        ax2.scatter(other_signals, other_lrs, c='lightblue', alpha=0.6, s=50, label='Other Agents', edgecolors='blue', linewidth=0.5)
    
    # Add regression line and p-value for type comparison
    if len(type_data) > 2:
        all_signals = [d['signal_correctness'] for d in type_data]
        all_lrs = [d['learning_rate'] for d in type_data]
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_signals, all_lrs)
        
        # Create regression line
        x_range = np.linspace(min(all_signals), max(all_signals), 100)
        y_pred = slope * x_range + intercept
        
        ax2.plot(x_range, y_pred, 'k--', alpha=0.8, linewidth=2, 
                label=f'Regression (p={p_value:.3f})')
        
    
    ax2.set_xlabel('Average Signal Correctness', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontweight='bold', fontsize=12)
    ax2.set_title(f'Signal Quality vs Learning Rate\n({fixed_agent_count} Agents - Network Type Comparison)', fontweight='bold', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=SIGNAL_ACCURACY, color='black', linestyle='--', alpha=0.7, label=f'Expected Signal Accuracy ({SIGNAL_ACCURACY})')
    
    plt.suptitle('Signal Quality vs Learning Rate Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    plot_path = RESULTS_DIR / "signal_quality_vs_learning_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved dedicated signal quality vs learning performance plot to {plot_path}")
    plt.close()


def plot_neighbor_count_vs_learning_performance(results, args):
    """Create plots showing neighbor count vs learning performance with fastest/slowest distinction."""
    print("Creating neighbor count vs learning performance plots...")
    
    # Only analyze star and random networks (complete and ring don't have meaningful neighbor count variations)
    relevant_networks = ['star', 'random']
    available_networks = [nt for nt in relevant_networks if nt in args.network_types]
    
    if not available_networks:
        print("No relevant network types (star, random) found for neighbor count analysis. Skipping...")
        return
    
    # Collect data for analysis
    all_agent_data = []
    
    for network_type in available_networks:
        for num_agents in args.agent_counts:
            if num_agents <= 2:  # Skip autarky and 2 agents (not meaningful for neighbor analysis)
                continue
                
            performance = results[network_type][num_agents]
            if performance['slowest_mean'] is None:
                continue
                
            for ep_idx, ep_data in enumerate(performance['all_episodes_data']):
                network_matrix = ep_data['network_matrix']
                learning_rates = ep_data['learning_rates']
                
                # Get fastest and slowest agent IDs for this episode
                slowest_agent_id = performance['slowest_positions'][ep_idx]
                fastest_agent_id = max(learning_rates.keys(), key=lambda k: learning_rates[k])
                
                # Calculate neighbor count for each agent
                for agent_id in range(num_agents):
                    neighbor_count = np.sum(network_matrix[agent_id])  # Number of neighbors
                    agent_lr = learning_rates.get(agent_id, 0.0)
                    
                    # Determine agent category
                    if agent_id == slowest_agent_id:
                        agent_category = 'slowest'
                    elif agent_id == fastest_agent_id:
                        agent_category = 'fastest'
                    else:
                        agent_category = 'other'
                    
                    all_agent_data.append({
                        'network_type': network_type,
                        'num_agents': num_agents,
                        'agent_id': agent_id,
                        'episode': ep_idx,
                        'neighbor_count': neighbor_count,
                        'learning_rate': agent_lr,
                        'agent_category': agent_category,
                        'final_incorrect_prob': ep_data['trajectories'][agent_id][-1]
                    })
    
    if not all_agent_data:
        print("No neighbor count data available for analysis")
        return
    
    # Create a single plot combining 4 and 8 agents with star and random networks
    valid_agent_counts = [n for n in args.agent_counts if n > 2]  # Exclude autarky and 2 agents
    
    if not valid_agent_counts:
        print("No valid agent counts (>2) found for neighbor count analysis. Skipping...")
        return
    
    # Create 2x2 subplot: 2 network types × 2 agent counts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for network_type in available_networks:
        for num_agents in valid_agent_counts[:2]:  # Take first 2 valid agent counts (typically 4 and 8)
            if plot_idx >= 4:
                break
                
            ax = axes[plot_idx]
            
            # Filter data for this specific size and network type
            type_data = [d for d in all_agent_data if d['network_type'] == network_type and d['num_agents'] == num_agents]
            
            if not type_data:
                ax.set_visible(False)
                plot_idx += 1
                continue
            
            slowest_data = [d for d in type_data if d['agent_category'] == 'slowest']
            fastest_data = [d for d in type_data if d['agent_category'] == 'fastest']
            other_data = [d for d in type_data if d['agent_category'] == 'other']
            
            # Plot data points
            if slowest_data:
                slowest_neighbors = [d['neighbor_count'] for d in slowest_data]
                slowest_lrs = [d['learning_rate'] for d in slowest_data]
                ax.scatter(slowest_neighbors, slowest_lrs, c='red', alpha=0.8, s=80, label='Slowest Agents', edgecolors='darkred', linewidth=1)
            
            if fastest_data:
                fastest_neighbors = [d['neighbor_count'] for d in fastest_data]
                fastest_lrs = [d['learning_rate'] for d in fastest_data]
                ax.scatter(fastest_neighbors, fastest_lrs, c='green', alpha=0.8, s=80, label='Fastest Agents', edgecolors='darkgreen', linewidth=1)
            
            if other_data:
                other_neighbors = [d['neighbor_count'] for d in other_data]
                other_lrs = [d['learning_rate'] for d in other_data]
                ax.scatter(other_neighbors, other_lrs, c='lightblue', alpha=0.6, s=50, label='Other Agents', edgecolors='blue', linewidth=0.5)
            
            # Add regression line and p-value if we have enough data points
            if len(type_data) > 2:
                all_neighbors = [d['neighbor_count'] for d in type_data]
                all_lrs = [d['learning_rate'] for d in type_data]
                
                # Check if there's variation in neighbor counts (avoid regression on constant x)
                if len(set(all_neighbors)) > 1:
                    # Perform linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(all_neighbors, all_lrs)
                    
                    # Create regression line
                    x_range = np.linspace(min(all_neighbors), max(all_neighbors), 100)
                    y_pred = slope * x_range + intercept
                    
                    ax.plot(x_range, y_pred, 'k--', alpha=0.8, linewidth=2, 
                           label=f'p={p_value:.3f}')
            
            ax.set_xlabel('Number of Neighbors', fontweight='bold', fontsize=12)
            ax.set_ylabel('Learning Rate', fontweight='bold', fontsize=12)
            ax.set_title(f'{network_type.capitalize()} Network ({num_agents} Agents)', fontweight='bold', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.suptitle('Neighbor Count vs Learning Rate Analysis\n(Star & Random Networks)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / "neighbor_count_vs_learning_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved neighbor count vs learning performance plot to {plot_path}")
    plt.close()


def plot_temporal_signal_analysis(results, args):
    """Create plots showing how signal quality changes over time comparing slowest vs fastest agents."""
    print("Creating temporal signal analysis plots...")
    
    # Collect temporal signal data
    temporal_data = {}
    
    for network_type in args.network_types:
        temporal_data[network_type] = {}
        
        for num_agents in args.agent_counts:
            if num_agents == 1:  # Skip autarky
                continue
                
            performance = results[network_type][num_agents]
            if performance['slowest_mean'] is None:
                continue
                
            slowest_temporal = []
            fastest_temporal = []
            
            for ep_idx, ep_data in enumerate(performance['all_episodes_data']):
                if 'signals' in ep_data and len(ep_data['signals']) > 0:
                    episode_signals = ep_data['signals']
                    true_state = ep_data['true_state']
                    learning_rates = ep_data['learning_rates']
                    
                    # Get fastest and slowest agent IDs for this episode
                    slowest_agent_id = performance['slowest_positions'][ep_idx]
                    fastest_agent_id = max(learning_rates.keys(), key=lambda k: learning_rates[k])
                    
                    # Calculate temporal signal correctness for slowest and fastest agents
                    for agent_id in [slowest_agent_id, fastest_agent_id]:
                        agent_temporal_signals = []
                        for step_signals in episode_signals:
                            if len(step_signals) > agent_id:
                                agent_signal = step_signals[agent_id]
                                correctness = 1.0 if agent_signal == true_state else 0.0
                                agent_temporal_signals.append(correctness)
                        
                        if len(agent_temporal_signals) > 0:
                            if agent_id == slowest_agent_id:
                                slowest_temporal.append(agent_temporal_signals)
                            elif agent_id == fastest_agent_id:
                                fastest_temporal.append(agent_temporal_signals)
            
            temporal_data[network_type][num_agents] = {
                'slowest': slowest_temporal,
                'fastest': fastest_temporal
            }
    
    # Create temporal analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for network_type in args.network_types:
        if plot_idx >= 4:
            break
            
        ax = axes[plot_idx]
        
        # Combine data across all agent counts for this network type
        all_slowest = []
        all_fastest = []
        
        for num_agents in args.agent_counts:
            if num_agents in temporal_data[network_type]:
                all_slowest.extend(temporal_data[network_type][num_agents]['slowest'])
                all_fastest.extend(temporal_data[network_type][num_agents]['fastest'])
        
        if all_slowest and all_fastest:
            # Calculate mean temporal patterns
            max_len = max(max(len(s) for s in all_slowest), max(len(f) for f in all_fastest))
            
            # Pad and average slowest agents
            padded_slowest = []
            for signals in all_slowest:
                if len(signals) < max_len:
                    padded = signals + [signals[-1]] * (max_len - len(signals))
                else:
                    padded = signals[:max_len]
                padded_slowest.append(padded)
            
            # Pad and average fastest agents
            padded_fastest = []
            for signals in all_fastest:
                if len(signals) < max_len:
                    padded = signals + [signals[-1]] * (max_len - len(signals))
                else:
                    padded = signals[:max_len]
                padded_fastest.append(padded)
            
            slowest_array = np.array(padded_slowest)
            fastest_array = np.array(padded_fastest)
            
            slowest_mean = np.mean(slowest_array, axis=0)
            fastest_mean = np.mean(fastest_array, axis=0)
            
            time_steps = range(len(slowest_mean))
            
            # Plot with confidence intervals
            ax.plot(time_steps, slowest_mean, label='Slowest Agents', color='red', linewidth=3.0)
            ax.plot(time_steps, fastest_mean, label='Fastest Agents', color='green', linewidth=3.0)
            
            if len(padded_slowest) > 1:
                slowest_sem = stats.sem(slowest_array, axis=0)
                slowest_ci = stats.t.interval(0.95, len(padded_slowest)-1, loc=slowest_mean, scale=slowest_sem)
                ax.fill_between(time_steps, slowest_ci[0], slowest_ci[1], color='red', alpha=0.2)
            
            if len(padded_fastest) > 1:
                fastest_sem = stats.sem(fastest_array, axis=0)
                fastest_ci = stats.t.interval(0.95, len(padded_fastest)-1, loc=fastest_mean, scale=fastest_sem)
                ax.fill_between(time_steps, fastest_ci[0], fastest_ci[1], color='green', alpha=0.2)
        
        ax.set_xlabel('Time Steps', fontweight='bold', fontsize=12)
        ax.set_ylabel('Signal Correctness', fontweight='bold', fontsize=12)
        ax.set_title(f'{network_type.capitalize()} Network\nTemporal Signal Quality', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=SIGNAL_ACCURACY, color='black', linestyle='--', alpha=0.7, label=f'Expected ({SIGNAL_ACCURACY})')
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.suptitle('Temporal Signal Quality: Slowest vs Fastest Agents', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / "temporal_signal_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved temporal signal analysis to {plot_path}")
    plt.close()


def plot_attention_analysis(results, args):
    """Create plots showing which agents receive the most attention, similar to network position analysis."""
    print("Creating attention analysis plots...")
    
    # Check if we have any attention data at all
    has_attention_data = False
    for network_type in args.network_types:
        for num_agents in args.agent_counts:
            if num_agents == 1:  # Skip autarky
                continue
            performance = results.get(network_type, {}).get(num_agents, {})
            if performance.get('slowest_mean') is not None:
                for ep_data in performance.get('all_episodes_data', []):
                    if 'attention_weights' in ep_data and len(ep_data['attention_weights']) > 0:
                        has_attention_data = True
                        break
                if has_attention_data:
                    break
        if has_attention_data:
            break
    
    if not has_attention_data:
        print("⚠️  No attention weights data found in the results.")
        print("   This analysis requires GNN-enabled agents to capture attention weights.")
        print("   Make sure you're running experiments with --use-gnn flag enabled.")
        print("   Skipping attention analysis...")
        return
    
    # Collect attention data across all configurations
    attention_data = {}
    
    for network_type in args.network_types:
        attention_data[network_type] = {}
        
        for num_agents in args.agent_counts:
            if num_agents == 1:  # Skip autarky
                continue
                
            performance = results[network_type][num_agents]
            if performance['slowest_mean'] is None:
                continue
                
            # Extract attention data from all episodes
            all_episode_attention = []
            
            for ep_idx, ep_data in enumerate(performance['all_episodes_data']):
                if 'attention_weights' in ep_data and len(ep_data['attention_weights']) > 0:
                    # ep_data['attention_weights'] should be a list of attention matrices, one per time step
                    episode_attention = ep_data['attention_weights']
                    
                    # Calculate average attention received by each agent over time
                    agent_attention_received = np.zeros(num_agents)
                    
                    for step_attention in episode_attention:
                        if isinstance(step_attention, np.ndarray) and step_attention.shape == (num_agents, num_agents):
                            # Sum attention received by each agent (column-wise sum)
                            agent_attention_received += np.sum(step_attention, axis=0)
                    
                    # Normalize by number of time steps
                    if len(episode_attention) > 0:
                        agent_attention_received /= len(episode_attention)
                        all_episode_attention.append(agent_attention_received)
            
            if len(all_episode_attention) > 0:
                attention_data[network_type][num_agents] = all_episode_attention

    # Check if we have any valid attention data after processing
    total_configs_with_data = sum(len(configs) for configs in attention_data.values())
    if total_configs_with_data == 0:
        print("⚠️  No valid attention data found after processing.")
        print("   This could happen if attention weights are not properly captured or")
        print("   if the attention matrices don't have the expected shape.")
        print("   Skipping attention analysis...")
        return
    
    print(f"✅ Found attention data for {total_configs_with_data} configurations")
    
    # Only analyze star and random networks (similar to network position analysis)
    relevant_networks = ['star', 'random']
    available_networks = [nt for nt in relevant_networks if nt in args.network_types and nt in attention_data]
    
    if not available_networks:
        print("No relevant network types (star, random) found for attention analysis. Skipping...")
        return
    
    # Create network graphs showing attention frequencies (similar to network position plot)
    n_networks = len(available_networks)
    n_cols = min(2, n_networks)
    n_rows = (n_networks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    if n_networks == 1:
        axes = [axes]
    elif n_networks <= 2:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    for network_type in available_networks:
        ax = axes[plot_idx]
        
        # Find the largest network size for this network type to use as the representative
        agent_counts = sorted([ac for ac in attention_data.get(network_type, {}).keys() if ac > 1])
        if not agent_counts:
            plot_idx += 1
            continue
            
        # Use the largest network size for visualization
        num_agents = agent_counts[-1]
        
        if num_agents not in attention_data[network_type]:
            plot_idx += 1
            continue
            
        episode_attention_data = attention_data[network_type][num_agents]
        performance = results[network_type][num_agents]
        
        # Get network matrix from the first episode
        network_matrix = performance['all_episodes_data'][0]['network_matrix']
        
        # Calculate average attention received by each agent across all episodes
        avg_attention_received = np.mean(episode_attention_data, axis=0)
        
        # Create NetworkX graph
        G = nx.from_numpy_array(network_matrix)
        
        # Calculate node positions based on network type with proper scaling
        if network_type == 'complete':
            pos = nx.circular_layout(G, scale=0.8)
        elif network_type == 'ring':
            pos = nx.circular_layout(G, scale=0.8)
        elif network_type == 'star':
            pos = nx.spring_layout(G, k=1.5, iterations=100, scale=0.8)
        elif network_type == 'random':
            pos = nx.spring_layout(G, k=1.0, iterations=100, scale=0.8)
        else:
            pos = nx.spring_layout(G, scale=0.8)
        
        # Normalize attention values to [0, 1] for color mapping
        max_attention = np.max(avg_attention_received) if len(avg_attention_received) > 0 else 1
        min_attention = np.min(avg_attention_received) if len(avg_attention_received) > 0 else 0
        
        # Create node colors based on attention received (white to red gradient)
        node_colors = []
        for node in range(num_agents):
            attention = avg_attention_received[node]
            # Normalize to [0, 1]
            if max_attention > min_attention:
                intensity = (attention - min_attention) / (max_attention - min_attention)
            else:
                intensity = 0.5
            # Color gradient from white (low attention) to red (high attention)
            node_colors.append((1.0, 1.0 - intensity, 1.0 - intensity))  # RGB: white to red
        
        # Draw the network with larger nodes and black frames
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.7, width=3.5, edge_color='gray')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=2500, alpha=0.9, edgecolors='black', linewidths=2)
        
        # Add labels showing attention values with better formatting
        labels = {}
        for node in range(num_agents):
            attention = avg_attention_received[node]
            labels[node] = f'{attention:.2f}'
        
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=18, font_weight='bold', 
                               font_color='black')
        
        ax.set_title(f'{network_type.capitalize()} Network ({num_agents} agents)\nAverage Attention Received', 
                    fontweight='bold', fontsize=16)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        
        plot_idx += 1
    
    # Hide unused subplots
    if len(axes) > plot_idx:
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
    
    # Add a centered color legend (only if we have plots)
    if plot_idx > 0:
        from matplotlib.colors import LinearSegmentedColormap
        
        # Position the legend below the plots with more space
        legend_x = 0.35  # Center horizontally
        legend_y = 0.15  # Position below the plots
        legend_width = 0.3
        legend_height = 0.04
        
        # Create gradient rectangle using figure coordinates
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        
        # Add the legend to the figure (not to a specific axis)
        fig_legend_ax = fig.add_axes([legend_x, legend_y, legend_width, legend_height])
        fig_legend_ax.imshow(gradient, aspect='auto', cmap='Reds', alpha=0.8)
        fig_legend_ax.set_xlim(0, 100)
        fig_legend_ax.set_ylim(0, 1)
        fig_legend_ax.set_xticks([0, 100])
        fig_legend_ax.set_xticklabels(['Low', 'High'], fontsize=12, fontweight='bold')
        fig_legend_ax.set_yticks([])
        fig_legend_ax.set_xlabel('Attention Received', fontsize=13, fontweight='bold', labelpad=10)
        
        # Remove the box around the legend
        for spine in fig_legend_ax.spines.values():
            spine.set_visible(False)
    
    plt.suptitle('Attention Analysis: Average Attention Received by Each Agent (Star & Random Networks)', 
                 fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    plot_path = RESULTS_DIR / "attention_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved attention analysis to {plot_path}")
    plt.close()


def plot_attention_vs_learning_performance(results, args):
    """Create plots showing attention received vs learning performance with fastest/slowest distinction."""
    print("Creating attention received vs learning performance plots...")
    
    # Check if we have any attention data at all
    has_attention_data = False
    for network_type in args.network_types:
        for num_agents in args.agent_counts:
            if num_agents == 1:  # Skip autarky
                continue
            performance = results.get(network_type, {}).get(num_agents, {})
            if performance.get('slowest_mean') is not None:
                for ep_data in performance.get('all_episodes_data', []):
                    if 'attention_weights' in ep_data and len(ep_data['attention_weights']) > 0:
                        has_attention_data = True
                        break
                if has_attention_data:
                    break
        if has_attention_data:
            break
    
    if not has_attention_data:
        print("⚠️  No attention weights data found in the results.")
        print("   This analysis requires GNN-enabled agents to capture attention weights.")
        print("   Make sure you're running experiments with --use-gnn flag enabled.")
        print("   Skipping attention vs learning performance analysis...")
        return
    
    # Collect data for analysis
    all_agent_data = []
    
    for network_type in args.network_types:
        for num_agents in args.agent_counts:
            if num_agents == 1:  # Skip autarky
                continue
                
            performance = results[network_type][num_agents]
            if performance['slowest_mean'] is None:
                continue
                
            for ep_idx, ep_data in enumerate(performance['all_episodes_data']):
                if 'attention_weights' in ep_data and len(ep_data['attention_weights']) > 0:
                    episode_attention = ep_data['attention_weights']
                    learning_rates = ep_data['learning_rates']
                    
                    # Get fastest and slowest agent IDs for this episode
                    slowest_agent_id = performance['slowest_positions'][ep_idx]
                    fastest_agent_id = max(learning_rates.keys(), key=lambda k: learning_rates[k])
                    
                    # Calculate average attention received by each agent
                    agent_attention_received = np.zeros(num_agents)
                    
                    for step_attention in episode_attention:
                        if isinstance(step_attention, np.ndarray) and step_attention.shape == (num_agents, num_agents):
                            # Sum attention received by each agent (column-wise sum)
                            agent_attention_received += np.sum(step_attention, axis=0)
                    
                    # Normalize by number of time steps
                    if len(episode_attention) > 0:
                        agent_attention_received /= len(episode_attention)
                        
                        # Store data for each agent
                        for agent_id in range(num_agents):
                            agent_lr = learning_rates.get(agent_id, 0.0)
                            attention_received = agent_attention_received[agent_id]
                            
                            # Determine agent category
                            if agent_id == slowest_agent_id:
                                agent_category = 'slowest'
                            elif agent_id == fastest_agent_id:
                                agent_category = 'fastest'
                            else:
                                agent_category = 'other'
                            
                            all_agent_data.append({
                                'network_type': network_type,
                                'num_agents': num_agents,
                                'agent_id': agent_id,
                                'episode': ep_idx,
                                'attention_received': attention_received,
                                'learning_rate': agent_lr,
                                'agent_category': agent_category,
                                'final_incorrect_prob': ep_data['trajectories'][agent_id][-1]
                            })
    
    if not all_agent_data:
        print("No attention vs learning performance data available for analysis")
        return
    
    # Create two plots: Size comparison and Type comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Attention vs Learning Rate - Size Comparison
    ax1 = axes[0]
    
    # Use the first network type for size comparison
    primary_network = args.network_types[0]
    size_data = [d for d in all_agent_data if d['network_type'] == primary_network]
    
    slowest_data = [d for d in size_data if d['agent_category'] == 'slowest']
    fastest_data = [d for d in size_data if d['agent_category'] == 'fastest']
    other_data = [d for d in size_data if d['agent_category'] == 'other']
    
    if slowest_data:
        slowest_attention = [d['attention_received'] for d in slowest_data]
        slowest_lrs = [d['learning_rate'] for d in slowest_data]
        ax1.scatter(slowest_attention, slowest_lrs, c='red', alpha=0.8, s=80, label='Slowest Agents', edgecolors='darkred', linewidth=1)
    
    if fastest_data:
        fastest_attention = [d['attention_received'] for d in fastest_data]
        fastest_lrs = [d['learning_rate'] for d in fastest_data]
        ax1.scatter(fastest_attention, fastest_lrs, c='green', alpha=0.8, s=80, label='Fastest Agents', edgecolors='darkgreen', linewidth=1)
    
    if other_data:
        other_attention = [d['attention_received'] for d in other_data]
        other_lrs = [d['learning_rate'] for d in other_data]
        ax1.scatter(other_attention, other_lrs, c='lightblue', alpha=0.6, s=50, label='Other Agents', edgecolors='blue', linewidth=0.5)
    
    # Add regression line and p-value for size comparison
    if len(size_data) > 2:
        all_attention = [d['attention_received'] for d in size_data]
        all_lrs = [d['learning_rate'] for d in size_data]
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_attention, all_lrs)
        
        # Create regression line
        x_range = np.linspace(min(all_attention), max(all_attention), 100)
        y_pred = slope * x_range + intercept
        
        ax1.plot(x_range, y_pred, 'k--', alpha=0.8, linewidth=2, 
                label=f'Regression (p={p_value:.3f})')
    
    ax1.set_xlabel('Average Attention Received', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Learning Rate', fontweight='bold', fontsize=12)
    ax1.set_title(f'Attention Received vs Learning Rate\n({primary_network.capitalize()} Network - Size Comparison)', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Attention vs Learning Rate - Type Comparison
    ax2 = axes[1]
    
    # Use a fixed agent count for type comparison
    fixed_agent_count = args.agent_counts[-1]
    type_data = [d for d in all_agent_data if d['num_agents'] == fixed_agent_count]
    
    slowest_data = [d for d in type_data if d['agent_category'] == 'slowest']
    fastest_data = [d for d in type_data if d['agent_category'] == 'fastest']
    other_data = [d for d in type_data if d['agent_category'] == 'other']
    
    if slowest_data:
        slowest_attention = [d['attention_received'] for d in slowest_data]
        slowest_lrs = [d['learning_rate'] for d in slowest_data]
        ax2.scatter(slowest_attention, slowest_lrs, c='red', alpha=0.8, s=80, label='Slowest Agents', edgecolors='darkred', linewidth=1)
    
    if fastest_data:
        fastest_attention = [d['attention_received'] for d in fastest_data]
        fastest_lrs = [d['learning_rate'] for d in fastest_data]
        ax2.scatter(fastest_attention, fastest_lrs, c='green', alpha=0.8, s=80, label='Fastest Agents', edgecolors='darkgreen', linewidth=1)
    
    if other_data:
        other_attention = [d['attention_received'] for d in other_data]
        other_lrs = [d['learning_rate'] for d in other_data]
        ax2.scatter(other_attention, other_lrs, c='lightblue', alpha=0.6, s=50, label='Other Agents', edgecolors='blue', linewidth=0.5)
    
    # Add regression line and p-value for type comparison
    if len(type_data) > 2:
        all_attention = [d['attention_received'] for d in type_data]
        all_lrs = [d['learning_rate'] for d in type_data]
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_attention, all_lrs)
        
        # Create regression line
        x_range = np.linspace(min(all_attention), max(all_attention), 100)
        y_pred = slope * x_range + intercept
        
        ax2.plot(x_range, y_pred, 'k--', alpha=0.8, linewidth=2, 
                label=f'Regression (p={p_value:.3f})')
    
    ax2.set_xlabel('Average Attention Received', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontweight='bold', fontsize=12)
    ax2.set_title(f'Attention Received vs Learning Rate\n({fixed_agent_count} Agents - Network Type Comparison)', fontweight='bold', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Attention Received vs Learning Rate Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    plot_path = RESULTS_DIR / "attention_vs_learning_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved attention vs learning performance plot to {plot_path}")
    plt.close()


def main():
    """Main function for agent performance comparison."""
    parser = argparse.ArgumentParser(description="Compare Slowest vs Fastest Learning Agents with Confidence Intervals")
    parser.add_argument('--agent-counts', nargs='+', type=int, default=AGENT_COUNTS,
                       help='List of agent counts to test')
    parser.add_argument('--network-types', nargs='+', type=str, default=NETWORK_TYPES,
                       choices=['complete', 'ring', 'star', 'random'],
                       help='List of network types to test')
    parser.add_argument('--episodes', type=int, default=EPISODES, help='Number of episodes per configuration')
    parser.add_argument('--horizon', type=int, default=HORIZON, help='Steps per episode')
    parser.add_argument('--signal-accuracy', type=float, default=SIGNAL_ACCURACY, help='Signal accuracy')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps', 'cuda'],
                       help='Device to use')
    parser.add_argument('--use-gnn', action='store_true', default=True, help='Enable Graph Neural Network for agents')
    parser.add_argument('--no-gnn', action='store_true', help='Disable Graph Neural Network for agents')
    
    args = parser.parse_args()
    
    # Handle GNN configuration
    if args.no_gnn:
        args.use_gnn = False
    
    print("=== POLARIS Brandl Social Learning Agent Performance Sweep ===")
    print("This script analyzes individual agent learning performance across different")
    print("network topologies and sizes in the Brandl social learning environment.")
    print()
    print(f"📊 Configuration:")
    print(f"   • Agent counts: {args.agent_counts}")
    print(f"   • Network types: {args.network_types}")
    print(f"   • Episodes per config: {args.episodes}")
    print(f"   • Horizon (steps): {args.horizon}")
    print(f"   • Signal accuracy: {args.signal_accuracy}")
    print(f"   • Device: {args.device}")
    print(f"   • Using GNN: {args.use_gnn}")
    print()
    print("🔍 Analysis Focus:")
    print("   • Fastest vs slowest agent trajectories with 95% confidence intervals")
    print("   • Learning rate calculations (r) averaged across episodes")
    print("   • Network topology impact on learning disparities")
    print("   • Network position analysis for slowest agents")
    print("   • Private signal analysis over time")
    print()
    
    # Set matplotlib style
    plt.rcParams["font.family"] = "serif"
    
    results = {}
    
    # Run experiments for all configurations
    for network_type in args.network_types:
        print(f"Testing {network_type} networks...")
        results[network_type] = {}
        
        for num_agents in args.agent_counts:
            print(f"  {num_agents} agents ({args.episodes} episodes)...", end=' ')
            
            performance = run_agent_experiment(
                num_agents, network_type, args.episodes, args.horizon,
                args.signal_accuracy, args.seed, args.device, args.use_gnn
            )
            
            results[network_type][num_agents] = performance
            
            if performance['slowest_mean'] is not None:
                slowest_final = performance['slowest_mean'][-1]
                fastest_final = performance['fastest_mean'][-1]
                print(f"Slowest: {slowest_final:.3f}, Fastest: {fastest_final:.3f}")
            else:
                print("No data")
    
    print("\n=== 📈 Generating Agent Performance Visualizations ===")
    print("Creating comprehensive plots showing:")
    print("• Fastest vs slowest agent trajectories with 95% confidence intervals")
    print("• Learning rates (r) averaged across episodes")
    print("• Network position analysis for slowest agents")
    print("• Private signal analysis over time")
    print()
    
    # Plot 1: Fastest vs Slowest trajectories across Network Sizes (using first network type)
    primary_network = args.network_types[0]
    plt.figure(figsize=(15, 10))
    
    agent_counts = sorted(results[primary_network].keys())
    
    # Create subplots for different network sizes
    n_sizes = len(agent_counts)
    n_cols = min(2, n_sizes)  # Use 2 columns instead of 3
    n_rows = (n_sizes + n_cols - 1) // n_cols
    
    for i, num_agents in enumerate(agent_counts):
        plt.subplot(n_rows, n_cols, i + 1)
        
        performance = results[primary_network][num_agents]
        if performance['slowest_mean'] is not None:
            time_steps = range(len(performance['slowest_mean']))
            
            slowest_lr = performance['learning_rates'].get(performance['slowest_agent_id'], 0.0)
            fastest_lr = performance['learning_rates'].get(performance['fastest_agent_id'], 0.0)
            
            # Plot slowest agent with confidence interval (removed agent ID from label)
            plt.plot(time_steps, performance['slowest_mean'], 
                    label=f'Slowest (r={slowest_lr:.4f})', 
                    color='red', linewidth=2.5, alpha=0.9)
            plt.fill_between(time_steps, 
                           performance['slowest_ci'][0], 
                           performance['slowest_ci'][1],
                           color='red', alpha=0.2)
            
            # Plot fastest agent with confidence interval (removed agent ID from label)
            plt.plot(time_steps, performance['fastest_mean'], 
                    label=f'Fastest (r={fastest_lr:.4f})', 
                    color='green', linewidth=2.5, alpha=0.9)
            plt.fill_between(time_steps, 
                           performance['fastest_ci'][0], 
                           performance['fastest_ci'][1],
                           color='green', alpha=0.2)
            
            plt.xlabel("Time Steps", fontsize=12, fontweight='bold')
            plt.ylabel("Incorrect Action Probability", fontsize=12, fontweight='bold')
            if num_agents == 1:
                plt.title(f"Autarky (Average over {performance['num_episodes']} episodes)", fontsize=14, fontweight='bold')
            else:
                plt.title(f"{num_agents} Agents (Average over {performance['num_episodes']} episodes)", fontsize=14, fontweight='bold')
            plt.legend(fontsize=12, loc='upper right')
            plt.grid(True, alpha=0.3)
    
    plt.suptitle(f"Fastest vs Slowest Learning Evolution Across Network Sizes\n({primary_network.capitalize()} Network, 95% Confidence Intervals)", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / "fastest_slowest_network_sizes_evolution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved network sizes evolution plot to {plot_path}")
    plt.close()
    
    # Plot 2: Fastest vs Slowest trajectories across Network Types (using middle agent count)
    middle_idx = len(args.agent_counts) // 2
    fixed_agent_count = args.agent_counts[-1]
    
    plt.figure(figsize=(15, 10))
    
    network_names = list(results.keys())
    n_networks = len(network_names)
    n_cols = min(2, n_networks)
    n_rows = (n_networks + n_cols - 1) // n_cols
    
    for i, network_type in enumerate(network_names):
        plt.subplot(n_rows, n_cols, i + 1)
        
        performance = results[network_type][fixed_agent_count]
        if performance['slowest_mean'] is not None:
            time_steps = range(len(performance['slowest_mean']))
            
            slowest_lr = performance['learning_rates'].get(performance['slowest_agent_id'], 0.0)
            fastest_lr = performance['learning_rates'].get(performance['fastest_agent_id'], 0.0)
            
            # Plot slowest agent with confidence interval (removed agent ID from label)
            plt.plot(time_steps, performance['slowest_mean'], 
                    label=f'Slowest (r={slowest_lr:.4f})', 
                    color='red', linewidth=2.5, alpha=0.9)
            plt.fill_between(time_steps, 
                           performance['slowest_ci'][0], 
                           performance['slowest_ci'][1],
                           color='red', alpha=0.2)
            
            # Plot fastest agent with confidence interval (removed agent ID from label)
            plt.plot(time_steps, performance['fastest_mean'], 
                    label=f'Fastest (r={fastest_lr:.4f})', 
                    color='green', linewidth=2.5, alpha=0.9)
            plt.fill_between(time_steps, 
                           performance['fastest_ci'][0], 
                           performance['fastest_ci'][1],
                           color='green', alpha=0.2)
            
            plt.xlabel("Time Steps", fontsize=12, fontweight='bold')
            plt.ylabel("Incorrect Action Probability", fontsize=12, fontweight='bold')
            plt.title(f"{network_type.capitalize()} Network (Average over {performance['num_episodes']} episodes)", fontsize=14, fontweight='bold')
            plt.legend(fontsize=12, loc='upper right')
            plt.grid(True, alpha=0.3)
    
    plt.suptitle(f"Fastest vs Slowest Learning Evolution Across Network Types\n({fixed_agent_count} Agents, 95% Confidence Intervals)", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / "fastest_slowest_network_types_evolution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved network types evolution plot to {plot_path}")
    plt.close()
    
    # Plot 3: Network position analysis
    plot_network_positions(results, args)
    
    # Plot 4: Signal analysis
    plot_signal_analysis(results, args)
    
    # Plot 5: Signal Quality vs Learning Performance
    plot_signal_quality_vs_learning_performance(results, args)
    
    # Plot 6: Neighbor Count vs Learning Performance
    plot_neighbor_count_vs_learning_performance(results, args)
    
    # Plot 7: Temporal Signal analysis
    plot_temporal_signal_analysis(results, args)
    
    # Plot 8: Attention analysis
    plot_attention_analysis(results, args)
    
    # Plot 9: Attention vs Learning Performance
    plot_attention_vs_learning_performance(results, args)
    
    # Save results (convert numpy arrays to lists for JSON serialization)
    import json
    
    def convert_for_json(obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_for_json(item) for item in obj)
        else:
            return obj
    
    serializable_results = convert_for_json(results)
    
    summary_file = RESULTS_DIR / "agent_performance_results.json"
    with open(summary_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Saved results to {summary_file}")
    
    # Print summary
    print("\n=== Agent Performance Summary ===")
    for network_type in results.keys():
        print(f"\n{network_type.capitalize()} Networks:")
        for num_agents in sorted(results[network_type].keys()):
            perf = results[network_type][num_agents]
            if perf['slowest_mean'] is not None:
                slowest_final = perf['slowest_mean'][-1]
                fastest_final = perf['fastest_mean'][-1]
                gap = slowest_final - fastest_final
                print(f"  {num_agents:2d} agents: Gap={gap:.3f} (Slowest={slowest_final:.3f}, Fastest={fastest_final:.3f}) [n={perf['num_episodes']}]")
            else:
                print(f"  {num_agents:2d} agents: No data available")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\n🎯 Generated Outputs:")
    print(f"   📊 fastest_slowest_network_sizes_evolution.png - Fastest/slowest trajectories with CIs across network sizes")
    print(f"   📊 fastest_slowest_network_types_evolution.png - Fastest/slowest trajectories with CIs across network types")
    print(f"   📊 slowest_agent_network_positions.png - Network position frequency analysis for slowest agents")
    print(f"   📊 average_private_signals.png - Average private signals received over time")
    print(f"   📊 signal_quality_vs_learning_performance.png - Signal quality vs learning rate analysis")
    print(f"   📊 neighbor_count_vs_learning_performance.png - Neighbor count vs learning rate analysis (star & random networks, 4 & 8 agents)")
    print(f"   📊 temporal_signal_analysis.png - Temporal signal quality comparison between slowest and fastest agents")
    print(f"   📊 attention_analysis.png - Attention analysis showing which agents receive the most attention")
    print(f"   📊 attention_vs_learning_performance.png - Attention received vs learning rate analysis")
    print(f"   📄 agent_performance_results.json - Complete numerical results with learning rates and CIs")
    print()
    print("🔬 Key Insights Available:")
    print("   • Statistical significance of learning rate differences")
    print("   • Confidence intervals showing variability across episodes")
    print("   • Network topology effects on fastest vs slowest learners")
    print("   • Network position analysis showing which positions tend to be slowest")
    print("   • Signal quality analysis over time and across configurations")
    print("   • Robust performance comparisons with proper error estimation")
    print()
    print("=== ✅ Brandl agent performance analysis completed successfully! ===")
    print("📖 See README.md for detailed documentation and interpretation guide.")


if __name__ == "__main__":
    main() 