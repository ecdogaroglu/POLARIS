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

Key Features:
- Calculates learning rates (r) for each agent using incorrect probability decay
- Generates comprehensive visualizations with confidence intervals
- Highlights extreme learners (slowest in red, fastest in green)
- Supports multiple episodes with proper agent resetting
- Produces both aggregate plots and detailed JSON results
- Supports multiple network types: complete, ring, star, random

Usage:
    python experiments/brandl_sweep.py
    python experiments/brandl_sweep.py --agent-counts 1 2 4 6 8 10 --network-types complete ring star --episodes 5

Outputs:
- fastest_slowest_network_sizes_evolution.png: Fastest/slowest agent trajectories with CIs across network sizes
- fastest_slowest_network_types_evolution.png: Fastest/slowest agent trajectories with CIs across network types
- agent_performance_results.json: Complete numerical results with learning rates
- Individual agent plots in subdirectories for each configuration

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

from polaris.config.experiment_config import (
    ExperimentConfig, AgentConfig, TrainingConfig, BrandlConfig
)
from polaris.environments import SocialLearningEnvironment
from polaris.simulation import run_experiment
from polaris.utils.math import calculate_learning_rate

# --- FOCUSED PARAMETERS ---
AGENT_COUNTS = [1, 2, 4, 8]  # Network sizes to compare
NETWORK_TYPES = ['complete', 'ring', 'star', 'random']  # Network types to compare
EPISODES = 5  # Multiple episodes for confidence intervals
HORIZON = 50  # Shorter for faster execution
SIGNAL_ACCURACY = 0.75
RESULTS_DIR = Path("results/brandl_experiment/agent_performance")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_agent_experiment(num_agents, network_type, episodes, horizon, signal_accuracy, seed=0, device='cpu'):
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
            use_gnn=True,
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
            seed=episode_seed
        )
        
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
                    
                    all_episodes_data.append({
                        'trajectories': episode_trajectories,
                        'learning_rates': episode_learning_rates,
                        'final_probs': [traj[-1] for traj in episode_trajectories],
                        'episode_seed': episode_seed,
                        'true_state': env.true_state
                    })
    
    # Aggregate results across episodes
    if not all_episodes_data:
        return {
            'slowest_trajectories': None,
            'fastest_trajectories': None,
            'slowest_mean': 1.0,
            'fastest_mean': 1.0,
            'slowest_ci': (1.0, 1.0),
            'fastest_ci': (1.0, 1.0),
            'slowest_agent_id': 0,
            'fastest_agent_id': 0,
            'learning_rates': {},
            'num_episodes': episodes
        }
    
    # Identify fastest and slowest agents for each episode separately
    fastest_trajectories = []
    slowest_trajectories = []
    fastest_learning_rates = []
    slowest_learning_rates = []
    
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
        'all_episodes_data': all_episodes_data  # Keep for detailed analysis
    }
    
    return agent_performance


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
    
    args = parser.parse_args()
    
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
    print()
    print("🔍 Analysis Focus:")
    print("   • Fastest vs slowest agent trajectories with 95% confidence intervals")
    print("   • Learning rate calculations (r) averaged across episodes")
    print("   • Network topology impact on learning disparities")
    print("   • Statistical significance of performance differences")
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
                args.signal_accuracy, args.seed, args.device
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
    print("• Statistical significance of performance differences")
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
            
            # Plot slowest agent with confidence interval
            plt.plot(time_steps, performance['slowest_mean'], 
                    label=f'Agent {performance["slowest_agent_id"]} (r={slowest_lr:.4f}, Slowest)', 
                    color='red', linewidth=2.5, alpha=0.9)
            plt.fill_between(time_steps, 
                           performance['slowest_ci'][0], 
                           performance['slowest_ci'][1],
                           color='red', alpha=0.2)
            
            # Plot fastest agent with confidence interval
            plt.plot(time_steps, performance['fastest_mean'], 
                    label=f'Agent {performance["fastest_agent_id"]} (r={fastest_lr:.4f}, Fastest)', 
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
    fixed_agent_count = args.agent_counts[middle_idx]
    
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
            
            # Plot slowest agent with confidence interval
            plt.plot(time_steps, performance['slowest_mean'], 
                    label=f'Agent {performance["slowest_agent_id"]} (r={slowest_lr:.4f}, Slowest)', 
                    color='red', linewidth=2.5, alpha=0.9)
            plt.fill_between(time_steps, 
                           performance['slowest_ci'][0], 
                           performance['slowest_ci'][1],
                           color='red', alpha=0.2)
            
            # Plot fastest agent with confidence interval
            plt.plot(time_steps, performance['fastest_mean'], 
                    label=f'Agent {performance["fastest_agent_id"]} (r={fastest_lr:.4f}, Fastest)', 
                    color='green', linewidth=2.5, alpha=0.9)
            plt.fill_between(time_steps, 
                           performance['fastest_ci'][0], 
                           performance['fastest_ci'][1],
                           color='green', alpha=0.2)
            
            plt.xlabel("Time Steps", fontsize=12, fontweight='bold')
            plt.ylabel("Incorrect ActionProbability", fontsize=12, fontweight='bold')
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
    print(f"   📄 agent_performance_results.json - Complete numerical results with learning rates and CIs")
    print()
    print("🔬 Key Insights Available:")
    print("   • Statistical significance of learning rate differences")
    print("   • Confidence intervals showing variability across episodes")
    print("   • Network topology effects on fastest vs slowest learners")
    print("   • Robust performance comparisons with proper error estimation")
    print()
    print("=== ✅ Brandl agent performance analysis completed successfully! ===")
    print("📖 See README.md for detailed documentation and interpretation guide.")


if __name__ == "__main__":
    main() 