#!/usr/bin/env python3
"""
POLARIS Strategic Experimentation Sweep Script (Updated)

Runs the Keller-Rady experiment for different numbers of agents and analyzes
highest vs lowest cumulative allocators across different network sizes, separated by true state.

Similar to brandl_sweep.py, this script provides detailed insights into:
- Individual agent cumulative allocation trajectories and performance
- Performance disparities between highest and lowest cumulative allocators
- Network size effects on cumulative allocation dynamics
- State-specific analysis (good vs bad state)
- Comparative analysis across different network sizes with confidence intervals

Key Features:
- Separates episodes by true state (good vs bad)
- Tracks cumulative allocations over time instead of incremental allocations
- Generates comprehensive visualizations with confidence intervals
- Highlights extreme cumulative allocators (lowest in red, highest in green) 
- Supports multiple episodes with proper agent resetting
- Produces state-specific plots with 4 subplots per network size
- Analyzes cumulative allocation patterns in good vs bad states

Usage:
    python experiments/keller_rady_sweep.py
    python experiments/keller_rady_sweep.py --agent-counts 2 4 6 8 --episodes 10

Outputs:
- highest_lowest_cumulative_allocators_good_state.png: Highest/lowest cumulative allocator trajectories with CIs for good state
- highest_lowest_cumulative_allocators_bad_state.png: Highest/lowest cumulative allocator trajectories with CIs for bad state
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
from types import SimpleNamespace

from polaris.config.experiment_config import (
    ExperimentConfig, AgentConfig, TrainingConfig, StrategicExpConfig
)
from polaris.environments import StrategicExperimentationEnvironment
from polaris.training.simulation import run_experiment
from polaris.utils.device import get_best_device

# --- FOCUSED PARAMETERS ---
AGENT_COUNTS = [1, 2, 4, 8]  # Network sizes to compare
EPISODES = 20  # Multiple episodes for confidence intervals
HORIZON = 50
RESULTS_DIR = Path("results/strategic_experimentation/sweep_allocations")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_agent_experiment(num_agents, episodes, horizon, seed=0, device='cpu', continuous_actions=False):
    """Run experiment for specified number of agents and return episode metrics with true states."""
    
    # Store results from all episodes, separated by true state
    good_state_episodes = []
    bad_state_episodes = []
    
    # Run each episode with fresh agents
    for episode_idx in range(episodes):
        episode_seed = seed + episode_idx
        print(f"    Episode {episode_idx + 1}/{episodes} (seed {episode_seed})...")
        
        # Configuration for agent performance tracking
        agent_config = AgentConfig(
            learning_rate=1e-3,
            discount_factor=0.0,  # Use average reward for strategic experimentation
            use_si=False,
            si_importance=10,
            si_damping=0.1,
            si_exclude_final_layers=False
        )
        
        training_config = TrainingConfig(
            batch_size=8,
            buffer_capacity=50,
            num_episodes=1,  # Single episode per run
            horizon=horizon
        )
        
        env_config = StrategicExpConfig(
            environment_type='strategic_experimentation',
            num_agents=num_agents,
            seed=episode_seed,  # Different seed for each episode
            safe_payoff=0.5,
            drift_rates=[0, 1],  # first values for bad state, second for good state
            jump_rates=[0, 0.1],
            jump_sizes=[1.0, 1.0],
            diffusion_sigma=0.0,
            background_informativeness=0.0,
            time_step=1.0,
            continuous_actions=continuous_actions
        )
        
        config = ExperimentConfig(
            agent=agent_config,
            training=training_config,
            environment=env_config,
            device=device,
            output_dir=str(RESULTS_DIR),
            exp_name=f"strategic_experimentation_sweep_agents_{num_agents}_ep{episode_idx}",
            save_model=False,
            load_model=None,
            eval_only=False,
            plot_internal_states=False,  # Disable internal plotting to avoid conflicts
            plot_allocations=False,      # Disable internal plotting to avoid conflicts
            latex_style=False,
            use_tex=False
        )
        
        # Create fresh environment for this episode
        env = StrategicExperimentationEnvironment(
            num_agents=num_agents,
            num_states=2,
            network_type='complete',  # Use complete network for Keller-Rady
            horizon=horizon,
            seed=episode_seed,
            safe_payoff=0.5,
            drift_rates=[0, 1],
            jump_rates=[0, 0.1],
            jump_sizes=[1.0, 1.0],
            diffusion_sigma=0.0,
            background_informativeness=0.0,
            time_step=1.0
        )
        
        # Run experiment with fresh agents
        episodic_metrics, processed_metrics = run_experiment(env, config)
        
        # Extract episode data and true state
        if 'episodes' in episodic_metrics and episodic_metrics['episodes']:
            episode_data = episodic_metrics['episodes'][0]
            
            # Get the true state for this episode (should be constant throughout)
            if 'true_states' in episode_data and episode_data['true_states']:
                true_state = episode_data['true_states'][0]  # Should be constant
                
                # Extract allocation data
                if 'allocations' in episode_data:
                    allocations_data = episode_data['allocations']
                    
                    # Convert allocations to time series format (cumulative)
                    episode_trajectories = []
                    for agent_id in range(num_agents):
                        if agent_id in allocations_data:
                            agent_allocations = allocations_data[agent_id]
                            # Compute cumulative sum of allocations
                            cumulative_allocations = np.cumsum(agent_allocations)
                            episode_trajectories.append(cumulative_allocations.tolist())
                        else:
                            # Fallback: create zero cumulative allocations
                            episode_trajectories.append([0.0] * horizon)
                    
                    # Store episode data based on true state
                    episode_info = {
                        'trajectories': episode_trajectories,
                        'true_state': true_state,
                        'episode_idx': episode_idx
                    }
                    
                    if true_state == 1:  # Good state
                        good_state_episodes.append(episode_info)
                    else:  # Bad state
                        bad_state_episodes.append(episode_info)
                    
                    print(f"      True state: {true_state} ({'Good' if true_state == 1 else 'Bad'})")
    
    return {
        'good_state_episodes': good_state_episodes,
        'bad_state_episodes': bad_state_episodes,
        'num_agents': num_agents,
        'total_episodes': episodes
    }


def calculate_trajectory_stats(trajectories_by_state):
    """
    Calculate statistics for highest and lowest allocators across episodes.
    
    Args:
        trajectories_by_state: List of episode data for a specific state
        
    Returns:
        Dictionary with highest/lowest allocator statistics
    """
    if not trajectories_by_state:
        return {
            'highest_mean': None,
            'highest_ci': (None, None),
            'lowest_mean': None,
            'lowest_ci': (None, None),
            'highest_agent_ids': [],
            'lowest_agent_ids': [],
            'num_episodes': 0
        }
    
    num_episodes = len(trajectories_by_state)
    num_agents = len(trajectories_by_state[0]['trajectories'])
    horizon = len(trajectories_by_state[0]['trajectories'][0])
    
    # Collect all agent trajectories across episodes
    all_highest_trajectories = []
    all_lowest_trajectories = []
    highest_agent_ids = []
    lowest_agent_ids = []
    
    for episode_data in trajectories_by_state:
        trajectories = episode_data['trajectories']
        
        # Calculate final allocations for each agent
        final_allocations = [traj[-1] for traj in trajectories]
        
        # Find highest and lowest allocators in this episode
        highest_idx = np.argmax(final_allocations)
        lowest_idx = np.argmin(final_allocations)
        
        all_highest_trajectories.append(trajectories[highest_idx])
        all_lowest_trajectories.append(trajectories[lowest_idx])
        highest_agent_ids.append(highest_idx)
        lowest_agent_ids.append(lowest_idx)
    
    # Calculate means and confidence intervals
    highest_trajectories = np.array(all_highest_trajectories)
    lowest_trajectories = np.array(all_lowest_trajectories)
    
    highest_mean = np.mean(highest_trajectories, axis=0)
    lowest_mean = np.mean(lowest_trajectories, axis=0)
    
    # Calculate 95% confidence intervals
    highest_sem = stats.sem(highest_trajectories, axis=0)
    lowest_sem = stats.sem(lowest_trajectories, axis=0)
    
    highest_ci = (
        highest_mean - 1.96 * highest_sem,
        highest_mean + 1.96 * highest_sem
    )
    lowest_ci = (
        lowest_mean - 1.96 * lowest_sem,
        lowest_mean + 1.96 * lowest_sem
    )
    
    return {
        'highest_mean': highest_mean,
        'highest_ci': highest_ci,
        'lowest_mean': lowest_mean,
        'lowest_ci': lowest_ci,
        'highest_agent_ids': highest_agent_ids,
        'lowest_agent_ids': lowest_agent_ids,
        'num_episodes': num_episodes
    }


def plot_state_specific_results(results, state_name, state_value, args):
    """
    Plot highest vs lowest allocators for a specific state with separate subplots.
    
    Args:
        results: Dictionary with results for each agent count
        state_name: Name of the state ('Good' or 'Bad')
        state_value: Value of the state (1 or 0)
        args: Command line arguments
    """
    plt.figure(figsize=(15, 6))
    
    agent_counts = sorted(results.keys())
    
    # Create two subplots: one for highest, one for lowest
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    
    # Subplot 1: Highest Allocators
    plt.subplot(1, 2, 1)
    
    for i, num_agents in enumerate(agent_counts):
        # Get results for this agent count and state
        agent_results = results[num_agents]
        state_key = 'good_state_episodes' if state_value == 1 else 'bad_state_episodes'
        state_episodes = agent_results[state_key]
        
        # Calculate statistics
        stats = calculate_trajectory_stats(state_episodes)
        
        if stats['highest_mean'] is not None:
            time_steps = range(len(stats['highest_mean']))
            color = colors[i % len(colors)]
            
            # Plot highest allocators with confidence interval
            plt.plot(time_steps, stats['highest_mean'], 
                    label=f'{num_agents} agents' if num_agents > 1 else 'Autarky', 
                    color=color, linewidth=2.5, alpha=0.9)
            plt.fill_between(time_steps, 
                           stats['highest_ci'][0], 
                           stats['highest_ci'][1],
                           color=color, alpha=0.2)
    
    plt.xlabel("Time Steps", fontsize=12, fontweight='bold')
    plt.ylabel("Cumulative Allocation to Risky Arm", fontsize=12, fontweight='bold')
    plt.title("Highest Cumulative Allocators", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Lowest Allocators
    plt.subplot(1, 2, 2)
    
    for i, num_agents in enumerate(agent_counts):
        # Get results for this agent count and state
        agent_results = results[num_agents]
        state_key = 'good_state_episodes' if state_value == 1 else 'bad_state_episodes'
        state_episodes = agent_results[state_key]
        
        # Calculate statistics
        stats = calculate_trajectory_stats(state_episodes)
        
        if stats['lowest_mean'] is not None:
            time_steps = range(len(stats['lowest_mean']))
            color = colors[i % len(colors)]
            
            # Plot lowest allocators with confidence interval
            plt.plot(time_steps, stats['lowest_mean'], 
                    label=f'{num_agents} agents' if num_agents > 1 else 'Autarky', 
                    color=color, linewidth=2.5, alpha=0.9)
            plt.fill_between(time_steps, 
                           stats['lowest_ci'][0], 
                           stats['lowest_ci'][1],
                           color=color, alpha=0.2)
    
    plt.xlabel("Time Steps", fontsize=12, fontweight='bold')
    plt.ylabel("Cumulative Allocation to Risky Arm", fontsize=12, fontweight='bold')
    plt.title("Lowest Cumulative Allocators", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f"Cumulative Allocators by Network Size ({state_name} State, 95% Confidence Intervals)", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / f"highest_lowest_cumulative_allocators_{state_name.lower()}_state.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved {state_name.lower()} state cumulative allocations plot to {plot_path}")
    plt.close()


def main():
    """Main function for strategic experimentation agent performance comparison."""
    parser = argparse.ArgumentParser(description="Compare Highest vs Lowest Cumulative Allocators in Strategic Experimentation")
    parser.add_argument('--agent-counts', nargs='+', type=int, default=AGENT_COUNTS,
                       help='List of agent counts to test')
    parser.add_argument('--episodes', type=int, default=EPISODES, help='Number of episodes per configuration')
    parser.add_argument('--horizon', type=int, default=HORIZON, help='Steps per episode')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps', 'cuda'],
                       help='Device to use')
    parser.add_argument('--continuous-actions', action='store_true', default=False,
                       help='Use continuous actions (default: discrete actions with action 1 prob as allocation)')
    
    args = parser.parse_args()
    
    print("=== POLARIS Strategic Experimentation Cumulative Allocation Sweep ===")
    print("This script analyzes individual agent cumulative allocation performance across different")
    print("network sizes in the strategic experimentation environment, separated by true state.")
    print()
    print(f"📊 Configuration:")
    print(f"   • Agent counts: {args.agent_counts}")
    print(f"   • Episodes per config: {args.episodes}")
    print(f"   • Horizon (steps): {args.horizon}")
    print(f"   • Device: {args.device}")
    print(f"   • Action type: {'Continuous' if args.continuous_actions else 'Discrete (action 1 prob as allocation)'}")
    print()
    print("🔍 Analysis Focus:")
    print("   • Highest vs lowest cumulative allocator trajectories with 95% confidence intervals")
    print("   • State-specific analysis (good state vs bad state)")
    print("   • Network size impact on cumulative allocation disparities")
    print()
    
    # Set matplotlib style
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "DejaVu Serif",
        "Times New Roman", 
        "Times",
        "serif",
    ]
    
    results = {}
    
    # Run experiments for all configurations
    for num_agents in args.agent_counts:
        print(f"Testing {num_agents} agents ({args.episodes} episodes)...", end=' ')
        
        agent_results = run_agent_experiment(
            num_agents, args.episodes, args.horizon, args.seed, args.device, args.continuous_actions
        )
        
        results[num_agents] = agent_results
        
        good_episodes = len(agent_results['good_state_episodes'])
        bad_episodes = len(agent_results['bad_state_episodes'])
        print(f"Good state: {good_episodes} episodes, Bad state: {bad_episodes} episodes")
    
    print("\n=== 📈 Generating State-Specific Cumulative Allocation Visualizations ===")
    print("Creating plots showing:")
    print("• Highest vs lowest cumulative allocator trajectories with 95% confidence intervals")
    print("• Separate analysis for good state and bad state episodes")
    print("• Two subplots per state: one for highest, one for lowest allocators")
    print("• Lines showing different network sizes in each subplot")
    print()
    
    # Plot results for good state (state = 1)
    plot_state_specific_results(results, "Good", 1, args)
    
    # Plot results for bad state (state = 0)
    plot_state_specific_results(results, "Bad", 0, args)
    
    print("\n=== Sweep completed successfully! ===")


if __name__ == "__main__":
    main() 