#!/usr/bin/env python3
"""
POLARIS Strategic Experimentation Sweep Script (Refactored)

Runs the Keller-Rady experiment for different numbers of agents and plots the average total 
allocation over time for each configuration.

This is a wrapper for the refactored POLARIS implementation.
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from polaris.config.experiment_config import (
    ExperimentConfig, AgentConfig, TrainingConfig, StrategicExpConfig
)
from polaris.environments import StrategicExperimentationEnvironment
from polaris.training.simulation import run_experiment
from polaris.utils.device import get_best_device

# --- CONFIGURABLE PARAMETERS ---
AGENT_COUNTS = [2, 3, 4, 5, 6, 7, 8]
EPISODES = 1  # Number of episodes to average over
HORIZON = 100
RESULTS_DIR = Path("results/strategic_experimentation/sweep_allocations")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment_for_agents(num_agents, episodes, horizon, seed=0, device='cpu', continuous_actions=True):
    """Run the Keller-Rady experiment for a given number of agents and return allocation data for all episodes."""
    
    # Create configuration
    config = create_sweep_config(num_agents, episodes, horizon, seed, device, continuous_actions)
    
    # Create environment
    env = StrategicExperimentationEnvironment(
        num_agents=config.environment.num_agents,
        num_states=config.environment.num_states,
        network_type=config.environment.network_type,
        horizon=config.training.horizon,
        seed=config.environment.seed,
        safe_payoff=config.environment.safe_payoff,
        drift_rates=config.environment.drift_rates,
        jump_rates=config.environment.jump_rates,
        jump_sizes=config.environment.jump_sizes,
        diffusion_sigma=config.environment.diffusion_sigma,
        background_informativeness=config.environment.background_informativeness,
        time_step=config.environment.time_step
    )
    
    # Run experiment and collect metrics
    episodic_metrics, processed_metrics = run_experiment(env, config)
    
    # Extract allocation data from metrics
    # The metrics structure needs to be compatible with the original format
    episode_allocations = []
    
    if 'episodes' in episodic_metrics:
        for ep in episodic_metrics["episodes"]:
            allocations = ep.get('allocations', {})
            if allocations:
                # Convert allocations dict to array
                alloc_arr = np.array([allocations[aid] for aid in sorted(allocations.keys(), key=int)])
            else:
                alloc_arr = np.zeros((num_agents, horizon))
            episode_allocations.append(alloc_arr)  # shape: (num_agents, time)
    else:
        # Fallback: try to get allocations from processed_metrics
        allocations = processed_metrics.get('allocations', {})
        if allocations:
            # Convert single episode allocations
            alloc_arr = np.array([allocations[aid] for aid in sorted(allocations.keys(), key=int)])
            episode_allocations.append(alloc_arr)
        else:
            # No allocation data available
            alloc_arr = np.zeros((num_agents, horizon))
            episode_allocations.append(alloc_arr)
    
    # Shape: (episodes, num_agents, time)
    return np.stack(episode_allocations, axis=0)


def create_sweep_config(num_agents, episodes, horizon, seed, device='cpu', continuous_actions=True) -> ExperimentConfig:
    """Create experiment configuration for the sweep."""
    # Agent configuration
    agent_config = AgentConfig(
        learning_rate=1e-3,
        discount_factor=0.0,  # Use average reward for strategic experimentation
        use_si=False,
        si_importance=10,
        si_damping=0.1,
        si_exclude_final_layers=False
    )
    
    # Training configuration
    training_config = TrainingConfig(
        batch_size=8,
        buffer_capacity=50,
        num_episodes=episodes,
        horizon=horizon
    )
    
    # Strategic experimentation environment configuration
    env_config = StrategicExpConfig(
        environment_type='strategic_experimentation',
        num_agents=num_agents,
        seed=seed,
        safe_payoff=0.5,
        drift_rates=[0, 1],  # first values for bad state, second for good state
        jump_rates=[0, 0.1],
        jump_sizes=[1.0, 1.0],
        diffusion_sigma=0.0,
        background_informativeness=0.0,
        time_step=1.0,
        continuous_actions=continuous_actions
    )
    
    # Experiment name
    exp_name = f"strategic_experimentation_sweep_agents_{num_agents}"
    
    # Create complete configuration
    config = ExperimentConfig(
        agent=agent_config,
        training=training_config,
        environment=env_config,
        device=device,  # Use specified device instead of auto-detection
        output_dir=str(RESULTS_DIR),
        exp_name=exp_name,
        save_model=False,  # Don't save models for sweep
        load_model=None,
        eval_only=False,
        plot_internal_states=False,
        plot_allocations=False,  # We'll generate custom plots
        latex_style=False,
        use_tex=False
    )
    
    return config


def main():
    """Main function to run the sweep experiment."""
    parser = argparse.ArgumentParser(description="Run Keller-Rady Strategic Experimentation Sweep")
    parser.add_argument('--agent-counts', nargs='+', type=int, default=AGENT_COUNTS,
                       help='List of agent counts to sweep over')
    parser.add_argument('--episodes', type=int, default=EPISODES, help='Number of episodes per configuration')
    parser.add_argument('--horizon', type=int, default=HORIZON, help='Steps per episode')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps', 'cuda'], 
                       help='Force specific device (cpu is default to avoid device switching issues)')
    parser.add_argument('--continuous-actions', action='store_true', default=False,
                       help='Use continuous actions (default: discrete actions with action 1 prob as allocation)')
    
    args = parser.parse_args()
    
    # Set font family to match POLARIS plotting system
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "DejaVu Serif",
        "Times New Roman", 
        "Times",
        "serif",
    ]
    
    print("=== Keller-Rady Strategic Experimentation Sweep ===")
    print(f"Agent counts: {args.agent_counts}")
    print(f"Episodes per configuration: {args.episodes}")
    print(f"Horizon: {args.horizon}")
    print(f"Device: {args.device}")
    
    # Print action space type
    if args.continuous_actions:
        print(f"Using continuous action space for resource allocation")
    else:
        print(f"Using discrete action space (action 1 probability mapped to risky arm allocation)")
    
    all_results = {}
    all_cis = {}
    
    for num_agents in args.agent_counts:
        print(f"\n=== Running for {num_agents} agents ===")
        
        # Run experiment
        alloc_arrs = run_experiment_for_agents(num_agents, args.episodes, args.horizon, 
                                               seed=args.seed, device=args.device, 
                                               continuous_actions=args.continuous_actions)
        print(f"    alloc_arrs shape: {alloc_arrs.shape}, sample: {alloc_arrs[0, :, :5]}")
        
        # For each episode, average over agents, then compute cumulative sum over time
        # Shape: (episodes, time)
        mean_over_agents = alloc_arrs.mean(axis=1)
        cumulative_alloc = np.cumsum(mean_over_agents, axis=1)
        
        # Compute mean and 95% CI across episodes at each time step
        mean_cum = cumulative_alloc.mean(axis=0)  # shape: (time,)
        sem = cumulative_alloc.std(axis=0, ddof=1) / np.sqrt(args.episodes)
        ci95 = 1.96 * sem
        
        all_results[num_agents] = mean_cum
        all_cis[num_agents] = ci95
    
    # --- Plotting: cumulative time series with 95% CI ---
    plt.figure(figsize=(8, 5))  # Slightly larger figure
    
    for num_agents in args.agent_counts:
        mean_cum = all_results[num_agents]
        ci = all_cis[num_agents]
        
        plt.plot(mean_cum, label=f"{num_agents} agents", linewidth=2.5)
        plt.fill_between(np.arange(len(mean_cum)), mean_cum - ci, mean_cum + ci, alpha=0.2)
    
    # Apply improved font sizes consistent with POLARIS plotting system (using serif fonts)
    plt.xlabel("Time Steps", fontsize=16, fontweight='bold', labelpad=10, family='serif')
    plt.ylabel("Average Cumulative Allocation", fontsize=16, fontweight='bold', labelpad=10, family='serif')
    plt.title("Average Cumulative Allocation per Agent", fontsize=20, fontweight='bold', pad=25, family='serif')
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=12)
    
    # Use constrained_layout for better spacing
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / "average_cumulative_allocation_per_agent_over_time.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved cumulative time series plot to {plot_path}")
    plt.close()
    
    print("\n=== Sweep completed successfully! ===")


if __name__ == "__main__":
    main() 