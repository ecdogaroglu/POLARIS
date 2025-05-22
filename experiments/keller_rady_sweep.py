#!/usr/bin/env python3
"""
POLARIS Strategic Experimentation Sweep Script

Runs the Keller-Rady experiment for different numbers of agents and plots the average total allocation over time for each configuration.
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from polaris.environments.strategic import StrategicExperimentationEnvironment
from polaris.utils.args import parse_args
from polaris.simulation import run_agents

# --- CONFIGURABLE PARAMETERS ---
AGENT_COUNTS = [2, 3, 4, 5, 6, 7, 8]
EPISODES = 1  # Number of episodes to average over
HORIZON = 400
RESULTS_DIR = Path("results/strategic_experimentation/sweep_allocations")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment_for_agents(num_agents, episodes, horizon, seed=0):
    """Run the Keller-Rady experiment for a given number of agents and return allocation data for all episodes."""
    args = parse_args()  # Use default, no arguments
    args.environment_type = 'strategic_experimentation'
    args.exp_name = 'strategic_experimentation'
    args.safe_payoff = 0.5
    args.drift_rates_list = [0, 1]
    args.jump_rates_list = [0, 0.1]
    args.jump_sizes_list = [1.0, 1.0]
    args.num_agents = num_agents
    args.num_episodes = episodes
    args.horizon = horizon
    args.eval_only = False
    args.save_model = False
    args.use_gnn = True
    args.use_si = False
    args.si_importance = 10
    args.visualize_si = False
    args.si_exclude_final_layers = False
    args.latex_style = False
    args.discount_factor = 0.0
    args.continuous_actions = True
    args.plot_allocations = False
    args.plot_internal_states = False
    args.seed = seed
    args.batch_size = 8
    args.buffer_capacity = 50

    # Add any missing required args with defaults
    if not hasattr(args, 'num_states'):
        args.num_states = 2
    if not hasattr(args, 'network_type'):
        args.network_type = 'complete'
    if not hasattr(args, 'network_density'):
        args.network_density = 1.0
    if not hasattr(args, 'diffusion_sigma'):
        args.diffusion_sigma = 0.0
    if not hasattr(args, 'background_informativeness'):
        args.background_informativeness = 0.0
    if not hasattr(args, 'time_step'):
        args.time_step = 1.0
    if not hasattr(args, 'output_dir'):
        args.output_dir = str(RESULTS_DIR)
    # Create environment
    env = StrategicExperimentationEnvironment(
        num_agents=args.num_agents,
        num_states=args.num_states,
        network_type=args.network_type,
        network_params={'density': args.network_density} if args.network_type == 'random' else None,
        horizon=args.horizon,
        seed=args.seed,
        safe_payoff=args.safe_payoff,
        drift_rates=args.drift_rates_list,
        diffusion_sigma=args.diffusion_sigma,
        jump_rates=args.jump_rates_list,
        jump_sizes=args.jump_sizes_list,
        background_informativeness=args.background_informativeness,
        time_step=args.time_step
    )
    # Run agents and collect metrics
    episodic_metrics, _ = run_agents(env, args, training=True, model_path=None)

    # Collect allocations for all episodes
    episode_allocations = []
    for ep in episodic_metrics["episodes"]:
        allocations = ep['allocations']
        if allocations:
            alloc_arr = np.array([allocations[aid] for aid in sorted(allocations.keys(), key=int)])
        else:
            alloc_arr = np.zeros((num_agents, horizon))
        episode_allocations.append(alloc_arr)  # shape: (num_agents, time)
    # Shape: (episodes, num_agents, time)
    return np.stack(episode_allocations, axis=0)


def main():
    all_results = {}
    all_cis = {}
    for num_agents in AGENT_COUNTS:
        print(f"\n=== Running for {num_agents} agents ===")
        alloc_arrs = run_experiment_for_agents(num_agents, EPISODES, HORIZON, seed=0)  # shape: (episodes, num_agents, time)
        print(f"    alloc_arrs shape: {alloc_arrs.shape}, sample: {alloc_arrs[0, :, :5]}")
        # For each episode, average over agents, then compute cumulative sum over time
        # Shape: (episodes, time)
        mean_over_agents = alloc_arrs.mean(axis=1)
        cumulative_alloc = np.cumsum(mean_over_agents, axis=1)
        # Compute mean and 95% CI across episodes at each time step
        mean_cum = cumulative_alloc.mean(axis=0)  # shape: (time,)
        sem = cumulative_alloc.std(axis=0, ddof=1) / np.sqrt(EPISODES)
        ci95 = 1.96 * sem
        all_results[num_agents] = mean_cum
        all_cis[num_agents] = ci95
    # --- Plotting: cumulative time series with 95% CI ---
    plt.figure(figsize=(5, 3))
    for num_agents in AGENT_COUNTS:
        mean_cum = all_results[num_agents]
        ci = all_cis[num_agents]
        plt.plot(mean_cum, label=f"{num_agents} agents")
        plt.fill_between(np.arange(len(mean_cum)), mean_cum - ci, mean_cum + ci, alpha=0.2)
    plt.xlabel("Time Steps")
    plt.ylabel("Average Cumulative Allocation")
    plt.title("Average Cumulative Allocation per Agent")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = RESULTS_DIR / "average_cumulative_allocation_per_agent_over_time.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved cumulative time series plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    main() 