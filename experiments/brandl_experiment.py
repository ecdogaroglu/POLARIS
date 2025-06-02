#!/usr/bin/env python3
"""
POLARIS Brandl Social Learning Experiment Script (Refactored)

This script runs experiments with POLARIS agents in a social learning environment
based on the Brandl framework, where agents learn without experimentation by
observing others' actions and receiving private signals.

This is a wrapper for the refactored POLARIS implementation.
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path

from polaris.config.experiment_config import (
    ExperimentConfig, AgentConfig, TrainingConfig, BrandlConfig
)
from polaris.environments import SocialLearningEnvironment
from polaris.training.simulation import run_experiment
from polaris.utils.device import get_best_device


def main():
    """Main function to run the Brandl experiment."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Brandl Social Learning Experiment")
    parser.add_argument('--agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--horizon', type=int, default=100, help='Steps per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--signal-accuracy', type=float, default=0.75, help='Signal accuracy')
    parser.add_argument('--network-type', type=str, default='complete', 
                       choices=['complete', 'ring', 'star', 'random'], help='Network type')
    parser.add_argument('--network-density', type=float, default=0.5, help='Network density for random networks')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--load', type=str, default=None, help='Path to load models')
    parser.add_argument('--use-gnn', action='store_true', default=True, help='Use GNN inference')
    parser.add_argument('--use-si', action='store_true', default=True, help='Use Synaptic Intelligence')
    parser.add_argument('--si-importance', type=float, default=1.0, help='SI importance weight')
    parser.add_argument('--plot-states', action='store_true', default=True, help='Plot internal states')
    parser.add_argument('--latex-style', action='store_true', default=True, help='Use LaTeX styling')
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'mps', 'cuda'], 
                       help='Force specific device (cpu is default to avoid device switching issues)')
    args = parser.parse_args()
    
    # Set initial random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create configuration
    config = create_brandl_config(args)
    
    print("\n=== Brandl Social Learning Experiment ===\n")
    
    # Create environment
    env = SocialLearningEnvironment(
        num_agents=config.environment.num_agents,
        num_states=config.environment.num_states,
        signal_accuracy=config.environment.signal_accuracy,
        network_type=config.environment.network_type,
        network_params={'density': config.environment.network_density} if config.environment.network_type == 'random' else None,
        horizon=config.training.horizon,
        seed=config.environment.seed
    )
    
    # Print experiment information
    print(f"Running Brandl social learning experiment")
    print(f"Network type: {config.environment.network_type}, Number of agents: {config.environment.num_agents}")
    print(f"Signal accuracy: {config.environment.signal_accuracy}")
    
    # Calculate theoretical bounds
    autarky_rate = env.get_autarky_rate()
    bound_rate = env.get_bound_rate()
    coordination_rate = env.get_coordination_rate()
    
    print(f"\nTheoretical learning rates:")
    print(f"- Autarky rate: {autarky_rate:.4f} (isolated agent)")
    print(f"- Bound rate: {bound_rate:.4f} (maximum possible for any agent)")
    print(f"- Coordination rate: {coordination_rate:.4f} (achievable with coordination)")
    
    # Print episode information if training
    if not config.eval_only and config.training.num_episodes > 1:
        print(f"\nTraining with {config.training.num_episodes} episodes, {config.training.horizon} steps per episode")
        print(f"True state will be randomly selected at the beginning of each episode with different seeds")
    
    # Print model architecture information
    if config.agent.use_gnn:
        print(f"Using Graph Neural Network")
    else:
        print("Using traditional encoder-decoder inference module")
    
    if config.agent.use_si:
        print(f"Using Synaptic Intelligence with importance {config.agent.si_importance} and damping {config.agent.si_damping}")
        if config.agent.si_exclude_final_layers:
            print("Excluding final layers from SI protection")
    
    # Run experiment
    print(f"\nRunning experiment...")
    print(f"Device: {config.device}")
    
    metrics, processed_metrics = run_experiment(env, config)
    
    print("\nExperiment completed!")
    print(f"Results saved to: {config.output_dir}/{config.exp_name}")


def create_brandl_config(args) -> ExperimentConfig:
    """Create experiment configuration for Brandl social learning."""
    # Agent configuration
    agent_config = AgentConfig(
        learning_rate=1e-3,
        discount_factor=0.99,  # Use discounted reward for social learning
        use_gnn=args.use_gnn,
        use_si=args.use_si,
        si_importance=args.si_importance,
        si_damping=0.1,
        si_exclude_final_layers=False
    )
    
    # Training configuration
    training_config = TrainingConfig(
        batch_size=128,
        buffer_capacity=1000,
        num_episodes=args.episodes,
        horizon=args.horizon
    )
    
    # Brandl environment configuration
    env_config = BrandlConfig(
        environment_type='brandl',
        num_agents=args.agents,
        seed=args.seed,
        signal_accuracy=args.signal_accuracy,
        network_type=args.network_type,
        network_density=args.network_density
    )
    
    # Experiment name
    exp_name = f"brandl_experiment_agents_{args.agents}"
    if args.use_gnn:
        exp_name += "_gnn"
    if args.use_si:
        exp_name += "_si"
    
    # Create complete configuration
    config = ExperimentConfig(
        agent=agent_config,
        training=training_config,
        environment=env_config,
        device=args.device,
        output_dir=args.output,
        exp_name=exp_name,
        save_model=not args.eval,
        load_model=args.load,
        eval_only=args.eval,
        plot_internal_states=args.plot_states,
        plot_allocations=False,  # Not relevant for Brandl
        latex_style=args.latex_style,
        use_tex=False  # Avoid LaTeX rendering issues with Unicode characters
    )
    
    return config


if __name__ == "__main__":
    main() 