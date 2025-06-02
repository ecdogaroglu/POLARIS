#!/usr/bin/env python3
"""
Main experiment runner for POLARIS.
"""

import argparse
from pathlib import Path
import torch
import numpy as np

from polaris.config.experiment_config import (
    ExperimentConfig, AgentConfig, TrainingConfig, 
    BrandlConfig, StrategicExpConfig
)
from polaris.environments import SocialLearningEnvironment, StrategicExperimentationEnvironment
from polaris.training.simulation import run_experiment
from polaris.utils.device import get_best_device

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run POLARIS experiments")
    
    # Environment selection
    parser.add_argument('--env', type=str, default='brandl',
                       choices=['brandl', 'strategic'],
                       help='Environment type')
    
    # Common parameters
    parser.add_argument('--agents', type=int, default=2,
                       help='Number of agents')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes')
    parser.add_argument('--horizon', type=int, default=1000,
                       help='Steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--use-gnn', action='store_true',
                       help='Use GNN inference')
    parser.add_argument('--use-si', action='store_true',
                       help='Use Synaptic Intelligence')
    
    # Mode
    parser.add_argument('--eval', action='store_true',
                       help='Evaluation mode')
    parser.add_argument('--load', type=str, default=None,
                       help='Path to load models')
    
    # Output
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name')
    
    # Visualization
    parser.add_argument('--plot-states', action='store_true',
                       help='Plot internal states')
    parser.add_argument('--plot-allocations', action='store_true',
                       help='Plot allocations (strategic exp only)')
    
    return parser.parse_args()

def create_config(args) -> ExperimentConfig:
    """Create experiment configuration from arguments."""
    # Agent configuration
    agent_config = AgentConfig(
        learning_rate=args.lr,
        use_gnn=args.use_gnn,
        use_si=args.use_si
    )
    
    # Training configuration
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_episodes=args.episodes,
        horizon=args.horizon
    )
    
    # Environment configuration
    if args.env == 'brandl':
        env_config = BrandlConfig(
            environment_type='brandl',
            num_agents=args.agents,
            seed=args.seed
        )
    else:
        env_config = StrategicExpConfig(
            environment_type='strategic_experimentation',
            num_agents=args.agents,
            seed=args.seed,
            continuous_actions=True
        )
        
    # Experiment name
    if args.name is None:
        args.name = f"{args.env}_a{args.agents}_{'gnn' if args.use_gnn else 'mlp'}"
        if args.use_si:
            args.name += "_si"
            
    # Create config
    config = ExperimentConfig(
        agent=agent_config,
        training=training_config,
        environment=env_config,
        device=get_best_device(),
        output_dir=args.output,
        exp_name=args.name,
        eval_only=args.eval,
        load_model=args.load,
        plot_internal_states=args.plot_states,
        plot_allocations=args.plot_allocations
    )
    
    return config

def main():
    """Main experiment runner."""
    args = parse_args()
    config = create_config(args)
    
    # Set seeds
    torch.manual_seed(config.environment.seed)
    np.random.seed(config.environment.seed)
    
    # Create environment
    if config.environment.environment_type == 'brandl':
        env = SocialLearningEnvironment(
            num_agents=config.environment.num_agents,
            num_states=config.environment.num_states,
            signal_accuracy=config.environment.signal_accuracy,
            network_type=config.environment.network_type,
            horizon=config.training.horizon,
            seed=config.environment.seed
        )
    else:
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
        
    # Run experiment
    print(f"Running {config.environment.environment_type} experiment")
    print(f"Agents: {config.environment.num_agents}")
    print(f"Episodes: {config.training.num_episodes}")
    print(f"Horizon: {config.training.horizon}")
    print(f"Device: {config.device}")
    
    metrics, processed_metrics = run_experiment(env, config)
    
    print("\nExperiment completed!")
    print(f"Results saved to: {config.output_dir}/{config.exp_name}")

if __name__ == "__main__":
    main()