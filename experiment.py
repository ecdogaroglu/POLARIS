#!/usr/bin/env python3
"""
POLARIS Social Learning Experiment Script (Modular Version)

This script runs experiments with POLARIS agents in a social learning environment.
It has been refactored into a modular structure for better organization and maintainability.
"""

import os
import torch
import numpy as np
from pathlib import Path

from modules.environment import SocialLearningEnvironment, StrategicExperimentationEnvironment
from modules.args import parse_args
from modules.simulation import run_agents


def main():
    """Main function to run the experiment."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set initial random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.train_then_evaluate:
        # Train
        args.num_episodes = 2
        args.horizon = 10000
        args.eval_only = False
        args.save_model = True
        args.use_gnn = True
        args.use_si = True
        args.si_importance = 100
        args.seed = 44
        args.visualize_si = True  # Visualize during training (where SI actually happens)
        args.si_exclude_final_layers = False
        run_experiment(args)

        # Evaluate
        args.num_episodes = 10
        args.use_si = True  # Keep SI enabled for consistency 
        args.horizon = 100
        args.eval_only = True
        args.load_model = 'auto'
        args.visualize_si = False  # No need to visualize during evaluation
        run_experiment(args)

    else:
        run_experiment(args)


def run_experiment(args):
    """Run the experiment with the given arguments."""
    # Create appropriate environment based on args
    if args.environment_type == 'brandl':
        # Create Brandl social learning environment
        env = SocialLearningEnvironment(
            num_agents=args.num_agents,
            num_states=args.num_states,
            signal_accuracy=args.signal_accuracy,
            network_type=args.network_type,
            network_params={'density': args.network_density} if args.network_type == 'random' else None,
            horizon=args.horizon,
            seed=args.seed
        )
        env_type = "brandl"
    elif args.environment_type == 'strategic_experimentation':
        # Create Strategic Experimentation environment
        env = StrategicExperimentationEnvironment(
            num_agents=args.num_agents,
            num_states=args.num_states,
            network_type=args.network_type,
            network_params={'density': args.network_density} if args.network_type == 'random' else None,
            horizon=args.horizon,
            seed=args.seed,
            safe_payoff=args.safe_payoff,
            drift_rates=args.drift_rates_list if hasattr(args, 'drift_rates_list') else None,
            diffusion_sigma=args.diffusion_sigma,
            jump_rates=args.jump_rates_list if hasattr(args, 'jump_rates_list') else None,
            jump_sizes=args.jump_sizes_list if hasattr(args, 'jump_sizes_list') else None,
            background_informativeness=args.background_informativeness,
            time_step=args.time_step
        )
        env_type = "strategic_experimentation"
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")
    
    # Keep track of environment type
    args.env_type = env_type
    
    # Determine model path for loading
    model_path = None
    if args.load_model:
        if args.load_model == 'auto':
            # Automatically find the final model directory
            model_path = Path(args.output_dir) / args.exp_name / f"network_{args.network_type}_agents_{args.num_agents}" / 'models' / 'final'
            print(f"Automatically loading final models from: {model_path}")
        else:
            # Use the specified path
            model_path = Path(args.load_model)
            print(f"Loading models from specified path: {model_path}")
    
    # Print experiment information
    print(f"Running experiment with {args.environment_type} environment")
    print(f"Network type: {args.network_type}, Number of agents: {args.num_agents}")
    
    if args.environment_type == 'brandl':
        print(f"Signal accuracy: {args.signal_accuracy}")
    else:
        print(f"Safe payoff: {args.safe_payoff}")
        print(f"Drift rates: {args.drift_rates_list if hasattr(args, 'drift_rates_list') else 'default'}")
    
    # Print episode information if training
    if not args.eval_only and args.num_episodes > 1:
        print(f"Training with {args.num_episodes} episodes, {args.horizon} steps per episode")
        print(f"True state will be randomly selected at the beginning of each episode with different seeds")
    
    # Run agents in training or evaluation mode
    if args.eval_only:
        run_agents(env, args, training=False, model_path=model_path)
    else:
        run_agents(env, args, training=True, model_path=model_path)


if __name__ == "__main__":
    main()