#!/usr/bin/env python3
"""
POLARIS Brandl Social Learning Experiment Script

This script runs experiments with POLARIS agents in a social learning environment
based on the Brandl framework, where agents learn without experimentation by
observing others' actions and receiving private signals.
"""

import os
import torch
import numpy as np
import argparse  # Import here to avoid shadowing the argparse from modules.args
from pathlib import Path

from polaris.environments.social import SocialLearningEnvironment
from polaris.utils.args import parse_args
from polaris.simulation import run_agents


def main():
    """Main function to run the Brandl experiment."""
    # Parse command-line arguments
    args = parse_args()
    
    # Force Brandl environment type
    args.environment_type = 'brandl'
    
    # Set experiment name 
    args.exp_name = 'brandl_experiment'
    
    # Set initial random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Training phase
    print("\n=== Training phase ===\n")
    train_args = argparse.Namespace(**vars(args))
    train_args.num_episodes = 1
    train_args.horizon = 1000
    train_args.eval_only = False
    train_args.save_model = True
    train_args.use_gnn = True
    train_args.use_si = False
    train_args.si_importance = 10
    train_args.visualize_si = False
    train_args.si_exclude_final_layers = False
    run_brandl_experiment(train_args)
    """""
    # Evaluation phase
    print("\n=== Evaluation phase ===\n")
    eval_args = argparse.Namespace(**vars(args))
    eval_args.num_episodes = 10
    eval_args.horizon = 100
    eval_args.eval_only = True
    eval_args.load_model = 'auto'
    eval_args.visualize_si = False
    eval_args.use_gnn = True  # Match the training setting
    run_brandl_experiment(eval_args)"""


def run_brandl_experiment(args):
    """Run the Brandl experiment with the given arguments."""
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
    
    # Keep track of environment type
    args.env_type = "brandl"
    
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
    print(f"Running Brandl social learning experiment")
    print(f"Network type: {args.network_type}, Number of agents: {args.num_agents}")
    print(f"Signal accuracy: {args.signal_accuracy}")
    
    # Calculate theoretical bounds
    autarky_rate = env.get_autarky_rate()
    bound_rate = env.get_bound_rate()
    coordination_rate = env.get_coordination_rate()
    
    print(f"\nTheoretical learning rates:")
    print(f"- Autarky rate: {autarky_rate:.4f} (isolated agent)")
    print(f"- Bound rate: {bound_rate:.4f} (maximum possible for any agent)")
    print(f"- Coordination rate: {coordination_rate:.4f} (achievable with coordination)")
    
    # Print episode information if training
    if not args.eval_only and args.num_episodes > 1:
        print(f"\nTraining with {args.num_episodes} episodes, {args.horizon} steps per episode")
        print(f"True state will be randomly selected at the beginning of each episode with different seeds")
    
    # Run agents in training or evaluation mode
    if args.eval_only:
        run_agents(env, args, training=False, model_path=model_path)
    else:
        run_agents(env, args, training=True, model_path=model_path)


if __name__ == "__main__":
    main() 