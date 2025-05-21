#!/usr/bin/env python3
"""
POLARIS Strategic Experimentation Experiment Script

This script runs experiments with POLARIS agents in a strategic experimentation environment
based on the Keller and Rady (2020) framework, where agents allocate resources between
a safe arm with known payoff and a risky arm with unknown state-dependent payoff.
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path

from polaris.environments.strategic import StrategicExperimentationEnvironment
from polaris.utils.args import parse_args
from polaris.simulation import run_agents


def main():
    """Main function to run the Strategic Experimentation experiment."""
    # Parse command-line arguments
    args = parse_args()
    
    # Force Strategic Experimentation environment type
    args.environment_type = 'strategic_experimentation'
    
    # Set experiment name
    args.exp_name = 'strategic_experimentation'
    
    # Set initial random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Training phase
    print("\n=== Training phase ===\n")
    train_args = argparse.Namespace(**vars(args))
    train_args.safe_payoff = 0.5
    # first values for the bad state, second values for the good state
    train_args.drift_rates_list = [0, 1] 
    train_args.jump_rates_list = [0, 0.1]
    train_args.jump_sizes_list = [1.0, 1.0]
    train_args.num_agents = 2
    train_args.num_episodes = 1
    train_args.horizon = 10
    train_args.eval_only = False
    train_args.save_model = True
    train_args.use_gnn = True
    train_args.use_si = False
    train_args.si_importance = 100
    train_args.visualize_si = False
    train_args.si_exclude_final_layers = False
    train_args.latex_style = True
    # Use average reward for strategic experimentation
    train_args.discount_factor = 0.0
    # Enable continuous action space for resource allocation
    train_args.continuous_actions = True
    # Plot agent allocations and internal states
    train_args.plot_allocations = True
    train_args.plot_internal_states = False  # Enable belief state plotting
    run_strategic_experiment(train_args)
    


def run_strategic_experiment(args):
    """Run the Strategic Experimentation experiment with the given arguments."""
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
    print(f"Running Strategic Experimentation experiment (Keller-Rady framework)")
    print(f"Network type: {args.network_type}, Number of agents: {args.num_agents}")
    print(f"Safe payoff: {args.safe_payoff}")
    print(f"Drift rates: {args.drift_rates_list if hasattr(args, 'drift_rates_list') else 'default'}")
    print(f"Jump rates: {args.jump_rates_list if hasattr(args, 'jump_rates_list') else 'default'}")
    print(f"Background informativeness: {args.background_informativeness}")
    print(f"Using continuous action space for resource allocation")
    
    
    # Print learning method
    if args.discount_factor == 0.0:
        print(f"Using average reward criterion (discount factor: 0.0)")
    else:
        print(f"Using discounted reward criterion (discount factor: {args.discount_factor})")
    
    # Print episode information if training
    if not args.eval_only and args.num_episodes > 1:
        print(f"\nTraining with {args.num_episodes} episodes, {args.horizon} steps per episode")
        print(f"True state will be randomly selected at the beginning of each episode with different seeds")
    
    # Print model architecture information
    if args.use_gnn:
        print(f"Using Graph Neural Network with {args.gnn_layers} layers, {args.attn_heads} attention heads")
        print(f"Temporal window size: {args.temporal_window}")
    else:
        print("Using traditional encoder-decoder inference module")
    
    if args.use_si:
        print(f"Using Synaptic Intelligence with importance {args.si_importance} and damping {args.si_damping}")
        if args.si_exclude_final_layers:
            print("Excluding final layers from SI protection")
    
    # Run agents in training or evaluation mode
    if args.eval_only:
        run_agents(env, args, training=False, model_path=model_path)
    else:
        run_agents(env, args, training=True, model_path=model_path)


if __name__ == "__main__":
    main() 