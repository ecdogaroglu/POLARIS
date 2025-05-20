#!/usr/bin/env python3
"""
Helper script to run POLARIS training with Synaptic Intelligence to prevent catastrophic forgetting.

This script simplifies the process of training agents with Synaptic Intelligence enabled,
creating runs with balanced importance weighting across network layers.
"""

import os
import time
import subprocess
import argparse

def main():
    """Main function to launch training with Synaptic Intelligence."""
    parser = argparse.ArgumentParser(description='Run POLARIS training with Synaptic Intelligence')
    parser.add_argument('--num-agents', type=int, default=4, help='Number of agents')
    parser.add_argument('--num-episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--network-type', type=str, default='complete', help='Network type')
    parser.add_argument('--horizon', type=int, default=1000, help='Steps per episode')
    parser.add_argument('--si-lambda', type=float, default=1.0, help='SI regularization strength')
    parser.add_argument('--si-damping', type=float, default=0.1, help='SI damping factor')
    parser.add_argument('--run-regular', action='store_true', help='Also run a regular experiment without SI')
    args = parser.parse_args()
    
    # Create timestamp for experiment naming
    timestamp = time.time()
    
    # Run experiment with Synaptic Intelligence
    print("Starting training with Synaptic Intelligence...")
    cmd = [
        "python", "experiment.py",
        "--num-agents", str(args.num_agents),
        "--num-episodes", str(args.num_episodes),
        "--network-type", args.network_type,
        "--horizon", str(args.horizon),
        "--enable-si",
        "--si-lambda", str(args.si_lambda),
        "--si-damping", str(args.si_damping),
        "--visualize-si",
        "--save-model",
        "--exp-name", f"si_balanced_importance_{timestamp}"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    # Run evaluation on the SI model
    print("\nRunning evaluation on the SI model...")
    eval_cmd = [
        "python", "experiment.py",
        "--num-agents", str(args.num_agents),
        "--network-type", args.network_type,
        "--horizon", str(args.horizon),
        "--eval-only",
        "--load-model", "auto",
        "--exp-name", f"si_balanced_importance_{timestamp}"
    ]
    
    print(f"Running command: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd)
    
    # Optionally run a regular experiment for comparison
    if args.run_regular:
        print("\nRunning regular training without Synaptic Intelligence for comparison...")
        regular_cmd = [
            "python", "experiment.py",
            "--num-agents", str(args.num_agents),
            "--num-episodes", str(args.num_episodes),
            "--network-type", args.network_type,
            "--horizon", str(args.horizon),
            "--save-model",
            "--exp-name", f"regular_training_{timestamp}"
        ]
        
        print(f"Running command: {' '.join(regular_cmd)}")
        subprocess.run(regular_cmd)
        
        # Run evaluation on the regular model
        print("\nRunning evaluation on the regular model...")
        regular_eval_cmd = [
            "python", "experiment.py",
            "--num-agents", str(args.num_agents),
            "--network-type", args.network_type,
            "--horizon", str(args.horizon),
            "--eval-only",
            "--load-model", "auto",
            "--exp-name", f"regular_training_{timestamp}"
        ]
        
        print(f"Running command: {' '.join(regular_eval_cmd)}")
        subprocess.run(regular_eval_cmd)
    
    # Analyze the SI visualizations
    print("\nExploring Synaptic Intelligence visualizations...")
    explore_cmd = [
        "python", "explore_si_visualizations.py",
        "--results_dir", "results",
        "--experiments", f"si_balanced_importance_{timestamp}"
    ]
    
    print(f"Running command: {' '.join(explore_cmd)}")
    subprocess.run(explore_cmd)
    
    print("\nTraining complete!")
    print(f"Results saved in results/si_balanced_importance_{timestamp}")
    
if __name__ == "__main__":
    main() 