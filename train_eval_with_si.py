#!/usr/bin/env python3
"""
Helper script to run POLARIS train-then-evaluate experiments with Synaptic Intelligence.

This script runs training followed by evaluation with Synaptic Intelligence enabled,
helping prevent catastrophic forgetting when learning across different tasks.
"""

import os
import time
import subprocess
import argparse

def main():
    """Main function to launch train-then-evaluate with Synaptic Intelligence."""
    parser = argparse.ArgumentParser(description='Run POLARIS train-then-evaluate with Synaptic Intelligence')
    parser.add_argument('--num-agents', type=int, default=4, help='Number of agents')
    parser.add_argument('--network-type', type=str, default='complete', help='Network type')
    parser.add_argument('--horizon', type=int, default=1000, help='Steps per episode')
    parser.add_argument('--train-episodes', type=int, default=4, help='Number of training episodes')
    parser.add_argument('--eval-episodes', type=int, default=4, help='Number of evaluation episodes')
    parser.add_argument('--si-lambda', type=float, default=1.0, help='SI regularization strength')
    parser.add_argument('--si-damping', type=float, default=0.1, help='SI damping factor')
    parser.add_argument('--run-comparison', action='store_true', help='Also run a comparison without SI')
    args = parser.parse_args()
    
    # Create timestamp for experiment naming
    timestamp = time.time()
    exp_name = f"si_visualization_test_{timestamp}"
    
    # Run train-then-evaluate with Synaptic Intelligence
    print(f"Starting train-then-evaluate with Synaptic Intelligence (exp_name: {exp_name})...")
    cmd = [
        "python", "experiment.py",
        "--num-agents", str(args.num_agents),
        "--network-type", args.network_type,
        "--horizon", str(args.horizon),
        "--train-then-evaluate",
        "--enable-si", 
        "--si-lambda", str(args.si_lambda),
        "--si-damping", str(args.si_damping),
        "--visualize-si",
        "--save-model",
        "--exp-name", exp_name
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    # If comparison requested, run without SI
    if args.run_comparison:
        # Create a new timestamp and experiment name
        comp_timestamp = time.time()
        comp_exp_name = f"regular_training_{comp_timestamp}"
        
        print(f"\nRunning comparison train-then-evaluate WITHOUT Synaptic Intelligence (exp_name: {comp_exp_name})...")
        
        comp_cmd = [
            "python", "experiment.py",
            "--num-agents", str(args.num_agents),
            "--network-type", args.network_type,
            "--horizon", str(args.horizon),
            "--train-then-evaluate",
            "--save-model",
            "--exp-name", comp_exp_name
        ]
        
        print(f"Running command: {' '.join(comp_cmd)}")
        subprocess.run(comp_cmd)
        
        # Compare the results
        print("\nComparing results with and without Synaptic Intelligence...")
        compare_cmd = [
            "python", "explore_si_visualizations.py",
            "--results_dir", "results",
            "--compare_layers",
            "--experiments", exp_name, comp_exp_name
        ]
        
        print(f"Running command: {' '.join(compare_cmd)}")
        subprocess.run(compare_cmd)
    
    print("\nTrain-then-evaluate experiment complete!")
    print(f"Results saved in results/{exp_name}")
    
if __name__ == "__main__":
    main() 