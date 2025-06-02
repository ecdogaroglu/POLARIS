#!/usr/bin/env python3
"""
POLARIS Brandl Policy Inversion Analysis Script

This script:
1. Trains POLARIS agents from scratch in the Brandl social learning environment
2. Creates linear grids of belief distributions and opponent belief distributions
3. Uses invertible belief heads to convert distributions to hidden representations when feeding to policy networks
4. Generates heatmaps showing policy allocation as a function of belief and opponent belief distributions

The script leverages the InvertibleBeliefHead in both the transformer network and the 
opponent belief head in the GNN network to perform the inversion from distributions to hidden space.

Key Implementation:
- Belief grids contain actual belief distributions (probability distributions over states)
- Latent grids contain actual opponent belief distributions (probability distributions over opponent states)
- Invertible transformations happen only when feeding to policy networks for evaluation
- This allows direct interpretation of results in terms of belief and opponent belief distributions
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import json

from polaris.config.experiment_config import (
    ExperimentConfig, AgentConfig, TrainingConfig, BrandlConfig
)
from polaris.environments import SocialLearningEnvironment
from polaris.training.simulation import run_experiment
from polaris.utils.device import get_best_device


def main():
    """Main function to run the Brandl policy inversion analysis."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Brandl Policy Inversion Analysis")
    parser.add_argument('--agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--episodes', type=int, default=1, help='Number of training episodes')
    parser.add_argument('--horizon', type=int, default=1000, help='Steps per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--signal-accuracy', type=float, default=0.75, help='Signal accuracy')
    parser.add_argument('--network-type', type=str, default='complete', 
                       choices=['complete', 'ring', 'star', 'random'], help='Network type')
    parser.add_argument('--network-density', type=float, default=0.5, help='Network density for random networks')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--use-gnn', action='store_true', default=True, help='Use GNN inference')
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'mps', 'cuda'], 
                       help='Device to use')
    parser.add_argument('--grid-resolution', type=int, default=50, help='Resolution of belief/latent grid for heatmap')
    parser.add_argument('--skip-training', action='store_true', help='Skip training and load existing models')
    parser.add_argument('--model-path', type=str, default=None, help='Path to load models from')
    args = parser.parse_args()
    
    # Set initial random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n=== Brandl Policy Inversion Analysis ===\n")
    
    # Create environment
    env = SocialLearningEnvironment(
        num_agents=args.agents,
        num_states=2,  # Binary state space for Brandl
        signal_accuracy=args.signal_accuracy,
        network_type=args.network_type,
        network_params={'density': args.network_density} if args.network_type == 'random' else None,
        horizon=args.horizon,
        seed=args.seed
    )
    
    print(f"Environment: {args.network_type} network with {args.agents} agents")
    print(f"Signal accuracy: {args.signal_accuracy}")
    
    # Step 1: Train agents from scratch (unless skipping)
    if not args.skip_training:
        print("\n=== Step 1: Training Agents from Scratch ===")
        
        # Create training configuration
        config = create_brandl_config(args)
        
        # Run training
        print(f"Training for {args.episodes} episodes...")
        metrics, processed_metrics = run_experiment(env, config)
        
        # The training saves models to a nested directory structure that includes network type
        model_path = Path(config.output_dir) / config.exp_name / f"network_{args.network_type}_agents_{args.agents}" / "models" / "final"
        print(f"Training completed. Models saved to: {model_path}")
    else:
        print("\n=== Step 1: Loading Pre-trained Models ===")
        if args.model_path is None:
            # Try to find the most recent model
            model_path = find_latest_model_path(args)
        else:
            model_path = Path(args.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        print(f"Loading models from: {model_path}")
    
    # Step 2: Load trained agents for analysis
    print("\n=== Step 2: Loading Trained Agents for Analysis ===")
    
    # Create agents for analysis
    agents = create_agents_for_analysis(env, args)
    
    # Load the trained models
    load_trained_models(agents, model_path, args.agents)
    
    # Set agents to evaluation mode
    for agent in agents.values():
        agent.set_eval_mode()
    
    print("Agents loaded and set to evaluation mode")
    
    # Step 3: Generate belief and latent grids using invertible heads
    print("\n=== Step 3: Generating Belief and Latent Grids ===")
    
    belief_grids, latent_grids = generate_invertible_grids(agents, args)
    
    print(f"Generated grids with resolution {args.grid_resolution}x{args.grid_resolution}")
    
    # Step 4: Generate policy allocation heatmaps
    print("\n=== Step 4: Generating Policy Allocation Heatmaps ===")
    
    output_dir = Path(args.output) / f"brandl_policy_inversion_agents_{args.agents}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_policy_heatmaps(agents, belief_grids, latent_grids, output_dir, args)
    
    print(f"\nAnalysis completed! Results saved to: {output_dir}")


def create_brandl_config(args) -> ExperimentConfig:
    """Create experiment configuration for Brandl social learning."""
    # Agent configuration
    agent_config = AgentConfig(
        learning_rate=1e-3,
        discount_factor=0.99,
        use_gnn=args.use_gnn,
        use_si=False,  # Disable SI for cleaner training
        hidden_dim=256,
        belief_dim=256,
        latent_dim=256
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
    exp_name = f"brandl_policy_inversion_training_agents_{args.agents}"
    if args.use_gnn:
        exp_name += "_gnn"
    
    # Create complete configuration
    config = ExperimentConfig(
        agent=agent_config,
        training=training_config,
        environment=env_config,
        device=args.device,
        output_dir=args.output,
        exp_name=exp_name,
        save_model=True,
        load_model=None,
        eval_only=False,
        plot_internal_states=False,
        plot_allocations=False,
        latex_style=False,
        use_tex=False
    )
    
    # Disable all plotting to avoid visualization errors during training
    config.plot_internal_states = False
    config.plot_allocations = False
    config.plot_learning_curves = False
    config.plot_belief_evolution = False
    config.plot_network_analysis = False
    config.generate_plots = False
    config.disable_plotting = True  # This is checked in the trainer to skip all plotting
    
    return config


def find_latest_model_path(args) -> Path:
    """Find the most recent model path for the given configuration."""
    base_dir = Path(args.output)
    
    # Look for directories matching the pattern
    pattern = f"brandl_*_agents_{args.agents}"
    if args.use_gnn:
        pattern += "_gnn"
    
    matching_dirs = list(base_dir.glob(pattern))
    
    if not matching_dirs:
        raise FileNotFoundError(f"No model directories found matching pattern: {pattern}")
    
    # Find the most recent one
    latest_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
    
    # The models are saved in a nested directory structure that includes network type
    model_path = latest_dir / f"network_{args.network_type}_agents_{args.agents}" / "models" / "final"
    
    return model_path


def create_agents_for_analysis(env, args):
    """Create agents for analysis with the same configuration as training."""
    from polaris.agents.polaris_agent import POLARISAgent
    from polaris.utils.encoding import calculate_observation_dimension
    
    obs_dim = calculate_observation_dimension(env)
    agents = {}
    
    for agent_id in range(env.num_agents):
        agent = POLARISAgent(
            agent_id=agent_id,
            num_agents=env.num_agents,
            num_states=env.num_states,
            observation_dim=obs_dim,
            action_dim=env.num_states,
            hidden_dim=256,
            belief_dim=256,
            latent_dim=256,
            learning_rate=1e-3,
            discount_factor=0.99,
            device=args.device,
            use_gnn=args.use_gnn,
            continuous_actions=False
        )
        agents[agent_id] = agent
    
    return agents


def load_trained_models(agents, model_path, num_agents):
    """Load trained models for all agents."""
    for agent_id in range(num_agents):
        model_file = model_path / f"agent_{agent_id}.pt"
        if model_file.exists():
            print(f"Loading model for agent {agent_id}")
            agents[agent_id].load(str(model_file), evaluation_mode=True)
        else:
            raise FileNotFoundError(f"Model file not found: {model_file}")


def generate_invertible_grids(agents, args) -> Tuple[Dict, Dict]:
    """
    Generate belief and latent grids using the invertible heads.
    
    Returns:
        belief_grids: Dict mapping agent_id to belief distribution grids (actual distributions)
        latent_grids: Dict mapping agent_id to opponent belief distribution grids (actual distributions)
    """
    resolution = args.grid_resolution
    device = args.device
    
    belief_grids = {}
    latent_grids = {}
    
    for agent_id, agent in agents.items():
        print(f"Generating grids for agent {agent_id}...")
        
        # Create linear grids for belief distributions
        # For Brandl environment, we have binary states, so belief is 2D
        belief_dim = 2  # Binary belief distribution
        
        # Create a grid of belief distributions (simplex sampling)
        belief_values = []
        for i in range(resolution):
            # Sample from simplex: belief in good state vs bad state
            p_good = i / (resolution - 1)  # From 0 to 1
            p_bad = 1 - p_good
            belief_dist = torch.tensor([p_bad, p_good], dtype=torch.float32, device=device)
            belief_values.append(belief_dist)
        
        # Store the actual belief distributions (not hidden representations)
        belief_grids[agent_id] = torch.stack(belief_values)
        
        # Create opponent belief grid - also store actual distributions, not hidden representations
        opponent_belief_values = []
        for i in range(resolution):
            # Create opponent belief distribution
            p_opponent_good = i / (resolution - 1)
            p_opponent_bad = 1 - p_opponent_good
            opponent_belief_dist = torch.tensor([p_opponent_bad, p_opponent_good], 
                                              dtype=torch.float32, device=device)
            opponent_belief_values.append(opponent_belief_dist)
        
        # Store the actual opponent belief distributions (not hidden representations)
        latent_grids[agent_id] = torch.stack(opponent_belief_values)
        
        print(f"  Belief grid shape: {belief_grids[agent_id].shape}")
        print(f"  Latent grid shape: {latent_grids[agent_id].shape}")
    
    return belief_grids, latent_grids


def generate_policy_heatmaps(agents, belief_grids, latent_grids, output_dir, args):
    """Generate policy allocation heatmaps for each agent."""
    resolution = args.grid_resolution
    
    for agent_id, agent in agents.items():
        print(f"Generating heatmap for agent {agent_id}...")
        
        belief_grid = belief_grids[agent_id]  # These are actual belief distributions
        latent_grid = latent_grids[agent_id]  # These are opponent belief distributions
        
        # Create meshgrid for heatmap
        policy_allocations = np.zeros((resolution, resolution))
        
        # Evaluate policy for each combination of belief and latent
        with torch.no_grad():
            for i, belief_dist in enumerate(belief_grid):
                for j, opponent_belief_dist in enumerate(latent_grid):
                    # Convert belief distribution to hidden representation for policy input
                    # Transformer case: use invertible belief head
                    belief_hidden = agent.belief_processor.inverse_belief_transform(
                        belief_dist.unsqueeze(0)
                    )

                    
                    # Convert opponent belief distribution to latent representation for policy input
                    # GNN case: use invertible opponent belief head
                    latent_repr = agent.inference_module.inverse_belief_transform(
                        opponent_belief_dist.unsqueeze(0)
                    )
                    latent_input = latent_repr  # [1, latent_dim]

                    
                    # Prepare inputs for policy network
                    belief_input = belief_hidden.unsqueeze(1)  # [1, 1, hidden_dim]
                    
                    # Get policy output
                    if agent.continuous_actions:
                        # For continuous actions, get the mean allocation
                        mean, _ = agent.policy(belief_input, latent_input)
                        allocation = mean.squeeze().cpu().numpy()
                    else:
                        # For discrete actions, get action probabilities
                        action_logits = agent.policy(belief_input, latent_input)
                        action_probs = torch.softmax(action_logits, dim=-1)
                        # Use probability of choosing action 1 (good state) as allocation
                        allocation = action_probs[0, 1].cpu().numpy()
                    
                    policy_allocations[i, j] = allocation
        print(f"  Policy allocations: {policy_allocations}")
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        # Create belief and latent axis labels
        belief_labels = [f"{i/(resolution-1):.2f}" for i in range(0, resolution, resolution//5)]
        latent_labels = [f"{i/(resolution-1):.2f}" for i in range(0, resolution, resolution//5)]
        
        # Plot heatmap
        sns.heatmap(
            policy_allocations,
            xticklabels=[latent_labels[i] if i < len(latent_labels) else "" 
                        for i in range(0, resolution, resolution//5)],
            yticklabels=[belief_labels[i] if i < len(belief_labels) else "" 
                        for i in range(0, resolution, resolution//5)],
            cmap='viridis',
            cbar_kws={'label': 'Policy Allocation'},
            square=True
        )
        
        plt.title(f'Agent {agent_id} Policy Allocation Heatmap\n'
                 f'Belief vs Opponent Belief Distribution',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Opponent Belief (Good State Probability)', fontsize=12)
        plt.ylabel('Own Belief (Good State Probability)', fontsize=12)
        
        # Add colorbar label
        cbar = plt.gca().collections[0].colorbar
        cbar.set_label('Policy Allocation (Probability of Choosing Good State)', 
                      fontsize=12)
        
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = output_dir / f"agent_{agent_id}_policy_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved heatmap to: {heatmap_path}")
        
        # Save raw data for further analysis
        data_path = output_dir / f"agent_{agent_id}_policy_data.npz"
        np.savez(
            data_path,
            policy_allocations=policy_allocations,
            belief_grid=belief_grid.detach().cpu().numpy(),  # Actual belief distributions
            latent_grid=latent_grid.detach().cpu().numpy()   # Opponent belief distributions
        )
        print(f"  Saved raw data to: {data_path}")
    
    # Generate combined heatmap for all agents
    if len(agents) > 1:
        generate_combined_heatmap(agents, belief_grids, latent_grids, output_dir, args)


def generate_combined_heatmap(agents, belief_grids, latent_grids, output_dir, args):
    """Generate a combined heatmap showing all agents' policies."""
    resolution = args.grid_resolution
    num_agents = len(agents)
    
    fig, axes = plt.subplots(1, num_agents, figsize=(5*num_agents, 4))
    if num_agents == 1:
        axes = [axes]
    
    for idx, (agent_id, agent) in enumerate(agents.items()):
        belief_grid = belief_grids[agent_id]  # These are actual belief distributions
        latent_grid = latent_grids[agent_id]  # These are opponent belief distributions
        
        # Create policy allocation matrix
        policy_allocations = np.zeros((resolution, resolution))
        
        with torch.no_grad():
            for i, belief_dist in enumerate(belief_grid):
                for j, opponent_belief_dist in enumerate(latent_grid):
                    # Convert belief distribution to hidden representation for policy input
                    if hasattr(agent.belief_processor, 'belief_head'):
                        # Transformer case: use invertible belief head
                        belief_hidden = agent.belief_processor.inverse_belief_transform(
                            belief_dist.unsqueeze(0)
                        )
                    else:
                        # Fallback: pad belief distribution to match hidden dimension
                        hidden_dim = agent.belief_processor.hidden_dim
                        belief_hidden = torch.zeros(1, hidden_dim, device=belief_dist.device)
                        belief_hidden[0, :len(belief_dist)] = belief_dist
                    
                    # Convert opponent belief distribution to latent representation for policy input
                    if args.use_gnn and hasattr(agent.inference_module, 'inverse_belief_transform'):
                        # GNN case: use invertible opponent belief head
                        latent_repr = agent.inference_module.inverse_belief_transform(
                            opponent_belief_dist.unsqueeze(0)
                        )
                        latent_input = latent_repr  # [1, latent_dim]
                    else:
                        # Fallback: pad opponent belief distribution to match latent dimension
                        latent_dim = agent.latent_dim
                        latent_repr = torch.zeros(1, latent_dim, device=opponent_belief_dist.device)
                        latent_repr[0, :len(opponent_belief_dist)] = opponent_belief_dist
                        latent_input = latent_repr  # [1, latent_dim]
                    
                    # Prepare inputs for policy network
                    belief_input = belief_hidden.unsqueeze(1)  # [1, 1, hidden_dim]
                    
                    # Get policy output
                    if agent.continuous_actions:
                        mean, _ = agent.policy(belief_input, latent_input)
                        allocation = mean.squeeze().cpu().numpy()
                    else:
                        action_logits = agent.policy(belief_input, latent_input)
                        action_probs = torch.softmax(action_logits, dim=-1)
                        allocation = action_probs[0, 1].cpu().numpy()
                    
                    policy_allocations[i, j] = allocation
        
        # Plot on subplot
        im = axes[idx].imshow(
            policy_allocations,
            cmap='viridis',
            aspect='auto',
            origin='lower',
            extent=[0, 1, 0, 1]
        )
        
        axes[idx].set_title(f'Agent {agent_id}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Opponent Belief (Good State Probability)', fontsize=10)
        if idx == 0:
            axes[idx].set_ylabel('Own Belief (Good State Probability)', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[idx])
        cbar.set_label('Policy\nAllocation', fontsize=9)
    
    plt.suptitle(f'Policy Allocation Comparison ({args.network_type.title()} Network)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save combined heatmap
    combined_path = output_dir / "combined_policy_heatmap.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined heatmap to: {combined_path}")
    
    # Save analysis summary
    save_analysis_summary(agents, output_dir, args)


def save_analysis_summary(agents, output_dir, args):
    """Save a summary of the analysis."""
    summary = {
        "experiment_config": {
            "num_agents": args.agents,
            "network_type": args.network_type,
            "signal_accuracy": args.signal_accuracy,
            "use_gnn": args.use_gnn,
            "grid_resolution": args.grid_resolution,
            "device": args.device
        },
        "agents_analyzed": list(agents.keys()),
        "grid_contents": {
            "belief_grids": "actual belief distributions (probability distributions over states)",
            "latent_grids": "actual opponent belief distributions (probability distributions over opponent states)"
        },
        "invertible_components": {
            "transformer_belief_head": "converts belief distributions to hidden representations for policy input",
            "gnn_opponent_belief_head": "converts opponent belief distributions to latent representations for policy input" if args.use_gnn else "not used"
        },
        "analysis_method": "invertible transformations applied only when feeding to policy networks",
        "output_files": [
            f"agent_{agent_id}_policy_heatmap.png" for agent_id in agents.keys()
        ] + [
            f"agent_{agent_id}_policy_data.npz" for agent_id in agents.keys()
        ] + ["combined_policy_heatmap.png", "analysis_summary.json"]
    }
    
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved analysis summary to: {summary_path}")


if __name__ == "__main__":
    main() 