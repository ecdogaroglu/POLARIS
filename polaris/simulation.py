from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from polaris.training.trainer import Trainer


def run_experiment(env, config) -> Tuple[Dict, Dict]:
    """
    Run complete experiment using the trainer interface.

    Args:
        env: Environment instance
        config: Configuration object with attributes matching expected args

    Returns:
        Tuple of (episodic_metrics, serializable_metrics)
    """
    # Convert config to args-like object if needed
    if hasattr(config, "training"):
        # Convert from ExperimentConfig to args-like object
        args = type("Args", (), {})()
        args.num_episodes = config.training.num_episodes
        args.horizon = config.training.horizon
        args.hidden_dim = config.agent.hidden_dim
        args.belief_dim = config.agent.belief_dim
        args.latent_dim = config.agent.latent_dim
        args.learning_rate = config.agent.learning_rate
        args.discount_factor = config.agent.discount_factor
        args.entropy_weight = config.agent.entropy_weight
        args.kl_weight = config.agent.kl_weight
        args.buffer_capacity = config.training.buffer_capacity
        args.batch_size = config.training.batch_size
        args.update_interval = getattr(config.training, "update_interval", 10)
        args.use_gnn = config.agent.use_gnn
        args.use_si = config.agent.use_si
        args.si_importance = config.agent.si_importance
        args.si_damping = config.agent.si_damping
        args.si_exclude_final_layers = config.agent.si_exclude_final_layers
        args.continuous_actions = getattr(
            config.environment, "continuous_actions", False
        )
        args.device = config.device
        args.output_dir = config.output_dir
        args.exp_name = config.exp_name
        args.eval_only = config.eval_only
        args.plot_internal_states = config.plot_internal_states
        args.plot_allocations = config.plot_allocations
        args.latex_style = getattr(config, "latex_style", False)
        args.use_tex = getattr(config, "use_tex", False)
        args.save_model = config.save_model
        args.seed = config.environment.seed

        # Environment specific args
        args.network_type = config.environment.network_type
        args.network_density = getattr(config.environment, "network_density", 0.5)

        # GNN specific args (with defaults)
        args.gnn_layers = 2
        args.attn_heads = 4
        args.temporal_window = 5
    else:
        # Assume config is already an args object
        args = config

    # Create trainer and run
    trainer = Trainer(env, args)

    # Determine if training or evaluation
    training = not args.eval_only
    model_path = getattr(args, "load_model", None)

    return trainer.run_agents(training=training, model_path=model_path)


def main():
    """Main entry point for the polaris-simulate console script."""
    try:
        from polaris.config.args import parse_args
        from polaris.environments.social_learning import SocialLearningEnvironment
        from polaris.environments.strategic_exp import StrategicExperimentationEnvironment
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("This might be due to missing dependencies. Please install with:")
        print("pip install polaris-marl[all]")
        return 1

    try:
        # Parse command line arguments
        args = parse_args()

        # Create environment based on type
        if args.environment_type == "brandl":
            env = SocialLearningEnvironment(
                num_agents=args.num_agents,
                num_states=args.num_states,
                network_type=getattr(args, "network_type", "complete"),
                signal_accuracy=getattr(args, "signal_accuracy", 0.75),
            )
        elif args.environment_type == "strategic_experimentation":
            env = StrategicExperimentationEnvironment(
                num_agents=args.num_agents,
                continuous_actions=getattr(args, "continuous_actions", False),
                safe_payoff=getattr(args, "safe_payoff", 1.0),
            )
        else:
            raise ValueError(f"Unknown environment type: {args.environment_type}")

        # Run experiment
        episodic_metrics, serializable_metrics = run_experiment(env, args)
        print("Experiment completed successfully!")
        print(f"Final metrics: {serializable_metrics}")
        return 0
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        print("Use --help for usage information")
        return 1


if __name__ == "__main__":
    exit(main())
