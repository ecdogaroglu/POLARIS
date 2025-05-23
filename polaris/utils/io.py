"""
I/O and file management utilities for POLARIS.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def setup_random_seeds(seed: int, env) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)


def create_output_directory(args, env, training: bool) -> Path:
    """Create and return the output directory for experiment results."""
    dir_prefix = "" if training else "eval_"
    output_dir = (
        Path(args.output_dir)
        / args.exp_name
        / f"{dir_prefix}network_{args.network_type}_agents_{env.num_agents}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_agent_models(
    agents: Dict, model_path: str, num_agents: int, training: bool = True
) -> None:
    """
    Load pre-trained models if a path is provided.

    Args:
        agents: Dictionary of agent objects
        model_path: Path to the directory containing model files
        num_agents: Number of agents to load models for
        training: Whether the models will be used for training (True) or evaluation (False)
    """
    # If no model path is provided, skip loading
    if model_path is None:
        print("No model path provided. Starting with fresh models.")
        return

    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"Warning: Model directory {model_dir} does not exist")
        return

    models_loaded = 0
    for agent_id in range(num_agents):
        model_file = model_dir / f"agent_{agent_id}.pt"
        if model_file.exists():
            print(f"Loading model for agent {agent_id} from {model_file}")
            # Set evaluation_mode=True if we're loading for evaluation
            agents[agent_id].load(str(model_file), evaluation_mode=not training)
            models_loaded += 1
        else:
            print(f"Warning: Model file {model_file} not found")

    if models_loaded == 0:
        print(
            f"No model files found in directory {model_dir} for any of the {num_agents} agents"
        )
    else:
        print(
            f"Successfully loaded {models_loaded} models in {'training' if training else 'evaluation'} mode"
        )


def save_checkpoint_models(agents: Dict, output_dir: Path, step: int) -> None:
    """Save checkpoint models during training."""
    checkpoint_dir = output_dir / "models" / f"checkpoint_{step+1}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for agent_id, agent in agents.items():
        agent.save(str(checkpoint_dir / f"agent_{agent_id}.pt"))

    print(f"Saved checkpoint models at step {step+1} to {checkpoint_dir}")


def save_final_models(agents: Dict, output_dir: Path) -> None:
    """Save final agent models."""
    final_model_dir = output_dir / "models" / "final"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    for agent_id, agent in agents.items():
        agent.save(str(final_model_dir / f"agent_{agent_id}.pt"))

    print(f"Saved final models to {final_model_dir}")


def write_config_file(args, env, bounds: Dict[str, Any], output_dir: Path) -> None:
    """Write configuration to a JSON file."""
    # Create base environment config
    env_config = {
        "num_agents": env.num_agents,
        "num_states": env.num_states,
        "network_type": args.network_type,
        "network_density": (
            args.network_density if args.network_type == "random" else None
        ),
    }

    # Add environment-specific attributes
    if hasattr(env, "signal_accuracy"):
        # Social Learning Environment
        env_config["signal_accuracy"] = env.signal_accuracy
    elif hasattr(env, "safe_payoff"):
        # Strategic Experimentation Environment
        env_config["safe_payoff"] = env.safe_payoff
        env_config["drift_rates"] = env.drift_rates
        env_config["jump_rates"] = env.jump_rates
        env_config["jump_sizes"] = env.jump_sizes
        env_config["background_informativeness"] = env.background_informativeness
        env_config["diffusion_sigma"] = env.diffusion_sigma
        env_config["time_step"] = env.time_step

    # Write config to file
    with open(output_dir / "config.json", "w") as f:
        config = {
            "args": vars(args),
            "theoretical_bounds": bounds,
            "environment": env_config,
        }
        json.dump(config, f, indent=2)


def flatten_episodic_metrics(episodic_metrics: Dict, num_agents: int) -> Dict:
    """
    Flatten episodic metrics into a single combined metrics dictionary for backward compatibility.

    Args:
        episodic_metrics: Dictionary with episodes list containing metrics for each episode
        num_agents: Number of agents in the environment

    Returns:
        combined_metrics: Flattened metrics dictionary
    """
    if not episodic_metrics["episodes"]:
        return {}  # No episodes to flatten

    # Initialize combined metrics based on the structure of the first episode
    first_episode = episodic_metrics["episodes"][0]
    combined_metrics = {}

    # Initialize each key in combined_metrics with the appropriate structure
    for key, value in first_episode.items():
        if isinstance(value, list):
            combined_metrics[key] = []
        elif isinstance(value, dict):
            combined_metrics[key] = {}
            for sub_key in value:
                if isinstance(value[sub_key], list):
                    combined_metrics[key][sub_key] = []
                else:
                    combined_metrics[key][sub_key] = value[sub_key]
        else:
            combined_metrics[key] = value

    # Combine metrics from all episodes
    for episode_metrics in episodic_metrics["episodes"]:
        for key, value in episode_metrics.items():
            if isinstance(value, list):
                combined_metrics[key].extend(value)
            elif isinstance(value, dict):
                for sub_key in value:
                    if isinstance(value[sub_key], list):
                        if sub_key not in combined_metrics[key]:
                            combined_metrics[key][sub_key] = []
                        combined_metrics[key][sub_key].extend(value[sub_key])
                    elif sub_key not in combined_metrics[key]:
                        combined_metrics[key][sub_key] = value[sub_key]

    return combined_metrics


# Agent and simulation utilities


def reset_agent_internal_states(agents: Dict) -> None:
    """Reset all agents' internal states to ensure fresh start."""
    for agent in agents.values():
        agent.reset_internal_state()


def select_agent_actions(agents: Dict, metrics: Dict) -> tuple:
    """Select actions for all agents and return with probabilities."""
    actions = {}
    action_probs = {}

    for agent_id, agent in agents.items():
        action, probs = agent.select_action()
        actions[agent_id] = action

        # Convert to numpy if it's a tensor
        if hasattr(probs, "cpu"):
            probs_np = probs.cpu().numpy()
        else:
            probs_np = probs

        action_probs[agent_id] = probs_np

    return actions, action_probs


def update_total_rewards(total_rewards: np.ndarray, rewards: Dict) -> None:
    """Update total rewards for each agent."""
    for agent_id, reward in rewards.items():
        if isinstance(reward, dict):
            # If reward is a dictionary (strategic experimentation environment),
            # use the 'total' field
            total_rewards[agent_id] += reward["total"]
        else:
            # Standard case - reward is a scalar
            total_rewards[agent_id] += reward


def update_progress_display(
    steps_iterator, info: Dict, total_rewards: np.ndarray, step: int, training: bool
) -> None:
    """Update the progress bar with current information."""
    if "incorrect_prob" in info:
        steps_iterator.set_postfix(incorrect_prob=np.array(info["incorrect_prob"]))

    if training and step > 0 and step % 1000 == 0:
        avg_rewards = total_rewards / (step + 1)
        print(f"\nStep {step}: Average rewards: {avg_rewards}")


# Global metrics tracking
_global_metrics = None


def set_metrics(metrics: Dict) -> None:
    """Set the global metrics dictionary."""
    global _global_metrics
    _global_metrics = metrics


def get_metrics() -> Dict:
    """Get the global metrics dictionary."""
    global _global_metrics
    return _global_metrics


def store_transition_in_buffer(
    buffer,
    signal,
    neighbor_actions,
    belief,
    latent,
    action,
    reward,
    next_signal,
    next_belief,
    next_latent,
    mean,
    logvar,
):
    """Store a transition in the replay buffer."""
    buffer.push(
        signal=signal,
        neighbor_actions=neighbor_actions,
        belief=belief,
        latent=latent,
        action=action,
        reward=reward,
        next_signal=next_signal,
        next_belief=next_belief,
        next_latent=next_latent,
        mean=mean,
        logvar=logvar,
    )
