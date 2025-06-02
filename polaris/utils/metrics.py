"""
Metrics collection and processing for POLARIS experiments.
"""

import json
import numpy as np

from ..utils.math import (
    calculate_learning_rate,
    calculate_incentive,
    calculate_dynamic_mpe,
    calculate_policy_kl_divergence,
)
from ..utils.io import make_json_serializable


class AllocationFormatError(ValueError):
    """Raised when allocation format is not supported."""
    pass


class IncompatibleMetricsError(ValueError):
    """Raised when metrics data is incompatible or missing."""
    pass


class UnknownEnvironmentError(ValueError):
    """Raised when environment type cannot be determined for theoretical bounds."""
    pass


class MetricsTracker:
    """
    A class to track and manage metrics for POLARIS experiments.
    
    This class encapsulates all metrics tracking functionality including initialization,
    updates, processing, and serialization.
    
    Example usage:
        # Initialize the tracker
        tracker = MetricsTracker(env, args, training=True)
        
        # Update metrics during simulation
        tracker.update(info, actions, action_probs)
        
        # Get learning rates
        learning_rates = tracker.get_learning_rates()
        
        # Get theoretical bounds
        bounds = tracker.get_theoretical_bounds()
        
        # Prepare for serialization
        serializable = tracker.prepare_for_serialization(learning_rates, bounds, num_steps)
        
        # Save to file
        tracker.save_to_file(output_dir)
        
        # For backward compatibility, you can still access raw metrics
        raw_metrics = tracker.get_raw_metrics()
    """
    
    def __init__(self, env, args, training=True):
        """
        Initialize the metrics tracker.
        
        Args:
            env: The environment object
            args: Configuration arguments
            training: Whether this is for training or evaluation
        """
        self.env = env
        self.args = args
        self.training = training
        self.metrics = self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize metrics dictionary for tracking experiment results."""
        metrics = {
            "mistake_rates": [],
            "incorrect_probs": [],
            "action_probs": {agent_id: [] for agent_id in range(self.env.num_agents)},
            "full_action_probs": {agent_id: [] for agent_id in range(self.env.num_agents)},
            "true_states": [],
            "agent_actions": {agent_id: [] for agent_id in range(self.env.num_agents)},
            "allocations": {agent_id: [] for agent_id in range(self.env.num_agents)},
            "signals": [],  # Track signals over time
        }

        # Add training-specific or evaluation-specific metrics
        if self.training:
            metrics["training_loss"] = []
            # Add training losses tracking for catastrophic forgetting diagnostics
            metrics["training_losses"] = {agent_id: [] for agent_id in range(self.env.num_agents)}
            # Add action logits tracking for catastrophic forgetting diagnostics  
            metrics["action_logits"] = {agent_id: [] for agent_id in range(self.env.num_agents)}
            # Add belief distribution tracking for strategic experimentation if plot_internal_states is enabled
            if hasattr(self.args, "plot_internal_states") and self.args.plot_internal_states:
                metrics["belief_distributions"] = {
                    agent_id: [] for agent_id in range(self.env.num_agents)
                }
                metrics["opponent_belief_distributions"] = {
                    agent_id: [] for agent_id in range(self.env.num_agents)
                }
        else:
            metrics["correct_actions"] = {agent_id: 0 for agent_id in range(self.env.num_agents)}
            # Add belief distribution tracking for evaluation
            metrics["belief_distributions"] = {
                agent_id: [] for agent_id in range(self.env.num_agents)
            }
            metrics["opponent_belief_distributions"] = {
                agent_id: [] for agent_id in range(self.env.num_agents)
            }

        # Add strategic experimentation specific metrics if applicable
        if hasattr(self.env, "safe_payoff"):
            # Add allocation tracking for continuous actions
            # Check if environment uses continuous actions by looking at action_space_type
            is_continuous = (
                hasattr(self.env, "action_space_type") and self.env.action_space_type == "continuous"
            ) or (hasattr(self.args, "continuous_actions") and self.args.continuous_actions)
            
            if is_continuous:
                metrics["allocations"] = {
                    agent_id: [] for agent_id in range(self.env.num_agents)
                }
                metrics["mpe_allocations"] = {
                    agent_id: [] for agent_id in range(self.env.num_agents)
                }
                # Add KL divergence tracking for policy convergence
                metrics["policy_kl_divergence"] = {
                    agent_id: [] for agent_id in range(self.env.num_agents)
                }
                metrics["policy_means"] = {
                    agent_id: [] for agent_id in range(self.env.num_agents)
                }
                metrics["policy_stds"] = {
                    agent_id: [] for agent_id in range(self.env.num_agents)
                }
                # Add incentive tracking
                metrics["agent_incentives"] = {
                    agent_id: [] for agent_id in range(self.env.num_agents)
                }

        metrics["belief_distributions"] = {
            agent_id: [] for agent_id in range(self.env.num_agents)
        }
        metrics["agent_beliefs"] = {agent_id: [] for agent_id in range(self.env.num_agents)}
        
        print(
            f"Initialized metrics dictionary with {len(metrics['action_probs'])} agent entries"
        )
        return metrics
    
    def update(self, info, actions, action_probs=None, beliefs=None, latent_states=None, opponent_beliefs=None):
        """
        Update metrics dictionary with the current step information.
        
        Args:
            info: Information dictionary from environment step
            actions: Dictionary of agent actions
            action_probs: Dictionary of action probabilities
            beliefs: Agent beliefs (optional)
            latent_states: Latent states (optional)
            opponent_beliefs: Opponent beliefs (optional)
        """
        # Update true state history
        if "true_state" in info:
            self.metrics["true_states"].append(info["true_state"])

        # Update signals history
        if "signals" in info:
            self.metrics["signals"].append(info["signals"])

        # Update mistake rates and incorrect probabilities
        if "mistake_rate" in info:
            self.metrics["mistake_rates"].append(info["mistake_rate"])
        if "incorrect_prob" in info:
            self.metrics["incorrect_probs"].append(info["incorrect_prob"])

            # Store incorrect probabilities in per-agent format for plotting
            incorrect_prob = info["incorrect_prob"]
            if isinstance(incorrect_prob, list):
                # Per-agent incorrect probabilities
                for agent_id, prob in enumerate(incorrect_prob):
                    if agent_id in self.metrics["action_probs"]:
                        self.metrics["action_probs"][agent_id].append(prob)
            else:
                # Scalar incorrect probability - apply to all agents
                for agent_id in self.metrics["action_probs"]:
                    self.metrics["action_probs"][agent_id].append(incorrect_prob)

        # Update action probabilities
        if action_probs is not None:
            for agent_id, probs in action_probs.items():
                self.metrics["full_action_probs"][agent_id].append(probs)

        # Update action history
        for agent_id, action in actions.items():
            self.metrics["agent_actions"][agent_id].append(action)

        # Update allocations for strategic experimentation if available
        if "allocations" in info and "allocations" in self.metrics:
            self._update_allocations(info["allocations"])

        # Update policy means and stds if available
        if "policy_means" in info and "policy_means" in self.metrics:
            self._update_policy_means(info["policy_means"])

        if "policy_stds" in info and "policy_stds" in self.metrics:
            self._update_policy_stds(info["policy_stds"])

        # Update agent beliefs and incentives for strategic experimentation
        if "agent_beliefs" in info and "env_params" in info:
            self._update_agent_beliefs_and_incentives(info)

        # Update KL divergence based on dynamic MPE allocation
        if "policy_means" in self.metrics and "policy_stds" in self.metrics:
            self._update_kl_divergence(info)

        # Update opponent beliefs if requested and available
        if opponent_beliefs is not None and "opponent_belief_distributions" in self.metrics:
            for agent_id, opponent_belief in opponent_beliefs.items():
                self.metrics["opponent_belief_distributions"][agent_id].append(opponent_belief)

    def _update_allocations(self, allocations):
        """Update allocation metrics."""
        if isinstance(allocations, dict):
            # Dictionary format
            for agent_id, allocation in allocations.items():
                if agent_id in self.metrics["allocations"]:
                    self.metrics["allocations"][agent_id].append(allocation)
        elif isinstance(allocations, (list, np.ndarray)):
            # List or array format
            for agent_id in range(len(allocations)):
                if agent_id in self.metrics["allocations"]:
                    self.metrics["allocations"][agent_id].append(allocations[agent_id])
        else:
            raise AllocationFormatError(
                f"Unsupported allocations format: {type(allocations)}. "
                f"Expected dict, list, or numpy array."
            )

    def _update_policy_means(self, policy_means):
        """Update policy means metrics."""
        for agent_id, mean in enumerate(policy_means):
            if agent_id in self.metrics["policy_means"]:
                self.metrics["policy_means"][agent_id].append(mean)

    def _update_policy_stds(self, policy_stds):
        """Update policy standard deviations metrics."""
        for agent_id, std in enumerate(policy_stds):
            if agent_id in self.metrics["policy_stds"]:
                self.metrics["policy_stds"][agent_id].append(std)

    def _update_agent_beliefs_and_incentives(self, info):
        """Update agent beliefs and calculate incentives."""
        env_params = info["env_params"]
        safe_payoff = env_params.get("safe_payoff")
        drift_rates = env_params.get("drift_rates")
        jump_rates = env_params.get("jump_rates")
        jump_sizes = env_params.get("jump_sizes")
        
        for agent_id, agent_belief in info["agent_beliefs"].items():
            # Store agent beliefs
            if "agent_beliefs" in self.metrics and agent_id in self.metrics["agent_beliefs"]:
                self.metrics["agent_beliefs"][agent_id].append(agent_belief)
            
            # Calculate and store incentives
            if "agent_incentives" in self.metrics and agent_id in self.metrics["agent_incentives"]:
                incentive = calculate_incentive(
                    agent_belief,
                    safe_payoff,
                    drift_rates,
                    jump_rates,
                    jump_sizes,
                )
                self.metrics["agent_incentives"][agent_id].append(incentive)

    def _update_kl_divergence(self, info):
        """Update KL divergence metrics based on dynamic MPE allocation."""
        if "policy_means" in info and "policy_stds" in info:
            for agent_id in self.metrics["policy_kl_divergence"]:
                if agent_id < len(info["policy_means"]) and agent_id < len(
                    info["policy_stds"]
                ):
                    # Try to get agent beliefs from info
                    if "agent_beliefs" in info and agent_id in info["agent_beliefs"]:
                        agent_belief = info["agent_beliefs"][agent_id]

                    # Get environment parameters from info if available
                    env_params = info.get("env_params", {})
                    safe_payoff = env_params.get("safe_payoff")

                    # If no drift rates provided, use defaults
                    drift_rates = env_params.get("drift_rates")
                    jump_rates = env_params.get("jump_rates")
                    jump_sizes = env_params.get("jump_sizes")
                    background_informativeness = env_params.get(
                        "background_informativeness"
                    )
                    num_agents = env_params.get("num_agents")
                    true_state = env_params.get("true_state")

                    # Calculate dynamic MPE based on current belief
                    mpe_allocation = calculate_dynamic_mpe(
                        true_state,
                        agent_belief,
                        safe_payoff,
                        drift_rates,
                        jump_rates,
                        jump_sizes,
                        background_informativeness,
                        num_agents,
                    )

                    # Calculate KL divergence between agent's policy and MPE
                    kl = calculate_policy_kl_divergence(
                        info["policy_means"][agent_id],
                        info["policy_stds"][agent_id],
                        mpe_allocation,
                    )

                    self.metrics["mpe_allocations"][agent_id].append(mpe_allocation)
                    self.metrics["policy_kl_divergence"][agent_id].append(kl)

    def store_incorrect_probabilities(self, info):
        """Store incorrect action probabilities in metrics."""
        if "incorrect_prob" in info:
            incorrect_prob = info["incorrect_prob"]

            # Handle both list and scalar incorrect probabilities
            if isinstance(incorrect_prob, list):
                self.metrics["incorrect_probs"].append(incorrect_prob)

                # Also store in per-agent metrics
                for agent_id, prob in enumerate(incorrect_prob):
                    if agent_id < self.env.num_agents:
                        self.metrics["action_probs"][agent_id].append(prob)
            else:
                # If we only have a scalar, store it and duplicate for all agents
                self.metrics["incorrect_probs"].append(incorrect_prob)
                for agent_id in range(self.env.num_agents):
                    self.metrics["action_probs"][agent_id].append(incorrect_prob)

    def process_incorrect_probabilities(self):
        """Process incorrect probabilities for plotting."""
        agent_incorrect_probs = self.metrics["action_probs"]

        # Check if action_probs is empty and raise error instead of fallback
        if not agent_incorrect_probs or not any(agent_incorrect_probs.values()):
            raise IncompatibleMetricsError(
                "No action probabilities data available. Cannot process incorrect probabilities. "
                "Ensure that metrics are properly updated with action probability data."
            )

        return agent_incorrect_probs

    def get_learning_rates(self):
        """Calculate learning rates for each agent from metrics."""
        learning_rates = {}

        # Calculate learning rate for each agent
        for agent_id, probs in self.metrics["action_probs"].items():
            if len(probs) > 0:
                learning_rates[agent_id] = calculate_learning_rate(probs)
            else:
                learning_rates[agent_id] = 0.0
        return learning_rates

    def prepare_for_serialization(self, learning_rates, theoretical_bounds, num_steps):
        """Prepare metrics for JSON serialization."""
        if not learning_rates:
            raise IncompatibleMetricsError("No learning rates available for serialization")
            
        # Find fastest and slowest learning agents
        fastest_agent = max(learning_rates.items(), key=lambda x: x[1])
        slowest_agent = min(learning_rates.items(), key=lambda x: x[1])
        avg_learning_rate = np.mean(list(learning_rates.values()))

        serializable_metrics = {
            "total_steps": num_steps,
            "mistake_rates": [float(m) for m in self.metrics["mistake_rates"]],
            "incorrect_probs": [
                (
                    [float(p) for p in agent_probs]
                    if isinstance(agent_probs, list)
                    else float(agent_probs)
                )
                for agent_probs in self.metrics["incorrect_probs"]
            ],
            "action_probs": {
                str(agent_id): [float(p) for p in probs]
                for agent_id, probs in self.metrics["action_probs"].items()
            },
            "full_action_probs": {
                str(agent_id): [[float(p) for p in dist] for dist in probs]
                for agent_id, probs in self.metrics["full_action_probs"].items()
            },
            "true_states": self.metrics["true_states"],
            "agent_actions": {
                str(agent_id): [
                    int(a) if isinstance(a, (int, np.integer)) else float(a)
                    for a in actions
                ]
                for agent_id, actions in self.metrics["agent_actions"].items()
            },
            "learning_rates": {str(k): float(v) for k, v in learning_rates.items()},
            # Add belief distributions if they exist
            "has_belief_distributions": "belief_distributions" in self.metrics,
            "fastest_agent": {"id": int(fastest_agent[0]), "rate": float(fastest_agent[1])},
            "slowest_agent": {"id": int(slowest_agent[0]), "rate": float(slowest_agent[1])},
            "avg_learning_rate": float(avg_learning_rate),
        }

        # Add KL divergence data if available
        if "policy_kl_divergence" in self.metrics:
            serializable_metrics["policy_kl_divergence"] = {
                str(agent_id): [float(kl) for kl in kl_values]
                for agent_id, kl_values in self.metrics["policy_kl_divergence"].items()
            }

            # Calculate average KL divergence per agent (over last 20% of steps)
            final_kl_divergences = {}
            for agent_id, kl_values in self.metrics["policy_kl_divergence"].items():
                if len(kl_values) > 0:
                    # Use the last 20% of values to calculate the average
                    last_idx = max(1, int(len(kl_values) * 0.2))
                    final_kl = np.mean(kl_values[-last_idx:])
                    final_kl_divergences[str(agent_id)] = float(final_kl)

            if final_kl_divergences:
                serializable_metrics["final_kl_divergence"] = final_kl_divergences
                serializable_metrics["avg_final_kl_divergence"] = float(
                    np.mean(list(final_kl_divergences.values()))
                )

        # Add allocations if they exist (for Strategic Experimentation)
        if "allocations" in self.metrics:
            serializable_metrics["allocations"] = {
                str(agent_id): [float(a) for a in allocs]
                for agent_id, allocs in self.metrics["allocations"].items()
            }

            # Calculate average allocation per agent (over last 20% of steps)
            final_allocations = {}
            for agent_id, allocs in self.metrics["allocations"].items():
                if len(allocs) > 0:
                    # Use the last 20% of values to calculate the average
                    last_idx = max(1, int(len(allocs) * 0.2))
                    final_alloc = np.mean(allocs[-last_idx:])
                    final_allocations[str(agent_id)] = float(final_alloc)

            if final_allocations:
                serializable_metrics["final_allocations"] = final_allocations
                serializable_metrics["avg_final_allocation"] = float(
                    np.mean(list(final_allocations.values()))
                )

        # Add theoretical bounds based on environment type
        if "autarky_rate" in theoretical_bounds:
            # Social Learning Environment
            serializable_metrics["theoretical_bounds"] = {
                "autarky_rate": float(theoretical_bounds["autarky_rate"]),
                "coordination_rate": float(theoretical_bounds["coordination_rate"]),
                "bound_rate": float(theoretical_bounds["bound_rate"]),
            }
        elif "mpe_neutral" in theoretical_bounds:
            # Strategic Experimentation Environment
            serializable_metrics["theoretical_bounds"] = {
                "mpe_neutral": float(theoretical_bounds["mpe_neutral"]),
                "mpe_good_state": float(theoretical_bounds["mpe_good_state"]),
                "mpe_bad_state": float(theoretical_bounds["mpe_bad_state"]),
            }
        else:
            # Default empty bounds
            serializable_metrics["theoretical_bounds"] = {}

        return serializable_metrics

    def save_to_file(self, output_dir, filename=None):
        """
        Save metrics to a JSON file.

        Args:
            output_dir: Directory to save the file in
            filename: Optional custom filename (if None, uses default naming)
        """
        if filename is None:
            filename = f"training_metrics.json" if self.training else "evaluation_metrics.json"

        # Convert metrics to JSON serializable format
        serializable_metrics = make_json_serializable(self.metrics)

        metrics_file = output_dir / filename
        with open(metrics_file, "w") as f:
            json.dump(serializable_metrics, f, indent=2)

    def get_raw_metrics(self):
        """Get the raw metrics dictionary."""
        return self.metrics

    def get_theoretical_bounds(self):
        """Calculate theoretical performance bounds based on environment type."""
        # Check environment type
        if hasattr(self.env, "get_autarky_rate"):
            # Social Learning Environment
            return {
                "autarky_rate": self.env.get_autarky_rate(),
                "bound_rate": self.env.get_bound_rate(),
                "coordination_rate": self.env.get_coordination_rate(),
            }
        elif hasattr(self.env, "get_theoretical_mpe"):
            # Strategic Experimentation Environment
            # Calculate MPE based on 0.5 beliefs (neutral prior)
            neutral_beliefs = [0.5] * self.env.num_agents
            mpe_allocations = self.env.get_theoretical_mpe(neutral_beliefs)

            # For good state (state 1), optimal allocation is usually higher
            good_beliefs = [0.8] * self.env.num_agents
            good_allocations = self.env.get_theoretical_mpe(good_beliefs)

            # For bad state (state 0), optimal allocation is usually lower
            bad_beliefs = [0.2] * self.env.num_agents
            bad_allocations = self.env.get_theoretical_mpe(bad_beliefs)

            return {
                "mpe_neutral": np.mean(mpe_allocations),
                "mpe_good_state": np.mean(good_allocations),
                "mpe_bad_state": np.mean(bad_allocations),
            }
        else:
            # Raise error instead of fallback
            raise UnknownEnvironmentError(
                f"Unknown environment type: {type(self.env)}. "
                f"Environment must implement either 'get_autarky_rate' (Social Learning) "
                f"or 'get_theoretical_mpe' (Strategic Experimentation) methods."
            )

    def update_training_losses(self, agent_id, losses_dict):
        """
        Update training losses for a specific agent.
        
        Args:
            agent_id: Agent identifier
            losses_dict: Dictionary containing loss values (policy_loss, belief_si_loss, etc.)
        """
        if self.training and "training_losses" in self.metrics:
            self.metrics["training_losses"][agent_id].append(losses_dict.copy())
    
    def update_action_logits(self, agent_id, action_logits):
        """
        Update action logits for a specific agent.
        
        Args:
            agent_id: Agent identifier  
            action_logits: Action logits tensor or array
        """
        if self.training and "action_logits" in self.metrics:
            # Convert tensor to numpy if needed and store
            if hasattr(action_logits, 'detach'):
                logits_data = action_logits.detach().cpu().numpy()
            else:
                logits_data = action_logits
            self.metrics["action_logits"][agent_id].append(logits_data.copy())
