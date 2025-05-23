"""
Evaluation logic for POLARIS agents.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..utils.encoding import encode_observation
from ..utils.io import (
    reset_agent_internal_states,
    select_agent_actions,
    setup_random_seeds,
    update_progress_display,
    update_total_rewards,
)
from ..utils.metrics import (
    calculate_agent_learning_rates_from_metrics,
    initialize_metrics,
    update_metrics,
)


class Evaluator:
    """
    Handles evaluation of trained POLARIS agents.

    This class provides comprehensive evaluation capabilities including:
    - Multi-episode evaluation
    - Performance metrics calculation
    - Belief state tracking
    - Action accuracy assessment
    - Environment-specific metric collection
    """

    def __init__(self, env, agents: Dict, args):
        """
        Initialize the evaluator.

        Args:
            env: The environment to evaluate in
            agents: Dictionary of trained agents
            args: Configuration arguments
        """
        self.env = env
        self.agents = agents
        self.args = args
        self.metrics = None

    def evaluate(
        self,
        num_episodes: int = None,
        num_steps: int = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate agents for specified episodes or steps.

        Args:
            num_episodes: Number of episodes to evaluate (overrides args if provided)
            num_steps: Number of steps per episode (overrides args if provided)
            output_dir: Directory to save evaluation results

        Returns:
            Dictionary containing evaluation results and metrics
        """
        # Use provided values or fall back to args
        eval_episodes = (
            num_episodes
            if num_episodes is not None
            else getattr(self.args, "num_episodes", 1)
        )
        eval_steps = (
            num_steps if num_steps is not None else getattr(self.args, "horizon", 1000)
        )

        print(
            f"Starting evaluation for {eval_episodes} episode(s) with {eval_steps} steps each"
        )

        # Set all agents to evaluation mode
        self._set_agents_eval_mode()

        # Initialize episodic metrics to store each episode separately
        episodic_metrics = {"episodes": []}

        # Run evaluation episodes
        for episode in range(eval_episodes):
            # Set different seed for each episode
            episode_seed = (
                self.args.seed + episode + 1000
            )  # Offset to avoid training seed overlap
            setup_random_seeds(episode_seed, self.env)
            print(
                f"\nEvaluating episode {episode+1}/{eval_episodes} with seed {episode_seed}"
            )

            # Initialize fresh metrics for this episode
            self.metrics = initialize_metrics(self.env, self.args, training=False)

            # Run single episode evaluation
            episode_results = self._evaluate_single_episode(eval_steps)

            # Store this episode's metrics
            episodic_metrics["episodes"].append(episode_results)

        # Aggregate results across episodes
        aggregated_results = self._aggregate_episode_results(episodic_metrics)

        # Calculate learning rates and performance metrics
        learning_rates = calculate_agent_learning_rates_from_metrics(aggregated_results)

        # Add evaluation summary
        evaluation_summary = self._calculate_evaluation_summary(
            episodic_metrics, learning_rates
        )

        # Combine all results
        final_results = {
            "episodic_metrics": episodic_metrics,
            "aggregated_metrics": aggregated_results,
            "learning_rates": learning_rates,
            "evaluation_summary": evaluation_summary,
            "num_episodes": eval_episodes,
            "steps_per_episode": eval_steps,
        }

        return final_results

    def _evaluate_single_episode(self, num_steps: int) -> Dict[str, Any]:
        """
        Evaluate agents for a single episode.

        Args:
            num_steps: Number of steps to run

        Returns:
            Episode results and metrics
        """
        start_time = time.time()

        # Initialize environment
        observations = self.env.initialize()
        total_rewards = np.zeros(self.env.num_agents)

        # Print environment state information
        self._print_environment_info()

        # Reset agent internal states
        reset_agent_internal_states(self.agents)

        # Main evaluation loop
        steps_iterator = tqdm(range(num_steps), desc="Evaluating")
        for step in steps_iterator:
            # Get agent actions (no exploration during evaluation)
            actions, action_probs = select_agent_actions(self.agents, self.metrics)

            # Collect additional information for continuous actions
            policy_info = self._collect_policy_information()

            # Take environment step
            next_observations, rewards, done, info = self.env.step(
                actions, action_probs
            )

            # Add policy information to info
            if policy_info:
                info.update(policy_info)

            # Update rewards
            if rewards:
                update_total_rewards(total_rewards, rewards)

            # Update agent states for metric tracking
            self._update_agent_states_for_evaluation(
                observations, next_observations, actions, rewards, step
            )

            # Update observations for next step
            observations = next_observations

            # Store metrics
            update_metrics(self.metrics, info, actions, action_probs)

            # Update progress display
            update_progress_display(
                steps_iterator, info, total_rewards, step, training=False
            )

            if done:
                break

        # Calculate episode duration
        episode_time = time.time() - start_time
        print(f"Episode completed in {episode_time:.2f} seconds")

        # Add episode summary to metrics
        self.metrics["episode_time"] = episode_time
        self.metrics["total_rewards"] = {
            i: total_rewards[i] for i in range(self.env.num_agents)
        }
        self.metrics["final_observations"] = observations

        return self.metrics

    def _update_agent_states_for_evaluation(
        self, observations, next_observations, actions, rewards, step
    ):
        """Update agent states during evaluation (no training updates)."""
        # Check if we're using continuous actions
        continuous_actions = (
            hasattr(self.args, "continuous_actions") and self.args.continuous_actions
        )

        for agent_id, agent in self.agents.items():
            # Get current and next observations
            obs_data = observations[agent_id]
            next_obs_data = next_observations[agent_id]

            # Extract signals and neighbor actions based on environment type
            if "signal" in obs_data:
                # Social Learning Environment format
                signal = obs_data["signal"]
                next_signal = next_obs_data["signal"]
                neighbor_actions = obs_data["neighbor_actions"]
                next_neighbor_actions = next_obs_data["neighbor_actions"]
            elif "background_signal" in obs_data:
                # Strategic Experimentation Environment format
                if (
                    "background_increment" in obs_data
                    and "background_increment" in next_obs_data
                ):
                    signal = obs_data["background_increment"]
                    next_signal = next_obs_data["background_increment"]
                else:
                    signal = obs_data["background_signal"]
                    next_signal = next_obs_data["background_signal"]

                neighbor_allocations = obs_data.get("neighbor_allocations", {})
                next_neighbor_allocations = next_obs_data.get(
                    "neighbor_allocations", {}
                )
                neighbor_actions = (
                    {} if neighbor_allocations is None else neighbor_allocations
                )
                next_neighbor_actions = (
                    {}
                    if next_neighbor_allocations is None
                    else next_neighbor_allocations
                )

            # Encode observations
            continuous_signal = "background_increment" in obs_data

            signal_encoded, actions_encoded = encode_observation(
                signal=signal,
                neighbor_actions=neighbor_actions,
                num_agents=self.env.num_agents,
                num_states=self.env.num_states,
                continuous_actions=continuous_actions,
                continuous_signal=continuous_signal,
            )

            # Update agent belief state (no parameter updates during evaluation)
            with torch.no_grad():
                next_belief, next_dstr = agent.observe(signal_encoded, actions_encoded)

                # Infer latent state for next observation
                next_latent = agent.infer_latent(
                    signal_encoded,
                    actions_encoded,
                    (
                        rewards[agent_id]
                        if isinstance(rewards[agent_id], float)
                        else rewards[agent_id]["total"]
                    ),
                    signal_encoded,  # Use current signal for latent inference during evaluation
                )

            # Store belief distribution
            belief_distribution = agent.get_belief_distribution()
            self.metrics["belief_distributions"][agent_id].append(
                belief_distribution.detach().cpu().numpy()
            )

    def _collect_policy_information(self) -> Dict[str, Any]:
        """Collect policy-specific information for analysis."""
        policy_info = {}

        if hasattr(self.args, "continuous_actions") and self.args.continuous_actions:
            policy_means = []
            policy_stds = []
            agent_beliefs = {}

            for agent_id, agent in self.agents.items():
                if hasattr(agent, "action_mean") and hasattr(agent.policy, "forward"):
                    # Get policy parameters
                    with torch.no_grad():
                        mean, log_std = agent.policy(
                            agent.current_belief, agent.current_latent
                        )
                        std = torch.exp(log_std)

                    policy_means.append(mean.item())
                    policy_stds.append(std.item())

                    # Extract agent's belief about the state
                    if (
                        hasattr(agent, "current_belief_distribution")
                        and agent.current_belief_distribution is not None
                    ):
                        if torch.isnan(agent.current_belief_distribution).any():
                            agent_beliefs[agent_id] = 0.5
                        else:
                            if agent.current_belief_distribution.size(1) == 1:
                                raw_value = agent.current_belief_distribution[
                                    0, 0
                                ].item()
                                agent_beliefs[agent_id] = max(0.0, min(1.0, raw_value))
                            elif agent.current_belief_distribution.shape[-1] == 2:
                                agent_beliefs[agent_id] = (
                                    agent.current_belief_distribution[0, 1].item()
                                )
                            else:
                                belief_weights = torch.arange(
                                    agent.current_belief_distribution.shape[-1],
                                    device=agent.current_belief_distribution.device,
                                ).float()
                                belief_weights = belief_weights / (
                                    agent.current_belief_distribution.shape[-1] - 1
                                )
                                agent_beliefs[agent_id] = torch.sum(
                                    agent.current_belief_distribution * belief_weights,
                                    dim=-1,
                                ).item()

            policy_info.update(
                {
                    "policy_means": policy_means,
                    "policy_stds": policy_stds,
                    "agent_beliefs": agent_beliefs,
                }
            )

        return policy_info

    def _print_environment_info(self):
        """Print information about the current environment state."""
        if hasattr(self.env, "safe_payoff"):
            # Strategic Experimentation Environment
            if self.env.true_state == 0:
                print(
                    f"True state is bad. Drift rate: {self.env.drift_rates[self.env.true_state]} "
                    f"Jump rate: {self.env.jump_rates[self.env.true_state]} "
                    f"Jump size: {self.env.jump_sizes[self.env.true_state]}"
                )
            else:
                print(
                    f"True state is good. Drift rate: {self.env.drift_rates[self.env.true_state]} "
                    f"Jump rate: {self.env.jump_rates[self.env.true_state]} "
                    f"Jump size: {self.env.jump_sizes[self.env.true_state]}"
                )
        else:
            # Social Learning Environment
            print(f"True state is {self.env.true_state}")

    def _set_agents_eval_mode(self):
        """Set all agents to evaluation mode."""
        for agent in self.agents.values():
            agent.set_eval_mode()
        print(f"Set {len(self.agents)} agents to evaluation mode")

    def _aggregate_episode_results(self, episodic_metrics: Dict) -> Dict[str, Any]:
        """Aggregate results across multiple episodes."""
        if not episodic_metrics["episodes"]:
            return {}

        # Get the structure from the first episode
        first_episode = episodic_metrics["episodes"][0]
        aggregated = {}

        # Aggregate different types of metrics
        for key, value in first_episode.items():
            if key == "episode_time":
                # Average episode time
                times = [
                    ep.get("episode_time", 0) for ep in episodic_metrics["episodes"]
                ]
                aggregated[key] = np.mean(times)
            elif key == "total_rewards":
                # Average total rewards per agent
                all_rewards = [
                    ep.get("total_rewards", {}) for ep in episodic_metrics["episodes"]
                ]
                if all_rewards and all_rewards[0]:
                    aggregated[key] = {}
                    for agent_id in all_rewards[0].keys():
                        rewards = [
                            ep_rewards.get(agent_id, 0) for ep_rewards in all_rewards
                        ]
                        aggregated[key][agent_id] = np.mean(rewards)
            elif isinstance(value, dict) and all(
                isinstance(v, list) for v in value.values()
            ):
                # Metrics with agent-specific lists (like belief_distributions)
                aggregated[key] = {}
                for agent_id in value.keys():
                    # Concatenate lists across episodes
                    agent_data = []
                    for ep in episodic_metrics["episodes"]:
                        if key in ep and agent_id in ep[key]:
                            agent_data.extend(ep[key][agent_id])
                    aggregated[key][agent_id] = agent_data
            elif isinstance(value, list):
                # Simple lists - concatenate across episodes
                aggregated_list = []
                for ep in episodic_metrics["episodes"]:
                    if key in ep:
                        aggregated_list.extend(ep[key])
                aggregated[key] = aggregated_list

        return aggregated

    def _calculate_evaluation_summary(
        self, episodic_metrics: Dict, learning_rates: Dict
    ) -> Dict[str, Any]:
        """Calculate summary statistics for the evaluation."""
        summary = {
            "num_episodes": len(episodic_metrics["episodes"]),
            "learning_rates": learning_rates,
        }

        # Calculate reward statistics
        if (
            episodic_metrics["episodes"]
            and "total_rewards" in episodic_metrics["episodes"][0]
        ):
            reward_stats = {}
            for agent_id in range(self.env.num_agents):
                rewards = [
                    ep["total_rewards"].get(agent_id, 0)
                    for ep in episodic_metrics["episodes"]
                    if "total_rewards" in ep
                ]
                if rewards:
                    reward_stats[agent_id] = {
                        "mean": np.mean(rewards),
                        "std": np.std(rewards),
                        "min": np.min(rewards),
                        "max": np.max(rewards),
                    }
            summary["reward_statistics"] = reward_stats

        # Calculate action accuracy if we have true states and actions
        if (
            episodic_metrics["episodes"]
            and any("true_states" in ep for ep in episodic_metrics["episodes"])
            and any("agent_actions" in ep for ep in episodic_metrics["episodes"])
        ):

            accuracy_stats = {}
            for agent_id in range(self.env.num_agents):
                correct_actions = 0
                total_actions = 0

                for ep in episodic_metrics["episodes"]:
                    if "true_states" in ep and "agent_actions" in ep:
                        true_states = ep["true_states"]
                        agent_actions = ep["agent_actions"].get(agent_id, [])

                        for i, (true_state, action) in enumerate(
                            zip(true_states, agent_actions)
                        ):
                            if action == true_state:
                                correct_actions += 1
                            total_actions += 1

                if total_actions > 0:
                    accuracy_stats[agent_id] = correct_actions / total_actions

            summary["action_accuracy"] = accuracy_stats

        return summary

    def quick_evaluate(self, num_steps: int = 100) -> Dict[str, float]:
        """
        Perform a quick evaluation for basic performance metrics.

        Args:
            num_steps: Number of steps for quick evaluation

        Returns:
            Dictionary with basic performance metrics
        """
        # Set agents to eval mode
        self._set_agents_eval_mode()

        # Initialize
        observations = self.env.initialize()
        total_rewards = np.zeros(self.env.num_agents)
        correct_actions = np.zeros(self.env.num_agents)

        print(f"Quick evaluation for {num_steps} steps...")

        for step in range(num_steps):
            # Get actions
            actions, _ = select_agent_actions(self.agents, {})

            # Environment step
            next_observations, rewards, done, info = self.env.step(actions, {})

            # Update rewards
            if rewards:
                for agent_id, reward in rewards.items():
                    if isinstance(reward, dict):
                        total_rewards[agent_id] += reward["total"]
                    else:
                        total_rewards[agent_id] += reward

            # Check action correctness if true state is available
            if hasattr(self.env, "true_state"):
                for agent_id, action in actions.items():
                    if action == self.env.true_state:
                        correct_actions[agent_id] += 1

            observations = next_observations

            if done:
                break

        # Calculate metrics
        results = {
            "average_reward": float(np.mean(total_rewards)),
            "total_rewards": {
                i: float(total_rewards[i]) for i in range(self.env.num_agents)
            },
        }

        if hasattr(self.env, "true_state"):
            results["action_accuracy"] = {
                i: float(correct_actions[i] / num_steps)
                for i in range(self.env.num_agents)
            }
            results["average_accuracy"] = float(np.mean(correct_actions / num_steps))

        return results
