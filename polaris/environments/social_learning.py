from typing import Dict, List, Optional, Tuple

import numpy as np

from ..environments.base import BaseEnvironment


class SocialLearningEnvironment(BaseEnvironment):
    """
    Multi-agent environment for social learning based on Brandl 2024.

    This environment models agents with partial observability learning
    the true state through private signals and observations of other agents' actions.

    This implementation uses a continuous stream of data without episode structure.
    """

    def __init__(
        self,
        num_agents: int = 10,
        num_states: int = 2,
        signal_accuracy: float = 0.75,
        network_type: str = "complete",
        network_params: Dict = None,
        horizon: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Initialize the social learning environment.

        Args:
            num_agents: Number of agents in the environment
            num_states: Number of possible states of the world
            signal_accuracy: Probability that a signal matches the true state
            network_type: Type of network structure ('complete', 'ring', 'star', 'random')
            network_params: Parameters for network generation if network_type is 'random'
            horizon: Total number of steps to run
            seed: Random seed for reproducibility
        """
        super().__init__(
            num_agents=num_agents,
            num_states=num_states,
            network_type=network_type,
            network_params=network_params,
            horizon=horizon,
            seed=seed,
        )

        self.signal_accuracy = signal_accuracy
        self.num_actions = num_states  # Actions correspond to states

        # Initialize additional state variables
        self.actions = np.zeros(self.num_agents, dtype=np.int32)
        self.signals = np.zeros(self.num_agents, dtype=np.int32)

        # Track metrics
        self.correct_actions = np.zeros(self.num_agents, dtype=np.int32)
        self.mistake_history = []
        self.incorrect_prob_history = []

    def _generate_signal(self, agent_id: int) -> int:
        """
        Generate a private signal for an agent based on the true state.
        """
        if self.rng.random() < self.signal_accuracy:
            # Signal matches the true state with probability signal_accuracy
            signal = self.true_state
        else:
            # Generate a random incorrect signal
            incorrect_states = [
                s for s in range(self.num_states) if s != self.true_state
            ]
            signal = self.rng.choice(incorrect_states)

        return signal

    def _compute_reward(self, agent_id: int, action: int) -> float:
        """
        Compute the reward for an agent's action using derived reward function.
        """
        signal = self.signals[agent_id]
        q = self.signal_accuracy

        # For binary case with signal accuracy q
        if action == signal:  # Action matches signal
            reward = q / (2 * q - 1)
        else:  # Action doesn't match signal
            reward = -(1 - q) / (2 * q - 1)

        return reward

    def initialize(self) -> Dict[int, Dict]:
        """
        Initialize the environment state.

        Returns:
            observations: Dictionary of initial observations for each agent
        """
        # Sample a new true state
        self.true_state = self.rng.randint(0, self.num_states)

        # Reset step counter
        self.current_step = 0

        # Reset actions
        self.actions = np.zeros(self.num_agents, dtype=np.int32)

        # Generate initial signals for all agents
        self.signals = np.array(
            [self._generate_signal(i) for i in range(self.num_agents)]
        )

        # Reset metrics
        self.correct_actions = np.zeros(self.num_agents, dtype=np.int32)
        self.mistake_history = []
        self.incorrect_prob_history = []  # Track incorrect probability assignments

        # Create initial observations for each agent
        observations = {}
        for agent_id in range(self.num_agents):
            # Initially, agents only observe their private signal
            observations[agent_id] = {
                "signal": self.signals[agent_id],
                "neighbor_actions": None,  # No actions observed yet
            }

        return observations

    def step(
        self, actions: Dict[int, int], action_probs: Dict[int, np.ndarray] = None
    ) -> Tuple[Dict[int, Dict], Dict[int, float], bool, Dict]:
        """
        Take a step in the environment given the actions of all agents.

        Args:
            actions: Dictionary mapping agent IDs to their chosen actions
            action_probs: Dictionary mapping agent IDs to their action probability distributions
        """
        # Update step counter
        self.current_step += 1

        # Update actions
        for agent_id, action in actions.items():
            self.actions[agent_id] = action

        # Generate new signals for all agents
        self.signals = np.array(
            [self._generate_signal(i) for i in range(self.num_agents)]
        )

        # Compute rewards and track correct actions
        rewards = {}
        for agent_id in range(self.num_agents):
            rewards[agent_id] = self._compute_reward(agent_id, actions[agent_id])

            # Track correct actions for metrics
            if actions[agent_id] == self.true_state:
                self.correct_actions[agent_id] += 1

        # Calculate mistake rate for this step (binary)
        mistake_rate = 1.0 - np.mean(
            [1.0 if a == self.true_state else 0.0 for a in self.actions]
        )
        self.mistake_history.append(mistake_rate)

        # Calculate incorrect probability assignment rate
        # For each agent, get the probability they assigned to incorrect states
        incorrect_probs = []
        for agent_id in range(self.num_agents):
            if agent_id in action_probs:
                # Sum probabilities assigned to all incorrect states
                incorrect_prob = 1.0 - action_probs[agent_id][self.true_state]
                incorrect_probs.append(incorrect_prob)
            else:
                # If we don't have probabilities for this agent, use 0.5 as a default
                print(f"Warning: No action probabilities for agent {agent_id}")
                incorrect_probs.append(0.5)

        # Average incorrect probability across all agents
        avg_incorrect_prob = np.mean(incorrect_probs)
        self.incorrect_prob_history.append(incorrect_probs.copy())

        # Create observations for each agent
        observations = {}
        for agent_id in range(self.num_agents):
            # Get actions of neighbors that this agent can observe
            neighbor_actions = {}
            for neighbor_id in range(self.num_agents):
                if self.network[agent_id, neighbor_id] == 1:
                    neighbor_actions[neighbor_id] = self.actions[neighbor_id]

            observations[agent_id] = {
                "signal": self.signals[agent_id],
                "neighbor_actions": neighbor_actions,
            }

        # Check if we've reached the total number of steps
        done = self.current_step >= self.horizon

        # Additional information
        info = {
            "true_state": self.true_state,
            "mistake_rate": mistake_rate,
            "incorrect_prob": (
                self.incorrect_prob_history[-1].copy()
                if isinstance(self.incorrect_prob_history[-1], list)
                else self.incorrect_prob_history[-1]
            ),
            "correct_actions": self.correct_actions.copy(),
        }

        return observations, rewards, done, info

    def get_autarky_rate(self) -> float:
        """
        Compute the theoretical learning rate for a single agent in isolation (autarky).

        For binary state with accuracy q: r_aut = -(1/t) log P(error) = -log(1-q)
        """
        if self.num_states == 2:
            return -np.log(1 - self.signal_accuracy)
        else:
            # For multi-state case
            p_correct = self.signal_accuracy
            p_error = (1 - p_correct) / (self.num_states - 1)
            return -np.log(p_error)

    def get_bound_rate(self) -> float:
        """
        Compute the theoretical upper bound on any agent's learning rate.

        For binary state with accuracy q: r_bdd = -(1/t) log P(error) = -(log(q) + log(1-q))
        """
        if self.num_states == 2:
            return -(np.log(self.signal_accuracy) + np.log(1 - self.signal_accuracy))
        else:
            # For multi-state case (Jeffreys divergence)
            p_correct = self.signal_accuracy
            p_error = (1 - p_correct) / (self.num_states - 1)
            return -(
                p_correct * np.log(p_error / p_correct)
                + p_error * np.log(p_correct / p_error)
            )

    def get_coordination_rate(self) -> float:
        """
        Compute the theoretical learning rate achievable with coordination.

        For binary state with accuracy q: r_crd = -(1/t) log P(error) = -log(q)
        """
        if self.num_states == 2:
            return -np.log(self.signal_accuracy)
        else:
            # For multi-state case (KL divergence)
            p_correct = self.signal_accuracy
            p_error = (1 - p_correct) / (self.num_states - 1)
            return -p_correct * np.log(p_error / p_correct)
