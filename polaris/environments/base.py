"""
Abstract base class for POLARIS environments.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np


class BaseEnvironment(ABC):
    """
    Abstract base class for social learning environments.

    This class defines the interface for all environment implementations
    and provides common functionality.
    """

    def __init__(
        self,
        num_agents: int,
        num_states: int,
        network_type: str,
        network_params: Dict = None,
        horizon: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Initialize the base environment.

        Args:
            num_agents: Number of agents in the environment
            num_states: Number of possible states of the world
            network_type: Type of network structure ('complete', 'ring', 'star', 'random')
            network_params: Parameters for network generation if network_type is 'random'
            horizon: Total number of steps to run
            seed: Random seed for reproducibility
        """
        self.num_agents = num_agents
        self.num_states = num_states
        self.horizon = horizon

        # Set random seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        # Initialize the network (who observes whom)
        self.network = self._create_network(network_type, network_params)

        # Initialize state and time variables
        self.true_state = None
        self.current_step = 0

    def _create_network(
        self, network_type: str, network_params: Dict = None
    ) -> np.ndarray:
        """
        Create a network structure based on the specified type.
        """
        # Initialize an empty graph
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_agents))

        # Create the specified network structure
        if network_type == "complete":
            # Every agent observes every other agent
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i != j:
                        G.add_edge(i, j)

        elif network_type == "ring":
            # Each agent observes only adjacent agents in a ring
            for i in range(self.num_agents):
                G.add_edge(i, (i + 1) % self.num_agents)
                G.add_edge(i, (i - 1) % self.num_agents)

        elif network_type == "star":
            # Central agent (0) observes all others, others observe only the central agent
            for i in range(1, self.num_agents):
                G.add_edge(0, i)  # Central agent observes periphery
                G.add_edge(i, 0)  # Periphery observes central agent

        elif network_type == "random":
            # Random network with specified density
            density = network_params.get("density", 0.5)
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i != j and self.rng.random() < density:
                        G.add_edge(i, j)

            # Ensure the graph is strongly connected
            if not nx.is_strongly_connected(G):
                components = list(nx.strongly_connected_components(G))
                # Connect all components
                for i in range(len(components) - 1):
                    u = self.rng.choice(list(components[i]))
                    v = self.rng.choice(list(components[i + 1]))
                    G.add_edge(u, v)
                    G.add_edge(v, u)

        else:
            raise ValueError(f"Unknown network type: {network_type}")

        # Convert NetworkX graph to adjacency matrix
        adjacency_matrix = nx.to_numpy_array(G, dtype=np.int32)
        return adjacency_matrix

    @abstractmethod
    def initialize(self) -> Dict[int, Dict]:
        """
        Initialize the environment state.

        Returns:
            observations: Dictionary of initial observations for each agent
        """
        pass

    @abstractmethod
    def step(
        self, actions: Dict[int, int], action_probs: Dict[int, np.ndarray] = None
    ) -> Tuple[Dict[int, Dict], Dict[int, float], bool, Dict]:
        """
        Take a step in the environment given the actions of all agents.

        Args:
            actions: Dictionary mapping agent IDs to their chosen actions
            action_probs: Dictionary mapping agent IDs to their action probability distributions

        Returns:
            observations: Dictionary of observations for each agent
            rewards: Dictionary of rewards for each agent
            done: Whether the episode is done
            info: Additional information
        """
        pass

    def reset(self) -> Dict[int, Dict]:
        """Reset the environment to an initial state."""
        return self.initialize()

    def seed(self, seed: Optional[int] = None) -> None:
        """Set the random seed for the environment."""
        self.rng = np.random.RandomState(seed)

    def get_neighbors(self, agent_id: int) -> List[int]:
        """Get the neighbors of an agent."""
        return [j for j in range(self.num_agents) if self.network[agent_id, j] == 1]
