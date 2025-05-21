import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod

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
        seed: Optional[int] = None
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
        
    def _create_network(self, network_type: str, network_params: Dict = None) -> np.ndarray:
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
    def step(self, actions: Dict[int, int], action_probs: Dict[int, np.ndarray] = None) -> Tuple[Dict[int, Dict], Dict[int, float], bool, Dict]:
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
        seed: Optional[int] = None
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
            seed=seed
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
            incorrect_states = [s for s in range(self.num_states) if s != self.true_state]
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
            reward = q / (2*q - 1)
        else:  # Action doesn't match signal
            reward = -(1 - q) / (2*q - 1)
        
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
        self.signals = np.array([self._generate_signal(i) for i in range(self.num_agents)])
        
        # Reset metrics
        self.correct_actions = np.zeros(self.num_agents, dtype=np.int32)
        self.mistake_history = []
        self.incorrect_prob_history = []  # Track incorrect probability assignments
        
        # Create initial observations for each agent
        observations = {}
        for agent_id in range(self.num_agents):
            # Initially, agents only observe their private signal
            observations[agent_id] = {
                'signal': self.signals[agent_id],
                'neighbor_actions': None,  # No actions observed yet
            }
        
        return observations
    
    def step(self, actions: Dict[int, int], action_probs: Dict[int, np.ndarray] = None) -> Tuple[Dict[int, Dict], Dict[int, float], bool, Dict]:
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
        self.signals = np.array([self._generate_signal(i) for i in range(self.num_agents)])
        
        # Compute rewards and track correct actions
        rewards = {}
        for agent_id in range(self.num_agents):
            rewards[agent_id] = self._compute_reward(agent_id, actions[agent_id])
            
            # Track correct actions for metrics
            if actions[agent_id] == self.true_state:
                self.correct_actions[agent_id] += 1
        
        # Calculate mistake rate for this step (binary)
        mistake_rate = 1.0 - np.mean([1.0 if a == self.true_state else 0.0 for a in self.actions])
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
                'signal': self.signals[agent_id],
                'neighbor_actions': neighbor_actions,
            }
        
        # Check if we've reached the total number of steps
        done = self.current_step >= self.horizon
        
        # Additional information
        info = {
            'true_state': self.true_state,
            'mistake_rate': mistake_rate,
            'incorrect_prob': self.incorrect_prob_history[-1].copy() if isinstance(self.incorrect_prob_history[-1], list) else self.incorrect_prob_history[-1],
            'correct_actions': self.correct_actions.copy()
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
            return -(p_correct * np.log(p_error/p_correct) + p_error * np.log(p_correct/p_error))
    
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
            return -p_correct * np.log(p_error/p_correct)


class StrategicExperimentationEnvironment(BaseEnvironment):
    """
    Environment for strategic experimentation based on Keller and Rady 2020.
    
    This environment models agents who allocate resources between a safe arm with
    known payoff and a risky arm with unknown state-dependent payoff.
    
    The payoff processes follow Lévy processes that combine diffusion and jumps.
    Agents can observe their own and others' rewards, allowing them to learn about
    the underlying state through experimentation.
    """
    
    def __init__(
        self,
        num_agents: int = 2,
        num_states: int = 2,
        network_type: str = "complete",
        network_params: Dict = None,
        horizon: int = 1000,
        seed: Optional[int] = None,
        safe_payoff: float = 0.0,
        drift_rates: List[float] = None,
        diffusion_sigma: float = 0.5,
        jump_rates: List[float] = None,
        jump_sizes: List[float] = None,
        background_informativeness: float = 0.1,
        time_step: float = 0.1
    ):
        """
        Initialize the strategic experimentation environment.
        
        Args:
            num_agents: Number of agents in the environment
            num_states: Number of possible states of the world
            network_type: Type of network structure ('complete', 'ring', 'star', 'random')
            network_params: Parameters for network generation if network_type is 'random'
            horizon: Total number of steps to run
            seed: Random seed for reproducibility
            safe_payoff: Deterministic payoff of the safe arm
            drift_rates: Drift rates of the risky arm for each state
            diffusion_sigma: Volatility of the diffusion component
            jump_rates: Poisson rates for jumps in each state
            jump_sizes: Expected jump sizes in each state
            background_informativeness: Informativeness of the background signal process
            time_step: Size of time step for discretizing the Lévy processes
        """
        super().__init__(
            num_agents=num_agents,
            num_states=num_states,
            network_type=network_type,
            network_params=network_params,
            horizon=horizon,
            seed=seed
        )
        
        self.safe_payoff = safe_payoff
        self.time_step = time_step
        
        # Set default parameters if not provided
        if drift_rates is None:
            # Default: state 0 has negative drift, state 1 has positive drift
            self.drift_rates = [-0.5, 0.5][:num_states]
        else:
            self.drift_rates = drift_rates[:num_states]
            
        self.diffusion_sigma = diffusion_sigma
        
        if jump_rates is None:
            # Default: higher jump rates in higher states
            self.jump_rates = [0.1 * (i + 1) for i in range(num_states)]
        else:
            self.jump_rates = jump_rates[:num_states]
            
        if jump_sizes is None:
            # Default: constant jump size of 1.0
            self.jump_sizes = [1.0] * num_states
        else:
            self.jump_sizes = jump_sizes[:num_states]
        
        self.background_informativeness = background_informativeness
        
        # Define action space: continuous allocation [0,1]
        self.num_actions = None  # Continuous action space
        self.action_space_type = "continuous"
        self.action_low = 0.0
        self.action_high = 1.0
        
        # Initialize state variables
        self.allocations = np.zeros(self.num_agents)
        self.last_background_signal = 0.0
        self.background_signal_history = []
        self.payoff_histories = [[] for _ in range(self.num_agents)]
        
        # Metrics
        self.correct_actions = np.zeros(self.num_agents)  # High allocation to risky arm in good state
        

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
        
        # Reset allocations
        self.allocations = np.zeros(self.num_agents)
        
        # Reset background signal
        self.last_background_signal = 0.0
        self.background_signal_history = [0.0]
        
        # Reset payoff histories
        self.payoff_histories = [[] for _ in range(self.num_agents)]
        
        # Reset metrics
        self.correct_actions = np.zeros(self.num_agents)
        
        # Create initial observations for each agent
        observations = {}
        for agent_id in range(self.num_agents):
            observations[agent_id] = {
                'background_signal': 0.0,
                'background_history': [0.0],
                'neighbor_allocations': None,  # No allocations observed yet
                'neighbor_payoffs': None,      # No payoffs observed yet
                'own_payoff_history': [],
            }
        
        return observations
    
    def step(self, actions: Dict[int, Union[int, float]], action_probs: Dict[int, np.ndarray] = None) -> Tuple[Dict[int, Dict], Dict[int, Dict], bool, Dict]:
        """
        Take a step in the environment given the allocations of all agents.
        
        Args:
            actions: Dictionary mapping agent IDs to their chosen allocations [0,1]
            action_probs: Dictionary mapping agent IDs to their action probability distributions (not used)
            
        Returns:
            observations: Dictionary of observations for each agent
            rewards: Dictionary of rewards for each agent
            done: Whether the episode is done
            info: Additional information
        """
        # Update step counter
        self.current_step += 1
        
        # Update allocations
        for agent_id, allocation in actions.items():
            # Ensure allocation is in the valid range [0,1]
            self.allocations[agent_id] = np.clip(allocation, 0.0, 1.0)
        
        # Generate new background signal increment
        background_increment = self._generate_background_signal()
        self.last_background_signal += background_increment
        self.background_signal_history.append(self.last_background_signal)
        
        # Compute rewards
        rewards = {}
        payoffs = {}
        for agent_id in range(self.num_agents):
            reward_info = self._compute_reward(agent_id, self.allocations[agent_id])
            rewards[agent_id] = reward_info
            payoffs[agent_id] = reward_info['total']
            
            # Store payoff history
            self.payoff_histories[agent_id].append(reward_info['total'])
            
            # Track "correct" actions (higher allocation to risky arm in good state)
            if self.true_state > 0:  # Assuming higher states are "good"
                if self.allocations[agent_id] > 0.5:
                    self.correct_actions[agent_id] += 1
            else:  # Lower states are "bad"
                if self.allocations[agent_id] < 0.5:
                    self.correct_actions[agent_id] += 1
        
        # Create observations for each agent
        observations = {}
        for agent_id in range(self.num_agents):
            # Get allocations and payoffs of neighbors that this agent can observe
            neighbor_allocations = {}
            neighbor_payoffs = {}
            for neighbor_id in range(self.num_agents):
                if self.network[agent_id, neighbor_id] == 1:
                    neighbor_allocations[neighbor_id] = self.allocations[neighbor_id]
                    neighbor_payoffs[neighbor_id] = payoffs[neighbor_id]
            
            # Provide own payoff history
            own_payoff_history = self.payoff_histories[agent_id].copy()
            
            observations[agent_id] = {
                'background_signal': self.last_background_signal,
                'background_history': self.background_signal_history.copy(),
                'neighbor_allocations': neighbor_allocations,
                'neighbor_payoffs': neighbor_payoffs,
                'own_payoff_history': own_payoff_history,
            }
        
        # Check if we've reached the total number of steps
        done = self.current_step >= self.horizon
        
        # Calculate metrics for info
        correct_action_rate = np.mean(self.correct_actions) / self.current_step if self.current_step > 0 else 0.0
        avg_allocation = np.mean(self.allocations)
        total_allocation = np.sum(self.allocations)
        
        # Additional information
        info = {
            'true_state': self.true_state,
            'correct_action_rate': correct_action_rate,
            'avg_allocation': avg_allocation,
            'total_allocation': total_allocation,
            'allocations': self.allocations.copy(),
            'background_signal': self.last_background_signal,
        }
        
        return observations, rewards, done, info
    

    def _generate_background_signal(self) -> float:
        """
        Generate background signal increment based on true state using Lévy process.
        
        Returns:
            background_signal_increment: Change in the background signal
        """
        # Drift component based on true state
        drift = self.background_informativeness * self.drift_rates[self.true_state] * self.time_step
        
        # Diffusion component (Brownian motion)
        diffusion = self.diffusion_sigma * np.sqrt(self.time_step) * self.rng.normal()
        
        # Jump component (compound Poisson process)
        # Determine if jump occurs in this time step
        jump_prob = self.jump_rates[self.true_state] * self.time_step
        jump_occurs = self.rng.random() < jump_prob
        jump = self.jump_sizes[self.true_state] if jump_occurs else 0.0
        
        # Total signal increment
        signal_increment = drift + diffusion + jump
        
        return signal_increment
    

    def _generate_risky_payoff(self, agent_id: int, allocation: float) -> float:
        """
        Generate payoff from the risky arm based on true state using Lévy process.
        
        Args:
            agent_id: ID of the agent
            allocation: Allocation to the risky arm [0,1]
            
        Returns:
            risky_payoff: Payoff from the risky arm
        """
        if allocation <= 0:
            return 0.0
            
        # Drift component based on true state
        drift = self.drift_rates[self.true_state] * self.time_step
        
        # Diffusion component (Brownian motion)
        diffusion = self.diffusion_sigma * np.sqrt(self.time_step) * self.rng.normal()
        
        # Jump component (compound Poisson process)
        # Determine if jump occurs in this time step
        jump_prob = self.jump_rates[self.true_state] * self.time_step
        jump_occurs = self.rng.random() < jump_prob
        jump = self.jump_sizes[self.true_state] if jump_occurs else 0.0
        
        # Total payoff scaled by allocation
        payoff = allocation * (drift + diffusion + jump)
        
        return payoff
    

    def _compute_reward(self, agent_id: int, allocation: float) -> Dict:
        """
        Compute rewards for an agent's allocation decision.
        
        Args:
            agent_id: ID of the agent
            allocation: Allocation to the risky arm [0,1]
            
        Returns:
            reward_info: Dictionary with reward components
        """
        # Safe arm gives deterministic payoff based on allocation
        safe_payoff = (1 - allocation) * self.safe_payoff
        
        # Risky arm gives stochastic payoff based on allocation and state
        risky_payoff = self._generate_risky_payoff(agent_id, allocation)
        
        # Total payoff is the sum
        total_payoff = safe_payoff + risky_payoff
        
        # Return detailed information
        return {
            'total': total_payoff,
            'safe': safe_payoff,
            'risky': risky_payoff,
            'allocation': allocation
        }
    

    def get_theoretical_mpe(self, beliefs: List[float]) -> List[float]:
        """
        Calculate the Markov perfect equilibrium allocations for the current game.
        
        Args:
            beliefs: List of agents' beliefs about being in the good state
            
        Returns:
            mpe_allocations: List of MPE allocations for each agent
        """
        # Implementation based on Keller and Rady 2020 with symmetric MPE
        mpe_allocations = []
        for belief in beliefs:
            # Compute incentive to experiment I(b)
            expected_risky_payoff = (
                belief * (self.drift_rates[1] + self.jump_rates[1] * self.jump_sizes[1]) +
                (1 - belief) * (self.drift_rates[0] + self.jump_rates[0] * self.jump_sizes[0])
            )
            
            # Full information payoff
            full_info_payoff = max(self.safe_payoff, self.drift_rates[1] + self.jump_rates[1] * self.jump_sizes[1])
            
            # Incentive defined in the paper
            incentive = (full_info_payoff - self.safe_payoff) / (self.safe_payoff - expected_risky_payoff)
            
            # Adjust for number of players and background signal
            k0 = self.background_informativeness
            n = self.num_agents
            
            if incentive <= k0:
                allocation = 0.0  # No experimentation
            elif k0 < incentive < k0 + n - 1:
                # Partial experimentation
                allocation = (incentive - k0) / (n - 1)
            else:
                allocation = 1.0  # Full experimentation

                
            mpe_allocations.append(allocation)
            
        return mpe_allocations