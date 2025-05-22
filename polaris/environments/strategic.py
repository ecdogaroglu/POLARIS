import numpy as np
from typing import Dict, List, Tuple, Optional, Union

from polaris.environments.base import BaseEnvironment

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
        
        # Set default parameters 
        self.drift_rates = drift_rates
        self.diffusion_sigma = diffusion_sigma
        self.jump_rates = jump_rates
        self.jump_sizes = jump_sizes

        self.background_informativeness = background_informativeness
        
        # Define action space: continuous allocation [0,1]
        self.num_actions = None  # Continuous action space
        self.action_space_type = "continuous"
        self.action_low = 0.0
        self.action_high = 1.0
        
        # Initialize state variables
        self.allocations = np.ones(self.num_agents) / self.num_states
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
        self.allocations = np.ones(self.num_agents) / self.num_states
        
        # Reset background signal
        self.last_background_signal = 0.0
        self.background_signal_history = [0.0]
        self.last_background_increment = 0.0
        
        # Reset payoff histories
        self.payoff_histories = [[] for _ in range(self.num_agents)]
        
        
        # Create initial observations for each agent
        observations = {}
        for agent_id in range(self.num_agents):
            observations[agent_id] = {
                'background_signal': 0.0,
                'background_increment': 0.0,
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
        # Save the increment for agent observations
        self.last_background_increment = background_increment
        
        # Compute rewards
        rewards = {}
        payoffs = {}
        for agent_id in range(self.num_agents):
            reward_info = self._compute_reward(agent_id, self.allocations[agent_id])
            rewards[agent_id] = reward_info
            payoffs[agent_id] = reward_info['total']
            
            # Store payoff history
            self.payoff_histories[agent_id].append(reward_info['total'])

        
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
                'background_increment': self.last_background_increment,
                'background_history': self.background_signal_history.copy(),
                'neighbor_allocations': neighbor_allocations,
                'neighbor_payoffs': neighbor_payoffs,
                'own_payoff_history': own_payoff_history,
            }
        
        # Check if we've reached the total number of steps
        done = self.current_step >= self.horizon
        
        # Calculate metrics for info
        avg_allocation = np.mean(self.allocations)
        total_allocation = np.sum(self.allocations)
        
        # Additional information
        info = {
            'true_state': self.true_state,
            'avg_allocation': avg_allocation,
            'total_allocation': total_allocation,
            'allocations': self.allocations.copy(),
            'background_signal': self.last_background_signal,
            'background_increment': self.last_background_increment,
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