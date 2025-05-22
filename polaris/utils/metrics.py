"""
Metrics collection and processing for POLARIS experiments.
"""

import json
import numpy as np
import torch
from pathlib import Path
from polaris.utils.utils import calculate_learning_rate


def calculate_policy_kl_divergence(policy_mean, policy_std, mpe_allocation):
    """
    Calculate KL divergence between the agent's policy distribution and the MPE allocation.
    
    For continuous actions, we treat the MPE allocation as a Dirac delta (deterministic policy)
    and the agent's policy as a truncated Gaussian distribution.
    
    Args:
        policy_mean: Mean of the agent's policy distribution
        policy_std: Standard deviation of the agent's policy distribution
        mpe_allocation: Theoretical MPE allocation (deterministic)
        
    Returns:
        kl_divergence: KL divergence between distributions
    """
    # Convert inputs to tensors if they aren't already
    if not isinstance(policy_mean, torch.Tensor):
        policy_mean = torch.tensor(policy_mean)
    if not isinstance(policy_std, torch.Tensor):
        policy_std = torch.tensor(policy_std)
    if not isinstance(mpe_allocation, torch.Tensor):
        mpe_allocation = torch.tensor(mpe_allocation)
    
    # For continuous action space with truncated Gaussian policy and deterministic target,
    # the KL divergence can be approximated as:
    # KL(p||δ) = -log(pdf(δ|μ,σ)) where pdf is the probability density function
    
    # Calculate the negative log probability of the MPE allocation under the policy distribution
    z_score = (mpe_allocation - policy_mean) / policy_std
    log_pdf = -0.5 * (z_score ** 2) - torch.log(policy_std) - 0.5 * torch.log(torch.tensor(2 * np.pi))
    kl_divergence = -log_pdf
    
    return kl_divergence.item()


def initialize_metrics(env, args, training):
    """Initialize metrics dictionary for tracking experiment results."""
    metrics = {
        'mistake_rates': [],
        'incorrect_probs': [],
        'action_probs': {agent_id: [] for agent_id in range(env.num_agents)},
        'full_action_probs': {agent_id: [] for agent_id in range(env.num_agents)},
        'true_states': [],
        'agent_actions': {agent_id: [] for agent_id in range(env.num_agents)},
        'allocations': {agent_id: [] for agent_id in range(env.num_agents)}
    }
    
    # Add training-specific or evaluation-specific metrics
    if training:
        metrics['training_loss'] = []
        # Add belief and latent state tracking for training if plot_internal_states is enabled
        if hasattr(args, 'plot_internal_states') and args.plot_internal_states:
            metrics['belief_states'] = {agent_id: [] for agent_id in range(env.num_agents)}
            metrics['latent_states'] = {agent_id: [] for agent_id in range(env.num_agents)}
            metrics['belief_distributions'] = {agent_id: [] for agent_id in range(env.num_agents)}
            metrics['opponent_belief_distributions'] = {agent_id: [] for agent_id in range(env.num_agents)}
    else:
        metrics['correct_actions'] = {agent_id: 0 for agent_id in range(env.num_agents)}
        # Add belief and latent state tracking for evaluation
        metrics['belief_states'] = {agent_id: [] for agent_id in range(env.num_agents)}
        metrics['latent_states'] = {agent_id: [] for agent_id in range(env.num_agents)}
        # Add belief distribution tracking
        metrics['belief_distributions'] = {agent_id: [] for agent_id in range(env.num_agents)}
        metrics['opponent_belief_distributions'] = {agent_id: [] for agent_id in range(env.num_agents)}
    
    # Add strategic experimentation specific metrics if applicable
    if hasattr(env, 'safe_payoff'):   
        # Add allocation tracking for continuous actions
        if hasattr(args, 'continuous_actions') and args.continuous_actions:
            metrics['allocations'] = {agent_id: [] for agent_id in range(env.num_agents)}
            metrics["mpe_allocations"] = {agent_id: [] for agent_id in range(env.num_agents)}
            # Add KL divergence tracking for policy convergence
            metrics['policy_kl_divergence'] = {agent_id: [] for agent_id in range(env.num_agents)}
            metrics['policy_means'] = {agent_id: [] for agent_id in range(env.num_agents)}
            metrics['policy_stds'] = {agent_id: [] for agent_id in range(env.num_agents)}
    
    metrics['belief_distributions'] = {agent_id: [] for agent_id in range(env.num_agents)}
    metrics['agent_beliefs'] = {agent_id: [] for agent_id in range(env.num_agents)}
    print(f"Initialized metrics dictionary with {len(metrics['action_probs'])} agent entries")
    return metrics


def update_metrics(metrics, info, actions, action_probs=None, beliefs=None, latent_states=None, opponent_beliefs=None):
    """Update metrics dictionary with the current step information."""
    # Update true state history
    if 'true_state' in info:
        metrics['true_states'].append(info['true_state'])
    
    # Update mistake rates and incorrect probabilities
    if 'mistake_rate' in info:
        metrics['mistake_rates'].append(info['mistake_rate'])
    if 'incorrect_prob' in info:
        metrics['incorrect_probs'].append(info['incorrect_prob'])
    
    # Update action probabilities
    if action_probs is not None:
        for agent_id, probs in action_probs.items():
            metrics['full_action_probs'][agent_id].append(probs)
    
    # Update action history
    for agent_id, action in actions.items():
        metrics['agent_actions'][agent_id].append(action)
    
    # Update allocations for strategic experimentation if available
    if 'allocations' in info and 'allocations' in metrics:
        # Handle different types of allocations data
        allocations = info['allocations']
        if isinstance(allocations, dict):
            # Dictionary format
            for agent_id, allocation in allocations.items():
                if agent_id in metrics['allocations']:
                    metrics['allocations'][agent_id].append(allocation)
        elif isinstance(allocations, (list, np.ndarray)):
            # List or array format
            for agent_id in range(len(allocations)):
                if agent_id in metrics['allocations']:
                    metrics['allocations'][agent_id].append(allocations[agent_id])
        else:
            print(f"Warning: Unsupported allocations format: {type(allocations)}")
    
    # Update policy means and stds if available
    if 'policy_means' in info and 'policy_means' in metrics:
        policy_means = info['policy_means']
        for agent_id, mean in enumerate(policy_means):
            if agent_id in metrics['policy_means']:
                metrics['policy_means'][agent_id].append(mean)
    
    if 'policy_stds' in info and 'policy_stds' in metrics:
        policy_stds = info['policy_stds']
        for agent_id, std in enumerate(policy_stds):
            if agent_id in metrics['policy_stds']:
                metrics['policy_stds'][agent_id].append(std)
    
    # Update KL divergence based on dynamic MPE allocation
    if 'policy_means' in metrics and 'policy_stds' in metrics:
        # Get policy parameters if available
        if 'policy_means' in info and 'policy_stds' in info:
            for agent_id in metrics['policy_kl_divergence']:
                if agent_id < len(info['policy_means']) and agent_id < len(info['policy_stds']):
                    # Determine current belief for this agent
                    agent_belief = 0.5  # Default: neutral belief
                    
                    # Try to get agent beliefs from info
                    if 'agent_beliefs' in info and agent_id in info['agent_beliefs']:
                        agent_belief = info['agent_beliefs'][agent_id]

                    # Get environment parameters from info if available
                    env_params = info.get('env_params', {})
                    safe_payoff = env_params.get('safe_payoff')
                    
                    # If no drift rates provided, use defaults
                    drift_rates = env_params.get('drift_rates')
                    jump_rates = env_params.get('jump_rates')
                    jump_sizes = env_params.get('jump_sizes')
                    background_informativeness = env_params.get('background_informativeness')
                    num_agents = env_params.get('num_agents')
                    true_state = env_params.get('true_state')

                    # Calculate dynamic MPE based on current belief
                    mpe_allocation = calculate_dynamic_mpe(
                        true_state,
                        agent_belief, 
                        safe_payoff,
                        drift_rates,
                        jump_rates,
                        jump_sizes,
                        background_informativeness,
                        num_agents
                    )
                    
                    # Calculate KL divergence between agent's policy and MPE
                    kl = calculate_policy_kl_divergence(
                        info['policy_means'][agent_id], 
                        info['policy_stds'][agent_id],
                        mpe_allocation
                    )
                    metrics["mpe_allocations"][agent_id].append(mpe_allocation)
                    metrics['policy_kl_divergence'][agent_id].append(kl)
                    metrics['agent_beliefs'][agent_id].append(agent_belief)
                
    # Update belief states if requested and available
    if beliefs is not None and 'belief_states' in metrics:
        for agent_id, belief in beliefs.items():
            metrics['belief_states'][agent_id].append(belief)
    
    # Update latent states if requested and available
    if latent_states is not None and 'latent_states' in metrics:
        for agent_id, latent in latent_states.items():
            metrics['latent_states'][agent_id].append(latent)
            
    # Update opponent beliefs if requested and available
    if opponent_beliefs is not None and 'opponent_belief_distributions' in metrics:
        for agent_id, opponent_belief in opponent_beliefs.items():
            metrics['opponent_belief_distributions'][agent_id].append(opponent_belief)
    
    return metrics


def calculate_dynamic_mpe(true_state, belief, safe_payoff, drift_rates, jump_rates, jump_sizes, background_informativeness, num_agents):
    """
    Calculate the Markov perfect equilibrium allocation dynamically based on current belief.
    
    This follows the Keller and Rady (2020) model with symmetric MPE.
    
    Args:
        belief: Agent's belief probability of being in the good state (state 1)
        safe_payoff: Deterministic payoff of the safe arm
        drift_rates: Drift rates for bad and good states
        jump_rates: Poisson jump rates for bad and good states
        jump_sizes: Jump sizes for bad and good states
        background_informativeness: Informativeness of background signals
        num_agents: Number of agents in the game
        
    Returns:
        mpe_allocation: The MPE allocation for the given belief
    """
    # Compute expected risky payoff based on current belief
    expected_risky_payoff = (
        belief * (drift_rates[1] + jump_rates[1] * jump_sizes[1]) +
        (1 - belief) * (drift_rates[0] + jump_rates[0] * jump_sizes[0])
    )
    
    
    # Full information payoff (value when state is known to be good)
    full_info_payoff = belief * max(safe_payoff, drift_rates[1] + jump_rates[1] * jump_sizes[1]) + (1 - belief) * max(safe_payoff, drift_rates[0] + jump_rates[0] * jump_sizes[0])
    

    # Incentive defined in the Keller and Rady paper
    incentive = (full_info_payoff - safe_payoff) / (safe_payoff - expected_risky_payoff)
    # Adjust for number of players and background signal
    k0 = background_informativeness
    n = num_agents

    if incentive <= k0:
        return 0.0  # No experimentation
    
    elif k0 < incentive < k0 + n - 1:
        # Partial experimentation
        return (incentive - k0) / (n - 1)
    else:
        return 1.0  # Full experimentation


def store_incorrect_probabilities(metrics, info, num_agents):
    """Store incorrect action probabilities in metrics."""
    if 'incorrect_prob' in info:
        incorrect_prob = info['incorrect_prob']
        
        # Handle both list and scalar incorrect probabilities
        if isinstance(incorrect_prob, list):
            metrics['incorrect_probs'].append(incorrect_prob)
            
            # Also store in per-agent metrics
            for agent_id, prob in enumerate(incorrect_prob):
                if agent_id < num_agents:
                    metrics['action_probs'][agent_id].append(prob)
        else:
            # If we only have a scalar, store it and duplicate for all agents
            metrics['incorrect_probs'].append(incorrect_prob)
            for agent_id in range(num_agents):
                metrics['action_probs'][agent_id].append(incorrect_prob)


def calculate_agent_learning_rates_from_metrics(metrics):
    """Calculate learning rates for each agent from metrics."""
    learning_rates = {}
    
    # Calculate learning rate for each agent
    for agent_id, probs in metrics['action_probs'].items():
        if len(probs) > 0:
            learning_rates[agent_id] = calculate_learning_rate(probs)
        else:
            learning_rates[agent_id] = 0.0
    return learning_rates


def prepare_serializable_metrics(metrics, learning_rates, theoretical_bounds, num_steps, training):
    """Prepare metrics for JSON serialization."""
    # Find fastest and slowest learning agents
    fastest_agent = max(learning_rates.items(), key=lambda x: x[1])
    slowest_agent = min(learning_rates.items(), key=lambda x: x[1])
    avg_learning_rate = np.mean(list(learning_rates.values()))
    
    serializable_metrics = {
        'total_steps': num_steps,
        'mistake_rates': [float(m) for m in metrics['mistake_rates']],
        'incorrect_probs': [[float(p) for p in agent_probs] if isinstance(agent_probs, list) else float(agent_probs) 
                           for agent_probs in metrics['incorrect_probs']],
        'action_probs': {str(agent_id): [float(p) for p in probs] 
                         for agent_id, probs in metrics['action_probs'].items()},
        'full_action_probs': {str(agent_id): [[float(p) for p in dist] for dist in probs] 
                              for agent_id, probs in metrics['full_action_probs'].items()},
        'true_states': metrics['true_states'],
        'agent_actions': {str(agent_id): [int(a) if isinstance(a, (int, np.integer)) else float(a) for a in actions] 
                         for agent_id, actions in metrics['agent_actions'].items()},
        'learning_rates': {str(k): float(v) for k, v in learning_rates.items()},
        
        # Add belief and latent states if they exist
        'has_belief_states': 'belief_states' in metrics,
        'has_latent_states': 'latent_states' in metrics,
        'has_belief_distributions': 'belief_distributions' in metrics,
        'fastest_agent': {
            'id': int(fastest_agent[0]),
            'rate': float(fastest_agent[1])
        },
        'slowest_agent': {
            'id': int(slowest_agent[0]),
            'rate': float(slowest_agent[1])
        },
        'avg_learning_rate': float(avg_learning_rate),
    }
    
    # Add KL divergence data if available
    if 'policy_kl_divergence' in metrics:
        serializable_metrics['policy_kl_divergence'] = {
            str(agent_id): [float(kl) for kl in kl_values] 
            for agent_id, kl_values in metrics['policy_kl_divergence'].items()
        }
        
        # Calculate average KL divergence per agent (over last 20% of steps)
        final_kl_divergences = {}
        for agent_id, kl_values in metrics['policy_kl_divergence'].items():
            if len(kl_values) > 0:
                # Use the last 20% of values to calculate the average
                last_idx = max(1, int(len(kl_values) * 0.2))
                final_kl = np.mean(kl_values[-last_idx:])
                final_kl_divergences[str(agent_id)] = float(final_kl)
        
        if final_kl_divergences:
            serializable_metrics['final_kl_divergence'] = final_kl_divergences
            serializable_metrics['avg_final_kl_divergence'] = float(np.mean(list(final_kl_divergences.values())))
    
    # Add allocations if they exist (for Strategic Experimentation)
    if 'allocations' in metrics:
        serializable_metrics['allocations'] = {
            str(agent_id): [float(a) for a in allocs] 
            for agent_id, allocs in metrics['allocations'].items()
        }
        
        # Calculate average allocation per agent (over last 20% of steps)
        final_allocations = {}
        for agent_id, allocs in metrics['allocations'].items():
            if len(allocs) > 0:
                # Use the last 20% of values to calculate the average
                last_idx = max(1, int(len(allocs) * 0.2))
                final_alloc = np.mean(allocs[-last_idx:])
                final_allocations[str(agent_id)] = float(final_alloc)
        
        if final_allocations:
            serializable_metrics['final_allocations'] = final_allocations
            serializable_metrics['avg_final_allocation'] = float(np.mean(list(final_allocations.values())))
    
    # Add theoretical bounds based on environment type
    if 'autarky_rate' in theoretical_bounds:
        # Social Learning Environment
        serializable_metrics['theoretical_bounds'] = {
            'autarky_rate': float(theoretical_bounds['autarky_rate']),
            'coordination_rate': float(theoretical_bounds['coordination_rate']),
            'bound_rate': float(theoretical_bounds['bound_rate'])
        }
    elif 'mpe_neutral' in theoretical_bounds:
        # Strategic Experimentation Environment
        serializable_metrics['theoretical_bounds'] = {
            'mpe_neutral': float(theoretical_bounds['mpe_neutral']),
            'mpe_good_state': float(theoretical_bounds['mpe_good_state']),
            'mpe_bad_state': float(theoretical_bounds['mpe_bad_state'])
        }
    else:
        # Default empty bounds
        serializable_metrics['theoretical_bounds'] = {}
    
    return serializable_metrics


def make_json_serializable(obj):
    """
    Convert a Python object to a JSON serializable format.
    
    Args:
        obj: Any Python object
        
    Returns:
        JSON serializable version of the object
    """
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {make_json_serializable(key): make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, set):
        return list(make_json_serializable(item) for item in obj)
    elif hasattr(obj, 'tolist'):  # For torch tensors
        return make_json_serializable(obj.tolist())
    elif hasattr(obj, 'item'):  # For scalar tensors
        return obj.item()
    else:
        return obj


def save_metrics_to_file(metrics, output_dir, training, filename=None):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics to save
        output_dir: Directory to save the file in
        training: Whether this is training or evaluation
        filename: Optional custom filename (if None, uses default naming)
    """
    if filename is None:
        filename = f'training_metrics.json' if training else 'evaluation_metrics.json'
    
    # Convert metrics to JSON serializable format
    serializable_metrics = make_json_serializable(metrics)
    
    metrics_file = output_dir / filename
    with open(metrics_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)


def process_incorrect_probabilities(metrics, num_agents):
    """Process incorrect probabilities for plotting."""
    agent_incorrect_probs = metrics['action_probs']
    
    # If action_probs is empty, try to process from incorrect_probs as fallback
    if not agent_incorrect_probs:
        print("Warning: Using fallback method to process incorrect probabilities")
        agent_incorrect_probs = {}
        for step_idx, step_probs in enumerate(metrics['incorrect_probs']):
            if isinstance(step_probs, list):
                # If we have per-agent probabilities
                for agent_id, prob in enumerate(step_probs):
                    if agent_id not in agent_incorrect_probs:
                        agent_incorrect_probs[agent_id] = []
                    agent_incorrect_probs[agent_id].append(prob)
            else:
                # If we only have an average probability
                for agent_id in range(num_agents):
                    if agent_id not in agent_incorrect_probs:
                        agent_incorrect_probs[agent_id] = []
                    agent_incorrect_probs[agent_id].append(step_probs)
    
    return agent_incorrect_probs
