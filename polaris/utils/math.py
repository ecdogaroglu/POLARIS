"""
Mathematical utilities for POLARIS.
"""

from typing import Dict, List

import numpy as np
import torch


def calculate_learning_rate(mistake_history: List[float]) -> float:
    """
    Calculate the learning rate (rate of decay of mistakes) using log-linear regression.

    Args:
        mistake_history: List of mistake rates over time

    Returns:
        learning_rate: Estimated learning rate
    """
    if len(mistake_history) < 3:  # Need at least 3 points for meaningful regression
        return 0.0

    mistake_history = np.array(mistake_history)
    
    # Handle edge cases
    if np.all(mistake_history == mistake_history[0]):
        # All values are the same (no learning)
        return 0.0
    
    if np.any(mistake_history <= 0):
        # Replace zeros and negative values with a small positive value
        mistake_history = np.clip(mistake_history, 1e-10, 1.0)

    # Time steps
    t = np.arange(len(mistake_history))

    # Log of mistake probability, avoiding log(0)
    log_mistakes = np.log(mistake_history)

    try:
        # Simple linear regression on log-transformed data
        # log(P(mistake)) = -rt + c
        A = np.vstack([t, np.ones_like(t)]).T
        result = np.linalg.lstsq(A, log_mistakes, rcond=None)
        minus_r, c = result[0]

        # Negate slope to get positive learning rate
        learning_rate = -minus_r
        
        # Handle invalid results
        if np.isnan(learning_rate) or np.isinf(learning_rate):
            return 0.0
            
        # For very short sequences, apply a correction factor
        if len(mistake_history) < 10:
            # Scale down the learning rate for short sequences as it's less reliable
            correction_factor = len(mistake_history) / 10.0
            learning_rate *= correction_factor

        return max(0.0, learning_rate)  # Ensure non-negative
        
    except Exception:
        # If regression fails, return 0
        return 0.0


def calculate_agent_learning_rates(
    incorrect_probs: Dict[int, List[List[float]]], min_length: int = 100
) -> Dict[int, float]:
    """
    Calculate learning rates for each agent from their incorrect action probabilities.

    Args:
        incorrect_probs: Dictionary mapping agent IDs to lists of incorrect action
                        probability histories (one list per episode)
        min_length: Minimum number of steps required for calculation

    Returns:
        learning_rates: Dictionary mapping agent IDs to their learning rates
    """
    learning_rates = {}

    for agent_id, prob_histories in incorrect_probs.items():
        # Truncate histories to a common length
        common_length = min(len(hist) for hist in prob_histories)
        if common_length < min_length:
            learning_rates[agent_id] = 0.0
            continue

        # Average across episodes for each time step
        avg_probs = []
        for t in range(common_length):
            avg_prob = np.mean([hist[t] for hist in prob_histories])
            avg_probs.append(avg_prob)

        # Calculate learning rate
        learning_rates[agent_id] = calculate_learning_rate(avg_probs)

    return learning_rates


def calculate_incentive(
    belief,
    safe_payoff,
    drift_rates,
    jump_rates,
    jump_sizes,
):
    """
    Calculate the experimentation incentive I(b) based on current belief.

    This follows the Keller and Rady (2020) model - calculates the value of information.

    Args:
        belief: Agent's belief probability of being in the good state (state 1)
        safe_payoff: Deterministic payoff of the safe arm
        drift_rates: List of drift rates for each state [bad_state, good_state]
        jump_rates: List of jump rates for each state [bad_state, good_state]
        jump_sizes: List of jump sizes for each state [bad_state, good_state]

    Returns:
        incentive: The experimentation incentive I(b) - value of information
    """
    if belief is None or np.isnan(belief):
        return 0.0

    # Calculate payoffs for each state
    bad_state_risky_payoff = drift_rates[0] + jump_rates[0] * jump_sizes[0]
    good_state_risky_payoff = drift_rates[1] + jump_rates[1] * jump_sizes[1]

    # Expected risky payoff based on current belief
    expected_risky_payoff = belief * good_state_risky_payoff + (1 - belief) * bad_state_risky_payoff

    # Expected payoff with current belief (choose best between safe and risky)
    expected_payoff_current_belief = max(safe_payoff, expected_risky_payoff)

    # Expected payoff with full information (knowing the true state)
    # With probability 'belief', state is good: choose max(safe, good_risky)
    # With probability '1-belief', state is bad: choose max(safe, bad_risky)
    expected_payoff_full_info = (
        belief * max(safe_payoff, good_state_risky_payoff) + 
        (1 - belief) * max(safe_payoff, bad_state_risky_payoff)
    )

    # Value of information (incentive to experiment)
    # This should always be non-negative
    incentive = expected_payoff_full_info - expected_payoff_current_belief

    return max(0.0, incentive)  # Ensure non-negative


def calculate_dynamic_mpe(
    true_state,
    belief,
    safe_payoff,
    drift_rates,
    jump_rates,
    jump_sizes,
    background_informativeness,
    num_agents,
):
    """
    Calculate the Markov perfect equilibrium allocation dynamically based on current belief.

    This follows the Keller and Rady (2020) model with symmetric MPE.

    Args:
        true_state: The true state of the world (0 for bad, 1 for good)
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

    # Calculate payoffs for each state
    bad_state_risky_payoff = drift_rates[0] + jump_rates[0] * jump_sizes[0]
    good_state_risky_payoff = drift_rates[1] + jump_rates[1] * jump_sizes[1]

    # Compute expected risky payoff based on current belief
    expected_risky_payoff = belief * good_state_risky_payoff + (1 - belief) * bad_state_risky_payoff

    # Expected payoff with current belief (choose best between safe and risky)
    expected_payoff_current_belief = max(safe_payoff, expected_risky_payoff)

    # Expected payoff with full information (knowing the true state)
    expected_payoff_full_info = (
        belief * max(safe_payoff, good_state_risky_payoff) + 
        (1 - belief) * max(safe_payoff, bad_state_risky_payoff)
    )

    # Value of information (incentive to experiment)
    incentive = expected_payoff_full_info - expected_payoff_current_belief
    incentive = max(0.0, incentive)  # Ensure non-negative

    # Check for invalid incentive
    if np.isnan(incentive) or np.isinf(incentive):
        # Return a reasonable default based on true state
        if true_state == 1:  # Good state
            return 1.0  # Full experimentation in good state
        else:
            return 0.0  # No experimentation in bad state

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

    # For continuous action space with truncated Gaussian policy and deterministic target,
    # the KL divergence can be approximated as:
    # KL(p||δ) = -log(pdf(δ|μ,σ)) where pdf is the probability density function

    # Convert to tensor
    policy_std = torch.tensor(policy_std)

    # Calculate the negative log probability of the MPE allocation under the policy distribution
    # The log probability density function (PDF) of a normal distribution N(μ, σ^2) at x is:
    # log p(x) = -0.5 * ((x - μ)/σ)^2 - log(σ) - 0.5 * log(2π)
    # Here, x = mpe_allocation, μ = policy_mean, σ = policy_std

    z_score = (mpe_allocation - policy_mean) / policy_std

    log_pdf = (
        -0.5 * (z_score**2)
        - torch.log(policy_std)
        - 0.5 * torch.log(torch.tensor(2 * np.pi))
    )
    kl_divergence = -log_pdf
    kl_divergence = torch.clamp(kl_divergence, min=0, max=100)
    return kl_divergence.item()


