"""
Mathematical utilities for POLARIS.
"""

import numpy as np
from typing import List, Dict


def calculate_learning_rate(mistake_history: List[float]) -> float:
    """
    Calculate the learning rate (rate of decay of mistakes) using log-linear regression.
    
    Args:
        mistake_history: List of mistake rates over time
        
    Returns:
        learning_rate: Estimated learning rate
    """
    if len(mistake_history) < 10:  # Need sufficient points for regression
        return 0.0
    
    mistake_history = np.array(mistake_history)

    # Time steps
    t = np.arange(len(mistake_history))
    
    # Log of mistake probability, avoiding log(0)
    log_mistakes = np.log(np.clip(mistake_history, 1e-10, 1.0))
    
    # Simple linear regression on log-transformed data
    # log(P(mistake)) = -rt + c
    A = np.vstack([t, np.ones_like(t)]).T
    result = np.linalg.lstsq(A, log_mistakes, rcond=None)
    minus_r, c = result[0]
    
    # Negate slope to get positive learning rate
    learning_rate = -minus_r
    
    return learning_rate


def calculate_agent_learning_rates(
    incorrect_probs: Dict[int, List[List[float]]], 
    min_length: int = 100
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
