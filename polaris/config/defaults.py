"""
Default configurations for POLARIS.
"""

from typing import Dict, Any


# Default agent hyperparameters
AGENT_DEFAULTS = {
    'hidden_dim': 256,
    'belief_dim': 256,
    'latent_dim': 256,
    'learning_rate': 1e-3,
    'discount_factor': 0.9,
    'entropy_weight': 0.5,
    'kl_weight': 10.0,
}

# Default training parameters
TRAINING_DEFAULTS = {
    'batch_size': 128,
    'buffer_capacity': 1000,
    'update_interval': 10,
    'num_episodes': 1,
    'horizon': 1000,
}

# Default network parameters
NETWORK_DEFAULTS = {
    'gnn_layers': 2,
    'attn_heads': 4,
    'temporal_window': 5,
}

# Default environment parameters
ENVIRONMENT_DEFAULTS = {
    'num_agents': 2,
    'num_states': 2,
    'network_type': 'complete',
    'network_density': 0.5,
}

# Default strategic experimentation parameters  
STRATEGIC_EXP_DEFAULTS = {
    'safe_payoff': 1.0,
    'drift_rates': [-0.5, 0.5],
    'diffusion_sigma': 0.5,
    'jump_rates': [0.1, 0.2],
    'jump_sizes': [1.0, 1.0],
    'background_informativeness': 0.1,
    'time_step': 0.1,
    'continuous_actions': True,
}

# Default Brandl social learning parameters
BRANDL_DEFAULTS = {
    'signal_accuracy': 0.75,
}

# Default SI parameters
SI_DEFAULTS = {
    'si_importance': 100.0,
    'si_damping': 0.1,
    'si_exclude_final_layers': False,
}

# Default visualization parameters
VISUALIZATION_DEFAULTS = {
    'latex_style': True,
    'use_tex': False,
    'plot_internal_states': False,
    'plot_allocations': False,
    'visualize_si': False,
}


def get_default_config(environment_type: str = 'brandl') -> Dict[str, Any]:
    """Get default configuration for a specific environment type."""
    config = {}
    config.update(AGENT_DEFAULTS)
    config.update(TRAINING_DEFAULTS)
    config.update(NETWORK_DEFAULTS)
    config.update(ENVIRONMENT_DEFAULTS)
    config.update(SI_DEFAULTS)
    config.update(VISUALIZATION_DEFAULTS)
    
    if environment_type == 'strategic_experimentation':
        config.update(STRATEGIC_EXP_DEFAULTS)
    elif environment_type == 'brandl':
        config.update(BRANDL_DEFAULTS)
    
    return config
