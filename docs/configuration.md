# POLARIS Configuration Guide

This guide provides detailed information about configuring POLARIS for different experimental setups and research scenarios.

## Table of Contents

- [Configuration Overview](#configuration-overview)
- [Environment Configuration](#environment-configuration)
- [Agent Configuration](#agent-configuration)
- [Training Configuration](#training-configuration)
- [Network Architecture Configuration](#network-architecture-configuration)
- [Visualization Configuration](#visualization-configuration)
- [Command-Line Arguments](#command-line-arguments)
- [Configuration Examples](#configuration-examples)

---

## Configuration Overview

POLARIS uses a hierarchical configuration system that supports:

- **Default configurations** for standard experimental setups
- **Environment-specific** parameter sets
- **Command-line argument** overrides
- **Programmatic configuration** through dictionaries

### Basic Configuration Loading

```python
from polaris.config.defaults import get_default_config

# Load default configuration for Brandl social learning
config = get_default_config('brandl')

# Load default configuration for strategic experimentation
config = get_default_config('strategic_experimentation')
```

### Configuration Structure

The configuration system includes several default parameter sets:

- `AGENT_DEFAULTS`: Agent hyperparameters
- `TRAINING_DEFAULTS`: Training loop parameters
- `NETWORK_DEFAULTS`: Neural network architecture parameters
- `ENVIRONMENT_DEFAULTS`: Environment parameters
- `STRATEGIC_EXP_DEFAULTS`: Strategic experimentation settings
- `BRANDL_DEFAULTS`: Brandl social learning settings
- `SI_DEFAULTS`: Synaptic Intelligence parameters
- `VISUALIZATION_DEFAULTS`: Plotting options

---

## Environment Configuration

### Brandl Social Learning Environment

```python
# Default parameters for Brandl environment
config = {
    'num_agents': 2,                     # Number of agents
    'num_states': 2,                     # Number of possible states
    'network_type': 'complete',          # Network topology
    'network_density': 0.5,              # Density for random networks
    'signal_accuracy': 0.75,             # Private signal accuracy
}
```

#### Available Network Types
- **`'complete'`**: Every agent connected to every other agent
- **`'ring'`**: Agents arranged in a ring topology
- **`'star'`**: Star topology with central hub
- **`'random'`**: Random network with specified density

### Strategic Experimentation Environment

```python
# Default parameters for strategic experimentation
config = {
    'num_agents': 2,                     # Number of agents
    'safe_payoff': 1.0,                  # Deterministic payoff of safe arm
    'drift_rates': [-0.5, 0.5],         # Drift rates for each state
    'diffusion_sigma': 0.5,              # Volatility of diffusion component
    'jump_rates': [0.1, 0.2],           # Poisson rates for jumps in each state
    'jump_sizes': [1.0, 1.0],           # Expected jump sizes in each state
    'background_informativeness': 0.1,   # Informativeness of background signal
    'time_step': 0.1,                    # Time step for discretizing Lévy processes
    'continuous_actions': False,         # Use continuous action space
}
```

---

## Agent Configuration

### Basic Agent Parameters

```python
agent_config = {
    'hidden_dim': 256,                   # Hidden layer dimension
    'belief_dim': 256,                   # Belief state dimension
    'latent_dim': 256,                   # Latent space dimension
    'learning_rate': 1e-3,               # Learning rate
    'discount_factor': 0.9,              # Discount factor
    'entropy_weight': 0.5,               # Entropy bonus weight
    'kl_weight': 10.0,                   # KL weight for inference
}
```

### Advanced Agent Features

```python
advanced_config = {
    # Graph Neural Networks (default and only architecture)
    'gnn_layers': 2,                     # Number of GNN layers
    'attn_heads': 4,                     # Number of attention heads
    'temporal_window': 5,                # Temporal window size
    
    # Synaptic Intelligence
    'use_si': False,                     # Enable Synaptic Intelligence
    'si_importance': 100.0,              # SI importance factor (lambda)
    'si_damping': 0.1,                   # SI damping factor
    'si_exclude_final_layers': False,    # Exclude final layers from SI
    
    # Action space
    'continuous_actions': False,         # Use continuous actions
}
```

---

## Training Configuration

### Basic Training Parameters

```python
training_config = {
    'batch_size': 128,                   # Training batch size
    'buffer_capacity': 1000,             # Replay buffer capacity
    'update_interval': 10,               # Steps between network updates
    'num_episodes': 1,                   # Number of training episodes
    'horizon': 1000,                     # Maximum steps per episode
}
```

### Environment Selection

```python
# Choose between environments
environment_configs = {
    'brandl': {
        'environment_type': 'brandl',
        'signal_accuracy': 0.75,
    },
    'strategic_experimentation': {
        'environment_type': 'strategic_experimentation',
        'safe_payoff': 1.0,
        'continuous_actions': True,
    }
}
```

---

## Network Architecture Configuration

### Standard Network Parameters

```python
network_config = {
    'hidden_dim': 256,                   # Hidden layer dimension
    'belief_dim': 256,                   # Belief state dimension
    'latent_dim': 256,                   # Latent space dimension
}
```

### Graph Neural Network Configuration

POLARIS uses Graph Neural Networks by default for all inference tasks. The GNN architecture provides superior performance for modeling agent interactions.

```python
gnn_config = {
    'gnn_layers': 2,                     # Number of GNN layers
    'attn_heads': 4,                     # Number of attention heads
    'temporal_window': 5,                # Temporal window size for memory
}
```

---

## Visualization Configuration

```python
visualization_config = {
    'latex_style': False,                # Use LaTeX-style formatting
    'use_tex': False,                    # Use actual LaTeX rendering
    'plot_internal_states': False,       # Plot belief and latent states
    'plot_allocations': False,           # Plot allocations over time
    'visualize_si': False,               # Visualize SI importance scores
}
```

---

## Command-Line Arguments

POLARIS provides comprehensive command-line argument support:

### Environment Arguments

```bash
# Environment selection
--environment-type {brandl,strategic_experimentation}
--num-agents INT                     # Number of agents
--num-states INT                     # Number of states
--network-type {complete,ring,star,random}
--network-density FLOAT              # For random networks

# Brandl-specific
--signal-accuracy FLOAT              # Private signal accuracy

# Strategic experimentation-specific
--safe-payoff FLOAT                  # Safe arm payoff
--drift-rates STR                    # Comma-separated drift rates
--diffusion-sigma FLOAT              # Diffusion volatility
--continuous-actions                 # Enable continuous actions
```

### Training Arguments

```bash
# Training parameters
--batch-size INT                     # Training batch size
--buffer-capacity INT                # Replay buffer capacity
--learning-rate FLOAT                # Learning rate
--update-interval INT                # Update frequency
--horizon INT                        # Episode length
--num-episodes INT                   # Number of episodes
```

### Agent Arguments

```bash
# Agent architecture
--hidden-dim INT                     # Hidden layer dimension
--belief-dim INT                     # Belief state dimension
--latent-dim INT                     # Latent space dimension
--discount-factor FLOAT              # Discount factor
--entropy-weight FLOAT               # Entropy bonus weight
--kl-weight FLOAT                    # KL divergence weight

# Advanced features
--gnn-layers INT                     # Number of GNN layers (default: 2)
--attn-heads INT                     # Number of attention heads (default: 4)
--temporal-window INT                # Temporal window size (default: 5)

# Synaptic Intelligence
--use-si                             # Enable Synaptic Intelligence
--si-importance FLOAT                # SI importance factor
--si-damping FLOAT                   # SI damping factor
--si-exclude-final-layers            # Exclude final layers from SI
```

### Experiment Arguments

```bash
# Experiment control
--seed INT                           # Random seed
--device STR                         # Device (cuda/mps/cpu)
--output-dir STR                     # Output directory
--exp-name STR                       # Experiment name
--save-model                         # Save trained models
--load-model [PATH]                  # Load model (auto or specific path)

# Evaluation
--eval-only                          # Only evaluate, no training
--train-then-evaluate                # Train then evaluate

# Comparisons
--compare-sizes                      # Compare different network sizes
--network-sizes STR                  # Comma-separated sizes
--compare-frameworks                 # Compare Brandl vs Strategic Exp
```

### Visualization Arguments

```bash
# Plotting options
--plot-internal-states               # Plot belief and latent states
--plot-type {belief,latent,both}     # Type of internal state to plot
--plot-allocations                   # Plot allocations over time
--latex-style                        # LaTeX-style formatting
--use-tex                            # Use LaTeX rendering
--visualize-si                       # Visualize SI importance
```

---

## Configuration Examples

### Quick Start Example

```python
from polaris.config.defaults import get_default_config

# Minimal configuration for Brandl environment
config = get_default_config('brandl')
config['num_agents'] = 5
config['num_states'] = 3
```

### Research Configuration Example

```python
# Configuration for research experiments
config = get_default_config('strategic_experimentation')
config.update({
    'num_agents': 4,
    'continuous_actions': True,
    'gnn_layers': 3,
    'attn_heads': 8,
    'use_si': True,
    'si_importance': 150.0,
    'hidden_dim': 512,
    'learning_rate': 5e-4,
})
```

### Command-Line Examples

```bash
# Train Brandl environment with GNN (default architecture)
python -m polaris.simulation \
    --environment-type brandl \
    --num-agents 10 \
    --num-states 5 \
    --network-type complete \
    --gnn-layers 3 \
    --save-model

# Strategic experimentation with continuous actions and SI
python -m polaris.simulation \
    --environment-type strategic_experimentation \
    --num-agents 4 \
    --continuous-actions \
    --use-si \
    --si-importance 200.0 \
    --plot-allocations

# Evaluation only
python -m polaris.simulation \
    --environment-type brandl \
    --eval-only \
    --load-model auto \
    --plot-internal-states
```

### Experiment Configuration Classes

POLARIS also provides structured configuration classes:

```python
from polaris.config.experiment_config import (
    AgentConfig, TrainingConfig, BrandlConfig, 
    StrategicExpConfig, ExperimentConfig
)

# Create structured configuration
agent_config = AgentConfig(
    hidden_dim=256,
    use_si=True
)

training_config = TrainingConfig(
    num_episodes=100,
    batch_size=64
)

env_config = BrandlConfig(
    num_agents=10,
    signal_accuracy=0.8
)

experiment = ExperimentConfig(
    agent_config=agent_config,
    training_config=training_config,
    environment_config=env_config
)
```

## Default Values Reference

### Agent Defaults

```python
AGENT_DEFAULTS = {
    'hidden_dim': 256,
    'belief_dim': 256,
    'latent_dim': 256,
    'learning_rate': 1e-3,
    'discount_factor': 0.9,
    'entropy_weight': 0.5,
    'kl_weight': 10.0,
}
```

### Training Defaults

```python
TRAINING_DEFAULTS = {
    'batch_size': 128,
    'buffer_capacity': 1000,
    'update_interval': 10,
    'num_episodes': 1,
    'horizon': 1000,
}
```

### Environment Defaults

```python
ENVIRONMENT_DEFAULTS = {
    'num_agents': 2,
    'num_states': 2,
    'network_type': 'complete',
    'network_density': 0.5,
}
```

For complete parameter descriptions and advanced usage, see the [API Reference](api.md) and examine the default configuration files in `polaris/config/`.