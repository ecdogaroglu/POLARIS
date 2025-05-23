# POLARIS API Reference

This document provides a comprehensive reference for the POLARIS API, covering all major modules, classes, and functions.

## Table of Contents

- [Agents](#agents)
- [Environments](#environments)
- [Networks](#networks)
- [Algorithms](#algorithms)
- [Training](#training)
- [Visualization](#visualization)
- [Configuration](#configuration)
- [Utilities](#utilities)

---

## Agents

### POLARISAgent

The main agent class implementing the POLARIS algorithm.

```python
class POLARISAgent:
    def __init__(self, agent_id, num_agents, num_states, observation_dim, action_dim, 
                 hidden_dim=256, belief_dim=256, latent_dim=256, learning_rate=1e-3, 
                 discount_factor=0.9, entropy_weight=0.5, kl_weight=10.0, device='cpu',
                 buffer_capacity=1000, max_trajectory_length=1000, use_gnn=False, 
                 use_si=False, si_importance=100.0, si_damping=0.1, 
                 si_exclude_final_layers=False, continuous_actions=False)
```

**Parameters:**
- `agent_id`: Unique identifier for the agent
- `num_agents`: Total number of agents in the environment
- `num_states`: Number of possible states in the environment
- `observation_dim`: Dimension of observations
- `action_dim`: Dimension of actions
- `hidden_dim`: Hidden layer dimension (default: 256)
- `belief_dim`: Belief state dimension (default: 256)
- `latent_dim`: Latent space dimension (default: 256)
- `learning_rate`: Learning rate (default: 1e-3)
- `use_gnn`: Whether to use Graph Neural Networks (default: False)
- `continuous_actions`: Whether to use continuous action space (default: False)

**Methods:**

#### `observe(signal, neighbor_actions)`
Update belief state based on new observation.

**Parameters:**
- `signal`: Current observation signal
- `neighbor_actions`: Actions of neighboring agents

**Returns:**
- `new_belief`: Updated belief state
- `belief_distribution`: Belief distribution

#### `act(observation, neighbor_actions=None)`
Select an action based on the current observation.

#### `infer_latent(signal, actions, reward, next_signal)`
Infer latent state based on experience.

#### `update(batch)`
Update the agent's networks based on a batch of experiences.

### Agent Components

#### BeliefComponent

```python
class BeliefComponent:
    def __init__(self, belief_dim, num_states, device='cpu')
```

Handles belief state management and updates.

#### PolicyComponent

```python
class PolicyComponent:
    def __init__(self, belief_dim, latent_dim, action_dim, hidden_dim, device='cpu')
```

Manages policy network and action selection.

#### CriticComponent

```python
class CriticComponent:
    def __init__(self, belief_dim, latent_dim, action_dim, hidden_dim, device='cpu')
```

Implements value function estimation.

#### InferenceComponent

```python
class InferenceComponent:
    def __init__(self, observation_dim, action_dim, latent_dim, hidden_dim, device='cpu')
```

Handles opponent modeling and latent state inference.

### Memory

#### ReplayBuffer

```python
class ReplayBuffer:
    def __init__(self, capacity, observation_dim, belief_dim, latent_dim, 
                 device='cpu', sequence_length=8)
```

Experience replay buffer with support for sequential sampling.

**Methods:**

#### `add_transition(obs, actions, belief, latent, action, reward, next_obs, next_belief, next_latent, mean, logvar)`
Add an experience tuple to the buffer.

#### `sample(batch_size)`
Sample a batch of experiences.

#### `can_sample(batch_size)`
Check if buffer has enough samples.

---

## Environments

### BaseEnvironment

Abstract base class for all POLARIS environments.

```python
class BaseEnvironment(ABC):
    def __init__(self, num_agents, **kwargs)
```

**Methods:**

#### `step(actions)`
Execute one step of the environment.

#### `reset()`
Reset the environment to initial state.

#### `get_network()`
Get the social network structure.

### SocialLearningEnvironment

Environment implementing the Brandl social learning model.

```python
class SocialLearningEnvironment(BaseEnvironment):
    def __init__(self, num_agents=2, num_states=2, network_type='complete',
                 signal_accuracy=0.75, network_density=0.5)
```

**Parameters:**
- `num_agents`: Number of agents in the environment (default: 2)
- `num_states`: Number of possible states (default: 2)
- `network_type`: Type of social network ('complete', 'ring', 'star', 'random')
- `signal_accuracy`: Accuracy of private signals (default: 0.75)

### StrategicExperimentationEnvironment

Environment implementing the Keller-Rady strategic experimentation model.

```python
class StrategicExperimentationEnvironment(BaseEnvironment):
    def __init__(self, num_agents=2, safe_payoff=1.0, drift_rates=[-0.5, 0.5],
                 diffusion_sigma=0.5, jump_rates=[0.1, 0.2], jump_sizes=[1.0, 1.0],
                 background_informativeness=0.1, time_step=0.1, continuous_actions=False)
```

**Parameters:**
- `num_agents`: Number of agents (default: 2)
- `safe_payoff`: Deterministic payoff of safe arm (default: 1.0)
- `drift_rates`: List of drift rates for each state
- `continuous_actions`: Whether to use continuous actions (default: False)

---

## Networks

### Graph Neural Networks

#### TemporalGNN

Graph Neural Network with temporal attention for multi-agent communication.

```python
class TemporalGNN:
    def __init__(self, hidden_dim, action_dim, latent_dim, num_agents,
                 num_belief_states, num_gnn_layers=2, num_attn_heads=4,
                 temporal_window_size=5, device=None)
```

#### TransformerBeliefProcessor

Transformer-based belief state processor.

```python
class TransformerBeliefProcessor:
    def __init__(self, hidden_dim, action_dim, num_belief_states,
                 nhead=4, num_layers=2, dropout=0.1, device=None)
```

### Standard Networks

#### PolicyNetwork

Policy network for action selection.

```python
class PolicyNetwork:
    def __init__(self, belief_dim, latent_dim, action_dim, hidden_dim, device=None)
```

#### ContinuousPolicyNetwork

Policy network for continuous action spaces.

```python
class ContinuousPolicyNetwork:
    def __init__(self, belief_dim, latent_dim, action_dim, hidden_dim,
                 min_action=0.0, max_action=1.0, device=None)
```

#### EncoderNetwork

Encoder network for inference.

```python
class EncoderNetwork:
    def __init__(self, action_dim, latent_dim, hidden_dim, num_agents,
                 num_belief_states, device=None)
```

#### DecoderNetwork

Decoder network for action prediction.

```python
class DecoderNetwork:
    def __init__(self, action_dim, latent_dim, hidden_dim, num_agents,
                 num_belief_states, device=None)
```

---

## Algorithms

### Regularization

#### SILoss

```python
class SILoss:
    def __init__(self, importance=100.0, damping=0.1, exclude_final_layers=False)
```

Implements Synaptic Intelligence for continual learning.

#### EWCLoss

```python
class EWCLoss:
    def __init__(self, fisher_weight=1000, num_samples=100)
```

Implements Elastic Weight Consolidation for continual learning.

---

## Training

### Trainer

Main training class implementation.

```python
class Trainer:
    def __init__(self, env, args)
```

**Methods:**

#### `run_agents(training=True, model_path=None)`
Main training/evaluation loop.

#### `evaluate(num_episodes=None, num_steps=None)`
Evaluate trained agents.

#### `quick_evaluate(num_steps=100)`
Perform quick evaluation.

### Function Interface

```python
def run_agents(env, args, training=True, model_path=None):
    """
    Backward compatibility wrapper for training.
    
    Parameters:
    -----------
    env : BaseEnvironment
        The environment to train in
    args : argparse.Namespace
        Parsed command-line arguments
    training : bool
        Whether to train (True) or evaluate (False)
    model_path : str, optional
        Path to load models from
        
    Returns:
    --------
    tuple : (learning_rates, serializable_metrics)
        Training results
    """
```

### Evaluator

```python
class Evaluator:
    def __init__(self, env, agents, args)
```

Evaluates trained agents and computes performance metrics.

---

## Visualization

### Available Plotters

#### LearningCurvePlotter
Plot learning curves for agent performance.

#### BeliefPlotter
Visualize belief state evolution.

#### AllocationPlotter
Plot resource allocations for strategic experimentation.

### Base Classes

#### BasePlotter
Abstract base class for all plotters.

#### MultiAgentPlotter
Base class for multi-agent visualizations.

---

## Configuration

### Default Configurations

```python
def get_default_config(environment_type='brandl'):
    """
    Get default configuration for a specific environment.
    
    Parameters:
    -----------
    environment_type : str
        Type of environment ('brandl' or 'strategic_experimentation')
        
    Returns:
    --------
    config : dict
        Default configuration dictionary
    """
```

**Available Defaults:**
- `AGENT_DEFAULTS`: Agent hyperparameters
- `TRAINING_DEFAULTS`: Training parameters  
- `NETWORK_DEFAULTS`: Network architecture settings
- `ENVIRONMENT_DEFAULTS`: Environment parameters
- `STRATEGIC_EXP_DEFAULTS`: Strategic experimentation settings
- `BRANDL_DEFAULTS`: Brandl social learning settings
- `SI_DEFAULTS`: Synaptic Intelligence parameters
- `VISUALIZATION_DEFAULTS`: Plotting options

### Argument Parsing

```python
def parse_args():
    """
    Parse command line arguments for training scripts.
    
    Returns:
    --------
    args : argparse.Namespace
        Parsed arguments with environment and training parameters
    """
```

**Key Arguments:**
- `--environment-type`: 'brandl' or 'strategic_experimentation'
- `--num-agents`: Number of agents
- `--num-states`: Number of states
- `--use-gnn`: Enable Graph Neural Networks
- `--continuous-actions`: Use continuous action space
- `--use-si`: Enable Synaptic Intelligence

---

## Utilities

### Device Management

```python
def get_best_device():
    """
    Automatically select the best available device.
    
    Returns:
    --------
    device : str
        Device name ('cuda', 'mps', or 'cpu')
    """
```

### Encoding

```python
def encode_observation(signal, neighbor_actions, num_agents, num_states,
                      continuous_actions=False, continuous_signal=False):
    """
    Encode observations for neural network input.
    
    Returns:
    --------
    tuple : (signal_encoded, actions_encoded)
        Encoded signal and actions
    """
```

### Metrics

The metrics module provides comprehensive tracking and analysis capabilities for training and evaluation.

### I/O Operations

Functions for loading/saving models and managing experiment data.

---

## Error Handling

POLARIS uses standard Python exceptions with specific error messages:

- `ValueError`: Invalid parameter values
- `RuntimeError`: Runtime errors during training or inference
- `TypeError`: Type mismatches in function arguments

## Memory Management

POLARIS automatically manages GPU memory but provides manual control through device utilities.

For more detailed examples and usage patterns, see the project's example scripts and configuration files. 