# POLARIS

<div align="center">

**P**artially **O**bservable **L**earning with **A**ctive **R**einforcement **I**n **S**ocial Environments

*A multi-agent reinforcement learning framework for strategic social learning*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Quick Start](#-quick-start) • [Examples](#-examples) • [Research Features](#-research-features)

</div>

---

## Overview

POLARIS is a multi-agent reinforcement learning framework for studying **strategic social learning**. It implements two canonical environments from economic theory and provides sophisticated neural architectures for modeling how agents learn from both private signals and social observations.

### Theoretical Foundation

POLARIS introduces **Partially Observable Active Markov Games (POAMGs)**, extending traditional multi-agent frameworks to handle strategic learning under partial observability. Key theoretical contributions include:

- **Convergence Guarantees**: Stochastically stable distributions ensure well-defined limiting behavior
- **Policy Gradient Theorems**: Novel gradients for belief-conditioned policies in non-stationary environments  
- **Active Equilibrium Concepts**: Strategic reasoning about influencing others' learning processes

**[Read the full theoretical treatment →](docs/thesis.pdf)**

### Key Features

- **Theoretical Foundation**: Based on Partially Observable Active Markov Games (POAMGs)
- **Strategic Learning**: Agents influence others' learning processes under partial observability
- **Advanced Architectures**: Graph Neural Networks with Temporal Attention and Transformers 
- **Continual Learning**: Synaptic Intelligence prevents catastrophic forgetting
- **Two Environments**: Brandl social learning and Keller-Rady strategic experimentation

## Quick Start

### Installation

```bash
# Basic installation
pip install polaris-marl

# With all features (recommended)
pip install polaris-marl[all]
```

### Command Line Usage

**General Purpose Simulation**
```bash
# Social learning experiment (Brandl framework)
polaris-simulate --environment-type brandl --num-agents 5 --num-states 3 --signal-accuracy 0.8

# Strategic experimentation (Keller-Rady framework)
polaris-simulate --environment-type strategic_experimentation --num-agents 4 --continuous-actions
```

**Research Scripts**
```bash
# Brandl social learning sweep - analyzes individual agent performance across network topologies
python experiments/brandl_sweep.py --agent-counts 1 2 4 6 8 --network-types complete ring star random --episodes 5

# Keller-Rady strategic experimentation sweep - compares aggregate performance across agent counts
python experiments/keller_rady_sweep.py --agent-counts 2 3 4 5 6 7 8 --episodes 3

# Individual experiments
python experiments/brandl_experiment.py --agents 8 --signal-accuracy 0.75 --plot-states
python experiments/keller_rady_experiment.py --agents 2 --horizon 10000 --plot-allocations

# List all available scripts
python -m polaris.experiments
```

### Python API

```python
from polaris.config.experiment_config import (
    ExperimentConfig, AgentConfig, TrainingConfig, BrandlConfig
)
from polaris.environments.social_learning import SocialLearningEnvironment
from polaris.training.simulation import run_experiment

# Create configuration
config = ExperimentConfig(
    agent=AgentConfig(
        learning_rate=1e-3,
        use_si=True,  # Enable Synaptic Intelligence
        num_gnn_layers=3  # Graph Neural Networks (default architecture)
    ),
    training=TrainingConfig(
        num_episodes=10,
        horizon=1000
    ),
    environment=BrandlConfig(
        num_agents=5,
        num_states=3,
        signal_accuracy=0.8,
        network_type='complete'
    )
)

# Create environment
env = SocialLearningEnvironment(
    num_agents=config.environment.num_agents,
    num_states=config.environment.num_states,
    signal_accuracy=config.environment.signal_accuracy,
    network_type=config.environment.network_type
)

# Run experiment
episodic_metrics, processed_metrics = run_experiment(env, config)
```

## Research Features

### Environments

**Brandl Social Learning**: Agents learn about a hidden state through private signals and social observation
- Discrete actions, configurable network topologies, theoretical bounds analysis

**Strategic Experimentation (Keller-Rady)**: Agents allocate resources between safe and risky options
- Continuous actions, Lévy processes, exploration-exploitation trade-offs

### Neural Architectures

- **Graph Neural Networks**: Temporal attention over social networks
- **Transformers**: Advanced belief state processing
- **Variational Inference**: Opponent modeling and belief updating

### Advanced Features

```bash
# Graph Neural Networks with temporal attention
polaris-simulate --gnn-layers 3 --attn-heads 8 --temporal-window 10

# Continual learning with Synaptic Intelligence
polaris-simulate --use-si --si-importance 150.0

# Custom network topologies
polaris-simulate --network-type ring --network-density 0.3
```

### Sweep Analysis Scripts

POLARIS provides specialized sweep scripts for comprehensive research analysis:

#### **Brandl Social Learning Sweep**

Analyzes individual agent learning performance across network topologies:

```bash
# Basic usage - analyze learning across network sizes and types
python experiments/brandl_sweep.py

# Custom configuration with statistical analysis
python experiments/brandl_sweep.py \
    --agent-counts 1 2 4 6 8 10 \
    --network-types complete ring star random \
    --episodes 5 \
    --horizon 100 \
    --signal-accuracy 0.75
```

**Key Features:**
- **Learning Rate Calculation**: Computes individual learning rates using log-linear regression
- **Statistical Analysis**: Multiple episodes with 95% confidence intervals
- **Extreme Agent Focus**: Shows fastest (green) and slowest (red) learners to avoid overcrowding
- **Network Topology Comparison**: Analyzes performance across complete, ring, star, and random networks

**Generated Outputs:**
- `fastest_slowest_network_sizes_evolution.png` - Performance trajectories across network sizes
- `fastest_slowest_network_types_evolution.png` - Performance trajectories across network types
- `agent_performance_results.json` - Complete numerical results with learning rates

#### **Keller-Rady Strategic Experimentation Sweep**

Compares aggregate performance across different agent counts:

```bash
# Basic usage - compare performance across agent counts
python experiments/keller_rady_sweep.py

# Custom configuration  
python experiments/keller_rady_sweep.py \
    --agent-counts 2 3 4 5 6 7 8 \
    --episodes 3 \
    --horizon 100
```

**Key Features:**
- **Multi-Agent Comparison**: Analyzes how performance scales with agent count
- **Statistical Analysis**: Confidence intervals across multiple episodes
- **Cumulative Allocation Tracking**: Resource allocation patterns over time
- **Convergence Analysis**: Studies optimal strategy convergence

**Generated Outputs:**
- `average_cumulative_allocation_per_agent_over_time.png` - Allocation trends with confidence intervals

## Examples

### Research Workflow
```bash
# 1. Individual agent analysis (Brandl social learning)
python experiments/brandl_sweep.py --agent-counts 2 4 6 8 --network-types complete ring star --episodes 5

# 2. Multi-agent comparison (Keller-Rady strategic experimentation)
python experiments/keller_rady_sweep.py --agent-counts 2 3 4 5 6 7 8 --episodes 3

# 3. Single experiments with visualization
python experiments/brandl_experiment.py --agents 8 --signal-accuracy 0.75 --plot-states --latex-style
python experiments/keller_rady_experiment.py --agents 2 --horizon 10000 --plot-allocations
```

### Advanced Configuration
```python
from polaris.config.experiment_config import ExperimentConfig, AgentConfig, TrainingConfig, StrategicExpConfig

# Strategic experimentation with continual learning
config = ExperimentConfig(
    agent=AgentConfig(
        use_si=True,
        si_importance=100.0,
        num_gnn_layers=3,
        temporal_window_size=10
    ),
    training=TrainingConfig(
        num_episodes=10, 
        horizon=1000
    ),
    environment=StrategicExpConfig(
        num_agents=4,
        continuous_actions=True,
        safe_payoff=1.0,
        drift_rates=[-0.5, 0.5]
    )
)
```

## Available Scripts

| Script | Purpose | Key Features |
|--------|---------|-------------|
| `polaris-simulate` | General experimentation | Flexible interface for both environments |
| `experiments/brandl_experiment.py` | Single Brandl experiment | Belief analysis, state plots |
| `experiments/keller_rady_experiment.py` | Single strategic experiment | Allocation plots, convergence analysis |
| `experiments/brandl_sweep.py` | Multi-agent Brandl analysis | Learning rates, network topology comparison |
| `experiments/keller_rady_sweep.py` | Multi-agent strategic analysis | Cumulative allocations, scaling analysis |

## Project Structure

```
polaris/
├── agents/          # Agent implementations with memory systems
├── algorithms/      # Regularization techniques (SI, EWC)
├── config/          # Configuration system
├── environments/    # Brandl and Keller-Rady environments
├── networks/        # Graph neural network architectures
├── training/        # Training loop and simulation runner
├── utils/           # Utilities for device management, etc.
└── visualization/   # Plotting and visualization tools

experiments/
├── brandl_experiment.py           # Single Brandl experiment
├── keller_rady_experiment.py      # Single strategic experimentation
├── brandl_sweep.py               # Multi-agent Brandl analysis
├── keller_rady_sweep.py          # Multi-agent strategic analysis
└── brandl_policy_inversion_analysis.py  # Policy analysis tools
```

## Development

```bash
# Development installation
git clone https://github.com/ecdogaroglu/polaris.git
cd polaris
pip install -e .

# Run tests
pytest tests/

# Check available experiments
python -m polaris.experiments
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{polaris2025,
  title={POLARIS: Partially Observable Learning with Active Reinforcement In Social Environments},
  author={Ege Can Doğaroğlu},
  year={2025},
  url={https://github.com/ecdogaroglu/polaris}
}
```



