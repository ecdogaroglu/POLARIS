# POLARIS

<div align="center">

**P**artially **O**bservable **L**earning with **A**ctive **R**einforcement **I**n **S**ocial Environments

*A multi-agent reinforcement learning framework for strategic social learning*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
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
- **Advanced Architectures**: Graph Neural Networks, Transformers, and Temporal Attention
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

#### **General Purpose**
```bash
# Social learning experiment
polaris-simulate --environment-type brandl --num-agents 5 --num-states 3

# Strategic experimentation
polaris-simulate --environment-type strategic_experimentation --num-agents 4 --continuous-actions
```

#### **Research Scripts**
```bash
# Social learning with enhanced analysis
polaris-brandl --agents 8 --signal-accuracy 0.75 --plot-states --latex-style

# Strategic experimentation with allocations
polaris-strategic --agents 2 --horizon 10000 --plot-allocations --use-gnn

# Individual agent performance analysis (Brandl social learning)
python experiments/brandl_sweep.py --agent-counts 1 2 4 6 8 10 --network-types complete ring star random --episodes 5

# Multi-agent comparison analysis (Keller-Rady strategic experimentation)
python experiments/keller_rady_sweep.py --agent-counts 2 3 4 5 6 7 8 --episodes 3

# List all available scripts and examples
python -m polaris.experiments
```

### Python API

```python
from polaris.environments.social_learning import SocialLearningEnvironment
from polaris.training.trainer import Trainer
from polaris.config.args import parse_args

# Create environment
env = SocialLearningEnvironment(
    num_agents=5,
    num_states=3,
    signal_accuracy=0.8,
    network_type='complete'
)

# Configure and train
args = parse_args()
trainer = Trainer(env, args)
results = trainer.run_agents(training=True)
```

## Research Features

### Environments

**Brandl Social Learning**: Agents learn about a hidden state through private signals and social observation
- Discrete actions, configurable networks, theoretical bounds analysis

**Strategic Experimentation (Keller-Rady)**: Agents allocate resources between safe and risky options
- Continuous actions, Lévy processes, exploration-exploitation trade-offs

### Neural Architectures

- **Graph Neural Networks**: Temporal attention over social networks
- **Transformers**: Advanced belief state processing
- **Variational Inference**: Opponent modeling and belief updating

### Advanced Features

```bash
# Graph Neural Networks with temporal attention
polaris-simulate --use-gnn --gnn-layers 3 --attn-heads 8

# Continual learning with Synaptic Intelligence
polaris-simulate --use-si --si-importance 150.0

# Enhanced visualizations
polaris-brandl --plot-states --latex-style
polaris-strategic --plot-allocations --save-model
```

### Sweep Analysis Scripts

POLARIS provides two specialized sweep scripts for comprehensive analysis across different experimental conditions:

#### **Brandl Social Learning Sweep**

The `brandl_sweep.py` script analyzes individual agent learning performance across network topologies:

```bash
# Basic usage - analyze learning across network sizes and types
python experiments/brandl_sweep.py

# Custom configuration with multiple episodes for statistical analysis
python experiments/brandl_sweep.py \
    --agent-counts 1 2 4 6 8 10 \
    --network-types complete ring star random \
    --episodes 5 \
    --horizon 100 \
    --signal-accuracy 0.75
```

**Key Features:**
- **Learning Rate Calculation**: Computes individual learning rates (r) using log-linear regression
- **Statistical Analysis**: Multiple episodes with 95% confidence intervals
- **Extreme Agent Focus**: Shows only fastest (green) and slowest (red) learners to avoid plot overcrowding
- **Agent Resetting**: Proper agent state reset between episodes to prevent information leakage
- **Network Topology Comparison**: Analyzes performance across complete, ring, star, and random networks

**Generated Outputs:**
- `fastest_slowest_network_sizes_evolution.png` - Fastest/slowest trajectories with CIs across network sizes
- `fastest_slowest_network_types_evolution.png` - Fastest/slowest trajectories with CIs across network types
- `agent_performance_results.json` - Complete numerical results with learning rates and confidence intervals

#### **Keller-Rady Strategic Experimentation Sweep**

The `keller_rady_sweep.py` script compares aggregate performance across different agent counts:

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
- **Statistical Analysis**: Provides confidence intervals across multiple episodes
- **Cumulative Allocation Tracking**: Shows resource allocation patterns over time
- **Convergence Analysis**: Studies how quickly agents reach optimal strategies

**Generated Outputs:**
- `average_cumulative_allocation_per_agent_over_time.png` - Allocation trends with confidence intervals
- Detailed performance metrics for each agent count configuration

## Examples

### Research Workflow
```bash
# 1. Train social learning agents
polaris-brandl --agents 8 --signal-accuracy 0.75 --use-gnn --plot-states

# 2. Strategic experimentation
polaris-strategic --agents 2 --horizon 10000 --plot-allocations --latex-style

# 3. Individual agent analysis (Brandl)
python experiments/brandl_sweep.py --agent-counts 2 4 6 8 --network-types complete ring star --episodes 5

# 4. Multi-agent comparison (Keller-Rady)
python experiments/keller_rady_sweep.py --agent-counts 2 3 4 5 6 7 8 --episodes 3
```

### Advanced Configuration
```python
from polaris.config.experiment_config import ExperimentConfig, AgentConfig, TrainingConfig

# Custom configuration
config = ExperimentConfig(
    agent=AgentConfig(use_gnn=True, use_si=True),
    training=TrainingConfig(num_episodes=10, horizon=1000)
)
```

## Console Scripts Reference

| Command | Purpose | Key Features |
|---------|---------|-------------|
| `polaris-simulate` | General experimentation | Flexible, all environments |
| `polaris-brandl` | Social learning research | Theoretical bounds, belief analysis |
| `polaris-strategic` | Strategic experimentation | Allocation plots, KL divergence |
| `experiments/brandl_sweep.py` | Multi-agent comparison (Brandl) | Learning rates, trajectory comparison, network topology effects, confidence intervals |
| `experiments/keller_rady_sweep.py` | Multi-agent comparison (Keller-Rady) | Cumulative allocations |
| `polaris-experiment` | Quick testing | Simplified interface |

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



