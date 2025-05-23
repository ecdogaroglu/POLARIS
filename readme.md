# POLARIS

<div align="center">

**P**artially **O**bservable **L**earning with **A**ctive **R**einforcement **I**n **S**ocial Environments

*A multi-agent reinforcement learning framework for strategic social learning*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[🚀 Quick Start](#-quick-start) • [💡 Examples](#-examples) • [🔬 Research Features](#-research-features)

</div>

---

## 🎯 Overview

POLARIS is a multi-agent reinforcement learning framework for studying **strategic social learning**. It implements two canonical environments from economic theory and provides sophisticated neural architectures for modeling how agents learn from both private signals and social observations.

### 🧮 Theoretical Foundation

POLARIS introduces **Partially Observable Active Markov Games (POAMGs)**, extending traditional multi-agent frameworks to handle strategic learning under partial observability. Key theoretical contributions include:

- **Convergence Guarantees**: Stochastically stable distributions ensure well-defined limiting behavior
- **Policy Gradient Theorems**: Novel gradients for belief-conditioned policies in non-stationary environments  
- **Active Equilibrium Concepts**: Strategic reasoning about influencing others' learning processes

📖 **[Read the full theoretical treatment →](docs/thesis.pdf)**

### 🏆 Key Features

- **🧠 Theoretical Foundation**: Based on Partially Observable Active Markov Games (POAMGs)
- **🎮 Strategic Learning**: Agents influence others' learning processes under partial observability
- **🤝 Advanced Architectures**: Graph Neural Networks, Transformers, and Temporal Attention
- **🔄 Continual Learning**: Synaptic Intelligence prevents catastrophic forgetting
- **📊 Two Environments**: Brandl social learning and Keller-Rady strategic experimentation
- **⚡ GPU Acceleration**: Optimized for CUDA/MPS/CPU

## 🚀 Quick Start

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

# Comparative analysis across agent counts
polaris-sweep --agent-counts 2 3 4 5 --horizon 400

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

## 🔬 Research Features

### 🌍 Environments

**Brandl Social Learning**: Agents learn about a hidden state through private signals and social observation
- Discrete actions, configurable networks, theoretical bounds analysis

**Strategic Experimentation (Keller-Rady)**: Agents allocate resources between safe and risky options
- Continuous actions, Lévy processes, exploration-exploitation trade-offs

### 🧠 Neural Architectures

- **Graph Neural Networks**: Temporal attention over social networks
- **Transformers**: Advanced belief state processing
- **Variational Inference**: Opponent modeling and belief updating

### 🎯 Advanced Features

```bash
# Graph Neural Networks with temporal attention
polaris-simulate --use-gnn --gnn-layers 3 --attn-heads 8

# Continual learning with Synaptic Intelligence
polaris-simulate --use-si --si-importance 150.0

# Enhanced visualizations
polaris-brandl --plot-states --latex-style
polaris-strategic --plot-allocations --save-model
```

## 💡 Examples

### Research Workflow
```bash
# 1. Train social learning agents
polaris-brandl --agents 8 --signal-accuracy 0.75 --use-gnn --plot-states

# 2. Strategic experimentation
polaris-strategic --agents 2 --horizon 10000 --plot-allocations --latex-style

# 3. Comparative analysis
polaris-sweep --agent-counts 2 3 4 5 6 7 8 --episodes 3
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

## 📊 Console Scripts Reference

| Command | Purpose | Key Features |
|---------|---------|-------------|
| `polaris-simulate` | General experimentation | Flexible, all environments |
| `polaris-brandl` | Social learning research | Theoretical bounds, belief analysis |
| `polaris-strategic` | Strategic experimentation | Allocation plots, KL divergence |
| `polaris-sweep` | Multi-agent comparison | Statistical analysis, confidence intervals |
| `polaris-experiment` | Quick testing | Simplified interface |

## 🛠️ Development

```bash
# Development installation
git clone https://github.com/ecdogaroglu/polaris.git
cd polaris
pip install -e .[all]

# Run tests
pytest tests/

# Check available experiments
python -m polaris.experiments
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📚 Citation

```bibtex
@software{polaris2025,
  title={POLARIS: Partially Observable Learning with Active Reinforcement In Social Environments},
  author={Ege Can Doğaroğlu},
  year={2025},
  url={https://github.com/ecdogaroglu/polaris}
}
```

---

<div align="center">

[⭐ Star on GitHub](https://github.com/ecdogaroglu/polaris) • [🐛 Report Issues](https://github.com/ecdogaroglu/polaris/issues)

</div> 