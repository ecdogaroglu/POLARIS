# POLARIS

<div align="center">

**P**artially **O**bservable **L**earning with **A**ctive **R**einforcement **I**n **S**ocial Environments

*A cutting-edge multi-agent reinforcement learning framework for social and strategic environments*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🔬 Research](#-research-features) • [💡 Examples](#-examples)

</div>

---

## 🎯 Overview

POLARIS is a state-of-the-art multi-agent reinforcement learning framework designed for complex social and strategic environments. It combines advanced neural architectures with sophisticated learning algorithms to model partially observable environments where agents must learn from limited information while inferring the strategies of other agents.

### 🏆 Key Features

- **🧠 Advanced Neural Architectures**: Graph Neural Networks, Transformers, and Temporal Attention
- **🤝 Multi-Agent Learning**: Sophisticated opponent modeling and belief inference
- **🎮 Flexible Environments**: Social learning and strategic experimentation scenarios
- **🔄 Continual Learning**: Synaptic Intelligence and Elastic Weight Consolidation
- **📊 Rich Visualization**: Comprehensive analysis and plotting tools
- **⚡ High Performance**: Optimized for GPU/MPS acceleration

## 🏗️ Architecture

POLARIS follows a modular design for maximum flexibility and extensibility:

```
polaris/
├── 🤖 agents/                 # Intelligent agent implementations
│   ├── polaris_agent.py      # Main POLARIS agent
│   ├── components/           # Modular agent components
│   │   ├── belief.py         # Belief state processing
│   │   ├── inference.py      # Opponent inference
│   │   ├── policy.py         # Policy networks
│   │   └── critics.py        # Value function networks
│   └── memory/               # Memory systems
│       └── replay_buffer.py  # Experience replay
├── 🌍 environments/          # Training environments
│   ├── social_learning.py    # Brandl social learning
│   └── strategic_exp.py      # Keller-Rady strategic experimentation
├── 🧠 networks/              # Neural network architectures
│   ├── gnn.py               # Graph Neural Networks
│   ├── transformer.py       # Transformer models
│   └── mlp.py               # Multi-layer perceptrons
├── 🎯 algorithms/            # Learning algorithms
│   └── regularization/      # Continual learning
│       ├── si.py           # Synaptic Intelligence
│       └── ewc.py          # Elastic Weight Consolidation
├── 🚂 training/             # Training infrastructure
│   ├── trainer.py          # Main training loop
│   └── evaluator.py        # Evaluation system
├── 📊 visualization/        # Analysis and plotting
│   └── plots/              # Visualization modules
├── 🔧 utils/               # Utility functions
│   ├── device.py          # Device management
│   ├── encoding.py        # Data encoding
│   └── metrics.py         # Performance metrics
└── ⚙️ config/              # Configuration system
    ├── args.py            # Argument parsing
    └── defaults.py        # Default configurations
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ecdogaroglu/polaris.git
cd polaris

# Install dependencies
pip install -r requirements.txt

# Install POLARIS
pip install -e .
```

### Basic Usage

```python
from polaris.environments.social_learning import SocialLearningEnvironment
from polaris.training.trainer import run_agents
from polaris.config.args import parse_args
from polaris.config.defaults import get_default_config

# Configure environment using defaults
config = get_default_config('brandl')
config.update({
    'num_agents': 5,
    'num_states': 3,
    'signal_accuracy': 0.8
})

# Create environment
env = SocialLearningEnvironment(
    num_agents=config['num_agents'],
    num_states=config['num_states'],
    signal_accuracy=config['signal_accuracy']
)

# Parse command line arguments
args = parse_args()
args.num_agents = config['num_agents']
args.num_states = config['num_states']

# Train the system
learning_rates, metrics = run_agents(env, args, training=True)
```

### Running Experiments

```bash
# Brandl Social Learning Environment
python -m polaris.simulation \
    --environment-type brandl \
    --num-agents 10 \
    --num-states 5 \
    --network-type complete \
    --signal-accuracy 0.8

# Strategic Experimentation (Keller-Rady)
python -m polaris.simulation \
    --environment-type strategic_experimentation \
    --num-agents 4 \
    --continuous-actions \
    --use-gnn \
    --gnn-layers 3
```

## 🔬 Research Features

### 🧪 Multi-Environment Support

#### Brandl Social Learning Environment
- **Discrete Action Spaces**: Binary or multi-choice decisions
- **Signal Accuracy Control**: Configurable information quality (default: 0.75)
- **Network Topologies**: Complete, ring, star, random networks
- **Belief Dynamics**: Sophisticated belief updating mechanisms

#### Strategic Experimentation Environment (Keller-Rady)
- **Continuous Action Spaces**: Real-valued strategic choices
- **Lévy Processes**: Advanced stochastic modeling
- **Exploration-Exploitation Trade-offs**: Dynamic strategy adaptation
- **Market-like Dynamics**: Competitive multi-agent scenarios

### 🧠 Advanced Neural Architectures

#### Graph Neural Networks (GNNs)
```python
# Enable GNN with temporal attention
python -m polaris.simulation \
    --environment-type brandl \
    --use-gnn \
    --gnn-layers 3 \
    --attn-heads 8 \
    --temporal-window 10
```

#### Transformer Belief Processors
The framework includes transformer-based belief processing for sophisticated state representation learning.

### 🔄 Continual Learning

Prevent catastrophic forgetting with advanced regularization:

```bash
# Enable Synaptic Intelligence
python -m polaris.simulation \
    --environment-type brandl \
    --use-si \
    --si-importance 150.0 \
    --si-damping 0.1
```

## 💡 Examples

### Example 1: Brandl Social Learning with Network Effects

```python
from polaris.environments.social_learning import SocialLearningEnvironment
from polaris.training.trainer import Trainer
from polaris.config.args import parse_args

# Setup environment with star network
env = SocialLearningEnvironment(
    num_agents=10,
    num_states=3,
    network_type='star',
    signal_accuracy=0.8
)

# Configure training
args = parse_args()
args.environment_type = 'brandl'
args.num_agents = 10
args.num_states = 3
args.use_gnn = True

# Train agents
trainer = Trainer(env, args)
results = trainer.run_agents(training=True)
```

### Example 2: Strategic Experimentation with GNNs

```python
from polaris.environments.strategic_exp import StrategicExperimentationEnvironment
from polaris.training.trainer import Trainer

# Create continuous action environment
env = StrategicExperimentationEnvironment(
    num_agents=4,
    continuous_actions=True,
    safe_payoff=1.0
)

# Configure with GNN
args = parse_args()
args.environment_type = 'strategic_experimentation'
args.continuous_actions = True
args.use_gnn = True
args.gnn_layers = 3
args.attn_heads = 8

trainer = Trainer(env, args)
results = trainer.run_agents(training=True)
```

### Example 3: Evaluation and Visualization

```bash
# Train then evaluate with visualization
python -m polaris.simulation \
    --environment-type brandl \
    --train-then-evaluate \
    --plot-internal-states \
    --plot-type both \
    --save-model

# Evaluation only with model loading
python -m polaris.simulation \
    --environment-type brandl \
    --eval-only \
    --load-model auto \
    --plot-internal-states
```

## 📊 Visualization & Analysis

POLARIS includes comprehensive visualization tools:

```bash
# Enable internal state visualization
python -m polaris.simulation \
    --environment-type brandl \
    --plot-internal-states \
    --plot-type belief \
    --latex-style

# Strategic experimentation allocation plots
python -m polaris.simulation \
    --environment-type strategic_experimentation \
    --plot-allocations \
    --continuous-actions

# Synaptic Intelligence visualization
python -m polaris.simulation \
    --environment-type brandl \
    --use-si \
    --visualize-si
```

## ⚙️ Configuration

POLARIS uses a flexible configuration system:

```python
from polaris.config.defaults import get_default_config

# Get default configuration for Brandl
config = get_default_config('brandl')

# Available defaults include:
# - AGENT_DEFAULTS: hidden_dim=256, learning_rate=1e-3, etc.
# - TRAINING_DEFAULTS: batch_size=128, buffer_capacity=1000, etc.
# - ENVIRONMENT_DEFAULTS: num_agents=2, num_states=2, etc.
# - BRANDL_DEFAULTS: signal_accuracy=0.75
# - STRATEGIC_EXP_DEFAULTS: safe_payoff=1.0, continuous_actions=False, etc.

# Customize parameters
config.update({
    'num_agents': 15,
    'num_states': 4,
    'signal_accuracy': 0.9,
    'hidden_dim': 512,
    'learning_rate': 5e-4,
    'use_gnn': True,
    'gnn_layers': 3,
})
```

## 🔧 Advanced Features

### Device Management
```python
from polaris.utils.device import get_best_device

# Automatic device selection with MPS support
device = get_best_device()  # Returns 'mps', 'cuda', or 'cpu'

# Force specific device via command line
python -m polaris.simulation --device cuda
```

### Memory Management
```python
from polaris.agents.memory.replay_buffer import ReplayBuffer

# Advanced replay buffer with sequence sampling
buffer = ReplayBuffer(
    capacity=1000,
    observation_dim=64,
    belief_dim=256,
    latent_dim=256,
    sequence_length=8
)
```

### Experiment Configuration
```python
from polaris.config.experiment_config import ExperimentConfig, AgentConfig

# Structured configuration
agent_config = AgentConfig(hidden_dim=256, use_gnn=True)
experiment = ExperimentConfig(agent_config=agent_config)
```

## 📚 Documentation

- **[API Reference](docs/api.md)**: Complete API documentation
- **[Configuration Guide](docs/configuration.md)**: Configuration options
- **[Research Paper](docs/thesis.pdf)**: Relevant publication

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/ecdogaroglu/polaris.git
cd polaris
pip install -e .

# Run experiments
python -m polaris.simulation --help
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use POLARIS in your research, please cite:

```bibtex
@software{polaris2025,
  title={POLARIS: Partially Observable Learning with Active Reinforcement In Social Environments},
  author={Ege Can Dogaroglu},
  year={2025},
  url={https://github.com/ecdogaroglu/polaris}
}
```

## 🙏 Acknowledgments

- Brandl et al. for social learning foundations
- Keller & Rady for strategic experimentation framework
- PyTorch team for the excellent deep learning framework
- PyTorch Geometric team for graph neural network tools

---

<div align="center">

**Built with ❤️ for the multi-agent learning community**

[⭐ Star on GitHub](https://github.com/ecdogaroglu/polaris) • [🐛 Report Issues](https://github.com/ecdogaroglu/polaris/issues) • [💬 Discussions](https://github.com/ecdogaroglu/polaris/discussions)

</div> 