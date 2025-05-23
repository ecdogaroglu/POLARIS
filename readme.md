# POLARIS

<div align="center">

**P**artially **O**bservable **L**earning with **A**ctive **R**einforcement **I**n **S**ocial Environments

*A theoretically-grounded multi-agent reinforcement learning framework for strategic social learning*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🔬 Theoretical Foundations](#-theoretical-foundations) • [💡 Examples](#-examples)

</div>

---

## 🎯 Overview

POLARIS bridges economic social learning theory and multi-agent reinforcement learning through a novel theoretical framework: **Partially Observable Active Markov Games (POAMGs)**. Unlike traditional approaches that treat multi-agent learning as a technical challenge, POLARIS models strategic adaptation and policy evolution as fundamental features of social learning environments.

### 🏆 Key Features

- **🧠 Theoretical Rigor**: Formal mathematical framework with convergence guarantees and equilibrium analysis
- **🎮 Strategic Sophistication**: Models how agents influence others' learning processes under partial observability
- **🤝 Advanced Architectures**: Graph Neural Networks, Transformers, and Temporal Attention for sophisticated belief processing
- **🔄 Continual Learning**: Synaptic Intelligence prevents catastrophic forgetting in evolving social environments
- **📊 Empirical Validation**: Validates theoretical predictions in canonical social learning scenarios
- **⚡ Scalable Implementation**: Optimized for modern GPU/MPS acceleration

## 🧮 Theoretical Foundations

POLARIS introduces **Partially Observable Active Markov Games (POAMGs)**, a formalism that extends Active Markov Games [[Kim et al., 2022]](https://arxiv.org/abs/2202.02546) to partially observable settings where agents must learn from limited information while reasoning about others' strategic adaptations.

### 🔬 Core Theoretical Contributions

#### **1. POAMGs Framework**
We formalize social learning as a tuple $M_n = \langle I, S, \mathbf{A}, T, \mathbf{O}, \mathbf{R}, \mathbf{\Theta}, \mathbf{U} \rangle$ where:
- Agents maintain **belief states** about hidden world states
- **Policy evolution** is explicitly modeled through update functions $\mathbf{U}$
- **Strategic influence** on others' learning is captured through active equilibrium concepts

Unlike traditional frameworks that assume static policies or treat adaptation as noise, POAMGs incorporate policy evolution as an integral part of the environment dynamics, extending the Active Markov Game formulation of Kim et al. (2022) to handle partial observability.

#### **2. Convergence Guarantees**
**Theorem (Stochastically Stable Distribution):** Under mild regularity conditions, the joint process of states, beliefs, and policy parameters converges to a unique stochastically stable distribution $\mu^*$, ensuring well-defined limiting behavior despite non-stationary learning dynamics.

This provides theoretical guarantees that social learning processes reach stable configurations regardless of initial conditions—a crucial property for long-term strategic planning.

#### **3. Policy Gradient Theorems**
We derive novel policy gradient theorems for both average and discounted reward criteria:

**Average Reward Policy Gradient:**

$$\nabla_{\theta^i} J^i(\theta^i) = \sum_{s,\mathbf{b},\boldsymbol{\theta}} \mu(s,\mathbf{b},\boldsymbol{\theta}) \sum_{a^i} \nabla_{\theta^i} \pi^i(a^i|b^i;\theta^i) \sum_{\mathbf{a}^{-i}} \pi^{-i}(\mathbf{a}^{-i}|\mathbf{b}^{-i};\boldsymbol{\theta}^{-i}) q^i(s,\mathbf{b},\boldsymbol{\theta},\mathbf{a})$$

**Discounted Return Policy Gradient:**

$$\nabla_{\theta^i} J^{i}_{\gamma}(\theta^i) = \frac{1}{1-\gamma} \sum_{s,\mathbf{b},\boldsymbol{\theta}} d^{\pi}(s,\mathbf{b},\boldsymbol{\theta}) \sum_{a^i} \nabla_{\theta^i} \pi^i(a^i|b^i;\theta^i) q^i(s,\mathbf{b},\boldsymbol{\theta},\mathbf{a})$$

where:
- $\theta^i$ are agent $i$'s policy parameters
- $\mathbf{b} = (b^1, \ldots, b^n)$ are joint belief states  
- $\boldsymbol{\theta} = (\theta^1, \ldots, \theta^n)$ are joint policy parameters
- $\pi^i(a^i|b^i;\theta^i)$ is agent $i$'s policy conditioned on beliefs
- $q^i(s,\mathbf{b},\boldsymbol{\theta},\mathbf{a})$ is the action-value function
- $\mu(\cdot)$ and $d^{\pi}(\cdot)$ are the stationary and discounted visitation measures

These theorems extend classical policy gradients to **belief-conditioned policies** in **non-stationary multi-agent environments**, providing the mathematical foundation for strategic learning algorithms.

#### **4. Equilibrium Concepts**
**Definition (Partially Observable Active Equilibrium):** A configuration $\boldsymbol{\theta}^*$ where no agent can improve their long-term reward by unilaterally changing their policy or learning strategy, accounting for:
- **Partial observability** through belief states
- **Strategic adaptation** through policy evolution
- **Long-term consequences** of influencing others' learning

This equilibrium concept captures sophisticated strategic reasoning while remaining computationally tractable through policy gradient optimization.

### 📈 Theoretical Insights

Our framework reveals several fundamental insights about social learning:

1. **Strategic Teaching**: Agents may choose seemingly suboptimal actions to influence others' beliefs and future behaviors
2. **Information Revelation**: Strategic considerations affect how agents reveal private information through their actions  
3. **Learning Barriers**: Even optimal social learning strategies face fundamental limits determined by signal informativeness
4. **Network Effects**: Social network topology significantly influences both learning speed and strategic behavior

These insights emerge from explicitly modeling how agents reason about and influence others' learning processes, building upon the active influence mechanisms introduced by Kim et al. (2022) but extended to settings with partial observability and belief-based decision making.

## 🏗️ Architecture

POLARIS follows a modular design that implements our theoretical framework:

```
polaris/
├── 🤖 agents/                 # POLARIS agent implementation
│   ├── polaris_agent.py      # Main POAMG-based agent
│   ├── components/           # Modular agent components
│   │   ├── belief.py         # Belief state processing (Transformers)
│   │   ├── inference.py      # Opponent modeling (Variational inference)
│   │   ├── policy.py         # Policy networks (Discrete/Continuous)
│   │   └── critics.py        # Value function approximation
│   └── memory/               # Experience replay systems
├── 🌍 environments/          # Canonical social learning scenarios
│   ├── social_learning.py    # Brandl social learning model
│   └── strategic_exp.py      # Keller-Rady strategic experimentation
├── 🧠 networks/              # Advanced neural architectures
│   ├── gnn.py               # Graph Neural Networks with temporal attention
│   └── transformer.py       # Transformer belief processors
├── 🎯 algorithms/            # Continual learning algorithms
│   └── regularization/      # Prevent catastrophic forgetting
│       ├── si.py           # Synaptic Intelligence
│       └── ewc.py          # Elastic Weight Consolidation
├── 🚂 training/             # Training infrastructure
│   ├── trainer.py          # Policy gradient optimization
│   └── evaluator.py        # Performance evaluation
├── 📊 visualization/        # Analysis and plotting tools
├── 🔧 utils/               # Utilities (device management, encoding)
└── ⚙️ config/              # Configuration system
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

### 🧪 Theoretical Validation

POLARIS validates key theoretical predictions from economic social learning:

#### **Learning Barrier Theorem**
For any strategy profile, some agent's learning rate is bounded by the **Jeffreys divergence** between signal distributions, regardless of network size:

$$\min_i r^i(\sigma) \leq r_{\text{bdd}} = \min_{\theta \neq \theta'} \left[ D_{\text{KL}}(\mu_\theta \| \mu_{\theta'}) + D_{\text{KL}}(\mu_{\theta'} \| \mu_\theta) \right]$$

#### **Coordination Benefits Theorem**
In large, well-connected networks, all agents can achieve learning rates above the coordination bound:

$$\min_i r^i(\sigma) \geq r_{\text{crd}} - \varepsilon = \min_{\theta \neq \theta'} D_{\text{KL}}(\mu_\theta \| \mu_{\theta'}) - \varepsilon$$

where:
- $r^i(\sigma)$ is agent $i$'s learning rate under strategy profile $\sigma$
- $D_{\text{KL}}(\mu_\theta \| \mu_{\theta'})$ is the Kullback-Leibler divergence between signal distributions
- $\varepsilon > 0$ is arbitrarily small for sufficiently large networks

### 🧪 Multi-Environment Support

#### Brandl Social Learning Environment
- **Discrete Action Spaces**: Binary or multi-choice decisions
- **Signal Accuracy Control**: Configurable information quality (default: 0.75)
- **Network Topologies**: Complete, ring, star, random networks
- **Belief Dynamics**: Sophisticated belief updating mechanisms

#### Strategic Experimentation Environment (Keller-Rady)
- **Continuous Action Spaces**: Real-valued strategic choices with Lévy processes
- **Exploration-Exploitation Trade-offs**: Dynamic strategy adaptation
- **Free-rider Problems**: Strategic coordination challenges
- **Background Information**: Exogenous information arrival

### 🧠 Advanced Neural Architectures

#### Graph Neural Networks (GNNs)
```bash
# Enable GNN with temporal attention
python -m polaris.simulation \
    --environment-type brandl \
    --use-gnn \
    --gnn-layers 3 \
    --attn-heads 8 \
    --temporal-window 10
```

Our GNN implementation features:
- **Temporal Attention**: Aggregates information across time horizons
- **Belief-Action Fusion**: Combines private beliefs with observed actions
- **Dynamic Network Topology**: Adapts to changing social connections

#### Transformer Belief Processors
Advanced belief state processing using Transformer architectures:
- **Sequence Modeling**: Processes observation histories for belief updating
- **Attention Mechanisms**: Focuses on relevant historical information
- **Continuous/Discrete Support**: Handles both signal types seamlessly

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

**Synaptic Intelligence (SI)** preserves important network parameters while allowing adaptation to new scenarios, crucial for modeling realistic social learning where environments gradually evolve.

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

POLARIS includes comprehensive visualization tools for analyzing social learning dynamics:

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

POLARIS uses a flexible configuration system that supports both programmatic and command-line configuration:

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
  author={Ege Can Doğaroğlu},
  year={2025},
  url={https://github.com/ecdogaroglu/polaris}
}

@article{kim2022influencing,
  title={Influencing Others via Information Design: Policy Optimization in Multi-Agent Environments},
  author={Kim, Bobak and Fazel, Maryam and Sadigh, Dorsa},
  journal={arXiv preprint arXiv:2202.02546},
  year={2022}
}
```


---

<div align="center">


[⭐ Star on GitHub](https://github.com/ecdogaroglu/polaris) • [🐛 Report Issues](https://github.com/ecdogaroglu/polaris/issues) • [💬 Discussions](https://github.com/ecdogaroglu/polaris/discussions)

</div> 