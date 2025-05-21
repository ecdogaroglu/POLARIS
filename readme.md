# POLARIS - Partially Observable Learning with Active Reinforcement In Social Environments

POLARIS is a framework for multi-agent reinforcement learning in social learning environments, implementing different approaches from economic theory to understand and model strategic learning behavior.

## Overview

This repository implements two key frameworks from economic social learning theory:

1. **Strategic Experimentation Framework** (Keller, Rady (2020) Model): Agents allocate resources between a safe arm with known payoff and a risky arm with unknown state-dependent payoff, learning from their own and others' observed rewards.

2. **Learning without Experimentation Framework** (Brandl (2024) Model): Agents learn from private signals and by observing other agents' actions, without direct payoff feedback from their own actions.

Both frameworks are implemented using the Partially Observable Active Markov Game (POAMG) formalism, which extends standard reinforcement learning to account for partial observability and strategic influence between agents.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/polaris.git
cd polaris
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### General Command

The main script `experiment.py` allows running experiments with either framework:

```bash
python experiment.py --environment-type [brandl|strategic_experimentation] [options]
```

### Specialized Scripts

For convenience, we provide specialized scripts for each framework:

- For Learning without Experimentation framework:
```bash
python brandl_experiment.py [options]
```

- For Strategic Experimentation framework:
```bash
python keller_rady_experiment.py [options]
```

## Key Parameters

### Shared Parameters

- `--num-agents`: Number of agents (default: 2)
- `--num-states`: Number of possible states (default: 2)
- `--network-type`: Network structure (choices: 'complete', 'ring', 'star', 'random', default: 'complete')
- `--network-density`: Density for random networks (default: 0.5)
- `--horizon`: Total number of steps per episode (default: 1000)
- `--num-episodes`: Number of episodes for training (default: 1)
- `--seed`: Random seed (default: 42)

### Strategic Experimentation Parameters

- `--safe-payoff`: Deterministic payoff of the safe arm (default: 1.0)
- `--drift-rates`: Comma-separated list of drift rates for each state (default: "-0.5,0.5")
- `--diffusion-sigma`: Volatility of the diffusion component (default: 0.5)
- `--jump-rates`: Comma-separated list of Poisson rates for jumps in each state (default: "0.1,0.2")
- `--jump-sizes`: Comma-separated list of expected jump sizes in each state (default: "1.0,1.0")
- `--background-informativeness`: Informativeness of the background signal process (default: 0.1)
- `--time-step`: Size of time step for discretizing the Lévy processes (default: 0.1)

### Learning without Experimentation Framework Parameters

- `--signal-accuracy`: Accuracy of private signals (default: 0.75)


### Agent Parameters

- `--discount-factor`: Discount factor for RL (0 = average reward, default: 0.9)
- `--entropy-weight`: Entropy bonus weight (default: 0.5)
- `--use-gnn`: Use Graph Neural Network with temporal attention
- `--use-si`: Use Synaptic Intelligence to prevent catastrophic forgetting


## Architecture

The framework is built with a modular architecture:

1. **Base Environment**: Abstract base class that defines the interface for all environment implementations
2. **Specific Environments**:
   - `StrategicExperimentationEnvironment`: Implements the Keller-Rady model
   - `SocialLearningEnvironment`: Implements the Brandl model
3. **POLARIS Agent**: Reinforcement learning agent with three key components:
   - Belief processing module using Transformers
   - Inference learning module with GNNs
   - RL module with MASAC (Multi-Agent Soft Actor-Critic)

## Theoretical Foundations

- **Strategic Experimentation**: Based on Keller and Rady (2020), focusing on resource allocation under uncertainty with observable rewards.

- **Brandl Framework**: Based on Brandl (2024), focusing on learning without experimentation through action observation.


Both implementations are within the Partially Observable Active Markov Game framework, which captures how agents strategically influence each other's learning processes.

## Visualization and Analysis

The framework includes tools for visualizing results:

- Learning curves and convergence rates
- Network influence visualizations
- Belief state evolution
- Comparison with theoretical bounds

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{dogaroglu2024polaris,
  author = {Doğaroğlu, Ege Can},
  title = {POLARIS: Partially Observable Learning with Active Reinforcement In Social Environments},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ecdogaroglu/POLARIS}}
}
```

## License

MIT License
