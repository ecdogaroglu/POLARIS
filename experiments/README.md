# POLARIS Refactored Experiment Scripts

This directory contains wrapper scripts for the refactored POLARIS implementation that replicate the functionality of the original experiment scripts. These scripts use the new configuration system and modular architecture while maintaining compatibility with the original experimental designs.

## Available Scripts

### 1. `keller_rady_experiment.py`
Replicates the functionality of `experiments/keller_rady_experiment.py` from the original implementation.

**Purpose**: Runs a single strategic experimentation experiment based on the Keller and Rady (2020) framework, where agents allocate resources between a safe arm with known payoff and a risky arm with unknown state-dependent payoff.

**Usage**:
```bash
python keller_rady_experiment.py [OPTIONS]
```

**Key Options**:
- `--agents N`: Number of agents (default: 2)
- `--episodes N`: Number of episodes (default: 1)
- `--horizon N`: Steps per episode (default: 10000)
- `--seed N`: Random seed (default: 42)
- `--use-gnn`: Use Graph Neural Network (default: True)
- `--use-si`: Use Synaptic Intelligence
- `--plot-allocations`: Plot agent allocations (default: True)
- `--plot-states`: Plot internal states (default: True)
- `--latex-style`: Use LaTeX styling for plots (default: True)
- `--eval`: Run in evaluation mode
- `--load PATH`: Load models from path

**Output**: 
- Agent allocations over time plots
- Internal state visualizations (beliefs, etc.)
- KL divergence to MPE (Markov Perfect Equilibrium)
- Model checkpoints (if training)

### 2. `brandl_experiment.py`
Replicates the functionality of `experiments/brandl_experiment.py` from the original implementation.

**Purpose**: Runs experiments with POLARIS agents in a social learning environment based on the Brandl framework, where agents learn without experimentation by observing others' actions and receiving private signals.

**Usage**:
```bash
python brandl_experiment.py [OPTIONS]
```

**Key Options**:
- `--agents N`: Number of agents (default: 8)
- `--episodes N`: Number of episodes (default: 1)
- `--horizon N`: Steps per episode (default: 1000)
- `--signal-accuracy F`: Signal accuracy (default: 0.75)
- `--network-type TYPE`: Network topology (complete, ring, star, random)
- `--network-density F`: Network density for random networks (default: 0.5)
- `--use-gnn`: Use Graph Neural Network (default: True)
- `--use-si`: Use Synaptic Intelligence
- `--plot-states`: Plot internal states (default: True)
- `--device TYPE`: Force specific device (cpu, mps, cuda) - overrides auto-detection

**Output**:
- Learning curves showing convergence to theoretical bounds
- Belief evolution visualizations
- Network structure plots
- Comparison with theoretical learning rates (autarky, bound, coordination)

### 3. `keller_rady_sweep.py`
Replicates the functionality of `experiments/keller_rady_sweep.py` from the original implementation.

**Purpose**: Runs the Keller-Rady experiment for different numbers of agents and creates a comparative plot of average cumulative allocation over time for each configuration.

**Usage**:
```bash
python keller_rady_sweep.py [OPTIONS]
```

**Key Options**:
- `--agent-counts N [N ...]`: List of agent counts to sweep over (default: [2,3,4,5,6,7,8])
- `--episodes N`: Number of episodes per configuration (default: 1)
- `--horizon N`: Steps per episode (default: 400)
- `--seed N`: Random seed (default: 0)

**Output**:
- Comparative plot: "Average Cumulative Allocation per Agent Over Time"
- Shows allocation trends for different agent group sizes
- Includes 95% confidence intervals
- Saved to `results/strategic_experimentation/sweep_allocations/`

## Key Differences from Original Scripts

### Architecture Changes
1. **Configuration System**: Uses the new `ExperimentConfig` dataclass system instead of argparse namespaces
2. **Modular Design**: Leverages the refactored trainer and visualization systems
3. **Type Safety**: Proper type hints and structured configurations
4. **Device Management**: Automatic GPU/CPU detection and management

### Maintained Compatibility
1. **Same Experiments**: Identical experimental designs and parameters
2. **Same Plots**: Generates the same visualization outputs
3. **Same Results**: Should produce equivalent experimental results
4. **Same CLI**: Similar command-line interfaces with enhanced options

### Enhanced Features
1. **Better Error Handling**: More robust error handling and validation
2. **Improved Logging**: Better progress reporting and debugging information
3. **Flexible Plotting**: Enhanced visualization options with LaTeX support
4. **Modular Components**: Easier to extend and customize

## Environment Setup

Make sure you're in the `Refactored/` directory and have installed the package:

```bash
cd Refactored/
pip install -e .
```

### Force CPU Usage (Apple Silicon Users)

If you're on Apple Silicon and want to force CPU usage (often faster than MPS for smaller models):

```bash
export TORCH_DEVICE=cpu
```

Add this to your shell profile (`.zshrc` or `.bash_profile`) to make it permanent:
```bash
echo 'export TORCH_DEVICE=cpu' >> ~/.zshrc
source ~/.zshrc
```

You can also force other devices:
- `export TORCH_DEVICE=mps` - Force MPS (Apple Silicon GPU)  
- `export TORCH_DEVICE=cuda` - Force CUDA (NVIDIA GPU)
- `export TORCH_DEVICE=cpu` - Force CPU

## Examples

### Quick Strategic Experimentation Test
```bash
python keller_rady_experiment.py --agents 2 --horizon 1000 --plot-allocations --plot-states
```

### Brandl Social Learning with 4 Agents
```bash
python brandl_experiment.py --agents 4 --signal-accuracy 0.8 --network-type complete
```

### Agent Count Sweep (Short)
```bash
python keller_rady_sweep.py --agent-counts 2 3 4 --horizon 200 --episodes 1
```

### Full Replication of Original Experiments
```bash
# Strategic experimentation (full)
python keller_rady_experiment.py --agents 2 --horizon 10000 --latex-style

# Brandl social learning (full)
python brandl_experiment.py --agents 8 --horizon 1000 --signal-accuracy 0.75

# Complete agent sweep
python keller_rady_sweep.py --agent-counts 2 3 4 5 6 7 8 --horizon 400
```

## Output Structure

Results are saved in the same directory structure as the original implementation:

```
results/
├── strategic_experimentation/
│   ├── strategic_experimentation_agents_N/
│   │   ├── plots/
│   │   ├── models/
│   │   └── config.json
│   └── sweep_allocations/
│       └── average_cumulative_allocation_per_agent_over_time.png
└── brandl_experiment_agents_N/
    ├── plots/
    ├── models/
    └── config.json
```

This maintains compatibility with existing analysis scripts and workflows. 