import json
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class AgentConfig:
    """Configuration for POLARIS agents."""

    hidden_dim: int = 256
    belief_dim: int = 256
    latent_dim: int = 256
    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    entropy_weight: float = 0.01
    kl_weight: float = 0.01
    target_update_rate: float = 0.005
    use_gnn: bool = True
    use_si: bool = False
    si_importance: float = 100.0
    si_damping: float = 0.1
    si_exclude_final_layers: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training."""

    batch_size: int = 128
    buffer_capacity: int = 1000
    update_interval: int = 10
    save_interval: int = 1000
    num_episodes: int = 1
    horizon: int = 1000


@dataclass
class EnvironmentConfig:
    """Base environment configuration."""

    environment_type: Literal["brandl", "strategic_experimentation"] = "brandl"
    num_agents: int = 2
    num_states: int = 2
    network_type: Literal["complete", "ring", "star", "random"] = "complete"
    network_density: float = 0.5
    seed: int = 42


@dataclass
class BrandlConfig(EnvironmentConfig):
    """Brandl environment specific configuration."""

    signal_accuracy: float = 0.75


@dataclass
class StrategicExpConfig(EnvironmentConfig):
    """Strategic experimentation specific configuration."""

    safe_payoff: float = 1.0
    drift_rates: List[float] = field(default_factory=lambda: [-0.5, 0.5])
    diffusion_sigma: float = 0.5
    jump_rates: List[float] = field(default_factory=lambda: [0.1, 0.2])
    jump_sizes: List[float] = field(default_factory=lambda: [1.0, 1.0])
    background_informativeness: float = 0.1
    time_step: float = 0.1
    continuous_actions: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    device: Optional[str] = None
    output_dir: str = "results"
    exp_name: str = "experiment"
    save_model: bool = True
    load_model: Optional[str] = None
    eval_only: bool = False
    plot_internal_states: bool = False
    plot_allocations: bool = False
    latex_style: bool = False
    use_tex: bool = False
    visualize_si: bool = False

    def save(self, path: str):
        """Save configuration to JSON."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=lambda o: o.__dict__)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
