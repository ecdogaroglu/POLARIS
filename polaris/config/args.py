"""
Command-line argument parsing for POLARIS experiments.
"""

import argparse

from ..utils.device import get_best_device


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate POLARIS in social learning"
    )

    # Environment selection
    parser.add_argument(
        "--environment-type",
        type=str,
        default="brandl",
        choices=["brandl", "strategic_experimentation"],
        help="Type of social learning environment to use",
    )

    # Shared environment parameters
    parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
    parser.add_argument(
        "--num-states", type=int, default=2, help="Number of possible states"
    )
    parser.add_argument(
        "--network-type",
        type=str,
        default="complete",
        choices=["complete", "ring", "star", "random"],
        help="Network structure",
    )
    parser.add_argument(
        "--network-density", type=float, default=0.5, help="Density for random networks"
    )
    parser.add_argument(
        "--horizon", type=int, default=1000, help="Total number of steps"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of episodes for training (true state is reset between episodes)",
    )

    # Brandl environment-specific parameters
    parser.add_argument(
        "--signal-accuracy",
        type=float,
        default=0.75,
        help="Accuracy of private signals (Brandl environment)",
    )

    # Strategic experimentation environment-specific parameters
    parser.add_argument(
        "--safe-payoff",
        type=float,
        default=1.0,
        help="Deterministic payoff of the safe arm (Strategic experimentation)",
    )
    parser.add_argument(
        "--drift-rates",
        type=str,
        default="-0.5,0.5",
        help="Comma-separated list of drift rates for each state (Strategic experimentation)",
    )
    parser.add_argument(
        "--diffusion-sigma",
        type=float,
        default=0.5,
        help="Volatility of the diffusion component (Strategic experimentation)",
    )
    parser.add_argument(
        "--jump-rates",
        type=str,
        default="0.1,0.2",
        help="Comma-separated list of Poisson rates for jumps in each state (Strategic experimentation)",
    )
    parser.add_argument(
        "--jump-sizes",
        type=str,
        default="1.0,1.0",
        help="Comma-separated list of expected jump sizes in each state (Strategic experimentation)",
    )
    parser.add_argument(
        "--background-informativeness",
        type=float,
        default=0.1,
        help="Informativeness of the background signal process (Strategic experimentation)",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.1,
        help="Size of time step for discretizing the Lévy processes (Strategic experimentation)",
    )
    parser.add_argument(
        "--continuous-actions",
        action="store_true",
        help="Use continuous action space for resource allocation in strategic experimentation",
    )

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--buffer-capacity", type=int, default=1000, help="Replay buffer capacity"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--update-interval", type=int, default=10, help="Steps between updates"
    )

    # Agent hyperparameters
    parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--belief-dim", type=int, default=256, help="Belief state dimension"
    )
    parser.add_argument(
        "--latent-dim", type=int, default=256, help="Latent space dimension"
    )
    parser.add_argument(
        "--discount-factor",
        type=float,
        default=0.9,
        help="Discount factor (0 = average reward)",
    )
    parser.add_argument(
        "--entropy-weight", type=float, default=0.5, help="Entropy bonus weight"
    )
    parser.add_argument(
        "--kl-weight", type=float, default=10, help="KL weight for inference"
    )

    # GNN configuration (always used)
    parser.add_argument(
        "--gnn-layers", type=int, default=2, help="Number of GNN layers"
    )
    parser.add_argument(
        "--attn-heads", type=int, default=4, help="Number of attention heads in GNN"
    )
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=5,
        help="Temporal window size for GNN memory",
    )

    # SI parameters (replacing EWC parameters)
    parser.add_argument(
        "--use-si",
        action="store_true",
        help="Use Synaptic Intelligence to prevent catastrophic forgetting",
    )
    parser.add_argument(
        "--si-importance",
        type=float,
        default=100.0,
        help="Importance factor for SI penalty (lambda)",
    )
    parser.add_argument(
        "--si-damping",
        type=float,
        default=0.1,
        help="Damping factor to prevent division by zero in SI calculation",
    )
    parser.add_argument(
        "--si-exclude-final-layers",
        action="store_true",
        help="Exclude final layers from SI protection to allow better adaptation to new tasks",
    )

    # Experiment settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default=get_best_device(),
        help="Device to use (cuda, mps, or cpu)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--exp-name", type=str, default="brandl_validation", help="Experiment name"
    )
    parser.add_argument("--save-model", action="store_true", help="Save agent models")
    parser.add_argument(
        "--load-model",
        type=str,
        nargs="?",
        const="auto",
        help="Load model (if specified without path: load final model, with path: load from specified path)",
    )

    # Network size comparison
    parser.add_argument(
        "--compare-sizes", action="store_true", help="Compare different network sizes"
    )
    parser.add_argument(
        "--network-sizes",
        type=str,
        default="2,4,6,10,20",
        help="Comma-separated list of network sizes",
    )

    # Training vs evaluation
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )
    parser.add_argument(
        "--train-then-evaluate",
        action="store_true",
        help="Train for specified episodes then evaluate",
    )

    # Compare environment frameworks
    parser.add_argument(
        "--compare-frameworks",
        action="store_true",
        help="Compare Brandl and Strategic Experimentation frameworks",
    )

    # Visualization options
    parser.add_argument(
        "--plot-internal-states",
        action="store_true",
        help="Generate detailed visualizations of internal states (belief, latent, decision boundaries)",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        default="both",
        choices=["belief", "latent", "both"],
        help="Type of internal state to plot (belief, latent, or both)",
    )
    parser.add_argument(
        "--plot-allocations",
        action="store_true",
        help="Plot agent allocations over time for strategic experimentation",
    )
    parser.add_argument(
        "--latex-style",
        action="store_true",
        help="Use LaTeX-style formatting for plots (publication quality)",
    )
    parser.add_argument(
        "--use-tex",
        action="store_true",
        help="Use actual LaTeX rendering for text in plots (requires LaTeX installation)",
    )
    parser.add_argument(
        "--visualize-si",
        action="store_true",
        help="Generate visualizations of Synaptic Intelligence importance scores",
    )

    args = parser.parse_args()

    # Process network sizes
    if hasattr(args, "network_sizes"):
        args.network_sizes_list = [int(size) for size in args.network_sizes.split(",")]

    # Process list parameters
    if hasattr(args, "drift_rates"):
        args.drift_rates_list = [float(rate) for rate in args.drift_rates.split(",")]
    if hasattr(args, "jump_rates"):
        args.jump_rates_list = [float(rate) for rate in args.jump_rates.split(",")]
    if hasattr(args, "jump_sizes"):
        args.jump_sizes_list = [float(size) for size in args.jump_sizes.split(",")]

    return args
