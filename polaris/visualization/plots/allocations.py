"""
Allocation plotting for POLARIS strategic experimentation experiments.
"""

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import MultiAgentPlotter


class AllocationPlotter(MultiAgentPlotter):
    """
    Plotter for agent allocations in strategic experimentation environments.

    Handles resource allocation visualization, policy analysis, and
    convergence to MPE (Markov Perfect Equilibrium).
    """

    def plot(self, metrics: Dict[str, Any], env, args, output_dir: Path):
        """
        Generate allocation plots for strategic experimentation.

        Args:
            metrics: Experiment metrics
            env: Environment object
            args: Command-line arguments
            output_dir: Directory to save plots
        """
        if not self.validate_metrics(metrics):
            return

        print("  💰 Generating allocation plots...")

        # Plot basic allocations over time
        if "allocations" in metrics:
            self.plot_allocations_over_time(metrics, output_dir)

        # Plot KL divergence to MPE
        if "policy_kl_divergence" in metrics:
            self.plot_kl_divergence(metrics, output_dir)

        # Plot policy cutoff analysis
        if "agent_beliefs" in metrics and "allocations" in metrics:
            self.plot_policy_cutoff_vs_belief(metrics, output_dir)

    def plot_allocations_over_time(self, metrics: Dict, output_dir: Path):
        """
        Plot agent allocations over time.

        Args:
            metrics: Dictionary of metrics including allocations
            output_dir: Directory to save plots
        """
        if "allocations" not in metrics:
            print("No allocation data available for plotting")
            return

        # Create figure
        fig, ax = self.create_figure()
        colors = self.get_colors()

        allocations = metrics["allocations"]

        # Plot allocations for each agent
        for i, (agent_id, agent_allocations) in enumerate(allocations.items()):
            if len(agent_allocations) == 0:
                continue

            # Convert tensors to numpy if needed
            if isinstance(agent_allocations[0], torch.Tensor):
                allocations_np = [a.item() for a in agent_allocations]
            else:
                allocations_np = agent_allocations

            timesteps = range(len(allocations_np))
            agent_color = self.get_agent_color(i, colors)

            try:
                agent_id_int = int(agent_id)
                label = f"Agent {agent_id_int}"
            except (ValueError, TypeError):
                label = f"Agent {agent_id}"

            ax.plot(
                timesteps, allocations_np, label=label, color=agent_color, linewidth=2.5
            )

        # Formatting
        self.set_title(ax, "Agent Resource Allocations Over Time")
        self.set_labels(ax, "Time Steps", "Allocation to Risky Arm")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)

        # Save figure
        save_path = output_dir / "agent_allocations.png"
        self.save_figure(fig, save_path, "Agent Allocations")

    def plot_kl_divergence(self, metrics: Dict, output_dir: Path):
        """
        Plot KL divergence between agent policies and the MPE over time.

        Args:
            metrics: Dictionary of metrics including policy_kl_divergence
            output_dir: Directory to save plots
        """
        if "policy_kl_divergence" not in metrics or not metrics["policy_kl_divergence"]:
            print("No KL divergence data available for plotting")
            return

        # Create figure
        fig, ax = self.create_figure()
        colors = self.get_colors()

        kl_divergences = metrics["policy_kl_divergence"]

        # Plot KL divergence for each agent
        for i, (agent_id, agent_kl) in enumerate(kl_divergences.items()):
            if len(agent_kl) == 0:
                continue

            timesteps = range(len(agent_kl))
            agent_color = self.get_agent_color(i, colors)

            # Apply smoothing for better visualization
            smoothed_kl = self._apply_smoothing(agent_kl)

            ax.plot(
                timesteps[: len(smoothed_kl)],
                smoothed_kl,
                label=f"Agent {agent_id}",
                color=agent_color,
                linewidth=2.5,
            )

        # Formatting
        self.set_title(ax, "KL Divergence to MPE Policy Over Time")
        self.set_labels(ax, "Time Steps", "KL Divergence")
        ax.legend(fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)

        # Save figure
        save_path = output_dir / "kl_divergence.png"
        self.save_figure(fig, save_path, "KL Divergence to MPE")

    def plot_policy_cutoff_vs_belief(self, metrics: Dict, output_dir: Path):
        """
        Plot agent allocation (policy) as a function of belief to visualize cutoff beliefs.

        Args:
            metrics: Dictionary of metrics including 'agent_beliefs' and 'allocations'
            output_dir: Directory to save plots
        """
        if "agent_beliefs" not in metrics or "allocations" not in metrics:
            print("No belief or allocation data available for plotting cutoff.")
            return

        # Create figure
        fig, ax = self.create_figure()
        colors = self.get_colors()

        # Plot for each agent
        for i, agent_id in enumerate(metrics["allocations"].keys()):
            if (
                agent_id not in metrics["agent_beliefs"]
                or agent_id not in metrics["allocations"]
            ):
                continue

            beliefs = []
            allocations = []

            # Collect data points (skip initial steps for stability)
            for t, alloc in enumerate(metrics["allocations"][agent_id]):
                if 100 < t < len(metrics["agent_beliefs"][agent_id]):
                    belief_good = metrics["agent_beliefs"][agent_id][t]
                    if belief_good is not None and not np.isnan(belief_good):
                        beliefs.append(belief_good)
                        allocations.append(alloc)

            if len(beliefs) < 2:
                continue

            agent_color = self.get_agent_color(i, colors)

            # Scatter plot of (belief, allocation) pairs
            ax.scatter(
                beliefs,
                allocations,
                label=f"Agent {int(agent_id)}",
                alpha=0.6,
                s=15,
                color=agent_color,
            )

            # Add regression line if enough points
            if len(beliefs) >= 2:
                x = np.array(beliefs)
                y = np.array(allocations)

                # Only fit if there is variance in x
                if np.std(x) > 1e-6:
                    coeffs = np.polyfit(x, y, 1)
                    reg_x = np.linspace(np.min(x), np.max(x), 100)
                    reg_y = np.polyval(coeffs, reg_x)
                    ax.plot(
                        reg_x,
                        reg_y,
                        linestyle="--",
                        linewidth=2,
                        color=agent_color,
                        alpha=0.8,
                    )

        # Formatting
        self.set_labels(ax, "Belief in Good State", "Allocation to Risky Arm")
        self.set_title(ax, "Policy Cutoff: Allocation vs. Belief")
        ax.legend(fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)

        # Save figure
        save_path = output_dir / "policy_cutoff_vs_belief.png"
        self.save_figure(fig, save_path, "Policy Cutoff vs Belief")

    def _apply_smoothing(self, data, window_size=None):
        """Apply moving average smoothing to data."""
        if window_size is None:
            window_size = min(10, len(data) // 10) if len(data) > 10 else 1

        if window_size <= 1:
            return data

        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode="valid")
