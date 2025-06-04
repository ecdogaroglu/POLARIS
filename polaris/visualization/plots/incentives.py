"""
Incentive plotting for POLARIS strategic experimentation experiments.
"""

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import MultiAgentPlotter


class IncentivePlotter(MultiAgentPlotter):
    """
    Plotter for agent incentives in strategic experimentation environments.

    Handles visualization of experimentation incentives over time based on
    the Keller and Rady (2020) framework.
    """

    def plot(self, metrics: Dict[str, Any], env, args, output_dir: Path):
        """
        Generate incentive plots for strategic experimentation.

        Args:
            metrics: Experiment metrics
            env: Environment object
            args: Command-line arguments
            output_dir: Directory to save plots
        """
        if not self.validate_metrics(metrics):
            return

        print("  📈 Generating incentive plots...")

        # Plot incentives over time
        if "agent_incentives" in metrics:
            self.plot_incentives_over_time(metrics, output_dir)

        # Plot incentive vs belief relationship
        if "agent_incentives" in metrics and "agent_beliefs" in metrics:
            self.plot_incentive_vs_belief(metrics, output_dir)

        # Plot incentive thresholds
        if "agent_incentives" in metrics:
            self.plot_incentive_thresholds(metrics, env, output_dir)

    def plot_incentives_over_time(self, metrics: Dict, output_dir: Path):
        """
        Plot agent incentives over time.

        Args:
            metrics: Dictionary of metrics including agent_incentives
            output_dir: Directory to save plots
        """
        if "agent_incentives" not in metrics:
            print("No incentive data available for plotting")
            return

        # Create figure
        fig, ax = self.create_figure()
        colors = self.get_colors()

        incentives = metrics["agent_incentives"]

        # Plot incentives for each agent
        for i, (agent_id, agent_incentives) in enumerate(incentives.items()):
            if len(agent_incentives) == 0:
                continue

            # Convert tensors to numpy if needed
            if isinstance(agent_incentives[0], torch.Tensor):
                incentives_np = [inc.item() for inc in agent_incentives]
            else:
                incentives_np = agent_incentives

            timesteps = range(len(incentives_np))
            agent_color = self.get_agent_color(i, colors)

            try:
                agent_id_int = int(agent_id)
                label = f"Agent {agent_id_int}"
            except (ValueError, TypeError):
                label = f"Agent {agent_id}"

            ax.plot(
                timesteps, incentives_np, label=label, color=agent_color, linewidth=2.5
            )

        # Formatting
        self.set_title(ax, "Agent Experimentation Incentives Over Time")
        self.set_labels(ax, "Time Steps", "Incentive to Experiment I(b)")
        ax.legend(fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)

        # Save figure
        save_path = output_dir / "agent_incentives.png"
        self.save_figure(fig, save_path, "Agent Incentives")

    def plot_incentive_vs_belief(self, metrics: Dict, output_dir: Path):
        """
        Plot agent incentives as a function of belief to show the relationship.

        Args:
            metrics: Dictionary of metrics including 'agent_beliefs' and 'agent_incentives'
            output_dir: Directory to save plots
        """
        if "agent_beliefs" not in metrics or "agent_incentives" not in metrics:
            print("No belief or incentive data available for plotting incentive vs belief.")
            return

        # Create figure
        fig, ax = self.create_figure()
        colors = self.get_colors()

        # Plot for each agent
        for i, agent_id in enumerate(metrics["agent_incentives"].keys()):
            if (
                agent_id not in metrics["agent_beliefs"]
                or agent_id not in metrics["agent_incentives"]
            ):
                continue

            beliefs = []
            incentives = []

            # Collect data points (skip initial steps for stability)
            for t, incentive in enumerate(metrics["agent_incentives"][agent_id]):
                if 100 < t < len(metrics["agent_beliefs"][agent_id]):
                    belief_good = metrics["agent_beliefs"][agent_id][t]
                    if belief_good is not None and not np.isnan(belief_good):
                        beliefs.append(belief_good)
                        incentives.append(incentive)

            if len(beliefs) < 2:
                continue

            agent_color = self.get_agent_color(i, colors)

            # Scatter plot of (belief, incentive) pairs
            ax.scatter(
                beliefs,
                incentives,
                label=f"Agent {int(agent_id)}",
                alpha=0.6,
                s=15,
                color=agent_color,
            )

            # Add regression line if enough points
            if len(beliefs) >= 2:
                x = np.array(beliefs)
                y = np.array(incentives)

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
        self.set_labels(ax, "Belief in Good State", "Incentive to Experiment I(b)")
        self.set_title(ax, "Experimentation Incentive vs. Belief")
        ax.legend(fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)

        # Save figure
        save_path = output_dir / "incentive_vs_belief.png"
        self.save_figure(fig, save_path, "Incentive vs Belief")

    def plot_incentive_thresholds(self, metrics: Dict, env, output_dir: Path):
        """
        Plot incentive thresholds and regions for experimentation decisions.

        Args:
            metrics: Dictionary of metrics including agent_incentives
            env: Environment object to get parameters
            output_dir: Directory to save plots
        """
        if "agent_incentives" not in metrics:
            print("No incentive data available for plotting thresholds")
            return

        # Create figure
        fig, ax = self.create_figure()
        colors = self.get_colors()

        # Get environment parameters
        k0 = env.background_informativeness
        n = env.num_agents
        
        # Find the maximum timesteps length to use for fill_between
        max_timesteps = 0
        for agent_incentives in metrics["agent_incentives"].values():
            if len(agent_incentives) > max_timesteps:
                max_timesteps = len(agent_incentives)
        
        # If no agent has incentives, return early
        if max_timesteps == 0:
            print("No incentive data found for any agent")
            return

        # Plot incentive thresholds
        ax.axhline(y=k0, color="red", linestyle="--", linewidth=2, 
                  label=f"Lower threshold (k₀ = {k0})")
        ax.axhline(y=k0 + n - 1, color="orange", linestyle="--", linewidth=2,
                  label=f"Upper threshold (k₀ + n - 1 = {k0 + n - 1})")

        # Plot incentives for each agent
        for i, (agent_id, agent_incentives) in enumerate(metrics["agent_incentives"].items()):
            if len(agent_incentives) == 0:
                continue

            # Convert tensors to numpy if needed
            if isinstance(agent_incentives[0], torch.Tensor):
                incentives_np = [inc.item() for inc in agent_incentives]
            else:
                incentives_np = agent_incentives

            timesteps = range(len(incentives_np))
            agent_color = self.get_agent_color(i, colors)

            try:
                agent_id_int = int(agent_id)
                label = f"Agent {agent_id_int}"
            except (ValueError, TypeError):
                label = f"Agent {agent_id}"

            ax.plot(
                timesteps, incentives_np, label=label, color=agent_color, linewidth=2.5
            )

        # Add shaded regions for different experimentation regimes
        y_min, y_max = ax.get_ylim()
        
        # No experimentation region (I ≤ k₀)
        ax.fill_between([0, max_timesteps], y_min, k0, alpha=0.1, color="red", 
                       label="No experimentation")
        
        # Partial experimentation region (k₀ < I < k₀ + n - 1)
        ax.fill_between([0, max_timesteps], k0, k0 + n - 1, alpha=0.1, color="yellow",
                       label="Partial experimentation")
        
        # Full experimentation region (I ≥ k₀ + n - 1)
        ax.fill_between([0, max_timesteps], k0 + n - 1, y_max, alpha=0.1, color="green",
                       label="Full experimentation")

        # Formatting
        self.set_title(ax, "Experimentation Incentives and Thresholds")
        self.set_labels(ax, "Time Steps", "Incentive to Experiment I(b)")
        ax.legend(fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)

        # Save figure
        save_path = output_dir / "incentive_thresholds.png"
        self.save_figure(fig, save_path, "Incentive Thresholds")

    def _apply_smoothing(self, data, window_size=None):
        """Apply moving average smoothing to data."""
        if window_size is None:
            window_size = min(10, len(data) // 10) if len(data) > 10 else 1

        if window_size <= 1:
            return data

        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode="valid") 