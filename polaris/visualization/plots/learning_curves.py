"""
Learning curve plotting for POLARIS experiments.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ...utils.math import calculate_learning_rate
from . import MultiAgentPlotter


class LearningCurvePlotter(MultiAgentPlotter):
    """
    Plotter for learning curves and performance metrics.

    Handles incorrect action probabilities, learning rates, and convergence analysis.
    """

    def plot(self, metrics: Dict, env, args, output_dir: Path, episodic_metrics=None):
        """
        Generate learning curve plots.

        Args:
            metrics: Combined metrics dictionary
            env: Environment object
            args: Command-line arguments
            output_dir: Directory to save plots
            episodic_metrics: Optional episodic metrics for confidence intervals
        """
        if not self.validate_metrics(metrics):
            self.create_empty_plot(
                output_dir, "learning_curves.png", "No learning data available"
            )
            return

        # Skip incorrect probability plots for strategic experimentation (Keller-Rady) experiments
        if hasattr(env, "safe_payoff"):
            print(
                "  📈 Skipping incorrect probability plots for strategic experimentation environment"
            )
            return

        print("  📈 Generating learning curves...")

        # If we have episodic metrics, plot with error bars
        if (
            episodic_metrics
            and "episodes" in episodic_metrics
            and len(episodic_metrics["episodes"]) > 1
        ):
            self._plot_episodic_learning_curves(episodic_metrics, args, output_dir)
        else:
            # Standard learning curves without confidence intervals
            self._plot_standard_learning_curves(metrics, args, output_dir)

    def _plot_episodic_learning_curves(
        self, episodic_metrics: Dict, args, output_dir: Path
    ):
        """Plot learning curves with confidence intervals from multiple episodes."""
        incorrect_probs_by_episode = []
        for ep in episodic_metrics["episodes"]:
            if "incorrect_probs" in ep:
                incorrect_probs_by_episode.append(ep["incorrect_probs"])

        if incorrect_probs_by_episode:
            episode_length = args.horizon
            num_episodes = len(episodic_metrics["episodes"])

            self.plot_mean_incorrect_action_probabilities_with_ci(
                incorrect_probs_by_episode,
                title=f"Mean Incorrect Action Probabilities with 95% CI ({num_episodes} episodes)",
                save_path=output_dir / "mean_incorrect_probs_with_ci.png",
                log_scale=True,
                episode_length=episode_length,
            )

    def _plot_standard_learning_curves(self, metrics: Dict, args, output_dir: Path):
        """Plot standard learning curves without confidence intervals."""
        incorrect_probs = self._process_incorrect_probabilities(
            metrics, metrics.get("num_agents", 2)
        )

        if incorrect_probs:
            self.plot_incorrect_action_probabilities(
                incorrect_probs,
                title="Incorrect Action Probabilities Over Time",
                save_path=output_dir / "incorrect_probs.png",
                log_scale=False,
                episode_length=args.horizon,
            )

    def plot_mean_incorrect_action_probabilities_with_ci(
        self,
        episodic_data: List[List[float]],
        title: str = "Mean Incorrect Action Probabilities with 95% CI",
        save_path: Optional[Path] = None,
        log_scale: bool = False,
        episode_length: Optional[int] = None,
    ):
        """
        Plot mean incorrect action probabilities across episodes with 95% confidence intervals.

        Args:
            episodic_data: List of incorrect probability lists for each episode
            title: Title of the plot
            save_path: Path to save the figure
            log_scale: Whether to use logarithmic scale for y-axis
            episode_length: Length of each episode
        """
        if not episodic_data or len(episodic_data) < 2:
            print("Need at least 2 episodes to plot mean with confidence intervals.")
            return

        num_episodes = len(episodic_data)
        min_length = min(len(ep_data) for ep_data in episodic_data)

        if min_length == 0:
            print("Episodes have no data. Skipping CI plot.")
            return

        # Create figure
        fig, ax = self.create_figure()
        colors = self.get_colors()

        # Check if we have multi-agent data (arrays at each time step) or scalar data
        first_episode_first_step = episodic_data[0][0]
        is_multi_agent = isinstance(first_episode_first_step, (list, np.ndarray))
        
        if is_multi_agent:
            # Handle multi-agent case: average across agents at each time step
            processed_data = []
            for ep_data in episodic_data:
                # For each episode, convert multi-agent data to average across agents
                episode_averages = []
                for step_data in ep_data[:min_length]:
                    if isinstance(step_data, (list, np.ndarray)):
                        # Average across agents for this time step
                        episode_averages.append(np.mean(step_data))
                    else:
                        # Single value (shouldn't happen in multi-agent case, but handle it)
                        episode_averages.append(step_data)
                processed_data.append(episode_averages)
        else:
            # Handle single-agent case: data is already scalar
            processed_data = [ep_data[:min_length] for ep_data in episodic_data]

        # Convert to numpy array for easier computation
        data_array = np.array(processed_data)

        # Calculate mean and confidence intervals
        mean_probs = np.mean(data_array, axis=0)
        std_probs = np.std(data_array, axis=0)

        # Use t-distribution for small sample sizes
        t_value = stats.t.ppf(0.975, num_episodes - 1)  # 95% CI
        ci = t_value * std_probs / np.sqrt(num_episodes)

        time_steps = np.arange(min_length)

        # Plot mean line
        if log_scale:
            ax.semilogy(
                time_steps,
                mean_probs,
                label=f"Mean ({num_episodes} episodes)",
                color=colors[0],
                linewidth=2,
            )
        else:
            ax.plot(
                time_steps,
                mean_probs,
                label=f"Mean ({num_episodes} episodes)",
                color=colors[0],
                linewidth=2,
            )

        # Plot confidence interval
        ax.fill_between(
            time_steps,
            mean_probs - ci,
            mean_probs + ci,
            alpha=0.3,
            color=colors[0],
            label="95% Confidence Interval",
        )

        # Formatting
        self.set_labels(ax, "Time Step", "Incorrect Action Probability")
        self.set_title(ax, title)
        ax.legend(fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)

        if save_path:
            self.save_figure(fig, save_path, title)

    def plot_incorrect_action_probabilities(
        self,
        incorrect_probs: Dict[int, List[float]],
        title: str = "Incorrect Action Probabilities Over Time",
        save_path: Optional[Path] = None,
        max_steps: Optional[int] = None,
        log_scale: bool = False,
        show_learning_rates: bool = True,
        episode_length: Optional[int] = None,
    ):
        """
        Plot incorrect action probabilities for each agent.

        Args:
            incorrect_probs: Dictionary mapping agent IDs to probability lists
            title: Title of the plot
            save_path: Path to save the figure
            max_steps: Maximum number of steps to plot
            log_scale: Whether to use logarithmic scale
            show_learning_rates: Whether to show learning rates in legend
            episode_length: Length of each episode
        """
        if not incorrect_probs:
            print("No incorrect probability data available.")
            return

        # Create figure
        fig, ax = self.create_figure()
        colors = self.get_colors()

        # Calculate learning rates if requested
        learning_rates = {}
        if show_learning_rates:
            for agent_id, probs in incorrect_probs.items():
                learning_rates[agent_id] = calculate_learning_rate(probs)

        # Plot each agent
        for i, (agent_id, probs) in enumerate(incorrect_probs.items()):
            if not probs:
                continue

            # Limit steps if specified
            if max_steps:
                probs = probs[:max_steps]

            time_steps = np.arange(len(probs))
            agent_color = self.get_agent_color(i, colors)

            # Create label
            if show_learning_rates and agent_id in learning_rates:
                label = f"Agent {agent_id} (r={learning_rates[agent_id]:.4f})"
            else:
                label = f"Agent {agent_id}"

            # Plot the line
            if log_scale:
                ax.semilogy(
                    time_steps, probs, label=label, color=agent_color, linewidth=2.5
                )
            else:
                ax.plot(
                    time_steps, probs, label=label, color=agent_color, linewidth=2.5
                )

        # Formatting
        self.set_labels(ax, "Time Step", "Incorrect Action Probability")
        self.set_title(ax, title)
        ax.legend(fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)

        if save_path:
            self.save_figure(fig, save_path, title)

    def _process_incorrect_probabilities(
        self, metrics: Dict, num_agents: int
    ) -> Dict[int, List[float]]:
        """
        Process incorrect probabilities from metrics for plotting.

        Args:
            metrics: Metrics dictionary
            num_agents: Number of agents

        Returns:
            Dictionary mapping agent IDs to incorrect probability lists
            
        Raises:
            ValueError: If no action probabilities data is available
        """
        agent_incorrect_probs = metrics.get("action_probs", {})

        # Check if action_probs is empty and raise error instead of fallback
        if not agent_incorrect_probs or not any(agent_incorrect_probs.values()):
            raise ValueError(
                "No action probabilities data available for plotting. "
                "Ensure that metrics are properly updated with action probability data."
            )

        return agent_incorrect_probs
