"""
Belief state visualization for POLARIS experiments.
"""

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import MultiAgentPlotter


class BeliefPlotter(MultiAgentPlotter):
    """
    Plotter for agent belief states and belief distributions.

    Handles visualization of belief evolution, belief distributions,
    and belief-dependent analysis.
    """

    def plot(self, metrics: Dict[str, Any], env, args, output_dir: Path):
        """
        Generate belief visualizations.

        Args:
            metrics: Experiment metrics
            env: Environment object
            args: Command-line arguments
            output_dir: Directory to save plots
        """
        if not self.validate_metrics(metrics):
            return

        print("  🧠 Generating belief visualizations...")

        agent_ids = self.get_agent_ids(metrics)

        # Skip individual belief state and distribution plots for Brandl experiment
        # Only generate the combined social learning beliefs plot

        # Plot good belief over time for strategic experimentation
        if hasattr(env, "safe_payoff") and "agent_beliefs" in metrics:
            self.plot_good_belief_over_time(metrics, output_dir)

        # Plot combined belief distributions for social learning (Brandl experiment)
        if (
            not hasattr(env, "safe_payoff")
            and "belief_distributions" in metrics
            and len(agent_ids) > 1
        ):
            self.plot_social_learning_beliefs(metrics, env, output_dir)

    def plot_belief_states(self, metrics: Dict, agent_id: int, output_dir: Path):
        """
        Plot belief states over time for a specific agent.

        Args:
            metrics: Dictionary of metrics including belief_states
            agent_id: ID of the agent to plot for
            output_dir: Directory to save plots
        """
        # Disabled for Brandl experiment - individual plots not needed
        return

        belief_states = metrics["belief_states"][agent_id]
        if not belief_states:
            print(f"Empty belief state history for agent {agent_id}")
            return

        # Convert tensors to numpy
        belief_arrays = self._process_belief_tensors(belief_states)
        if not belief_arrays:
            print(f"Could not process belief state data for agent {agent_id}")
            return

        # Create figure
        fig, ax = self.create_figure()

        # Plot each belief dimension (limit to 10 for readability)
        belief_dim = len(belief_arrays[0])
        timesteps = range(len(belief_arrays))
        colors = self.get_colors()

        for dim in range(min(belief_dim, 10)):
            values = [belief[dim] for belief in belief_arrays]
            ax.plot(
                timesteps, values, label=f"Dim {dim}", color=colors[dim % len(colors)]
            )

        # Formatting
        ax.set_title(f"Agent {agent_id} Belief State Evolution")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Belief State Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save figure
        save_path = output_dir / f"agent_{agent_id}_belief_states.png"
        self.save_figure(fig, save_path, f"Agent {agent_id} Belief States")

    def plot_belief_distributions(
        self, metrics: Dict, agent_id: int, num_states: int, output_dir: Path
    ):
        """
        Plot belief distributions over time for a specific agent.

        Args:
            metrics: Dictionary of metrics
            agent_id: Agent ID to plot for
            num_states: Number of possible states
            output_dir: Directory to save plots
        """
        # Disabled for Brandl experiment - individual plots not needed
        return

        belief_distributions = metrics["belief_distributions"][agent_id]
        if not belief_distributions:
            print(f"Empty belief distribution history for agent {agent_id}")
            return

        # Process belief distributions
        belief_values = []
        for belief_dist in belief_distributions:
            if isinstance(belief_dist, torch.Tensor):
                if belief_dist.dim() > 1:
                    belief_np = belief_dist.squeeze().detach().cpu().numpy()
                else:
                    belief_np = belief_dist.detach().cpu().numpy()
            else:
                belief_np = np.array(belief_dist)

            # Ensure correct dimensionality
            if belief_np.size >= num_states:
                belief_np = belief_np[:num_states]
            else:
                padded = np.zeros(num_states)
                padded[: belief_np.size] = belief_np
                belief_np = padded

            belief_values.append(belief_np)

        belief_values = np.array(belief_values)
        time_steps = np.arange(len(belief_values))

        # Create figure
        fig, ax = self.create_figure(figsize=(10, 6))

        # Plot belief for each state
        colors = plt.cm.viridis(np.linspace(0, 1, num_states))

        # Handle case where belief_values might not have the expected shape
        if belief_values.ndim == 1:
            # Single dimension - assume this is belief in state 1 (good state)
            ax.plot(
                time_steps,
                belief_values,
                label=f"State 1 (Good)",
                color=colors[1 % num_states],
                linewidth=2,
            )
            # Plot complement for state 0
            ax.plot(
                time_steps,
                1 - belief_values,
                label=f"State 0 (Bad)",
                color=colors[0],
                linewidth=2,
            )
        elif belief_values.shape[1] >= num_states:
            # Multiple dimensions - plot each state
            for state in range(min(num_states, belief_values.shape[1])):
                ax.plot(
                    time_steps,
                    belief_values[:, state],
                    label=f"State {state}",
                    color=colors[state],
                    linewidth=2,
                )
        else:
            # Fallback: plot whatever dimensions we have
            for state in range(belief_values.shape[1]):
                ax.plot(
                    time_steps,
                    belief_values[:, state],
                    label=f"Dimension {state}",
                    color=colors[state % num_states],
                    linewidth=2,
                )

        # Highlight true state changes if available
        if "true_states" in metrics:
            self._add_true_state_highlights(ax, metrics["true_states"], time_steps)

        # Add reference line at 0.5
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

        # Formatting
        ax.set_title(f"Agent {agent_id} Belief Distribution Over Time")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save figure
        save_path = output_dir / f"agent_{agent_id}_belief_distribution.png"
        self.save_figure(fig, save_path, f"Agent {agent_id} Belief Distribution")

    def plot_good_belief_over_time(self, metrics: Dict, output_dir: Path):
        """
        Plot the belief in the good state over time for each agent.

        Args:
            metrics: Dictionary of metrics including 'agent_beliefs'
            output_dir: Directory to save plots
        """
        if "agent_beliefs" not in metrics:
            print("No belief data available for plotting good belief over time.")
            return

        # Create figure
        fig, ax = self.create_figure()
        colors = self.get_colors()

        for i, (agent_id, beliefs) in enumerate(metrics["agent_beliefs"].items()):
            good_beliefs = []
            time_steps = []

            for t, belief in enumerate(beliefs):
                # Skip invalid values
                if belief is None or (isinstance(belief, float) and np.isnan(belief)):
                    continue

                good_beliefs.append(belief)
                time_steps.append(t)

            if good_beliefs:  # Only plot if we have valid beliefs
                agent_color = self.get_agent_color(i, colors)
                ax.plot(
                    time_steps,
                    good_beliefs,
                    label=f"Agent {agent_id}",
                    color=agent_color,
                )

        # Formatting
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Belief in Good State")
        ax.set_title("Belief in Good State Over Time")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save figure
        save_path = output_dir / "good_belief_over_time.png"
        self.save_figure(fig, save_path, "Good Belief Over Time")

    def plot_social_learning_beliefs(self, metrics: Dict, env, output_dir: Path):
        """
        Plot belief distributions for social learning (Brandl) experiment.
        Shows each agent's belief in the wrong state over time.

        Args:
            metrics: Dictionary of metrics
            env: Environment object
            output_dir: Directory to save plots
        """
        if "belief_distributions" not in metrics:
            print("No belief distribution data available for social learning plot.")
            return

        # Determine the true state (assume it's constant for Brandl experiment)
        true_state = None
        if "true_states" in metrics and metrics["true_states"]:
            true_state = metrics["true_states"][0]  # Use first true state
        else:
            # Fallback: check if environment has current true state
            if hasattr(env, "true_state"):
                true_state = env.true_state

        if true_state is None:
            print("Cannot determine true state for wrong belief calculation.")
            return

        # Wrong state is the opposite of true state (for binary case)
        wrong_state = 1 - true_state

        # Create single figure
        fig, ax = self.create_figure(figsize=(12, 8))
        colors = self.get_colors()

        # Track if any agent has valid data
        has_data = False

        for i, (agent_id, belief_distributions) in enumerate(
            metrics["belief_distributions"].items()
        ):
            if not belief_distributions:
                continue

            # Process belief distributions for this agent
            belief_in_wrong_state = []
            time_steps = []

            for t, belief_dist in enumerate(belief_distributions):
                try:
                    if isinstance(belief_dist, torch.Tensor):
                        if belief_dist.dim() > 1:
                            belief_np = belief_dist.squeeze().detach().cpu().numpy()
                        else:
                            belief_np = belief_dist.detach().cpu().numpy()
                    else:
                        belief_np = np.array(belief_dist)

                    # Handle the nested list structure [[belief0, belief1]]
                    if belief_np.ndim > 1:
                        belief_np = belief_np.flatten()

                    # Extract belief in wrong state
                    if belief_np.size >= 2:
                        wrong_state_belief = belief_np[wrong_state]
                    else:
                        continue

                    # Validate the belief value
                    if (
                        not np.isnan(wrong_state_belief)
                        and 0 <= wrong_state_belief <= 1
                    ):
                        belief_in_wrong_state.append(wrong_state_belief)
                        time_steps.append(t)

                except Exception as e:
                    continue  # Skip problematic data points

            # Plot this agent's belief if we have valid data
            if belief_in_wrong_state:
                has_data = True
                agent_color = self.get_agent_color(i, colors)
                ax.plot(
                    time_steps,
                    belief_in_wrong_state,
                    label=f"Agent {agent_id}",
                    color=agent_color,
                    linewidth=2,
                )

        if not has_data:
            print("No valid belief distribution data found for social learning plot.")
            return

        # Add reference line at 0.5
        ax.axhline(
            y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random Belief"
        )

        # Formatting
        ax.set_title(
            f"Agents' Belief in Wrong State (State {wrong_state}) Over Time\nTrue State: {true_state}"
        )
        ax.set_xlabel("Time Steps")
        ax.set_ylabel(f"Belief in Wrong State (State {wrong_state})")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save figure
        save_path = output_dir / "social_learning_beliefs.png"
        self.save_figure(fig, save_path, "Social Learning Beliefs")

    def _process_belief_tensors(self, belief_states):
        """Convert belief tensors to numpy arrays with consistent format."""
        belief_arrays = []

        for belief in belief_states:
            if isinstance(belief, torch.Tensor):
                # Handle different tensor shapes
                if belief.dim() == 3:  # [1, batch_size, belief_dim]
                    belief = belief.squeeze(0).squeeze(0).detach().cpu().numpy()
                elif belief.dim() == 2:  # [batch_size, belief_dim]
                    belief = belief.squeeze(0).detach().cpu().numpy()
                elif belief.dim() == 1:  # [belief_dim]
                    belief = belief.detach().cpu().numpy()
            elif isinstance(belief, (list, np.ndarray)):
                belief = np.array(belief).flatten()
            else:
                continue  # Skip invalid beliefs

            belief_arrays.append(belief)

        return belief_arrays

    def _add_true_state_highlights(self, ax, true_states, time_steps):
        """Add background highlighting for true state changes."""
        if not true_states or len(true_states) == 0:
            return

        # Limit true states to match time steps
        if len(true_states) > len(time_steps):
            true_states = true_states[: len(time_steps)]
        elif len(true_states) < len(time_steps):
            # Pad with the last state
            true_states = true_states + [true_states[-1]] * (
                len(time_steps) - len(true_states)
            )

        # Add vertical lines at state changes
        prev_state = true_states[0] if true_states else None
        for t, state in enumerate(true_states):
            if state != prev_state:
                ax.axvline(x=t, color="red", linestyle="--", alpha=0.5)
                prev_state = state

        # Color background based on true state
        state_changes = [0]  # Start of first state
        current_state = true_states[0]
        for t, state in enumerate(true_states[1:], 1):
            if state != current_state:
                state_changes.append(t)
                current_state = state
        state_changes.append(len(true_states))  # End of last state

        # Color regions
        for i in range(len(state_changes) - 1):
            start = state_changes[i]
            end = state_changes[i + 1]
            state = true_states[start]
            color = "lightgreen" if state > 0 else "lightcoral"
            ax.axvspan(start, end, alpha=0.2, color=color)

    def _add_true_state_highlights_social_learning(self, ax, true_states, time_steps):
        """Add background highlighting for social learning - highlight when each state is true."""
        if not true_states or len(true_states) == 0:
            return

        # Limit true states to match time steps
        if len(true_states) > len(time_steps):
            true_states = true_states[: len(time_steps)]
        elif len(true_states) < len(time_steps):
            # Pad with the last state
            true_states = true_states + [true_states[-1]] * (
                len(time_steps) - len(true_states)
            )

        # Add vertical lines at state changes
        prev_state = true_states[0] if true_states else None
        for t, state in enumerate(true_states):
            if state != prev_state:
                ax.axvline(x=t, color="red", linestyle="--", alpha=0.5)
                prev_state = state

        # Color background based on true state
        state_changes = [0]  # Start of first state
        current_state = true_states[0]
        for t, state in enumerate(true_states[1:], 1):
            if state != current_state:
                state_changes.append(t)
                current_state = state
        state_changes.append(len(true_states))  # End of last state

        # Color regions based on which state is true
        state_0_labeled = False
        state_1_labeled = False
        for i in range(len(state_changes) - 1):
            start = state_changes[i]
            end = state_changes[i + 1]
            state = true_states[start]

            if state == 0:
                label = "True State = 0" if not state_0_labeled else ""
                ax.axvspan(start, end, alpha=0.2, color="lightcoral", label=label)
                state_0_labeled = True
            else:  # state == 1
                label = "True State = 1" if not state_1_labeled else ""
                ax.axvspan(start, end, alpha=0.2, color="lightgreen", label=label)
                state_1_labeled = True
