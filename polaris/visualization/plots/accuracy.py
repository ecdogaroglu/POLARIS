"""
Accuracy plotting for POLARIS strategic experimentation experiments.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from . import MultiAgentPlotter


class AccuracyPlotter(MultiAgentPlotter):
    """
    Plotter for belief accuracy and allocation accuracy in strategic experimentation environments.

    Handles visualization of accuracy metrics with confidence intervals across multiple episodes.
    """

    def plot(self, metrics: Dict[str, Any], env, args, output_dir: Path):
        """
        Generate accuracy plots for strategic experimentation.

        Args:
            metrics: Experiment metrics
            env: Environment object
            args: Command-line arguments
            output_dir: Directory to save plots
        """
        if not self.validate_metrics(metrics):
            return

        print("  📊 Generating accuracy plots...")

        # Plot belief accuracy over time with confidence intervals
        if "agent_beliefs" in metrics and "true_states" in metrics:
            self.plot_belief_accuracy_over_time(metrics, output_dir)
        elif "episodes" in metrics and len(metrics["episodes"]) > 0:
            # Check if episodic data has the required fields
            has_beliefs = any("agent_beliefs" in ep for ep in metrics["episodes"])
            has_states = any("true_states" in ep for ep in metrics["episodes"])
            if has_beliefs and has_states:
                self.plot_belief_accuracy_over_time(metrics, output_dir)

        # Plot allocation accuracy over time with confidence intervals
        if hasattr(env, "safe_payoff") and "allocations" in metrics and "true_states" in metrics:
            self.plot_allocation_accuracy_over_time(metrics, env, output_dir)
        elif hasattr(env, "safe_payoff") and "episodes" in metrics and len(metrics["episodes"]) > 0:
            # Check if episodic data has the required fields
            has_allocations = any("allocations" in ep for ep in metrics["episodes"])
            has_states = any("true_states" in ep for ep in metrics["episodes"])
            if has_allocations and has_states:
                self.plot_allocation_accuracy_over_time(metrics, env, output_dir)

    def plot_belief_accuracy_over_time(self, metrics: Dict, output_dir: Path):
        """
        Plot belief accuracy over time with confidence intervals across episodes.

        Args:
            metrics: Dictionary of metrics including 'agent_beliefs' and 'true_states'
            output_dir: Directory to save plots
        """
        # Check if we have the required data either at top level or in episodes
        has_required_data = False
        
        if "agent_beliefs" in metrics and "true_states" in metrics:
            has_required_data = True
        elif "episodes" in metrics and len(metrics["episodes"]) > 0:
            # Check if episodic data has the required fields
            has_beliefs = any("agent_beliefs" in ep for ep in metrics["episodes"])
            has_states = any("true_states" in ep for ep in metrics["episodes"])
            if has_beliefs and has_states:
                has_required_data = True
        
        if not has_required_data:
            print("No belief or true state data available for belief accuracy plotting.")
            return

        # Calculate belief accuracy for each episode
        episodic_accuracies = self._calculate_episodic_belief_accuracies(metrics)
        
        if not episodic_accuracies:
            print("No valid belief accuracy data found.")
            return

        # Calculate mean and confidence intervals across episodes
        mean_accuracies, confidence_intervals, time_steps = self._calculate_accuracy_statistics(episodic_accuracies)

        # Create figure
        fig, ax = self.create_figure()
        colors = self.get_colors()

        # Plot each agent
        for i, agent_id in enumerate(mean_accuracies.keys()):
            agent_color = self.get_agent_color(i, colors)
            
            mean_acc = mean_accuracies[agent_id]
            ci_lower, ci_upper = confidence_intervals[agent_id]
            
            # Add confidence interval first (so it appears behind the line)
            ax.fill_between(
                time_steps,
                ci_lower,
                ci_upper,
                alpha=0.3,  # Increased alpha for better visibility
                color=agent_color,
                edgecolor=agent_color,
                linewidth=0.5,
            )
            
            # Plot mean accuracy on top
            ax.plot(
                time_steps,
                mean_acc,
                label=f"Agent {agent_id}",
                color=agent_color,
                linewidth=2.5,
            )

        # Formatting
        self.set_labels(ax, "Time Step", "Belief Accuracy")
        self.set_title(ax, "Belief Accuracy Over Time (Mean ± 95% CI)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)

        # Add reference line at 0.5 (random chance)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random Chance")

        # Save figure
        save_path = output_dir / "belief_accuracy_over_time.png"
        self.save_figure(fig, save_path, "Belief Accuracy Over Time")

    def plot_allocation_accuracy_over_time(self, metrics: Dict, env, output_dir: Path):
        """
        Plot allocation accuracy over time with confidence intervals across episodes.

        Args:
            metrics: Dictionary of metrics including 'allocations' and 'true_states'
            env: Environment object
            output_dir: Directory to save plots
        """
        # Check if we have the required data either at top level or in episodes
        has_required_data = False
        
        if "allocations" in metrics and "true_states" in metrics:
            has_required_data = True
        elif "episodes" in metrics and len(metrics["episodes"]) > 0:
            # Check if episodic data has the required fields
            has_allocations = any("allocations" in ep for ep in metrics["episodes"])
            has_states = any("true_states" in ep for ep in metrics["episodes"])
            if has_allocations and has_states:
                has_required_data = True
        
        if not has_required_data:
            print("No allocation or true state data available for allocation accuracy plotting.")
            return

        # Calculate allocation accuracy for each episode
        episodic_accuracies = self._calculate_episodic_allocation_accuracies(metrics, env)
        
        if not episodic_accuracies:
            print("No valid allocation accuracy data found.")
            return

        # Calculate mean and confidence intervals across episodes
        mean_accuracies, confidence_intervals, time_steps = self._calculate_accuracy_statistics(episodic_accuracies)

        # Create figure
        fig, ax = self.create_figure()
        colors = self.get_colors()

        # Plot each agent
        for i, agent_id in enumerate(mean_accuracies.keys()):
            agent_color = self.get_agent_color(i, colors)
            
            mean_acc = mean_accuracies[agent_id]
            ci_lower, ci_upper = confidence_intervals[agent_id]
            
            # Add confidence interval first (so it appears behind the line)
            ax.fill_between(
                time_steps,
                ci_lower,
                ci_upper,
                alpha=0.5,  # Increased alpha for better visibility
                color=agent_color,
                edgecolor=agent_color,
                linewidth=0.5,
            )
            
            # Plot mean accuracy on top
            ax.plot(
                time_steps,
                mean_acc,
                label=f"Agent {agent_id}",
                color=agent_color,
                linewidth=2.5,
            )

        # Formatting
        self.set_labels(ax, "Time Step", "Allocation Accuracy")
        self.set_title(ax, "Allocation Accuracy Over Time (Mean ± 95% CI)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)

        # Add reference line at 0.5 (random chance)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random Chance")

        # Save figure
        save_path = output_dir / "allocation_accuracy_over_time.png"
        self.save_figure(fig, save_path, "Allocation Accuracy Over Time")

    def _calculate_episodic_belief_accuracies(self, metrics: Dict) -> List[Dict]:
        """
        Calculate belief accuracy for each episode.
        
        Returns:
            List of dictionaries, each containing agent belief accuracies for one episode
        """
        episodic_accuracies = []
        
        # Check if we have episodic data structure
        if "episodes" in metrics and isinstance(metrics["episodes"], list):
            # Multi-episode structure
            for episode_data in metrics["episodes"]:
                episode_accuracies = {}
                
                if "agent_beliefs" not in episode_data or "true_states" not in episode_data:
                    continue
                    
                for agent_id, beliefs in episode_data["agent_beliefs"].items():
                    accuracies = []
                    true_states = episode_data["true_states"]
                    
                    for t, belief in enumerate(beliefs):
                        if t < len(true_states) and belief is not None and not np.isnan(belief):
                            true_state = true_states[t]
                            # Belief accuracy: how close the belief is to the true state
                            if true_state == 1:  # Good state
                                accuracy = belief  # Higher belief in good state is better
                            else:  # Bad state
                                accuracy = 1 - belief  # Lower belief in good state is better
                            accuracies.append(accuracy)
                    
                    if accuracies:
                        episode_accuracies[agent_id] = accuracies
                
                if episode_accuracies:
                    episodic_accuracies.append(episode_accuracies)
        else:
            # Single episode structure (fallback)
            episode_accuracies = {}
            
            if "agent_beliefs" in metrics and "true_states" in metrics:
                for agent_id, beliefs in metrics["agent_beliefs"].items():
                    accuracies = []
                    true_states = metrics["true_states"]
                    
                    for t, belief in enumerate(beliefs):
                        if t < len(true_states) and belief is not None and not np.isnan(belief):
                            true_state = true_states[t]
                            if true_state == 1:  # Good state
                                accuracy = belief
                            else:  # Bad state
                                accuracy = 1 - belief
                            accuracies.append(accuracy)
                    
                    if accuracies:
                        episode_accuracies[agent_id] = accuracies
                
                if episode_accuracies:
                    episodic_accuracies.append(episode_accuracies)
        
        return episodic_accuracies

    def _calculate_episodic_allocation_accuracies(self, metrics: Dict, env) -> List[Dict]:
        """
        Calculate allocation accuracy for each episode.
        
        Returns:
            List of dictionaries, each containing agent allocation accuracies for one episode
        """
        episodic_accuracies = []
        
        # Check if we have episodic data structure
        if "episodes" in metrics and isinstance(metrics["episodes"], list):
            # Multi-episode structure
            for episode_data in metrics["episodes"]:
                episode_accuracies = {}
                
                if "allocations" not in episode_data or "true_states" not in episode_data:
                    continue
                    
                for agent_id, allocations in episode_data["allocations"].items():
                    accuracies = []
                    true_states = episode_data["true_states"]
                    
                    for t, allocation in enumerate(allocations):
                        if t < len(true_states):
                            true_state = true_states[t]
                            # Calculate optimal allocation for this state
                            optimal_allocation = self._calculate_optimal_allocation(true_state, env)
                            
                            # Allocation accuracy: 1 - normalized distance from optimal
                            accuracy = 1 - abs(allocation - optimal_allocation)
                            accuracies.append(max(0, accuracy))  # Ensure non-negative
                    
                    if accuracies:
                        episode_accuracies[agent_id] = accuracies
                
                if episode_accuracies:
                    episodic_accuracies.append(episode_accuracies)
        else:
            # Single episode structure (fallback)
            episode_accuracies = {}
            
            if "allocations" in metrics and "true_states" in metrics:
                for agent_id, allocations in metrics["allocations"].items():
                    accuracies = []
                    true_states = metrics["true_states"]
                    
                    for t, allocation in enumerate(allocations):
                        if t < len(true_states):
                            true_state = true_states[t]
                            optimal_allocation = self._calculate_optimal_allocation(true_state, env)
                            accuracy = 1 - abs(allocation - optimal_allocation)
                            accuracies.append(max(0, accuracy))
                    
                    if accuracies:
                        episode_accuracies[agent_id] = accuracies
                
                if episode_accuracies:
                    episodic_accuracies.append(episode_accuracies)
        
        return episodic_accuracies

    def _calculate_optimal_allocation(self, true_state: int, env) -> float:
        """
        Calculate the optimal allocation given the true state.
        
        Args:
            true_state: The true state (0 for bad, 1 for good)
            env: Environment object with parameters
            
        Returns:
            optimal_allocation: Optimal allocation to risky arm [0,1]
        """
        if true_state == 1:  # Good state
            # Compare risky arm payoff in good state vs safe payoff
            risky_payoff = env.drift_rates[1] + env.jump_rates[1] * env.jump_sizes[1]
            if risky_payoff > env.safe_payoff:
                return 1.0  # Full allocation to risky arm
            else:
                return 0.0  # Full allocation to safe arm
        else:  # Bad state
            # Compare risky arm payoff in bad state vs safe payoff
            risky_payoff = env.drift_rates[0] + env.jump_rates[0] * env.jump_sizes[0]
            if risky_payoff > env.safe_payoff:
                return 1.0  # Full allocation to risky arm
            else:
                return 0.0  # Full allocation to safe arm

    def _calculate_accuracy_statistics(self, episodic_accuracies: List[Dict]) -> Tuple[Dict, Dict, List]:
        """
        Calculate mean and confidence intervals across episodes.
        
        Args:
            episodic_accuracies: List of episode accuracy dictionaries
            
        Returns:
            Tuple of (mean_accuracies, confidence_intervals, time_steps)
        """
        if not episodic_accuracies:
            return {}, {}, []

        # Find common agents and maximum time steps
        all_agent_ids = set()
        max_time_steps = 0
        
        for episode_data in episodic_accuracies:
            all_agent_ids.update(episode_data.keys())
            for agent_data in episode_data.values():
                max_time_steps = max(max_time_steps, len(agent_data))

        time_steps = list(range(max_time_steps))
        mean_accuracies = {}
        confidence_intervals = {}

        for agent_id in all_agent_ids:
            # Collect accuracy data for this agent across all episodes
            agent_accuracies = []
            
            for episode_data in episodic_accuracies:
                if agent_id in episode_data:
                    # Pad with NaN if episode is shorter
                    episode_acc = episode_data[agent_id]
                    padded_acc = episode_acc + [np.nan] * (max_time_steps - len(episode_acc))
                    agent_accuracies.append(padded_acc)
            
            if agent_accuracies:
                # Convert to numpy array for easier computation
                agent_accuracies = np.array(agent_accuracies)
                
                # Calculate mean and confidence intervals (ignoring NaN values)
                mean_acc = np.nanmean(agent_accuracies, axis=0)
                std_acc = np.nanstd(agent_accuracies, axis=0)
                n_episodes = np.sum(~np.isnan(agent_accuracies), axis=0)
                
                # 95% confidence interval
                ci_margin = 1.96 * std_acc / np.sqrt(np.maximum(n_episodes, 1))
                ci_lower = mean_acc - ci_margin
                ci_upper = mean_acc + ci_margin
                
                # Clip to valid range [0, 1]
                ci_lower = np.clip(ci_lower, 0, 1)
                ci_upper = np.clip(ci_upper, 0, 1)
                
                mean_accuracies[agent_id] = mean_acc
                confidence_intervals[agent_id] = (ci_lower, ci_upper)

        return mean_accuracies, confidence_intervals, time_steps 