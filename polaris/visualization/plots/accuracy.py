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

        # Create unified accuracy plot with both belief and allocation accuracy
        self.plot_unified_accuracy(metrics, env, output_dir)

    def plot_unified_accuracy(self, metrics: Dict, env, output_dir: Path):
        """
        Plot unified accuracy plots showing both belief and allocation accuracy over time with confidence intervals.

        Args:
            metrics: Dictionary of metrics including 'agent_beliefs', 'allocations', and 'true_states'
            env: Environment object
            output_dir: Directory to save plots
        """
        # Check if we have the required data for both belief and allocation accuracy
        has_belief_data = False
        has_allocation_data = False
        
        # Check belief data
        if "agent_beliefs" in metrics and "true_states" in metrics:
            has_belief_data = True
        elif "episodes" in metrics and len(metrics["episodes"]) > 0:
            has_beliefs = any("agent_beliefs" in ep for ep in metrics["episodes"])
            has_states = any("true_states" in ep for ep in metrics["episodes"])
            if has_beliefs and has_states:
                has_belief_data = True
        
        # Check allocation data
        if hasattr(env, "safe_payoff") and "allocations" in metrics and "true_states" in metrics:
            has_allocation_data = True
        elif hasattr(env, "safe_payoff") and "episodes" in metrics and len(metrics["episodes"]) > 0:
            has_allocations = any("allocations" in ep for ep in metrics["episodes"])
            has_states = any("true_states" in ep for ep in metrics["episodes"])
            if has_allocations and has_states:
                has_allocation_data = True
        
        if not has_belief_data and not has_allocation_data:
            print("No belief or allocation data available for accuracy plotting.")
            return

        # Create figure with 2 subplots (1x2 layout)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        
        # Use grayscale colors similar to the sweep plot (darker grey instead of light grey)
        colors = ['#000000', '#808080']  # Black and medium gray
        
        # Plot belief accuracy if available
        if has_belief_data:
            ax1 = axes[0]
            
            # Calculate belief accuracy for each episode
            episodic_accuracies = self._calculate_episodic_belief_accuracies(metrics)
            
            if episodic_accuracies:
                # Calculate mean and confidence intervals across episodes
                mean_accuracies, confidence_intervals, time_steps = self._calculate_accuracy_statistics(episodic_accuracies)

                # Plot each agent
                for i, agent_id in enumerate(mean_accuracies.keys()):
                    agent_color = colors[i % len(colors)]
                    
                    mean_acc = mean_accuracies[agent_id]
                    ci_lower, ci_upper = confidence_intervals[agent_id]
                    
                    # Add confidence interval first (so it appears behind the line)
                    ax1.fill_between(
                        time_steps,
                        ci_lower,
                        ci_upper,
                        alpha=0.1,
                        color=agent_color,
                    )
                    
                    # Plot mean accuracy on top (no markers)
                    ax1.plot(
                        time_steps,
                        mean_acc,
                        label=f"Agent {int(agent_id) + 1}",  # Convert 0,1 to 1,2
                        color=agent_color,
                        linewidth=2,
                        alpha=0.9,
                    )

                # Formatting
                self.set_labels(ax1, "Time Steps", "Belief Accuracy", fontsize=12)
                self.set_title(ax1, "Belief Accuracy Over Time", fontsize=14)
                ax1.set_ylim(0, 1)
                ax1.legend(fontsize=12, loc='lower right', framealpha=0.9)
                ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax1.tick_params(labelsize=10)
                
                # Set clean white background
                ax1.set_facecolor('white')

                # Add reference line at 0.5 (random chance)
                ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
            else:
                ax1.text(0.5, 0.5, "No valid belief accuracy data", 
                        ha='center', va='center', transform=ax1.transAxes)
        else:
            axes[0].text(0.5, 0.5, "No belief data available", 
                        ha='center', va='center', transform=axes[0].transAxes)
        
        # Plot allocation accuracy if available
        if has_allocation_data:
            ax2 = axes[1]
            
            # Calculate allocation accuracy for each episode
            episodic_accuracies = self._calculate_episodic_allocation_accuracies(metrics, env)
            
            if episodic_accuracies:
                # Calculate mean and confidence intervals across episodes
                mean_accuracies, confidence_intervals, time_steps = self._calculate_accuracy_statistics(episodic_accuracies)

                # Plot each agent
                for i, agent_id in enumerate(mean_accuracies.keys()):
                    agent_color = colors[i % len(colors)]
                    
                    mean_acc = mean_accuracies[agent_id]
                    ci_lower, ci_upper = confidence_intervals[agent_id]
                    
                    # Add confidence interval first (so it appears behind the line)
                    ax2.fill_between(
                        time_steps,
                        ci_lower,
                        ci_upper,
                        alpha=0.1,
                        color=agent_color,
                    )
                    
                    # Plot mean accuracy on top (no markers)
                    ax2.plot(
                        time_steps,
                        mean_acc,
                        label=f"Agent {int(agent_id) + 1}",  # Convert 0,1 to 1,2
                        color=agent_color,
                        linewidth=2,
                        alpha=0.9,
                    )

                # Formatting
                self.set_labels(ax2, "Time Steps", "Allocation Accuracy", fontsize=12)
                self.set_title(ax2, "Allocation Accuracy Over Time", fontsize=14)
                ax2.set_ylim(0, 1)
                ax2.legend(fontsize=12, loc='lower right', framealpha=0.9)
                ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax2.tick_params(labelsize=10)
                
                # Set clean white background
                ax2.set_facecolor('white')

                # Add reference line at 0.5 (random chance)
                ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
            else:
                ax2.text(0.5, 0.5, "No valid allocation accuracy data", 
                        ha='center', va='center', transform=ax2.transAxes)
        else:
            axes[1].text(0.5, 0.5, "No allocation data available", 
                        ha='center', va='center', transform=axes[1].transAxes)

        # Overall title
        fig.suptitle(f"Accuracy Over Time (Mean over {len(metrics['episodes'])} episodes with ± 95% CI)", 
                    fontsize=16, fontweight='bold')

        # Save figure
        save_path = output_dir / "unified_accuracy_over_time.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
        plt.close()

    def plot_belief_accuracy_over_time(self, metrics: Dict, output_dir: Path):
        """
        Plot belief accuracy over time with confidence intervals across episodes.
        
        Note: This method is kept for backward compatibility but the unified plot is preferred.

        Args:
            metrics: Dictionary of metrics including 'agent_beliefs' and 'true_states'
            output_dir: Directory to save plots
        """
        # This method is now deprecated in favor of the unified plot
        # Keeping it for backward compatibility
        pass

    def plot_allocation_accuracy_over_time(self, metrics: Dict, env, output_dir: Path):
        """
        Plot allocation accuracy over time with confidence intervals across episodes.
        
        Note: This method is kept for backward compatibility but the unified plot is preferred.

        Args:
            metrics: Dictionary of metrics including 'allocations' and 'true_states'
            env: Environment object
            output_dir: Directory to save plots
        """
        # This method is now deprecated in favor of the unified plot
        # Keeping it for backward compatibility
        pass

    def _calculate_episodic_belief_accuracies(self, metrics: Dict) -> List[Dict]:
        """
        Calculate belief accuracy for each episode.
        
        Returns:
            List of dictionaries, each containing agent belief accuracies for one episode
        """
        episodic_accuracies = []
        
        # Check if we have episodic data structure
        if "episodes" in metrics and isinstance(metrics["episodes"], list) and len(metrics["episodes"]) > 0:
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
            # Single episode structure (fallback) or flat aggregated structure
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
        if "episodes" in metrics and isinstance(metrics["episodes"], list) and len(metrics["episodes"]) > 0:
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
            # Single episode structure (fallback) or flat aggregated structure
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