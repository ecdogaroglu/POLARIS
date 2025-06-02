"""
Catastrophic Forgetting Diagnostic Plots for POLARIS.

This module provides specialized plotting functions to diagnose catastrophic forgetting
by tracking policy belief accuracy, SI losses, and action logits across episodes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import MultiAgentPlotter


class CatastrophicForgettingDiagnosticPlotter(MultiAgentPlotter):
    """
    Specialized plotter for diagnosing catastrophic forgetting in POLARIS agents.
    
    Tracks and visualizes:
    - Policy belief accuracy over episodes
    - SI losses (policy and belief) over episodes
    - Action logits distribution over episodes
    """

    def plot(self, metrics: Dict[str, Any], env, args, output_dir: Path):
        """
        Generate catastrophic forgetting diagnostic plots.

        Args:
            metrics: Experiment metrics containing episodic data
            env: Environment object
            args: Command-line arguments
            output_dir: Directory to save plots
        """
        if not self.validate_metrics(metrics):
            return

        print("  🧠 Generating catastrophic forgetting diagnostic plots...")

        # Only generate these plots if we have episodic data and SI enabled
        if "episodes" not in metrics or not getattr(args, "use_si", False):
            print("  ⚠️  Skipping CF diagnostics: No episodic data or SI not enabled")
            return

        episodes = metrics["episodes"]
        if len(episodes) < 2:
            print("  ⚠️  Skipping CF diagnostics: Need at least 2 episodes")
            return

        # Generate combined SI loss and policy belief plot
        self.plot_si_losses_and_belief_accuracy(episodes, output_dir)
        
        # Generate action logits distribution plots
        self.plot_action_logits_evolution(episodes, output_dir)
        
        # Generate episode-by-episode loss progression
        self.plot_episode_loss_progression(episodes, output_dir)

    def plot_si_losses_and_belief_accuracy(self, episodes: List[Dict], output_dir: Path):
        """
        Plot SI losses (policy and belief) together with belief accuracy across episodes.
        
        Args:
            episodes: List of episode metrics
            output_dir: Directory to save plots
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Extract data for each episode
        episode_nums = list(range(len(episodes)))
        
        # Track SI losses and belief accuracy per agent
        agents_policy_si_losses = {}
        agents_belief_si_losses = {}
        agents_belief_accuracy = {}
        
        for ep_idx, episode in enumerate(episodes):
            # Extract training losses if available
            if "training_losses" in episode:
                for agent_id, agent_losses in episode["training_losses"].items():
                    if agent_id not in agents_policy_si_losses:
                        agents_policy_si_losses[agent_id] = []
                        agents_belief_si_losses[agent_id] = []
                    
                    # Handle both list and dict formats for agent_losses
                    if isinstance(agent_losses, list):
                        # If it's a list, aggregate the losses
                        if agent_losses:
                            # Take the mean of all loss dictionaries in the list
                            policy_si_losses = []
                            belief_si_losses = []
                            
                            for loss_dict in agent_losses:
                                if isinstance(loss_dict, dict):
                                    policy_loss_val = loss_dict.get("policy_si_loss", 0.0)
                                    belief_loss_val = loss_dict.get("belief_si_loss", 0.0)
                                    
                                    # Convert tensors to floats
                                    if hasattr(policy_loss_val, 'detach'):
                                        policy_loss_val = policy_loss_val.detach().cpu().numpy()
                                    policy_si_losses.append(float(policy_loss_val))
                                    
                                    if hasattr(belief_loss_val, 'detach'):
                                        belief_loss_val = belief_loss_val.detach().cpu().numpy()
                                    belief_si_losses.append(float(belief_loss_val))
                            
                            policy_si_loss = np.mean(policy_si_losses) if policy_si_losses else 0.0
                            belief_si_loss = np.mean(belief_si_losses) if belief_si_losses else 0.0
                        else:
                            policy_si_loss = 0.0
                            belief_si_loss = 0.0
                    else:
                        # If it's a dict, extract directly
                        policy_si_loss = agent_losses.get("policy_si_loss", 0.0)
                        belief_si_loss = agent_losses.get("belief_si_loss", 0.0)
                        
                        # Convert tensors to floats
                        if hasattr(policy_si_loss, 'detach'):
                            policy_si_loss = policy_si_loss.detach().cpu().numpy()
                        policy_si_loss = float(policy_si_loss)
                        
                        if hasattr(belief_si_loss, 'detach'):
                            belief_si_loss = belief_si_loss.detach().cpu().numpy()
                        belief_si_loss = float(belief_si_loss)
                    
                    agents_policy_si_losses[agent_id].append(policy_si_loss)
                    agents_belief_si_losses[agent_id].append(belief_si_loss)
            
            # Calculate belief accuracy for this episode
            if "agent_beliefs" in episode and "true_states" in episode:
                for agent_id, beliefs in episode["agent_beliefs"].items():
                    if agent_id not in agents_belief_accuracy:
                        agents_belief_accuracy[agent_id] = []
                    
                    # Calculate mean belief accuracy for this episode
                    accuracies = []
                    true_states = episode["true_states"]
                    
                    for t, belief in enumerate(beliefs):
                        if t < len(true_states) and belief is not None and not np.isnan(belief):
                            true_state = true_states[t]
                            if true_state == 1:  # Good state
                                accuracy = belief
                            else:  # Bad state
                                accuracy = 1 - belief
                            accuracies.append(accuracy)
                    
                    if accuracies:
                        mean_accuracy = np.mean(accuracies)
                        agents_belief_accuracy[agent_id].append(mean_accuracy)
                    else:
                        agents_belief_accuracy[agent_id].append(0.0)
        
        colors = self.get_colors()
        
        # Plot SI losses (top subplot)
        for i, agent_id in enumerate(sorted(agents_policy_si_losses.keys())):
            agent_color = self.get_agent_color(i, colors)
            
            if len(agents_policy_si_losses[agent_id]) > 0:
                ax1.plot(episode_nums[:len(agents_policy_si_losses[agent_id])], 
                        agents_policy_si_losses[agent_id], 
                        label=f"Agent {agent_id} Policy SI", 
                        color=agent_color, 
                        linestyle='-', 
                        linewidth=2.5,
                        marker='o',
                        markersize=6)
            
            if len(agents_belief_si_losses[agent_id]) > 0:
                ax1.plot(episode_nums[:len(agents_belief_si_losses[agent_id])], 
                        agents_belief_si_losses[agent_id], 
                        label=f"Agent {agent_id} Belief SI", 
                        color=agent_color, 
                        linestyle='--', 
                        linewidth=2.5,
                        marker='s',
                        markersize=6)
        
        self.set_title(ax1, "SI Losses Across Episodes")
        self.set_labels(ax1, "Episode", "SI Loss")
        ax1.legend(fontsize=self.legend_fontsize)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for SI losses
        
        # Plot belief accuracy (bottom subplot)
        for i, agent_id in enumerate(sorted(agents_belief_accuracy.keys())):
            agent_color = self.get_agent_color(i, colors)
            
            if len(agents_belief_accuracy[agent_id]) > 0:
                ax2.plot(episode_nums[:len(agents_belief_accuracy[agent_id])], 
                        agents_belief_accuracy[agent_id], 
                        label=f"Agent {agent_id}", 
                        color=agent_color, 
                        linewidth=2.5,
                        marker='o',
                        markersize=6)
        
        # Add reference line at 0.5 (random chance)
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random Chance")
        
        self.set_title(ax2, "Belief Accuracy Across Episodes")
        self.set_labels(ax2, "Episode", "Mean Belief Accuracy")
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=self.legend_fontsize)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = output_dir / "si_losses_and_belief_accuracy.png"
        self.save_figure(fig, save_path, "SI Losses and Belief Accuracy")

    def plot_action_logits_evolution(self, episodes: List[Dict], output_dir: Path):
        """
        Plot the evolution of action logits across episodes to detect forgetting.
        
        Args:
            episodes: List of episode metrics
            output_dir: Directory to save plots
        """
        # Extract action logits data from episodes
        agents_action_logits = {}
        
        for ep_idx, episode in enumerate(episodes):
            if "action_logits" in episode:
                for agent_id, logits_sequence in episode["action_logits"].items():
                    if agent_id not in agents_action_logits:
                        agents_action_logits[agent_id] = []
                    
                    # Calculate statistics for this episode's logits
                    if logits_sequence:
                        # Convert to numpy if needed
                        if isinstance(logits_sequence[0], torch.Tensor):
                            logits_array = np.array([logits.detach().cpu().numpy() for logits in logits_sequence])
                        else:
                            logits_array = np.array(logits_sequence)
                        
                        # Calculate mean, std, and entropy for each action dimension
                        episode_stats = {
                            'mean': np.mean(logits_array, axis=0),
                            'std': np.std(logits_array, axis=0),
                            'entropy': self._calculate_entropy_from_logits(logits_array)
                        }
                        agents_action_logits[agent_id].append(episode_stats)
        
        if not agents_action_logits:
            print("  ⚠️  No action logits data found for plotting")
            return
        
        # Create subplot for each agent
        num_agents = len(agents_action_logits)
        fig, axes = plt.subplots(num_agents, 3, figsize=(18, 6 * num_agents))
        
        if num_agents == 1:
            axes = axes.reshape(1, -1)
        
        colors = self.get_colors()
        episode_nums = list(range(len(episodes)))
        
        for agent_idx, (agent_id, agent_logits) in enumerate(agents_action_logits.items()):
            agent_color = self.get_agent_color(agent_idx, colors)
            
            if not agent_logits:
                continue
                
            # Extract data for plotting
            ep_indices = list(range(len(agent_logits)))
            action_dims = len(agent_logits[0]['mean'])
            
            # Plot mean logits evolution
            ax = axes[agent_idx, 0]
            for action_dim in range(action_dims):
                means = [stats['mean'][action_dim] for stats in agent_logits]
                ax.plot(ep_indices, means, 
                       label=f"Action {action_dim}", 
                       linewidth=2.5,
                       marker='o',
                       markersize=4)
            
            self.set_title(ax, f"Agent {agent_id} - Mean Action Logits")
            self.set_labels(ax, "Episode", "Mean Logit Value")
            ax.legend(fontsize=self.legend_fontsize-2)
            ax.grid(True, alpha=0.3)
            
            # Plot logits standard deviation (measure of uncertainty)
            ax = axes[agent_idx, 1]
            for action_dim in range(action_dims):
                stds = [stats['std'][action_dim] for stats in agent_logits]
                ax.plot(ep_indices, stds, 
                       label=f"Action {action_dim}", 
                       linewidth=2.5,
                       marker='s',
                       markersize=4)
            
            self.set_title(ax, f"Agent {agent_id} - Action Logits Std")
            self.set_labels(ax, "Episode", "Logit Standard Deviation")
            ax.legend(fontsize=self.legend_fontsize-2)
            ax.grid(True, alpha=0.3)
            
            # Plot entropy (measure of policy uncertainty)
            ax = axes[agent_idx, 2]
            entropies = [stats['entropy'] for stats in agent_logits]
            ax.plot(ep_indices, entropies, 
                   color=agent_color,
                   linewidth=2.5,
                   marker='D',
                   markersize=6)
            
            self.set_title(ax, f"Agent {agent_id} - Policy Entropy")
            self.set_labels(ax, "Episode", "Entropy")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = output_dir / "action_logits_evolution.png"
        self.save_figure(fig, save_path, "Action Logits Evolution")

    def plot_episode_loss_progression(self, episodes: List[Dict], output_dir: Path):
        """
        Plot detailed loss progression within and across episodes.
        
        Args:
            episodes: List of episode metrics
            output_dir: Directory to save plots
        """
        # Extract all loss types from episodes
        loss_types = ["policy_loss", "transformer_loss", "policy_si_loss", "belief_si_loss", "inference_loss", "critic_loss"]
        
        agents_losses = {}
        
        for ep_idx, episode in enumerate(episodes):
            if "training_losses" in episode:
                for agent_id, agent_losses in episode["training_losses"].items():
                    if agent_id not in agents_losses:
                        agents_losses[agent_id] = {loss_type: [] for loss_type in loss_types}
                    
                    # Handle both list and dict formats for agent_losses
                    if isinstance(agent_losses, list):
                        # If it's a list, aggregate the losses
                        for loss_type in loss_types:
                            loss_values = []
                            for loss_dict in agent_losses:
                                if isinstance(loss_dict, dict):
                                    loss_val = loss_dict.get(loss_type, 0.0)
                                    # Handle both scalar and list values
                                    if isinstance(loss_val, list):
                                        loss_values.extend(loss_val)
                                    else:
                                        # Convert tensors to floats
                                        if hasattr(loss_val, 'detach'):
                                            loss_val = loss_val.detach().cpu().numpy()
                                        loss_values.append(float(loss_val))
                            
                            mean_loss = np.mean(loss_values) if loss_values else 0.0
                            agents_losses[agent_id][loss_type].append(mean_loss)
                    else:
                        # If it's a dict, extract each type of loss for this episode
                        for loss_type in loss_types:
                            loss_value = agent_losses.get(loss_type, 0.0)
                            # Handle both scalar and list values
                            if isinstance(loss_value, list):
                                mean_loss = np.mean(loss_value) if loss_value else 0.0
                            else:
                                # Convert tensors to floats
                                if hasattr(loss_value, 'detach'):
                                    loss_value = loss_value.detach().cpu().numpy()
                                mean_loss = float(loss_value)
                            
                            agents_losses[agent_id][loss_type].append(mean_loss)
        
        if not agents_losses:
            print("  ⚠️  No training loss data found for episode progression plot")
            return
        
        # Create subplots for each loss type
        num_loss_types = len(loss_types)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = self.get_colors()
        episode_nums = list(range(len(episodes)))
        
        for loss_idx, loss_type in enumerate(loss_types):
            ax = axes[loss_idx]
            
            for agent_idx, (agent_id, agent_losses) in enumerate(agents_losses.items()):
                agent_color = self.get_agent_color(agent_idx, colors)
                
                loss_values = agent_losses[loss_type]
                if loss_values and any(v > 0 for v in loss_values):  # Only plot if we have non-zero values
                    ep_indices = list(range(len(loss_values)))
                    ax.plot(ep_indices, loss_values,
                           label=f"Agent {agent_id}",
                           color=agent_color,
                           linewidth=2.5,
                           marker='o',
                           markersize=4)
            
            # Format loss type name for title
            title = loss_type.replace('_', ' ').title()
            self.set_title(ax, f"{title} Across Episodes")
            self.set_labels(ax, "Episode", f"{title}")
            
            # Use log scale for SI losses
            if "si_loss" in loss_type and any(
                any(v > 0 for v in agent_losses[loss_type]) 
                for agent_losses in agents_losses.values()
            ):
                ax.set_yscale('log')
            
            ax.legend(fontsize=self.legend_fontsize-2)
            ax.grid(True, alpha=0.3)
        
        # Hide the last subplot if we don't have 6 loss types
        if len(loss_types) < 6:
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        save_path = output_dir / "episode_loss_progression.png"
        self.save_figure(fig, save_path, "Episode Loss Progression")

    def _calculate_entropy_from_logits(self, logits_array):
        """
        Calculate the mean entropy of action distributions from logits.
        
        Args:
            logits_array: Array of shape (timesteps, action_dims)
            
        Returns:
            float: Mean entropy across timesteps
        """
        # Convert logits to probabilities
        probs = np.exp(logits_array) / np.sum(np.exp(logits_array), axis=1, keepdims=True)
        
        # Calculate entropy for each timestep: H = -sum(p * log(p))
        entropy_per_step = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        
        # Return mean entropy
        return np.mean(entropy_per_step) 