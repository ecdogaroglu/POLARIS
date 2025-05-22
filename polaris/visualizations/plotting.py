"""
Visualization functions for POLARIS experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import torch
import os
from scipy import stats
from pathlib import Path

from polaris.utils.metrics import process_incorrect_probabilities
from polaris.utils.utils import calculate_learning_rate
from polaris.visualizations.latex_style import set_latex_style, format_axis_in_latex_style, save_figure_for_publication

def generate_plots(metrics, env, args, output_dir, training, episodic_metrics=None, use_latex=False):
    """
    Generate plots from experiment results.
    
    Args:
        metrics: Combined metrics dictionary
        env: Environment object
        args: Command-line arguments
        output_dir: Directory to save plots
        training: Whether this is training or evaluation
        episodic_metrics: Optional dictionary of metrics for each episode
        use_latex: Whether to use LaTeX styling for plots
    """
    # Check if metrics are available
    if not metrics:
        print("No metrics available for plotting")
        return
    
    # Check if the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Add environment information to metrics for plotting
    metrics['num_states'] = env.num_states
    metrics['num_agents'] = env.num_agents
    metrics['environment_type'] = env.__class__.__name__
    
    # Plot belief states for each agent if requested
    if hasattr(args, 'plot_internal_states') and args.plot_internal_states:
        for agent_id in range(env.num_agents):
            plot_belief_states(metrics, agent_id, output_dir, use_latex)
            plot_latent_states(metrics, agent_id, output_dir, use_latex)
            
    # For Strategic Experimentation environment, plot agent allocations
    if hasattr(args, 'plot_allocations') and args.plot_allocations and hasattr(env, 'safe_payoff'):
        print("Plotting agent allocations...")
        plot_allocations(metrics, output_dir, use_latex)
        
        # Also plot KL divergence if available
        if 'policy_kl_divergence' in metrics:
            print("Plotting KL divergence towards MPE...")
            plot_kl_divergence(metrics, output_dir, use_latex)
            print("Plotting policy cutoff vs belief...")
            plot_policy_cutoff_vs_belief(metrics, output_dir, use_latex)
            print("Plotting good belief over time...")
            plot_good_belief_over_time(metrics, output_dir, use_latex)
    
    # If we have episodic metrics, plot with error bars
    if episodic_metrics and 'episodes' in episodic_metrics and len(episodic_metrics['episodes']) > 1:
        episode_length = args.horizon
        num_episodes = len(episodic_metrics['episodes'])
        
        # Create mean incorrect action probability plots with confidence intervals
        incorrect_probs_by_episode = []
        for ep in episodic_metrics['episodes']:
            incorrect_probs_by_episode.append(ep['incorrect_probs'])
            
        # Plot mean incorrect probabilities with confidence intervals
        plot_mean_incorrect_action_probabilities_with_ci(
            incorrect_probs_by_episode,
            title=f"Mean Incorrect Action Probabilities with 95% CI ({num_episodes} episodes)",
            save_path=output_dir / "mean_incorrect_probs_with_ci.png",
            log_scale=True,
            episode_length=episode_length
        )
    else:
        # Just plot the standard curves without CIs
        incorrect_probs = process_incorrect_probabilities(metrics, env.num_agents)
        
        plot_incorrect_action_probabilities(
            incorrect_probs,
            title="Incorrect Action Probabilities Over Time",
            save_path=output_dir / "incorrect_probs.png",
            log_scale=True,
            episode_length=args.horizon
        )

def create_empty_plot(output_dir):
    """Create an empty plot with a message when no data is available."""
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0.5, "No data available for plotting", 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=20)
    plt.savefig(str(output_dir / 'incorrect_action_probs.png'))
    plt.close()


def plot_mean_incorrect_action_probabilities_with_ci(
    episodic_metrics: Dict,
    title: str = "Mean Incorrect Action Probabilities with 95% CI",
    save_path: Optional[str] = None,
    log_scale: bool = False,
    episode_length: Optional[int] = None,
    subtitle: Optional[str] = None
) -> None:
    """
    Plot mean incorrect action probabilities across episodes with 95% confidence intervals.
    
    Args:
        episodic_metrics: Dictionary containing episode-separated metrics
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        log_scale: Whether to use logarithmic scale for y-axis
        episode_length: Length of each episode
        subtitle: Optional subtitle to display below the main title
    """
    if not episodic_metrics or 'episodes' not in episodic_metrics or not episodic_metrics['episodes']:
        print("No episodic metrics available for plotting mean incorrect action probabilities with CI.")
        return
    
    # Extract episodes data
    episodes = episodic_metrics['episodes']
    
    # Keep track of the total number of episodes for the title
    total_episodes = len(episodes)
    
    # We'll use all episodes for calculation (not limiting to 10)
    num_episodes = total_episodes
    
    if num_episodes < 2:
        print("Need at least 2 episodes to plot mean with confidence intervals.")
        return
    
    # Get the number of agents from the first episode
    if 'action_probs' not in episodes[0]:
        print("No action probabilities found in episodic metrics.")
        return
    
    agent_ids = list(episodes[0]['action_probs'].keys())
    
    # Create figure with LaTeX-style dimensions (golden ratio)
    fig_width = 8  # Width suitable for single-column journal format
    fig_height = fig_width * 0.618  # Golden ratio
    
    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    
    # Apply LaTeX-style formatting to the axis
    format_axis_in_latex_style(ax)
    
    # Get colors from the LaTeX-style color cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    # For each agent, collect data across episodes and calculate mean and CI
    for i, agent_id in enumerate(sorted(agent_ids, key=int)):
        agent_color = colors[i % len(colors)]  # Cycle through colors if more agents than colors
        
        # Collect data for this agent across all episodes
        # We need to ensure all episodes have the same length for this agent
        min_length = min(len(episodes[ep_idx]['action_probs'][agent_id]) for ep_idx in range(num_episodes))
        
        if min_length == 0:
            print(f"Agent {agent_id} has no data in at least one episode. Skipping.")
            continue
        
        # Create a 2D array where each row is an episode and each column is a time step
        agent_data = np.zeros((num_episodes, min_length))
        for ep_idx in range(num_episodes):
            agent_data[ep_idx, :] = episodes[ep_idx]['action_probs'][agent_id][:min_length]
        
        # Calculate mean and standard deviation across episodes for each time step
        mean_probs = np.mean(agent_data, axis=0)
        std_probs = np.std(agent_data, axis=0)
        
        # Calculate 95% confidence interval
        # For small sample sizes, use t-distribution
        t_value = stats.t.ppf(0.975, num_episodes - 1)  # 95% CI (two-tailed)
        ci = t_value * std_probs / np.sqrt(num_episodes)
        
        # Create time steps array
        time_steps = np.arange(min_length)
        
        # Plot mean line with markers at regular intervals
        marker_interval = max(1, min_length // 10)  # Show about 10 markers
        
        if log_scale:
            line, = ax.semilogy(time_steps, mean_probs, 
                             label=f"Agent {agent_id}",
                             color=agent_color,
                             linewidth=1.5,
                             marker='o',
                             markersize=3,
                             markevery=marker_interval)
        else:
            line, = ax.plot(time_steps, mean_probs, 
                         label=f"Agent {agent_id}",
                         color=agent_color,
                         linewidth=1.5,
                         marker='o',
                         markersize=3,
                         markevery=marker_interval)
        
        # Plot confidence interval with more elegant styling
        ax.fill_between(time_steps, 
                      mean_probs - ci, 
                      mean_probs + ci, 
                      color=agent_color, 
                      alpha=0.15,
                      edgecolor=agent_color,
                      linewidth=0.5,
                      linestyle=':')
        
        # Calculate and display learning rate for the mean curve
        if len(mean_probs) >= 10:
            learning_rate = calculate_learning_rate(mean_probs)
            
            # Update the label with learning rate
            line.set_label(f"Agent {agent_id} (r = {learning_rate:.4f})")
            
            # Plot fitted exponential decay with more elegant styling
            x = np.arange(len(mean_probs))
            initial_value = mean_probs[0]
            y = np.exp(-learning_rate * x) * initial_value
            
            if log_scale:
                ax.semilogy(x, y, '--', alpha=0.5, color=line.get_color(), linewidth=0.8)
            else:
                ax.plot(x, y, '--', alpha=0.5, color=line.get_color(), linewidth=0.8)
    
    # Set labels with LaTeX-style formatting
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Incorrect Action Probability" + (" (log scale)" if log_scale else ""))
    
    # Set y-axis limits for better visualization
    if log_scale:
        ax.set_ylim(0.001, 1.0)
    else:
        ax.set_ylim(0, 1.0)
    
    # Add legend with LaTeX-style formatting
    ax.legend(loc='best', framealpha=0.9, edgecolor='black', fancybox=False)
    
    # Set title with LaTeX-style formatting
    ax.set_title(title)
    
    # Add a subtitle if provided
    if subtitle:
        ax.text(0.5, 0.97, subtitle, 
               horizontalalignment='center',
               verticalalignment='top',
               transform=ax.transAxes,
               fontsize=10,
               alpha=0.8)
    
    # Number of episodes is now included in the title, so we don't need a separate text box
    
    # Save the figure in publication-quality formats
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Get base filename without extension
            base_path = os.path.splitext(save_path)[0]
            
            # Save in multiple formats for publication
            save_figure_for_publication(
                fig=fig,
                filename=base_path,
                dpi=300,
                formats=['png']
            )
            
            plt.close()
        except Exception as e:
            print(f"Error saving plot: {e}")
            plt.close()
    else:
        plt.show()
        plt.close()


def generate_internal_state_plots(metrics, env, args, output_dir, episode_num=None):
    """
    Generate plots of internal agent states during evaluation.
    
    Args:
        metrics: Dictionary of metrics
        env: Environment object
        args: Command-line arguments
        output_dir: Directory to save plots
        episode_num: Optional episode number for episode-specific plots
    """
    
    # Plot belief distributions if available
    # Add episode number to title if provided
    episode_suffix = f" - Episode {episode_num}" if episode_num is not None else ""
    
    plot_belief_distributions(
        belief_distributions=metrics['belief_distributions'],
        true_states=metrics['true_states'],
        title=f"Belief Distributions Evolution ({args.network_type.capitalize()} Network, {env.num_agents} Agents){episode_suffix}",
        save_path=str(output_dir / 'belief_distributions.png'),
        episode_length=args.horizon,  # Use horizon directly
        num_episodes=1 if episode_num is not None else args.num_episodes  # Single episode if episode_num is provided
    )
    
    # Plot opponent belief distributions if available
    if ('opponent_belief_distributions' in metrics and 
        any(len(beliefs) > 0 for beliefs in metrics['opponent_belief_distributions'].values())):
        
        # Add episode number to title if provided
        episode_suffix = f" - Episode {episode_num}" if episode_num is not None else ""
        
        plot_belief_distributions(
            belief_distributions=metrics['opponent_belief_distributions'],
            true_states=metrics['true_states'],
            title=f"Opponent Belief Distributions Evolution ({args.network_type.capitalize()} Network, {env.num_agents} Agents){episode_suffix}",
            save_path=str(output_dir / 'opponent_belief_distributions.png'),
            episode_length=args.horizon,  # Use horizon directly
            num_episodes=1 if episode_num is not None else args.num_episodes  # Single episode if episode_num is provided
        )
    
    # Plot agent actions if available
    if ('agent_actions' in metrics and 
        any(len(actions) > 0 for actions in metrics['agent_actions'].values())):
        
        # Add episode number to title if provided
        episode_suffix = f" - Episode {episode_num}" if episode_num is not None else ""
        
        plot_agent_actions(
            actions=metrics['agent_actions'],
            true_states=metrics['true_states'],
            title=f"Agent Actions Over Time ({args.network_type.capitalize()} Network, {env.num_agents} Agents){episode_suffix}",
            save_path=str(output_dir / 'agent_actions.png'),
            episode_length=args.horizon,  # Use horizon directly
            num_episodes=1 if episode_num is not None else args.num_episodes  # Single episode if episode_num is provided
        )
    

def plot_incorrect_action_probabilities(
    incorrect_probs: Dict[int, List[float]],
    title: str = "Incorrect Action Probabilities Over Time",
    save_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    log_scale: bool = False,
    show_learning_rates: bool = True,
    episode_length: Optional[int] = None
) -> None:
    """
    Plot incorrect action probabilities for all agents with separate subplots for each episode.
    
    Args:
        incorrect_probs: Dictionary mapping agent IDs to lists of incorrect action probabilities over time
        title: Title of the plot
        save_path: Path to save the figure (if None, figure is displayed)
        max_steps: Maximum number of steps to plot (if None, plots all steps)
        log_scale: Whether to use logarithmic scale for y-axis
        show_learning_rates: Whether to calculate and display learning rates in the legend
        episode_length: Length of each episode (if None, treats all data as a single episode)
    """
    # Determine total steps and number of episodes
    if not incorrect_probs:
        print("No incorrect action probability data available for plotting")
        if save_path:
            create_empty_plot(Path(save_path).parent)
        return
        
    total_steps = max(len(probs) for probs in incorrect_probs.values())
    
    if episode_length is None:
        # If episode_length is not provided, treat all data as a single episode
        num_episodes = 1
        episode_length = total_steps
    else:
        # Calculate number of episodes based on total steps and episode length
        num_episodes = (total_steps + episode_length - 1) // episode_length  # Ceiling division
        
        # Limit to first 10 episodes for readability
        num_episodes = min(num_episodes, 10)
        # Limit total steps to only include the episodes we're plotting
        total_steps = min(total_steps, num_episodes * episode_length)
    
    # Ensure we have at least one episode
    num_episodes = max(1, num_episodes)
    
    # Create a figure with subplots for each episode (2 per row)
    num_rows = (num_episodes + 1) // 2  # Ceiling division to get number of rows
    num_cols = min(2, num_episodes)  # At most 2 columns
    
    # Ensure we have at least one column
    num_cols = max(1, num_cols)
    
    # Use LaTeX-style figure dimensions (golden ratio)
    if num_cols == 1:
        fig_width = 6  # Single column width for publications
    else:
        fig_width = 10  # Two-column width for publications
    
    fig_height = fig_width * (num_rows / num_cols) * 0.75  # Approximate golden ratio
    
    # Create figure and subplots with constrained layout for better spacing
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_width, fig_height), 
                            sharey=True, squeeze=False, constrained_layout=True)
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Define a scientific color scheme suitable for publications
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    # Store learning rates for each agent across episodes
    learning_rates = {agent_id: [] for agent_id in incorrect_probs.keys()}
    
    # Plot each episode in a separate subplot
    for episode in range(num_episodes):
        ax = axes[episode]
        
        # Apply LaTeX-style formatting to this subplot
        format_axis_in_latex_style(ax)
        
        # Set subplot title with LaTeX-style formatting
        ax.set_title(f"Episode {episode+1}", fontsize=11)
        
        # Calculate start and end indices for this episode
        start_idx = episode * episode_length
        end_idx = min(start_idx + episode_length, total_steps)
        
        # Plot each agent's data for this episode
        for i, (agent_id, probs) in enumerate(sorted(incorrect_probs.items())):
            agent_color = colors[i % len(colors)]  # Cycle through colors if more agents than colors
            
            # Skip if we're out of data for this agent
            if start_idx >= len(probs):
                continue
                
            # Extract data for this episode
            episode_probs = probs[start_idx:min(end_idx, len(probs))]
            time_steps = np.arange(len(episode_probs))
            
            # Create label for this agent
            label = f"Agent {agent_id}"
            
            # Plot with appropriate scale
            if log_scale:
                line, = ax.semilogy(time_steps, episode_probs, 
                                  label=label,
                                  color=agent_color,
                                  linewidth=1.5,
                                  marker='o',
                                  markersize=3,
                                  markevery=max(1, len(episode_probs)//10))  # Show markers at regular intervals
            else:
                line, = ax.plot(time_steps, episode_probs, 
                              label=label,
                              color=agent_color,
                              linewidth=1.5,
                              marker='o',
                              markersize=3,
                              markevery=max(1, len(episode_probs)//10))  # Show markers at regular intervals
            
            # Calculate and display learning rate if requested
            if show_learning_rates and len(episode_probs) >= 10:
                learning_rate = calculate_learning_rate(episode_probs)
                learning_rates[agent_id].append(learning_rate)
                
                # Update the label with learning rate
                line.set_label(f"{label} (r = {learning_rate:.4f})")
                
                # Plot fitted exponential decay
                x = np.arange(len(episode_probs))
                initial_value = episode_probs[0]
                y = np.exp(-learning_rate * x) * initial_value
                
                if log_scale:
                    ax.semilogy(x, y, '--', alpha=0.3, color=line.get_color())
                else:
                    ax.plot(x, y, '--', alpha=0.3, color=line.get_color())
        
        # Set labels and grid
        ax.set_xlabel("Time Steps")
        if episode == 0:  # Only set y-label on the first subplot
            ax.set_ylabel("Incorrect Action Probability" + (" (log scale)" if log_scale else ""))
        
        ax.grid(True, which="both" if log_scale else "major", ls="--", alpha=0.7)
        
        # Set y-axis limits for better visualization
        if log_scale:
            ax.set_ylim(0.001, 1.0)
        else:
            ax.set_ylim(0, 1.0)
        
        # Add legend to each subplot
        ax.legend(loc='best', fontsize='small')
    
    # Set a common title for the entire figure with LaTeX-style formatting
    fig.suptitle(title, fontsize=13, y=1.02)
    
    # Add a text box with average learning rates across episodes
    if show_learning_rates and num_episodes > 1:
        avg_rates_text = "Average Learning Rates:\n"
        for agent_id, rates in learning_rates.items():
            if rates:  # Only include if we have rates
                avg_rate = np.mean(rates)
                avg_rates_text += f"Agent {agent_id}: {avg_rate:.4f}\n"
        
        # Add text box to the figure with LaTeX-style formatting
        fig.text(0.01, 0.01, avg_rates_text, fontsize=9, 
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Handle unused subplots
    for i in range(num_episodes, len(axes)):
        axes[i].set_visible(False)
    
    # Save the figure in publication-quality formats
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Get base filename without extension
            base_path = os.path.splitext(save_path)[0]
            
            # Save in multiple formats for publication
            save_figure_for_publication(
                fig=fig,
                filename=base_path,
                dpi=300,
                formats=['png']
            )
            
            plt.close()
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
            plt.close()
    else:
        plt.show()
    

def plot_belief_distributions(
    metrics,
    agent_id,
    num_states,
    output_dir,
    use_latex=False
) -> None:
    """
    Plot belief distributions over time for a specific agent.
    
    Args:
        metrics: Dictionary of metrics
        agent_id: Agent ID to plot for
        num_states: Number of possible states (for dimensionality)
        output_dir: Directory to save plots
        use_latex: Whether to use LaTeX styling
    """
    if 'belief_distributions' not in metrics or agent_id not in metrics['belief_distributions']:
        print(f"No belief distribution data available for agent {agent_id}")
        return
    
    # Get belief distributions for this agent
    belief_distributions = metrics['belief_distributions'][agent_id]
    
    if not belief_distributions:
        print(f"Empty belief distribution history for agent {agent_id}")
        return
    
    # Get true states if available
    true_states = metrics.get('true_states', None)
    
    # Create a single figure for this agent
    fig_width = 10
    fig_height = 6
    
    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    
    if use_latex:
        set_latex_style()
    
    # Convert tensors to numpy
    belief_values = []
    for belief_dist in belief_distributions:
        if isinstance(belief_dist, torch.Tensor):
            # Handle different tensor shapes
            if belief_dist.dim() > 1:
                belief_np = belief_dist.squeeze().detach().cpu().numpy()
            else:
                belief_np = belief_dist.detach().cpu().numpy()
        else:
            belief_np = np.array(belief_dist)
        
        # Ensure we have the right dimensionality
        if belief_np.size >= num_states:
            belief_np = belief_np[:num_states]  # Truncate if larger
        else:
            # Pad with zeros if smaller
            padded = np.zeros(num_states)
            padded[:belief_np.size] = belief_np
            belief_np = padded
        
        belief_values.append(belief_np)
    
    # Convert to numpy array
    belief_values = np.array(belief_values)
    
    # Get time steps
    time_steps = np.arange(len(belief_values))
    
    # Plot each state belief
    colors = plt.cm.viridis(np.linspace(0, 1, num_states))
    for state in range(num_states):
        plt.plot(
            time_steps, 
            belief_values[:, state], 
            label=f'State {state}',
            color=colors[state],
            linewidth=2
        )
    
    # Highlight true state changes if available
    if true_states:
        # Limit true states to the length of belief values if needed
        if len(true_states) > len(time_steps):
            true_states = true_states[:len(time_steps)]
        elif len(true_states) < len(time_steps):
            # Pad with the last state
            true_states = true_states + [true_states[-1]] * (len(time_steps) - len(true_states))
        
        # Add vertical lines at state changes
        prev_state = true_states[0] if true_states else None
        for t, state in enumerate(true_states):
            if state != prev_state:
                plt.axvline(x=t, color='red', linestyle='--', alpha=0.5)
                prev_state = state
        
        # Color the background based on true state
        state_changes = [0]  # Start of first state
        current_state = true_states[0]
        for t, state in enumerate(true_states[1:], 1):
            if state != current_state:
                state_changes.append(t)
                current_state = state
        state_changes.append(len(true_states))  # End of last state
        
        # Color regions based on state
        for i in range(len(state_changes)-1):
            start = state_changes[i]
            end = state_changes[i+1]
            state = true_states[start]
            color = 'lightgreen' if state > 0 else 'lightcoral'  # Green for good state, red for bad
            plt.axvspan(start, end, alpha=0.2, color=color)
    
    # Add horizontal line at 0.5 for reference
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.title(f'Agent {agent_id} Belief Distribution Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Probability')
    plt.ylim(0, 1.05)  # Set y-limit for probabilities
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format axis in LaTeX style
    format_axis_in_latex_style(ax)
    
    # Save figure
    output_path = output_dir / f'agent_{agent_id}_belief_distribution.png'
    if use_latex:
        save_figure_for_publication(output_dir / f'agent_{agent_id}_belief_distribution', formats=['pdf', 'png'])
    else:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_agent_actions(actions, true_states, title, save_path=None, episode_length=None, num_episodes=1):
    """
    Plot the actions taken by agents over time.
    
    Args:
        actions: Dictionary mapping agent IDs to lists of actions
        true_states: List of true states at each time step
        title: Plot title
        save_path: Path to save the plot (if None, displays the plot)
        episode_length: Length of each episode (for marking episode boundaries)
        num_episodes: Number of episodes in the data
    """
    # Limit to first 10 episodes for readability
    if num_episodes > 10:
        print(f"Limiting agent actions plot to first 10 episodes out of {num_episodes} total episodes")
        num_episodes = 10
        
        # Limit the data to only include the episodes we're plotting
        if episode_length:
            max_steps = num_episodes * episode_length
            true_states = true_states[:max_steps] if true_states else []
            for agent_id in actions:
                if len(actions[agent_id]) > max_steps:
                    actions[agent_id] = actions[agent_id][:max_steps]
    # Determine number of agents to plot (limit to 8 for readability)
    agent_ids = list(actions.keys())
    num_agents_to_plot = min(8, len(agent_ids))
    selected_agents = agent_ids[:num_agents_to_plot]
    
    # Create figure with LaTeX-style dimensions (golden ratio)
    fig_width = 8  # Width suitable for single-column journal format
    fig_height = fig_width * 0.618  # Golden ratio
    
    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    
    # Apply LaTeX-style formatting to the axis
    format_axis_in_latex_style(ax)
    
    # Get colors from the LaTeX-style color cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    # Plot true state with improved styling
    if true_states:
        ax.plot(true_states, '-', linewidth=1.5, label='True State', color='black', 
               marker='s', markersize=3, markevery=max(1, len(true_states)//15))
    
    # Plot actions for each agent with improved styling
    for i, agent_id in enumerate(selected_agents):
        agent_actions = actions[agent_id]
        if agent_actions:
            agent_color = colors[i % len(colors)]  # Cycle through colors if more agents than colors
            marker_style = ['o', 's', '^', 'v', 'D', 'p', '*', 'x'][i % 8]  # Different markers for each agent
            
            ax.plot(agent_actions, marker=marker_style, markersize=3, alpha=0.8, 
                   label=f'Agent {agent_id}', color=agent_color, linewidth=1.2,
                   markevery=max(1, len(agent_actions)//15))  # Show fewer markers for clarity
    
    # Add episode boundaries if episode_length is provided
    if episode_length and num_episodes > 1:
        for ep in range(1, num_episodes):
            ax.axvline(x=ep*episode_length, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Set labels and title with LaTeX-style formatting
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Action / State')
    ax.set_title(title)
    
    # Add legend with LaTeX-style formatting
    ax.legend(loc='best', framealpha=0.9, edgecolor='black', fancybox=False)
    
    # Set y-ticks to match state indices if we can determine the number of states
    if true_states:
        num_states = max(true_states) + 1
        ax.set_yticks(range(num_states))
        ax.set_yticklabels([f'State {s}' for s in range(num_states)])
    
    # Save the figure in publication-quality formats
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Get base filename without extension
            base_path = os.path.splitext(save_path)[0]
            
            # Save in multiple formats for publication
            save_figure_for_publication(
                fig=fig,
                filename=base_path,
                dpi=300,
                formats=['png']
            )
            
            plt.close()
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
            plt.close()
    else:
        plt.show()
        plt.close()


def plot_allocations(metrics, output_dir, use_latex=False):
    """Plot agent allocations over time for the Strategic Experimentation environment.
    
    Args:
        metrics: Dictionary of metrics including allocations
        output_dir: Directory to save plots
        use_latex: Whether to use LaTeX styling
    """
    if 'allocations' not in metrics:
        print("No allocation data available for plotting")
        return
    
    # Create figure with consistent dimensions
    plt.figure(figsize=(5, 3))
    
    # Apply LaTeX style if requested
    if use_latex:
        set_latex_style()
    
    # Get axes object for formatting
    ax = plt.gca()
    
    # Apply LaTeX formatting to axes if needed
    if use_latex:
        format_axis_in_latex_style(ax)
    
    # Get allocations for each agent
    allocations = metrics['allocations']
    num_agents = len(allocations)
    
    # Plot allocations for each agent
    for agent_id, agent_allocations in allocations.items():
        if len(agent_allocations) == 0:
            continue
            
        # Convert to numpy array
        if isinstance(agent_allocations[0], torch.Tensor):
            allocations_np = [a.item() for a in agent_allocations]
        else:
            allocations_np = agent_allocations
            
        # Create x-axis (timesteps)
        timesteps = range(len(allocations_np))
        
        # Try to convert agent_id to int for consistent formatting
        try:
            agent_id_int = int(agent_id)
            plt.plot(timesteps, allocations_np, label=f'Agent {agent_id_int}')
        except (ValueError, TypeError):
            # Fallback to original agent_id if conversion fails
            plt.plot(timesteps, allocations_np, label=f'Agent {agent_id}')
    
    # Add MPE allocation lines if available
    if 'theoretical_bounds' in metrics and 'mpe_neutral' in metrics['theoretical_bounds']:
        plt.axhline(y=metrics['theoretical_bounds']['mpe_neutral'], color='k', linestyle='--', alpha=0.5, label='MPE (neutral)')
        plt.axhline(y=metrics['theoretical_bounds']['mpe_good_state'], color='g', linestyle='--', alpha=0.5, label='MPE (good state)')
        plt.axhline(y=metrics['theoretical_bounds']['mpe_bad_state'], color='r', linestyle='--', alpha=0.5, label='MPE (bad state)')
    
    # Add title and labels
    plt.title('Agent Resource Allocations Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Allocation to Risky Arm')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure properly based on LaTeX setting
    if use_latex:
        save_figure_for_publication(output_dir / 'agent_allocations', formats=['pdf', 'png'])
    else:
        plt.savefig(output_dir / 'agent_allocations.png', dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_kl_divergence(metrics, output_dir, use_latex=False):
    """
    Plot KL divergence between agent policies and the MPE over time.
    
    Args:
        metrics: Dictionary of metrics including policy_kl_divergence
        output_dir: Directory to save plots
        use_latex: Whether to use LaTeX styling
    """
    if 'policy_kl_divergence' not in metrics or not metrics['policy_kl_divergence']:
        print("No KL divergence data available for plotting")
        return
    
    # Create figure with LaTeX-style dimensions if requested
    fig_width = 5
    fig_height = 3
    
    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    
    if use_latex:
        set_latex_style()
    
    # Get KL divergence for each agent
    kl_divergences = metrics['policy_kl_divergence']
    true_states = metrics.get('true_states', [])
    
    # Get number of agents and timesteps
    num_agents = len(kl_divergences)
    
    # Plot KL divergence for each agent
    for agent_id, agent_kl in kl_divergences.items():
        if len(agent_kl) == 0:
            continue
            
        # Create x-axis (timesteps)
        timesteps = range(len(agent_kl))
        
        # Apply moving average for smoother plot
        window_size = min(10, len(agent_kl) // 10) if len(agent_kl) > 10 else 1
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            smooth_kl = np.convolve(agent_kl, kernel, mode='valid')
            smooth_timesteps = timesteps[window_size-1:]
            plt.plot(smooth_timesteps, smooth_kl, label=f'Agent {agent_id}')
        else:
            plt.plot(timesteps, agent_kl, label=f'Agent {agent_id}')
    
    # Add title and labels
    plt.title('KL Divergence to MPE Policy Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('KL Divergence')
    plt.yscale('log')  # Log scale for better visualization
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add indicator lines for convergence thresholds
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='High divergence')
    plt.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Medium divergence')
    plt.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='Low divergence')
    
    # Set y-axis limits
    plt.ylim(bottom=0.001)  # Set minimum to avoid log scale issues
    
    # Format axis in LaTeX style
    format_axis_in_latex_style(ax)
    
    # Save figure
    if use_latex:
        save_figure_for_publication(output_dir / 'kl_divergence', formats=['pdf', 'png'])
    else:
        plt.savefig(output_dir / 'kl_divergence.png', dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_belief_states(metrics, agent_id, output_dir, use_latex=False):
    """
    Plot belief states over time for a specific agent.
    
    Args:
        metrics: Dictionary of metrics including belief_states
        agent_id: ID of the agent to plot for
        output_dir: Directory to save plots
        use_latex: Whether to use LaTeX styling
    """
    if 'belief_states' not in metrics or agent_id not in metrics['belief_states']:
        print(f"No belief state data available for agent {agent_id}")
        return
    
    # Get agent's belief state history
    belief_states = metrics['belief_states'][agent_id]
    if not belief_states:
        print(f"Empty belief state history for agent {agent_id}")
        return
    
    # Convert tensors to numpy if needed
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
            belief = np.array(belief).flatten()  # Ensure it's a flat array
        
        belief_arrays.append(belief)
    
    # Get dimensions based on first belief state
    if not belief_arrays:
        print(f"Could not process belief state data for agent {agent_id}")
        return
        
    belief_dim = len(belief_arrays[0])
    timesteps = range(len(belief_arrays))
    
    # Create figure with LaTeX-style dimensions if requested
    fig_width = 5
    fig_height = 3
    
    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    
    if use_latex:
        set_latex_style()
    
    # Plot each belief dimension
    for dim in range(min(belief_dim, 10)):  # Limit to 10 dimensions to avoid overcrowding
        values = [belief[dim] for belief in belief_arrays]
        plt.plot(timesteps, values, label=f'Dim {dim}')
    
    # Add title and labels
    plt.title(f'Agent {agent_id} Belief State Evolution')
    plt.xlabel('Time Steps')
    plt.ylabel('Belief State Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format axis in LaTeX style
    format_axis_in_latex_style(ax)
    
    # Save figure
    output_path = output_dir / f'agent_{agent_id}_belief_states.png'
    if use_latex:
        save_figure_for_publication(output_dir / f'agent_{agent_id}_belief_states', formats=['pdf', 'png'])
    else:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    # Also plot belief distributions if available
    if 'belief_distributions' in metrics and agent_id in metrics['belief_distributions'] and 'num_states' in dir(metrics):
        # Try to get number of states from metrics or environment reference
        num_states = metrics.get('num_states', 2)  # Default to 2 if not available
        plot_belief_distributions(metrics, agent_id, num_states, output_dir, use_latex)

def plot_latent_states(metrics, agent_id, output_dir, use_latex=False):
    """
    Plot latent states over time for a specific agent.
    
    Args:
        metrics: Dictionary of metrics including latent_states
        agent_id: ID of the agent to plot for
        output_dir: Directory to save plots
        use_latex: Whether to use LaTeX styling
    """
    if 'latent_states' not in metrics or agent_id not in metrics['latent_states']:
        print(f"No latent state data available for agent {agent_id}")
        return
    
    # Get agent's latent state history
    latent_states = metrics['latent_states'][agent_id]
    if not latent_states:
        print(f"Empty latent state history for agent {agent_id}")
        return
    
    # Convert tensors to numpy if needed
    latent_arrays = []
    for latent in latent_states:
        if isinstance(latent, torch.Tensor):
            # Handle different tensor shapes
            if latent.dim() == 3:  # [1, batch_size, latent_dim]
                latent = latent.squeeze(0).squeeze(0).detach().cpu().numpy()
            elif latent.dim() == 2:  # [batch_size, latent_dim]
                latent = latent.squeeze(0).detach().cpu().numpy()
            elif latent.dim() == 1:  # [latent_dim]
                latent = latent.detach().cpu().numpy()
        elif isinstance(latent, (list, np.ndarray)):
            latent = np.array(latent).flatten()  # Ensure it's a flat array
        
        latent_arrays.append(latent)
    
    # Get dimensions based on first latent state
    if not latent_arrays:
        print(f"Could not process latent state data for agent {agent_id}")
        return
        
    latent_dim = len(latent_arrays[0])
    timesteps = range(len(latent_arrays))
    
    # Create figure with LaTeX-style dimensions if requested
    fig_width = 5
    fig_height = 3
    
    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    
    if use_latex:
        set_latex_style()
    
    # Plot each latent dimension
    for dim in range(min(latent_dim, 10)):  # Limit to 10 dimensions to avoid overcrowding
        values = [latent[dim] for latent in latent_arrays]
        plt.plot(timesteps, values, label=f'Dim {dim}')
    
    # Add title and labels
    plt.title(f'Agent {agent_id} Latent State Evolution')
    plt.xlabel('Time Steps')
    plt.ylabel('Latent State Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format axis in LaTeX style
    format_axis_in_latex_style(ax)
    
    # Save figure
    output_path = output_dir / f'agent_{agent_id}_latent_states.png'
    if use_latex:
        save_figure_for_publication(output_dir / f'agent_{agent_id}_latent_states', formats=['pdf', 'png'])
    else:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_policy_cutoff_vs_belief(metrics, output_dir, use_latex=False):
    """
    Plot agent allocation (policy) as a function of belief, to visualize cutoff beliefs.
    Args:
        metrics: Dictionary of metrics including 'belief_distributions' and 'allocations'
        output_dir: Directory to save plots
        use_latex: Whether to use LaTeX styling
    """

    if 'agent_beliefs' not in metrics or 'allocations' not in metrics:
        print("No belief or allocation data available for plotting cutoff.")
        return

    num_agents = len(metrics['allocations'])

    good_state_index = 1

    # Create figure with consistent dimensions
    plt.figure(figsize=(5, 3))
    
    # Apply LaTeX style if requested
    if use_latex:
        set_latex_style()
    
    # Get axes object for formatting
    ax = plt.gca()
    
    # Apply LaTeX formatting to axes if needed
    if use_latex:
        format_axis_in_latex_style(ax)
        
    for agent_id in metrics['allocations']:
        beliefs = []
        allocations = []
        for t, alloc in enumerate(metrics['allocations'][agent_id]):
            # Get belief in 'good' state at time t
            if t < len(metrics['agent_beliefs'][agent_id]):
                belief_good = metrics['agent_beliefs'][agent_id][t]
                beliefs.append(belief_good)
                allocations.append(alloc)
                print(f"Agent {agent_id} belief at time {t}: {belief_good} allocation: {alloc}")

        # Scatter plot of (belief, allocation) pairs
        plt.scatter(beliefs, allocations, label=f'Agent {int(agent_id)}', alpha=0.5, s=15)

        # Add regression line if enough points
        if len(beliefs) >= 2:
            x = np.array(beliefs)
            y = np.array(allocations)
            # Only fit if there is some variance in x
            if np.std(x) > 0:
                coeffs = np.polyfit(x, y, 1)
                reg_x = np.linspace(np.min(x), np.max(x), 100)
                reg_y = np.polyval(coeffs, reg_x)
                plt.plot(reg_x, reg_y, linestyle='--', linewidth=2, label=f'Regression Agent {int(agent_id)}')

    plt.xlabel("Belief in Good State")
    plt.ylabel("Allocation to Risky Arm")
    plt.title("Policy Cutoff: Allocation vs. Belief")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure properly based on LaTeX setting
    if use_latex:
        save_figure_for_publication(output_dir / 'policy_cutoff_vs_belief', formats=['pdf', 'png'])
    else:
        plt.savefig(output_dir / 'policy_cutoff_vs_belief.png', dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_good_belief_over_time(metrics, output_dir, use_latex=False):
    """
    Plot the belief in the good state over time for each agent.
    Args:
        metrics: Dictionary of metrics including 'agent_beliefs' or 'belief_distributions'
        output_dir: Directory to save plots
        use_latex: Whether to use LaTeX styling
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import math
    from pathlib import Path

    # Try to use agent_beliefs if available, otherwise fallback to belief_distributions
    if 'agent_beliefs' in metrics:
        agent_beliefs = metrics['agent_beliefs']
        use_direct = True
        print("Using agent_beliefs")
    else:
        print("No belief data available for plotting good belief over time.")
        return

    # Try to infer the 'good' state index (default to 1, fallback to 0 if only 1 state)
    good_state_index = 1

    # Create figure with consistent dimensions
    plt.figure(figsize=(5, 3))

    # Apply LaTeX style if requested
    if use_latex:
        set_latex_style()
    
    # Get axes object for formatting
    ax = plt.gca()
    
    # Apply LaTeX formatting to axes if needed
    if use_latex:
        format_axis_in_latex_style(ax)

    for agent_id in agent_beliefs:
        good_beliefs = []
        time_steps = []
        for t, belief in enumerate(agent_beliefs[agent_id]):
            # Skip NaN values in the plot
            if belief is None or (isinstance(belief, float) and math.isnan(belief)):
                continue
                
            if use_direct:
                good_belief = belief
            else:
                if isinstance(belief, torch.Tensor):
                    belief = belief.detach().cpu().numpy()
                good_belief = belief[good_state_index]
                if isinstance(good_belief, np.ndarray):
                    if good_belief.size == 1:
                        good_belief = float(good_belief.squeeze())
                    else:
                        good_belief = float(good_belief.flat[0])
            
            # Skip any remaining NaN values after processing
            if isinstance(good_belief, float) and math.isnan(good_belief):
                continue
                
            good_beliefs.append(good_belief)
            time_steps.append(t)
            
        if good_beliefs:  # Only plot if we have valid beliefs
            # Try to convert agent_id to int for consistent formatting with other plots
            try:
                agent_id_int = int(agent_id)
                plt.plot(time_steps, good_beliefs, label=f'Agent {agent_id_int}')
            except (ValueError, TypeError):
                # Fallback to string if conversion fails
                agent_id_str = str(agent_id)
                plt.plot(time_steps, good_beliefs, label=f'Agent {agent_id_str}')

    plt.xlabel("Time Step")
    plt.ylabel("Belief in Good State")
    plt.title("Belief in Good State Over Time")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Make sure output_dir is a Path object and create directory if needed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the figure properly based on LaTeX setting
    if use_latex:
        save_figure_for_publication(output_dir / 'good_belief_over_time', formats=['pdf', 'png'])
    else:
        # Save file with explicit path
        save_path = output_dir / 'good_belief_over_time.png'
        print(f"Saving belief plot to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
