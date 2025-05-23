"""
Synaptic Intelligence Visualization Module for POLARIS.

This module provides functionality to visualize parameter importance in Synaptic Intelligence
when running POLARIS agents with SI enabled.
"""

import os
import signal
import time
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec


def visualize_parameter_importance(agent, output_dir, layer_name=None):
    """
    Visualize parameter importance scores for a specific agent.

    Args:
        agent: A POLARISAgent instance with SI enabled
        output_dir: Directory to save visualizations
        layer_name: Optional specific layer to visualize (if None, will visualize all major layers)
    """
    if not hasattr(agent, "use_si") or not agent.use_si:
        print(
            f"Agent {agent.agent_id} does not use Synaptic Intelligence, skipping visualization"
        )
        return

    # Create visualization directory
    vis_dir = Path(output_dir) / "si_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Get SI trackers from the agent
    si_trackers = {"belief": agent.belief_si, "policy": agent.policy_si}

    # Get models from the agent
    models = {"belief": agent.belief_processor, "policy": agent.policy}

    # Print diagnostic information about SI state
    print(f"\n=== SI Diagnostic Information for Agent {agent.agent_id} ===")
    print(f"Path integrals calculated: {agent.path_integrals_calculated}")
    print(f"Seen true states: {agent.seen_true_states}")
    print(f"SI debug counter: {agent.si_debug_counter}")

    # Check if final layers are excluded
    if hasattr(agent, "si_exclude_final_layers") and agent.si_exclude_final_layers:
        print(f"Excluding final layers from SI protection:")
        print(f"  Belief layers excluded: {agent.excluded_belief_layers}")
        print(f"  Policy layers excluded: {agent.excluded_policy_layers}")

    # For each SI tracker
    for tracker_name, si_tracker in si_trackers.items():
        model = models[tracker_name]

        print(f"\n--- {tracker_name.capitalize()} Network SI Diagnostics ---")
        print(f"Importance factor: {si_tracker.importance}")
        print(f"Damping factor: {si_tracker.damping}")

        # Print excluded layers for this tracker
        if hasattr(si_tracker, "excluded_layers") and si_tracker.excluded_layers:
            print(f"Excluded layers: {si_tracker.excluded_layers}")

        # Check all layers with their importance statistics
        total_params = 0
        nonzero_params = 0
        all_importances = []

        # Overall statistics
        if all_importances:
            all_imp = torch.cat([imp.flatten() for imp in all_importances])
            print(
                f"\nOverall: {nonzero_params}/{total_params} non-zero parameters ({nonzero_params/total_params*100:.2f}%)"
            )
            print(
                f"Global min: {torch.min(all_imp).item():.8f}, Global max: {torch.max(all_imp).item():.8f}"
            )
            print(
                f"Global mean: {torch.mean(all_imp).item():.8f}, Global std: {torch.std(all_imp).item():.8f}"
            )

        # Check if there are any non-zero importance scores
        has_nonzero_importance = False
        for name, param in model.named_parameters():
            if param.requires_grad and name in si_tracker.importance_scores:
                if torch.sum(torch.abs(si_tracker.importance_scores[name])) > 0:
                    has_nonzero_importance = True
                    break

        if not has_nonzero_importance:
            print(
                f"WARNING: All importance scores for {tracker_name} are zero. SI may not be properly accumulating importance."
            )
            print(f"This can happen if:")
            print(f"  1. No true state transitions have occurred")
            print(f"  2. calculate_path_integrals() hasn't been called")
            print(f"  3. Parameters aren't changing during training")
            print(
                f"  4. register_task() wasn't called after accumulating path integrals"
            )

        # Layers to visualize
        if layer_name:
            layers_to_visualize = {layer_name}
        else:
            # Find all weight parameters
            layers_to_visualize = set()
            for name, param in model.named_parameters():
                if (
                    param.requires_grad
                    and "weight" in name
                    and name in si_tracker.importance_scores
                ):
                    # Only include main layers (exclude small linear layers, biases, etc.)
                    if (
                        param.numel() > 100
                    ):  # Only include layers with more than 100 parameters
                        layers_to_visualize.add(name)

        # For each layer we want to visualize
        for layer_name in layers_to_visualize:
            # Skip if not in importance scores
            if layer_name not in si_tracker.importance_scores:
                continue

            # Get importance scores for this layer
            importance = si_tracker.importance_scores[layer_name].detach().cpu().numpy()

            # Apply transformation to enhance small values (log transformation)
            # Add a small epsilon to avoid log(0)
            epsilon = 1e-10
            abs_importance = np.abs(importance)

            # Check if all values are zero or nearly zero
            if np.max(abs_importance) < epsilon:
                print(
                    f"Warning: All importance values in {layer_name} are near zero. Still attempting visualization with enhancement."
                )

            # Get number of dimensions
            ndim = len(importance.shape)

            # Visualize differently based on dimensions
            if ndim == 1:  # 1D tensor - create a simple bar chart
                visualize_1d_importance(
                    abs_importance, layer_name, vis_dir, tracker_name, agent.agent_id
                )
            elif ndim == 2:  # 2D tensor - create a heatmap
                visualize_2d_importance(
                    abs_importance, layer_name, vis_dir, tracker_name, agent.agent_id
                )
            else:  # Higher dimensions - flatten and create histogram
                visualize_nd_importance(
                    abs_importance, layer_name, vis_dir, tracker_name, agent.agent_id
                )

    print(f"\nGenerated SI visualizations for agent {agent.agent_id} in {vis_dir}")
    print("=" * 50)


def visualize_1d_importance(importance, layer_name, vis_dir, tracker_name, agent_id):
    """Visualize importance for 1D tensor."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Check if all values are very similar (homogeneous)
    max_val = np.max(importance)
    min_val = np.min(importance)
    range_val = max_val - min_val

    # Print diagnostics
    print(
        f"Layer {layer_name} 1D importance stats: min={min_val:.8f}, max={max_val:.8f}, range={range_val:.8f}"
    )

    if range_val < 1e-6 or max_val < 1e-8:
        print(
            f"WARNING: 1D Importance values for {layer_name} are nearly homogeneous or all near zero."
        )

    # Sort values by importance for better visualization
    sorted_indices = np.argsort(importance)
    sorted_importance = importance[sorted_indices]

    # Create bar chart with color gradient
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 1.0, len(sorted_importance)))

    # Plot only the top 50 most important values if there are many
    if len(sorted_importance) > 50:
        indices = sorted_indices[-50:]
        values = importance[indices]
        pos = np.arange(len(indices))
        ax.bar(pos, values, color=colors[-50:])
        ax.set_xticks(pos)
        ax.set_xticklabels([str(i) for i in indices], rotation=90, fontsize=8)
        title_suffix = " (Top 50 parameters)"
    else:
        ax.bar(np.arange(len(importance)), importance, color=colors)
        title_suffix = ""

    # Add titles and labels
    ax.set_title(
        f"Parameter Importance - {tracker_name} - {layer_name} (Agent {agent_id}){title_suffix}"
    )
    ax.set_xlabel("Parameter Index")
    ax.set_ylabel("Absolute Importance")

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Relative Importance")

    # Save figure
    safe_layer_name = layer_name.replace(".", "_")
    fig.savefig(
        vis_dir
        / f"si_importance_1d_{tracker_name}_{safe_layer_name}_agent{agent_id}.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def visualize_2d_importance(importance, layer_name, vis_dir, tracker_name, agent_id):
    """Visualize importance for 2D tensor as heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Apply log transformation for better visualization
    epsilon = 1e-10
    log_importance = np.log(importance + epsilon)

    # Check if all values are very similar (homogeneous)
    max_val = np.max(importance)
    min_val = np.min(importance)
    range_val = max_val - min_val

    # Print diagnostics
    print(
        f"Layer {layer_name} importance stats: min={min_val:.8f}, max={max_val:.8f}, range={range_val:.8f}"
    )

    # If values are all very similar or near zero, add a warning
    if range_val < 1e-6 or max_val < 1e-8:
        print(
            f"WARNING: Importance values for {layer_name} are nearly homogeneous or all near zero."
        )

    # Use a robust color normalization to enhance contrast
    vmax = np.percentile(log_importance, 95)  # 95th percentile for upper bound
    vmin = np.percentile(log_importance, 5)  # 5th percentile for lower bound

    # Create heatmap with enhanced contrast
    heatmap = sns.heatmap(
        log_importance,
        cmap="viridis",
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Log Importance"},
    )

    # Add titles and labels
    ax.set_title(
        f"Parameter Importance Heatmap - {tracker_name} - {layer_name} (Agent {agent_id})"
    )
    ax.set_xlabel("Output Dimension")
    ax.set_ylabel("Input Dimension")

    # Save figure without tight_layout
    safe_layer_name = layer_name.replace(".", "_")
    fig.savefig(
        vis_dir
        / f"si_importance_2d_{tracker_name}_{safe_layer_name}_agent{agent_id}.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Also create a version showing the most important connections
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create a mask for the top 5% most important connections
    flat_importance = importance.flatten()
    threshold = np.percentile(flat_importance, 95)
    mask = importance < threshold

    # Create heatmap with only important connections visible
    heatmap = sns.heatmap(
        importance,
        cmap="viridis",
        ax=ax,
        mask=mask,
        cbar_kws={"label": "Importance (Top 5%)"},
    )

    # Add titles and labels
    ax.set_title(
        f"Top 5% Important Connections - {tracker_name} - {layer_name} (Agent {agent_id})"
    )
    ax.set_xlabel("Output Dimension")
    ax.set_ylabel("Input Dimension")

    # Save figure
    fig.savefig(
        vis_dir
        / f"si_importance_2d_top_connections_{tracker_name}_{safe_layer_name}_agent{agent_id}.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def visualize_nd_importance(importance, layer_name, vis_dir, tracker_name, agent_id):
    """Visualize importance for higher dimensional tensor by flattening."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Flatten importance
    flat_importance = importance.flatten()

    # Apply log transformation for better visualization
    epsilon = 1e-10
    log_importance = np.log(flat_importance + epsilon)

    # Create histogram with better binning
    counts, bins, patches = ax.hist(log_importance, bins=50, alpha=0.7)

    # Color the histogram bars using a gradient
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    norm = plt.Normalize(min(bin_centers), max(bin_centers))
    for c, p in zip(bin_centers, patches):
        plt.setp(p, "facecolor", plt.cm.viridis(norm(c)))

    # Add titles and labels
    ax.set_title(
        f"Parameter Importance Distribution - {tracker_name} - {layer_name} (Agent {agent_id})"
    )
    ax.set_xlabel("Log Absolute Importance")
    ax.set_ylabel("Count")

    # Add vertical line at 95th percentile
    percentile_95 = np.percentile(log_importance, 95)
    ax.axvline(x=percentile_95, color="r", linestyle="--")
    ax.text(
        percentile_95,
        ax.get_ylim()[1] * 0.9,
        "95th percentile",
        rotation=90,
        verticalalignment="top",
        color="r",
    )

    # Save figure
    safe_layer_name = layer_name.replace(".", "_")
    fig.savefig(
        vis_dir
        / f"si_importance_dist_{tracker_name}_{safe_layer_name}_agent{agent_id}.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def visualize_si_mechanism():
    """Create an explanatory diagram of the SI mechanism."""
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(12, 8))

    # Disable axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set the title
    ax.set_title("Synaptic Intelligence (SI) in POLARIS", fontsize=16)

    # Draw the main components
    task_box_1 = Rectangle(
        (0.1, 0.7), 0.2, 0.2, fill=True, alpha=0.3, ec="black", fc="blue"
    )
    task_box_2 = Rectangle(
        (0.4, 0.7), 0.2, 0.2, fill=True, alpha=0.3, ec="black", fc="green"
    )
    param_box = Rectangle(
        (0.25, 0.4), 0.2, 0.2, fill=True, alpha=0.3, ec="black", fc="gray"
    )
    importance_box = Rectangle(
        (0.55, 0.4), 0.2, 0.2, fill=True, alpha=0.3, ec="black", fc="red"
    )
    loss_box = Rectangle(
        (0.4, 0.1), 0.2, 0.2, fill=True, alpha=0.3, ec="black", fc="purple"
    )

    # Add boxes to the plot
    ax.add_patch(task_box_1)
    ax.add_patch(task_box_2)
    ax.add_patch(param_box)
    ax.add_patch(importance_box)
    ax.add_patch(loss_box)

    # Add text labels
    ax.text(0.2, 0.8, "True State 1", ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.8, "True State 2", ha="center", va="center", fontsize=12)
    ax.text(0.35, 0.5, "Parameters\nθ", ha="center", va="center", fontsize=12)
    ax.text(0.65, 0.5, "Importance\nΩ", ha="center", va="center", fontsize=12)
    ax.text(
        0.5, 0.2, "SI Loss\nL = λ/2 * Ω(θ-θ*)²", ha="center", va="center", fontsize=12
    )

    # Add arrows
    arrow_style = dict(arrowstyle="->", linewidth=2, color="black")

    # Task 1 to Parameters
    task1_param = FancyArrowPatch(
        (0.2, 0.7), (0.3, 0.6), connectionstyle="arc3,rad=-0.2", **arrow_style
    )
    ax.add_patch(task1_param)
    ax.text(0.2, 0.65, "Update θ", fontsize=10, ha="center")

    # Task 2 to Parameters
    task2_param = FancyArrowPatch(
        (0.5, 0.7), (0.4, 0.6), connectionstyle="arc3,rad=0.2", **arrow_style
    )
    ax.add_patch(task2_param)
    ax.text(0.5, 0.65, "Update θ", fontsize=10, ha="center")

    # Parameters to Importance (calculation)
    param_imp = FancyArrowPatch((0.45, 0.5), (0.55, 0.5), **arrow_style)
    ax.add_patch(param_imp)
    ax.text(0.5, 0.52, "Calculate\nΩ = ∫g·dθ / (Δθ²+ξ)", fontsize=10, ha="center")

    # Parameters and Importance to Loss
    param_loss = FancyArrowPatch(
        (0.35, 0.4), (0.45, 0.3), connectionstyle="arc3,rad=-0.2", **arrow_style
    )
    imp_loss = FancyArrowPatch(
        (0.65, 0.4), (0.55, 0.3), connectionstyle="arc3,rad=0.2", **arrow_style
    )
    ax.add_patch(param_loss)
    ax.add_patch(imp_loss)

    # Loss back to parameters (regularization)
    loss_param = FancyArrowPatch(
        (0.4, 0.2), (0.3, 0.4), connectionstyle="arc3,rad=0.2", **arrow_style
    )
    ax.add_patch(loss_param)
    ax.text(0.3, 0.3, "Regularize", fontsize=10, ha="center")

    # Add explanation text
    explanation = """
    Synaptic Intelligence (SI) in POLARIS:
    
    1. As agents learn during a true state, parameter trajectories are tracked
    2. When transitioning to a new true state, importance scores are calculated:
       • Path integrals accumulate gradient * parameter change during learning
       • Importance = path_integral / (delta_parameter² + damping)
    3. When learning in the new true state:
       • SI loss penalizes changes to parameters important for previous states
       • Loss = λ/2 * Ω(θ-θ*)², where θ* are parameter values after previous state
    4. This preserves knowledge from earlier true states while adapting to new ones
    """

    plt.figtext(0.02, 0.02, explanation, fontsize=12, ha="left", va="bottom")

    # Return without tight_layout
    return fig


def visualize_all_agents_si(agents, output_dir):
    """
    Create visualizations for all agents with SI enabled.

    Args:
        agents: Dictionary of agent_id to POLARISAgent instances
        output_dir: Directory to save visualizations
    """
    # Create visualization directory
    vis_dir = Path(output_dir) / "si_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Create SI mechanism diagram
    diagram = visualize_si_mechanism()
    diagram.savefig(vis_dir / "si_mechanism_diagram.png", dpi=200, bbox_inches="tight")
    plt.close(diagram)

    # Count how many agents use SI
    si_agents = 0

    # Visualize each agent's parameter importance
    for agent_id, agent in agents.items():
        if hasattr(agent, "use_si") and agent.use_si:
            si_agents += 1
            visualize_parameter_importance(agent, output_dir)

    if si_agents > 0:
        print(f"Generated SI visualizations for {si_agents} agents in {vis_dir}")
    else:
        print("No agents with SI enabled found.")


def visualize_layer_importances_across_agents(agents, output_dir, layer_name):
    """
    Compare layer importance across different agents.

    Args:
        agents: Dictionary of agent_id to POLARISAgent instances
        output_dir: Directory to save visualizations
        layer_name: Layer name to compare across agents
    """
    # Create visualization directory
    vis_dir = Path(output_dir) / "si_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Get all agents with SI
    si_agents = {
        agent_id: agent
        for agent_id, agent in agents.items()
        if hasattr(agent, "use_si") and agent.use_si
    }

    if not si_agents:
        print("No agents with SI enabled found.")
        return

    # Components to visualize
    components = ["belief", "policy"]

    for component in components:
        # Get the corresponding SI tracker for each agent
        si_trackers = {}
        for agent_id, agent in si_agents.items():
            if component == "belief":
                si_trackers[agent_id] = agent.belief_si
            elif component == "policy":
                si_trackers[agent_id] = agent.policy_si

        # Check if the layer exists in any agent
        layer_exists = False
        for agent_id, tracker in si_trackers.items():
            if layer_name in tracker.importance_scores:
                layer_exists = True
                break

        if not layer_exists:
            print(f"Layer {layer_name} not found in any agent's {component} tracker")
            continue

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # For each agent
        for agent_id, tracker in si_trackers.items():
            if layer_name in tracker.importance_scores:
                # Get importance scores
                importance = (
                    tracker.importance_scores[layer_name].detach().cpu().numpy()
                )

                # Calculate mean absolute importance per input neuron
                if len(importance.shape) == 2:
                    # For 2D tensor, take mean across output dimension
                    mean_importance = np.mean(np.abs(importance), axis=1)
                else:
                    # Otherwise flatten
                    mean_importance = np.abs(importance.flatten())

                # Find top 20 values and their indices
                top_indices = np.argsort(mean_importance)[-22:]
                top_values = mean_importance[top_indices]

                # Plot top 20 values
                width = 0.35  # width of bars
                offset = 0.4 * (
                    agent_id - list(si_agents.keys())[0]
                )  # offset for agent
                ax.bar(
                    top_indices + offset,  # Position bars with offset
                    top_values,
                    width=width,
                    alpha=0.7,
                    label=f"Agent {agent_id}",
                )

        ax.set_title(f"Top Parameter Importances - {component} - {layer_name}")
        ax.set_xlabel("Parameter Index")
        ax.set_ylabel("Mean Absolute Importance")
        ax.legend()
        fig.savefig(
            vis_dir
            / f"si_importance_comparison_{component}_{layer_name.replace('.', '_')}.png",
            bbox_inches="tight",
        )
        plt.close(fig)

    print(f"Generated layer importance comparison visualizations in {vis_dir}")


def create_si_visualizations(agents, output_dir):
    """
    Create a complete set of SI visualizations.

    Args:
        agents: Dictionary of agent_id to POLARISAgent instances
        output_dir: Directory to save visualizations
    """
    try:
        # Create visualization directory
        vis_dir = Path(output_dir) / "si_visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Count how many agents use SI
        si_agents = 0
        for agent_id, agent in agents.items():
            if hasattr(agent, "use_si") and agent.use_si:
                si_agents += 1

        if si_agents == 0:
            print("No agents with SI enabled found. Skipping SI visualizations.")
            return

        print(f"Generating SI visualizations for {si_agents} agents...")

        # Create gradient masking visualizations
        print("Creating gradient masking visualizations...")
        for agent_id, agent in agents.items():
            if hasattr(agent, "use_si") and agent.use_si:
                try:
                    visualize_gradient_masking(agent, output_dir)
                except Exception as e:
                    print(
                        f"Error creating gradient masking visualizations for agent {agent_id}: {e}"
                    )

        # Create parameter specialization visualizations
        print("Creating parameter specialization visualizations...")
        visualize_parameter_specialization(agents, output_dir)

        # Create parameter specialization matrix
        print("Creating parameter specialization matrix...")
        for agent_id, agent in agents.items():
            if hasattr(agent, "use_si") and agent.use_si:
                try:
                    visualize_task_parameter_specialization_matrix(agent, output_dir)
                except Exception as e:
                    print(
                        f"Error creating specialization matrix for agent {agent_id}: {e}"
                    )

        # Create task comparison visualizations
        for agent_id, agent in agents.items():
            if hasattr(agent, "use_si") and agent.use_si:
                try:
                    print(f"Creating visualizations for agent {agent_id}")

                    # Create consolidated task comparison visualization
                    print(f"  Creating consolidated importance visualization...")
                    visualize_consolidated_importance_across_tasks(agent, output_dir)
                    print(f"  Completed consolidated importance visualization")
                except Exception as e:
                    print(f"Error creating visualizations for agent {agent_id}: {e}")

        # Also create comparisons for key layers
        key_layers = [
            "transformer.transformer_encoder.layers.0.linear1.weight",
            "transformer.transformer_encoder.layers.0.self_attn.out_proj.weight",
            "fc_belief.weight",
            "policy_network.0.weight",
        ]

        for layer in key_layers:
            try:
                print(f"Creating layer comparison for {layer}")
                visualize_layer_importances_across_agents(agents, output_dir, layer)
            except Exception as e:
                print(f"Error visualizing layer {layer}: {e}")

        print("SI visualization generation complete!")

    except Exception as e:
        import traceback

        print(f"Error creating SI visualizations: {e}")
        print("Detailed error:")
        traceback.print_exc()


def visualize_importance_across_tasks(agent, output_dir):
    """
    Visualize how importance scores differ across different tasks (true states).

    Args:
        agent: A POLARISAgent instance with SI enabled
        output_dir: Directory to save visualizations
    """
    if not hasattr(agent, "use_si") or not agent.use_si:
        print(
            f"Agent {agent.agent_id} does not use Synaptic Intelligence, skipping visualization"
        )
        return

    if (
        not hasattr(agent, "state_belief_si_trackers")
        or not agent.state_belief_si_trackers
    ):
        print(
            f"Agent {agent.agent_id} has no state-specific SI trackers, skipping comparison"
        )
        return

    # Create visualization directory
    vis_dir = Path(output_dir) / "si_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Print diagnostic info about available states
    print(
        f"\n=== Task Comparison Diagnostic Information for Agent {agent.agent_id} ==="
    )
    print(
        f"Available true states for comparison: {list(agent.state_belief_si_trackers.keys())}"
    )

    # Components to visualize
    components = [
        ("belief", agent.state_belief_si_trackers),
        ("policy", agent.state_policy_si_trackers),
    ]

    # If there's only one state, we can't compare
    if len(agent.state_belief_si_trackers) < 2:
        print(f"Only one true state available, can't compare across tasks")
        return

    # Get layers common to all states
    for component_name, trackers in components:
        # Find common layers across all states
        common_layers = None
        for state, tracker in trackers.items():
            if common_layers is None:
                common_layers = set(tracker.importance_scores.keys())
            else:
                common_layers &= set(tracker.importance_scores.keys())

        if not common_layers:
            print(f"No common layers found across states for {component_name}")
            continue

        # Sort layers by name for consistent order
        common_layers = sorted(list(common_layers))

        # For each common layer
        for layer_name in common_layers:
            # Collect importance scores for this layer across all states
            state_importance = {}
            for state, tracker in trackers.items():
                if layer_name in tracker.importance_scores:
                    state_importance[state] = (
                        tracker.importance_scores[layer_name].detach().cpu().numpy()
                    )

            # Get layer dimensions
            sample_importance = next(iter(state_importance.values()))
            ndim = len(sample_importance.shape)

            if ndim == 1:
                # For 1D layers, create bar charts comparing top important parameters
                visualize_1d_importance_across_tasks(
                    state_importance,
                    layer_name,
                    vis_dir,
                    component_name,
                    agent.agent_id,
                )
            elif ndim == 2:
                # For 2D layers, create heatmap comparisons
                visualize_2d_importance_across_tasks(
                    state_importance,
                    layer_name,
                    vis_dir,
                    component_name,
                    agent.agent_id,
                )

    print(
        f"Generated task comparison visualizations for agent {agent.agent_id} in {vis_dir}"
    )
    print("=" * 50)


def visualize_1d_importance_across_tasks(
    state_importance, layer_name, vis_dir, component_name, agent_id
):
    """Visualize 1D importance scores across different tasks/true states."""
    # Number of states to compare
    num_states = len(state_importance)

    # Create figure with subplots for each state and a comparison
    fig, axes = plt.subplots(
        num_states + 1, 1, figsize=(12, 4 * (num_states + 1)), constrained_layout=True
    )

    # Colors for the states
    colors = plt.cm.tab10(np.linspace(0, 1, num_states))

    # Track global min/max for consistent scaling
    global_min = float("inf")
    global_max = float("-inf")

    # For each state, find min/max
    for state, importance in state_importance.items():
        min_val = np.min(importance)
        max_val = np.max(importance)
        global_min = min(global_min, min_val)
        global_max = max(global_max, max_val)

    # If all values are very close to zero, adjust scaling
    if global_max < 1e-8:
        global_max = 1.0  # Just a default value for visualization

    # For each state, plot individual importance
    for i, (state, importance) in enumerate(state_importance.items()):
        ax = axes[i]

        # Get top 50 indices
        top_indices = np.argsort(importance)[-50:]
        values = importance[top_indices]

        # Plot bar chart
        bars = ax.bar(np.arange(len(top_indices)), values, color=colors[i], alpha=0.7)

        # Add state label
        ax.set_title(
            f"True State {state}: Top Important Parameters - {component_name} - {layer_name}"
        )
        ax.set_xlabel("Parameter Index")
        ax.set_ylabel("Importance")

        # Set y limits consistently
        ax.set_ylim(0, global_max * 1.1)

        # Add parameter indices as x-tick labels
        ax.set_xticks(np.arange(len(top_indices)))
        ax.set_xticklabels([str(i) for i in top_indices], rotation=90, fontsize=8)

    # Now create a comparison plot showing top parameters from each state
    ax = axes[-1]

    # For each state, find top 20 parameters
    top_params_by_state = {}
    all_top_params = set()

    for state, importance in state_importance.items():
        top_indices = np.argsort(importance)[-22:]
        top_params_by_state[state] = top_indices
        all_top_params.update(top_indices)

    # Sort all top parameters
    all_top_params = sorted(list(all_top_params))

    # Width of bars
    width = 0.8 / num_states

    # For each state, plot its importance for all top parameters
    for i, (state, importance) in enumerate(state_importance.items()):
        # Extract values for top parameters
        values = importance[all_top_params]
        # Position for this state's bars
        positions = np.arange(len(all_top_params)) + (i - num_states / 2 + 0.5) * width
        # Plot bars
        ax.bar(
            positions, values, width, label=f"State {state}", color=colors[i], alpha=0.7
        )

    # Add labels and legend
    ax.set_title(
        f"Parameter Importance Comparison Across Tasks - {component_name} - {layer_name}"
    )
    ax.set_xlabel("Parameter Index")
    ax.set_ylabel("Importance")
    ax.legend()

    # Set x-ticks to show parameter indices
    ax.set_xticks(np.arange(len(all_top_params)))
    ax.set_xticklabels([str(p) for p in all_top_params], rotation=90, fontsize=8)

    # Save figure
    safe_layer_name = layer_name.replace(".", "_")
    fig.savefig(
        vis_dir
        / f"si_task_comparison_1d_{component_name}_{safe_layer_name}_agent{agent_id}.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def visualize_2d_importance_across_tasks(
    state_importance, layer_name, vis_dir, component_name, agent_id
):
    """Visualize 2D importance scores across different tasks/true states."""
    # Number of states to compare
    num_states = len(state_importance)

    # Create figure with subplots in a grid
    fig = plt.figure(figsize=(15, 5 * (num_states + 1)))
    gs = GridSpec(num_states + 1, 3, figure=fig)

    # Get global min/max for consistent color scaling
    global_min = float("inf")
    global_max = float("-inf")

    # For each state, find min/max
    for state, importance in state_importance.items():
        epsilon = 1e-10
        log_importance = np.log(importance + epsilon)
        min_val = np.min(log_importance)
        max_val = np.max(log_importance)
        global_min = min(global_min, min_val)
        global_max = max(global_max, max_val)

    # If all values are very close to 0, adjust the scale
    if global_max - global_min < 1e-8:
        global_min = -1
        global_max = 1

    # For each state, create a heatmap
    for i, (state, importance) in enumerate(state_importance.items()):
        ax = fig.add_subplot(gs[i, :])

        # Log transform for better visualization
        epsilon = 1e-10
        log_importance = np.log(importance + epsilon)

        # Create heatmap
        sns.heatmap(
            log_importance,
            cmap="viridis",
            ax=ax,
            vmin=global_min,
            vmax=global_max,
            cbar_kws={"label": "Log Importance"},
        )

        # Add title
        ax.set_title(
            f"True State {state}: Parameter Importance - {component_name} - {layer_name}"
        )
        ax.set_xlabel("Output Dimension")
        ax.set_ylabel("Input Dimension")

    # Create a difference plot showing how importance changes between states
    if num_states >= 2:
        # Take the first two states for difference visualization
        states = list(state_importance.keys())
        importance1 = state_importance[states[0]]
        importance2 = state_importance[states[1]]

        # Calculate difference
        diff = importance2 - importance1

        # Create a diverging colormap for the difference
        ax = fig.add_subplot(gs[-1, :])

        # Calculate max absolute difference for symmetric colormap
        max_diff = np.max(np.abs(diff))

        # Create a diverging colormap centered at zero
        sns.heatmap(
            diff,
            cmap="RdBu_r",
            ax=ax,
            vmin=-max_diff,
            vmax=max_diff,
            cbar_kws={"label": "Importance Difference"},
        )

        # Add title
        ax.set_title(
            f"Importance Difference: State {states[1]} - State {states[0]} - {component_name} - {layer_name}"
        )
        ax.set_xlabel("Output Dimension")
        ax.set_ylabel("Input Dimension")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    safe_layer_name = layer_name.replace(".", "_")
    fig.savefig(
        vis_dir
        / f"si_task_comparison_2d_{component_name}_{safe_layer_name}_agent{agent_id}.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def visualize_consolidated_importance_across_tasks(agent, output_dir):
    """
    Create a consolidated figure showing 1D parameter importance comparison
    across tasks for all layers' weights and biases.

    Args:
        agent: A POLARISAgent instance with SI enabled
        output_dir: Directory to save visualizations
    """

    # Define a timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError("Visualization timed out")

    # Set a timeout of 60 seconds
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)

    start_time = time.time()

    try:
        if not hasattr(agent, "use_si") or not agent.use_si:
            print(
                f"Agent {agent.agent_id} does not use Synaptic Intelligence, skipping visualization"
            )
            return

        if (
            not hasattr(agent, "state_belief_si_trackers")
            or not agent.state_belief_si_trackers
        ):
            print(
                f"Agent {agent.agent_id} has no state-specific SI trackers, skipping comparison"
            )
            return

        # Create visualization directory
        vis_dir = Path(output_dir) / "si_visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Get the true states we've seen
        true_states = sorted(list(agent.state_belief_si_trackers.keys()))
        num_states = len(true_states)

        if num_states < 2:
            print(
                f"Agent {agent.agent_id} has only seen {num_states} true states, not enough for comparison"
            )
            return

        # Collect all parameter names and their types (weight/bias) from belief and policy networks
        all_params = {}

        # First, get belief processor parameters
        for name, param in agent.belief_processor.named_parameters():
            # Skip excluded layers
            excluded = (
                hasattr(agent, "excluded_belief_layers")
                and name in agent.excluded_belief_layers
            )

            # Skip parameters that are too large (to avoid memory issues)
            if param.numel() > 10000:
                print(f"Skipping large parameter {name} with {param.numel()} elements")
                continue

            # Only include weights and biases of main layers
            if not ("weight" in name or "bias" in name) or param.numel() < 10:
                continue

            # Determine layer name and type
            if "weight" in name:
                param_type = "weight"
                layer_name = name.replace(".weight", "")
            elif "bias" in name:
                param_type = "bias"
                layer_name = name.replace(".bias", "")
            else:
                param_type = "other"
                layer_name = name

            # Key combines network, layer, and parameter type
            key = f"belief_{layer_name}_{param_type}"
            all_params[key] = {
                "network": "belief",
                "layer": layer_name,
                "type": param_type,
                "name": name,
                "excluded": excluded,
                "shape": param.shape,
            }

        # Then, get policy parameters
        for name, param in agent.policy.named_parameters():
            # Skip excluded layers
            excluded = (
                hasattr(agent, "excluded_policy_layers")
                and name in agent.excluded_policy_layers
            )

            # Skip parameters that are too large (to avoid memory issues)
            if param.numel() > 10000:
                print(f"Skipping large parameter {name} with {param.numel()} elements")
                continue

            # Only include weights and biases of main layers
            if not ("weight" in name or "bias" in name) or param.numel() < 10:
                continue

            # Determine layer name and type
            if "weight" in name:
                param_type = "weight"
                layer_name = name.replace(".weight", "")
            elif "bias" in name:
                param_type = "bias"
                layer_name = name.replace(".bias", "")
            else:
                param_type = "other"
                layer_name = name

            # Key combines network, layer, and parameter type
            key = f"policy_{layer_name}_{param_type}"
            all_params[key] = {
                "network": "policy",
                "layer": layer_name,
                "type": param_type,
                "name": name,
                "excluded": excluded,
                "shape": param.shape,
            }

        # Limit number of parameters to visualize (to avoid memory issues)
        max_params = 22
        if len(all_params) > max_params:
            print(
                f"Limiting visualization to {max_params} parameters (out of {len(all_params)})"
            )
            # Sort by parameter size and select a subset
            params_by_size = sorted(
                all_params.items(), key=lambda x: np.prod(x[1]["shape"]), reverse=True
            )
            selected_params = dict(params_by_size[:max_params])
            all_params = selected_params

        # Create a large figure for the consolidated visualization
        fig = plt.figure(figsize=(22, 14))
        fig.suptitle(
            f"Parameter Importance Comparison Across Tasks for Agent {agent.agent_id}",
            fontsize=18,
        )

        # Determine grid layout
        num_params = len(all_params)
        grid_cols = min(2, num_params)  # 2 columns for weight/bias pairs
        grid_rows = (num_params + grid_cols - 1) // grid_cols  # Ceiling division

        # Counter for subplot position
        subplot_idx = 1

        # Get color mapping for states
        colors = plt.cm.tab10.colors[:num_states]
        color_map = {state: colors[i] for i, state in enumerate(true_states)}

        # Process each parameter
        for key, param_info in sorted(all_params.items()):
            # Get parameter name and info
            name = param_info["name"]
            network = param_info["network"]
            layer = param_info["layer"]
            param_type = param_info["type"]
            is_excluded = param_info["excluded"]

            # Create subplot
            ax = fig.add_subplot(grid_rows, grid_cols, subplot_idx)
            subplot_idx += 1

            # Custom title based on layer and type
            ax_title = f"{network.capitalize()} - {layer} ({param_type})"
            if is_excluded:
                ax_title += " [EXCLUDED FROM SI]"
                ax.set_facecolor("#FFEEEE")  # Light red background for excluded layers
            ax.set_title(ax_title, fontsize=10)

            # Get parameter shape
            param_shape = param_info["shape"]
            param_size = np.prod(param_shape)

            # For each state
            bar_width = 0.8 / num_states

            # If parameter is too large, sample every N elements to reduce size
            if param_size > 100:
                # Sample at most 100 elements
                sample_rate = max(1, param_size // 100)
                x_pos = np.arange(0, param_size, sample_rate)
            else:
                x_pos = np.arange(param_size)

            for state_idx, true_state in enumerate(true_states):
                # Get the appropriate SI tracker
                if network == "belief":
                    tracker = agent.state_belief_si_trackers.get(true_state)
                else:
                    tracker = agent.state_policy_si_trackers.get(true_state)

                if tracker is None:
                    continue

                # Get importance scores for this parameter
                if name in tracker.importance_scores:
                    # Get importance scores
                    importance = tracker.importance_scores[name].detach().cpu().numpy()

                    # Flatten importance scores for 1D plotting
                    flat_importance = importance.flatten()

                    # Sample the importance scores if necessary
                    if param_size > 100:
                        sampled_importance = flat_importance[::sample_rate]
                    else:
                        sampled_importance = flat_importance

                    # Plot bar for each state with slight offset
                    state_offset = state_idx * bar_width - 0.4 + (bar_width / 2)
                    ax.bar(
                        x_pos + state_offset,
                        sampled_importance,
                        width=bar_width,
                        label=f"Task {true_state}" if subplot_idx == 2 else "",
                        color=color_map[true_state],
                        alpha=0.7,
                    )

            # Add axis labels and grid
            ax.set_xlabel("Parameter Index", fontsize=8)
            ax.set_ylabel("Importance", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(alpha=0.3)

        # Add legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_map[state]) for state in true_states
        ]
        labels = [f"Task {state}" for state in true_states]

        # Add "Excluded from SI" to legend
        handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor="#FFEEEE", edgecolor="black")
        )
        labels.append("Excluded from SI Protection")

        fig.legend(handles, labels, loc="upper right", fontsize=10)

        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

        # Save the figure
        fig_path = (
            vis_dir / f"consolidated_parameter_importance_agent{agent.agent_id}.png"
        )
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        elapsed_time = time.time() - start_time
        print(
            f"Saved consolidated parameter importance comparison to {fig_path} in {elapsed_time:.2f} seconds"
        )

    except TimeoutError:
        print(
            f"WARNING: Visualization for agent {agent.agent_id} timed out after 60 seconds. Skipping."
        )
        # Clean up any open plots
        plt.close("all")
    except Exception as e:
        print(
            f"Error in visualize_consolidated_importance_across_tasks for agent {agent.agent_id}: {e}"
        )
        # Clean up any open plots
        plt.close("all")
    finally:
        # Cancel the alarm
        signal.alarm(0)


def visualize_parameter_specialization(agents, output_dir):
    """
    Visualize parameter specialization across tasks.

    This function creates visualizations showing how different tasks use different parameters
    by comparing importance scores across tasks.

    Args:
        agents: Dictionary of agent_id to POLARISAgent instances
        output_dir: Directory to save visualizations
    """
    # Create visualization directory
    vis_dir = Path(output_dir) / "si_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Count how many agents use SI
    si_agents = 0

    for agent_id, agent in agents.items():
        if not hasattr(agent, "use_si") or not agent.use_si:
            continue

        si_agents += 1

        # Check if we have task-specific trackers
        if (
            not hasattr(agent, "state_belief_si_trackers")
            or not agent.state_belief_si_trackers
        ):
            print(
                f"Agent {agent_id} has no state-specific SI trackers, skipping specialization visualization"
            )
            continue

        if len(agent.state_belief_si_trackers) < 2:
            print(
                f"Agent {agent_id} has only seen {len(agent.state_belief_si_trackers)} tasks, not enough for specialization comparison"
            )
            continue

        # Get list of tasks
        tasks = list(agent.state_belief_si_trackers.keys())

        # Components to visualize
        components = [
            ("belief", agent.state_belief_si_trackers),
            ("policy", agent.state_policy_si_trackers),
        ]

        # Track layer-level specialization data
        layer_specialization = {}

        # For each component
        for component_name, trackers in components:
            # Find common layers across all tasks
            common_layers = None
            for task_id, tracker in trackers.items():
                if common_layers is None:
                    common_layers = set(tracker.importance_scores.keys())
                else:
                    common_layers &= set(tracker.importance_scores.keys())

            if not common_layers:
                print(f"No common layers found for {component_name} across tasks")
                continue

            # Create a figure for task parameter specialization
            plt.figure(figsize=(15, 10))

            # Collect data for visualization
            layers_to_show = []
            specialization_scores = []

            # For each layer, calculate a specialization score
            for layer_name in sorted(common_layers):
                # Get importance for this layer across tasks
                layer_importances = {}

                for task_id, tracker in trackers.items():
                    if layer_name in tracker.importance_scores:
                        # Calculate mean absolute importance
                        importance = tracker.importance_scores[layer_name]
                        mean_importance = torch.mean(torch.abs(importance)).item()
                        layer_importances[task_id] = mean_importance

                if not layer_importances:
                    continue

                # Calculate specialization: variance of importance across tasks
                # Higher variance means more specialization
                importances = list(layer_importances.values())
                if len(importances) > 1:
                    # Only include layers with non-zero importance
                    if sum(importances) > 0:
                        # Normalize importances
                        total_importance = sum(importances)
                        normalized_importances = [
                            imp / total_importance for imp in importances
                        ]

                        # Calculate variance as a measure of specialization
                        mean_importance = sum(normalized_importances) / len(
                            normalized_importances
                        )
                        variance = sum(
                            (imp - mean_importance) ** 2
                            for imp in normalized_importances
                        ) / len(normalized_importances)

                        # Store data for plotting
                        layers_to_show.append(layer_name.replace(".", "_"))
                        specialization_scores.append(variance)

                        # Store for layer-level visualization
                        layer_specialization[layer_name] = {
                            "component": component_name,
                            "specialization": variance,
                            "importances": layer_importances,
                        }

            # Plot specialization scores
            if not layers_to_show:
                print(f"No layers with sufficient variance found for {component_name}")
                continue

            # Sort layers by specialization score
            sorted_indices = np.argsort(specialization_scores)
            sorted_layers = [layers_to_show[i] for i in sorted_indices]
            sorted_scores = [specialization_scores[i] for i in sorted_indices]

            # Plot top 15 layers with highest specialization
            if len(sorted_layers) > 15:
                sorted_layers = sorted_layers[-15:]
                sorted_scores = sorted_scores[-15:]

            # Create bar plot
            plt.barh(sorted_layers, sorted_scores)
            plt.xlabel("Specialization Score (Variance)")
            plt.ylabel("Layer")
            plt.title(f"Parameter Specialization - {component_name} - Agent {agent_id}")
            plt.tight_layout()

            # Save the figure
            plt.savefig(
                vis_dir
                / f"parameter_specialization_{component_name}_agent{agent_id}.png"
            )
            plt.close()

        # Create a more detailed visualization of task-specific parameter usage
        # for the top specialized layers
        if layer_specialization:
            # Sort layers by specialization score
            top_layers = sorted(
                layer_specialization.items(),
                key=lambda x: x[1]["specialization"],
                reverse=True,
            )

            # Show top 5 layers
            top_n = min(5, len(top_layers))

            fig, axes = plt.subplots(top_n, 1, figsize=(12, 3 * top_n))
            if top_n == 1:
                axes = [axes]

            for i, (layer_name, layer_data) in enumerate(top_layers[:top_n]):
                ax = axes[i]

                # Get importances across tasks
                importances = layer_data["importances"]

                # Normalize importances
                total_importance = sum(importances.values())
                if total_importance > 0:
                    normalized_importances = {
                        t: imp / total_importance for t, imp in importances.items()
                    }
                else:
                    normalized_importances = importances

                # Create bar chart
                tasks = list(normalized_importances.keys())
                values = list(normalized_importances.values())

                # Use different colors for each task
                colors = plt.cm.tab10.colors[: len(tasks)]

                ax.bar(range(len(tasks)), values, color=colors)
                ax.set_xticks(range(len(tasks)))
                ax.set_xticklabels([f"Task {t}" for t in tasks])
                ax.set_title(f"{layer_data['component']} - {layer_name}")
                ax.set_ylabel("Normalized Importance")

            plt.tight_layout()
            plt.savefig(vis_dir / f"task_specific_importance_agent{agent_id}.png")
            plt.close()

    if si_agents > 0:
        print(
            f"Generated parameter specialization visualizations for {si_agents} agents in {vis_dir}"
        )
    else:
        print("No agents with SI enabled found.")


def visualize_gradient_masking(agent, output_dir):
    """
    Visualize the effect of gradient masking on parameters.

    This helps understand how gradient masking promotes parameter specialization
    by reducing gradient flow to parameters that are important for other tasks.

    Args:
        agent: A POLARISAgent instance with SI enabled
        output_dir: Directory to save visualizations
    """
    if not hasattr(agent, "use_si") or not agent.use_si:
        print(
            f"Agent {agent.agent_id} does not use Synaptic Intelligence, skipping visualization"
        )
        return

    # Create visualization directory
    vis_dir = Path(output_dir) / "si_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Get SI trackers from the agent
    si_trackers = {"belief": agent.belief_si, "policy": agent.policy_si}

    # For each SI tracker, check if it has stored gradients
    for tracker_name, si_tracker in si_trackers.items():
        if (
            not hasattr(si_tracker, "original_gradients")
            or not si_tracker.original_gradients
        ):
            print(f"No gradient masking data available for {tracker_name}")
            continue

        if (
            not hasattr(si_tracker, "masked_gradients")
            or not si_tracker.masked_gradients
        ):
            print(f"No masked gradients data available for {tracker_name}")
            continue

        # Select a few representative layers
        # Try to find layers with significant masking
        layers_to_visualize = []

        # Calculate masking ratio for each layer
        masking_ratios = {}

        for name in si_tracker.original_gradients:
            if name in si_tracker.masked_gradients:
                orig_grad = si_tracker.original_gradients[name]
                masked_grad = si_tracker.masked_gradients[name]

                # Calculate masking ratio as mean absolute difference / mean absolute original
                orig_abs_mean = torch.mean(torch.abs(orig_grad)).item()
                if orig_abs_mean > 0:
                    diff_abs_mean = torch.mean(
                        torch.abs(orig_grad - masked_grad)
                    ).item()
                    ratio = diff_abs_mean / orig_abs_mean
                    masking_ratios[name] = ratio

        # Sort layers by masking ratio
        if masking_ratios:
            sorted_layers = sorted(
                masking_ratios.items(), key=lambda x: x[1], reverse=True
            )

            # Select top layers for visualization
            for name, ratio in sorted_layers[:3]:
                if ratio > 0.01:  # Only show if there's at least 1% masking
                    layers_to_visualize.append(name)

        if not layers_to_visualize:
            # If no significant masking, just pick some representative layers
            for name in si_tracker.original_gradients:
                if "weight" in name and name in si_tracker.masked_gradients:
                    # Only include main layers
                    param = si_tracker.masked_gradients[name]
                    if (
                        param.numel() > 100
                    ):  # Only include layers with more than 100 parameters
                        layers_to_visualize.append(name)
                        if len(layers_to_visualize) >= 3:
                            break

        # For each layer we want to visualize
        for layer_name in layers_to_visualize:
            if (
                layer_name in si_tracker.original_gradients
                and layer_name in si_tracker.masked_gradients
            ):
                orig_grad = si_tracker.original_gradients[layer_name]
                masked_grad = si_tracker.masked_gradients[layer_name]

                # Get number of dimensions
                ndim = len(orig_grad.shape)

                if ndim == 2:  # 2D tensor - create a heatmap
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                    # Plot original gradient
                    orig_abs = torch.abs(orig_grad).detach().cpu().numpy()
                    im0 = axes[0].imshow(orig_abs, cmap="viridis")
                    axes[0].set_title(f"Original Gradient\n{layer_name}")
                    plt.colorbar(im0, ax=axes[0])

                    # Plot masked gradient
                    masked_abs = torch.abs(masked_grad).detach().cpu().numpy()
                    im1 = axes[1].imshow(masked_abs, cmap="viridis")
                    axes[1].set_title(f"Masked Gradient\n{layer_name}")
                    plt.colorbar(im1, ax=axes[1])

                    # Plot difference
                    diff = orig_abs - masked_abs
                    im2 = axes[2].imshow(diff, cmap="viridis")
                    axes[2].set_title(f"Difference (Removed Gradient)\n{layer_name}")
                    plt.colorbar(im2, ax=axes[2])

                    # Save figure
                    safe_layer_name = layer_name.replace(".", "_")
                    fig.tight_layout()
                    fig.savefig(
                        vis_dir
                        / f"gradient_masking_{tracker_name}_{safe_layer_name}_agent{agent.agent_id}.png"
                    )
                    plt.close(fig)
                else:
                    # Create a histogram of the masking effect
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                    # Flatten gradients
                    orig_flat = orig_grad.flatten().detach().cpu().numpy()
                    masked_flat = masked_grad.flatten().detach().cpu().numpy()
                    diff_flat = orig_flat - masked_flat

                    # Plot histograms
                    axes[0].hist(orig_flat, bins=50, alpha=0.7)
                    axes[0].set_title(f"Original Gradient\n{layer_name}")
                    axes[0].set_xlabel("Gradient Value")
                    axes[0].set_ylabel("Count")

                    axes[1].hist(masked_flat, bins=50, alpha=0.7)
                    axes[1].set_title(f"Masked Gradient\n{layer_name}")
                    axes[1].set_xlabel("Gradient Value")

                    axes[2].hist(diff_flat, bins=50, alpha=0.7)
                    axes[2].set_title(f"Difference (Removed Gradient)\n{layer_name}")
                    axes[2].set_xlabel("Gradient Value")

                    # Save figure
                    safe_layer_name = layer_name.replace(".", "_")
                    fig.tight_layout()
                    fig.savefig(
                        vis_dir
                        / f"gradient_masking_hist_{tracker_name}_{safe_layer_name}_agent{agent.agent_id}.png"
                    )
                    plt.close(fig)

    print(
        f"Generated gradient masking visualizations for agent {agent.agent_id} in {vis_dir}"
    )


def visualize_task_parameter_specialization_matrix(agent, output_dir):
    """
    Visualize a matrix showing how parameters are specialized across different tasks.

    Args:
        agent: A POLARISAgent instance with SI enabled
        output_dir: Directory to save visualizations
    """
    if not hasattr(agent, "use_si") or not agent.use_si:
        print(
            f"Agent {agent.agent_id} does not use Synaptic Intelligence, skipping visualization"
        )
        return

    if (
        not hasattr(agent, "state_belief_si_trackers")
        or not agent.state_belief_si_trackers
    ):
        print(
            f"Agent {agent.agent_id} has no state-specific SI trackers, skipping visualization"
        )
        return

    # Create visualization directory
    vis_dir = Path(output_dir) / "si_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Components to visualize
    components = [
        ("belief", agent.state_belief_si_trackers, agent.belief_processor),
        ("policy", agent.state_policy_si_trackers, agent.policy),
    ]

    # Get the true states we've seen
    true_states = sorted(list(agent.state_belief_si_trackers.keys()))
    num_states = len(true_states)

    if num_states < 2:
        print(
            f"Agent {agent.agent_id} has only seen {num_states} true states, not enough for comparison"
        )
        return

    # For each component (belief/policy)
    for component_name, trackers, model in components:
        # Select top layers by parameter count for analysis
        selected_layers = []
        layer_params = {}

        for name, param in model.named_parameters():
            if param.requires_grad and "weight" in name and param.numel() > 100:
                # Only include main weight layers
                layer_params[name] = param.numel()

        # Sort by parameter count and take top 10
        top_layers = sorted(layer_params.items(), key=lambda x: x[1], reverse=True)[:10]
        selected_layers = [name for name, _ in top_layers]

        if not selected_layers:
            continue

        # Create a specialization matrix: tasks x layers
        specialization_matrix = np.zeros((num_states, len(selected_layers)))

        # For each task, calculate relative importance for each layer
        for i, task_id in enumerate(true_states):
            if task_id not in trackers:
                continue

            task_tracker = trackers[task_id]

            for j, layer_name in enumerate(selected_layers):
                if layer_name in task_tracker.importance_scores:
                    # Calculate mean absolute importance
                    importance = task_tracker.importance_scores[layer_name]
                    mean_importance = torch.mean(torch.abs(importance)).item()
                    specialization_matrix[i, j] = mean_importance

        # Normalize by column (layer) to show relative importance across tasks
        col_sums = specialization_matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1.0  # Avoid division by zero
        norm_matrix = specialization_matrix / col_sums

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            norm_matrix,
            cmap="viridis",
            annot=True,
            fmt=".2f",
            xticklabels=[l.split(".")[-2:][0] for l in selected_layers],
            yticklabels=[f"Task {t}" for t in true_states],
        )
        plt.title(
            f"Parameter Specialization Matrix - {component_name} - Agent {agent.agent_id}"
        )
        plt.xlabel("Layer")
        plt.ylabel("Task")
        plt.tight_layout()

        # Save the figure
        plt.savefig(
            vis_dir
            / f"task_parameter_specialization_matrix_{component_name}_agent{agent.agent_id}.png"
        )
        plt.close()

        # Also create a difference matrix
        if num_states >= 2:
            # Calculate pairwise differences between tasks
            diff_matrix = np.zeros((num_states, num_states, len(selected_layers)))

            for i in range(num_states):
                for j in range(num_states):
                    if i != j:
                        diff_matrix[i, j] = norm_matrix[i] - norm_matrix[j]

            # Calculate task dissimilarity (how different each task's parameter usage is)
            dissimilarity = np.zeros((num_states, num_states))

            for i in range(num_states):
                for j in range(num_states):
                    if i != j:
                        # Use sum of absolute differences
                        dissimilarity[i, j] = np.sum(np.abs(diff_matrix[i, j]))

            # Create heatmap of task dissimilarity
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                dissimilarity,
                cmap="coolwarm",
                annot=True,
                fmt=".2f",
                xticklabels=[f"Task {t}" for t in true_states],
                yticklabels=[f"Task {t}" for t in true_states],
            )
            plt.title(
                f"Task Parameter Usage Dissimilarity - {component_name} - Agent {agent.agent_id}"
            )
            plt.xlabel("Task")
            plt.ylabel("Task")
            plt.tight_layout()

            # Save the figure
            plt.savefig(
                vis_dir
                / f"task_dissimilarity_matrix_{component_name}_agent{agent.agent_id}.png"
            )
            plt.close()

    print(f"Generated parameter specialization matrix for agent {agent.agent_id}")
