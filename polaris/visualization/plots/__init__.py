"""
Base plotting infrastructure for POLARIS visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from ..styles.latex import format_axis_in_latex_style, save_figure_for_publication


class BasePlotter(ABC):
    """
    Abstract base class for all POLARIS plotters.
    
    Provides common functionality and interface for specialized plotting classes.
    """
    
    def __init__(self, use_latex: bool = False):
        """
        Initialize the base plotter.
        
        Args:
            use_latex: Whether to use LaTeX styling for plots
        """
        self.use_latex = use_latex
        
        # Set up figure defaults
        self.fig_width = 8  # Width suitable for single-column journal format
        self.fig_height = self.fig_width * 0.618  # Golden ratio
        
    def create_figure(self, figsize: Optional[tuple] = None) -> tuple:
        """
        Create a figure with consistent styling.
        
        Args:
            figsize: Optional figure size (width, height)
            
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        if figsize is None:
            figsize = (self.fig_width, self.fig_height)
            
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        
        if self.use_latex:
            format_axis_in_latex_style(ax)
            
        return fig, ax
    
    def save_figure(self, fig, save_path: Path, title: str = ""):
        """
        Save figure with appropriate format and styling.
        
        Args:
            fig: Matplotlib figure object
            save_path: Path to save the figure (should include .png extension)
            title: Optional title for the figure
        """
        if self.use_latex:
            # Only save PNG format, no PDF
            save_figure_for_publication(fig, save_path, formats=['png'])
        else:
            # Ensure we have the right extension
            if not str(save_path).endswith('.png'):
                save_path = save_path.with_suffix('.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)
        print(f"  Saved: {save_path}")
    
    def get_colors(self) -> list:
        """Get color cycle for consistent styling."""
        return plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    def validate_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Validate that required metrics are present.
        
        Args:
            metrics: Metrics dictionary to validate
            
        Returns:
            bool: True if metrics are valid for this plotter
        """
        return metrics is not None and len(metrics) > 0
    
    @abstractmethod
    def plot(self, metrics: Dict[str, Any], env, args, output_dir: Path, **kwargs):
        """
        Generate plots for this plotter type.
        
        Args:
            metrics: Experiment metrics
            env: Environment object
            args: Command-line arguments
            output_dir: Directory to save plots
            **kwargs: Additional keyword arguments
        """
        pass
    
    def create_empty_plot(self, output_dir: Path, filename: str, message: str = "No data available"):
        """
        Create an empty plot with a message when no data is available.
        
        Args:
            output_dir: Directory to save the plot
            filename: Name of the plot file
            message: Message to display
        """
        fig, ax = self.create_figure()
        ax.text(0.5, 0.5, message, 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        save_path = output_dir / filename
        self.save_figure(fig, save_path)


class MultiAgentPlotter(BasePlotter):
    """
    Base class for plotters that handle multi-agent data.
    
    Provides utilities for iterating over agents and consistent agent coloring.
    """
    
    def get_agent_color(self, agent_id: int, colors: list) -> str:
        """Get consistent color for an agent."""
        return colors[agent_id % len(colors)]
    
    def get_agent_ids(self, metrics: Dict[str, Any]) -> list:
        """Extract agent IDs from metrics."""
        # Try different possible locations for agent data
        for key in ['action_probs', 'agent_actions', 'allocations']:
            if key in metrics and isinstance(metrics[key], dict):
                return sorted(metrics[key].keys(), key=int)
        
        # Fallback to num_agents if available
        if 'num_agents' in metrics:
            return list(range(metrics['num_agents']))
            
        return []
