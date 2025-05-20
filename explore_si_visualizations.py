#!/usr/bin/env python3
"""
Explore Synaptic Intelligence Visualizations.

This script provides a simple browser for exploring the SI visualizations 
generated during training and evaluation of POLARIS agents.
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class SIVisualizationBrowser:
    """Browser for exploring SI visualizations."""
    
    def __init__(self, root, visualization_dir):
        """Initialize the browser."""
        self.root = root
        self.visualization_dir = Path(visualization_dir)
        
        # Set window title
        self.root.title("Synaptic Intelligence Visualization Browser")
        
        # Configure window size
        self.root.geometry("1200x800")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel for visualization list
        self.left_panel = ttk.Frame(self.main_frame, width=300)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create right panel for visualization display
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create visualization list
        self.create_visualization_list()
        
        # Create visualization display
        self.create_visualization_display()
        
        # Find all visualization files
        self.visualization_files = self.find_visualization_files()
        
        # Populate visualization list
        self.populate_visualization_list()
        
        # Current visualization file
        self.current_file = None
        
        # Add resize event handler
        self.root.bind("<Configure>", self.on_resize)
        
        # After canvas is properly set up, show mechanism diagram
        self.root.after(100, self.show_mechanism_diagram)
    
    def create_visualization_list(self):
        """Create the visualization list widget."""
        # Create list frame
        self.list_frame = ttk.Frame(self.left_panel)
        self.list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create filter section
        self.filter_frame = ttk.Frame(self.list_frame)
        self.filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.filter_frame, text="Filter:").pack(side=tk.LEFT)
        
        self.filter_var = tk.StringVar()
        self.filter_entry = ttk.Entry(self.filter_frame, textvariable=self.filter_var)
        self.filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.filter_var.trace_add("write", self.filter_visualizations)
        
        # Create agent filter
        self.agent_frame = ttk.Frame(self.list_frame)
        self.agent_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.agent_frame, text="Agent:").pack(side=tk.LEFT)
        
        self.agent_var = tk.StringVar(value="All")
        self.agent_combo = ttk.Combobox(self.agent_frame, textvariable=self.agent_var,
                                        values=["All", "Agent 0", "Agent 1"])
        self.agent_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.agent_combo.bind("<<ComboboxSelected>>", self.filter_visualizations)
        
        # Create visualization type filter
        self.type_frame = ttk.Frame(self.list_frame)
        self.type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.type_frame, text="Type:").pack(side=tk.LEFT)
        
        self.type_var = tk.StringVar(value="All")
        self.type_combo = ttk.Combobox(self.type_frame, textvariable=self.type_var,
                                      values=["All", "belief", "policy", "1d", "2d"])
        self.type_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.type_combo.bind("<<ComboboxSelected>>", self.filter_visualizations)
        
        # Create visualization list
        self.visualization_list = tk.Listbox(self.list_frame, width=40, height=20)
        self.visualization_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.visualization_list.bind("<<ListboxSelect>>", self.on_visualization_selected)
        
        # Add scrollbar to list
        self.list_scrollbar = ttk.Scrollbar(self.visualization_list)
        self.list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.visualization_list.config(yscrollcommand=self.list_scrollbar.set)
        self.list_scrollbar.config(command=self.visualization_list.yview)
    
    def create_visualization_display(self):
        """Create the visualization display widget."""
        # Create display frame with minimum size
        self.display_frame = ttk.Frame(self.right_panel, width=800, height=600)
        self.display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Make sure the frame keeps its size
        self.display_frame.pack_propagate(False)
        
        # Create canvas for displaying images
        self.canvas = tk.Canvas(self.display_frame, bg="white", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add description label
        self.description_var = tk.StringVar()
        self.description_label = ttk.Label(self.right_panel, textvariable=self.description_var,
                                          wraplength=800, justify=tk.LEFT)
        self.description_label.pack(fill=tk.X, padx=5, pady=5)
    
    def find_visualization_files(self):
        """Find all visualization files in the directory."""
        visualization_files = []
        
        # Look for PNG files
        for file_path in sorted(self.visualization_dir.glob("*.png")):
            visualization_files.append(str(file_path))
        
        return visualization_files
    
    def populate_visualization_list(self):
        """Populate the visualization list with files."""
        self.visualization_list.delete(0, tk.END)
        
        for file_path in self.visualization_files:
            file_name = os.path.basename(file_path)
            self.visualization_list.insert(tk.END, file_name)
    
    def filter_visualizations(self, *args):
        """Filter the visualization list based on the filter text."""
        filter_text = self.filter_var.get().lower()
        agent_filter = self.agent_var.get()
        type_filter = self.type_var.get()
        
        self.visualization_list.delete(0, tk.END)
        
        for file_path in self.visualization_files:
            file_name = os.path.basename(file_path)
            
            # Check if file matches all filters
            if filter_text and filter_text not in file_name.lower():
                continue
            
            if agent_filter != "All":
                agent_num = agent_filter.split()[-1]
                if f"agent{agent_num}" not in file_name:
                    continue
            
            if type_filter != "All":
                if type_filter not in file_name:
                    continue
            
            self.visualization_list.insert(tk.END, file_name)
    
    def on_visualization_selected(self, event):
        """Handle visualization selection."""
        selection = self.visualization_list.curselection()
        
        if not selection:
            return
        
        file_name = self.visualization_list.get(selection[0])
        file_path = self.visualization_dir / file_name
        self.current_file = file_path
        
        self.display_visualization(file_path)
    
    def on_resize(self, event):
        """Handle window resize."""
        # Only respond to actual size changes of the window
        if event.widget == self.root and self.current_file:
            # Slight delay to ensure the canvas has been resized
            self.root.after(100, lambda: self.display_visualization(self.current_file))
    
    def display_visualization(self, file_path):
        """Display a visualization."""
        try:
            # Store current file
            self.current_file = file_path
            
            # Clear canvas
            self.canvas.delete("all")
            
            # Load image
            image = Image.open(file_path)
            
            # Get canvas size (use update_idletasks to ensure size is current)
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Make sure we have valid dimensions
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = self.display_frame.winfo_width()
                canvas_height = self.display_frame.winfo_height()
            
            # Fallback to minimum size if still invalid
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 800
                canvas_height = 600
            
            # Get image size
            image_width, image_height = image.size
            
            # Calculate scale to fit image in canvas
            scale = min(canvas_width / image_width, canvas_height / image_height)
            
            # Calculate new dimensions
            new_width = int(image_width * scale)
            new_height = int(image_height * scale)
            
            # Resize image
            if new_width > 0 and new_height > 0:
                image = image.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert image for Tkinter
                self.tk_image = ImageTk.PhotoImage(image)
                
                # Display image centered in canvas
                self.canvas.create_image(canvas_width // 2, canvas_height // 2, 
                                        image=self.tk_image, anchor=tk.CENTER)
            
            # Set description
            file_name = os.path.basename(file_path)
            
            # Format description
            if "mechanism" in file_name:
                description = "Synaptic Intelligence Mechanism Diagram: Shows how SI tracks parameter importance and prevents catastrophic forgetting"
            elif "1d" in file_name:
                layer_name = file_name.split("_")[-2]
                agent_id = file_name.split("_")[-1].replace("agent", "").replace(".png", "")
                description = f"1D Parameter Importance for {layer_name} (Agent {agent_id}): Bar chart showing absolute importance of individual parameters"
            elif "2d" in file_name:
                layer_name = file_name.split("_")[-2]
                agent_id = file_name.split("_")[-1].replace("agent", "").replace(".png", "")
                description = f"2D Parameter Importance Heatmap for {layer_name} (Agent {agent_id}): Shows importance scores for each parameter in the weight matrix"
            elif "comparison" in file_name:
                description = "Parameter Importance Comparison: Shows the most important parameters for each agent"
            else:
                description = file_name
            
            self.description_var.set(description)
        
        except Exception as e:
            print(f"Error displaying visualization: {e}")
    
    def show_mechanism_diagram(self):
        """Show the mechanism diagram by default."""
        mechanism_file = next((f for f in self.visualization_files if "mechanism" in f), None)
        
        if mechanism_file:
            self.display_visualization(mechanism_file)


def main():
    """Main function to run the browser."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Browse SI Visualizations")
    parser.add_argument("--dir", type=str, default=None,
                        help="Directory containing SI visualizations")
    
    args = parser.parse_args()
    
    # Find visualization directory if not specified
    if args.dir is None:
        results_dir = Path("results")
        vis_dirs = []
        
        # Look for SI visualization directories
        for exp_dir in results_dir.glob("*/eval_*/si_visualizations"):
            vis_dirs.append(exp_dir)
            
        for exp_dir in results_dir.glob("*/*/si_visualizations"):
            vis_dirs.append(exp_dir)
        
        if not vis_dirs:
            print("No SI visualization directories found")
            return
        
        # Use the most recent directory
        args.dir = max(vis_dirs, key=os.path.getmtime)
        print(f"Using most recent visualization directory: {args.dir}")
    
    # Create Tkinter root window
    root = tk.Tk()
    
    # Create browser
    browser = SIVisualizationBrowser(root, args.dir)
    
    # Run main loop
    root.mainloop()


if __name__ == "__main__":
    main() 