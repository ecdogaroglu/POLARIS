"""
POLARIS Experiment Scripts Access

This module provides information about accessing the specialized experiment scripts
that come with POLARIS for research purposes.
"""

import pkg_resources
from pathlib import Path


def get_experiments_path() -> Path:
    """Get the path to the experiments directory."""
    try:
        # Get the path to the installed experiments directory
        package_path = pkg_resources.resource_filename('polaris_marl', 'experiments')
        return Path(package_path)
    except (ImportError, pkg_resources.DistributionNotFound):
        # Fallback for development installations
        import polaris
        polaris_path = Path(polaris.__file__).parent.parent
        return polaris_path / 'experiments'


def list_experiment_scripts():
    """List available experiment scripts."""
    experiments_path = get_experiments_path()
    
    scripts = {
        'brandl_experiment.py': 'Social learning experiments (Brandl framework)',
        'keller_rady_experiment.py': 'Strategic experimentation (Keller-Rady framework)',
        'keller_rady_sweep.py': 'Multi-agent comparative analysis',
        'run_experiment.py': 'General purpose experiment runner'
    }
    
    print("Available POLARIS experiment scripts:")
    print("=" * 50)
    
    for script, description in scripts.items():
        script_path = experiments_path / script
        status = "✓" if script_path.exists() else "✗"
        print(f"{status} {script}")
        print(f"   {description}")
        if script_path.exists():
            print(f"   Path: {script_path}")
        print()
    
    if experiments_path.exists():
        print(f"Experiments directory: {experiments_path}")
        print("\nAfter pip installation, you can also use these console commands:")
        print("• polaris-brandl       (brandl_experiment.py)")
        print("• polaris-strategic    (keller_rady_experiment.py)")  
        print("• polaris-sweep        (keller_rady_sweep.py)")
        print("• polaris-experiment   (run_experiment.py)")
    else:
        print("⚠️  Experiments directory not found!")
        print("Install POLARIS with: pip install polaris-marl")


def usage_examples():
    """Show usage examples for experiment scripts."""
    examples = """
POLARIS Experiment Scripts Usage Examples:

1. Console Commands (after pip install):
   polaris-brandl --agents 8 --signal-accuracy 0.75 --network-type complete
   polaris-strategic --agents 2 --horizon 10000 --plot-allocations
   polaris-sweep --agent-counts 2 3 4 5 --horizon 400
   polaris-experiment --env brandl --agents 4 --use-gnn

2. Direct Python Execution:
   python -m polaris.experiments.brandl_experiment --agents 8 --plot-states
   python -m polaris.experiments.keller_rady_experiment --agents 2 --latex-style
   
3. From Experiments Directory:
   cd $(python -c "import polaris.experiments; print(polaris.experiments.get_experiments_path())")
   python brandl_experiment.py --agents 8 --plot-states
   python keller_rady_experiment.py --agents 2 --latex-style

4. Research Workflow:
   # Train and analyze social learning
   polaris-brandl --agents 8 --signal-accuracy 0.75 --plot-states --latex-style
   
   # Run strategic experimentation
   polaris-strategic --agents 2 --horizon 10000 --plot-allocations --save-model
   
   # Comparative analysis across agent counts  
   polaris-sweep --agent-counts 2 3 4 5 6 7 8 --horizon 400 --episodes 3
"""
    print(examples)


if __name__ == "__main__":
    list_experiment_scripts()
    print("\n" + "=" * 50 + "\n")
    usage_examples() 