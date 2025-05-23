#!/usr/bin/env python3
"""
Build and publish script for POLARIS package.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and handle errors."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result


def clean_build():
    """Clean build artifacts."""
    print("🧹 Cleaning build artifacts...")
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    for pattern in dirs_to_clean:
        run_command(f"rm -rf {pattern}", check=False)


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    result = run_command("python -m pytest tests/ -v", check=False)
    if result.returncode != 0:
        print("❌ Tests failed!")
        return False
    print("✅ Tests passed!")
    return True


def check_code_quality():
    """Run code quality checks."""
    print("🔍 Running code quality checks...")
    
    # Run black
    print("Running black...")
    result = run_command("python -m black --check polaris/", check=False)
    if result.returncode != 0:
        print("⚠️ Code formatting issues found. Run 'black polaris/' to fix.")
        return False
    
    # Run isort
    print("Running isort...")
    result = run_command("python -m isort --check-only polaris/", check=False)
    if result.returncode != 0:
        print("⚠️ Import sorting issues found. Run 'isort polaris/' to fix.")
        return False
    
    # Run flake8
    print("Running flake8...")
    result = run_command("python -m flake8 polaris/", check=False)
    if result.returncode != 0:
        print("⚠️ Linting issues found.")
        return False
    
    print("✅ Code quality checks passed!")
    return True


def build_package():
    """Build the package."""
    print("📦 Building package...")
    
    # Build source distribution and wheel
    result = run_command("python -m build")
    if result.returncode != 0:
        print("❌ Package build failed!")
        return False
        
    print("✅ Package built successfully!")
    return True


def check_package():
    """Check the package with twine."""
    print("🔍 Checking package...")
    result = run_command("python -m twine check dist/*")
    if result.returncode != 0:
        print("❌ Package check failed!")
        return False
    print("✅ Package check passed!")
    return True


def publish_test():
    """Publish to test PyPI."""
    print("📤 Publishing to test PyPI...")
    result = run_command("python -m twine upload --repository testpypi dist/*")
    if result.returncode != 0:
        print("❌ Test PyPI upload failed!")
        return False
    print("✅ Published to test PyPI!")
    return True


def publish_prod():
    """Publish to PyPI."""
    print("📤 Publishing to PyPI...")
    result = run_command("python -m twine upload dist/*")
    if result.returncode != 0:
        print("❌ PyPI upload failed!")
        return False
    print("✅ Published to PyPI!")
    return True


def main():
    """Main build and publish workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build and publish POLARIS package")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-quality", action="store_true", help="Skip code quality checks")
    parser.add_argument("--test-pypi", action="store_true", help="Publish to test PyPI")
    parser.add_argument("--prod-pypi", action="store_true", help="Publish to production PyPI")
    parser.add_argument("--clean-only", action="store_true", help="Only clean build artifacts")
    
    args = parser.parse_args()
    
    # Change to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    # Clean build artifacts
    clean_build()
    
    if args.clean_only:
        return
    
    # Run tests
    if not args.skip_tests:
        if not run_tests():
            sys.exit(1)
    
    # Check code quality
    if not args.skip_quality:
        if not check_code_quality():
            sys.exit(1)
    
    # Build package
    if not build_package():
        sys.exit(1)
    
    # Check package
    if not check_package():
        sys.exit(1)
    
    # Publish if requested
    if args.test_pypi:
        if not publish_test():
            sys.exit(1)
    elif args.prod_pypi:
        response = input("⚠️ This will publish to production PyPI. Are you sure? (y/N): ")
        if response.lower() == 'y':
            if not publish_prod():
                sys.exit(1)
        else:
            print("🚫 Publication cancelled.")
    
    print("🎉 Build process completed successfully!")


if __name__ == "__main__":
    main() 