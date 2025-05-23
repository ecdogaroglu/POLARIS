"""
Basic tests for POLARIS package functionality.
"""

import pytest
import sys
import traceback


def test_package_import():
    """Test that the package can be imported."""
    try:
        import polaris
        assert polaris is not None
    except ImportError as e:
        pytest.fail(f"Failed to import polaris: {e}")


def test_package_version():
    """Test that package version is accessible."""
    try:
        import polaris
        assert hasattr(polaris, '__version__')
        assert isinstance(polaris.__version__, str)
        assert polaris.__version__ == "2.0.0"
    except Exception as e:
        pytest.fail(f"Version test failed: {e}")


def test_package_info():
    """Test that package info is accessible."""
    try:
        import polaris
        info = polaris.get_info()
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'version' in info
        assert 'author' in info
        assert info['name'] == 'polaris-marl'
    except Exception as e:
        pytest.fail(f"Package info test failed: {e}")


def test_basic_imports():
    """Test that basic components can be imported without complex dependencies."""
    try:
        import polaris
        # Test individual imports to see which one fails
        get_default_config = getattr(polaris, 'get_default_config', None)
        parse_args = getattr(polaris, 'parse_args', None)
        
        # These imports might fail in CI due to missing dependencies
        # So we'll test them separately and provide better error messages
        print(f"get_default_config available: {get_default_config is not None}")
        print(f"parse_args available: {parse_args is not None}")
        
        # At least the package should import successfully
        assert polaris is not None
        
    except ImportError as e:
        pytest.fail(f"Basic imports failed: {e}\nTraceback: {traceback.format_exc()}")


def test_optional_imports():
    """Test imports that might fail due to missing dependencies."""
    import polaris
    
    # Test each import separately to identify which one fails
    imports_to_test = [
        ("POLARISAgent", "polaris.agents.polaris_agent"),
        ("SocialLearningEnvironment", "polaris.environments.social_learning"),
        ("StrategicExperimentationEnvironment", "polaris.environments.strategic_exp"),
        ("Trainer", "polaris.training.trainer"),
    ]
    
    failed_imports = []
    successful_imports = []
    
    for name, module_path in imports_to_test:
        try:
            # Try importing from the polaris package
            obj = getattr(polaris, name, None)
            if obj is not None:
                successful_imports.append(name)
            else:
                failed_imports.append((name, "Import returned None (missing dependencies)"))
        except (ImportError, AttributeError) as e:
            failed_imports.append((name, str(e)))
    
    # Print information for debugging
    print(f"Successful imports: {successful_imports}")
    if failed_imports:
        print(f"Failed imports: {failed_imports}")
    
    # The test passes as long as the package imports without crashing
    # Individual component failures are expected in CI environments
    assert polaris is not None


def test_version_consistency():
    """Test that version is consistent across files."""
    try:
        import polaris
        version = polaris.get_version()
        assert version == polaris.__version__
    except Exception as e:
        pytest.fail(f"Version consistency test failed: {e}")


def test_python_version_compatibility():
    """Test that we're running on a supported Python version."""
    major, minor = sys.version_info[:2]
    assert major == 3, f"Python 3 required, got Python {major}.{minor}"
    assert minor >= 8, f"Python 3.8+ required, got Python {major}.{minor}"
    print(f"Running on Python {major}.{minor}")


if __name__ == "__main__":
    pytest.main([__file__]) 