"""
Basic tests for POLARIS package functionality.
"""

import pytest
import polaris


def test_package_version():
    """Test that package version is accessible."""
    assert hasattr(polaris, '__version__')
    assert isinstance(polaris.__version__, str)
    assert polaris.__version__ == "2.0.0"


def test_package_info():
    """Test that package info is accessible."""
    info = polaris.get_info()
    assert isinstance(info, dict)
    assert 'name' in info
    assert 'version' in info
    assert 'author' in info
    assert info['name'] == 'polaris-marl'


def test_main_imports():
    """Test that main components can be imported."""
    from polaris import (
        POLARISAgent,
        SocialLearningEnvironment,
        StrategicExperimentationEnvironment,
        Trainer,
        parse_args,
        get_default_config
    )
    
    # Check that these are not None
    assert POLARISAgent is not None
    assert SocialLearningEnvironment is not None
    assert StrategicExperimentationEnvironment is not None
    assert Trainer is not None
    assert parse_args is not None
    assert get_default_config is not None


def test_version_consistency():
    """Test that version is consistent across files."""
    version = polaris.get_version()
    assert version == polaris.__version__


if __name__ == "__main__":
    pytest.main([__file__]) 