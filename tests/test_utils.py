"""
Tests for utilities including YAML config loading.
"""

from utils import load_config


def test_load_config_parses_yaml(tmp_path):
    """A valid YAML config parses into a nested dict."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "training:\n  epochs_phase1: 3\n  batch_size: 2\ndata:\n  source: synthetic\n"
    )
    config = load_config(str(config_file))
    assert config["training"]["epochs_phase1"] == 3
    assert config["training"]["batch_size"] == 2
    assert config["data"]["source"] == "synthetic"


def test_load_config_missing_file_returns_empty():
    """A missing config file returns an empty dict without raising."""
    assert load_config("does_not_exist.yaml") == {}


def test_load_config_invalid_yaml_returns_empty(tmp_path):
    """Malformed YAML returns an empty dict without raising."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(":\n  - not: [valid")
    assert load_config(str(config_file)) == {}
