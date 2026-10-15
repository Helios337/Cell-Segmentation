"""
Tests for utilities including YAML config loading.
"""

import numpy as np

from utils import load_config, prepare_dataset


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


def test_config_wires_capacity_and_regularization(tmp_path):
    """decoder_scale and weight_decay from config become CLI defaults."""
    from main import apply_config_defaults, build_parser, load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text("model:\n  decoder_scale: 0.5\n  weight_decay: 0.0001\n")
    parser = build_parser()
    apply_config_defaults(parser, load_config(str(config_file)))
    args = parser.parse_args([])
    assert args.decoder_scale == 0.5
    assert args.weight_decay == 0.0001


def test_prepare_dataset_basic_shapes():
    """A plain dataset yields correctly batched (image, mask) pairs."""
    X = np.zeros((16, 64, 64, 3), dtype=np.float32)
    y = np.zeros((16, 64, 64), dtype=np.float32)
    ds = prepare_dataset(X, y, augment=False, batch_size=4)
    for batch_x, batch_y in ds.take(2):
        assert batch_x.shape == (4, 64, 64, 3)
        assert batch_y.shape == (4, 64, 64, 1)


def test_prepare_dataset_augment_shapes():
    """Augmented dataset keeps image/mask shapes and same dtype."""
    X = np.zeros((16, 64, 64, 3), dtype=np.float32)
    y = np.zeros((16, 64, 64), dtype=np.float32)
    ds = prepare_dataset(X, y, augment=True, batch_size=4)
    for batch_x, batch_y in ds.take(2):
        assert batch_x.shape == (4, 64, 64, 3)
        assert batch_y.shape == (4, 64, 64, 1)
        assert batch_x.dtype == np.float32
        assert batch_y.dtype == np.float32


def test_prepare_dataset_augment_keeps_mask_in_range():
    """Augmented masks remain binary after geometric transforms."""
    X = np.random.rand(8, 32, 32, 3).astype(np.float32)
    y = (np.random.rand(8, 32, 32) > 0.5).astype(np.float32)
    ds = prepare_dataset(X, y, augment=True, batch_size=4)
    for _batch_x, batch_y in ds.take(1):
        unique = np.unique(batch_y.numpy())
        assert set(unique).issubset({0.0, 1.0})
