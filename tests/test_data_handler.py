"""
Tests for BBBC data loader and synthetic data generation.
"""

import numpy as np

from data_handler import BBBCDataLoader


def test_synthetic_data_shape():
    """Synthetic data honors the requested image size and mask channels."""
    loader = BBBCDataLoader()
    X, y = loader.load_synthetic_data(n_samples=8, img_size=(64, 64))
    assert X.shape == (8, 64, 64, 3)
    assert y.shape == (8, 64, 64)
    assert X.dtype == np.float32
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_synthetic_data_has_cells():
    """Synthetic masks should contain at least some foreground pixels."""
    loader = BBBCDataLoader()
    _X, y = loader.load_synthetic_data(n_samples=3, img_size=(128, 128))
    assert (y > 0).any(), "Synthetic masks should contain foreground cells"


def test_unknown_dataset_download_is_safe():
    """download_dataset on an unknown dataset should not raise."""
    loader = BBBCDataLoader(dataset_name="NOPE")
    loader.download_dataset(save_dir="./data")
    assert True
