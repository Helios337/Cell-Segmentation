"""
Tests for cell segmentation U-Net model.
"""

import numpy as np
import pytest

from model import CellSegmentationTool


def test_model_build():
    """Test that the U-Net model with ResNet50 encoder builds correctly."""
    tool = CellSegmentationTool(input_shape=(128, 128, 3))
    model = tool.build_unet()
    assert model is not None
    assert model.output_shape == (None, 128, 128, 1)


def test_dice_coef():
    """Test dice coefficient with perfect, partial, and no overlap."""
    tool = CellSegmentationTool()
    perfect = tool.dice_coef(
        np.ones((2, 128, 128, 1)),
        np.ones((2, 128, 128, 1)),
    ).numpy()
    assert perfect == pytest.approx(1.0, abs=1e-5), f"Perfect overlap should be 1.0, got {perfect}"

    no_overlap = tool.dice_coef(
        np.ones((2, 128, 128, 1)),
        np.zeros((2, 128, 128, 1)),
    ).numpy()
    assert no_overlap == pytest.approx(0.0, abs=1e-5), f"No overlap should be 0.0, got {no_overlap}"

    half = tool.dice_coef(
        np.ones((2, 128, 128, 1)),
        np.concatenate([np.ones((1, 128, 128, 1)), np.zeros((1, 128, 128, 1))], axis=0),
    ).numpy()
    assert half == pytest.approx(2 / 3, abs=1e-4), f"Half overlap should be 2/3, got {half}"


def test_dice_loss():
    """Test that dice loss = 1 - dice coefficient."""
    tool = CellSegmentationTool()
    y_true = np.ones((2, 128, 128, 1))
    y_pred = np.ones((2, 128, 128, 1))
    loss = tool.dice_loss(y_true, y_pred).numpy()
    assert loss == pytest.approx(0.0, abs=1e-5), f"Perfect overlap loss should be 0.0, got {loss}"


def test_focal_loss():
    """Test that focal loss runs without error and produces a scalar."""
    tool = CellSegmentationTool()
    y_true = np.ones((2, 128, 128, 1))
    y_pred = np.full((2, 128, 128, 1), 0.7)
    loss = tool.focal_loss(y_true, y_pred).numpy()
    assert np.isscalar(loss) or loss.ndim == 0, "Loss must be a scalar"
    assert loss > 0, f"Loss should be positive, got {loss}"


def test_combined_loss():
    """Test that combined loss runs without error and produces a scalar."""
    tool = CellSegmentationTool()
    y_true = np.ones((2, 128, 128, 1))
    y_pred = np.full((2, 128, 128, 1), 0.7)
    loss = tool.combined_loss(y_true, y_pred).numpy()
    assert np.isscalar(loss) or loss.ndim == 0, "Loss must be a scalar"
    assert loss > 0, f"Loss should be positive, got {loss}"


def test_iou_metric():
    """Test IoU metric with perfect and no overlap."""
    tool = CellSegmentationTool()
    perfect = tool.iou_metric(
        np.ones((2, 128, 128, 1)),
        np.ones((2, 128, 128, 1)),
    ).numpy()
    assert perfect == pytest.approx(1.0, abs=1e-5)

    no_overlap = tool.iou_metric(
        np.ones((2, 128, 128, 1)),
        np.zeros((2, 128, 128, 1)),
    ).numpy()
    assert no_overlap == pytest.approx(0.0, abs=1e-5)


def test_post_process_and_count():
    """Test post-processing with a known binary mask."""
    mask = np.zeros((100, 100), dtype=np.float32)
    from skimage.draw import disk

    rr, cc = disk((30, 30), 15)
    mask[rr, cc] = 1.0
    rr2, cc2 = disk((70, 70), 12)
    mask[rr2, cc2] = 1.0

    count, labels = CellSegmentationTool.post_process_and_count(mask, threshold=0.5, min_size=10)
    assert count >= 2, f"Expected at least 2 cells, got {count}"
    assert labels.shape == mask.shape, f"Label shape {labels.shape} != mask shape {mask.shape}"


def test_compile_model():
    """Test model compilation."""
    tool = CellSegmentationTool(input_shape=(128, 128, 3))
    tool.build_unet()
    model = tool.compile_model()
    assert model.optimizer is not None
    assert model.loss is not None


def test_predict_without_training():
    """Test that predict returns expected shape even without training."""
    tool = CellSegmentationTool()
    tool.build_unet()
    result = tool.predict_segmentation(np.zeros((256, 256, 3)))
    assert result.shape == (256, 256), f"Expected (256, 256), got {result.shape}"
    assert result.dtype == np.float32


def test_predict_tta():
    """Test test-time augmentation prediction."""
    tool = CellSegmentationTool()
    tool.build_unet()
    result = tool.predict_segmentation_tta(np.zeros((256, 256, 3)), num_augmentations=2)
    assert result.shape == (256, 256), f"Expected (256, 256), got {result.shape}"
    assert result.dtype == np.float32


def test_evaluate_metrics():
    """Test comprehensive evaluation metrics."""
    y_true = np.ones((64, 64), dtype=np.float32)
    y_pred = np.ones((64, 64), dtype=np.float32) * 0.9
    metrics = CellSegmentationTool.evaluate_metrics(y_true, y_pred, threshold=0.5)
    assert "iou" in metrics
    assert "dice" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "count_mae" in metrics
    assert metrics["iou"] > 0.9


def test_optimize_post_processing():
    """Test post-processing grid search optimization."""
    mask = np.zeros((100, 100), dtype=np.float32)
    from skimage.draw import disk

    rr, cc = disk((30, 30), 15)
    mask[rr, cc] = 1.0
    rr2, cc2 = disk((70, 70), 12)
    mask[rr2, cc2] = 1.0

    _best_thresh, _best_min_size, best_count, best_iou = (
        CellSegmentationTool.optimize_post_processing(mask, mask)
    )
    assert best_iou >= 0, f"IoU should be non-negative, got {best_iou}"
    assert best_count >= 2, f"Expected at least 2 cells, got {best_count}"
