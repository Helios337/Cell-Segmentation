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


def test_decoder_scale_reduces_params():
    """A smaller decoder scale must produce a smaller model, same output."""
    full = CellSegmentationTool(input_shape=(128, 128, 3)).build_unet(decoder_scale=1.0)
    slim = CellSegmentationTool(input_shape=(128, 128, 3)).build_unet(decoder_scale=0.5)
    assert full.output_shape == slim.output_shape == (None, 128, 128, 1)
    assert slim.count_params() < full.count_params()
    # the ResNet50 encoder dominates, so expect a clear but not extreme cut
    assert slim.count_params() < full.count_params() * 0.8


def test_compile_model_weight_decay():
    """Weight decay is applied to the Adam optimizer."""
    tool = CellSegmentationTool(input_shape=(128, 128, 3))
    tool.build_unet()
    tool.compile_model(weight_decay=1e-4)
    assert tool.model.optimizer.weight_decay == 1e-4


def test_target_score_stopping():
    """TargetScoreStopping halts training only once the score is reached."""
    from model import TargetScoreStopping

    class FakeModel:
        stop_training = False

    model = FakeModel()
    cb = TargetScoreStopping(monitor="val_dice_coef", target=0.75)
    cb.set_model(model)

    cb.on_epoch_end(1, {"val_dice_coef": 0.60})
    assert model.stop_training is False

    cb.on_epoch_end(2, {"val_dice_coef": 0.80})
    assert model.stop_training is True


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


def test_evaluate_metrics_2d_counts_whole_image():
    """2D input must be treated as ONE image, not per-row.

    Regression test: np.atleast_3d turns (H, W) into (H, W, 1), so the
    metric loop previously iterated over rows instead of images, producing
    meaningless precision/recall/count values.
    """
    from skimage.draw import disk

    mask = np.zeros((128, 128), dtype=np.float32)
    for center in [(30, 30), (70, 30), (50, 70)]:
        rr, cc = disk(center, 15)
        mask[rr, cc] = 1.0

    # model predicts nothing: count_pred = 0, truth has 3 cells -> MAE = 3
    zero_pred = np.zeros((128, 128), dtype=np.float32)
    metrics = CellSegmentationTool.evaluate_metrics(mask, zero_pred, threshold=0.5)
    assert metrics["count_mae"] == pytest.approx(3.0)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0

    # perfect prediction: metrics should be ~perfect for the whole image
    perfect = CellSegmentationTool.evaluate_metrics(mask, mask, threshold=0.5)
    assert perfect["iou"] > 0.9
    assert perfect["count_mae"] == pytest.approx(0.0)


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
