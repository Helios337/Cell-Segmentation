"""
End-to-end cell segmentation pipeline using U-Net with ResNet50 encoder.

Trains on real fluorescence microscopy data (BBBC038/BBBC039),
evaluates with comprehensive metrics, and visualizes results.

Supported modes:
    train    - two-phase U-Net training (default)
    eval     - evaluate a saved model on the test split
    predict  - segment and count cells in a single image
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from data_handler import BBBCDataLoader
from model import CellSegmentationTool
from utils import load_config, prepare_dataset


def plot_training_history(history):
    """Plot training and validation loss + dice coefficient over epochs.

    Accepts either a Keras ``History`` object or a plain dict (as returned
    by two-phase training).
    """
    hist = history.history if hasattr(history, "history") else history

    _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].plot(hist["loss"], label="Training Loss")
    axes[0].plot(hist["val_loss"], label="Validation Loss")
    axes[0].set_title("Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(hist["dice_coef"], label="Training Dice")
    axes[1].plot(hist["val_dice_coef"], label="Validation Dice")
    axes[1].set_title("Dice Coefficient")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].legend()

    axes[2].plot(hist["iou_metric"], label="Training IoU")
    axes[2].plot(hist["val_iou_metric"], label="Validation IoU")
    axes[2].set_title("IoU")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()


def load_data(args):
    """Load training data based on the specified source."""
    data_loader = BBBCDataLoader(dataset_name=args.data_source)

    if args.data_source == "BBBC038":
        data_loader.download_dataset(save_dir="./data")
        image_dir = "./data"
        X, y = data_loader.load_real_data(image_dir=image_dir, img_size=tuple(args.img_size))
        print(f"Loaded {len(X)} images from BBBC038")
        print(f"  X shape: {X.shape}, y shape: {y.shape}")
        print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"  y nonzero: {(y > 0).sum()} / {y.size}")
        return X, y
    elif args.data_source == "BBBC039":
        data_loader.download_dataset(save_dir="./data")
        image_dir = os.path.join("./data", "images")
        mask_dir = os.path.join("./data", "masks")
        X, y = data_loader.load_real_data(
            image_dir=image_dir, annotation_dir=mask_dir, img_size=tuple(args.img_size)
        )
        print(f"Loaded {len(X)} images from BBBC039")
        return X, y
    else:
        X, y = data_loader.load_synthetic_data(
            n_samples=args.n_samples, img_size=tuple(args.img_size)
        )
        if len(y.shape) == 3:
            y = np.expand_dims(y, axis=-1)
        return X, y


def split_data(X, y, args):
    """Split data into train/validation/test sets using configured ratios."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1.0 - args.train_split, random_state=42
    )
    val_ratio = args.val_split / (args.val_split + args.test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1.0 - val_ratio, random_state=42
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_tool(args):
    """Build and compile the U-Net model with the configured input size."""
    print("[1/6] Building U-Net model with ResNet50 encoder ...")
    cell_tool = CellSegmentationTool(
        input_shape=tuple(args.img_size) + (3,),
    )
    cell_tool.build_unet(decoder_scale=args.decoder_scale)
    cell_tool.compile_model(learning_rate=args.lr_phase1, weight_decay=args.weight_decay)
    cell_tool.model.summary()
    return cell_tool


def print_metrics(metrics, prefix="  "):
    """Pretty-print evaluation metrics returned by evaluate_model."""
    print(f"{prefix}Test Loss:       {metrics['loss']:.4f}")
    print(f"{prefix}Test Dice Coef:  {metrics['dice_coefficient']:.4f}")
    print(f"{prefix}Test IoU:        {metrics['iou']:.4f}")
    print(f"{prefix}Test Accuracy:   {metrics['binary_accuracy']:.4f}")
    print(f"{prefix}Avg Count MAE:   {metrics['average']['count_mae']:.2f}")
    print(f"{prefix}Avg Precision:   {metrics['average']['precision']:.4f}")
    print(f"{prefix}Avg Recall:      {metrics['average']['recall']:.4f}")
    print(f"{prefix}Avg F1:          {metrics['average']['f1']:.4f}")


def run_training(args):
    """Run the full training pipeline."""
    print("=" * 60)
    print("Cell Segmentation Tool - U-Net Training")
    print("=" * 60)

    cell_tool = build_tool(args)

    print("\n[2/6] Loading data ...")
    X, y = load_data(args)
    if len(y.shape) == 3:
        y = np.expand_dims(y, axis=-1)
    print(f"  Total samples: {len(X)}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args)

    if args.augment:
        print("  Augmentation: enabled (on-the-fly, every epoch)")
        X_train = prepare_dataset(X_train, y_train, augment=True, batch_size=args.batch_size)
        y_train = None
    else:
        print("  Augmentation: disabled")

    total_epochs = args.epochs_phase1 + args.epochs_phase2
    print(
        f"\n[3/6] Training U-Net ({total_epochs} epochs total, "
        f"{args.epochs_phase1} frozen + {args.epochs_phase2} fine-tune) ..."
    )
    history = cell_tool.train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=total_epochs,
        batch_size=args.batch_size,
        freeze_encoder_epochs=args.epochs_phase1,
        lr_phase1=args.lr_phase1,
        lr_phase2=args.lr_phase2,
        target_dice=args.target_dice,
    )

    print("\n[4/6] Evaluating on test set ...")
    metrics = cell_tool.evaluate_model(X_test, y_test, threshold=args.threshold)
    print_metrics(metrics)

    print("\n[5/6] Post-processing optimization ...")
    if args.optimize_thresholds:
        test_idx = np.random.randint(0, len(X_test))
        test_image = X_test[test_idx]
        test_mask = y_test[test_idx, :, :, 0]
        raw_pred = cell_tool.predict_segmentation(test_image)
        best_thresh, best_min_size, best_count, best_iou = cell_tool.optimize_post_processing(
            raw_pred, test_mask
        )
        print(f"  Best threshold: {best_thresh}")
        print(f"  Best min_size:  {best_min_size}")
        print(f"  Best IoU:       {best_iou:.4f}")
        print(f"  Best count:     {best_count}")

    print("\n[6/6] Visualizing results ...")
    test_idx = np.random.randint(0, len(X_test))
    test_image = X_test[test_idx]
    test_mask = y_test[test_idx, :, :, 0]

    if args.tta:
        print("  Using test-time augmentation ...")
        raw_prediction = cell_tool.predict_segmentation_tta(test_image)
    else:
        raw_prediction = cell_tool.predict_segmentation(test_image)

    final_count, final_labels = cell_tool.post_process_and_count(
        raw_prediction, threshold=args.threshold, min_size=args.min_size
    )
    cell_tool.visualize_results(test_image, test_mask, raw_prediction, final_labels, final_count)
    print(f"  Predicted cell count: {final_count}")

    print("\n--- Training History ---")
    plot_training_history(history)

    # Save a canonical checkpoint so eval/predict can use a stable default path.
    cell_tool.model.save("best_model.keras")
    print("  Saved final model to best_model.keras")

    if args.save_to_drive:
        cell_tool.save_checkpoint_to_drive(
            filepath="best_model.keras",
            drive_path=args.drive_path,
        )

    print("\nPipeline complete!")
    return cell_tool, metrics


def run_eval(args):
    """Evaluate a saved model on the test split without training."""
    print("=" * 60)
    print("Cell Segmentation Tool - Evaluation")
    print("=" * 60)

    if not os.path.exists(args.model_path):
        sys.exit(f"Model file not found: {args.model_path}")

    cell_tool = CellSegmentationTool(input_shape=tuple(args.img_size) + (3,))
    cell_tool.load_model(args.model_path)
    # Use the loaded model's input size for data, not the CLI default.
    h, w, _ = cell_tool.input_shape
    args.img_size = [h, w]

    print("\nLoading data ...")
    X, y = load_data(args)
    if len(y.shape) == 3:
        y = np.expand_dims(y, axis=-1)

    _, _, X_test, _, _, y_test = split_data(X, y, args)

    print("\nEvaluating on test set ...")
    metrics = cell_tool.evaluate_model(X_test, y_test, threshold=args.threshold)
    print_metrics(metrics)

    return metrics


def run_predict(args):
    """Predict and count cells in a single image using a saved model."""
    print("=" * 60)
    print("Cell Segmentation Tool - Prediction")
    print("=" * 60)

    if not os.path.exists(args.model_path):
        sys.exit(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.image_path):
        sys.exit(f"Image file not found: {args.image_path}")

    cell_tool = CellSegmentationTool(input_shape=tuple(args.img_size) + (3,))
    cell_tool.load_model(args.model_path)
    # Use the loaded model's input size for the image, not the CLI default.
    h, w, _ = cell_tool.input_shape
    args.img_size = [h, w]

    img = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Could not read image: {args.image_path}")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, tuple(args.img_size))
    image = img.astype(np.float32) / 255.0

    if args.tta:
        print("  Using test-time augmentation ...")
        raw_prediction = cell_tool.predict_segmentation_tta(image)
    else:
        raw_prediction = cell_tool.predict_segmentation(image)

    final_count, final_labels = cell_tool.post_process_and_count(
        raw_prediction, threshold=args.threshold, min_size=args.min_size
    )
    print(f"  Predicted cell count: {final_count}")

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image_path))[0]

    mask_path = os.path.join(args.output_dir, f"{base}_mask.png")
    cv2.imwrite(mask_path, (raw_prediction > args.threshold).astype(np.uint8) * 255)
    print(f"  Saved probability mask: {mask_path}")

    labeled_path = os.path.join(args.output_dir, f"{base}_labels.png")
    cv2.imwrite(labeled_path, (final_labels > 0).astype(np.uint8) * 255)
    print(f"  Saved labeled mask: {labeled_path}")

    cell_tool.visualize_results(
        image,
        np.zeros(tuple(args.img_size)),
        raw_prediction,
        final_labels,
        final_count,
    )
    return final_count


def build_parser():
    """Build the CLI argument parser with defaults from config.yaml."""
    parser = argparse.ArgumentParser(description="Cell Segmentation U-Net Training and Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (overridable by CLI args)",
    )
    parser.add_argument(
        "--mode", choices=["train", "eval", "predict"], default="train", help="Run mode"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="BBBC038",
        help="Dataset source (BBBC038, BBBC039, synthetic)",
    )
    parser.add_argument(
        "--img-size", type=int, nargs=2, default=[256, 256], help="Input image size (height width)"
    )
    parser.add_argument("--epochs-phase1", type=int, default=10, help="Epochs with frozen encoder")
    parser.add_argument("--epochs-phase2", type=int, default=20, help="Epochs for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr-phase1", type=float, default=1e-3, help="Learning rate for phase 1")
    parser.add_argument(
        "--lr-phase2", type=float, default=1e-5, help="Learning rate for phase 2 (fine-tuning)"
    )
    parser.add_argument(
        "--decoder-scale",
        type=float,
        default=1.0,
        help="Scale decoder channel counts (e.g. 0.5 halves them) to reduce capacity "
        "and curb overfitting",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam weight decay (L2-style regularization)",
    )
    parser.add_argument(
        "--target-dice",
        type=float,
        default=0.0,
        help="Stop training early once validation Dice reaches this score (0 disables)",
    )
    parser.add_argument(
        "--augment", action="store_true", default=True, help="Enable on-the-fly augmentation"
    )
    parser.add_argument(
        "--no-augment", dest="augment", action="store_false", help="Disable augmentation"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Post-processing threshold")
    parser.add_argument(
        "--min-size", type=int, default=50, help="Minimum cell size for post-processing"
    )
    parser.add_argument(
        "--optimize-thresholds",
        action="store_true",
        default=True,
        help="Optimize threshold and min_size via grid search",
    )
    parser.add_argument(
        "--no-optimize",
        dest="optimize_thresholds",
        action="store_false",
        help="Skip threshold optimization",
    )
    parser.add_argument(
        "--tta", action="store_true", default=True, help="Use test-time augmentation"
    )
    parser.add_argument(
        "--no-tta", dest="tta", action="store_false", help="Disable test-time augmentation"
    )
    parser.add_argument(
        "--save-to-drive", action="store_true", default=False, help="Save model to Google Drive"
    )
    parser.add_argument(
        "--drive-path",
        type=str,
        default="/content/drive/MyDrive/cell_segmentation/best_model.keras",
        help="Google Drive path for model save",
    )
    parser.add_argument(
        "--n-samples", type=int, default=200, help="Number of synthetic samples (for demo mode)"
    )
    parser.add_argument("--train-split", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.15, help="Test split ratio")
    parser.add_argument(
        "--model-path",
        type=str,
        default="best_model.keras",
        help="Path to a saved model (for eval/predict modes)",
    )
    parser.add_argument(
        "--image-path", type=str, default="", help="Path to an input image (for predict mode)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Directory for prediction outputs"
    )
    return parser


def apply_config_defaults(parser, config):
    """Set parser defaults from a config dict where keys are present.

    CLI arguments always take precedence over config values because
    defaults are applied before parsing the command line.
    """
    if not config:
        return

    mapping = {
        "data.source": "data_source",
        "data.img_size": "img_size",
        "data.augment": "augment",
        "data.train_split": "train_split",
        "data.val_split": "val_split",
        "data.test_split": "test_split",
        "training.epochs_phase1": "epochs_phase1",
        "training.epochs_phase2": "epochs_phase2",
        "training.batch_size": "batch_size",
        "training.learning_rate_phase1": "lr_phase1",
        "training.learning_rate_phase2": "lr_phase2",
        "training.target_dice": "target_dice",
        "model.decoder_scale": "decoder_scale",
        "model.weight_decay": "weight_decay",
        "post_processing.threshold": "threshold",
        "post_processing.min_size": "min_size",
        "post_processing.optimize_thresholds": "optimize_thresholds",
        "checkpoint.save_to_drive": "save_to_drive",
        "checkpoint.drive_path": "drive_path",
        "misc.tta": "tta",
    }
    defaults = {}
    for dotted, arg_name in mapping.items():
        section, _, key = dotted.partition(".")
        value = config.get(section, {}).get(key)
        if value is not None:
            defaults[arg_name] = value
    if defaults:
        parser.set_defaults(**defaults)


def main():
    parser = build_parser()
    args, _ = parser.parse_known_args()
    apply_config_defaults(parser, load_config(args.config))
    args = parser.parse_args()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()
