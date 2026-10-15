"""
U-Net model for cell segmentation with post-processing and counting.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import ndimage
from skimage import measure, morphology
from skimage.segmentation import watershed
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50


class CellSegmentationTool:
    """
    A comprehensive tool for cell segmentation using a U-Net architecture
    with ResNet50 pretrained encoder.

    Encapsulates model building, training, prediction, evaluation, and
    post-processing with watershed-based cell counting.

    Usage::

        tool = CellSegmentationTool(input_shape=(256, 256, 3))
        tool.build_unet()
        tool.compile_model()
        tool.train_model(X_train, y_train, X_val, y_val, epochs=25)
        metrics = tool.evaluate_model(X_test, y_test)
        mask = tool.predict_segmentation(image)
        count, labels = tool.post_process_and_count(mask)
    """

    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.encoder_layers = []

    def build_unet(self):
        """Build the U-Net architecture with ResNet50 pretrained encoder."""
        inputs = keras.Input(shape=self.input_shape)

        resnet = ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            pooling=None,
        )
        resnet.trainable = True
        s1 = resnet.get_layer("conv1_relu").output
        s2 = resnet.get_layer("conv2_block3_out").output
        s3 = resnet.get_layer("conv3_block4_out").output
        s4 = resnet.get_layer("conv4_block6_out").output
        b5 = resnet.get_layer("conv5_block3_out").output
        self.encoder_layers = resnet.layers

        u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(b5)
        u6 = layers.concatenate([u6, s4])
        c6 = layers.Conv2D(
            512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
        )(u6)
        c6 = layers.Dropout(0.2)(c6)
        c6 = layers.Conv2D(
            512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
        )(c6)

        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c6)
        u7 = layers.concatenate([u7, s3])
        c7 = layers.Conv2D(
            256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
        )(u7)
        c7 = layers.Dropout(0.2)(c7)
        c7 = layers.Conv2D(
            256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
        )(c7)

        u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c7)
        u8 = layers.concatenate([u8, s2])
        c8 = layers.Conv2D(
            128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
        )(u8)
        c8 = layers.Dropout(0.1)(c8)
        c8 = layers.Conv2D(
            128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
        )(c8)

        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c8)
        u9 = layers.concatenate([u9, s1], axis=3)
        c9 = layers.Conv2D(
            64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
        )(u9)
        c9 = layers.Dropout(0.1)(c9)
        c9 = layers.Conv2D(
            64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
        )(c9)

        u10 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c9)
        c10 = layers.Conv2D(
            32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
        )(u10)
        c10 = layers.Dropout(0.1)(c10)
        c10 = layers.Conv2D(
            32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
        )(c10)

        outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c10)

        self.model = keras.Model(inputs=[inputs], outputs=[outputs])
        return self.model

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1e-6):
        """Dice coefficient for segmentation evaluation."""
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
        )

    @staticmethod
    def dice_loss(y_true, y_pred):
        """Dice loss = 1 - Dice coefficient."""
        return 1 - CellSegmentationTool.dice_coef(y_true, y_pred)

    @staticmethod
    def iou_metric(y_true, y_pred):
        """IoU (Jaccard Index) for segmentation evaluation."""
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        smooth = 1e-6
        return (intersection + smooth) / (union + smooth)

    @staticmethod
    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
        """Focal Loss for handling extreme class imbalance in cell segmentation."""
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        bce = tf.keras.losses.binary_crossentropy(y_true_flat, y_pred_flat)
        p_t = y_true_flat * y_pred_flat + (1 - y_true_flat) * (1 - y_pred_flat)
        alpha_factor = y_true_flat * alpha + (1 - y_true_flat) * (1 - alpha)
        focal_weight = alpha_factor * tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(focal_weight * bce)

    @staticmethod
    def combined_loss(y_true, y_pred):
        """Combined BCE + Dice + Focal Loss."""
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        bce = tf.keras.losses.binary_crossentropy(y_true_flat, y_pred_flat)
        dice = CellSegmentationTool.dice_loss(y_true, y_pred)
        focal = CellSegmentationTool.focal_loss(y_true, y_pred)
        return tf.reduce_mean(bce) + dice + focal

    def compile_model(self, learning_rate=1e-4):
        """Compile the U-Net model with Adam optimizer and combined loss."""
        if self.model is None:
            raise ValueError("Model must be built first using build_unet()")
        if tf.config.list_physical_devices("GPU"):
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.combined_loss,
            metrics=[self.dice_coef, self.iou_metric, "binary_accuracy"],
        )
        return self.model

    def train_model(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=50,
        batch_size=8,
        freeze_encoder_epochs=0,
        lr_phase1=None,
        lr_phase2=None,
    ):
        """Train the U-Net model with two-phase training.

        Phase 1: Frozen encoder (if freeze_encoder_epochs > 0).
        Phase 2: Fine-tune entire model.

        Args:
            X_train: Training images.
            y_train: Training masks.
            X_val: Validation images.
            y_val: Validation masks.
            epochs: Total epochs (split across phases).
            batch_size: Batch size.
            freeze_encoder_epochs: Number of epochs to freeze the encoder.
            lr_phase1: Learning rate for phase 1. If None, uses the
                learning rate the model was compiled with.
            lr_phase2: Learning rate for phase 2 (fine-tuning). Defaults
                to 1e-4 when None.

        Returns:
            Training history (combined across phases when two-phase).
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled first")

        if freeze_encoder_epochs > 0:
            if lr_phase1 is not None:
                self.compile_model(learning_rate=lr_phase1)
            self._freeze_encoder()
            print(f"\nPhase 1: Training with frozen encoder for {freeze_encoder_epochs} epochs ...")
            callbacks_phase1 = [
                keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True, monitor="val_loss"
                ),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss"),
                keras.callbacks.ModelCheckpoint(
                    "best_model_phase1.keras",
                    save_best_only=True,
                    monitor="val_loss",
                ),
            ]
            history_phase1 = self.model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                epochs=freeze_encoder_epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks_phase1,
                verbose=1,
            )
            self._unfreeze_encoder()
            self.compile_model(learning_rate=lr_phase2 if lr_phase2 is not None else 1e-4)
            print(
                f"\nPhase 2: Fine-tuning entire model for "
                f"{epochs - freeze_encoder_epochs} epochs ..."
            )
            callbacks_phase2 = [
                keras.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True, monitor="val_loss"
                ),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor="val_loss"),
                keras.callbacks.ModelCheckpoint(
                    "best_model_phase2.keras", save_best_only=True, monitor="val_loss"
                ),
            ]
            remaining_epochs = epochs - freeze_encoder_epochs
            history_phase2 = self.model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                epochs=remaining_epochs,
                initial_epoch=0,
                validation_data=(X_val, y_val),
                callbacks=callbacks_phase2,
                verbose=1,
            )
            self.history = self._merge_histories(history_phase1, history_phase2)
            return self.history
        else:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True, monitor="val_loss"
                ),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor="val_loss"),
                keras.callbacks.ModelCheckpoint(
                    "best_model.keras", save_best_only=True, monitor="val_loss"
                ),
            ]
            self.history = self.model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1,
            )
            return self.history

    def _freeze_encoder(self):
        """Freeze the ResNet50 encoder layers, keep decoder trainable."""
        if self.model is None:
            return
        for layer in self.encoder_layers:
            layer.trainable = False

    def _unfreeze_encoder(self):
        """Unfreeze all layers for fine-tuning."""
        if self.model is None:
            return
        for layer in self.model.layers:
            layer.trainable = True

    @staticmethod
    def _merge_histories(history_phase1, history_phase2):
        """Combine phase 1 and phase 2 training histories into one dict.

        Returns a plain dict (mirroring ``History.history``) with each
        metric's values concatenated in training order.
        """
        keys = set(history_phase1.history) | set(history_phase2.history)
        merged = {}
        for key in keys:
            merged[key] = history_phase1.history.get(key, []) + history_phase2.history.get(key, [])
        return merged

    def load_model(self, filepath):
        """Load a saved .keras model along with the custom losses/metrics.

        Args:
            filepath: Path to a Keras model checkpoint.
        """
        custom_objects = {
            "dice_coef": self.dice_coef,
            "dice_loss": self.dice_loss,
            "iou_metric": self.iou_metric,
            "focal_loss": self.focal_loss,
            "combined_loss": self.combined_loss,
        }
        self.model = keras.models.load_model(filepath, custom_objects=custom_objects)
        self.input_shape = self.model.input_shape[1:]
        print(f"Loaded model from {filepath}")
        return self.model

    def save_checkpoint_to_drive(self, filepath="best_model.keras", drive_path=None):
        """Save model checkpoint to Google Drive for persistence.

        Args:
            filepath: Path to the model file to copy.
            drive_path: Destination path on Google Drive.
        """
        if drive_path is None:
            drive_path = "/content/drive/MyDrive/cell_segmentation/best_model.keras"
        try:
            from google.colab import drive

            drive.mount("/content/drive")
            os.makedirs(os.path.dirname(drive_path), exist_ok=True)
            import shutil

            shutil.copy(filepath, drive_path)
            print(f"Model saved to Google Drive: {drive_path}")
        except ImportError:
            print("Google Colab drive not available. Skipping Drive save.")
        except OSError as e:
            print(f"Failed to save to Drive: {e}")

    def predict_segmentation(self, image):
        """Predict the raw segmentation probability map for a single image."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image, verbose=0)
        return prediction[0, :, :, 0]

    def predict_segmentation_tta(self, image, num_augmentations=4):
        """Predict with test-time augmentation for more robust results.

        Averages predictions across the original image and its flips.

        Args:
            image: Input image, shape (H, W, 3), float32 in [0, 1].
            num_augmentations: Number of augmentations to average.

        Returns:
            Averaged probability map.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        predictions = []
        for i in range(num_augmentations):
            aug_image = image.copy()
            if i % 2 == 1:
                aug_image = np.flip(aug_image, axis=2).copy()
            if i % 4 >= 2:
                aug_image = np.flip(aug_image, axis=1).copy()
            pred = self.model.predict(aug_image, verbose=0)
            predictions.append(pred[0, :, :, 0])

        avg_pred = np.mean(predictions, axis=0)
        return avg_pred

    @staticmethod
    def post_process_and_count(mask, threshold=0.5, min_size=50):
        """Post-process a raw probability mask and return cell count + labels.

        Uses morphological cleaning, distance transform, and watershed
        to separate touching cells.

        Args:
            mask: 2D float array, raw prediction (values in [0, 1]).
            threshold: Binarization threshold.
            min_size: Minimum cell size in pixels (smaller objects removed).

        Returns:
            Tuple of (cell_count, labeled_mask).
        """
        binary_mask = (mask > threshold).astype(np.uint8)
        # remove_small_objects: `min_size` is deprecated since skimage 0.26
        # in favor of `max_size` (which removes objects <= its value).
        # max_size = min_size - 1 keeps objects >= min_size (same behavior).
        try:
            cleared_mask = morphology.remove_small_objects(
                binary_mask.astype(bool), max_size=min_size - 1
            )
        except TypeError:
            cleared_mask = morphology.remove_small_objects(
                binary_mask.astype(bool), min_size=min_size
            )
        distance = ndimage.distance_transform_edt(cleared_mask)
        coords = morphology.local_maxima(distance)
        markers = measure.label(coords)
        labels = watershed(-distance, markers, mask=cleared_mask)
        cell_count = len(np.unique(labels)) - 1
        return cell_count, labels

    @staticmethod
    def optimize_post_processing(mask, true_mask, threshold_grid=None, min_size_grid=None):
        """Grid search over threshold and min_size to optimize cell count accuracy.

        Args:
            mask: Raw prediction probability map(s), shape (H, W) or (N, H, W).
            true_mask: Ground truth binary mask(s), shape (H, W) or (N, H, W).
            threshold_grid: List of thresholds to try.
            min_size_grid: List of min_size values to try.

        Returns:
            Tuple of (best_threshold, best_min_size, best_count, best_iou).
        """
        if threshold_grid is None:
            threshold_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
        if min_size_grid is None:
            min_size_grid = [10, 20, 30, 50, 100]

        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]
            true_mask = true_mask[np.newaxis, ...]

        best_iou = -1
        best_threshold = 0.5
        best_min_size = 50
        best_count = 0

        for thresh in threshold_grid:
            for min_sz in min_size_grid:
                iou_sum = 0
                count_sum = 0
                n = mask.shape[0]
                for i in range(n):
                    count, labels = CellSegmentationTool.post_process_and_count(
                        mask[i], threshold=thresh, min_size=min_sz
                    )
                    pred_binary = (labels > 0).astype(np.uint8)
                    true_binary = (true_mask[i] > 0.5).astype(np.uint8)
                    intersection = np.sum(pred_binary * true_binary)
                    union = np.sum(pred_binary) + np.sum(true_binary) - intersection
                    iou_sum += intersection / (union + 1e-6)
                    count_sum += count
                avg_iou = iou_sum / n
                avg_count = count_sum / n
                if avg_iou > best_iou:
                    best_iou = avg_iou
                    best_threshold = thresh
                    best_min_size = min_sz
                    best_count = avg_count

        return best_threshold, best_min_size, best_count, best_iou

    @staticmethod
    def evaluate_metrics(y_true, y_pred, threshold=0.5):
        """Compute comprehensive evaluation metrics.

        Args:
            y_true: Ground truth binary mask(s), shape (H, W) or (N, H, W).
            y_pred: Predicted probability map(s), shape (H, W) or (N, H, W).
            threshold: Binarization threshold.

        Returns:
            Dict with IoU, Dice, Precision, Recall, F1, Count MAE.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim == 2:
            y_true = y_true[np.newaxis, ...]
            y_pred = y_pred[np.newaxis, ...]
        n = y_true.shape[0]

        iou_sum = 0
        dice_sum = 0
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0
        count_mae_sum = 0

        for i in range(n):
            y_pred_binary = (y_pred[i] > threshold).astype(np.uint8)
            y_true_binary = (y_true[i] > 0.5).astype(np.uint8)

            intersection = np.sum(y_pred_binary * y_true_binary)
            union = np.sum(y_pred_binary) + np.sum(y_true_binary) - intersection
            iou_sum += intersection / (union + 1e-6)

            dice_sum += (2.0 * intersection) / (
                np.sum(y_pred_binary) + np.sum(y_true_binary) + 1e-6
            )

            tp = intersection
            fp = np.sum(y_pred_binary) - tp
            fn = np.sum(y_true_binary) - tp
            p = tp / (tp + fp + 1e-6)
            r = tp / (tp + fn + 1e-6)
            f = 2 * p * r / (p + r + 1e-6) if (p + r) > 0 else 0
            precision_sum += p
            recall_sum += r
            f1_sum += f

            count_pred, _ = CellSegmentationTool.post_process_and_count(
                y_pred[i], threshold=threshold
            )
            count_true, _ = CellSegmentationTool.post_process_and_count(
                y_true[i].astype(float), threshold=0.5
            )
            count_mae_sum += abs(count_pred - count_true)

        return {
            "iou": iou_sum / n,
            "dice": dice_sum / n,
            "precision": precision_sum / n,
            "recall": recall_sum / n,
            "f1": f1_sum / n,
            "count_mae": count_mae_sum / n,
        }

    @staticmethod
    def visualize_results(image, true_mask, raw_prediction, labeled_mask, cell_count):
        """Display a 4-panel comparison: Original, Ground Truth,
        Raw Prediction, and Final Labeled Segmentation."""
        _, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(true_mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")
        axes[2].imshow(raw_prediction, cmap="gray")
        axes[2].set_title("Raw U-Net Prediction")
        axes[2].axis("off")
        colored_labels = plt.cm.nipy_spectral(labeled_mask / (labeled_mask.max() or 1))
        colored_labels[labeled_mask == 0] = 0
        axes[3].imshow(colored_labels)
        axes[3].set_title(f"Segmented Cells  Count: {cell_count}")
        axes[3].axis("off")
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, X_test, y_test, threshold=0.5):
        """Evaluate model performance on the test set with comprehensive metrics.

        Returns:
            Dict with keys 'loss', 'dice_coefficient', 'iou', 'binary_accuracy',
            plus per-image metrics and averages.
        """
        results = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.model.predict(X_test, verbose=0)

        per_image_metrics = []
        for i in range(len(X_test)):
            metrics = self.evaluate_metrics(
                y_test[i, :, :, 0], y_pred[i, :, :, 0], threshold=threshold
            )
            per_image_metrics.append(metrics)

        avg_metrics = {
            "iou": np.mean([m["iou"] for m in per_image_metrics]),
            "dice": np.mean([m["dice"] for m in per_image_metrics]),
            "precision": np.mean([m["precision"] for m in per_image_metrics]),
            "recall": np.mean([m["recall"] for m in per_image_metrics]),
            "f1": np.mean([m["f1"] for m in per_image_metrics]),
            "count_mae": np.mean([m["count_mae"] for m in per_image_metrics]),
        }

        return {
            "loss": results[0],
            "dice_coefficient": results[1],
            "iou": results[2],
            "binary_accuracy": results[3],
            "per_image": per_image_metrics,
            "average": avg_metrics,
        }
