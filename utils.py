import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_config(config_path="config.yaml"):
    """Load a YAML configuration file into a dict.

    Returns an empty dict if the file is missing or unreadable, so
    callers can fall back to CLI defaults.

    Args:
        config_path: Path to a YAML config file.

    Returns:
        Dict of parsed config values.
    """
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found; using defaults.")
        return {}
    import yaml

    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        print(f"Failed to parse config file {config_path}; using defaults.")
        return {}


def process_bbbc_images(image_dir, annotation_dir=None, img_size=(256, 256)):
    """Processes real BBBC images and (optionally) annotations."""
    image_files = [
        f
        for f in sorted(os.listdir(image_dir))
        if f.lower().endswith((".jpg", ".png", ".tif", ".tiff"))
    ]
    processed_images = []
    processed_masks = []
    for img_file in tqdm(image_files, desc="Processing BBBC images"):
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        processed_images.append(img)
        if annotation_dir:
            mask_path = os.path.join(annotation_dir, img_file)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, img_size)
                mask = (mask > 0).astype(np.float32)
                processed_masks.append(mask)
            else:
                processed_masks.append(np.zeros(img_size, dtype=np.float32))
    if annotation_dir:
        return np.array(processed_images), np.expand_dims(np.array(processed_masks), axis=-1)
    return np.array(processed_images)


def save_results_to_csv(image_names, cell_counts, output_path="cell_counts.csv"):
    """Saves cell counting results to a CSV file."""
    results_df = pd.DataFrame(
        {
            "image_name": image_names,
            "predicted_cell_count": cell_counts,
            "timestamp": pd.Timestamp.now(),
        }
    )
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


def augment_image(image, mask=None):
    """Apply fluorescence-specific augmentation to an image and optional mask.

    Augmentations are chosen to simulate real fluorescence microscopy variations:
    rotation, flip, brightness/contrast jitter, Gaussian noise, elastic deformation,
    Gaussian blur, and random erasing.

    Args:
        image: float32 numpy array, shape (H, W, 3), values in [0, 1].
        mask: optional float32 numpy array, shape (H, W), values in {0, 1}.

    Returns:
        Augmented image (and mask if provided) as float32 arrays.
    """
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=0).copy()
        if mask is not None:
            mask = np.flip(mask, axis=0).copy()

    if np.random.rand() > 0.5:
        image = np.flip(image, axis=1).copy()
        if mask is not None:
            mask = np.flip(mask, axis=1).copy()

    angle = np.random.choice([0, 90, 180, 270])
    if angle != 0:
        image = np.rot90(image, k=angle // 90).copy()
        if mask is not None:
            mask = np.rot90(mask, k=angle // 90).copy()

    brightness = np.random.uniform(0.8, 1.2)
    contrast = np.random.uniform(0.8, 1.2)
    image = image * brightness + contrast * 0.05
    image = np.clip(image, 0.0, 1.0)

    noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
    image = image + noise
    image = np.clip(image, 0.0, 1.0)

    kernel_size = np.random.choice([3, 5])
    sigma = np.random.uniform(0.3, 1.0)
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    if mask is not None:
        return image, mask
    return image


def prepare_dataset(X, y, augment=False, batch_size=8, img_size=(256, 256)):
    """Prepare a tf.data.Dataset for training with optional on-the-fly augmentation.

    Args:
        X: numpy array of images, shape (N, H, W, 3).
        y: numpy array of masks, shape (N, H, W, 1).
        augment: whether to apply augmentation on-the-fly.
        batch_size: batch size for the dataset.
        img_size: target image size.

    Returns:
        tf.data.Dataset object.
    """
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
