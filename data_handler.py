"""
Data loader for BBBC datasets and synthetic cell image generation.
"""

import os
import zipfile

import cv2
import numpy as np
import requests
from tqdm import tqdm

BBBC_URLS = {
    "BBBC005": "https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip",
    "BBBC038": "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip",
    "BBBC039": "https://data.broadinstitute.org/bbbc/BBBC039/images.zip",
}

BBBC_MASK_URLS = {
    "BBBC039": "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip",
}


class BBBCDataLoader:
    """
    Handles loading of BBBC biomedical datasets and synthetic data generation.

    Usage::

        loader = BBBCDataLoader(dataset_name="BBBC038")
        loader.download_dataset(save_dir="./data")
        X, y = loader.load_real_data(image_dir="./data")
    """

    def __init__(self, dataset_name="BBBC038"):
        self.dataset_name = dataset_name

    def download_dataset(self, save_dir="./data"):
        """Download a BBBC dataset zip file from the Broad Institute."""
        os.makedirs(save_dir, exist_ok=True)

        if self.dataset_name not in BBBC_URLS:
            print(f"Unknown dataset: {self.dataset_name}. Available: {list(BBBC_URLS.keys())}")
            return

        zip_url = BBBC_URLS[self.dataset_name]
        zip_filename = os.path.basename(zip_url)
        zip_path = os.path.join(save_dir, zip_filename)

        if os.path.exists(zip_path):
            print(f"{zip_filename} already downloaded at {zip_path}.")
        else:
            print(f"Downloading {self.dataset_name} from {zip_url} ...")
            try:
                response = requests.get(zip_url, stream=True, timeout=120)
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                with (
                    tqdm(
                        total=total_size, unit="B", unit_scale=True, desc=self.dataset_name
                    ) as pbar,
                    open(zip_path, "wb") as f,
                ):
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                print("Download complete.")
            except requests.exceptions.RequestException as e:
                print(f"Download failed: {e}")
                return

        print("Extracting files ...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(save_dir)
        print(f"Extraction complete. Files in: {save_dir}")

        if self.dataset_name == "BBBC039":
            mask_url = BBBC_MASK_URLS.get("BBBC039")
            if mask_url:
                mask_filename = os.path.basename(mask_url)
                mask_path = os.path.join(save_dir, mask_filename)
                if not os.path.exists(mask_path):
                    print(f"Downloading masks for {self.dataset_name} ...")
                    try:
                        response = requests.get(mask_url, stream=True, timeout=120)
                        response.raise_for_status()
                        total_size = int(response.headers.get("content-length", 0))
                        with (
                            tqdm(total=total_size, unit="B", unit_scale=True, desc="Masks") as pbar,
                            open(mask_path, "wb") as f,
                        ):
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                                pbar.update(len(chunk))
                        print("Masks download complete.")
                    except requests.exceptions.RequestException as e:
                        print(f"Masks download failed: {e}")
                if os.path.exists(mask_path):
                    print("Extracting masks ...")
                    with zipfile.ZipFile(mask_path, "r") as zip_ref:
                        zip_ref.extractall(save_dir)
                    print("Masks extraction complete.")

    def load_real_data(self, image_dir, annotation_dir=None, img_size=(256, 256)):
        """Load real BBBC images and optional annotations.

        Handles both flat directory structures (BBBC039) and
        BBBC038-style nested folders (each image in its own folder
        with images/ and masks/ subfolders).

        Args:
            image_dir: Path to directory containing microscopy images.
            annotation_dir: Path to directory containing ground-truth masks (optional).
            img_size: Target image size (height, width).

        Returns:
            Tuple of (images, masks) as float32 numpy arrays normalized to [0, 1].
        """
        image_files = []
        for root, dirs, files in os.walk(image_dir):
            if "masks" in root.split(os.sep):
                continue
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".tif", ".tiff")):
                    image_files.append(os.path.join(root, f))
        image_files = sorted(image_files)

        images = []
        masks = []
        masks_found = 0
        for img_path in tqdm(image_files, desc="Loading real images"):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            mask = np.zeros(img_size, dtype=np.float32)
            sample_dir = os.path.dirname(os.path.dirname(img_path))
            masks_dir = os.path.join(sample_dir, "masks")
            if os.path.isdir(masks_dir):
                mask_files = [
                    f
                    for f in os.listdir(masks_dir)
                    if f.lower().endswith((".png", ".jpg", ".tif", ".tiff"))
                ]
                for mf in mask_files:
                    mp = os.path.join(masks_dir, mf)
                    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        m = cv2.resize(m, img_size)
                        mask = np.maximum(mask, (m > 0).astype(np.float32))
                if mask.sum() > 0:
                    masks_found += 1
            elif annotation_dir:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = os.path.join(annotation_dir, base_name + ".png")
                if os.path.exists(mask_path):
                    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        m = cv2.resize(m, img_size)
                        mask = (m > 0).astype(np.float32)
                else:
                    mask_files = [
                        f
                        for f in os.listdir(annotation_dir)
                        if f.lower().endswith((".png", ".jpg", ".tif", ".tiff"))
                    ]
                    for mf in mask_files:
                        mp = os.path.join(annotation_dir, mf)
                        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                        if m is not None:
                            m = cv2.resize(m, img_size)
                            mask = np.maximum(mask, (m > 0).astype(np.float32))
            masks.append(mask)
        print(f"  Masks found with nonzero pixels: {masks_found} / {len(images)}")
        return np.array(images), np.expand_dims(np.array(masks), axis=-1)

    def load_synthetic_data(self, n_samples=100, img_size=(256, 256)):
        """Generate synthetic cell-like images for demonstration."""
        print(f"Generating {n_samples} synthetic cell images ...")
        X, y = [], []
        for _ in tqdm(range(n_samples), desc="Synthetic data"):
            img = np.zeros((*img_size, 3), dtype=np.uint8)
            mask = np.zeros(img_size, dtype=np.uint8)
            n_cells = np.random.randint(5, 25)
            for _ in range(n_cells):
                cx = np.random.randint(20, img_size[0] - 20)
                cy = np.random.randint(20, img_size[1] - 20)
                radius = np.random.randint(10, 25)
                brightness = np.random.randint(100, 200)
                highlight = np.random.randint(brightness + 10, 255)
                cv2.circle(img, (cx, cy), radius, (brightness,) * 3, -1)
                cv2.circle(img, (cx, cy), int(radius * 0.6), (highlight,) * 3, -1)
                cv2.circle(mask, (cx, cy), radius, 255, -1)
            noise = np.random.normal(0, 15, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            X.append(img.astype(np.float32) / 255.0)
            y.append(mask.astype(np.float32) / 255.0)
        return np.array(X), np.array(y)
