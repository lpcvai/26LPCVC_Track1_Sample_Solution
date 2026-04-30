import os
from pathlib import Path


import datasets
import numpy as np
from PIL import Image
from datasets import Dataset, DatasetDict

from utils.refcoco_utils import RefCocoSplit

REFCOCO_UNC_DIR = Path("data/annotations/refcoco-unc")

def load_annotations(split: RefCocoSplit | None = None) -> Dataset | DatasetDict:
    if split is None:
        return datasets.load_from_disk(REFCOCO_UNC_DIR)
    return datasets.load_from_disk(REFCOCO_UNC_DIR / split)


def process_image(image_path, target_size=(224, 224)):
    """Loads and processes an image to the required input shape (C, H, W)."""
    image = Image.open(image_path).convert('RGB').resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
    return np.transpose(image_array, (2, 0, 1))[np.newaxis, :]  # Convert to (1, C, H, W)


def load_images(folder_path: Path, split=None, target_size=(224, 224)):
    """Loads and processes all the RefCOCO images in the given folder."""
    image_paths = load_annotations(split)["image_path"]
    return [process_image(folder_path / path, target_size) for path in image_paths]




