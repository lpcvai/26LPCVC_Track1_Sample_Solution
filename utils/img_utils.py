from pathlib import Path

import numpy as np
from PIL import Image

from utils.text_utils import load_annotations


def process_image(image_path, target_size=(224, 224)):
    """Loads and processes an image to the required input shape (C, H, W)."""
    image = Image.open(image_path).convert('RGB').resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
    return np.transpose(image_array, (2, 0, 1))[np.newaxis, :]  # Convert to (1, C, H, W)


def load_images(folder_path: Path, split=None, target_size=(224, 224)):
    """Loads and processes all the RefCOCO images in the given folder."""
    image_paths = load_annotations(split)["image_path"]
    return [process_image(folder_path / path, target_size) for path in image_paths]




