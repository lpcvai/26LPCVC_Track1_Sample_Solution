import os

import numpy as np
from PIL import Image


def process_image(image_path, target_size=(224, 224)):
    """Loads and processes an image to the required input shape (C, H, W)."""
    image = Image.open(image_path).convert('RGB').resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
    return np.transpose(image_array, (2, 0, 1))[np.newaxis, :]  # Convert to (1, C, H, W)

def load_images_from_folder(folder_path, target_size=(224, 224)):
    """Loads and processes all images in a folder, sorted by name."""
    image_paths = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))
    ])
    return [process_image(path, target_size) for path in image_paths]
