from pathlib import Path

import qai_hub

from utils.img_utils import load_images
from utils.refcoco_utils import RefCocoSplit

data_folder = Path("data")

# Process images
input_data = load_images(data_folder, RefCocoSplit.VAL)
print(len(input_data))

# Check dataset properties
if input_data:
    print(f"Processed {len(input_data)} images.")
    print(f"First image shape: {input_data[0].shape}")  # Should be (1, 3, 224, 224)

# Upload dataset
print(qai_hub.upload_dataset({"image": input_data}))