import qai_hub

from utils import load_images_from_folder


# TODO: Define image folder path
image_folder = ""

# Process images
input_data = load_images_from_folder(image_folder)
print(len(input_data))

# Check dataset properties
if input_data:
    print(f"Processed {len(input_data)} images.")
    print(f"First image shape: {input_data[0].shape}")  # Should be (1, 3, 224, 224)

# Upload dataset
print(qai_hub.upload_dataset({"image": input_data}))