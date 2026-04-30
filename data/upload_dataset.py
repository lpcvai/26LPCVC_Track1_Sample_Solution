import pandas as pd
from transformers import CLIPTokenizer
import qai_hub


from utils.img_utils import load_images_from_folder

# TODO: Define image folder path
image_folder = "dataset/images"  # change to your folder

# Process images
input_image = load_images_from_folder(image_folder)
print(len(input_image))

# Check dataset properties
print(f"Processed {len(input_image)} images.")
print(f"First image shape: {input_image[0].shape}")  # Should be (1, 3, 224, 224)

# Upload dataset
print(qai_hub.upload_dataset({"image": input_image}))

# TODO: Load txt CSV
csv_path = "dataset/txt_list.csv"
df = pd.read_csv(csv_path)

# Get unique text prompts in order from the second column, drop NaN
prompts = df.iloc[:, 1].dropna().tolist()

# Load CLIP tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Tokenize prompts into numpy arrays of shape (1, 77) and dtype int32
tokenized_texts = []
for prompt in prompts:
    tokens = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )["input_ids"].to(torch.int32)  # torch tensor [1, 77], int32
    tokenized_texts.append(tokens.numpy())  # convert to numpy array

# Example: check first element
print(tokenized_texts[0].shape)  # (1, 77)
print(tokenized_texts[0].dtype)  # int32

print(qai_hub.upload_dataset({"text": tokenized_texts}))