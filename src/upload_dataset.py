import argparse
import io
import urllib.request

import numpy as np
import open_clip
import qai_hub
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

from utils import MODELS, NUM_IMAGE_SAMPLES, CAPTIONS_PER_IMAGE, NUM_TEXT_SAMPLES, JOB_IDS

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=MODELS.keys(), default="MobileCLIP2-S0", help="Model to target")
args = parser.parse_args()

MODEL_NAME = args.model
MODEL_PRETRAINED = MODELS[args.model]

# Load preprocess transform and tokenizer (model weights not needed for upload)
print(f"Loading preprocess and tokenizer for {MODEL_NAME}...")
_, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=MODEL_PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# Take the first NUM_IMAGE_SAMPLES examples from the test split
print("Loading dataset...")
ds = load_dataset("yerevann/coco-karpathy")
samples = list(ds["test"])[:NUM_IMAGE_SAMPLES]

# -----------------------------
# Images
# -----------------------------
print(f"Downloading and preprocessing {NUM_IMAGE_SAMPLES} images...")
input_images = []
for example in tqdm(samples, desc="images"):
    with urllib.request.urlopen(example["url"], timeout=10) as resp:
        img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    tensor = preprocess(img)                    # [3, H, W], float32
    input_images.append(tensor.unsqueeze(0).numpy())  # [1, 3, H, W]

print("Uploading image dataset...")
image_dataset = qai_hub.upload_dataset({"image": input_images})
JOB_IDS["image", "dataset_id"] = image_dataset.dataset_id
print(image_dataset)

# -----------------------------
# Texts
# -----------------------------
# Upload all captions for each image so ground truth has CAPTIONS_PER_IMAGE
# entries per image (matching how the baseline scripts compute recall).
# Texts are ordered: all captions for image 0, then image 1, etc.
print(f"Tokenizing {NUM_TEXT_SAMPLES} captions...")
tokenized_texts = []
for example in tqdm(samples, desc="texts"):
    for caption in example["sentences"][:CAPTIONS_PER_IMAGE]:
        tokens = tokenizer([caption])  # [1, context_length], int64
        tokenized_texts.append(tokens.numpy().astype(np.int32))

print("Uploading text dataset...")
text_dataset = qai_hub.upload_dataset({"text": tokenized_texts})
JOB_IDS["text", "dataset_id"] = text_dataset.dataset_id
print(text_dataset)

