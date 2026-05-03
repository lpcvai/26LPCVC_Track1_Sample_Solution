import argparse
import io
import urllib.request

import numpy as np
import open_clip
import qai_hub
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

from utils import MODELS, NUM_CALIBRATION_SAMPLES

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=MODELS.keys(), default="MobileCLIP2-S3", help="Model to target")
parser.add_argument("--num-samples", type=int, default=NUM_CALIBRATION_SAMPLES,
                    help="Number of validation samples to use for calibration")
args = parser.parse_args()

MODEL_NAME = args.model
MODEL_PRETRAINED = MODELS[args.model]

print(f"Loading preprocess and tokenizer for {MODEL_NAME}...")
_, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=MODEL_PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

print("Loading validation split...")
ds = load_dataset("yerevann/coco-karpathy")
samples = list(ds["validation"])[:args.num_samples]

# ── Images ────────────────────────────────────────────────────────────────────
print(f"Downloading and preprocessing {len(samples)} calibration images...")
calib_images = []
for example in tqdm(samples, desc="images"):
    with urllib.request.urlopen(example["url"], timeout=10) as resp:
        img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    calib_images.append(preprocess(img).unsqueeze(0).numpy())  # [1, 3, H, W]

print("Uploading image calibration dataset...")
image_calib_dataset = qai_hub.upload_dataset({"image": calib_images})
print(f"Image calibration dataset ID: {image_calib_dataset.dataset_id}")

# ── Texts ─────────────────────────────────────────────────────────────────────
# One caption per image is sufficient for calibration — we don't need full
# ground-truth coverage here, just representative token distributions.
print(f"Tokenizing {len(samples)} calibration captions...")
calib_texts = []
for example in tqdm(samples, desc="texts"):
    tokens = tokenizer([example["sentences"][0]])  # [1, context_length]
    calib_texts.append(tokens.numpy().astype(np.int32))

print("Uploading text calibration dataset...")
text_calib_dataset = qai_hub.upload_dataset({"text": calib_texts})
print(f"Text calibration dataset ID: {text_calib_dataset.dataset_id}")
