import argparse
import io
import os
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image
from datasets import load_dataset
from timm.utils import reparameterize_model
from tqdm import tqdm

from utils import RESULTS_PATH, MODELS, BATCH_SIZE, NUM_DOWNLOAD_WORKERS, K

parser = argparse.ArgumentParser()
parser.add_argument("--test-only", action="store_true", help="Run on the test split only")
parser.add_argument("--gpu", action="store_true", help="Run model inference on GPU")
parser.add_argument("--model", choices=MODELS.keys(), default="MobileCLIP2-S3", help="Model to use")
args = parser.parse_args()

MODEL_NAME = args.model
MODEL_PRETRAINED = MODELS[args.model]


def download_image(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


print("Loading model...")
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=MODEL_PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model.eval()
model = reparameterize_model(model)

device = "cuda" if args.gpu else "cpu"
model = model.to(device)

print("Loading dataset...")
ds = load_dataset("yerevann/coco-karpathy")
splits = ["test"] if args.test_only else list(ds.keys())

# Collect texts and images across selected splits, tracking cocoid for recall
all_texts = []     # flat list of all captions
text_cocoids = []  # parallel: which image each caption belongs to
image_urls = []    # one entry per image
image_cocoids = [] # parallel: cocoid for each image

for split in splits:
    for example in ds[split]:
        cid = example["cocoid"]
        for sentence in example["sentences"]:
            all_texts.append(sentence)
            text_cocoids.append(cid)
        image_urls.append(example["url"])
        image_cocoids.append(cid)

# cocoid -> list of indices into all_texts (ground-truth captions for that image)
cocoid_to_text_indices = defaultdict(list)
for idx, cid in enumerate(text_cocoids):
    cocoid_to_text_indices[cid].append(idx)

print(f"Total captions: {len(all_texts)}, Total images: {len(image_urls)}")

# ── Text encoding ──────────────────────────────────────────────────────────────
os.makedirs(RESULTS_PATH, exist_ok=True)

print("Encoding texts...")
all_text_embeddings = []
with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda"):
    for i in tqdm(range(0, len(all_texts), BATCH_SIZE), unit="batch", desc="texts"):
        batch = all_texts[i : i + BATCH_SIZE]
        tokens = tokenizer(batch).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        all_text_embeddings.append(features.cpu().float().numpy())

text_embeddings = np.concatenate(all_text_embeddings, axis=0)
np.save(os.path.join(RESULTS_PATH, "text_embeddings.npy"), text_embeddings)
print(f"Saved text_embeddings.npy {text_embeddings.shape}")

# ── Image encoding ─────────────────────────────────────────────────────────────
print("Encoding images...")
all_image_embeddings = []

image_pbar = tqdm(range(0, len(image_urls), BATCH_SIZE), unit="batch", desc="images")
for batch_start in image_pbar:
    batch_urls = image_urls[batch_start : batch_start + BATCH_SIZE]

    # Download the batch in parallel
    images = [None] * len(batch_urls)
    with ThreadPoolExecutor(max_workers=NUM_DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(download_image, url): i for i, url in enumerate(batch_urls)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                images[idx] = future.result()
            except Exception as e:
                image_pbar.write(f"Warning: failed to download {batch_urls[idx]}: {e}")

    # Skip any images that failed to download (keep index alignment via NaN row later)
    valid = [(i, img) for i, img in enumerate(images) if img is not None]
    if not valid:
        continue

    pixel_values = torch.stack([preprocess(img) for _, img in valid]).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda"):
        features = model.encode_image(pixel_values)
        features = features / features.norm(dim=-1, keepdim=True)

    # Place embeddings back in order, NaN for failures
    batch_embeddings = np.full((len(batch_urls), features.shape[-1]), np.nan, dtype=np.float32)
    for rank, (i, _) in enumerate(valid):
        batch_embeddings[i] = features[rank].cpu().float().numpy()
    all_image_embeddings.append(batch_embeddings)

image_embeddings = np.concatenate(all_image_embeddings, axis=0)
np.save(os.path.join(RESULTS_PATH, "image_embeddings.npy"), image_embeddings)
print(f"Saved image_embeddings.npy {image_embeddings.shape}")

# ── FAISS KNN index over text embeddings ───────────────────────────────────────
print("Building FAISS index...")
dim = text_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # inner product == cosine sim for unit vectors
index.add(text_embeddings)
print(f"Index built: {index.ntotal} vectors")

# Search all valid image embeddings at once
valid_mask = ~np.any(np.isnan(image_embeddings), axis=1)
valid_image_emb = np.ascontiguousarray(image_embeddings[valid_mask])
valid_indices = np.where(valid_mask)[0]

print(f"Searching top-{K} texts for {len(valid_image_emb)} images...")
_, top_k_indices = index.search(valid_image_emb, K)  # [N_valid, K]

# ── Recall@10 ─────────────────────────────────────────────────────────────────
print("Computing recall@10...")
recalls = []
for rank, img_idx in enumerate(tqdm(valid_indices, desc="recall@10")):
    cid = image_cocoids[img_idx]
    gt_indices = set(cocoid_to_text_indices[cid])
    top10 = set(top_k_indices[rank].tolist())
    recall_i = len(gt_indices & top10) / len(gt_indices)
    recalls.append(recall_i)

recall_at_10 = float(np.mean(recalls))
print(f"Recall@10: {recall_at_10:.4f}  ({recall_at_10*100:.2f}%)")
