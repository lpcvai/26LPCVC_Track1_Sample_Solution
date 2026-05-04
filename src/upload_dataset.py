import argparse
import io
import urllib.request

import numpy as np
import open_clip
import qai_hub
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

from utils import MODELS, NUM_IMAGE_SAMPLES, CAPTIONS_PER_IMAGE, JOB_IDS, IMAGES_PER_BATCH


def upload_datasets(
    model_name: str,
    *,
    images_per_batch: int = IMAGES_PER_BATCH,
    persist_job_ids: bool = True,
    num_image_samples: int = NUM_IMAGE_SAMPLES,
    captions_per_image: int = CAPTIONS_PER_IMAGE,
):
    """Upload the COCO-Karpathy subset to QAI Hub and persist dataset IDs.

    For large NUM_IMAGE_SAMPLES, images are uploaded in multiple datasets to avoid QAI Hub's ~2GB dataset limit.
    """
    model_pretrained = MODELS[model_name]
    if images_per_batch <= 0:
        raise ValueError("images_per_batch must be > 0")

    # Load preprocess transform and tokenizer (model weights not needed for upload)
    print(f"Loading preprocess and tokenizer for {model_name}...")
    # Force preprocessing to output 224x224 to match our ONNX export + Hub compile input specs.
    _, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_pretrained, force_image_size=224)
    tokenizer = open_clip.get_tokenizer(model_name)

    if num_image_samples <= 0:
        raise ValueError("num_image_samples must be > 0")
    if captions_per_image <= 0:
        raise ValueError("captions_per_image must be > 0")

    # Take the first num_image_samples examples from the test split
    print("Loading dataset...")
    ds = load_dataset("yerevann/coco-karpathy")
    samples = list(ds["test"])[:num_image_samples]

    # -----------------------------
    # Images
    # -----------------------------
    print(f"Downloading, preprocessing, and uploading {num_image_samples} images...")
    image_dataset_ids = []
    batch_images = []
    batch_start = 0
    for example in tqdm(samples, desc="images"):
        with urllib.request.urlopen(example["url"], timeout=10) as resp:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        tensor = preprocess(img)  # [3, H, W], float32
        # Some model variants use a different default input resolution (e.g., 256).
        # For this project, we force everything to 224x224 to match the compiled models.
        if tensor.shape[-2:] != (224, 224):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        batch_images.append(tensor.unsqueeze(0).numpy())  # [1, 3, H, W]
        if len(batch_images) >= images_per_batch:
            end = batch_start + len(batch_images)
            ds = qai_hub.upload_dataset({"image": batch_images}, name=f"coco_images_{batch_start}_{end}")
            image_dataset_ids.append(ds.dataset_id)
            print(ds)
            batch_images = []
            batch_start = end

    if batch_images:
        end = batch_start + len(batch_images)
        ds = qai_hub.upload_dataset({"image": batch_images}, name=f"coco_images_{batch_start}_{end}")
        image_dataset_ids.append(ds.dataset_id)
        print(ds)

    # Persist batched dataset IDs.
    if persist_job_ids:
        JOB_IDS["image", "dataset_ids"] = image_dataset_ids

    # -----------------------------
    # Texts
    # -----------------------------
    # Upload all captions for each image so ground truth has captions_per_image
    # entries per image (matching how the baseline scripts compute recall).
    # Texts are ordered: all captions for image 0, then image 1, etc.
    num_text_samples = num_image_samples * captions_per_image
    print(f"Tokenizing {num_text_samples} captions...")
    tokenized_texts = []
    for example in tqdm(samples, desc="texts"):
        for caption in example["sentences"][:captions_per_image]:
            tokens = tokenizer([caption])  # [1, context_length], int64
            tokenized_texts.append(tokens.numpy().astype(np.int32))

    print("Uploading text dataset...")
    text_dataset = qai_hub.upload_dataset({"text": tokenized_texts})
    if persist_job_ids:
        JOB_IDS["text", "dataset_id"] = text_dataset.dataset_id
    print(text_dataset)

    return image_dataset_ids, text_dataset.dataset_id


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), default="MobileCLIP2-S0", help="Model to target")
    parser.add_argument("--images-per-batch", type=int, default=None,
                        help=f"Images per uploaded dataset batch (default from utils.py: {IMAGES_PER_BATCH}).")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    upload_datasets(args.model, images_per_batch=(args.images_per_batch or IMAGES_PER_BATCH), persist_job_ids=True)


if __name__ == "__main__":
    main()
