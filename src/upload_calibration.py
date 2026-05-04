import argparse
import io
import urllib.request

import open_clip
import qai_hub
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

from utils import MODELS, NUM_CALIBRATION_SAMPLES


def upload_calibration_datasets(model_name: str, *, num_samples: int = NUM_CALIBRATION_SAMPLES):
    """
    Upload small calibration datasets for post-training quantization.

    Returns (image_calibration_id, text_calibration_id).
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    model_pretrained = MODELS[model_name]
    print(f"Loading preprocess and tokenizer for {model_name}...")
    # Force preprocessing to output 224x224 to match our ONNX export + Hub compile input specs.
    _, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_pretrained, force_image_size=224)
    tokenizer = open_clip.get_tokenizer(model_name)

    print("Loading validation split...")
    ds = load_dataset("yerevann/coco-karpathy")
    samples = list(ds["validation"])[:num_samples]

    # ── Images ───────────────────────────────────────────────────────────────
    print(f"Downloading and preprocessing {len(samples)} calibration images...")
    calib_images = []
    for example in tqdm(samples, desc="images"):
        with urllib.request.urlopen(example["url"], timeout=10) as resp:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        tensor = preprocess(img)  # [3, H, W]
        if tensor.shape[-2:] != (224, 224):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        calib_images.append(tensor.unsqueeze(0).numpy())  # [1, 3, 224, 224]

    print("Uploading image calibration dataset...")
    image_calib_dataset = qai_hub.upload_dataset({"image": calib_images}, name=f"calib_images_{model_name}_{num_samples}")
    print(f"Image calibration dataset ID: {image_calib_dataset.dataset_id}")

    # ── Texts ───────────────────────────────────────────────────────────────
    # One caption per image is sufficient for calibration.
    print(f"Tokenizing {len(samples)} calibration captions...")
    calib_texts = []
    for example in tqdm(samples, desc="texts"):
        tokens = tokenizer([example["sentences"][0]])  # [1, context_length]
        calib_texts.append(tokens.numpy().astype("int32"))

    print("Uploading text calibration dataset...")
    text_calib_dataset = qai_hub.upload_dataset({"text": calib_texts}, name=f"calib_text_{model_name}_{num_samples}")
    print(f"Text calibration dataset ID: {text_calib_dataset.dataset_id}")

    return image_calib_dataset.dataset_id, text_calib_dataset.dataset_id


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), default="MobileCLIP2-S3", help="Model to target")
    parser.add_argument("--num-samples", type=int, default=NUM_CALIBRATION_SAMPLES,
                        help="Number of validation samples to use for calibration")
    args = parser.parse_args(argv)

    upload_calibration_datasets(args.model, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
