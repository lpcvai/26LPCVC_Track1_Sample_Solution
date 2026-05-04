import argparse
import io
import urllib.request

import open_clip
import torch.nn.functional as F
import torchvision.transforms as T  # type: ignore
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

from dataset_store import get_or_upload_dataset, try_resolve_cached_dataset_id
from utils import MODELS, MODEL_PRETRAINED, NUM_CALIBRATION_SAMPLES


def upload_calibration_datasets(
        model_name: str,
        *,
        num_samples: int = NUM_CALIBRATION_SAMPLES,
        cache: bool = True,
        cache_write: bool = True,
):
    """
    Upload small calibration datasets for post-training quantization.

    Returns (image_calibration_id, text_calibration_id).
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    model_pretrained = MODEL_PRETRAINED[model_name]
    print(f"Loading preprocess and tokenizer for {model_name}...")
    # Force preprocessing to output 224x224 to match our ONNX export + Hub compile input specs.
    _, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_pretrained,
                                                             force_image_size=224)
    tokenizer = open_clip.get_tokenizer(model_name)

    # Model-agnostic signatures for safe reuse across models when tensorization matches.
    mean = None
    std = None
    transforms = getattr(preprocess, "transforms", None) or []
    for tr in transforms:
        if isinstance(tr, T.Normalize):
            mean = tuple(float(x) for x in tr.mean)
            std = tuple(float(x) for x in tr.std)
            break

    fmt_mean = "None" if mean is None else "[" + ",".join(f"{float(x):.8f}".rstrip("0").rstrip(".") for x in mean) + "]"
    fmt_std = "None" if std is None else "[" + ",".join(f"{float(x):.8f}".rstrip("0").rstrip(".") for x in std) + "]"
    preproc_sig = f"img224:mean={fmt_mean}:std={fmt_std}"

    tok_shape = None
    tok_dtype = None
    try:
        tok = tokenizer(["a test"])
        tok_shape = tuple(int(x) for x in getattr(tok, "shape", ()))
        tok_dtype = str(getattr(tok, "dtype", "unknown"))
    except Exception:
        pass
    tok_sig = f"shape={tok_shape}:dtype={tok_dtype}"

    print("Loading validation split...")
    ds = load_dataset("yerevann/coco-karpathy")
    samples = list(ds["validation"])[:num_samples]

    # ── Images ───────────────────────────────────────────────────────────────
    image_key = f"coco-karpathy:validation:firstN={num_samples}:kind=calib_image:preproc={preproc_sig}"
    image_calib_id = try_resolve_cached_dataset_id(key=image_key, cache=cache, cache_write=cache_write)
    if image_calib_id is None:
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
        image_calib_id = get_or_upload_dataset(
            {"image": calib_images},
            name=f"calib_images_{model_name}_{num_samples}",
            key=image_key,
            kind="calib_image",
            meta={"split": "validation", "num_samples": num_samples},
            cache=cache,
            cache_write=cache_write,
        )

    print(f"Image calibration dataset ID: {image_calib_id}")

    # ── Texts ───────────────────────────────────────────────────────────────
    # One caption per image is sufficient for calibration.
    text_key = f"coco-karpathy:validation:firstN={num_samples}:kind=calib_text:tok={tok_sig}"
    text_calib_id = try_resolve_cached_dataset_id(key=text_key, cache=cache, cache_write=cache_write)
    if text_calib_id is None:
        print(f"Tokenizing {len(samples)} calibration captions...")
        calib_texts = []
        for example in tqdm(samples, desc="texts"):
            tokens = tokenizer([example["sentences"][0]])  # [1, context_length]
            calib_texts.append(tokens.numpy().astype("int32"))

        print("Uploading text calibration dataset...")
        text_calib_id = get_or_upload_dataset(
            {"text": calib_texts},
            name=f"calib_text_{model_name}_{num_samples}",
            key=text_key,
            kind="calib_text",
            meta={"split": "validation", "num_samples": num_samples},
            cache=cache,
            cache_write=cache_write,
        )
    print(f"Text calibration dataset ID: {text_calib_id}")

    return image_calib_id, text_calib_id


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="MobileCLIP2-S3", help="Model to target")
    parser.add_argument("--num-samples", type=int, default=NUM_CALIBRATION_SAMPLES,
                        help="Number of validation samples to use for calibration")
    parser.add_argument("--cache", default=True, action=argparse.BooleanOptionalAction,
                        help="Reuse datasets via datasets.json (default: enabled).")
    parser.add_argument("--cache-write", default=True, action=argparse.BooleanOptionalAction,
                        help="Write uploaded dataset info into datasets.json (default: enabled).")
    args = parser.parse_args(argv)

    upload_calibration_datasets(
        args.model,
        num_samples=args.num_samples,
        cache=bool(getattr(args, "cache", True)),
        cache_write=bool(getattr(args, "cache_write", True)),
    )


if __name__ == "__main__":
    main()
