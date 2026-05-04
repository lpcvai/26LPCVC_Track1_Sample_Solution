import argparse
import io
import urllib.request
from datetime import datetime, timezone, timedelta
import os

import open_clip
import qai_hub
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

from utils import MODELS, MODEL_PRETRAINED, NUM_CALIBRATION_SAMPLES, DATASETS
from dataset_store import get_or_upload_dataset


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
    _, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_pretrained, force_image_size=224)
    tokenizer = open_clip.get_tokenizer(model_name)

    # Model-agnostic signatures for safe reuse across models when tensorization matches.
    mean = None
    std = None
    transforms = getattr(preprocess, "transforms", None) or []
    try:
        import torchvision.transforms as T  # type: ignore
        for tr in transforms:
            if isinstance(tr, T.Normalize):
                mean = tuple(float(x) for x in tr.mean)
                std = tuple(float(x) for x in tr.std)
                break
    except Exception:
        pass
    def _fmt_floats(xs):
        if xs is None:
            return "None"
        return "[" + ",".join(f"{float(x):.8f}".rstrip("0").rstrip(".") for x in xs) + "]"

    preproc_sig = f"img224:mean={_fmt_floats(mean)}:std={_fmt_floats(std)}"

    tok_shape = None
    tok_dtype = None
    try:
        tok = tokenizer(["a test"])
        tok_shape = tuple(int(x) for x in getattr(tok, "shape", ()))
        tok_dtype = str(getattr(tok, "dtype", "unknown"))
    except Exception:
        pass
    tok_sig = f"shape={tok_shape}:dtype={tok_dtype}"

    debug = os.getenv("QAI_DATASET_CACHE_DEBUG", "").strip().lower() in ("1", "true", "yes", "y", "on")

    def _try_resolve_cached_id(key: str) -> str | None:
        if not cache:
            if debug:
                print(f"[dataset-cache] request key={key!r} cache=False -> bypass")
            return None
        DATASETS.load()
        existing = DATASETS.find_by_key(key)
        if debug:
            if existing is None:
                print(f"[dataset-cache] request key={key!r} lookup: MISS")
            else:
                print(f"[dataset-cache] request key={key!r} lookup: HIT id={existing.dataset_id} expires={existing.expiration_time}")
        if existing is None:
            return None
        now = datetime.now(timezone.utc)
        if existing.expiration_time is not None and existing.expiration_time > (now + timedelta(minutes=5)):
            if debug:
                print("[dataset-cache] reuse: local expiration OK; returning cached dataset id")
            return existing.dataset_id
        try:
            remote = qai_hub.get_dataset(existing.dataset_id)
            if not remote.is_expired():
                from dataset_registry import DatasetInfo
                exp = getattr(remote, "expiration_time", None)
                if exp is not None and getattr(exp, "tzinfo", None) is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                refreshed = DatasetInfo(
                    dataset_id=remote.dataset_id,
                    name=getattr(remote, "name", None),
                    expiration_time=exp,
                    key=existing.key,
                    kind=existing.kind,
                    meta=existing.meta,
                )
                if cache_write:
                    DATASETS.upsert(refreshed)
                if debug:
                    print("[dataset-cache] reuse: revalidated against Hub; returning cached dataset id")
                return existing.dataset_id
        except Exception:
            if debug:
                print("[dataset-cache] reuse: Hub get_dataset failed; will treat as MISS")
            return None
        return None

    print("Loading validation split...")
    ds = load_dataset("yerevann/coco-karpathy")
    samples = list(ds["validation"])[:num_samples]

    # ── Images ───────────────────────────────────────────────────────────────
    image_key = f"coco-karpathy:validation:firstN={num_samples}:kind=calib_image:preproc={preproc_sig}"
    image_calib_id = _try_resolve_cached_id(image_key)
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
    elif debug:
        print("[dataset-cache] calib images: cached; skipping download/preprocess/upload")
    print(f"Image calibration dataset ID: {image_calib_id}")

    # ── Texts ───────────────────────────────────────────────────────────────
    # One caption per image is sufficient for calibration.
    text_key = f"coco-karpathy:validation:firstN={num_samples}:kind=calib_text:tok={tok_sig}"
    text_calib_id = _try_resolve_cached_id(text_key)
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
    elif debug:
        print("[dataset-cache] calib texts: cached; skipping tokenization/upload")
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
