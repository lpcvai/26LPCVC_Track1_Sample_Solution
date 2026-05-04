import argparse
import io
import urllib.request

import numpy as np
import open_clip
import torch.nn.functional as F
import torchvision.transforms as T  # type: ignore
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

from dataset_store import get_or_upload_dataset, try_resolve_cached_dataset_id
from utils import MODELS, MODEL_PRETRAINED, NUM_IMAGE_SAMPLES, CAPTIONS_PER_IMAGE, JOB_IDS, IMAGES_PER_BATCH


def upload_datasets(
    model_name: str,
    *,
    images_per_batch: int = IMAGES_PER_BATCH,
    persist_job_ids: bool = True,
    num_image_samples: int = NUM_IMAGE_SAMPLES,
    captions_per_image: int = CAPTIONS_PER_IMAGE,
    cache: bool = True,
    cache_write: bool = True,
):
    """Upload the COCO-Karpathy subset to QAI Hub and persist dataset IDs.

    For large NUM_IMAGE_SAMPLES, images are uploaded in multiple datasets to avoid QAI Hub's ~2GB dataset limit.
    """
    model_pretrained = MODEL_PRETRAINED[model_name]
    if images_per_batch <= 0:
        raise ValueError("images_per_batch must be > 0")

    # Load preprocess transform and tokenizer (model weights not needed for upload)
    print(f"Loading preprocess and tokenizer for {model_name}...")
    # Force preprocessing to output 224x224 to match our ONNX export + Hub compile input specs.
    _, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_pretrained, force_image_size=224)
    tokenizer = open_clip.get_tokenizer(model_name)

    # Build a model-agnostic cache signature from preprocess/tokenizer.
    # This lets datasets be reused across models as long as the actual tensorization matches.
    mean = None
    std = None
    transforms = getattr(preprocess, "transforms", None) or []
    for tr in transforms:
        if isinstance(tr, T.Normalize):
            mean = tuple(float(x) for x in tr.mean)
            std = tuple(float(x) for x in tr.std)
            break

    tok_shape = None
    tok_dtype = None
    try:
        tok = tokenizer(["a test"])
        tok_shape = tuple(int(x) for x in getattr(tok, "shape", ()))
        tok_dtype = str(getattr(tok, "dtype", "unknown"))
    except Exception:
        pass

    fmt_mean = "None" if mean is None else "[" + ",".join(f"{float(x):.8f}".rstrip("0").rstrip(".") for x in mean) + "]"
    fmt_std = "None" if std is None else "[" + ",".join(f"{float(x):.8f}".rstrip("0").rstrip(".") for x in std) + "]"
    preproc_sig = f"img224:mean={fmt_mean}:std={fmt_std}"
    tok_sig = f"shape={tok_shape}:dtype={tok_dtype}"

    if num_image_samples <= 0:
        raise ValueError("num_image_samples must be > 0")
    if captions_per_image <= 0:
        raise ValueError("captions_per_image must be > 0")

    # Take the first num_image_samples examples from the test split
    print("Loading dataset...")
    hf_ds = load_dataset("yerevann/coco-karpathy")
    split_name = "test"
    split_ds = hf_ds[split_name]
    samples = list(split_ds)[:num_image_samples]

    # Best-effort stable-ish fingerprint from HF datasets. Not guaranteed, so we
    # still include explicit params (split, firstN, etc.) in keys.
    # HF dataset fingerprints are not guaranteed stable across runs, so do not use them
    # as part of the cache identity.
    base_key = f"coco-karpathy:{split_name}:firstN={num_image_samples}:preproc={preproc_sig}"

    # -----------------------------
    # Images
    # -----------------------------
    # First, try to resolve *all* image batch dataset IDs from cache so we can avoid
    # downloading/preprocessing entirely when reusing.
    image_dataset_ids: list[str | None] = []
    batch_specs: list[tuple[int, int, str]] = []  # (start,end,key)
    batch_start = 0
    while batch_start < num_image_samples:
        end = min(num_image_samples, batch_start + images_per_batch)
        batch_index = batch_start // images_per_batch
        key = f"{base_key}:kind=image_batch:batch={images_per_batch}:batch_index={batch_index}:offset={batch_start}:count={end - batch_start}"
        batch_specs.append((batch_start, end, key))
        image_dataset_ids.append(try_resolve_cached_dataset_id(key=key, cache=cache, cache_write=cache_write))
        batch_start = end

    if all(x is not None for x in image_dataset_ids):
        image_dataset_ids_final = [str(x) for x in image_dataset_ids]
    else:
        print(f"Downloading, preprocessing, and uploading {num_image_samples} images...")
        # Only build/upload batches that were cache-misses.
        image_dataset_ids_final: list[str] = []
        for (start, end, key), cached_id in zip(batch_specs, image_dataset_ids):
            if cached_id is not None:
                image_dataset_ids_final.append(str(cached_id))
                continue
            batch_images = []
            for example in tqdm(samples[start:end], desc=f"images {start}:{end}"):
                with urllib.request.urlopen(example["url"], timeout=10) as resp:
                    img = Image.open(io.BytesIO(resp.read())).convert("RGB")
                tensor = preprocess(img)  # [3, H, W], float32
                if tensor.shape[-2:] != (224, 224):
                    tensor = F.interpolate(
                        tensor.unsqueeze(0),
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                batch_images.append(tensor.unsqueeze(0).numpy())

            ds_name = f"coco_karpathy_{split_name}_img224_first{num_image_samples}_{start}_{end}"
            ds_id = get_or_upload_dataset(
                {"image": batch_images},
                name=ds_name,
                key=key,
                kind="image_batch",
                meta={"split": split_name, "offset": start, "count": end - start, "num_images": num_image_samples},
                cache=cache,
                cache_write=cache_write,
            )
            image_dataset_ids_final.append(ds_id)

    # Persist batched dataset IDs.
    if persist_job_ids:
        JOB_IDS["image", "dataset_ids"] = image_dataset_ids_final

    # -----------------------------
    # Texts
    # -----------------------------
    # Upload all captions for each image so ground truth has captions_per_image
    # entries per image (matching how the baseline scripts compute recall).
    # Texts are ordered: all captions for image 0, then image 1, etc.
    text_ds_name = f"coco_karpathy_{split_name}_text_first{num_image_samples}_cap{captions_per_image}"
    text_key = f"coco-karpathy:{split_name}:firstN={num_image_samples}:kind=text:cap={captions_per_image}:tok={tok_sig}"
    text_dataset_id = try_resolve_cached_dataset_id(key=text_key, cache=cache, cache_write=cache_write)
    if text_dataset_id is None:
        num_text_samples = num_image_samples * captions_per_image
        print(f"Tokenizing {num_text_samples} captions...")
        tokenized_texts = []
        for example in tqdm(samples, desc="texts"):
            for caption in example["sentences"][:captions_per_image]:
                tokens = tokenizer([caption])  # [1, context_length], int64
                tokenized_texts.append(tokens.numpy().astype(np.int32))

        print("Uploading text dataset...")
        text_dataset_id = get_or_upload_dataset(
            {"text": tokenized_texts},
            name=text_ds_name,
            key=text_key,
            kind="text",
            meta={"split": split_name, "num_images": num_image_samples, "captions_per_image": captions_per_image},
            cache=cache,
            cache_write=cache_write,
        )
    if persist_job_ids:
        JOB_IDS["text", "dataset_id"] = text_dataset_id

    return image_dataset_ids_final, text_dataset_id


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="MobileCLIP2-S0", help="Model to target")
    parser.add_argument("--images-per-batch", type=int, default=None,
                        help=f"Images per uploaded dataset batch (default from utils.py: {IMAGES_PER_BATCH}).")
    parser.add_argument("--cache", default=True, action=argparse.BooleanOptionalAction,
                        help="Reuse datasets via datasets.json (default: enabled).")
    parser.add_argument("--cache-write", default=True, action=argparse.BooleanOptionalAction,
                        help="Write uploaded dataset info into datasets.json (default: enabled).")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    upload_datasets(
        args.model,
        images_per_batch=(args.images_per_batch or IMAGES_PER_BATCH),
        persist_job_ids=True,
        cache=bool(getattr(args, "cache", True)),
        cache_write=bool(getattr(args, "cache_write", True)),
    )


if __name__ == "__main__":
    main()
