import argparse

import numpy as np
import qai_hub

from utils import CAPTIONS_PER_IMAGE, JOB_IDS


def first_output(job):
    """QAI Hub renames outputs to output_0, output_1, … regardless of export names."""
    return next(iter(job.download_output_data().values()))


parser = argparse.ArgumentParser(description="Re-run inference on already-compiled MobileCLIP models")
parser.add_argument("--img-compiled-id", default=JOB_IDS["image", "compiled_id"],
                    help="Compile job ID for the image encoder. If omitted, uses job_ids.json.")
parser.add_argument("--txt-compiled-id", default=JOB_IDS["text", "compiled_id"],
                    help="Compile job ID for the text encoder (required for --topk device). "
                         "If omitted, uses job_ids.json.")
parser.add_argument("--topk-compiled-id", default=JOB_IDS["topk", "compiled_id"],
                    help="Compile job ID for the top-k model (topk_retrieval or faiss_index). "
                         "If omitted, uses job_ids.json.")
parser.add_argument("--image-dataset-id", default=JOB_IDS["image", "dataset_id"],
                    help="QAI Hub dataset ID for images. If omitted, uses job_ids.json.")
parser.add_argument("--text-dataset-id", default=JOB_IDS["text", "dataset_id"],
                    help="QAI Hub dataset ID for texts (required for --topk device). If omitted, uses job_ids.json.")
parser.add_argument("--topk", choices=["device", "faiss"], default="device",
                    help="Top-k method used when the model was compiled")
parser.add_argument("--faiss-compute-unit", choices=["all", "npu", "gpu", "cpu"], default="all",
                    help="Compute unit for the FAISS inference job (only applies with --topk faiss)")
parser.add_argument("--device", default="XR2 Gen 2 (Proxy)", help="QAI Hub target device")
args = parser.parse_args()

missing = []
if args.img_compiled_id is None:
    missing.append("--img-compiled-id")
if args.topk_compiled_id is None:
    missing.append("--topk-compiled-id")
if args.image_dataset_id is None:
    missing.append("--image-dataset-id")
if args.topk == "device":
    if args.txt_compiled_id is None:
        missing.append("--txt-compiled-id")
    if args.text_dataset_id is None:
        missing.append("--text-dataset-id")
if missing:
    parser.error(
        "Missing required parameters (and no defaults in job_ids.json): "
        + ", ".join(missing)
        + "."
    )

target_device = qai_hub.Device(args.device)
image_dataset = qai_hub.get_dataset(args.image_dataset_id)

# Retrieve compiled models from their job IDs
img_compiled = qai_hub.get_job(args.img_compiled_id).get_target_model()
topk_compiled = qai_hub.get_job(args.topk_compiled_id).get_target_model()

# ── Encoder inference ──────────────────────────────────────────────────────────
print("Submitting image encoder inference...")
img_inf_job = qai_hub.submit_inference_job(
    model=img_compiled,
    device=target_device,
    inputs=image_dataset,
)

if args.topk == "device":
    txt_compiled = qai_hub.get_job(args.txt_compiled_id).get_target_model()
    text_dataset = qai_hub.get_dataset(args.text_dataset_id)
    print("Submitting text encoder inference...")
    txt_inf_job = qai_hub.submit_inference_job(
        model=txt_compiled,
        device=target_device,
        inputs=text_dataset,
    )
    text_embs = np.concatenate(first_output(txt_inf_job), axis=0)
    text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)

image_embs = np.concatenate(first_output(img_inf_job), axis=0)
image_embs = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)

# ── Top-k on device ────────────────────────────────────────────────────────────
if args.topk == "device":
    topk_dataset = qai_hub.upload_dataset({
        "image_embs": [image_embs],
        "text_embs": [text_embs],
    })
else:
    # faiss: text embeddings are baked into the compiled model; only image
    # embeddings are needed as runtime input
    topk_dataset = qai_hub.upload_dataset({"image_embs": [image_embs]})

print("Submitting top-k inference...")
topk_options = f"--compute_unit {args.faiss_compute_unit}" if args.topk == "faiss" else None
topk_inf_job = qai_hub.submit_inference_job(
    model=topk_compiled,
    device=target_device,
    inputs=topk_dataset,
    options=topk_options,
)
topk_indices = first_output(topk_inf_job)[0]  # [N, K]

# ── Recall@10 ──────────────────────────────────────────────────────────────────
# Texts are ordered: CAPTIONS_PER_IMAGE captions per image, so ground truth
# for image i is text indices [i*C, (i+1)*C).
N = image_embs.shape[0]
recalls = []
for i in range(N):
    gt = set(range(i * CAPTIONS_PER_IMAGE, (i + 1) * CAPTIONS_PER_IMAGE))
    recalls.append(len(gt & set(topk_indices[i].tolist())) / CAPTIONS_PER_IMAGE)
recall_at_10 = float(np.mean(recalls))
print(f"Recall@10: {recall_at_10:.4f}  ({recall_at_10 * 100:.2f}%)")
