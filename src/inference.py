import argparse
import time

import numpy as np
import qai_hub

from utils import CAPTIONS_PER_IMAGE, JOB_IDS, TOPK_IMAGES_PER_BATCH
from utils import MAX_INFERENCE_INFLIGHT


def first_output(job):
    """QAI Hub renames outputs to output_0, output_1, … regardless of export names."""
    # QAI Hub's `download_output_data()` blocks until completion, but returns None if the
    # inference job failed (no output dataset).
    job_id = getattr(job, "job_id", "<unknown>")
    status = "<unknown>"
    try:
        status = job.wait()
    except Exception:
        # Even if waiting fails (network, transient API), try downloading; we'll still
        # produce a useful error if outputs are missing.
        pass

    outputs = job.download_output_data()
    if outputs is None:
        raise RuntimeError(
            f"QAI Hub inference job produced no outputs (job_id={job_id}, status={status}). "
            f"Check the job page/logs for details."
        )
    return next(iter(outputs.values()))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Re-run inference on already-compiled MobileCLIP models")
    parser.add_argument("--image-compiled-id", default=JOB_IDS["image", "compiled_id"],
                        help="Compile job ID for the image encoder. If omitted, uses job_ids.json.")
    parser.add_argument("--text-compiled-id", default=JOB_IDS["text", "compiled_id"],
                        help="Compile job ID for the text encoder. If omitted, uses job_ids.json.")
    parser.add_argument("--topk-compiled-id", default=None,
                        help="Compile job ID for the cosine top-k model (topk_retrieval.onnx). "
                             "If omitted, uses job_ids.json.")
    parser.add_argument("--image-dataset-ids", default=None,
                        help="Comma-separated list of image dataset ids (batched). Overrides job_ids.json.")
    parser.add_argument("--text-dataset-id", default=JOB_IDS["text", "dataset_id"],
                        help="QAI Hub dataset ID for texts. If omitted, uses job_ids.json.")
    parser.add_argument("--device", default="XR2 Gen 2 (Proxy)", help="QAI Hub target device")
    parser.add_argument("--job-name-prefix", default="",
                        help="Optional prefix for QAI Hub job names to make jobs easier to identify in the UI.")
    parser.add_argument("--topk-images-per-batch", type=int, default=None,
                        help=f"Images per top-k inference job (default from utils.py: {TOPK_IMAGES_PER_BATCH}).")
    return parser


def run_inference(
    *,
    device: str,
    image_compiled_id: str,
    topk_compiled_id: str,
    image_dataset_ids: list[str],
    text_compiled_id: str,
    text_dataset_id: str,
    job_name_prefix: str = "",
    max_inference_inflight: int = 2,
    topk_images_per_batch: int | None = None,
):
    missing = []
    if image_compiled_id is None:
        missing.append("image_compiled_id")
    if topk_compiled_id is None:
        missing.append("topk_compiled_id")
    if not image_dataset_ids:
        missing.append("image_dataset_ids")
    if text_compiled_id is None:
        missing.append("text_compiled_id")
    if text_dataset_id is None:
        missing.append("text_dataset_id")
    if missing:
        raise ValueError(f"Missing required inputs: {missing}")

    target_device = qai_hub.Device(device)
    prefix = (job_name_prefix.strip() + " ") if job_name_prefix and job_name_prefix.strip() else ""

    # Retrieve compiled models from their job IDs
    image_compiled = qai_hub.get_job(image_compiled_id).get_target_model()
    topk_compiled = qai_hub.get_job(topk_compiled_id).get_target_model()

    # ── Encoder inference ───────────────────────────────────────────────────
    # Submit the text encoder job early so it can run in parallel with the batched image encoder jobs.
    text_embs = None
    text_inf_job = None
    text_compiled = qai_hub.get_job(text_compiled_id).get_target_model()
    text_dataset = qai_hub.get_dataset(text_dataset_id)
    print("Submitting text encoder inference...")
    text_inf_job = qai_hub.submit_inference_job(
        model=text_compiled,
        device=target_device,
        inputs=text_dataset,
        name=f"{prefix}text-encoder",
    )

    image_embs_parts = []
    # Keep multiple inference jobs in-flight to improve throughput.
    max_inflight = max(1, int(max_inference_inflight))
    pending = list(enumerate(image_dataset_ids))
    active: list[tuple[int, str, qai_hub.InferenceJob]] = []
    results_by_idx: dict[int, np.ndarray] = {}

    def _submit_one(idx: int, ds_id: str):
        image_dataset = qai_hub.get_dataset(ds_id)
        print(f"Submitting image encoder inference (batch {idx + 1}/{len(image_dataset_ids)})...")
        job = qai_hub.submit_inference_job(
            model=image_compiled,
            device=target_device,
            inputs=image_dataset,
            name=f"{prefix}image b{idx + 1}/{len(image_dataset_ids)}",
        )
        active.append((idx, ds_id, job))

    while pending or active:
        while pending and len(active) < max_inflight:
            idx, ds_id = pending.pop(0)
            _submit_one(idx, ds_id)

        progressed = False

        # Check text encoder completion opportunistically.
        if text_inf_job is not None and text_embs is None:
            st = text_inf_job.get_status()
            if not (st.running or st.pending):
                text_embs = np.concatenate(first_output(text_inf_job), axis=0)
                text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)
                progressed = True

        still_active: list[tuple[int, str, qai_hub.InferenceJob]] = []
        for idx, ds_id, job in active:
            st = job.get_status()
            if st.running or st.pending:
                still_active.append((idx, ds_id, job))
                continue
            part = np.concatenate(first_output(job), axis=0)
            part = part / np.linalg.norm(part, axis=1, keepdims=True)
            results_by_idx[idx] = part
            progressed = True
        active = still_active

        if not progressed and (active or (text_inf_job is not None and text_embs is None)):
            time.sleep(2)

    for i in range(len(image_dataset_ids)):
        image_embs_parts.append(results_by_idx[i])

    if text_embs is None:
        # If it didn't finish during the image loop, block now.
        text_embs = np.concatenate(first_output(text_inf_job), axis=0)
        text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)

    image_embs = np.concatenate(image_embs_parts, axis=0)

    # ── Top-k inference ─────────────────────────────────────────────────────
    total_images = image_embs.shape[0]

    topk_batch = int(topk_images_per_batch or TOPK_IMAGES_PER_BATCH)
    if topk_batch <= 0:
        raise ValueError(f"Invalid topk batch size: {topk_batch}")

    num_topk_batches = (total_images + topk_batch - 1) // topk_batch
    print(f"Submitting cosine top-k inference (batch_size={topk_batch}, batches={num_topk_batches})...")

    topk_max_inflight = max(1, int(max_inference_inflight))
    pending = list(range(num_topk_batches))
    active: list[tuple[int, int, qai_hub.InferenceJob]] = []  # (bi, valid, job)
    results_by_batch: dict[int, np.ndarray] = {}

    def _submit_one(bi: int):
        start = bi * topk_batch
        end = min(total_images, start + topk_batch)
        chunk = image_embs[start:end]
        valid = int(chunk.shape[0])
        if valid < topk_batch:
            pad = np.zeros((topk_batch - valid, chunk.shape[1]), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad], axis=0)

        topk_dataset = qai_hub.upload_dataset({"image_embs": [chunk], "text_embs": [text_embs]})
        job = qai_hub.submit_inference_job(
            model=topk_compiled,
            device=target_device,
            inputs=topk_dataset,
            name=f"{prefix}topk-cosine {bi + 1}/{num_topk_batches}",
        )
        active.append((bi, valid, job))

    while pending or active:
        while pending and len(active) < topk_max_inflight:
            _submit_one(pending.pop(0))

        progressed = False
        still_active: list[tuple[int, int, qai_hub.InferenceJob]] = []
        for bi, valid, job in active:
            st = job.get_status()
            if st.running or st.pending:
                still_active.append((bi, valid, job))
                continue
            out = first_output(job)[0]
            results_by_batch[bi] = out[:valid]
            progressed = True
        active = still_active

        if not progressed and active:
            time.sleep(2)

    parts = [results_by_batch[i] for i in range(num_topk_batches)]
    topk_indices = np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), dtype=np.int64)

    # ── Recall@10 ───────────────────────────────────────────────────────────
    N = image_embs.shape[0]
    recalls = []
    for i in range(N):
        gt = set(range(i * CAPTIONS_PER_IMAGE, (i + 1) * CAPTIONS_PER_IMAGE))
        recalls.append(len(gt & set(topk_indices[i].tolist())) / CAPTIONS_PER_IMAGE)
    return float(np.mean(recalls))


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    missing = []
    if args.image_compiled_id is None:
        missing.append("--image-compiled-id")
    if args.topk_compiled_id is None:
        args.topk_compiled_id = (JOB_IDS.data.get("topk") or {}).get("compiled_id")
        if args.topk_compiled_id is None:
            missing.append("--topk-compiled-id")
    # Resolve batched image dataset ids.
    if args.image_dataset_ids:
        image_dataset_ids = [s.strip() for s in args.image_dataset_ids.split(",") if s.strip()]
    else:
        image_dataset_ids = (JOB_IDS.data.get("image") or {}).get("dataset_ids") or []

    if not image_dataset_ids:
        missing.append("--image-dataset-ids")
    if args.text_compiled_id is None:
        missing.append("--text-compiled-id")
    if args.text_dataset_id is None:
        missing.append("--text-dataset-id")
    if missing:
        parser.error(
            "Missing required parameters (and no defaults in job_ids.json): "
            + ", ".join(missing)
            + "."
        )

    recall_at_10 = run_inference(
        device=args.device,
        image_compiled_id=args.image_compiled_id,
        topk_compiled_id=args.topk_compiled_id,
        image_dataset_ids=image_dataset_ids,
        text_compiled_id=args.text_compiled_id,
        text_dataset_id=args.text_dataset_id,
        job_name_prefix=args.job_name_prefix,
        max_inference_inflight=int(MAX_INFERENCE_INFLIGHT),
        topk_images_per_batch=args.topk_images_per_batch,
    )
    print(f"Recall@10: {recall_at_10:.4f}  ({recall_at_10 * 100:.2f}%)")


if __name__ == "__main__":
    main()
