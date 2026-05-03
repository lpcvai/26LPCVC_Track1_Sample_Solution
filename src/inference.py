import argparse

import numpy as np
import qai_hub

from utils import CAPTIONS_PER_IMAGE, JOB_IDS


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
                        help="Compile job ID for the text encoder (required for --topk cosine). "
                             "If omitted, uses job_ids.json.")
    parser.add_argument("--topk-compiled-id", default=JOB_IDS["topk", "compiled_id"],
                        help="Compile job ID for the top-k model (topk_retrieval or faiss_index). "
                             "If omitted, uses job_ids.json.")
    parser.add_argument("--image-dataset-id", default=JOB_IDS["image", "dataset_id"],
                        help="QAI Hub dataset ID for images. If omitted, uses job_ids.json.")
    parser.add_argument("--text-dataset-id", default=JOB_IDS["text", "dataset_id"],
                        help="QAI Hub dataset ID for texts (required for --topk cosine). If omitted, uses job_ids.json.")
    parser.add_argument("--topk", choices=["cosine", "faiss"], default="cosine",
                        help="Top-k method used when the model was compiled")
    parser.add_argument("--faiss-compute-unit", choices=["all", "npu", "gpu", "cpu"], default="all",
                        help="Compute unit for the FAISS inference job (only applies with --topk faiss)")
    parser.add_argument("--device", default="XR2 Gen 2 (Proxy)", help="QAI Hub target device")
    parser.add_argument("--job-name-prefix", default="",
                        help="Optional prefix for QAI Hub job names to make jobs easier to identify in the UI.")
    return parser


def run_inference(
    *,
    device: str,
    topk: str,
    faiss_compute_unit: str,
    image_compiled_id: str,
    topk_compiled_id: str,
    image_dataset_id: str,
    text_compiled_id: str | None = None,
    text_dataset_id: str | None = None,
    job_name_prefix: str = "",
):
    missing = []
    if image_compiled_id is None:
        missing.append("image_compiled_id")
    if topk_compiled_id is None:
        missing.append("topk_compiled_id")
    if image_dataset_id is None:
        missing.append("image_dataset_id")
    if topk == "cosine":
        if text_compiled_id is None:
            missing.append("text_compiled_id")
        if text_dataset_id is None:
            missing.append("text_dataset_id")
    if missing:
        raise ValueError(f"Missing required inputs: {missing}")

    target_device = qai_hub.Device(device)
    image_dataset = qai_hub.get_dataset(image_dataset_id)
    prefix = (job_name_prefix.strip() + " ") if job_name_prefix and job_name_prefix.strip() else ""

    # Retrieve compiled models from their job IDs
    image_compiled = qai_hub.get_job(image_compiled_id).get_target_model()
    topk_compiled = qai_hub.get_job(topk_compiled_id).get_target_model()

    # ── Encoder inference ───────────────────────────────────────────────────
    print("Submitting image encoder inference...")
    image_inf_job = qai_hub.submit_inference_job(
        model=image_compiled,
        device=target_device,
        inputs=image_dataset,
        name=f"{prefix}image-encoder",
    )

    if topk == "cosine":
        text_compiled = qai_hub.get_job(text_compiled_id).get_target_model()
        text_dataset = qai_hub.get_dataset(text_dataset_id)
        print("Submitting text encoder inference...")
        text_inf_job = qai_hub.submit_inference_job(
            model=text_compiled,
            device=target_device,
            inputs=text_dataset,
            name=f"{prefix}text-encoder",
        )
        text_embs = np.concatenate(first_output(text_inf_job), axis=0)
        text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)

    image_embs = np.concatenate(first_output(image_inf_job), axis=0)
    image_embs = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)

    # ── Top-k inference ─────────────────────────────────────────────────────
    if topk == "cosine":
        topk_dataset = qai_hub.upload_dataset({
            "image_embs": [image_embs],
            "text_embs": [text_embs],
        })
        topk_options = None
    else:
        # faiss: text embeddings are baked into the compiled model; only image
        # embeddings are needed as runtime input
        topk_dataset = qai_hub.upload_dataset({"image_embs": [image_embs]})
        topk_options = f"--compute_unit {faiss_compute_unit}"

    print("Submitting top-k inference...")
    # qai_hub expects options to be a string; passing None can break in some versions.
    if topk_options is None:
        topk_inf_job = qai_hub.submit_inference_job(
            model=topk_compiled,
            device=target_device,
            inputs=topk_dataset,
            name=f"{prefix}topk-cosine",
        )
    else:
        topk_inf_job = qai_hub.submit_inference_job(
            model=topk_compiled,
            device=target_device,
            inputs=topk_dataset,
            options=topk_options,
            name=f"{prefix}topk-faiss",
        )
    topk_indices = first_output(topk_inf_job)[0]  # [N, K]

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
        missing.append("--topk-compiled-id")
    if args.image_dataset_id is None:
        missing.append("--image-dataset-id")
    if args.topk == "cosine":
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
        topk=args.topk,
        faiss_compute_unit=args.faiss_compute_unit,
        image_compiled_id=args.image_compiled_id,
        topk_compiled_id=args.topk_compiled_id,
        image_dataset_id=args.image_dataset_id,
        text_compiled_id=args.text_compiled_id,
        text_dataset_id=args.text_dataset_id,
        job_name_prefix=args.job_name_prefix,
    )
    print(f"Recall@10: {recall_at_10:.4f}  ({recall_at_10 * 100:.2f}%)")


if __name__ == "__main__":
    main()
