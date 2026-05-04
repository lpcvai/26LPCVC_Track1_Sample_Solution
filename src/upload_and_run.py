import argparse

import qai_hub

import compile_and_profile
from inference import run_inference
from upload_dataset import upload_datasets
from utils import MODELS, JOB_IDS, IMAGES_PER_BATCH, NUM_IMAGE_SAMPLES, CAPTIONS_PER_IMAGE
from utils import MAX_INFERENCE_INFLIGHT


def build_arg_parser():
    # Parent parser provides compile flags (model/all, device, topk, quantize, etc.).
    cap_parser = compile_and_profile.build_arg_parser()
    parser = argparse.ArgumentParser(
        description="Upload datasets to QAI Hub, then compile + run inference (combined runner).",
        parents=[cap_parser],
        conflict_handler="resolve",
    )
    parser.add_argument("--upload-model", choices=MODELS.keys(), default="MobileCLIP2-S0",
                        help="Model name used to build preprocess/tokenizer for dataset upload.")
    parser.add_argument("--no-upload", action="store_true",
                        help="Do not upload datasets; error if dataset IDs are missing.")
    parser.add_argument("--image-dataset-ids", default=None,
                        help="Comma-separated list of image dataset ids (batched). Overrides --image-dataset-id/job_ids.json.")
    parser.add_argument("--images-per-batch", type=int, default=None,
                        help=f"Images per uploaded dataset batch when uploading (default from utils.py: {IMAGES_PER_BATCH}).")
    parser.add_argument("--num-images", type=int, default=None,
                        help=f"Number of images to evaluate/upload (default from utils.py: {NUM_IMAGE_SAMPLES}).")
    parser.add_argument("--max-inflight", type=int, default=None,
                        help=f"Max number of inference jobs to keep in-flight when batching (default: {MAX_INFERENCE_INFLIGHT}).")
    return parser


def _resolve_dataset_ids(args):
    # If args were defaulted from an old JOB_IDS snapshot at import time, refresh them.
    image_dataset_ids = None
    if getattr(args, "image_dataset_ids", None):
        image_dataset_ids = [s.strip() for s in args.image_dataset_ids.split(",") if s.strip()]
    if not image_dataset_ids:
        image_dataset_ids = (JOB_IDS.data.get("image") or {}).get("dataset_ids") or None
    if not image_dataset_ids:
        if args.image_dataset_id is None:
            args.image_dataset_id = JOB_IDS["image", "dataset_id"]
        image_dataset_ids = [args.image_dataset_id] if args.image_dataset_id is not None else []

    if args.text_dataset_id is None:
        args.text_dataset_id = JOB_IDS["text", "dataset_id"]

    if not image_dataset_ids or args.text_dataset_id is None:
        if args.no_upload:
            raise SystemExit(
                "Dataset IDs are missing (CLI and job_ids.json). Re-run without --no-upload to upload them, "
                "or pass --image-dataset-id/--text-dataset-id."
            )
        image_dataset_ids, text_dataset_id = upload_datasets(
            args.upload_model,
            images_per_batch=int(args.images_per_batch or IMAGES_PER_BATCH),
            num_image_samples=int(args.num_images or NUM_IMAGE_SAMPLES),
            captions_per_image=CAPTIONS_PER_IMAGE,
        )
        args.image_dataset_id = image_dataset_ids[0] if image_dataset_ids else None
        args.image_dataset_ids = ",".join(image_dataset_ids)
        args.text_dataset_id = text_dataset_id

    return image_dataset_ids, args.text_dataset_id


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if getattr(args, "all", False):
        raise SystemExit("upload_and_run.py only supports a single --model. Use experiments.py for --all.")

    image_dataset_ids, text_dataset_id = _resolve_dataset_ids(args)

    compile_jobs = compile_and_profile.run_pipeline(args)
    cj = compile_jobs.get(args.model)
    if cj is None:
        raise SystemExit("No compile jobs were submitted (missing ONNX artifacts?).")

    topk_compiled_id = cj.get("topk_compiled_id")
    if args.topk == "faiss":
        # Build/compile the FAISS index topk model (requires text encoder compiled model + text dataset).
        target_device = qai_hub.Device(args.device)
        text_dataset = qai_hub.get_dataset(args.text_dataset_id)
        text_compiled = qai_hub.get_job(cj["text_compiled_id"]).get_target_model()
        faiss = compile_and_profile.compile_faiss_index_model(
            target_device=target_device,
            text_compiled=text_compiled,
            text_dataset=text_dataset,
            onnx_dir=cj["onnx_dir"],
            faiss_compute_unit=args.faiss_compute_unit,
            images_per_batch=int(args.num_images or NUM_IMAGE_SAMPLES),
        )
        if faiss is None:
            raise SystemExit("Failed to compile FAISS index model.")
        topk_compiled_id = faiss["topk_compiled_id"]

    if topk_compiled_id is None:
        raise SystemExit("Missing topk compiled id. Did the topk compile step run successfully?")

    recall_at_10 = run_inference(
        device=args.device,
        topk=args.topk,
        faiss_compute_unit=args.faiss_compute_unit,
        image_compiled_id=cj["image_compiled_id"],
        topk_compiled_id=topk_compiled_id,
        image_dataset_ids=image_dataset_ids,
        text_compiled_id=cj["text_compiled_id"],
        text_dataset_id=text_dataset_id,
        max_inference_inflight=int(args.max_inflight or MAX_INFERENCE_INFLIGHT),
    )
    print(f"Recall@10: {recall_at_10:.4f}  ({recall_at_10 * 100:.2f}%)")


if __name__ == "__main__":
    main()
