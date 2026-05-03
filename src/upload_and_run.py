import argparse

import qai_hub

import compile_and_profile
from inference import run_inference
from upload_dataset import upload_datasets
from utils import MODELS, JOB_IDS


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
    return parser


def _resolve_dataset_ids(args):
    # If args were defaulted from an old JOB_IDS snapshot at import time, refresh them.
    if args.image_dataset_id is None:
        args.image_dataset_id = JOB_IDS["image", "dataset_id"]
    if args.text_dataset_id is None:
        args.text_dataset_id = JOB_IDS["text", "dataset_id"]

    if args.image_dataset_id is None or args.text_dataset_id is None:
        if args.no_upload:
            raise SystemExit(
                "Dataset IDs are missing (CLI and job_ids.json). Re-run without --no-upload to upload them, "
                "or pass --image-dataset-id/--text-dataset-id."
            )
        image_dataset_id, text_dataset_id = upload_datasets(args.upload_model)
        args.image_dataset_id = image_dataset_id
        args.text_dataset_id = text_dataset_id


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if getattr(args, "all", False):
        raise SystemExit("upload_and_run.py only supports a single --model. Use experiments.py for --all.")

    _resolve_dataset_ids(args)

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
        image_dataset_id=args.image_dataset_id,
        text_compiled_id=cj["text_compiled_id"],
        text_dataset_id=args.text_dataset_id,
    )
    print(f"Recall@10: {recall_at_10:.4f}  ({recall_at_10 * 100:.2f}%)")


if __name__ == "__main__":
    main()

