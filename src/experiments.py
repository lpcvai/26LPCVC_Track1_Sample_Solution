import argparse
import copy
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import compile_and_profile
import qai_hub
from utils import MODELS, RESULTS_PATH, JOB_IDS


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run compile+inference benchmarks across models/topk modes.")
    parser.add_argument("--models", nargs="*", default=None, choices=list(MODELS.keys()),
                        help="Models to benchmark. If omitted, benchmarks all models.")
    parser.add_argument("--topk", choices=["cosine", "faiss", "both"], default="both",
                        help="Which topk mode(s) to benchmark.")
    parser.add_argument("--device", default="XR2 Gen 2 (Proxy)", help="QAI Hub target device")
    parser.add_argument("--faiss-compute-unit", choices=["all", "npu", "gpu", "cpu"], default="all",
                        help="Compute unit for FAISS compile and inference jobs (only applies with --topk faiss)")
    parser.add_argument("--quantize", action="store_true", help="Apply post-training quantization during compilation")
    parser.add_argument("--quantize-type", default="int16", choices=["w4a8", "w8a16", "w4a16", "int8", "int16"],
                        help="Quantization type (only applies with --quantize)")
    parser.add_argument("--image-calibration-id", default=None,
                        help="QAI Hub dataset ID for image calibration data (required for --quantize)")
    parser.add_argument("--text-calibration-id", default=None,
                        help="QAI Hub dataset ID for text calibration data (required for --quantize)")
    parser.add_argument("--image-dataset-id", default=None,
                        help="Override image dataset id (otherwise uses job_ids.json)")
    parser.add_argument("--text-dataset-id", default=None,
                        help="Override text dataset id (otherwise uses job_ids.json)")
    parser.add_argument("--output", default=None, help="Optional path to write a JSON summary.")
    return parser


def _resolve_dataset_ids(args):
    image_dataset_id = args.image_dataset_id or JOB_IDS["image", "dataset_id"]
    text_dataset_id = args.text_dataset_id or JOB_IDS["text", "dataset_id"]
    if image_dataset_id is None or text_dataset_id is None:
        raise SystemExit(
            "Missing dataset IDs. Provide --image-dataset-id/--text-dataset-id or run upload_dataset.py first "
            "(which updates job_ids.json)."
        )
    return image_dataset_id, text_dataset_id


def _iter_topk_modes(topk_arg: str):
    if topk_arg == "both":
        return ["cosine", "faiss"]
    return [topk_arg]


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    image_dataset_id, text_dataset_id = _resolve_dataset_ids(args)
    models = args.models if args.models else list(MODELS.keys())
    topk_modes = _iter_topk_modes(args.topk)

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": args.device,
        "image_dataset_id": image_dataset_id,
        "text_dataset_id": text_dataset_id,
        "runs": [],
    }

    # Phase 1: submit all compile jobs first so Hub can compile them concurrently.
    planned_runs = []
    for model_name in models:
        for topk_mode in topk_modes:
            print(f"\n{'=' * 70}")
            print(f"Experiment: model={model_name} topk={topk_mode} quantize={args.quantize}")
            print(f"{'=' * 70}")

            run_args = argparse.Namespace(
                model=model_name,
                all=False,
                image_dataset_id=image_dataset_id,
                text_dataset_id=text_dataset_id,
                device=args.device,
                topk=topk_mode,
                faiss_compute_unit=args.faiss_compute_unit,
                profile=False,
                quantize=args.quantize,
                quantize_type=args.quantize_type,
                image_calibration_id=args.image_calibration_id,
                text_calibration_id=args.text_calibration_id,
            )

            # NOTE: Even though inference must wait for compile completion, submitting all compile jobs
            # up-front avoids serializing compiles across runs.
            try:
                compile_jobs = compile_and_profile.run_pipeline(run_args) or {}
            except Exception as e:
                results["runs"].append({
                    "model": model_name,
                    "topk": topk_mode,
                    "quantize": bool(args.quantize),
                    "quantize_type": args.quantize_type,
                    "faiss_compute_unit": args.faiss_compute_unit,
                    "recall_at_10": None,
                    "job_ids_snapshot": copy.deepcopy(JOB_IDS.data),
                    "error": f"compile_submit_failed: {type(e).__name__}: {e}",
                })
                print(f"  ERROR: compile submit failed: {type(e).__name__}: {e}")
                continue

            cj = compile_jobs.get(model_name)
            if cj is None:
                results["runs"].append({
                    "model": model_name,
                    "topk": topk_mode,
                    "quantize": bool(args.quantize),
                    "quantize_type": args.quantize_type,
                    "faiss_compute_unit": args.faiss_compute_unit,
                    "recall_at_10": None,
                    "job_ids_snapshot": copy.deepcopy(JOB_IDS.data),
                    "error": "compile_skipped_or_failed",
                })
                continue

            planned_runs.append({
                "model": model_name,
                "topk": topk_mode,
                "compile_job_ids": cj,
            })

    # Phase 2: run inference for the runs that support it here.
    from inference import run_inference
    target_device = qai_hub.Device(args.device)

    def _run_one(run_dict):
        cj = run_dict["compile_job_ids"]
        topk_mode = run_dict["topk"]
        text_dataset = qai_hub.get_dataset(text_dataset_id)

        if topk_mode == "faiss":
            # FAISS top-k: build an index model by running text inference, baking embeddings,
            # exporting an ONNX wrapper, then compiling that wrapper as the "topk model".
            text_compiled = qai_hub.get_job(cj["text_compiled_id"]).get_target_model()
            faiss_compile = compile_and_profile.compile_faiss_index_model(
                target_device=target_device,
                text_compiled=text_compiled,
                text_dataset=text_dataset,
                onnx_dir=cj["onnx_dir"],
                faiss_compute_unit=args.faiss_compute_unit,
            )
            if not faiss_compile or not faiss_compile.get("topk_compiled_id"):
                raise RuntimeError("FAISS index compile did not return a topk_compiled_id")
            topk_compiled_id = faiss_compile["topk_compiled_id"]

            recall = run_inference(
                device=args.device,
                topk="faiss",
                faiss_compute_unit=args.faiss_compute_unit,
                image_compiled_id=cj["image_compiled_id"],
                topk_compiled_id=topk_compiled_id,
                image_dataset_id=image_dataset_id,
            )
            return recall

        # cosine
        recall = run_inference(
            device=args.device,
            topk="cosine",
            faiss_compute_unit=args.faiss_compute_unit,
            image_compiled_id=cj["image_compiled_id"],
            topk_compiled_id=cj.get("topk_compiled_id"),
            image_dataset_id=image_dataset_id,
            text_compiled_id=cj["text_compiled_id"],
            text_dataset_id=text_dataset_id,
        )
        return recall

    # A modest worker count avoids hammering the Hub API/client while still preventing
    # "wait for compile then infer" from becoming serialized across many runs.
    max_workers = min(4, max(1, len(planned_runs)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_run = {ex.submit(_run_one, run): run for run in planned_runs}
        for fut in as_completed(future_to_run):
            run = future_to_run[fut]
            model_name = run["model"]
            topk_mode = run["topk"]
            try:
                recall_at_10 = fut.result()
            except Exception as e:
                results["runs"].append({
                    "model": model_name,
                    "topk": topk_mode,
                    "quantize": bool(args.quantize),
                    "quantize_type": args.quantize_type,
                    "faiss_compute_unit": args.faiss_compute_unit,
                    "recall_at_10": None,
                    "job_ids_snapshot": copy.deepcopy(JOB_IDS.data),
                    "error": f"{type(e).__name__}: {e}",
                })
                print(f"  ERROR: {model_name} {topk_mode}: {type(e).__name__}: {e}")
                continue

            results["runs"].append({
                "model": model_name,
                "topk": topk_mode,
                "quantize": bool(args.quantize),
                "quantize_type": args.quantize_type,
                "faiss_compute_unit": args.faiss_compute_unit,
                "recall_at_10": recall_at_10,
                "job_ids_snapshot": copy.deepcopy(JOB_IDS.data),
            })

    out_path = args.output
    if out_path is None:
        os.makedirs(os.path.join(RESULTS_PATH, "experiments"), exist_ok=True)
        out_path = os.path.join(
            RESULTS_PATH,
            "experiments",
            f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote experiment summary to: {out_path}")


if __name__ == "__main__":
    main()
