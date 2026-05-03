import argparse
import copy
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import compile_and_profile
import numpy as np
import qai_hub
from utils import MODELS, RESULTS_PATH, JOB_IDS, CAPTIONS_PER_IMAGE


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

def _job_ids_snapshot_for_run(*, cj: dict, image_dataset_id: str, text_dataset_id: str, topk_compiled_id: str | None):
    """
    Build a stable per-run snapshot of the exact compile job IDs used.

    Don't read from JOB_IDS here: experiments submits many compiles and JOB_IDS is a
    mutable global backed by job_ids.json, so it will reflect whichever compile ran last.
    """
    return {
        "text": {
            "compiled_id": cj.get("text_compiled_id"),
            "dataset_id": text_dataset_id,
        },
        "image": {
            "compiled_id": cj.get("image_compiled_id"),
            "dataset_id": image_dataset_id,
        },
        "topk": {
            "compiled_id": topk_compiled_id,
        },
    }


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

    # Phase 1: submit compile jobs once per model (shared across cosine/faiss modes).
    # Previously we compiled per (model, topk_mode), which doubled compile jobs and caused
    # extra encoder inference work downstream.
    compile_by_model: dict[str, dict] = {}
    need_cosine = "cosine" in topk_modes
    need_faiss = "faiss" in topk_modes
    compile_topk_mode = "cosine" if need_cosine else "faiss"

    for model_name in models:
        print(f"\n{'=' * 70}")
        print(f"Compile: model={model_name} topk={compile_topk_mode} quantize={args.quantize}")
        print(f"{'=' * 70}")

        run_args = argparse.Namespace(
            model=model_name,
            all=False,
            image_dataset_id=image_dataset_id,
            text_dataset_id=text_dataset_id,
            device=args.device,
            topk=compile_topk_mode,
            faiss_compute_unit=args.faiss_compute_unit,
            profile=False,
            quantize=args.quantize,
            quantize_type=args.quantize_type,
            image_calibration_id=args.image_calibration_id,
            text_calibration_id=args.text_calibration_id,
        )

        try:
            # Avoid mutating job_ids.json during experiments. We want each run to carry
            # its own compile job IDs, not "whatever was last written".
            compile_jobs = compile_and_profile.run_pipeline(run_args, persist_job_ids=False) or {}
        except Exception as e:
            for topk_mode in topk_modes:
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
            for topk_mode in topk_modes:
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

        compile_by_model[model_name] = cj

    # Phase 2: run inference for the runs that support it here.
    from inference import first_output as first_output_inference
    target_device = qai_hub.Device(args.device)

    def _recall_from_topk_indices(*, topk_indices: np.ndarray, num_images: int) -> float:
        recalls = []
        for i in range(num_images):
            gt = set(range(i * CAPTIONS_PER_IMAGE, (i + 1) * CAPTIONS_PER_IMAGE))
            recalls.append(len(gt & set(topk_indices[i].tolist())) / CAPTIONS_PER_IMAGE)
        return float(np.mean(recalls))

    def _eval_model(model_name: str):
        """
        Evaluate one model. This is intentionally "one model at a time" to avoid exploding
        the number of concurrent Hub jobs.

        Optimization: when both cosine+faiss are requested, run image encoder inference
        once and (if needed) text encoder inference once, then reuse embeddings for both
        top-k modes.
        """
        cj = compile_by_model[model_name]

        image_dataset = qai_hub.get_dataset(image_dataset_id)
        text_dataset = qai_hub.get_dataset(text_dataset_id) if (need_cosine or need_faiss) else None

        image_compiled = qai_hub.get_job(cj["image_compiled_id"]).get_target_model()
        text_compiled = qai_hub.get_job(cj["text_compiled_id"]).get_target_model() if (need_cosine or need_faiss) else None

        # Submit encoder inference once.
        print(f"Submitting image encoder inference for {model_name}...")
        image_inf_job = qai_hub.submit_inference_job(model=image_compiled, device=target_device, inputs=image_dataset)
        text_embs = None
        if need_cosine or need_faiss:
            print(f"Submitting text encoder inference for {model_name}...")
            text_inf_job = qai_hub.submit_inference_job(model=text_compiled, device=target_device, inputs=text_dataset)
            text_embs = np.concatenate(first_output_inference(text_inf_job), axis=0)
            text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)

        image_embs = np.concatenate(first_output_inference(image_inf_job), axis=0)
        image_embs = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)

        num_images = image_embs.shape[0]

        # Cosine top-k
        if need_cosine:
            topk_cosine_id = cj.get("topk_compiled_id")
            topk_compiled = qai_hub.get_job(topk_cosine_id).get_target_model()
            topk_dataset = qai_hub.upload_dataset({"image_embs": [image_embs], "text_embs": [text_embs]})
            print(f"Submitting cosine top-k inference for {model_name}...")
            topk_inf_job = qai_hub.submit_inference_job(model=topk_compiled, device=target_device, inputs=topk_dataset)
            topk_indices = first_output_inference(topk_inf_job)[0]
            recall_at_10 = _recall_from_topk_indices(topk_indices=topk_indices, num_images=num_images)
            results["runs"].append({
                "model": model_name,
                "topk": "cosine",
                "quantize": bool(args.quantize),
                "quantize_type": args.quantize_type,
                "faiss_compute_unit": args.faiss_compute_unit,
                "recall_at_10": recall_at_10,
                "job_ids_snapshot": _job_ids_snapshot_for_run(
                    cj=cj,
                    image_dataset_id=image_dataset_id,
                    text_dataset_id=text_dataset_id,
                    topk_compiled_id=topk_cosine_id,
                ),
            })

        # FAISS top-k
        if need_faiss:
            if text_embs is None:
                raise RuntimeError("Internal error: missing text embeddings for FAISS evaluation")

            # Build+compile FAISS index model using the already-computed text embeddings.
            # This avoids a second text encoder inference job when both modes are requested.
            faiss_compile = compile_and_profile.compile_faiss_index_model(
                target_device=target_device,
                onnx_dir=cj["onnx_dir"],
                faiss_compute_unit=args.faiss_compute_unit,
                persist_job_ids=False,
                text_embs=text_embs,
            )
            if not faiss_compile or not faiss_compile.get("topk_compiled_id"):
                raise RuntimeError("FAISS index compile did not return a topk_compiled_id")
            topk_faiss_id = faiss_compile["topk_compiled_id"]

            topk_compiled = qai_hub.get_job(topk_faiss_id).get_target_model()
            topk_dataset = qai_hub.upload_dataset({"image_embs": [image_embs]})
            print(f"Submitting FAISS top-k inference for {model_name}...")
            topk_inf_job = qai_hub.submit_inference_job(
                model=topk_compiled,
                device=target_device,
                inputs=topk_dataset,
                options=f"--compute_unit {args.faiss_compute_unit}",
            )
            topk_indices = first_output_inference(topk_inf_job)[0]
            recall_at_10 = _recall_from_topk_indices(topk_indices=topk_indices, num_images=num_images)
            results["runs"].append({
                "model": model_name,
                "topk": "faiss",
                "quantize": bool(args.quantize),
                "quantize_type": args.quantize_type,
                "faiss_compute_unit": args.faiss_compute_unit,
                "recall_at_10": recall_at_10,
                "job_ids_snapshot": _job_ids_snapshot_for_run(
                    cj=cj,
                    image_dataset_id=image_dataset_id,
                    text_dataset_id=text_dataset_id,
                    topk_compiled_id=topk_faiss_id,
                ),
            })

    # Evaluate models with limited concurrency to avoid creating a large number of Hub jobs.
    planned_models = [m for m in models if m in compile_by_model]
    max_workers = min(2, max(1, len(planned_models)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_model = {ex.submit(_eval_model, m): m for m in planned_models}
        for fut in as_completed(future_to_model):
            model_name = future_to_model[fut]
            try:
                fut.result()
            except Exception as e:
                cj = compile_by_model.get(model_name) or {}
                for topk_mode in topk_modes:
                    results["runs"].append({
                        "model": model_name,
                        "topk": topk_mode,
                        "quantize": bool(args.quantize),
                        "quantize_type": args.quantize_type,
                        "faiss_compute_unit": args.faiss_compute_unit,
                        "recall_at_10": None,
                        "job_ids_snapshot": _job_ids_snapshot_for_run(
                            cj=cj,
                            image_dataset_id=image_dataset_id,
                            text_dataset_id=text_dataset_id,
                            topk_compiled_id=cj.get("topk_compiled_id"),
                        ),
                        "error": f"{type(e).__name__}: {e}",
                    })
                print(f"  ERROR: {model_name}: {type(e).__name__}: {e}")

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
