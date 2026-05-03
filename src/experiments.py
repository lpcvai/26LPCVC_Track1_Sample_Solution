import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import compile_and_profile
import numpy as np
import qai_hub
import time
import qai_hub.public_rest_api as hub_api

from utils import (
    MODELS,
    RESULTS_PATH,
    CAPTIONS_PER_IMAGE,
    IMAGES_PER_BATCH,
    MAX_INFERENCE_INFLIGHT,
    NUM_IMAGE_SAMPLES,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run compile+inference benchmarks across models/topk modes.")
    parser.add_argument("--models", nargs="*", default=None, choices=list(MODELS.keys()),
                        help="Models to benchmark. If omitted, benchmarks all models.")
    parser.add_argument("--topk", choices=["cosine", "faiss", "both"], default="both",
                        help="Which topk mode(s) to benchmark.")
    parser.add_argument("--num-images", type=int, default=None,
                        help=f"Number of images to evaluate (default from utils.py: {NUM_IMAGE_SAMPLES}).")
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
    parser.add_argument("--images-per-batch", type=int, default=None,
                        help=f"Images per uploaded dataset and per image-encoder inference job (default: {IMAGES_PER_BATCH}).")
    parser.add_argument("--max-inflight", type=int, default=None,
                        help=f"Max number of inference jobs to keep in-flight when batching (default: {MAX_INFERENCE_INFLIGHT}).")
    parser.add_argument("--export-onnx", action="store_true", default=True,
                        help="Export ONNX artifacts for each model before compiling (default: enabled).")
    parser.add_argument("--no-export-onnx", action="store_false", dest="export_onnx",
                        help="Skip ONNX export step (assume results/onnx/<model>/ already exists).")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="Max number of models to evaluate concurrently (each model evaluation submits multiple jobs).")
    parser.add_argument("--job-name-prefix", default="",
                        help="Optional prefix for QAI Hub inference job names to make jobs easier to identify in the UI.")
    parser.add_argument("--output", default=None, help="Optional path to write a JSON summary.")
    return parser


def _dataset_profile(model_name: str) -> str:
    """
    Produce a small signature for the model's preprocess+tokenizer, so models with identical
    input requirements can reuse the same uploaded datasets.
    """
    # Import lazily so importing experiments.py doesn't require deps unless executed.
    import open_clip  # type: ignore
    import torchvision.transforms as T  # type: ignore

    pretrained = MODELS[model_name]
    # Force preprocessing to output 224x224 to match our ONNX export + Hub compile input specs.
    _, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, force_image_size=224)
    tokenizer = open_clip.get_tokenizer(model_name)

    mean = None
    std = None
    transforms = getattr(preprocess, "transforms", None) or []
    for tr in transforms:
        if isinstance(tr, T.Normalize):
            mean = tuple(float(x) for x in tr.mean)
            std = tuple(float(x) for x in tr.std)
            break

    tok = tokenizer(["a test"])
    shape = tuple(int(x) for x in getattr(tok, "shape", ()))
    dtype = str(getattr(tok, "dtype", "unknown"))
    return f"mean={mean},std={std},tok_shape={shape},tok_dtype={dtype}"


def _upload_datasets_for_models(models: list[str], *, images_per_batch: int, num_images: int):
    """
    Upload datasets per distinct (preprocess, tokenizer) profile, then map each model -> datasets.
    """
    from upload_dataset import upload_datasets

    profile_to_datasets: dict[str, dict] = {}
    model_to_datasets: dict[str, dict] = {}
    for model_name in models:
        profile = _dataset_profile(model_name)
        if profile not in profile_to_datasets:
            print(f"\n{'=' * 70}")
            print(f"Uploading datasets for: {model_name}")
            print(f"Dataset profile: {profile}")
            print(f"{'=' * 70}")
            image_dataset_ids, text_dataset_id = upload_datasets(
                model_name,
                images_per_batch=images_per_batch,
                persist_job_ids=False,
                num_image_samples=num_images,
                captions_per_image=CAPTIONS_PER_IMAGE,
            )
            profile_to_datasets[profile] = {
                "image_dataset_ids": image_dataset_ids,
                "text_dataset_id": text_dataset_id,
                "representative_model": model_name,
            }
        ds = profile_to_datasets[profile]
        model_to_datasets[model_name] = {
            "profile": profile,
            "image_dataset_ids": ds["image_dataset_ids"],
            "text_dataset_id": ds["text_dataset_id"],
        }

    return model_to_datasets, profile_to_datasets


def _iter_topk_modes(topk_arg: str):
    if topk_arg == "both":
        return ["cosine", "faiss"]
    return [topk_arg]

def _snapshot_for_run(*, cj: dict, image_dataset_ids: list[str], text_dataset_id: str, topk_compiled_id: str | None):
    """
    Build a stable per-run snapshot of the exact compile job IDs used.

    Record the exact compile job IDs and dataset IDs used for this run.
    """
    return {
        "text": {
            "compiled_id": cj.get("text_compiled_id"),
            "dataset_id": text_dataset_id,
        },
        "image": {
            "compiled_id": cj.get("image_compiled_id"),
            "dataset_ids": image_dataset_ids,
        },
        "topk": {
            "compiled_id": topk_compiled_id,
        },
    }


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    models = args.models if args.models else list(MODELS.keys())
    num_images = int(args.num_images or NUM_IMAGE_SAMPLES)
    num_text = num_images * CAPTIONS_PER_IMAGE
    images_per_batch = int(args.images_per_batch or IMAGES_PER_BATCH)
    max_inflight = int(args.max_inflight or MAX_INFERENCE_INFLIGHT)
    topk_modes = _iter_topk_modes(args.topk)

    model_to_datasets, profile_to_datasets = _upload_datasets_for_models(
        models,
        images_per_batch=images_per_batch,
        num_images=num_images,
    )

    if args.export_onnx:
        # Ensure ONNX artifacts match current NUM_IMAGE_SAMPLES / expected topk batch size.
        # We force image_size=224 (repo convention) and top-k batch to NUM_IMAGE_SAMPLES
        # so one-shot topk works without shape mismatches.
        for model_name in models:
            print(f"\n{'=' * 70}")
            print(f"Export ONNX: {model_name}")
            print(f"{'=' * 70}")
            subprocess.run(
                [
                    sys.executable,
                    "src/export_onnx.py",
                    "--model",
                    model_name,
                    "--image-size",
                    "224",
                    "--num-images",
                    str(num_images),
                    "--images-per-batch",
                    str(num_images),
                ],
                check=True,
            )

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": args.device,
        "num_image_samples": num_images,
        "num_text_samples": num_text,
        "images_per_batch": images_per_batch,
        "dataset_profiles": profile_to_datasets,
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
            # compile doesn't need datasets; keep these for compatibility with the compile module signature.
            image_dataset_id=None,
            text_dataset_id=None,
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
                    "datasets": model_to_datasets.get(model_name),
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
                    "datasets": model_to_datasets.get(model_name),
                    "error": "compile_skipped_or_failed",
                })
            continue

        compile_by_model[model_name] = cj

    # Phase 2: run inference for the runs that support it here.
    from inference import first_output as first_output_inference
    target_device = qai_hub.Device(args.device)
    prefix = (args.job_name_prefix.strip() + " ") if args.job_name_prefix and args.job_name_prefix.strip() else ""

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
        ds = model_to_datasets[model_name]
        image_dataset_ids = ds["image_dataset_ids"]
        text_dataset_id = ds["text_dataset_id"]

        text_dataset = qai_hub.get_dataset(text_dataset_id) if (need_cosine or need_faiss) else None

        image_compiled = qai_hub.get_job(cj["image_compiled_id"]).get_target_model()
        text_compiled = qai_hub.get_job(cj["text_compiled_id"]).get_target_model() if (need_cosine or need_faiss) else None

        def _inference_detail(job: qai_hub.InferenceJob):
            """
            Fetch Hub-side timing/memory detail for an inference job.
            Values are typically in microseconds and bytes.
            """
            try:
                res = job._owner._api_call(hub_api.get_job_results, job.job_id)
                detail = res.inference_job_result.detail
                return {
                    "hub_execution_time_us": detail.execution_time,
                    "hub_load_time_us": detail.load_time,
                    "hub_peak_memory_bytes": detail.peak_memory_usage,
                    "hub_warm_load_time_us": getattr(detail, "warm_load_time", None),
                    "hub_cold_load_time_us": getattr(detail, "cold_load_time", None),
                }
            except Exception as e:
                return {"detail_error": f"{type(e).__name__}: {e}"}

        # Submit text encoder inference early so it can run in parallel with image batches.
        text_embs = None
        text_inf_job = None
        text_job_id = None
        text_job_url = None
        text_wall_s = None
        text_submitted_at = None
        text_detail = None
        if need_cosine or need_faiss:
            print(f"Submitting text encoder inference for {model_name}...")
            text_submitted_at = time.time()
            text_name = f"{prefix}{model_name} :: text"
            text_inf_job = qai_hub.submit_inference_job(
                model=text_compiled,
                device=target_device,
                inputs=text_dataset,
                name=text_name,
            )
            text_job_id = text_inf_job.job_id
            text_job_url = text_inf_job.url
            # We'll compute wall time when we actually download outputs.

        # Submit image encoder inference in batches (multiple image datasets) to avoid Hub size limits.
        image_embs_parts = []
        pending = list(enumerate(image_dataset_ids))
        active: list[tuple[int, str, qai_hub.InferenceJob]] = []
        results_by_idx: dict[int, np.ndarray] = {}
        image_jobs: list[dict] = []

        def _submit_one(idx: int, ds_id: str):
            img_ds = qai_hub.get_dataset(ds_id)
            print(f"Submitting image encoder inference for {model_name} (batch {idx + 1}/{len(image_dataset_ids)})...")
            t0 = time.time()
            name = f"{prefix}{model_name} :: image b{idx + 1}/{len(image_dataset_ids)}"
            job = qai_hub.submit_inference_job(
                model=image_compiled,
                device=target_device,
                inputs=img_ds,
                name=name,
            )
            active.append((idx, ds_id, job))
            image_jobs.append({
                "batch_index": idx,
                "dataset_id": ds_id,
                "job_id": job.job_id,
                "url": job.url,
                "name": name,
                "submitted_at_epoch_s": t0,
            })

        max_jobs = max(1, int(max_inflight))
        while pending or active:
            while pending and len(active) < max_jobs:
                idx, ds_id = pending.pop(0)
                _submit_one(idx, ds_id)

            progressed = False

            # Opportunistically capture completed text embeddings.
            if text_inf_job is not None and text_embs is None:
                st = text_inf_job.get_status()
                if not (st.running or st.pending):
                    t_done = time.time()
                    text_embs = np.concatenate(first_output_inference(text_inf_job), axis=0)
                    text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)
                    if text_submitted_at is not None:
                        text_wall_s = t_done - text_submitted_at
                    text_detail = _inference_detail(text_inf_job)
                    progressed = True

            still_active: list[tuple[int, str, qai_hub.InferenceJob]] = []
            for idx, ds_id, job in active:
                st = job.get_status()
                if st.running or st.pending:
                    still_active.append((idx, ds_id, job))
                    continue
                t_done = time.time()
                part = np.concatenate(first_output_inference(job), axis=0)
                part = part / np.linalg.norm(part, axis=1, keepdims=True)
                results_by_idx[idx] = part
                # Fill wall time for this batch job.
                for rec in image_jobs:
                    if rec.get("batch_index") == idx and rec.get("job_id") == job.job_id and "wall_time_s" not in rec:
                        rec["wall_time_s"] = t_done - rec["submitted_at_epoch_s"]
                        rec["hub_detail"] = _inference_detail(job)
                        break
                progressed = True
            active = still_active

            if not progressed and (active or (text_inf_job is not None and text_embs is None)):
                time.sleep(2)

        for i in range(len(image_dataset_ids)):
            image_embs_parts.append(results_by_idx[i])

        if (need_cosine or need_faiss) and text_embs is None:
            # If text inference didn't complete during image batching, block now.
            t_done = time.time()
            text_embs = np.concatenate(first_output_inference(text_inf_job), axis=0)
            text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)
            if text_submitted_at is not None:
                text_wall_s = t_done - text_submitted_at
            text_detail = _inference_detail(text_inf_job)

        image_embs = np.concatenate(image_embs_parts, axis=0)

        num_images = image_embs.shape[0]

        # Cosine top-k
        if need_cosine:
            topk_cosine_id = cj.get("topk_compiled_id")
            topk_compiled = qai_hub.get_job(topk_cosine_id).get_target_model()
            topk_in = image_embs

            print(f"Submitting cosine top-k inference for {model_name}...")
            topk_submit_t = time.time()
            topk_dataset = qai_hub.upload_dataset({"image_embs": [topk_in], "text_embs": [text_embs]})
            topk_name = f"{prefix}{model_name} :: topk-cosine"
            topk_inf_job = qai_hub.submit_inference_job(
                model=topk_compiled,
                device=target_device,
                inputs=topk_dataset,
                name=topk_name,
            )
            topk_indices = first_output_inference(topk_inf_job)[0]
            topk_wall_s = time.time() - topk_submit_t
            topk_detail = _inference_detail(topk_inf_job)
            recall_at_10 = _recall_from_topk_indices(topk_indices=topk_indices, num_images=num_images)
            results["runs"].append({
                "model": model_name,
                "topk": "cosine",
                "quantize": bool(args.quantize),
                "quantize_type": args.quantize_type,
                "faiss_compute_unit": args.faiss_compute_unit,
                "recall_at_10": recall_at_10,
                "jobs": {
                    "text": {"job_id": text_job_id, "url": text_job_url, "name": text_name, "wall_time_s": text_wall_s, "hub_detail": text_detail},
                    "images": sorted(image_jobs, key=lambda r: r.get("batch_index", 0)),
                    "topk": {"job_id": topk_inf_job.job_id, "url": topk_inf_job.url, "name": topk_name, "wall_time_s": topk_wall_s, "hub_detail": topk_detail},
                },
                "job_ids_snapshot": _snapshot_for_run(
                    cj=cj,
                    image_dataset_ids=image_dataset_ids,
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
                job_name_prefix=f"{prefix}{model_name} ::",
            )
            if not faiss_compile or not faiss_compile.get("topk_compiled_id"):
                raise RuntimeError("FAISS index compile did not return a topk_compiled_id")
            topk_faiss_id = faiss_compile["topk_compiled_id"]

            topk_compiled = qai_hub.get_job(topk_faiss_id).get_target_model()
            topk_in = image_embs

            print(f"Submitting FAISS top-k inference for {model_name}...")
            topk_submit_t = time.time()
            topk_dataset = qai_hub.upload_dataset({"image_embs": [topk_in]})
            topk_name = f"{prefix}{model_name} :: topk-faiss"
            topk_inf_job = qai_hub.submit_inference_job(
                model=topk_compiled,
                device=target_device,
                inputs=topk_dataset,
                options=f"--compute_unit {args.faiss_compute_unit}",
                name=topk_name,
            )
            topk_indices = first_output_inference(topk_inf_job)[0]
            topk_wall_s = time.time() - topk_submit_t
            topk_detail = _inference_detail(topk_inf_job)
            recall_at_10 = _recall_from_topk_indices(topk_indices=topk_indices, num_images=num_images)
            results["runs"].append({
                "model": model_name,
                "topk": "faiss",
                "quantize": bool(args.quantize),
                "quantize_type": args.quantize_type,
                "faiss_compute_unit": args.faiss_compute_unit,
                "recall_at_10": recall_at_10,
                "jobs": {
                    "text": {"job_id": text_job_id, "url": text_job_url, "name": text_name, "wall_time_s": text_wall_s, "hub_detail": text_detail},
                    "images": sorted(image_jobs, key=lambda r: r.get("batch_index", 0)),
                    "topk": {"job_id": topk_inf_job.job_id, "url": topk_inf_job.url, "name": topk_name, "wall_time_s": topk_wall_s, "hub_detail": topk_detail},
                },
                "job_ids_snapshot": _snapshot_for_run(
                    cj=cj,
                    image_dataset_ids=image_dataset_ids,
                    text_dataset_id=text_dataset_id,
                    topk_compiled_id=topk_faiss_id,
                ),
            })

    # Evaluate models with limited concurrency to avoid creating a large number of Hub jobs.
    planned_models = [m for m in models if m in compile_by_model]
    max_workers = min(max(1, int(args.max_workers)), max(1, len(planned_models)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_model = {ex.submit(_eval_model, m): m for m in planned_models}
        for fut in as_completed(future_to_model):
            model_name = future_to_model[fut]
            try:
                fut.result()
            except Exception as e:
                cj = compile_by_model.get(model_name) or {}
                ds = model_to_datasets.get(model_name) or {}
                image_ids = ds.get("image_dataset_ids") or []
                text_id = ds.get("text_dataset_id")
                for topk_mode in topk_modes:
                    results["runs"].append({
                        "model": model_name,
                        "topk": topk_mode,
                        "quantize": bool(args.quantize),
                        "quantize_type": args.quantize_type,
                        "faiss_compute_unit": args.faiss_compute_unit,
                        "recall_at_10": None,
                        "job_ids_snapshot": _snapshot_for_run(
                            cj=cj,
                            image_dataset_ids=image_ids,
                            text_dataset_id=text_id,
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
