import argparse
import os

import numpy as np
import onnx
import torch
import qai_hub

from utils import RESULTS_PATH, MODELS, NUM_IMAGE_SAMPLES, CAPTIONS_PER_IMAGE, K, JOB_IDS
# NOTE: This module is responsible for submitting compile/profile jobs.
# Inference/evaluation lives in src/inference.py and src/upload_and_run.py.

# ONNX element type codes → QAI Hub dtype strings
ONNX_ELEM_TYPE = {1: "float32", 6: "int32", 7: "int64"}


class FAISSIndexWrapper(torch.nn.Module):
    """Mimics faiss.IndexFlatIP: text embeddings are baked in as weights (the
    index), and only image embeddings are passed at inference time."""

    def __init__(self, text_embs: np.ndarray, k: int):
        super().__init__()
        self.register_buffer("index", torch.from_numpy(text_embs))
        self.k = k

    def forward(self, image_embs):
        # image_embs: [N_img, D] — L2-normalized
        sims = image_embs @ self.index.T  # [N_img, N_txt]
        return torch.topk(sims, self.k, dim=-1).indices.to(torch.int32)  # [N_img, K]


def build_arg_parser():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=MODELS.keys(), help="Single model to compile and profile")
    group.add_argument("--all", action="store_true", help="Compile and profile all models")
    parser.add_argument("--image-dataset-id", default=JOB_IDS["image", "dataset_id"],
                        help="QAI Hub dataset ID for images (from upload_dataset.py). "
                             "If omitted, uses job_ids.json.")
    parser.add_argument("--text-dataset-id", default=JOB_IDS["text", "dataset_id"],
                        help="QAI Hub dataset ID for texts  (from upload_dataset.py). "
                             "If omitted, uses job_ids.json.")
    parser.add_argument("--device", default="XR2 Gen 2 (Proxy)", help="QAI Hub target device")
    parser.add_argument("--topk", choices=["cosine", "faiss"], default="cosine",
                        help="How to compute top-k on the QAS device: cosine(default) or via FAISS")
    parser.add_argument("--faiss-compute-unit", choices=["all", "npu", "gpu", "cpu"], default="all",
                        help="Compute unit for FAISS compile and inference jobs (only applies with --topk faiss)")
    parser.add_argument("--profile", action="store_true", help="Submit profile jobs after recall is computed")
    parser.add_argument("--quantize", action="store_true", help="Apply post-training quantization during compilation")
    parser.add_argument("--quantize-type", default="int16", choices=["w4a8", "w8a16", "w4a16", "int8", "int16"],
                        help="Quantization type (only applies with --quantize)")
    parser.add_argument("--image-calibration-id", default=None,
                        help="QAI Hub dataset ID for image calibration data (from upload_calibration.py)")
    parser.add_argument("--text-calibration-id", default=None,
                        help="QAI Hub dataset ID for text calibration data (from upload_calibration.py)")
    return parser


def get_input_specs(onnx_model):
    """Read input names, shapes, and dtypes directly from the ONNX graph."""
    specs = {}
    for inp in onnx_model.graph.input:
        shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
        elem_type = inp.type.tensor_type.elem_type
        if elem_type == 1:  # float32 is the QAI Hub default; omit dtype
            specs[inp.name] = shape
        else:
            specs[inp.name] = (shape, ONNX_ELEM_TYPE[elem_type])
    return specs


def first_output(job):
    """Return the first output array list from an inference job.
    QAI Hub renames outputs to output_0, output_1, … regardless of export names."""
    job_id = getattr(job, "job_id", "<unknown>")
    status = "<unknown>"
    try:
        status = job.wait()
    except Exception:
        pass

    outputs = job.download_output_data()
    if outputs is None:
        raise RuntimeError(
            f"QAI Hub inference job produced no outputs (job_id={job_id}, status={status}). "
            f"Check the job page/logs for details."
        )
    return next(iter(outputs.values()))


def load_and_validate(path, label):
    print(f"  Loading {label} from {path}...")
    model = onnx.load(path)
    try:
        onnx.checker.check_model(model)
        print(f"  {label} valid ✅")
    except onnx.checker.ValidationError as e:
        print(f"  {label} invalid ❌: {e}")
        return None
    return model


def compile_model(
    *,
    target_device,
    onnx_img,
    onnx_txt,
    onnx_topk,
    compile_options: str,
    image_calib_dataset,
    text_calib_dataset,
    topk: str,
    faiss_compute_unit: str,
):
    """Submit compile jobs and persist compile job IDs to job_ids.json.

    This function only *submits* compile jobs; it does not block waiting for
    completion. Downstream steps should use the returned job IDs.
    """
    print("  Submitting compile jobs...")
    image_compile_job = qai_hub.submit_compile_job(
        model=onnx_img,
        device=target_device,
        input_specs=get_input_specs(onnx_img),
        options=compile_options,
        calibration_data=image_calib_dataset,
    )
    text_compile_job = qai_hub.submit_compile_job(
        model=onnx_txt,
        device=target_device,
        input_specs=get_input_specs(onnx_txt),
        options=compile_options,
        calibration_data=text_calib_dataset,
    )
    JOB_IDS["image", "compiled_id"] = image_compile_job.job_id
    JOB_IDS["text", "compiled_id"] = text_compile_job.job_id

    topk_compile_job = None
    if topk == "cosine":
        topk_compile_job = qai_hub.submit_compile_job(
            model=onnx_topk,
            device=target_device,
            input_specs=get_input_specs(onnx_topk),
            options=compile_options,
        )
        JOB_IDS["topk", "compiled_id"] = topk_compile_job.job_id
        print(f"  Compile job IDs — image: {image_compile_job.job_id}, text: {text_compile_job.job_id}, topk: {topk_compile_job.job_id}")
    else:
        print(f"  Compile job IDs — image: {image_compile_job.job_id}, text: {text_compile_job.job_id}")

    return {
        "image_compiled_id": image_compile_job.job_id,
        "text_compiled_id": text_compile_job.job_id,
        "topk_compiled_id": topk_compile_job.job_id if topk_compile_job is not None else None,
    }


def compile_faiss_index_model(
    *,
    target_device,
    text_compiled,
    text_dataset,
    onnx_dir: str,
    faiss_compute_unit: str,
):
    """Build a FAISS index model by running text inference, baking embeddings, and compiling."""
    if text_compiled is None or text_dataset is None:
        raise ValueError("compile_faiss_index_model requires text_compiled and text_dataset")
    print("  Running text encoder inference to build FAISS index...")
    text_inf_job = qai_hub.submit_inference_job(model=text_compiled, device=target_device, inputs=text_dataset)
    text_embs = np.concatenate(first_output(text_inf_job), axis=0)
    text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)

    faiss_model = FAISSIndexWrapper(text_embs, k=K).eval()
    faiss_onnx_path = os.path.join(onnx_dir, "faiss_index.onnx")
    dummy_img_embs = torch.rand(NUM_IMAGE_SAMPLES, text_embs.shape[1], dtype=torch.float32)
    print(f"  Exporting FAISS index model → {faiss_onnx_path}...")
    torch.onnx.export(
        faiss_model,
        dummy_img_embs,
        faiss_onnx_path,
        input_names=["image_embs"],
        output_names=["topk_indices"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=True,
    )
    onnx_faiss = load_and_validate(faiss_onnx_path, "FAISS index")
    if onnx_faiss is None:
        return None

    faiss_compile_job = qai_hub.submit_compile_job(
        model=onnx_faiss,
        device=target_device,
        input_specs=get_input_specs(onnx_faiss),
        options=f"--target_runtime qnn_dlc --compute_unit {faiss_compute_unit}",
    )
    JOB_IDS["topk", "compiled_id"] = faiss_compile_job.job_id
    print(f"  FAISS compile job ID: {faiss_compile_job.job_id}")
    return {
        "topk_compiled_id": faiss_compile_job.job_id,
    }


def run_pipeline(args):
    parser = build_arg_parser()
    if args.quantize and (args.image_calibration_id is None or args.text_calibration_id is None):
        parser.error("--quantize requires --image-calibration-id and --text-calibration-id")

    compile_options = "--target_runtime qnn_dlc"
    if args.quantize:
        compile_options += f" --quantize_full_type {args.quantize_type}"

    image_calib_dataset = qai_hub.get_dataset(args.image_calibration_id) if args.quantize else None
    text_calib_dataset = qai_hub.get_dataset(args.text_calibration_id) if args.quantize else None

    targets = MODELS if args.all else {args.model: MODELS[args.model]}

    target_device = qai_hub.Device(args.device)
    compiled_models = {}  # model_name -> list of compiled models to profile

    # Submit compile jobs for all models first (parallel on Hub side).
    compile_jobs = {}  # model_name -> dict of compile job IDs
    for model_name in targets:
        print(f"\n{'─' * 60}")
        print(f"Model: {model_name}")
        print(f"{'─' * 60}")

        onnx_dir = os.path.join(RESULTS_PATH, "onnx", model_name)
        image_onnx_path = os.path.join(onnx_dir, "image_encoder.onnx")
        text_onnx_path = os.path.join(onnx_dir, "text_encoder.onnx")
        topk_onnx_path = os.path.join(onnx_dir, "topk_retrieval.onnx")

        required_paths = [image_onnx_path, text_onnx_path]
        if args.topk == "cosine":
            required_paths.append(topk_onnx_path)
        if not all(os.path.exists(p) for p in required_paths):
            print("  Skipping: ONNX not found. Run: python src/export_onnx.py --all")
            continue

        onnx_img = load_and_validate(image_onnx_path, "image encoder")
        onnx_txt = load_and_validate(text_onnx_path, "text encoder")
        onnx_topk = load_and_validate(topk_onnx_path, "top-k retrieval") if args.topk == "cosine" else None
        if onnx_img is None or onnx_txt is None or (args.topk == "cosine" and onnx_topk is None):
            continue

        compile_jobs[model_name] = {
            "onnx_dir": onnx_dir,
            **compile_model(
                target_device=target_device,
                onnx_img=onnx_img,
                onnx_txt=onnx_txt,
                onnx_topk=onnx_topk,
                compile_options=compile_options,
                image_calib_dataset=image_calib_dataset,
                text_calib_dataset=text_calib_dataset,
                topk=args.topk,
                faiss_compute_unit=args.faiss_compute_unit,
            ),
        }

    # For profiling, we need compiled model objects (this blocks until the compile jobs finish).
    if args.profile:
        for model_name, cj in compile_jobs.items():
            image_compiled = qai_hub.get_job(cj["image_compiled_id"]).get_target_model()
            text_compiled = qai_hub.get_job(cj["text_compiled_id"]).get_target_model()
            compiled_models[model_name] = [image_compiled, text_compiled]
            if args.topk == "cosine" and cj.get("topk_compiled_id") is not None:
                topk_compiled = qai_hub.get_job(cj["topk_compiled_id"]).get_target_model()
                compiled_models[model_name].append(topk_compiled)

    if args.profile:
        print(f"\n{'─' * 60}")
        print("Submitting profile jobs...")
        for model_name, models in compiled_models.items():
            jobs = [
                qai_hub.submit_profile_job(model=m, device=target_device, options="--max_profiler_iterations 100")
                for m in models
            ]
            print(f"  {model_name}: {[j.job_id for j in jobs]}")

    # Return compile job IDs for programmatic use (experiments / combined runner).
    return {k: {kk: vv for kk, vv in v.items() if kk.endswith("_id") or kk == "onnx_dir"} for k, v in compile_jobs.items()}


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
