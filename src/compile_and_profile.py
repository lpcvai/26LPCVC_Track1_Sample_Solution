import argparse
import os

import onnx
import qai_hub

from utils import RESULTS_PATH, MODELS, JOB_IDS

# NOTE: This module is responsible for submitting compile/profile jobs.
# Inference/evaluation lives in src/inference.py and src/upload_and_run.py.

# ONNX element type codes → QAI Hub dtype strings
ONNX_ELEM_TYPE = {1: "float32", 6: "int32", 7: "int64"}


def build_arg_parser():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=MODELS, help="Single model to compile and profile")
    group.add_argument("--all", action="store_true", help="Compile and profile all models")
    parser.add_argument("--text-dataset-id", default=JOB_IDS["text", "dataset_id"],
                        help="QAI Hub dataset ID for texts  (from upload_dataset.py). "
                             "If omitted, uses job_ids.json.")
    parser.add_argument("--device", default="XR2 Gen 2 (Proxy)", help="QAI Hub target device")
    parser.add_argument("--onnx-root", default=os.path.join(RESULTS_PATH, "onnx"),
                        help="Root directory where ONNX artifacts live (default: <results_path>/onnx).")
    parser.add_argument("--profile", action="store_true", help="Submit profile jobs after recall is computed")
    # Optional arg:
    #   no flag => no quantization
    #   --quantize <type> => quantize both encoders using the given quantization type
    parser.add_argument(
        "--quantize",
        nargs="?",
        const="int16",
        default=None,
        choices=["w4a8", "w8a16", "w4a16", "int8", "int16"],
        help="Apply post-training quantization to both encoders during compilation. "
             "If provided without a value, defaults to int16. (uses Quantize Job API).",
    )
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
        persist_job_ids: bool = True,
        job_name_prefix: str = "",
        quantize_type: str | None = None,
):
    """Submit compile jobs and persist compile job IDs to job_ids.json.

    This function only *submits* compile jobs; it does not block waiting for
    completion. Downstream steps should use the returned job IDs.
    """
    print("  Submitting compile jobs...")
    prefix = (job_name_prefix.strip() + " ") if job_name_prefix and job_name_prefix.strip() else ""

    # Quantize Job API (preferred) – replaces deprecated --quantize_full_type.
    quantized_img = onnx_img
    quantized_txt = onnx_txt
    if quantize_type is not None:
        if image_calib_dataset is None or text_calib_dataset is None:
            raise ValueError("quantize_type requires image_calib_dataset and text_calib_dataset")
        qt = str(quantize_type).lower()
        if qt == "int8":
            w_dtype = qai_hub.QuantizeDtype.INT8
            a_dtype = qai_hub.QuantizeDtype.INT8
        elif qt == "int16":
            w_dtype = qai_hub.QuantizeDtype.INT16
            a_dtype = qai_hub.QuantizeDtype.INT16
        elif qt == "w4a8":
            w_dtype = qai_hub.QuantizeDtype.INT4
            a_dtype = qai_hub.QuantizeDtype.INT8
        elif qt == "w4a16":
            w_dtype = qai_hub.QuantizeDtype.INT4
            a_dtype = qai_hub.QuantizeDtype.INT16
        elif qt == "w8a16":
            w_dtype = qai_hub.QuantizeDtype.INT8
            a_dtype = qai_hub.QuantizeDtype.INT16
        else:
            raise ValueError(f"Unsupported quantize_type for Quantize Job API: {quantize_type}")

        print("  Submitting quantize jobs...")
        q_img = qai_hub.submit_quantize_job(
            onnx_img,
            calibration_data=image_calib_dataset,
            weights_dtype=w_dtype,
            activations_dtype=a_dtype,
            name=f"{prefix}quantize :: image-encoder",
        )
        q_txt = qai_hub.submit_quantize_job(
            onnx_txt,
            calibration_data=text_calib_dataset,
            weights_dtype=w_dtype,
            activations_dtype=a_dtype,
            name=f"{prefix}quantize :: text-encoder",
        )

        # Block until quantization completes, then compile the quantized ONNX model artifacts.
        quantized_img = q_img.get_target_model()
        if quantized_img is None:
            raise RuntimeError("Quantize job (image) failed to produce a target model.")
        quantized_txt = q_txt.get_target_model()
        if quantized_txt is None:
            raise RuntimeError("Quantize job (text) failed to produce a target model.")

    image_compile_job = qai_hub.submit_compile_job(
        model=quantized_img,
        device=target_device,
        input_specs=get_input_specs(onnx_img),
        options=compile_options,
        name=f"{prefix}image-encoder",
    )
    text_compile_job = qai_hub.submit_compile_job(
        model=quantized_txt,
        device=target_device,
        input_specs=get_input_specs(onnx_txt),
        options=compile_options,
        name=f"{prefix}text-encoder",
    )
    if persist_job_ids:
        JOB_IDS["image", "compiled_id"] = image_compile_job.job_id
        JOB_IDS["text", "compiled_id"] = text_compile_job.job_id

    topk_compile_job = qai_hub.submit_compile_job(
        model=onnx_topk,
        device=target_device,
        input_specs=get_input_specs(onnx_topk),
        options=compile_options,
        name=f"{prefix}topk-cosine",
    )
    if persist_job_ids:
        JOB_IDS["topk", "compiled_id"] = topk_compile_job.job_id
    print(
        f"  Compile job IDs — image: {image_compile_job.job_id}, text: {text_compile_job.job_id}, topk: {topk_compile_job.job_id}"
    )

    return {
        "image_compiled_id": image_compile_job.job_id,
        "text_compiled_id": text_compile_job.job_id,
        "topk_compiled_id": topk_compile_job.job_id,
    }


def run_pipeline(args, *, persist_job_ids: bool = True):
    parser = build_arg_parser()
    if args.quantize and (args.image_calibration_id is None or args.text_calibration_id is None):
        parser.error("--quantize requires --image-calibration-id and --text-calibration-id")

    compile_options = "--target_runtime qnn_dlc"
    # Quantization is handled via submit_quantize_job (Quantize Job API). No compile flag needed.

    image_calib_dataset = qai_hub.get_dataset(args.image_calibration_id) if args.quantize else None
    text_calib_dataset = qai_hub.get_dataset(args.text_calibration_id) if args.quantize else None

    targets = list(MODELS) if args.all else [args.model]

    target_device = qai_hub.Device(args.device)
    compiled_models = {}  # model_name -> list of compiled models to profile

    # Submit compile jobs for all models first (parallel on Hub side).
    compile_jobs = {}  # model_name -> dict of compile job IDs
    for model_name in targets:
        print(f"\n{'─' * 60}")
        print(f"Model: {model_name}")
        print(f"{'─' * 60}")

        onnx_dir = os.path.join(args.onnx_root, model_name)
        image_onnx_path = os.path.join(onnx_dir, "image_encoder.onnx")
        text_onnx_path = os.path.join(onnx_dir, "text_encoder.onnx")
        topk_onnx_path = os.path.join(onnx_dir, "topk_retrieval.onnx")

        required_paths = [image_onnx_path, text_onnx_path, topk_onnx_path]
        if not all(os.path.exists(p) for p in required_paths):
            print("  Skipping: ONNX not found. Run: python src/export_onnx.py --all")
            continue

        onnx_img = load_and_validate(image_onnx_path, "image encoder")
        onnx_txt = load_and_validate(text_onnx_path, "text encoder")
        onnx_topk = load_and_validate(topk_onnx_path, "top-k retrieval")
        if onnx_img is None or onnx_txt is None or onnx_topk is None:
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
                persist_job_ids=persist_job_ids,
                job_name_prefix=f"{model_name} :: ",
                quantize_type=args.quantize,
            ),
        }

    # For profiling, we need compiled model objects (this blocks until the compile jobs finish).
    if args.profile:
        for model_name, cj in compile_jobs.items():
            image_compiled = qai_hub.get_job(cj["image_compiled_id"]).get_target_model()
            text_compiled = qai_hub.get_job(cj["text_compiled_id"]).get_target_model()
            topk_compiled = qai_hub.get_job(cj["topk_compiled_id"]).get_target_model()
            compiled_models[model_name] = [
                ("image-encoder", image_compiled),
                ("text-encoder", text_compiled),
                ("topk-cosine", topk_compiled),
            ]

    if args.profile:
        print(f"\n{'─' * 60}")
        print("Submitting profile jobs...")

        def _submit_profile_job(*, m, name: str):
            # Not all qai_hub versions accept a `name=` kwarg for profile jobs. Try it first,
            # then fall back without naming to avoid breaking functionality.
            try:
                return qai_hub.submit_profile_job(
                    model=m,
                    device=target_device,
                    options="--max_profiler_iterations 100",
                    name=name,
                )
            except TypeError:
                return qai_hub.submit_profile_job(
                    model=m,
                    device=target_device,
                    options="--max_profiler_iterations 100",
                )

        for model_name, models in compiled_models.items():
            jobs = []
            for label, m in models:
                jobs.append(_submit_profile_job(m=m, name=f"{model_name} :: profile {label}"))
            print(f"  {model_name}: {[j.job_id for j in jobs]}")

    # Return compile job IDs for programmatic use (experiments / combined runner).
    return {k: {kk: vv for kk, vv in v.items() if kk.endswith("_id") or kk == "onnx_dir"} for k, v in
            compile_jobs.items()}


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
