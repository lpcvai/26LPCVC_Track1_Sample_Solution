import argparse
import os

import numpy as np
import onnx
import qai_hub
import torch

from utils import RESULTS_PATH, MODELS, NUM_IMAGE_SAMPLES, CAPTIONS_PER_IMAGE, K, JOB_IDS

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
parser.add_argument("--topk", choices=["device", "faiss"], default="device",
                    help="How to compute top-k: on the QAI device (default) or via FAISS on host")
parser.add_argument("--faiss-compute-unit", choices=["all", "npu", "gpu", "cpu"], default="all",
                    help="Compute unit for FAISS compile and inference jobs (only applies with --topk faiss)")
parser.add_argument("--profile", action="store_true", help="Submit profile jobs after recall is computed")
args = parser.parse_args()

targets = MODELS if args.all else {args.model: MODELS[args.model]}


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
    return next(iter(job.download_output_data().values()))


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


if args.image_dataset_id is None or args.text_dataset_id is None:
    raise SystemExit(
        "Missing dataset IDs. Run upload_dataset.py first (it updates job_ids.json), "
        "or pass --image-dataset-id/--text-dataset-id."
    )

target_device = qai_hub.Device(args.device)
image_dataset = qai_hub.get_dataset(args.image_dataset_id)
text_dataset = qai_hub.get_dataset(args.text_dataset_id)

recall_summary = {}
compiled_models = {}  # model_name -> list of compiled models to profile

for model_name in targets:
    print(f"\n{'─' * 60}")
    print(f"Model: {model_name}")
    print(f"{'─' * 60}")

    onnx_dir = os.path.join(RESULTS_PATH, "onnx", model_name)
    image_onnx_path = os.path.join(onnx_dir, "image_encoder.onnx")
    text_onnx_path = os.path.join(onnx_dir, "text_encoder.onnx")
    topk_onnx_path = os.path.join(onnx_dir, "topk_retrieval.onnx")

    required_paths = [image_onnx_path, text_onnx_path]
    if args.topk == "device":
        required_paths.append(topk_onnx_path)
    if not all(os.path.exists(p) for p in required_paths):
        print(f"  Skipping: ONNX not found. Run: python src/export_onnx.py --all")
        continue

    onnx_img = load_and_validate(image_onnx_path, "image encoder")
    onnx_txt = load_and_validate(text_onnx_path, "text encoder")
    onnx_topk = load_and_validate(topk_onnx_path, "top-k retrieval") if args.topk == "device" else None
    if onnx_img is None or onnx_txt is None or (args.topk == "device" and onnx_topk is None):
        continue

    # ── Compile (submit all jobs first, then await) ───────────────────────────
    print("  Submitting compile jobs...")
    img_compile_job = qai_hub.submit_compile_job(
        model=onnx_img,
        device=target_device,
        input_specs=get_input_specs(onnx_img),
        options="--target_runtime qnn_dlc",
    )
    txt_compile_job = qai_hub.submit_compile_job(
        model=onnx_txt,
        device=target_device,
        input_specs=get_input_specs(onnx_txt),
        options="--target_runtime qnn_dlc",
    )
    # Persist latest compile job IDs for convenience re-runs (CLI always overrides).
    JOB_IDS["image", "compiled_id"] = img_compile_job.job_id
    JOB_IDS["text", "compiled_id"] = txt_compile_job.job_id
    if args.topk == "device":
        topk_compile_job = qai_hub.submit_compile_job(
            model=onnx_topk,
            device=target_device,
            input_specs=get_input_specs(onnx_topk),
            options="--target_runtime qnn_dlc",
        )
        JOB_IDS["topk", "compiled_id"] = topk_compile_job.job_id
        print(
            f"  Compile job IDs — image: {img_compile_job.job_id}, text: {txt_compile_job.job_id}, topk: {topk_compile_job.job_id}")
    else:
        print(f"  Compile job IDs — image: {img_compile_job.job_id}, text: {txt_compile_job.job_id}")

    img_compiled = img_compile_job.get_target_model()
    txt_compiled = txt_compile_job.get_target_model()

    if args.topk == "device":
        topk_compiled = topk_compile_job.get_target_model()

        # ── Encoder inference ─────────────────────────────────────────────────
        print("  Submitting encoder inference jobs...")
        img_inf_job = qai_hub.submit_inference_job(model=img_compiled, device=target_device, inputs=image_dataset)
        txt_inf_job = qai_hub.submit_inference_job(model=txt_compiled, device=target_device, inputs=text_dataset)
        print(f"  Inference job IDs — image: {img_inf_job.job_id}, text: {txt_inf_job.job_id}")

        image_embs = np.concatenate(first_output(img_inf_job), axis=0)
        text_embs = np.concatenate(first_output(txt_inf_job), axis=0)
        image_embs = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)
        text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)

        # ── Top-k on device ───────────────────────────────────────────────────
        topk_dataset = qai_hub.upload_dataset({
            "image_embs": [image_embs],  # single sample: [N, D]
            "text_embs": [text_embs],  # single sample: [N, D]
        })
        topk_inf_job = qai_hub.submit_inference_job(model=topk_compiled, device=target_device, inputs=topk_dataset)
        topk_indices = first_output(topk_inf_job)[0]  # [N, K]

    else:
        # ── FAISS path: run text encoder first to build the on-device index ───
        print("  Running text encoder inference to build FAISS index...")
        txt_inf_job = qai_hub.submit_inference_job(model=txt_compiled, device=target_device, inputs=text_dataset)
        text_embs = np.concatenate(txt_inf_job.download_output_data()["text_embedding"], axis=0)
        text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)

        # Bake the text embeddings into a FAISSIndexWrapper and compile it —
        # this mirrors how faiss.IndexFlatIP stores the database as weights.
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
            continue

        faiss_compile_job = qai_hub.submit_compile_job(
            model=onnx_faiss,
            device=target_device,
            input_specs=get_input_specs(onnx_faiss),
            options=f"--target_runtime qnn_dlc --compute_unit {args.faiss_compute_unit}",
        )
        JOB_IDS["topk", "compiled_id"] = faiss_compile_job.job_id
        print(f"  FAISS compile job ID: {faiss_compile_job.job_id}")
        faiss_compiled = faiss_compile_job.get_target_model()

        # ── Image encoder inference → upload embeddings → FAISS inference ─────
        print("  Running image encoder inference...")
        img_inf_job = qai_hub.submit_inference_job(model=img_compiled, device=target_device, inputs=image_dataset)
        image_embs = np.concatenate(first_output(img_inf_job), axis=0)
        image_embs = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)

        faiss_dataset = qai_hub.upload_dataset({"image_embs": [image_embs]})
        faiss_inf_job = qai_hub.submit_inference_job(
            model=faiss_compiled,
            device=target_device,
            inputs=faiss_dataset,
            options=f"--compute_unit {args.faiss_compute_unit}",
        )
        topk_indices = first_output(faiss_inf_job)[0]  # [N, K]

    # ── Recall@10 ─────────────────────────────────────────────────────────────
    # Texts are ordered: CAPTIONS_PER_IMAGE captions per image, so ground truth
    # for image i is text indices [i*C, (i+1)*C).
    N = image_embs.shape[0]
    recalls = []
    for i in range(N):
        gt = set(range(i * CAPTIONS_PER_IMAGE, (i + 1) * CAPTIONS_PER_IMAGE))
        recalls.append(len(gt & set(topk_indices[i].tolist())) / CAPTIONS_PER_IMAGE)
    recall_at_10 = float(np.mean(recalls))

    print(f"  Recall@10: {recall_at_10:.4f}  ({recall_at_10 * 100:.2f}%)")
    recall_summary[model_name] = recall_at_10
    compiled_models[model_name] = [img_compiled, txt_compiled,
                                   topk_compiled if args.topk == "device" else faiss_compiled]

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═' * 60}")
print("Summary — Recall@10 on uploaded dataset")
print(f"{'═' * 60}")
for model_name, recall in recall_summary.items():
    print(f"  {model_name:<20} {recall * 100:.2f}%")

# ── Profiling (optional, submitted after recall so it doesn't block inference) ─
if args.profile:
    print(f"\n{'─' * 60}")
    print("Submitting profile jobs...")
    for model_name, models in compiled_models.items():
        jobs = [
            qai_hub.submit_profile_job(model=m, device=target_device, options="--max_profiler_iterations 100")
            for m in models
        ]
        print(f"  {model_name}: {[j.job_id for j in jobs]}")
