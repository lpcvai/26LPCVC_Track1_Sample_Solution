import numpy as np
import qai_hub
from sklearn.metrics.pairwise import cosine_similarity

from utils.ground_truth import get_ground_truth
from utils.refcoco_utils import RefCocoSplit


def run_inference(model, device, input_dataset, job_name=None):
    """Submits an inference job for the model and returns the output data."""
    inference_job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs=input_dataset,
        options="--max_profiler_iterations 1",
        name=job_name
    )
    # return inference_job.download_output_data()
    return inference_job.job_id


def evaluate_track1(img_output, txt_output, split: RefCocoSplit, k=10):
    """
    Compute Recall@K between image and text embeddings using HF RefCOCO annotations.
    """

    # Stack them into a single 2D array: [batch, D]
    img_embeds = np.vstack([x for x in img_output])  # shape: [N, D]
    txt_embeds = np.vstack([x for x in txt_output])  # shape: [M, D]

    # Normalize
    img_embeds = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
    txt_embeds = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)

    # Now similarity will work
    sim_matrix = cosine_similarity(img_embeds, txt_embeds)

    txt_id, gt = get_ground_truth(split)

    assert len(img_embeds) == len(gt), (len(img_embeds), len(gt))
    assert len(txt_embeds) == len(txt_id), (len(txt_embeds), len(txt_id))

    recalls = []

    for i in range(len(img_embeds)):
        gt_ids = gt[i]["captions"]

        # print(txt_id[i])
        # print(gt[i])
        txt_id, gt = get_ground_truth(split)

        # Top-K text indices by similarity
        k = 10
        # Top-K text indices by similarity
        top_k = np.argsort(-sim_matrix[i])[:k]

        # Map to real text IDs
        predicted_txt_ids = [txt_id[idx] for idx in top_k]

        # Fractional recall: how many GTs are in top-K
        # print(predicted_txt_ids)
        # print(gt_ids)
        matched = len(set(predicted_txt_ids) & set(gt_ids))
        recall_i = matched / len(gt_ids)
        # print(recall_i)
        recalls.append(recall_i)

    return np.mean(recalls)


#Define target device
device = qai_hub.Device("XR2 Gen 2 (Proxy)")

# TODO: Automate this :/
# TODO: Define tasks with their corresponding compiled job IDs and dataset IDs
tasks = {
    "text": {
        "compiled_id": "",
        "dataset_id": ""
    },
    "image": {
        "compiled_id": "",
        "dataset_id": ""
    }
}

inference_jobs = {}

for task_name, info in tasks.items():
    compiled_id = info["compiled_id"]
    input_dataset = qai_hub.get_dataset(info["dataset_id"])

    job = qai_hub.get_job(compiled_id)
    compiled_model = job.get_target_model()

    print(f"Submitting inference for {task_name} model {compiled_model.model_id} on device {device.name}")

    inference_id = run_inference(compiled_model, device, input_dataset)
    inference_jobs[task_name] = qai_hub.get_job(inference_id)

# Then collect outputs
outputs = {}

for task_name, inference_job in inference_jobs.items():
    inference_output = inference_job.download_output_data() # waits here
    outputs[task_name] = inference_output["output_0"]


text_output = outputs["text"]
image_output = outputs["image"]

result = evaluate_track1(image_output, text_output, RefCocoSplit.TEST)
print(result)
