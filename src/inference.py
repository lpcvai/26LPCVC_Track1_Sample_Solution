import qai_hub
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def run_inference(model, device, input_dataset):
    """Submits an inference job for the model and returns the output data."""
    inference_job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs=input_dataset,
        options="--max_profiler_iterations 1"
    )
    # return inference_job.download_output_data()
    inference_job.wait()
    return inference_job.job_id

def parse_ground_truth(txt_list, img_list):
    # Load your CSV
    df_img = pd.read_csv(img_list)
    df_txt = pd.read_csv(txt_list)

    # Get unique text prompts in order from the second column
    txt_id = df_txt.iloc[:, 0].dropna().astype(np.int16).tolist()
    gt = df_img.iloc[:, 1].dropna().tolist() # list of txt id for each image
    return txt_id, gt


def evaluate_track1(img_output, txt_output, txt_list, img_list, k=10):
    """
    Compute Recall@K between image and text embeddings.

    Args:
        img_output (np.ndarray): Image encoder output, shape (N, D)
        txt_output (np.ndarray): Text encoder output, shape (M, D)
        ground_truth_dir (str): Path to ground truth JSON file
        k (int): Top-K for recall computation

    Returns:
        float: Mean recall@K (accuracy)
    """

    # Stack them into a single 2D array: [batch, D]
    img_embeds = np.vstack([x for x in img_output])  # shape: [N, D]
    txt_embeds = np.vstack([x for x in txt_output])  # shape: [M, D]

    # Normalize
    img_embeds = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
    txt_embeds = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)

    # Now similarity will work
    sim_matrix = cosine_similarity(img_embeds, txt_embeds)

    # Load ground truth
    txt_id, gt = parse_ground_truth(txt_list, img_list)

    recalls = []

    # print(len(img_embeds))

    for i in range(len(img_embeds)):

        # print(txt_id[i])
        # print(gt[i])
        gt_ids = [int(x) for x in gt[i].split(';')]

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

# Dictionary to store outputs separately
outputs = {}

for task_name, info in tasks.items():
    compiled_id = info["compiled_id"]
    input_dataset = qai_hub.get_dataset(info["dataset_id"])

    # Retrieve the compiled model
    job = qai_hub.get_job(compiled_id)
    compiled_model = job.get_target_model()

    # Run inference
    print(f"Running inference for {task_name} model {compiled_model.model_id} on device {device.name}")
    inference_id = run_inference(compiled_model, device, input_dataset)
    inference_job = qai_hub.get_job(inference_id)

    if inference_job.get_status().failure:
        print(f"{task_name.capitalize()} inference failed")
        outputs[task_name] = None
    else:
        inference_output = inference_job.download_output_data()
        outputs[task_name] = inference_output['output_0']

text_output = outputs["text"]
image_output = outputs["image"]

result = evaluate_track1(image_output, text_output, "dataset/txt_list.csv", "dataset/img_list.csv")
print(result)