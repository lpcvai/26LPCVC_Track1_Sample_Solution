# LPCVC 2026 Track 1 - Image-to-Text Retrieval Sample Solution

## For Submissions

Check out [this repo](https://github.com/lpcvai/25LPCVC_AIHub_Guide) for more details on how to run models on AIHub.

## Overview

This repository contains Python scripts designed to extract, compile, and profile the OpenAI-CLIP's image and text encoders using the `qai_hub` library. It also includes scripts for uploading datasets and running inference with evaluation metrics such as Recall@10.

## **Table of Contents**

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)

---

## **Features**

* **Preprocessing Scripts**: Includes resizing and normalization for image inputs, and tokenization for text inputs.
* Extract CLIP Encoders: Extract image and text encoders from OpenAI-CLIP model and export as ONNX models.
* **Model Compilation**: Supports compiling the model for a specific target device using QAI Hub.
* **Model Profiling**: Submit and retrieve profiling results via QAI Hub.
* **Dataset Upload**: Upload image and text datasets to AI Hub for inference.
* **Inference & Evaluation**: Run inference on datasets and compute metrics such as Recall@10.

---

## **Requirements**

* Python 3.9+
* Torch and torchvision
* QAI Hub
* Required packages listed in `requirements.txt`

---

## **Installation**

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/lpcvai/26LPCVC_Track1_Sample_Solution.git
cd 26LPCVC_Track1_Sample_Solution
```

### **Step 2: Install Dependencies**

Ensure you have Python 3.9+ installed. Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Export ONNX Models**

Execute the script to export the encoders as ONNX models:

```bash
python export_onnx.py
```

### **2. Compile and Profile**

```bash
python compile_and_profile.py
```

This script will:

* Upload the ONNX models to AI Hub and submit a compile job.
* Submit a profiling job with the compiled models.

### **3. Upload Dataset**

Before running inference, datasets must be uploaded to AI Hub using `upload_dataset.py`. This script handles:

* Formatting images and text data into the structure expected by QAI Hub. (image: (1,3,224,224), txt: (1,77))
* Uploading the dataset and returning a dataset ID to be used in inference scripts.

```bash
python upload_dataset.py
```

This will print a `dataset_id` that you can use in `inference.py`.

### **4. Run Inference and Evaluate**

The `inference.py` script runs the compiled models on the uploaded datasets:

1. Retrieves the compiled image and text encoders from AI Hub.
2. Runs inference on the uploaded datasets.
3. Collects output embeddings for images and text.
4. Computes evaluation metrics, such as **Recall@10**, which measures how often the correct text is among the top-10 retrieved results for each image.

```bash
python inference.py
```

After completion, the script prints the Recall@10 score for the dataset.
