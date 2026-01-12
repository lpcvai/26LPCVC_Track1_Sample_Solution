# LPCVC 2026 Track 1 - Image-to-Text Retrieval Sample Solution

## For Submissions

Check out [this repo](https://github.com/lpcvai/25LPCVC_AIHub_Guide) for more details on how to run models on AIHub.

## Overview

This repository contains Python scripts designed to extract, compile, and profile the OpenAI-CLIP's image and text encoders using the `qai_hub` library.

## **Table of Contents**

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Requirements](#requirements)

---

## **Features**

- **Preprocessing Scripts**: Includes resizing and normalization for image inputs, and tokenization for text inputs.
- Extract CLIP Encoders: Extract image and text encoders from OpenAI-CLIP model and export as ONNX models.
- **Model Compilation**: Supports compiling the model for a specific target device using QAI Hub.
- **Model Profiling**: Submit and retrieve profiling results via QAI Hub.

---

## **Installation**

### **Step 1: Clone the Repository**

### **Step 2: Install Dependencies**

Ensure you have Python 3.9+ installed. Install the required Python packages:

`pip install -r requirements.txt`

---

## **Requirements**

- Python 3.9+
- Torch and torchvision
- QAI Hub
- Required packages listed in `requirements.txt`

---

## **Usage**

### **1. Export ONNX Models**

Execute the script to export the encoders as ONNX models:

`python export_onnx.py`

### **2. Compile and Profile**

`python compile_and_profile.py`

This python scipt will:

- Upload the ONNX models to AI Hub and submit a compile job.
- Submit a profiling job with the compiled models.
