# MobileCLIP QAI Hub Benchmarking

This repo benchmarks multiple MobileCLIP variants on Qualcomm AI Hub (QAI Hub) by:

1. Exporting models to ONNX
2. Uploading evaluation datasets to QAI Hub
3. Compiling for a target device
4. Running inference and computing Recall@10

The main “end-to-end” entrypoint is `src/experiments.py`.

## Environment Setup

**Prereqs**

- Python 3.10
- A working QAI Hub account and credentials configured locally (so `qai_hub` can submit jobs)



```bash
conda create -n clipenv python=3.10
conda activate clipenv
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## Project Configuration

This project uses `config.ini` for a couple repo-relative paths:

- `COCO_PATH`: expected local COCO location for some scripts
- `RESULTS_PATH`: where outputs (ONNX artifacts, experiment summaries, etc.) are written

Default:

```ini
[DEFAULT]
COCO_PATH=data/coco
RESULTS_PATH=results
```

Most of the QAI-Hub-driven pipeline (including `experiments.py`) uses the HuggingFace
dataset `yerevann/coco-karpathy` for upload/eval and will write outputs under `results/`.



## Using `experiments.py`

`src/experiments.py` runs the full pipeline:

1. Upload image/text datasets to QAI Hub (with caching in `datasets.json`)
2. Export ONNX artifacts under `results/onnx_experiments/...`
3. Submit compile jobs once per model
4. Submit quantization jobs once per model (if `--quantize` is set)
5. Submit inference jobs (text encoder, image encoder in batches, and top-k)
6. Write a JSON summary to `results/experiments/experiments_<timestamp>.json` or a custom path.

### Basic Runs

Run all models (cosine top-k):

```bash
python src/experiments.py
```

Run a single model:

```bash
python src/experiments.py --models "MobileCLIP2-S0"
```

Top-k is always cosine (FAISS support has been removed).

Write results to a specific file:

```bash
python src/experiments.py --output results/experiments/my_run.json
```

### Dataset Size / Batching

Control the evaluation size (number of images; texts are `num_images * 5` captions):

```bash
python src/experiments.py --num-images 200
```

Control image upload and image-encoder inference batch sizing:

```bash
python src/experiments.py --images-per-batch 250
```

Control top-k inference batch sizing (defaults to `--images-per-batch`):

```bash
python src/experiments.py --topk-images-per-batch 250
```

### Quantization

You can quantize both encoders during compilation:

```bash
python src/experiments.py --quantize int16
```

Quantization requires calibration datasets. You can either:

1. Let the script upload calibration datasets automatically (default behavior when `--quantize` is set and no IDs are provided), or
2. Provide existing QAI Hub calibration dataset IDs:

```bash
python src/experiments.py \
  --quantize int8 \
  --image-calibration-id <QAI_HUB_IMAGE_CAL_ID> \
  --text-calibration-id <QAI_HUB_TEXT_CAL_ID>
```

Control how many samples are used when uploading calibration datasets:

```bash
python src/experiments.py --quantize int16 --calibration-samples 200
```

### Caching Behavior

Dataset uploads are cached in `datasets.json` by a stable key derived from:

- dataset split + size
- preprocessing signature (mean/std, forced 224x224)
- tokenizer signature (shape/dtype)

To disable reuse:

```bash
python src/experiments.py --no-cache
```

To prevent writing new cache entries:

```bash
python src/experiments.py --no-cache-write
```

### Device / Job Naming

Choose a target device (must exist in your QAI Hub account):

```bash
python src/experiments.py --device "XR2 Gen 2 (Proxy)"
```

Add a prefix to the QAI Hub job names (useful in the Hub UI when running multiple trials):

```bash
python src/experiments.py --job-name-prefix "cmsc472"
```

### CLI Help (`--help`)

The flags for `src/experiments.py` are defined via `argparse`. Help text:

```text
usage: experiments.py [-h]
                      [--models [{MobileCLIP-S1,MobileCLIP2-S0,MobileCLIP2-S2,MobileCLIP2-B,MobileCLIP2-S3} ...]]
                      [--num-images NUM_IMAGES] [--device DEVICE]
                      [--quantize [{w4a8,w8a16,w4a16,int8,int16}]]
                      [--image-calibration-id IMAGE_CALIBRATION_ID]
                      [--text-calibration-id TEXT_CALIBRATION_ID]
                      [--calibration-samples CALIBRATION_SAMPLES]
                      [--images-per-batch IMAGES_PER_BATCH]
                      [--topk-images-per-batch TOPK_IMAGES_PER_BATCH]
                      [--job-name-prefix JOB_NAME_PREFIX] [--output OUTPUT]
                      [--cache | --no-cache]
                      [--cache-write | --no-cache-write]

Run compile and inference benchmarks across models (cosine top-k).

options:
  -h, --help            show this help message and exit
  --models [{MobileCLIP-S1,MobileCLIP2-S0,MobileCLIP2-S2,MobileCLIP2-B,MobileCLIP2-S3} ...]
                        Models to benchmark. If omitted, benchmarks all
                        models.
  --num-images NUM_IMAGES
                        Number of images to evaluate (default from utils.py:
                        1000).
  --device DEVICE       QAI Hub target device
  --quantize [{w4a8,w8a16,w4a16,int8,int16}]
                        Apply post-training quantization to both encoders
                        during compilation. If provided without a value,
                        defaults to int16.
  --image-calibration-id IMAGE_CALIBRATION_ID
                        QAI Hub dataset ID for image calibration data
                        (required for --quantize)
  --text-calibration-id TEXT_CALIBRATION_ID
                        QAI Hub dataset ID for text calibration data (required
                        for --quantize)
  --calibration-samples CALIBRATION_SAMPLES
                        Number of validation samples to use for quantization
                        calibration (default from utils.py: 200).
  --images-per-batch IMAGES_PER_BATCH
                        Images per uploaded dataset and per image-encoder
                        inference job (default: 1000).
  --topk-images-per-batch TOPK_IMAGES_PER_BATCH
                        Images per top-k inference job (default: same as
                        --images-per-batch).
  --job-name-prefix JOB_NAME_PREFIX
                        Optional prefix for QAI Hub inference job names to
                        make jobs easier to identify in the UI.
  --output OUTPUT       Optional path to write a JSON summary.
  --cache, --no-cache   Reuse datasets via datasets.json. (default: True)
  --cache-write, --no-cache-write
                        Write uploaded dataset info into datasets.json.
                        (default: True)
```


## Output Artifacts

- Experiment summaries: `results/experiments/experiments_YYYYMMDD_HHMMSS.json` or a custom path.
- ONNX exports for experiments: `results/onnx_experiments/<key>/<model_name>/...`
- Dataset cache registry: `datasets.json`
- Some scripts may also update `job_ids.json` with the “last run” IDs
