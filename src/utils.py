"""
Definitions of constants used across files
"""

import configparser
import os
import json
from pathlib import Path

from dataset_registry import DatasetRegistry

# List of supported model names (kept as a list for CLI choices / iteration).
# Mapping from model name -> open_clip pretrained tag.

MODEL_PRETRAINED = {
    "MobileCLIP-S1": "datacompdr",
    "MobileCLIP2-S0": "dfndr2b",
    "MobileCLIP2-S2": "dfndr2b",
    "MobileCLIP2-B": "dfndr2b",
    "MobileCLIP2-S3": "dfndr2b",
}
MODELS = list(MODEL_PRETRAINED.keys())

NUM_IMAGE_SAMPLES = 1000
# When NUM_IMAGE_SAMPLES is large, uploading/running inference on a single image dataset can exceed
# QAI Hub's 2GB flatbuffer limit. Upload images in batches and run multiple inference jobs.
IMAGES_PER_BATCH = NUM_IMAGE_SAMPLES
# When running top-k on-device, the full (N_images x N_text) similarity can exceed max runtime
# for large N. Run top-k over image embeddings in chunks of this size.
TOPK_IMAGES_PER_BATCH = IMAGES_PER_BATCH
# When running many batched inference jobs, limit how many we keep in-flight at once
# to avoid hammering the Hub API while still keeping throughput decent.
MAX_INFERENCE_INFLIGHT = 2
CAPTIONS_PER_IMAGE = 5
K = 10
NUM_CALIBRATION_SAMPLES = NUM_IMAGE_SAMPLES

# Used in baseline scripts, not in QAI exports
BATCH_SIZE = 512
NUM_DOWNLOAD_WORKERS = 16

# Anchor the working directory to the project root so all relative paths
# (RESULTS_PATH, config.ini, etc.) resolve correctly regardless of invocation dir.
_src_dir = os.path.dirname(__file__)
os.chdir(os.path.join(_src_dir, "..") if _src_dir else "..")

config = configparser.ConfigParser()
config.read("config.ini")
RESULTS_PATH = config["DEFAULT"]["results_path"]


class JobIds:
    """
    Class to manage JobIds data and dynamically update the corresponding JSON file
    """
    path: Path
    data: dict

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.loader()

    def loader(self):
        """
        Loads JobIds data from JSON file
        Creates and add null values if the file does not exist
        """
        raw: dict = {}
        if self.path.exists():
            try:
                with self.path.open("r") as f:
                    raw = json.load(f) or {}
            except Exception:
                raw = {}

        text_raw = raw.get("text") if isinstance(raw.get("text"), dict) else {}
        image_raw = raw.get("image") if isinstance(raw.get("image"), dict) else {}
        topk_raw = raw.get("topk") if isinstance(raw.get("topk"), dict) else {}

        dataset_ids = image_raw.get("dataset_ids")
        if not isinstance(dataset_ids, list):
            dataset_ids = []

        self.data = {
            "text": {
                "compiled_id": text_raw.get("compiled_id"),
                "dataset_id": text_raw.get("dataset_id"),
            },
            "image": {
                "compiled_id": image_raw.get("compiled_id"),
                "dataset_ids": dataset_ids,
            },
            "topk": {
                "compiled_id": topk_raw.get("compiled_id"),
            },
        }

        # Ensure a valid normalized file exists.
        self.save()

    def save(self):
        """
        Saves JobIds data to json file
        """
        with self.path.open("w") as f:
            json.dump(self.data, f, indent=2)

    def __getitem__(self, key):
        outer, inner = key
        return self.data[outer][inner]

    def __setitem__(self, key, value):
        outer, inner = key
        if outer not in self.data or not isinstance(self.data.get(outer), dict):
            self.data[outer] = {}
        self.data[outer][inner] = value
        self.save()

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)

    def __bool__(self):
        return bool(self.data)

    def __iter__(self):
        return iter(self.data.items())

    def __len__(self):
        return len(self.data)


JOB_IDS = JobIds("job_ids.json")

# Local cache of uploaded dataset metadata (id, name, expiration, and a stable content key).
# The registry prunes expired/invalid entries on load.
DATASETS = DatasetRegistry("datasets.json")
