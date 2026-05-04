"""
Definitions of constants used across files
"""

import configparser
import os
import json
from pathlib import Path

from dataset_registry import DatasetRegistry

def _load_models_json(path: str = "models.json") -> tuple[list[str], dict[str, str], dict[str, dict]]:
    p = Path(path)
    if not p.exists():
        return [], {}, {}
    try:
        obj = json.loads(p.read_text()) or {}
    except Exception:
        return [], {}, {}
    models = obj.get("models")
    if not isinstance(models, list):
        return [], {}, {}
    names: list[str] = []
    pretrained: dict[str, str] = {}
    meta: dict[str, dict] = {}
    for m in models:
        if not isinstance(m, dict):
            continue
        name = m.get("name")
        mid = m.get("id")
        if isinstance(name, str) and name.strip() and isinstance(mid, str) and mid.strip():
            names.append(name.strip())
            pretrained[name.strip()] = mid.strip()
            meta[name.strip()] = m
    return names, pretrained, meta


# List of supported model names (kept as a list for CLI choices / iteration).
# Mapping from model name -> open_clip pretrained tag.
# These are primarily sourced from repo-root models.json.
MODELS, MODEL_PRETRAINED, MODEL_META = _load_models_json()

# Backwards-compatible fallback if models.json is missing/corrupt.
if not MODELS or not MODEL_PRETRAINED:
    MODELS = [
        "MobileCLIP-S1",
        "MobileCLIP2-S0",
        "MobileCLIP2-S2",
        "MobileCLIP2-B",
        "MobileCLIP2-S3",
    ]
    MODEL_PRETRAINED = {
        "MobileCLIP-S1": "datacompdr",
        "MobileCLIP2-S0": "dfndr2b",
        "MobileCLIP2-S2": "dfndr2b",
        "MobileCLIP2-B": "dfndr2b",
        "MobileCLIP2-S3": "dfndr2b",
    }
    MODEL_META = {}

NUM_IMAGE_SAMPLES = 1000
# When NUM_IMAGE_SAMPLES is large, uploading/running inference on a single image dataset can exceed
# QAI Hub's 2GB flatbuffer limit. Upload images in batches and run multiple inference jobs.
IMAGES_PER_BATCH = 1000
# When running top-k on-device, the full (N_images x N_text) similarity can exceed max runtime
# for large N. Run top-k over image embeddings in chunks of this size.
TOPK_IMAGES_PER_BATCH = 1000
# When running many batched inference jobs, limit how many we keep in-flight at once
# to avoid hammering the Hub API while still keeping throughput decent.
MAX_INFERENCE_INFLIGHT = 2
CAPTIONS_PER_IMAGE = 5
NUM_TEXT_SAMPLES = NUM_IMAGE_SAMPLES * CAPTIONS_PER_IMAGE
K = 10
NUM_CALIBRATION_SAMPLES = 200

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
        if self.path.exists():
            with self.path.open("r") as f:
                self.data = json.load(f) or {}
            # Backwards-compatible defaults for new fields.
            if "text" not in self.data or not isinstance(self.data.get("text"), dict):
                self.data["text"] = {}
            if "image" not in self.data or not isinstance(self.data.get("image"), dict):
                self.data["image"] = {}
            if "topk" not in self.data or not isinstance(self.data.get("topk"), dict):
                self.data["topk"] = {}
            self.data["text"].setdefault("compiled_id", None)
            self.data["text"].setdefault("dataset_id", None)
            self.data["image"].setdefault("compiled_id", None)
            # Images are always batched now; keep only dataset_ids.
            self.data["image"].setdefault("dataset_ids", [])
            # Top-k can have multiple compiled IDs (cosine vs faiss), so store a dict.
            self.data["topk"].setdefault("compiled_ids", {})

            # Remove legacy single-id fields if present to avoid accidental use.
            if "dataset_id" in self.data["image"]:
                self.data["image"].pop("dataset_id", None)
            if "compiled_id" in self.data["topk"]:
                # Migrate to compiled_ids, best-effort (assume cosine if unknown).
                legacy = self.data["topk"].pop("compiled_id", None)
                if legacy and "cosine" not in self.data["topk"]["compiled_ids"]:
                    self.data["topk"]["compiled_ids"]["cosine"] = legacy
            # Ensure file is normalized to include new fields.
            self.save()
            return

        self.data = {
            "text": {"compiled_id": None, "dataset_id": None},
            "image": {"compiled_id": None, "dataset_ids": []},
            "topk": {"compiled_ids": {}},
        }
        # Ensure a valid initial file exists.
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

# Local cache of uploaded dataset metadata (ids, names, expiration, image range).
# The registry prunes expired/invalid entries on load.
DATASETS = DatasetRegistry("datasets.json")
