"""
Definitions of constants used across files
"""

import configparser
import os
import json
from pathlib import Path


MODELS = {
    "MobileCLIP-S1":  "datacompdr",
    "MobileCLIP2-S0": "dfndr2b",
    "MobileCLIP2-S2": "dfndr2b",
    "MobileCLIP2-B":  "dfndr2b",
    "MobileCLIP2-S3": "dfndr2b",
}

NUM_IMAGE_SAMPLES  = 500
CAPTIONS_PER_IMAGE = 5
NUM_TEXT_SAMPLES   = NUM_IMAGE_SAMPLES * CAPTIONS_PER_IMAGE
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
    Class to manage JobIds data and dynamically update the corresponding json file
    """
    path: Path
    data: dict

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.loader()

    def loader(self):
        """
        Loads JobIds data from json file
        Creates and add null values if the file does not exist
        """
        if self.path.exists():
            with self.path.open("r") as f:
                self.data = json.load(f)
                return
        self.data = {
            "text": {"compiled_id": None, "dataset_id": None},
            "image": {"compiled_id": None, "dataset_id": None}
        }
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


