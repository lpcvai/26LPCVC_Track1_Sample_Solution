from pathlib import Path

from utils.refcoco_utils import RefCocoSplit
import datasets
from datasets import Dataset, DatasetDict

REFCOCO_UNC_DIR = Path("data/annotations/refcoco-unc")

def load_annotations(split: RefCocoSplit | None = None) -> Dataset | DatasetDict:
    if split is None:
        return datasets.load_from_disk(REFCOCO_UNC_DIR)
    return datasets.load_from_disk(REFCOCO_UNC_DIR / split)
