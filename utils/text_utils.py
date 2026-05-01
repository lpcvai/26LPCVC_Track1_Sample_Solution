from pathlib import Path

from constants import RANDOM_SEED
from utils.refcoco_utils import RefCocoSplit
import datasets
from datasets import Dataset, DatasetDict

REFCOCO_UNC_DIR = Path("data/annotations/refcoco-unc")

def load_annotations(split: RefCocoSplit, limit: int | None = None) -> Dataset | DatasetDict:
    data = datasets.load_from_disk(REFCOCO_UNC_DIR / split)

    if limit is not None:
        data.shuffle(seed=RANDOM_SEED)
        data = data.take(limit)

    return data
