import os
from enum import StrEnum

from datasets import load_dataset, DatasetDict

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

class RefCocoSplit(StrEnum):
    """
    TODO: Write this
    """
    TRAIN = "train"
    TEST = "test"
    TESTB = "testB"
    VAL = "validation"


def download_refcoco_dataset(split: RefCocoSplit | list[RefCocoSplit] | None = None) -> DatasetDict:
    """
    TODO: Write this
    """
    return load_dataset("jxu124/RefCOCO", split=split)  # UNC partition
