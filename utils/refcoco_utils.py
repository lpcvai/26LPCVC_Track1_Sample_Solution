from enum import StrEnum

from datasets import load_dataset, DatasetDict
from datasets.arrow_dataset import Dataset


class RefCocoSplit(StrEnum):
    """
    TODO: Write this
    """
    TRAIN = "train"
    TEST = "test"
    TESTB = "testB"
    VAL = "validation"


def get_refcoco_dataset(split: RefCocoSplit | None = None) -> DatasetDict:
    """
    TODO: Write this
    """
    return load_dataset("jxu124/RefCOCO", split=split)  # UNC partition
