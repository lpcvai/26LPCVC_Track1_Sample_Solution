import os
from enum import StrEnum

from datasets import Dataset, DatasetDict, load_dataset

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


class RefCocoSplit(StrEnum):
    """
    RefCOCO dataset splits using the UNC partition.
    """

    TRAIN = "train"
    VAL = "validation"
    TEST = "test"
    TESTA = "testA"
    TESTB = "testB"


def download_refcoco_dataset(split: RefCocoSplit | list[RefCocoSplit] | None = None) \
        -> Dataset | list[Dataset] | DatasetDict:
    """
    Download the RefCOCO dataset from Hugging Face.

    :param split: The split or splits to download. If None, downloads all available splits.
    :return: The downloaded dataset split or splits.
    """
    if isinstance(split, list):
        split = [s.value for s in split]
    elif split is not None:
        split = split.value

    return load_dataset("jxu124/RefCOCO", split=split)  # UNC partition
