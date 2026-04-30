from collections import defaultdict
from itertools import chain

from utils.refcoco_utils import RefCocoSplit
from utils.text_utils import load_annotations


# function currently doesn't work with split=None
def get_ground_truth(split: RefCocoSplit):
    """Loads the ground truth annotations."""
    dataset = load_annotations(split)
    dataset.sort("ann_id")

    captions = sorted(set(chain.from_iterable(dataset["captions"])))

    return captions, dataset