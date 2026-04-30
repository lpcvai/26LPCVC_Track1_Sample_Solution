import pandas as pd
from transformers import CLIPTokenizer
import torch
import numpy as np  # import numpy
from itertools import chain

from utils.refcoco_utils import RefCocoSplit
from utils.text_utils import load_annotations

dataset = load_annotations(RefCocoSplit.VAL)
prompts = list(chain.from_iterable(dataset["captions"]))

# Load CLIP tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Tokenize prompts into numpy arrays of shape (1, 77) and dtype int32
tokenized_texts = []
for prompt in prompts:
    tokens = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )["input_ids"].to(torch.int32)  # torch tensor [1, 77], int32
    tokenized_texts.append(tokens.numpy())  # convert to numpy array

# Example: check first element
print(tokenized_texts[0].shape)  # (1, 77)
print(tokenized_texts[0].dtype)  # int32

# Optional: check total number of prompts
print(len(tokenized_texts))  # batch size