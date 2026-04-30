from pathlib import Path
from itertools import chain

from transformers import CLIPTokenizer
import qai_hub
import numpy as np
import torch

from utils.img_utils import load_images
from utils.refcoco_utils import RefCocoSplit
from utils.text_utils import load_annotations

SPLIT = RefCocoSplit.VAL
DATA_FOLDER = Path("data")

# Process images
input_data = load_images(DATA_FOLDER, SPLIT)
print(len(input_data))

# Check dataset properties
if input_data:
    print(f"Processed {len(input_data)} images.")
    print(f"First image shape: {input_data[0].shape}")  # Should be (1, 3, 224, 224)
    assert input_data[0].shape == (1, 3, 224, 224)

# Upload dataset
print(qai_hub.upload_dataset({"image": input_data}))

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
assert tokenized_texts[0].shape == (1, 77)
assert tokenized_texts[0].dtype == np.int32

print(qai_hub.upload_dataset({"text": tokenized_texts}))
