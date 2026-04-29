import torch
import os
from qai_hub_models.models.openai_clip.model import OpenAIClip

# --- Configuration for File Saving ---
ONNX_DIR = "exported_onnx"
device = torch.device("cpu") # use CPU to export onnx model to avoid GPU device issues
# -----------------------------------

# -----------------------------
# 1. Prepare Environment
# -----------------------------
os.makedirs(ONNX_DIR, exist_ok=True)
print(f"Saving ONNX files to directory: {os.path.abspath(ONNX_DIR)}")

# -----------------------------
# 2. Dummy inputs
# -----------------------------
DUMMY_IMAGE_INPUT = torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device)
DUMMY_TEXT_INPUT = torch.randint(0, 49408, (1, 77), dtype=torch.int64, device=device)

# -----------------------------
# 3. Load OpenAIClip wrapper and define encoders
# -----------------------------
print("Loading OpenAIClip wrapper model...")
clip_wrapper_model = OpenAIClip.from_pretrained().to(device)
clip_wrapper_model.eval()

clip_model = clip_wrapper_model.clip.to(device)
clip_model = clip_model.to(torch.float32) # convert all model params to float32 type, consistent with input type in compiling and profiling via AIHub
clip_model.eval()

class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, images):
        return self.visual(images)

class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        eos_index = token_ids.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_index]
        x = x @ self.text_projection
        return x

# -----------------------------
# 4. Create wrapper instances
# -----------------------------
image_encoder = ImageEncoderWrapper(clip_model)
text_encoder = TextEncoderWrapper(clip_model)
image_encoder.eval()
text_encoder.eval()


# -----------------------------
# 5. Export Image Encoder
# -----------------------------
image_onnx_path = os.path.join(ONNX_DIR, "image_encoder.onnx")
print(f"\nExporting Image Encoder to {image_onnx_path}...")

torch.onnx.export(
    image_encoder,
    DUMMY_IMAGE_INPUT,
    image_onnx_path,
    input_names=["image"],
    output_names=["embedding"],
    opset_version=18,
    do_constant_folding=True,
    dynamic_axes=None,
    verbose=False,
    export_params=True,
    training=torch.onnx.TrainingMode.EVAL,
    dynamo=True,
)

# -----------------------------
# 6. Export Text Encoder
# -----------------------------
text_onnx_path = os.path.join(ONNX_DIR, "text_encoder.onnx")
print(f"\nExporting Text Encoder to {text_onnx_path}...")

torch.onnx.export(
    text_encoder,
    DUMMY_TEXT_INPUT,
    text_onnx_path,
    input_names=["text"],
    output_names=["text_embedding"],
    opset_version=18,
    do_constant_folding=True,
    dynamic_axes=None,
    verbose=False,
    export_params=True,
    training=torch.onnx.TrainingMode.EVAL,
    dynamo=True,
)

print("\nExport complete.")
