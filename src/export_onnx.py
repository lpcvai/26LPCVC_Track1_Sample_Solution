import argparse
import os

import open_clip
import torch
from timm.utils import reparameterize_model

from utils import MODELS, RESULTS_PATH, NUM_IMAGE_SAMPLES, NUM_TEXT_SAMPLES, K

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--model", choices=MODELS.keys(), help="Single model to export")
group.add_argument("--all", action="store_true", help="Export all models")
args = parser.parse_args()

device = torch.device("cpu")  # use CPU to avoid GPU device issues during export


class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model.encode_image(images)


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, token_ids):
        return self.model.encode_text(token_ids.to(torch.int64))


class TopKWrapper(torch.nn.Module):
    """Computes cosine similarity and returns top-k text indices for each image."""
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, image_embs, text_embs):
        # image_embs: [N_img, D], text_embs: [N_txt, D], both L2-normalised
        sims = image_embs @ text_embs.T              # [N_img, N_txt]
        return torch.topk(sims, self.k, dim=-1).indices.to(torch.int32)  # [N_img, K]


def export_model(model_name: str, pretrained: str):
    onnx_dir = os.path.join(RESULTS_PATH, "onnx", model_name)
    os.makedirs(onnx_dir, exist_ok=True)
    print(f"\n── {model_name} ──")
    print(f"Loading model (pretrained={pretrained})...")

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(torch.float32)  # consistent with input type for QAI Hub compiling
    model.eval()
    model = reparameterize_model(model)
    model = model.to(device)

    image_size = model.visual.image_size
    if isinstance(image_size, (tuple, list)):
        image_h, image_w = image_size
    else:
        image_h = image_w = image_size

    dummy_image = torch.rand(1, 3, image_h, image_w, dtype=torch.float32, device=device)
    dummy_text = torch.randint(0, model.vocab_size, (1, model.context_length), dtype=torch.int32, device=device)

    with torch.no_grad():
        embed_dim = model.encode_image(dummy_image).shape[-1]

    image_encoder = ImageEncoderWrapper(model).eval()
    text_encoder  = TextEncoderWrapper(model).eval()
    topk_model    = TopKWrapper(k=K).eval()

    image_onnx_path = os.path.join(onnx_dir, "image_encoder.onnx")
    print(f"Exporting image encoder → {image_onnx_path}...")
    torch.onnx.export(
        image_encoder,
        dummy_image,
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

    text_onnx_path = os.path.join(onnx_dir, "text_encoder.onnx")
    print(f"Exporting text encoder  → {text_onnx_path}...")
    torch.onnx.export(
        text_encoder,
        dummy_text,
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
    topk_onnx_path = os.path.join(onnx_dir, "topk_retrieval.onnx")
    dummy_image_embs = torch.rand(NUM_IMAGE_SAMPLES, embed_dim, dtype=torch.float32, device=device)
    dummy_text_embs  = torch.rand(NUM_TEXT_SAMPLES,  embed_dim, dtype=torch.float32, device=device)
    print(f"Exporting top-k retrieval  → {topk_onnx_path}...")
    torch.onnx.export(
        topk_model,
        (dummy_image_embs, dummy_text_embs),
        topk_onnx_path,
        input_names=["image_embs", "text_embs"],
        output_names=["topk_indices"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=True,
    )
    print(f"Export complete for {model_name}.")


targets = MODELS if args.all else {args.model: MODELS[args.model]}
for model_name, pretrained in targets.items():
    export_model(model_name, pretrained)
