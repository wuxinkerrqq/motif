"""
 clip_embedder.py — 基于 CLIP 的视觉 embedding 工具

 用 CLIP image encoder 给场景关键帧编码，用 CLIP text encoder 给查询编码。
 两者在同一个 embedding 空间，可以直接做余弦相似度比较。

 模型：openai/clip-vit-base-patch32（512 维）
 """
from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger

CLIP_MODEL_ID = r"D:\Python_Programes\motif\clip-vit-base-patch32"
CLIP_DIM = 512
CLIP_BATCH_SIZE = 16


@lru_cache(maxsize=1)
def _load_clip():
    """懒加载 CLIP 模型，只加载一次。"""
    from transformers import CLIPModel, CLIPProcessor
    import torch

    logger.info(f"[CLIP] 加载模型 {CLIP_MODEL_ID}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    logger.info(f"[CLIP] 模型加载完成，device={device}")
    return model, processor, device


def embed_images_clip(image_paths: list[str]) -> np.ndarray:
    """
    用 CLIP image encoder 批量编码图片，返回 shape=(N, 512) 的向量矩阵。
    无法加载的图片用零向量占位。
    """
    import torch
    from PIL import Image

    model, processor, device = _load_clip()
    all_embeddings = []

    for i in range(0, len(image_paths), CLIP_BATCH_SIZE):
        batch_paths = image_paths[i: i + CLIP_BATCH_SIZE]
        images = []
        valid_mask = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_mask.append(True)
            except Exception as e:
                logger.warning(f"[CLIP] 无法加载图片 {p}: {e}，用零向量占位")
                valid_mask.append(False)

        if not images:
            all_embeddings.extend([[0.0] * CLIP_DIM] * len(batch_paths))
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=pixel_values)
            feats = model.visual_projection(vision_out.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy()

        feat_iter = iter(feats)
        for valid in valid_mask:
            if valid:
                all_embeddings.append(next(feat_iter).tolist())
            else:
                all_embeddings.append([0.0] * CLIP_DIM)

    return np.array(all_embeddings, dtype=np.float32)


def embed_texts_clip(texts: list[str]) -> np.ndarray:
    """
    用 CLIP text encoder 批量编码文本，返回 shape=(N, 512) 的向量矩阵。
    与 embed_images_clip 在同一 embedding 空间，可直接做相似度比较。
    """
    import torch

    model, processor, device = _load_clip()
    all_embeddings = []

    for i in range(0, len(texts), CLIP_BATCH_SIZE):
        batch = texts[i: i + CLIP_BATCH_SIZE]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            text_out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            feats = model.text_projection(text_out.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy()

        all_embeddings.extend(feats.tolist())

    return np.array(all_embeddings, dtype=np.float32)