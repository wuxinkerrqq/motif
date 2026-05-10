from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from utils.embedder import build_scene_text, embed_texts


def build_scene_index(
    scene_table_path: str,
    output_dir: str | None = None,
) -> tuple[np.ndarray, list[int]]:
    """
    读取 scene_table JSON，批量计算所有场景的 embedding，保存为 .npy 文件。

    返回：
        embeddings: shape (N, 512) 的向量矩阵
        scene_ids: 每行对应的 scene_id 列表
    """
    scene_table_p = Path(scene_table_path)
    scenes = json.load(open(scene_table_p, encoding="utf-8"))

    if output_dir is None:
        output_dir = str(scene_table_p.parent)

    stem = scene_table_p.stem  # e.g. "edgerunners_scene_table"
    embeddings_path = Path(output_dir) / f"{stem}_embeddings.npy"
    ids_path = Path(output_dir) / f"{stem}_ids.json"

    # 如果已经有缓存，直接加载
    if embeddings_path.exists() and ids_path.exists():
        logger.info(f"[scene_embedder] 加载缓存: {embeddings_path.name}")
        embeddings = np.load(str(embeddings_path))
        scene_ids = json.load(open(ids_path, encoding="utf-8"))
        return embeddings, scene_ids

    logger.info(f"[scene_embedder] 开始计算 {len(scenes)} 个场景的 embedding...")

    texts = [build_scene_text(s) for s in scenes]
    scene_ids = [s["scene_id"] for s in scenes]

    embeddings = embed_texts(texts)

    # 保存缓存
    np.save(str(embeddings_path), embeddings)
    json.dump(scene_ids, open(ids_path, "w", encoding="utf-8"))

    logger.info(f"[scene_embedder] 完成，shape={embeddings.shape}，已保存到 {embeddings_path.name}")
    return embeddings, scene_ids


def build_clip_index(
    scene_table_path: str,
    output_dir: str | None = None,
    force: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """
    用 CLIP image encoder 给每个场景的第一帧关键帧编码，保存为 _clip_embeddings.npy。

    返回：
        clip_embeddings: shape (N, 512)
        scene_ids: 每行对应的 scene_id
    """
    from utils.clip_embedder import embed_images_clip

    scene_table_p = Path(scene_table_path)
    scenes = json.load(open(scene_table_p, encoding="utf-8"))

    if output_dir is None:
        output_dir = str(scene_table_p.parent)

    stem = scene_table_p.stem
    embeddings_path = Path(output_dir) / f"{stem}_clip_embeddings.npy"
    ids_path = Path(output_dir) / f"{stem}_clip_ids.json"

    if not force and embeddings_path.exists() and ids_path.exists():
        logger.info(f"[scene_embedder] 加载 CLIP 缓存: {embeddings_path.name}")
        embeddings = np.load(str(embeddings_path))
        scene_ids = json.load(open(ids_path, encoding="utf-8"))
        return embeddings, scene_ids

    logger.info(f"[scene_embedder] 开始计算 {len(scenes)} 个场景的 CLIP embedding...")

    # 每个场景取第一帧关键帧
    image_paths = []
    scene_ids = []
    for s in scenes:
        kfs = s.get("keyframes", [])
        image_paths.append(kfs[0] if kfs else "")
        scene_ids.append(s["scene_id"])

    embeddings = embed_images_clip(image_paths)

    np.save(str(embeddings_path), embeddings)
    json.dump(scene_ids, open(ids_path, "w", encoding="utf-8"))

    logger.info(f"[scene_embedder] CLIP 完成，shape={embeddings.shape}，已保存到 {embeddings_path.name}")
    return embeddings, scene_ids


def load_clip_index(
    scene_table_path: str,
    output_dir: str | None = None,
) -> tuple[np.ndarray, list[int]] | tuple[None, None]:
    """加载已有的 CLIP embedding 缓存，不存在则返回 (None, None)。"""
    scene_table_p = Path(scene_table_path)
    if output_dir is None:
        output_dir = str(scene_table_p.parent)

    stem = scene_table_p.stem
    embeddings_path = Path(output_dir) / f"{stem}_clip_embeddings.npy"
    ids_path = Path(output_dir) / f"{stem}_clip_ids.json"

    if not embeddings_path.exists() or not ids_path.exists():
        return None, None

    embeddings = np.load(str(embeddings_path))
    scene_ids = json.load(open(ids_path, encoding="utf-8"))
    return embeddings, scene_ids


def load_scene_index(
    scene_table_path: str,
    output_dir: str | None = None,
) -> tuple[np.ndarray, list[int]] | tuple[None, None]:
    """
    加载已有的 scene embedding 缓存，不存在则返回 (None, None)。
    """
    scene_table_p = Path(scene_table_path)
    if output_dir is None:
        output_dir = str(scene_table_p.parent)

    stem = scene_table_p.stem
    embeddings_path = Path(output_dir) / f"{stem}_embeddings.npy"
    ids_path = Path(output_dir) / f"{stem}_ids.json"

    if not embeddings_path.exists() or not ids_path.exists():
        return None, None

    embeddings = np.load(str(embeddings_path))
    scene_ids = json.load(open(ids_path, encoding="utf-8"))
    return embeddings, scene_ids