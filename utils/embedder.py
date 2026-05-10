from __future__ import annotations

import os
import time
from functools import lru_cache

import numpy as np
from loguru import logger

# Dashscope 需要显式赋值 api_key
import dashscope
from dotenv import load_dotenv

load_dotenv()
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIM = 512          # 512 维足够，省 token
BATCH_SIZE = 10              # Dashscope 单次最多 25 条


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    批量计算文本向量，返回 shape=(N, EMBEDDING_DIM) 的 numpy 数组。
    自动分批，每批最多 BATCH_SIZE 条。
    """
    from dashscope import TextEmbedding

    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        retry = 0
        while retry < 3:
            try:
                resp = TextEmbedding.call(
                    model=EMBEDDING_MODEL,
                    input=batch,
                    dimension=EMBEDDING_DIM,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Dashscope 返回错误: {resp.status_code} {resp.message}")

                # 按 text_index 排序，保证顺序和输入一致
                embeddings = sorted(
                    resp.output["embeddings"],
                    key=lambda x: x["text_index"],
                )
                all_embeddings.extend([e["embedding"] for e in embeddings])
                break

            except Exception as e:
                retry += 1
                logger.warning(f"  [embedder] 第 {i//BATCH_SIZE+1} 批失败 (retry {retry}/3): {e}")
                if retry < 3:
                    time.sleep(1)
                else:
                    # 兜底：用零向量占位，不让整个流程崩溃
                    logger.error(f"  [embedder] 批次彻底失败，用零向量占位")
                    all_embeddings.extend([[0.0] * EMBEDDING_DIM] * len(batch))

    return np.array(all_embeddings, dtype=np.float32)


def cosine_similarity_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    计算 query (M, D) 和 corpus (N, D) 的余弦相似度矩阵，返回 (M, N)。
    """
    query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    return query_norm @ corpus_norm.T


def top_k_similar(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int = 10,
    exclude_ids: set[int] | None = None,
    id_list: list[int] | None = None,
) -> list[tuple[int, float]]:
    """
    找到与 query 最相似的 Top-K 个向量。

    参数：
        query_embedding: shape (D,) 单个查询向量
        corpus_embeddings: shape (N, D) 语料库向量
        k: 返回前 K 个
        exclude_ids: 要排除的索引集合（对应 id_list 里的 id）
        id_list: corpus 中每个向量对应的 scene_id 列表

    返回：[(scene_id, similarity_score), ...]
    """
    sims = cosine_similarity_matrix(
        query_embedding.reshape(1, -1),
        corpus_embeddings
    )[0]

    if exclude_ids and id_list:
        for i, sid in enumerate(id_list):
            if sid in exclude_ids:
                sims[i] = -1.0  # 排除已用场景

    top_indices = np.argsort(sims)[::-1][:k]

    if id_list:
        return [(id_list[i], float(sims[i])) for i in top_indices if sims[i] > -0.5]
    else:
        return [(int(i), float(sims[i])) for i in top_indices if sims[i] > -0.5]


def build_scene_text(scene: dict) -> str:
    """
    把 scene_table 中的一条记录转成用于 embedding 的文本。
    综合情绪、动感强度、场景描述三个维度。
    """
    mood = scene.get("mood") or scene.get("editing_metrics", {}).get("emotion_mood", "")
    density = scene.get("density") or scene.get("editing_metrics", {}).get("action_density", 5)
    desc = scene.get("desc") or scene.get("scene_description", "")

    # 把 action_density 数值映射成自然语言，增强语义
    density_text = _density_to_text(int(density))

    return f"情绪：{mood}，动感：{density_text}，画面：{desc}"


def build_segment_text(segment: dict) -> str:
    """
    把音频段落转成用于 embedding 的查询文本。
    """
    mood = segment.get("mood", "")
    energy = segment.get("energy", 5)
    description = segment.get("description", "")
    name = segment.get("name", "")

    energy_text = _density_to_text(int(energy))

    base = f"情绪：{mood}，能量：{energy_text}，段落：{name}"
    if description:
        base += f"，描述：{description[:50]}"
    return base


def _density_to_text(density: int) -> str:
    mapping = {
        1: "极静止", 2: "静止", 3: "轻微动作",
        4: "缓慢运动", 5: "适中动作", 6: "较活跃",
        7: "激烈动作", 8: "高强度战斗", 9: "极限爆发", 10: "最高强度"
    }
    return mapping.get(density, "适中动作")