from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


@lru_cache(maxsize=1)
def get_gemini_client():
    """
    Gemini 客户端（通过 GPTSAPI 中转）
    使用新版 google.genai SDK，旧版 google.generativeai 已废弃
    """
    from google import genai

    client = genai.Client(
        api_key=os.environ["GPTSAPI_KEY"],
        http_options={"base_url": "https://api.gptsapi.net"},
    )
    return client


@lru_cache(maxsize=1)
def get_qwen_client() -> AsyncOpenAI:
    """
    Qwen 客户端（Dashscope OpenAI 兼容模式，直连）
    用于：Audio Analyzer Round2 / Edit Planner / Editor / Reviewer 叙事质量
    """
    return AsyncOpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


@lru_cache(maxsize=1)
def get_openai_client() -> AsyncOpenAI:
    """
    OpenAI 客户端（jiekou.ai 中转）
    用于：Audio Analyzer L3 语义层（GPT-5.5）
    """
    return AsyncOpenAI(
        api_key=os.environ["OPENAI_KEY"],
        base_url="https://api.jiekou.ai/openai",
    )


# ── 模型名常量 ────────────────────────────────────────────────────────────────

GEMINI_FLASH = "gemini-2.5-flash"       # Audio Analyzer Round1 / Video Tagger (Gemini 路径)
QWEN_MAX = "qwen-max"                    # Edit Planner / Reviewer 叙事质量 / Manager
QWEN_PLUS = "qwen-plus"                 # Audio Analyzer Round2 / Editor Agent
QWEN_OMNI = "qwen-omni-turbo"           # 备用音频分析（若 Gemini 音频不可用时）
QWEN_VL = "qwen3-vl-plus"              # Video Tagger (Qwen3-VL 路径)
GPT_5_5 = "gpt-5.5"                     # Audio Analyzer L3 语义层（jiekou.ai 中转）