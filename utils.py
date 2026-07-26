"""
Utility functions including AsyncOpenAI client, RAG helpers, and benchmark loading.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import time
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from openai import AsyncOpenAI
from PIL import Image
import streamlit as st

from observability import log_api_call, log_cache_hit

T = TypeVar("T")
logger = logging.getLogger("dd_copilot")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_handler)

# Load external benchmarks
BENCHMARKS = {}
try:
    bench_path = Path(__file__).parent / "benchmarks.json"
    if bench_path.exists():
        BENCHMARKS = json.loads(bench_path.read_text())
except Exception as e:
    logger.error(f"Failed to load benchmarks.json: {e}")

def hash_content(content: bytes | str) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()

def get_async_openai_client(api_key: str) -> AsyncOpenAI:
    if not api_key or not api_key.strip():
        raise ValueError("OpenAI API key is required.")
    return AsyncOpenAI(api_key=api_key.strip())

def parse_json_safely(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for marker in ("```json", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.find("```", start)
            if end != -1:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError(f"Could not parse JSON from response: {text[:300]}...", text, 0)

async def call_openai_json_async(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 4000,
    agent_name: str = "Unknown"
) -> dict[str, Any]:
    start_time = time.time()
    
    # Exponential backoff for rate limits
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = time.time() - start_time
            content = response.choices[0].message.content
            
            usage = response.usage
            if usage:
                log_api_call(
                    model=model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    latency=latency,
                    agent_name=agent_name
                )
            
            if content is None:
                raise json.JSONDecodeError("Empty response from model", "", 0)
            return parse_json_safely(content)
            
        except Exception as exc:
            if "rate_limit_exceeded" in str(exc).lower() and attempt < max_retries - 1:
                delay = (2 ** attempt) + 2.0  # 3s, 5s, 9s
                logger.warning(f"Rate limit hit for {agent_name}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"API call failed for {agent_name}: {exc}")
                raise

async def get_embeddings(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    """Fetch embeddings for a list of texts."""
    start_time = time.time()
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    latency = time.time() - start_time
    
    # Log embedding usage (approximate)
    total_tokens = sum(len(t.split()) for t in texts) * 1.3  # rough estimate
    log_api_call(
        model="text-embedding-3-small",
        prompt_tokens=int(total_tokens),
        completion_tokens=0,
        latency=latency,
        agent_name="Embeddings"
    )
    
    return [d.embedding for d in response.data]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))

def image_to_base64(image: Image.Image, fmt: str = "JPEG", quality: int = 75) -> str:
    buffer = BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffer, format=fmt, quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def prepare_image_for_api(image: Image.Image, max_size: int = 1536) -> Image.Image:
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def safe_float(value: Any) -> float | None:
    if value is None: return None
    if isinstance(value, (int, float)): return float(value)
    try:
        cleaned = str(value).replace("$", "").replace(",", "").replace("%", "").strip()
        return float(cleaned)
    except (ValueError, TypeError): return None

def safe_int(value: Any) -> int | None:
    f = safe_float(value)
    return int(f) if f is not None else None

def format_currency(value: float | None) -> str:
    if value is None: return "N/A"
    if abs(value) >= 1_000_000_000: return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000: return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000: return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"

def format_percentage(value: float | None) -> str:
    if value is None: return "N/A"
    return f"{value:.1f}%"

# import asyncio here to avoid circular imports if utils is imported before asyncio
import asyncio
