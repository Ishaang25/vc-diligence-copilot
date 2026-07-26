"""
Disk-based caching layer.
"""
from __future__ import annotations
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any
import streamlit as st
from observability import log_cache_hit

logger = logging.getLogger("dd_copilot.cache")
CACHE_DIR = Path(".dd_cache")

def init_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if "cache_hits" not in st.session_state:
        st.session_state.cache_hits = 0

def cache_key(*parts: Any) -> str:
    combined = "|".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

def disk_cache_get(key: str) -> Any | None:
    path = CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            log_cache_hit()
            return data
    except Exception as exc:
        logger.warning("Failed to read cache entry %s: %s", key, exc)
        return None

def disk_cache_set(key: str, value: Any) -> None:
    init_cache()
    path = CACHE_DIR / f"{key}.pkl"
    try:
        with open(path, "wb") as f:
            pickle.dump(value, f)
    except Exception as exc:
        logger.warning("Failed to write cache entry %s: %s", key, exc)

def clear_cache() -> None:
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.pkl"):
            try: f.unlink()
            except OSError: pass

def get_cache_size() -> int:
    if not CACHE_DIR.exists(): return 0
    return sum(f.stat().st_size for f in CACHE_DIR.glob("*.pkl"))

def get_cache_entry_count() -> int:
    if not CACHE_DIR.exists(): return 0
    return len(list(CACHE_DIR.glob("*.pkl")))
