"""
Disk cache for LLM responses.
Keyed deterministically by sha256(model, prompt, params).
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class DiskCache:
    """
    Persistent disk cache for LLM responses to ensure deterministic replay,
    zero-cost re-runs, and instant test execution.
    """

    def __init__(self, cache_dir: str | Path = ".cache/llm"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_key(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Compute a deterministic sha256 hash key across model, messages, and generation params.
        """
        params_to_hash = params or {}
        # Ensure stable serialization by sorting keys
        payload = {
            "model": model,
            "messages": messages,
            "params": params_to_hash,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _file_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if present on disk."""
        target = self._file_path(key)
        if not target.is_file():
            return None
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, data: Dict[str, Any]) -> None:
        """Persist response data to disk atomically."""
        target = self._file_path(key)
        tmp_target = self._file_path(f"{key}.tmp")
        try:
            with open(tmp_target, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_target, target)
        except OSError:
            if tmp_target.exists():
                try:
                    tmp_target.unlink()
                except OSError:
                    pass

    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self._file_path(key).is_file()

    def clear(self) -> None:
        """Delete all cached entries."""
        for p in self.cache_dir.glob("*.json"):
            try:
                p.unlink()
            except OSError:
                pass
