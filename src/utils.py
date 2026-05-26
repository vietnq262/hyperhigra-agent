"""Shared utilities for HyperHiGraAgent."""

from __future__ import annotations

import json
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, TypeVar

import yaml

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover - optional until dependencies are installed
    unidecode = None

F = TypeVar("F", bound=Callable[..., Any])


def load_json(path: str | Path) -> Any:
    """Load JSON data from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    """Save JSON data to disk, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def flatten(values: Iterable[Iterable[Any]]) -> List[Any]:
    """Flatten a nested iterable into a list."""
    return [item for group in values for item in group]


def deduplicate(values: Sequence[Any]) -> List[Any]:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def normalize_entity(value: str) -> str:
    """Normalize an entity surface form for matching."""
    text = (value or "").strip().lower()
    if unidecode is not None:
        text = unidecode(text)
    return " ".join(text.split())


def timer(func: F) -> F:
    """Simple timing decorator that prints runtime."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"[timer] {func.__name__} finished in {duration:.2f}s")
        return result

    return wrapper  # type: ignore[return-value]
