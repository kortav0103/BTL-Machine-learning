"""Shared configuration loading for the vision pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Return a config dict from a mapping or a JSON file path."""
    if isinstance(config, Mapping):
        return dict(config)
    if isinstance(config, (str, Path)):
        path = Path(config)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    raise TypeError(f"Expected a mapping or a path, got {type(config).__name__}")
