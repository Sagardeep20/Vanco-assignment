"""Local file access for indexed assets (preview + open-original).

Security model: callers reference assets by database ID only — never by
filesystem path. The stored ``original_path`` is resolved and required to
stay inside the configured dataset directory; anything else is rejected
before any file is touched. No shell is ever used.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import subprocess
import sys
from pathlib import Path

from app.services.scanner import resolve_dataset_dir

logger = logging.getLogger(__name__)


def resolve_dataset_file(original_path: str) -> Path:
    """Resolve an asset's stored path, confined to the dataset directory.

    Returns the resolved file path. Raises FileNotFoundError when the
    file does not exist and ValueError when the path escapes the dataset
    directory (path traversal protection).
    """
    root = resolve_dataset_dir().resolve()
    candidate = Path(original_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        inside = resolved.is_relative_to(root)
    except (OSError, ValueError):
        inside = False
    if not inside:
        raise ValueError(
            "Asset file is outside the configured dataset directory; "
            "access refused."
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Asset file not found: {resolved.name}")
    return resolved


def guess_media_type(filename: str, stored: str | None = None) -> str:
    """Best-effort media type: stored DB value, else extension guess."""
    if stored:
        return stored
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def open_in_os(path: Path) -> None:
    """Open a file with the OS default application (local desktop use).

    Windows: os.startfile; macOS: ``open``; Linux: ``xdg-open``.
    No shell, fixed command + validated path argument only.
    """
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # noqa: S606 (Windows-only, no shell)
        return
    command = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.run([command, str(path)], check=True, timeout=30)
