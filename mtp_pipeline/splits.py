from __future__ import annotations

import random
from pathlib import Path

from .protocol_3dssg import OCRL_INVALID_ALIGNED_SCAN_IDS


def read_scan_list(path: str | Path | None) -> list[str]:
    """Read one reference scan ID per line."""
    if path is None:
        return []
    scan_path = Path(path)
    if not scan_path.exists():
        return []
    return [line.strip() for line in scan_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _reference_scan_id(scene_token: str) -> str:
    """Map an official scan_split token back to the ID stored in split files."""
    base, separator, suffix = scene_token.rpartition("_")
    if separator and suffix.isdigit():
        return base
    return scene_token


def get_3rscan_splits(
    scene_tokens: list[str],
    train_scans: str | Path | None = None,
    val_scans: str | Path | None = None,
    seed: int = 42,
    exclude_ocr_invalid_scan: bool = False,
) -> tuple[list[str], list[str]]:
    """Return the official split, including every scan_split annotation entry.

    The fixed text files contain 1,178/157 reference scan IDs. The official JSON
    expands each reference ID into one or more annotation versions named
    scan_id_split_number; the official repository trains/evaluates those entries.
    """
    official_train = set(read_scan_list(train_scans))
    official_val = set(read_scan_list(val_scans))
    if official_train and official_val:
        excluded = OCRL_INVALID_ALIGNED_SCAN_IDS if exclude_ocr_invalid_scan else frozenset()
        train_tokens = [
            token
            for token in scene_tokens
            if _reference_scan_id(token) in official_train and _reference_scan_id(token) not in excluded
        ]
        val_tokens = [token for token in scene_tokens if _reference_scan_id(token) in official_val]
        if train_tokens or val_tokens:
            return train_tokens, val_tokens

    tokens = sorted(scene_tokens)
    if len(tokens) < 2:
        return tokens, []
    rng = random.Random(seed)
    rng.shuffle(tokens)
    split_index = max(1, int(len(tokens) * 0.8))
    return tokens[:split_index], tokens[split_index:]
