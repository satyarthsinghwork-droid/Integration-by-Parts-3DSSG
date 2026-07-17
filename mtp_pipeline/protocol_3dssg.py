from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# The official OCRL loader skips this scan because the V2 aligned PLY and its
# segmentation annotations disagree. Seven training graph entries use this ID.
OCRL_INVALID_ALIGNED_SCAN_IDS = frozenset({"fa79392f-7766-2d5c-869a-f5d6cfb62fc6"})


def load_official_objects(official_subset_dir: Path) -> dict[str, dict[str, str]]:
    """Return the union of objects used by the official train/validation files."""
    objects_by_scan: dict[str, dict[str, str]] = {}
    for filename in ("relationships_train.json", "relationships_validation.json"):
        path = official_subset_dir / filename
        annotations = json.loads(path.read_text(encoding="utf-8"))
        for scan in annotations.get("scans", []):
            scan_objects = objects_by_scan.setdefault(str(scan["scan"]), {})
            for object_id, category in scan.get("objects", {}).items():
                scan_objects[str(object_id)] = str(category)
    return objects_by_scan


def official_objects_manifest(official_subset_dir: Path) -> dict[str, Any]:
    """Build the legacy object-manifest shape directly from official annotations."""
    objects_by_scan = load_official_objects(official_subset_dir)
    return {
        "source": "included OCRL/3DSSG official train and validation annotations",
        "scans": [
            {
                "scan": scan_id,
                "objects": [
                    {"id": object_id, "label": category}
                    for object_id, category in sorted(
                        scan_objects.items(),
                        key=lambda item: int(item[0]),
                    )
                ],
            }
            for scan_id, scan_objects in sorted(objects_by_scan.items())
        ],
    }


def write_official_objects_manifest(official_subset_dir: Path, output_path: Path) -> Path:
    """Materialize the generated manifest used by CLIP text and RGB extraction."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(official_objects_manifest(official_subset_dir), indent=2),
        encoding="utf-8",
    )
    return output_path
