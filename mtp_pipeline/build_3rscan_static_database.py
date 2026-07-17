from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .config import ProjectPaths
from .protocol_3dssg import OCRL_INVALID_ALIGNED_SCAN_IDS


LABEL_SPACE_FILE = "label_space.json"


def load_token_file(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _select_static_rgb_tokens(
    rgb_scene: dict[str, Any] | None,
    max_views: int = 1,
    tokens_per_view: int = 49,
    feature_dim: int = 768,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Pack top visible RGB views into fixed-size tokens plus a validity mask."""
    candidates: dict[str, list[tuple[float, int, torch.Tensor]]] = defaultdict(list)
    if not rgb_scene:
        return {}
    for frame_index, frame in enumerate(rgb_scene.get("frames", [])):
        for obj in frame.get("objects", []):
            tokens = obj.get("patch_tokens")
            if tokens is None:
                continue
            bbox = obj.get("bbox", [0.0, 0.0, 0.0, 0.0])
            area = max(float(bbox[2]) - float(bbox[0]), 0.0) * max(float(bbox[3]) - float(bbox[1]), 0.0)
            candidates[str(obj["instance_token"])].append((area, frame_index, tokens.float().cpu()))
    packed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    total_tokens = max_views * tokens_per_view
    for instance, views in candidates.items():
        output = torch.zeros(total_tokens, feature_dim, dtype=torch.float32)
        mask = torch.zeros(total_tokens, dtype=torch.bool)
        for view_index, (_area, _frame, tokens) in enumerate(sorted(views, key=lambda item: (-item[0], item[1]))[:max_views]):
            usable = min(tokens.size(0), tokens_per_view)
            start = view_index * tokens_per_view
            output[start : start + usable, : min(tokens.size(1), feature_dim)] = tokens[:usable, :feature_dim]
            mask[start : start + usable] = True
        packed[instance] = (output, mask)
    return packed

def _load_official_scans(official_subset_dir: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Load the official OCRL 3DSSG subset and retain its canonical ordering."""
    object_labels = _read_lines(official_subset_dir / "classes.txt")
    relation_labels = _read_lines(official_subset_dir / "relations.txt")
    if len(object_labels) != 160 or len(relation_labels) != 26:
        raise ValueError(
            "Expected the official OCRL 3DSSG subset (160 object classes and 26 positive relations)."
        )

    selected_by_split = {
        "train": set(_read_lines(official_subset_dir / "train_scans.txt")),
        "validation": set(_read_lines(official_subset_dir / "validation_scans.txt")),
    }
    scans: list[dict[str, Any]] = []
    for split, filename in (("train", "relationships_train.json"), ("validation", "relationships_validation.json")):
        with open(official_subset_dir / filename, "r", encoding="utf-8") as handle:
            annotations = json.load(handle)
        for scan in annotations.get("scans", []):
            if scan.get("scan") in selected_by_split[split]:
                scans.append({**scan, "official_split": split, "scene_token": f"{scan['scan']}_{scan['split']}"})
    return scans, object_labels, relation_labels


def _legacy_scans(objects_json: Path, relationships_json: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Fallback for older local databases; it is not an official comparison protocol."""
    with open(objects_json, "r", encoding="utf-8") as handle:
        object_data = json.load(handle)
    with open(relationships_json, "r", encoding="utf-8") as handle:
        relationship_data = json.load(handle)

    relationships_by_scan: dict[str, list[list[Any]]] = {}
    for scan in relationship_data.get("scans", []):
        relationships_by_scan[str(scan["scan"])] = list(scan.get("relationships", []))

    scans = []
    object_labels = set()
    relation_labels = set()
    for scan in object_data.get("scans", []):
        scan_id = str(scan["scan"])
        objects = {str(obj["id"]): obj.get("label", "unknown") for obj in scan.get("objects", [])}
        relationships = relationships_by_scan.get(scan_id, [])
        object_labels.update(objects.values())
        relation_labels.update(str(rel[3]) for rel in relationships)
        scans.append({"scan": scan_id, "objects": objects, "relationships": relationships, "official_split": "legacy"})
    return scans, sorted(object_labels), sorted(relation_labels)


def build_3rscan_static_scene_database(
    objects_json: Path | None,
    relationships_json: Path | None,
    lidar_dir: Path,
    text_dir: Path,
    output_dir: Path,
    rgb_dir: Path | None = None,
    rgb_fallback_tokens: int = 49,
    rgb_dim: int = 768,
    rgb_views_per_object: int = 1,
    official_subset_dir: Path | None = None,
    exclude_ocr_invalid_scan: bool = False,
    require_aligned_lidar: bool = False,
    refresh_if_inputs_newer: bool = False,
) -> int:
    """Build static scenes using the official OCRL/3DSSG labels when supplied.

    The official subset stores more than one positive predicate for some directed
    object pairs. Every relationship entry is retained here; the multi-hot target
    is constructed later by the graph builder.
    """
    if official_subset_dir is not None:
        scans, object_labels, relation_labels = _load_official_scans(official_subset_dir)
        if exclude_ocr_invalid_scan:
            scans = [scan for scan in scans if str(scan["scan"]) not in OCRL_INVALID_ALIGNED_SCAN_IDS]
        protocol = "OCRL-3DSSG official subset: 160 objects, 26 positive multi-label relations"
    else:
        if objects_json is None or relationships_json is None:
            raise ValueError("objects_json and relationships_json are required without --official-subset-dir.")
        scans, object_labels, relation_labels = _legacy_scans(objects_json, relationships_json)
        protocol = "legacy local labels; not comparable to the OCRL 3DSSG protocol"

    if rgb_views_per_object < 1:
        raise ValueError("rgb_views_per_object must be at least one.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / LABEL_SPACE_FILE).write_text(
        json.dumps(
            {
                "protocol": protocol,
                "object_labels": object_labels,
                "relation_labels": relation_labels,
                "num_object_classes": len(object_labels),
                "num_relation_classes": len(relation_labels),
                "reference_scan_groups": len({str(scan["scan"]) for scan in scans}),
                "annotation_entries": len(scans),
                "excluded_scan_ids": sorted(OCRL_INVALID_ALIGNED_SCAN_IDS) if exclude_ocr_invalid_scan else [],
                "rgb_policy": f"top-{rgb_views_per_object} largest-visible RGB views per object; unavailable RGB is masked",
                "lidar_requirement": "official pretrained PointNet tokens with world-space point_stats",
                "lidar_coordinate_frame": "official reference-aligned" if require_aligned_lidar else "not enforced",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    allowed_objects = set(object_labels)
    allowed_relations = set(relation_labels)
    saved = 0
    skipped_without_objects = 0
    zero_rgb = torch.zeros(rgb_fallback_tokens * rgb_views_per_object, rgb_dim, dtype=torch.float32)
    zero_rgb_mask = torch.zeros(rgb_fallback_tokens * rgb_views_per_object, dtype=torch.bool)

    for scan in tqdm(scans, desc="Building official static 3DSSG database"):
        scan_id = str(scan["scan"])
        scene_token = str(scan.get("scene_token", scan_id))
        scene_path = output_dir / f"{scene_token}.pt"
        if scene_path.exists():
            should_refresh = False
            if refresh_if_inputs_newer:
                scene_mtime = scene_path.stat().st_mtime_ns
                source_paths = [
                    path
                    for raw_id, category_name in scan.get("objects", {}).items()
                    if str(category_name) in allowed_objects
                    for path in (
                        lidar_dir / f"{scan_id}_{raw_id}.pt",
                        text_dir / f"{scan_id}_{raw_id}.pt",
                    )
                ]
                if rgb_dir is not None:
                    source_paths.append(rgb_dir / f"{scan_id}.pt")
                should_refresh = any(
                    path.exists() and path.stat().st_mtime_ns > scene_mtime for path in source_paths
                )
            if not should_refresh:
                # A timed-out build can safely resume because each scene is self-contained.
                saved += 1
                continue
        rgb_by_instance: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if rgb_dir is not None:
            rgb_path = rgb_dir / f"{scan_id}.pt"
            if rgb_path.exists():
                rgb_by_instance = _select_static_rgb_tokens(
                    load_token_file(rgb_path),
                    max_views=rgb_views_per_object,
                    tokens_per_view=rgb_fallback_tokens,
                    feature_dim=rgb_dim,
                )

        objects_out: list[dict[str, Any]] = []
        for raw_id, category_name in scan.get("objects", {}).items():
            obj_id = str(raw_id)
            category_name = str(category_name)
            if category_name not in allowed_objects:
                continue
            lidar_file = lidar_dir / f"{scan_id}_{obj_id}.pt"
            text_file = text_dir / f"{scan_id}_{obj_id}.pt"
            if not lidar_file.exists() or not text_file.exists():
                continue

            lidar_data = load_token_file(lidar_file)
            text_data = load_token_file(text_file)
            point_stats = lidar_data.get("point_stats")
            if lidar_data.get("encoder_name") != "OCRL official PointNetEncoder" or not point_stats:
                raise ValueError(
                    f"{lidar_file} is not a valid official-pretrained LiDAR token file. "
                    "Run the corrected LiDAR extraction before building this database."
                )
            if require_aligned_lidar and lidar_data.get("coordinate_frame") != "official reference-aligned":
                raise ValueError(
                    f"{lidar_file} is not in the official reference-aligned coordinate frame. "
                    "Re-run LiDAR extraction with the official 3RScan.json metadata."
                )
            centroid_3d = torch.tensor(point_stats["center"], dtype=torch.float32)
            std_3d = torch.tensor(point_stats["std"], dtype=torch.float32)
            bbox_3d = torch.tensor(point_stats["bbox_size"], dtype=torch.float32)
            rgb_tokens, rgb_token_mask = rgb_by_instance.get(obj_id, (zero_rgb, zero_rgb_mask))
            objects_out.append(
                {
                    "instance_token": obj_id,
                    "category_name": category_name,
                    "bbox": [0, 0, 1, 1],
                    "patch_tokens": rgb_tokens.clone(),
                    "rgb_token_mask": rgb_token_mask.clone(),
                    "lidar_tokens": lidar_data["lidar_tokens"].float().cpu(),
                    "text_features": text_data["text_features"].float().cpu(),
                    "num_points": lidar_data.get("num_points", 0),
                    "centroid_3d": centroid_3d,
                    "std_3d": std_3d,
                    "bbox_3d": bbox_3d,
                    "point_stats": point_stats,
                    "has_rgb": obj_id in rgb_by_instance,
                }
            )

        if not objects_out:
            skipped_without_objects += 1
            continue

        relationships = []
        for rel in scan.get("relationships", []):
            if len(rel) < 4:
                continue
            subject_id, object_id, _predicate_id, predicate = rel[:4]
            predicate = str(predicate)
            if predicate not in allowed_relations:
                continue
            relationships.append(
                {
                    "source_instance": str(subject_id),
                    "target_instance": str(object_id),
                    "predicate": predicate,
                }
            )

        scene = {
            "scene_token": scene_token,
            "official_split": scan.get("official_split", "legacy"),
            "frames": [
                {
                    "sample_token": f"{scene_token}_static",
                    "timestamp": 0,
                    "objects": objects_out,
                }
            ],
            "relationships": relationships,
        }
        torch.save(scene, scene_path)
        saved += 1

    print(f"Saved {saved}/{len(scans)} scenes; skipped {skipped_without_objects} with no usable object embeddings.")
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static database for the official OCRL/3DSSG protocol.")
    parser.add_argument("--objects-json", type=Path, default=None)
    parser.add_argument("--relationships-json", type=Path, default=None)
    parser.add_argument("--official-subset-dir", type=Path, default=Path("official_splits"))
    parser.add_argument("--exclude-ocr-invalid-scan", action="store_true")
    parser.add_argument("--require-aligned-lidar", action="store_true")
    parser.add_argument("--refresh-if-inputs-newer", action="store_true")
    parser.add_argument("--lidar-dir", type=Path, default=ProjectPaths().reference_root / "3rscan_pointnet_pretrained_tokens")
    parser.add_argument("--text-dir", type=Path, default=ProjectPaths().reference_root / "3rscan_text_embeddings")
    parser.add_argument("--rgb-dir", type=Path, default=ProjectPaths().reference_root / "3rscan_rgb_temporal")
    parser.add_argument("--output", type=Path, default=ProjectPaths().output_root / "3rscan_official_static_database_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = build_3rscan_static_scene_database(
        objects_json=args.objects_json,
        relationships_json=args.relationships_json,
        lidar_dir=args.lidar_dir,
        text_dir=args.text_dir,
        rgb_dir=args.rgb_dir,
        output_dir=args.output,
        official_subset_dir=args.official_subset_dir,
        exclude_ocr_invalid_scan=args.exclude_ocr_invalid_scan,
        require_aligned_lidar=args.require_aligned_lidar,
        refresh_if_inputs_newer=args.refresh_if_inputs_newer,
    )
    print(f"Saved {saved} static scenes to {args.output}")


if __name__ == "__main__":
    main()
