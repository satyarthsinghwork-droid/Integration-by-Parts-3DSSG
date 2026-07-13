from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import os

import torch
from tqdm import tqdm

from .config import PipelineConfig, ProjectPaths


def load_token_file(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_3rscan_scene_database(
    rgb_dir: Path,
    lidar_dir: Path,
    text_dir: Path,
    relationships_json: Path,
    output_dir: Path
) -> int:
    # 1. Load ground truth relationships
    with open(relationships_json, "r") as f:
        rel_data = json.load(f)
        
    scan_relationships = {}
    for scan in rel_data.get("scans", []):
        scan_id = scan["scan"]
        rels = []
        for r in scan.get("relationships", []):
            # Format: [subject_id, object_id, predicate_id, predicate_name]
            subj_id, obj_id, pred_id, pred_name = r
            rels.append({
                "source_instance": str(subj_id),
                "target_instance": str(obj_id),
                "predicate": pred_name
            })
        scan_relationships[scan_id] = rels

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We iterate over the scans that have RGB temporal sequences extracted
    rgb_files = list(rgb_dir.glob("*.pt"))
    saved_count = 0
    
    for rgb_file in tqdm(rgb_files, desc="Building lazy-loading database"):
        scan_id = rgb_file.stem
        rgb_data = load_token_file(rgb_file)
        
        frames_out = []
        for frame in rgb_data.get("frames", []):
            timestamp = frame["timestamp"]
            objects_out = []
            
            for obj in frame.get("objects", []):
                obj_id = obj["instance_token"]
                
                lidar_file = lidar_dir / f"{scan_id}_{obj_id}.pt"
                text_file = text_dir / f"{scan_id}_{obj_id}.pt"
                
                # We need all 3 modalities to exist
                if not lidar_file.exists() or not text_file.exists():
                    continue
                    
                lidar_data = load_token_file(lidar_file)
                text_data = load_token_file(text_file)
                
                fps = lidar_data["fps_centers"].float().cpu()
                centroid_3d = fps.mean(dim=0)
                std_3d = fps.std(dim=0)
                bbox_3d = fps.max(dim=0)[0] - fps.min(dim=0)[0]
                
                objects_out.append({
                    "instance_token": obj_id,
                    "category_name": text_data["category_name"],
                    "bbox": obj["bbox"],
                    "patch_tokens": obj["patch_tokens"].float().cpu(),
                    "lidar_tokens": lidar_data["lidar_tokens"].float().cpu(),
                    "text_features": text_data["text_features"].float().cpu(),
                    "num_points": lidar_data["num_points"],
                    "centroid_3d": centroid_3d,
                    "std_3d": std_3d,
                    "bbox_3d": bbox_3d,
                    "point_stats": lidar_data.get("point_stats"),
                })
                
            if objects_out:
                frames_out.append({
                    "sample_token": f"{scan_id}_{timestamp}",
                    "timestamp": timestamp,
                    "objects": objects_out,
                })
                
        if frames_out:
            # Sort frames temporally
            frames_out.sort(key=lambda x: x["timestamp"])
            scene_dict = {
                "frames": frames_out,
                "relationships": scan_relationships.get(scan_id, [])
            }
            # Save individually to prevent RAM overflow!
            torch.save(scene_dict, output_dir / f"{scan_id}.pt")
            saved_count += 1
            
    return saved_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean 3RScan lazy-loading database.")
    parser.add_argument("--rgb-dir", type=Path, default=ProjectPaths().reference_root / "3rscan_rgb_temporal")
    parser.add_argument("--lidar-dir", type=Path, default=ProjectPaths().reference_root / "3rscan_lidar_tokens")
    parser.add_argument("--text-dir", type=Path, default=ProjectPaths().reference_root / "3rscan_text_embeddings")
    parser.add_argument("--relationships-json", type=Path, default=Path(r"D:\MTP_Project\3DSSG\relationships.json"))
    parser.add_argument("--output", type=Path, default=ProjectPaths().output_root / "3rscan_database")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_count = build_3rscan_scene_database(
        args.rgb_dir, args.lidar_dir, args.text_dir, args.relationships_json, args.output
    )
    print(f"Saved {saved_count} scenes individually to {args.output}")


if __name__ == "__main__":
    main()
