from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import ProjectPaths
from .lidar_utils import PointMAEObjectEncoder, farthest_point_sampling, group_points, preprocess_object_points


def prepare_3rscan_lidar_tokens(
    scan_dir: Path, 
    objects_json: Path,
    output_dir: Path, 
    checkpoint: Path | None, 
    device: str | None
) -> None:
    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Initialize the encoder
    encoder = PointMAEObjectEncoder().to(device_obj).eval()
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device_obj, weights_only=False)
        state_dict = state.get("base_model", state.get("model", state))
        encoder.load_state_dict(state_dict, strict=False)
        
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(objects_json, "r") as f:
        data = json.load(f)
    scans = data.get("scans", [])

    # Process each scan
    for scan in tqdm(scans, desc="Extracting LiDAR tokens for 3RScan objects"):
        scan_id = scan["scan"]
        scan_path = scan_dir / scan_id
        ply_path = scan_path / "labels.instances.annotated.v2.ply"
        
        if not ply_path.exists():
            continue  # Skip if scan not downloaded yet

        # Fast reading of PLY ascii file using pandas
        with open(ply_path, "r") as f:
            lines = f.readlines()
            
        header_end = 0
        num_vertices = 0
        for i, line in enumerate(lines):
            if line.startswith("element vertex"):
                num_vertices = int(line.strip().split()[-1])
            elif line.startswith("end_header"):
                header_end = i + 1
                break
                
        # Parse vertices: x, y, z, r, g, b, objectId, globalId, ...
        # We only need the first num_vertices lines after the header
        vertex_lines = lines[header_end : header_end + num_vertices]
        
        # Fast parse using numpy
        # Columns: [0:x, 1:y, 2:z, 3:r, 4:g, 5:b, 6:objectId, 7:globalId]
        # We handle this efficiently by splitting strings
        try:
            vertex_data = np.loadtxt(vertex_lines, dtype=np.float32, usecols=(0, 1, 2, 6))
        except Exception as e:
            print(f"Error parsing PLY for {scan_id}: {e}")
            continue

        points = vertex_data[:, :3]
        object_ids = vertex_data[:, 3].astype(np.int32)
        
        for obj in scan.get("objects", []):
            obj_id = int(obj["id"])
            save_path = output_dir / f"{scan_id}_{obj_id}.pt"
            
            if save_path.exists():
                continue
                
            # Extract points for this object
            mask = (object_ids == obj_id)
            object_points = points[mask]
            
            # Filter objects with too few points
            if object_points.shape[0] < 10:
                continue
                
            center = object_points.mean(axis=0)
            std = object_points.std(axis=0)
            bbox_min = object_points.min(axis=0)
            bbox_max = object_points.max(axis=0)
            bbox_size = bbox_max - bbox_min
            point_stats = {
                "center": center.astype(float).tolist(),
                "std": std.astype(float).tolist(),
                "bbox_size": bbox_size.astype(float).tolist(),
                "volume": float(np.prod(np.maximum(bbox_size, 1e-6))),
                "max_side": float(np.max(bbox_size)),
            }

            # Base paper preprocessing
            # Shape should be (3, N) for the existing preprocess function
            processed = preprocess_object_points(object_points.T, num_points=256)
            centers, _ = farthest_point_sampling(processed, num_centers=64)
            groups = group_points(processed, centers, group_size=32)
            
            groups_tensor = torch.tensor(groups, dtype=torch.float32, device=device_obj).unsqueeze(0)
            with torch.no_grad():
                lidar_tokens = encoder(groups_tensor).squeeze(0).cpu()
                
            torch.save(
                {
                    "dataset_version": "3rscan",
                    "scene_token": scan_id,
                    "annotation_token": f"{scan_id}_{obj_id}",
                    "instance_token": str(obj_id),
                    "category_name": obj["label"],
                    "num_points": object_points.shape[0],
                    "point_stats": point_stats,
                    "fps_centers": torch.tensor(centers, dtype=torch.float32),
                    "lidar_tokens": lidar_tokens,
                },
                save_path,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PointMAE LiDAR tokens for 3DSSG objects.")
    parser.add_argument("--scan-dir", type=Path, default=Path(r"D:\MTP_Project\3RScan"))
    parser.add_argument("--objects-json", type=Path, default=Path(r"D:\MTP_Project\3DSSG\objects.json"))
    parser.add_argument("--output-dir", type=Path, default=ProjectPaths().reference_root / "3rscan_lidar_tokens")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_3rscan_lidar_tokens(args.scan_dir, args.objects_json, args.output_dir, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
