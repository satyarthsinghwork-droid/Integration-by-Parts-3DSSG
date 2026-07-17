from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .lidar_utils import OfficialPointNetEncoder, farthest_point_sampling
from .protocol_3dssg import load_official_objects


def _stable_seed(scene_id: str, object_id: str) -> int:
    digest = hashlib.sha256(f"{scene_id}:{object_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _read_ascii_mesh(
    ply_path: Path,
    alignment: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the 3RScan annotated mesh and calculate vertex normals from faces."""
    with open(ply_path, "r", encoding="utf-8") as handle:
        header: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {ply_path}")
            header.append(line.strip())
            if line.strip() == "end_header":
                break
        if not header[1].startswith("format ascii"):
            raise ValueError(f"Only ASCII 3RScan PLY files are supported: {ply_path}")

        vertex_count = next(int(line.split()[-1]) for line in header if line.startswith("element vertex"))
        face_count = next(int(line.split()[-1]) for line in header if line.startswith("element face"))
        vertex_lines = [handle.readline() for _ in range(vertex_count)]
        vertices = np.loadtxt(vertex_lines, dtype=np.float32, usecols=(0, 1, 2, 3, 4, 5, 6))
        face_lines = [handle.readline() for _ in range(face_count)]

    xyz = vertices[:, :3]
    if alignment is not None:
        homogeneous = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1)
        xyz = np.asarray(homogeneous @ alignment, dtype=np.float32)[:, :3]
    rgb = vertices[:, 3:6] / 255.0
    instance_ids = vertices[:, 6].astype(np.int64)
    normals = np.zeros_like(xyz, dtype=np.float32)
    for line in face_lines:
        values = line.split()
        if len(values) < 4:
            continue
        count = int(values[0])
        indices = np.asarray(values[1 : 1 + count], dtype=np.int64)
        if indices.size < 3:
            continue
        anchor = xyz[indices[0]]
        for index in range(1, indices.size - 1):
            normal = np.cross(xyz[indices[index]] - anchor, xyz[indices[index + 1]] - anchor)
            normals[indices[0]] += normal
            normals[indices[index]] += normal
            normals[indices[index + 1]] += normal
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-12)
    return xyz, rgb.astype(np.float32), normals.astype(np.float32), instance_ids


def _alignment_transforms(metadata_path: Path) -> dict[str, np.ndarray]:
    """Load the rescan-to-reference transforms used by OCRL's transform_ply.py."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    transforms: dict[str, np.ndarray] = {}
    for scene in metadata:
        reference = str(scene["reference"])
        transforms[reference] = np.eye(4, dtype=np.float32)
        for scan in scene.get("scans", []):
            scan_id = str(scan["reference"])
            transform = scan.get("transform")
            if transform is not None:
                transforms[scan_id] = np.asarray(transform, dtype=np.float32).reshape(4, 4)
    return transforms


def _sample_object_points(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=count, replace=len(points) < count)
    sampled = points[indices].copy()
    # The official OCRL data loader mean-centers XYZ per object, but does not
    # scale it. Geometry stats are saved separately in world coordinates.
    sampled[:, :3] -= sampled[:, :3].mean(axis=0, keepdims=True)
    return sampled.astype(np.float32)


def _point_stats(xyz: np.ndarray) -> dict[str, Any]:
    center = xyz.mean(axis=0)
    std = xyz.std(axis=0)
    size = xyz.max(axis=0) - xyz.min(axis=0)
    return {
        "center": center.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "bbox_size": size.astype(float).tolist(),
        "volume": float(np.prod(np.maximum(size, 1e-6))),
        "max_side": float(size.max()),
    }


def prepare_3rscan_lidar_tokens(
    scan_dir: Path,
    objects_json: Path | None,
    output_dir: Path,
    checkpoint: Path,
    device: str | None,
    official_subset_dir: Path | None = None,
    alignment_metadata: Path | None = None,
    points_per_object: int = 128,
    tokens_per_object: int = 64,
) -> None:
    """Create deterministic OCRL-pretrained PointNet tokens for official 3DSSG.

    ``objects_json`` is retained for notebook compatibility; official subset
    annotations are required for a base-paper-comparable run.
    """
    del objects_json
    if official_subset_dir is None:
        raise ValueError("official_subset_dir is required for the comparable 3DSSG pipeline.")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Official OCRL object encoder checkpoint is missing: {checkpoint}")

    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder = OfficialPointNetEncoder().to(device_obj).eval()
    encoder.load_state_dict(state, strict=True)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    targets = load_official_objects(official_subset_dir)
    transforms = _alignment_transforms(alignment_metadata) if alignment_metadata is not None else {}
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = saved = skipped = 0
    for scan_id, scan_objects in tqdm(sorted(targets.items()), desc="Extracting official pretrained PointNet tokens"):
        pending_objects = {
            object_id: category_name
            for object_id, category_name in scan_objects.items()
            if not (output_dir / f"{scan_id}_{object_id}.pt").exists()
        }
        if not pending_objects:
            continue
        ply_path = scan_dir / scan_id / "labels.instances.annotated.v2.ply"
        if not ply_path.exists():
            skipped += len(pending_objects)
            continue
        try:
            if alignment_metadata is not None and scan_id not in transforms:
                raise ValueError(f"No official 3RScan alignment transform found for {scan_id}")
            xyz, rgb, normals, instance_ids = _read_ascii_mesh(ply_path, transforms.get(scan_id))
        except (OSError, ValueError) as error:
            print(f"Skipping {scan_id}: {error}")
            skipped += len(pending_objects)
            continue

        for object_id, category_name in pending_objects.items():
            processed += 1
            save_path = output_dir / f"{scan_id}_{object_id}.pt"
            mask = instance_ids == int(object_id)
            object_xyz = xyz[mask]
            # OCRL samples with replacement, so even a one-point annotation is
            # valid. Only truly empty annotations cannot be encoded.
            if object_xyz.shape[0] == 0:
                skipped += 1
                continue
            object_points = np.concatenate([object_xyz, rgb[mask], normals[mask]], axis=1)
            sampled = _sample_object_points(object_points, points_per_object, _stable_seed(scan_id, object_id))
            _, token_indices = farthest_point_sampling(sampled[:, :3], num_centers=tokens_per_object, initial_index=0)
            point_tensor = torch.from_numpy(sampled.T).unsqueeze(0).to(device_obj)
            with torch.no_grad():
                all_tokens = encoder.forward_point_tokens(point_tensor).squeeze(0).cpu()
            torch.save(
                {
                    "dataset_version": "3rscan_official_static_v2",
                    "scene_token": scan_id,
                    "annotation_token": f"{scan_id}_{object_id}",
                    "instance_token": str(object_id),
                    "category_name": category_name,
                    "num_points": int(object_xyz.shape[0]),
                    "point_stats": _point_stats(object_xyz),
                    "lidar_tokens": all_tokens[torch.from_numpy(token_indices)].contiguous(),
                    "encoder_name": "OCRL official PointNetEncoder",
                    "encoder_checkpoint": str(checkpoint),
                    "encoder_checkpoint_sha256": checkpoint_sha256,
                    "input_channels": "XYZ+RGB+normal",
                    "coordinate_frame": "official reference-aligned" if alignment_metadata is not None else "scan-local",
                    "alignment_metadata": str(alignment_metadata) if alignment_metadata is not None else None,
                    "points_per_object": points_per_object,
                    "tokens_per_object": tokens_per_object,
                },
                save_path,
            )
            saved += 1
    print(f"Processed {processed} official objects; saved {saved}; skipped {skipped}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OCRL-pretrained PointNet object tokens for official 3DSSG.")
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--official-subset-dir", type=Path, default=Path("official_splits"))
    parser.add_argument("--alignment-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("pipeline_data/3rscan_pointnet_aligned_tokens_v6"))
    parser.add_argument("--checkpoint", type=Path, default=Path("reference/pretrained/obj_enc.pth"))
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_3rscan_lidar_tokens(
        scan_dir=args.scan_dir,
        objects_json=None,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        official_subset_dir=args.official_subset_dir,
        alignment_metadata=args.alignment_metadata,
    )


if __name__ == "__main__":
    main()
